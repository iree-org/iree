// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVVectorize.cpp -------------------------------------------------===//
//
// This pass vectorizes Linalg ops with buffer semantics.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-spirv-vectorize"

namespace mlir {
namespace iree_compiler {
namespace {

struct SwapYieldedInsertStridedSliceOutOfIfRegion final
    : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (ifOp.getNumResults() == 0) return failure();

    vector::InsertStridedSliceOp insertOp;
    unsigned yieldIndex;
    for (auto operand : llvm::enumerate(ifOp.thenYield().getOperands())) {
      insertOp = operand.value().getDefiningOp<vector::InsertStridedSliceOp>();
      yieldIndex = operand.index();
    }
    if (!insertOp) return failure();

    if (insertOp.getSourceVectorType().getRank() !=
        insertOp.getDestVectorType().getRank()) {
      auto srcShape = llvm::reverse(insertOp.getSourceVectorType().getShape());
      auto dstShape = llvm::reverse(insertOp.getDestVectorType().getShape());
      for (auto sd : llvm::zip_first(srcShape, dstShape)) {
        if (std::get<0>(sd) != std::get<1>(sd)) return failure();
      }
    }

    bool isDestDefinedAbove = areValuesDefinedAbove(ValueRange{insertOp.dest()},
                                                    ifOp.getThenRegion());
    SmallVector<Type, 4> resultTypes =
        llvm::to_vector<4>(ifOp.getResultTypes());
    resultTypes.erase(resultTypes.begin() + yieldIndex);
    resultTypes.push_back(insertOp.getSourceVectorType());
    if (!isDestDefinedAbove)
      resultTypes.push_back(insertOp.getDestVectorType());

    auto newIfOp = rewriter.create<scf::IfOp>(
        ifOp.getLoc(), resultTypes, ifOp.getCondition(), /*else=*/true);
    rewriter.eraseBlock(&newIfOp.getThenRegion().getBlocks().front());
    rewriter.eraseBlock(&newIfOp.getElseRegion().getBlocks().front());
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), newIfOp.getThenRegion(),
                                newIfOp.getThenRegion().begin());
    rewriter.inlineRegionBefore(ifOp.getElseRegion(), newIfOp.getElseRegion(),
                                newIfOp.getElseRegion().begin());

    {
      auto yieldOp = newIfOp.thenYield();
      rewriter.setInsertionPoint(yieldOp);

      SmallVector<Value, 4> values = llvm::to_vector<4>(yieldOp.getOperands());
      values.erase(values.begin() + yieldIndex);
      values.push_back(insertOp.source());
      if (!isDestDefinedAbove) values.push_back(insertOp.dest());
      rewriter.create<scf::YieldOp>(yieldOp.getLoc(), values);
      rewriter.eraseOp(yieldOp);
    }

    {
      auto yieldOp = newIfOp.elseYield();
      rewriter.setInsertionPoint(yieldOp);

      auto oldValue = yieldOp.getOperand(yieldIndex);
      Value extractValue;
      if (insertOp.getSourceVectorType().getRank() ==
          insertOp.getDestVectorType().getRank()) {
        extractValue = rewriter.create<vector::ExtractStridedSliceOp>(
            ifOp.getLoc(), insertOp.getSourceVectorType(), oldValue,
            insertOp.offsets(),
            rewriter.getI64ArrayAttr(insertOp.getSourceVectorType().getShape()),
            insertOp.strides());
      } else {
        auto offsets = llvm::to_vector<4>(
            llvm::map_range(insertOp.offsets(), [](Attribute attr) {
              return attr.cast<IntegerAttr>().getValue().getSExtValue();
            }));

        extractValue = rewriter.create<vector::ExtractOp>(
            ifOp.getLoc(), oldValue,
            llvm::makeArrayRef(offsets).drop_back(
                insertOp.getDestVectorType().getRank() -
                insertOp.getSourceVectorType().getRank()));
      }

      SmallVector<Value, 4> values = llvm::to_vector<4>(yieldOp.getOperands());
      values.erase(values.begin() + yieldIndex);
      values.push_back(extractValue);
      if (!isDestDefinedAbove) values.push_back(oldValue);
      rewriter.create<scf::YieldOp>(yieldOp.getLoc(), values);
      rewriter.eraseOp(yieldOp);
    }

    rewriter.setInsertionPointAfter(newIfOp);
    BlockAndValueMapping bvm;
    if (isDestDefinedAbove) {
      bvm.map(insertOp.source(),
              newIfOp.getResult(newIfOp.getNumResults() - 1));
    } else {
      bvm.map(insertOp.source(),
              newIfOp.getResult(newIfOp.getNumResults() - 2));
      bvm.map(insertOp.dest(), newIfOp.getResult(newIfOp.getNumResults() - 1));
    }
    auto newInsertOp = rewriter.clone(*insertOp, bvm);

    SmallVector<Value, 4> replacements;
    replacements.reserve(newIfOp.getNumResults());
    for (int i = 0; i < newIfOp.getNumResults(); ++i)
      replacements.push_back(newIfOp.getResult(i));
    replacements.pop_back_n(2 - isDestDefinedAbove);
    replacements.insert(replacements.begin() + yieldIndex,
                        newInsertOp->getResult(0));
    rewriter.replaceOp(ifOp, replacements);

    return success();
  }
};

int getNativeVectorSize(int64_t size) {
  // Try to use 4 first, and then 2, and then 1.
  return size % 4 == 0 ? 4 : (size % 2 == 0 ? 2 : 1);
}

Optional<SmallVector<int64_t, 4>> getNativeVectorShape(Operation *op) {
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>()) {
      SmallVector<int64_t, 4> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = getNativeVectorSize(vecType.getShape().back());
      return nativeSize;
    }
  } else if (auto vtOp = dyn_cast<VectorTransferOpInterface>(op)) {
    auto vecType = vtOp.getVectorType();
    SmallVector<int64_t, 4> nativeSize(vecType.getRank(), 1);
    for (const auto &dim :
         llvm::enumerate(vtOp.permutation_map().getResults())) {
      if (auto dimExpr = dim.value().dyn_cast<AffineDimExpr>()) {
        if (dimExpr.getPosition() == vtOp.permutation_map().getNumDims() - 1) {
          nativeSize[dim.index()] =
              getNativeVectorSize(vecType.getShape()[dim.index()]);
        }
      }
    }
    return nativeSize;
  } else if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
    unsigned lastParalleldim = 0;
    for (const auto &it : llvm::enumerate(contractOp.iterator_types())) {
      if (isParallelIterator(it.value())) lastParalleldim = it.index();
    }
    SmallVector<int64_t, 4> nativeSize(contractOp.iterator_types().size(), 1);
    nativeSize[lastParalleldim] = 4;  // Map to vec4 fma operations.
    return nativeSize;
  }
  return llvm::None;
}

/// Add patterns to vectorize any supported Linalg ops.
void populateVectorizationPatterns(RewritePatternSet &patterns) {
  linalg::LinalgVectorizationOptions opt;
  linalg::LinalgTransformationFilter f;
  linalg::VectorizationPatterns<linalg::FillOp, linalg::GenericOp>::insert(
      patterns, opt, f);
  patterns.add<linalg::LinalgVectorizationPattern>(
      patterns.getContext(), f.addOpFilter<linalg::ContractionOpInterface>(),
      opt);
  populateVectorizePadPatterns(patterns);
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorReductionToContractPatterns(patterns);
}

/// Adds patterns to unroll vector ops to SPIR-V native vector size.
void populateVectorUnrollPatterns(RewritePatternSet &patterns) {
  auto options =
      vector::UnrollVectorOptions().setNativeShapeFn(getNativeVectorShape);
  vector::populateVectorUnrollPatterns(patterns, options);
}

/// Vectorizes Linalg ops on buffer semantics.
class SPIRVVectorizePass : public SPIRVVectorizeBase<SPIRVVectorizePass> {
 public:
  SPIRVVectorizePass() = default;
  SPIRVVectorizePass(const SPIRVVectorizePass &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FuncOp funcOp = getOperation();

    {
      RewritePatternSet patterns(context);
      populateVectorizationPatterns(patterns);
      populateLinalgToVectorVectorizeConvPatterns(context, patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }

      RewritePatternSet foldPatterns(context);
      // Fold consumer add ops into the contraction op itself.
      vector::ContractionOp::getCanonicalizationPatterns(foldPatterns, context);
      if (failed(
              applyPatternsAndFoldGreedily(funcOp, std::move(foldPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After vectorization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      RewritePatternSet unrollPatterns(context);
      populateVectorUnrollPatterns(unrollPatterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(unrollPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After unrolling vector ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    linalg::hoistRedundantVectorTransfersOnTensor(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After hoisting vector transfers ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {  // Canonicalization
      RewritePatternSet patterns(context);
      vector::ExtractStridedSliceOp::getCanonicalizationPatterns(patterns,
                                                                 context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After canonicalizing vectors ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      RewritePatternSet contractLoweringPatterns(context);
      vector::populateVectorBroadcastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorContractLoweringPatterns(
          contractLoweringPatterns,
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct));
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(contractLoweringPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After handling contraction ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {  // Canonicalization
      RewritePatternSet patterns(context);
      // We need to pull in casting way leading one dims to allow cancelling
      // some read/write ops.
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After remove leading one dims ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      RewritePatternSet hoistPatterns(context);
      hoistPatterns.add<SwapYieldedInsertStridedSliceOutOfIfRegion>(context);
      if (failed(
              applyPatternsAndFoldGreedily(funcOp, std::move(hoistPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After hoisting inserts ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {  // Canonicalization
      RewritePatternSet patterns(context);
      vector::populateVectorInsertExtractStridedSliceTransforms(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createSPIRVVectorizePass() {
  return std::make_unique<SPIRVVectorizePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
