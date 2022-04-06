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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-vectorize"

namespace mlir {
namespace iree_compiler {
namespace {

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
    for (const auto &it : llvm::enumerate(contractOp.getIteratorTypes())) {
      if (isParallelIterator(it.value())) lastParalleldim = it.index();
    }
    SmallVector<int64_t, 4> nativeSize(contractOp.getIteratorTypes().size(), 1);
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
    func::FuncOp funcOp = getOperation();

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
      vector::TransposeOp::getCanonicalizationPatterns(foldPatterns, context);
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
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVVectorizePass() {
  return std::make_unique<SPIRVVectorizePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
