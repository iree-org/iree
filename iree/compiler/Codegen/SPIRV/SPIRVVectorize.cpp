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
#include "iree/compiler/Codegen/SPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-vectorize"

namespace mlir {
namespace iree_compiler {
namespace {

void populateVectorizationPatterns(MLIRContext *context,
                                   RewritePatternSet &patterns) {
  linalg::insertVectorizationPatterns<linalg::FillOp, linalg::GenericOp,
                                      linalg::ContractionOpInterface>(
      patterns, linalg::LinalgVectorizationOptions(),
      linalg::LinalgTransformationFilter(
          Identifier::get(getVectorizeMarker(), context)));
}

void populateVectorUnrollPatterns(MLIRContext *context,
                                  RewritePatternSet &patterns) {
  vector::populateVectorUnrollPatterns(
      patterns,
      vector::UnrollVectorOptions().setNativeShapeFn(getSPIRVNativeVectorSize));
}

/// Workaround SPIR-V backend limitations. SPIR-V vetorization pass relies on
/// unrolling to reduce instructions to a vector size we can convert to SPIR-V.
/// When vectorization creates transpose those block unrolling and result in
/// large vector we currently cannot lower. For now we always merge the
/// transpose into the contract op so that it can be unrolled.
// TODO(thomasraoux): Make transpose work with the current unrolling mechanism
// or replace unrolling.
class CombineContractTranspose final
    : public OpRewritePattern<vector::ContractionOp> {
 public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    // Perform lhs + rhs transpositions to conform to matmul row-major
    // semantics. Bail out if the contraction cannot be put in this form.
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    bool foundTranspose = false;
    std::array<Value, 3> sources = {op.lhs(), op.rhs(), op.acc()};
    SmallVector<AffineMap> newMaps;
    SmallVector<Value> newSources;
    for (auto source : llvm::enumerate(sources)) {
      auto map =
          op.indexing_maps()[source.index()].cast<AffineMapAttr>().getValue();
      auto tranposeOp = source.value().getDefiningOp<vector::TransposeOp>();
      if (!tranposeOp) {
        newSources.push_back(source.value());
        newMaps.push_back(map);
        continue;
      }
      SmallVector<int64_t, 3> perm;
      tranposeOp.getTransp(perm);
      SmallVector<AffineExpr> exprs(perm.size());
      for (auto remap : llvm::enumerate(perm)) {
        exprs[remap.value()] = map.getResult(remap.index());
      }
      newMaps.push_back(
          AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, ctx));
      newSources.push_back(tranposeOp.vector());
      foundTranspose = true;
    }
    if (!foundTranspose) return failure();

    Value res = rewriter.create<vector::ContractionOp>(
        loc, newSources[0], newSources[1], newSources[2],
        rewriter.getAffineMapArrayAttr(newMaps), op.iterator_types());
    rewriter.replaceOp(op, res);
    return success();
  }
};

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

    auto entryPointOp = getEntryPoint(funcOp);
    if (!entryPointOp) return;

    {
      RewritePatternSet vectorizationPatterns(&getContext());
      populateVectorizationPatterns(context, vectorizationPatterns);
      populateLinalgToVectorVectorizeConvPatterns(context,
                                                  vectorizationPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorizationPatterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After vectorization ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    // TODO: This should be a folding of Add into Contract in core but while
    // they live in different dialects, it is not possible without unnatural
    // dependencies.
    funcOp.walk([&](Operation *op) {
      if (auto contract = canonicalizeContractionAdd(op))
        op->replaceAllUsesWith(contract);
    });

    auto targetEnv = spirv::TargetEnv(spirv::lookupTargetEnv(funcOp));
    bool useCooperativeMatrix =
        targetEnv.allows(spirv::Capability::CooperativeMatrixNV) &&
        targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix);

    {
      RewritePatternSet vectorUnrollPatterns(funcOp.getContext());
      populateVectorUnrollPatterns(funcOp.getContext(), vectorUnrollPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorUnrollPatterns));
    }

    {
      linalg::hoistRedundantVectorTransfers(funcOp);

      LLVM_DEBUG({
        llvm::dbgs() << "--- After hoisting vector transfers ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    {
      RewritePatternSet canonicalizationPatterns(funcOp.getContext());
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          canonicalizationPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns));

      if (useCooperativeMatrix) {
        // When using cooperative matrix we don't want to lower the contract,
        // instead we want to merge contract and transpose so that they can be
        // converted to cooperative matrix matmul op.
        // TODO(thomasraoux): remove that once we support cooperative matrix
        // lowering in MLIR core.
        RewritePatternSet combineTransposePatterns(funcOp.getContext());
        combineTransposePatterns.add<CombineContractTranspose>(
            funcOp.getContext());
        (void)applyPatternsAndFoldGreedily(funcOp,
                                           std::move(combineTransposePatterns));
      } else {
        RewritePatternSet contractLoweringPatterns(funcOp.getContext());
        vector::populateVectorContractLoweringPatterns(
            contractLoweringPatterns,
            vector::VectorTransformsOptions().setVectorTransformsOptions(
                vector::VectorContractLowering::OuterProduct));
        (void)applyPatternsAndFoldGreedily(funcOp,
                                           std::move(contractLoweringPatterns));
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After unrolling vector ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createSPIRVVectorizePass() {
  return std::make_unique<SPIRVVectorizePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
