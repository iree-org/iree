// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-lower-vectors"

namespace mlir {
namespace iree_compiler {
namespace {
/// Pass to lower Vector ops before conversion to LLVM.
class LLVMCPUVectorLoweringPass
    : public LLVMCPUVectorLoweringBase<LLVMCPUVectorLoweringPass> {
 public:
  using LLVMCPUVectorLoweringBase::LLVMCPUVectorLoweringBase;
  LLVMCPUVectorLoweringPass(const LLVMCPUVectorLoweringPassOptions &options) {
    this->splitVectorTransfersTo = options.splitVectorTransfersTo;
    this->lowerVectorTransposeTo = options.lowerVectorTransposeTo;
    this->lowerVectorTransposeToAVX2 = options.lowerVectorTransposeToAVX2;
    this->lowerVectorMultiReductionTo = options.lowerVectorMultiReductionTo;
    this->lowerVectorContractionTo = options.lowerVectorContractionTo;
    this->unrollVectorTransfers = options.unrollVectorTransfers;
    this->maxTransferRank = options.maxTransferRank;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVectorLoweringPass::runOnOperation() {
  // Lower high level vector operations like contract or multidim reduce ops
  // to lower level vector ops.
  {
    vector::VectorTransposeLowering vectorTransposeLowering =
        llvm::StringSwitch<vector::VectorTransposeLowering>(
            lowerVectorTransposeTo.getValue())
            .Case("eltwise", vector::VectorTransposeLowering::EltWise)
            .Case("flat_transpose", vector::VectorTransposeLowering::Flat)
            .Case("shuffle", vector::VectorTransposeLowering::Shuffle)
            .Default(vector::VectorTransposeLowering::EltWise);
    vector::VectorMultiReductionLowering vectorMultiReductionLowering =
        llvm::StringSwitch<vector::VectorMultiReductionLowering>(
            lowerVectorMultiReductionTo.getValue())
            .Case("innerreduction",
                  vector::VectorMultiReductionLowering::InnerReduction)
            .Default(vector::VectorMultiReductionLowering::InnerParallel);
    vector::VectorContractLowering vectorContractLowering =
        llvm::StringSwitch<vector::VectorContractLowering>(
            lowerVectorContractionTo.getValue())
            .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
            .Case("dot", vector::VectorContractLowering::Dot)
            .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
            .Default(vector::VectorContractLowering::OuterProduct);
    vector::VectorTransferSplit vectorTransferSplit =
        llvm::StringSwitch<vector::VectorTransferSplit>(
            splitVectorTransfersTo.getValue())
            .Case("none", vector::VectorTransferSplit::None)
            .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
            .Case("vector-transfers",
                  vector::VectorTransferSplit::VectorTransfer)
            .Default(vector::VectorTransferSplit::None);

    // Per-function lowering pipeline.
    vector::VectorTransformsOptions vectorTransformOptions =
        vector::VectorTransformsOptions()
            .setVectorTransposeLowering(vectorTransposeLowering)
            .setVectorTransformsOptions(vectorContractLowering)
            .setVectorMultiReductionLowering(vectorMultiReductionLowering)
            .setVectorTransferSplit(vectorTransferSplit);
    RewritePatternSet patterns(&getContext());
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    vector::populateVectorContractLoweringPatterns(
        patterns, vectorTransformOptions,
        /*benefit=*/1,
        /*disableOuterProductLowering=*/true);
    vector::populateVectorShapeCastLoweringPatterns(patterns);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorMultiReductionLoweringPatterns(
        patterns, vectorMultiReductionLowering);
    populateVectorTransferFullPartialPatterns(patterns, vectorTransformOptions);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

  {
    RewritePatternSet patterns(&getContext());
    vector::populateVectorTransferLoweringPatterns(patterns, maxTransferRank);
    VectorTransferToSCFOptions vectorTransferToSCFOptions =
        VectorTransferToSCFOptions().enableFullUnroll(unrollVectorTransfers);
    populateVectorToSCFConversionPatterns(
        patterns, vectorTransferToSCFOptions.setTargetRank(maxTransferRank));
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

  {
    // Lowering for vector.transpose ops.
    RewritePatternSet patterns(&getContext());
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    vector::populateVectorTransposeLoweringPatterns(patterns,
                                                    vectorTransformOptions);
    if (lowerVectorTransposeToAVX2) {
      auto avx2LoweringOptions =
          x86vector::avx2::LoweringOptions().setTransposeOptions(
              x86vector::avx2::TransposeLoweringOptions()
                  .lower4x8xf32(lowerVectorTransposeToAVX2)
                  .lower8x8xf32(lowerVectorTransposeToAVX2));
      x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
          patterns, avx2LoweringOptions, /*benefit=*/10);
    }
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUVectorLoweringPass() {
  return std::make_unique<LLVMCPUVectorLoweringPass>();
}
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUVectorLoweringPass(
    const LLVMCPUVectorLoweringPassOptions &options) {
  return std::make_unique<LLVMCPUVectorLoweringPass>(options);
}
}  // namespace iree_compiler
}  // namespace mlir
