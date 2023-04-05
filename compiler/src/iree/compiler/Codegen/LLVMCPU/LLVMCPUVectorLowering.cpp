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
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
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
    this->lowerVectorTransposeToAVX2 = options.lowerVectorTransposeToAVX2;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVectorLoweringPass::runOnOperation() {
  // Per-function lowering pipeline.
  auto vectorTransposeLowering = vector::VectorTransposeLowering::Shuffle;
  auto vectorMultiReductionLowering =
      vector::VectorMultiReductionLowering::InnerReduction;
  auto vectorContractLowering = vector::VectorContractLowering::OuterProduct;
  auto vectorTransferSplit = vector::VectorTransferSplit::None;
  auto vectorTransformOptions =
      vector::VectorTransformsOptions()
          .setVectorTransposeLowering(vectorTransposeLowering)
          .setVectorTransformsOptions(vectorContractLowering)
          .setVectorMultiReductionLowering(vectorMultiReductionLowering)
          .setVectorTransferSplit(vectorTransferSplit);
  // Lower high level vector operations like contract or multidim reduce ops
  // to lower level vector ops.
  {
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
    vector::populateVectorTransferLoweringPatterns(patterns,
                                                   /*maxTransferRank=*/1);
    auto vectorTransferToSCFOptions =
        VectorTransferToSCFOptions().enableFullUnroll();
    populateVectorToSCFConversionPatterns(patterns, vectorTransferToSCFOptions);
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
                  .lower4x8xf32()
                  .lower8x8xf32());
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
