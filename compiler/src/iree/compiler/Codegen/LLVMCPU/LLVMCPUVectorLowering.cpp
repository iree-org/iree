// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-vector-lowering"

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
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

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
    RewritePatternSet patterns(ctx);
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
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After lowering high level vector ops to lower level "
                    "vector ops ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Make sure we remove redundant vector ops (e.g., vector tranposes) before we
  // lower them and can't be optimized away anymore.
  {
    RewritePatternSet patterns(ctx);
    SmallVector<Dialect *> dialects;
    dialects.push_back(ctx->getLoadedDialect<vector::VectorDialect>());
    dialects.push_back(ctx->getLoadedDialect<memref::MemRefDialect>());
    dialects.push_back(ctx->getLoadedDialect<linalg::LinalgDialect>());
    for (auto &dialect : dialects)
      dialect->getCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorTransferLoweringPatterns(patterns,
                                                   /*maxTransferRank=*/1);
    auto vectorTransferToSCFOptions =
        VectorTransferToSCFOptions().enableFullUnroll();
    populateVectorToSCFConversionPatterns(patterns, vectorTransferToSCFOptions);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After lowering vector transfers to SCF ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Lowering for vector.transpose ops.
  {
    RewritePatternSet patterns(ctx);
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
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After lowering vector transpose ops ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
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
