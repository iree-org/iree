// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-virtual-vector-lowering"

namespace mlir::iree_compiler {
namespace {
class LLVMCPUVirtualVectorLoweringPass
    : public LLVMCPUVirtualVectorLoweringBase<
          LLVMCPUVirtualVectorLoweringPass> {
public:
  using LLVMCPUVirtualVectorLoweringBase::LLVMCPUVirtualVectorLoweringBase;
  LLVMCPUVirtualVectorLoweringPass(std::string splitVectorTransfersTo) {
    this->splitVectorTransfersTo = splitVectorTransfersTo;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVirtualVectorLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  auto vectorMultiReductionLowering =
      vector::VectorMultiReductionLowering::InnerReduction;
  auto vectorContractLowering = vector::VectorContractLowering::OuterProduct;
  auto vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  auto vectorTransformOptions =
      vector::VectorTransformsOptions()
          .setVectorTransformsOptions(vectorContractLowering)
          .setVectorMultiReductionLowering(vectorMultiReductionLowering)
          .setVectorTransferSplit(vectorTransferSplit);

  RewritePatternSet patterns(ctx);
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorGatherLoweringPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(
      patterns, vectorTransformOptions,
      /*benefit=*/1,
      /*disableOuterProductLowering=*/false);
  // This pattern will transform vector loads whose elements are used in a
  // scalar fashion into scalar loads. This will let scalar loads to be folded
  // into broadcast/arithmetic operations and reduce register pressure.
  vector::populateScalarVectorTransferLoweringPatterns(
      patterns, /*benefit=*/1, /*allowMultipleUses=*/true);
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vectorMultiReductionLowering);
  populateVectorTransferFullPartialPatterns(patterns, vectorTransformOptions);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUVirtualVectorLoweringPass(std::string splitVectorTransfersTo) {
  return std::make_unique<LLVMCPUVirtualVectorLoweringPass>(
      splitVectorTransfersTo);
}

} // namespace mlir::iree_compiler
