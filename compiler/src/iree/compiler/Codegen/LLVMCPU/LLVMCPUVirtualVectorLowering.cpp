// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmNeon/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-virtual-vector-lowering"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUVIRTUALVECTORLOWERINGPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {
class LLVMCPUVirtualVectorLoweringPass
    : public impl::LLVMCPUVirtualVectorLoweringPassBase<
          LLVMCPUVirtualVectorLoweringPass> {
public:
  using Base::Base;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect,
                    arm_neon::ArmNeonDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVirtualVectorLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

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

  DictionaryAttr targetConfig;
  if (auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp)) {
    targetConfig = targetAttr.getConfiguration();
  }

  // Target-dependenet patterns.
  {
    if (enableArmI8mm) {
      RewritePatternSet patterns(ctx);
      arm_neon::populateLowerContractionToNeonI8MMPatterns(patterns);
      (void)applyPatternsGreedily(funcOp, std::move(patterns));
    }
  }

  // Target-independent patterns.
  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    // RVV should be able to lower most of the gather / scatter with indexed
    // load / store.
    if (!targetConfig || !isRISCV(targetConfig) ||
        !hasAnyVFeature(targetConfig)) {
      vector::populateVectorGatherToConditionalLoadPatterns(patterns);
    }
    vector::populateVectorGatherLoweringPatterns(patterns);
    vector::populateVectorContractLoweringPatterns(
        patterns, vectorTransformOptions.vectorContractLowering,
        /*benefit=*/1,
        /*disableOuterProductLowering=*/false);
    // This pattern will transform vector loads whose elements are used in a
    // scalar fashion into scalar loads. This will let scalar loads to be folded
    // into broadcast/arithmetic operations and reduce register pressure.
    vector::populateScalarVectorTransferLoweringPatterns(
        patterns, /*benefit=*/1, /*allowMultipleUses=*/true);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorMultiReductionReorderAndExpandPatterns(
        patterns, vectorMultiReductionLowering);
    vector::populateVectorMultiReductionFlatteningPatterns(
        patterns, vectorMultiReductionLowering);
    vector::populateVectorMultiReductionUnrollingPatterns(
        patterns, vectorMultiReductionLowering);
    populateVectorTransferFullPartialPatterns(patterns, vectorTransformOptions);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }
}
} // namespace
} // namespace mlir::iree_compiler
