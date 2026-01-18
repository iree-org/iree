// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUSELECTLOWERINGSTRATEGYPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {
/// Selects a lowering strategy for taking a hal.executable.variant operation
/// to scalar/native-vector code.
class LLVMGPUSelectLoweringStrategyPass final
    : public impl::LLVMGPUSelectLoweringStrategyPassBase<
          LLVMGPUSelectLoweringStrategyPass> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Codegen::IREECodegenDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override;
};
} // namespace

/// Verify that valid configuration is set for all ops within the compiled
/// module.
static LogicalResult verifyLoweringConfiguration(
    FunctionOpInterface funcOp,
    IREE::Codegen::TranslationInfoAttr translationInfo) {
  auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    auto loweringConfig = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
    if (!loweringConfig) {
      return success();
    }

    if (translationInfo.getDispatchLoweringPassPipeline() ==
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUVectorDistribute) {
      return verifyLLVMGPUVectorDistributePipeline(op, loweringConfig);
    }
    return success();
  });

  return failure(walkResult.wasInterrupted());
}

static LogicalResult
verifyEntryPoint(FunctionOpInterface funcOp,
                 IREE::Codegen::TranslationInfoAttr translationInfo) {
  std::optional<SmallVector<int64_t>> workgroupSize = getWorkgroupSize(funcOp);
  if (!workgroupSize) {
    return funcOp->emitOpError(
        "failed to get workgroup size needed for verification");
  }

  // Verify GPU-specific configuration
  if (failed(verifyLoweringConfiguration(funcOp, translationInfo))) {
    return failure();
  }

  return success();
}

void LLVMGPUSelectLoweringStrategyPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    if (failed(initGPULaunchConfig(funcOp))) {
      return signalPassFailure();
    }

    IREE::Codegen::TranslationInfoAttr translationInfo =
        getTranslationInfo(funcOp);
    if (!translationInfo) {
      // Dont do anything if translation info is not set.
      return;
    }

    // Verify the properties of each entry point based on the target pipeline.
    if (failed(verifyEntryPoint(funcOp, translationInfo))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
