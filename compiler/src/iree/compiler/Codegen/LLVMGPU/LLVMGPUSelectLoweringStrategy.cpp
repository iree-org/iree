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
  using impl::LLVMGPUSelectLoweringStrategyPassBase<
      LLVMGPUSelectLoweringStrategyPass>::LLVMGPUSelectLoweringStrategyPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Codegen::IREECodegenDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override;
};
} // namespace

/// Verify that valid configuration is set for all ops within the compiled
/// module.
template <typename ConfigTy>
static LogicalResult
verifyLoweringConfiguration(FunctionOpInterface funcOp,
                            IREE::Codegen::TranslationInfoAttr translationInfo,
                            ArrayRef<int64_t> workgroupSize) {
  auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    auto loweringConfig = getLoweringConfig<ConfigTy>(op);
    if (!loweringConfig)
      return WalkResult::advance();

    // Calls the correct overloaded function based on ConfigTy.
    if constexpr (std::is_same_v<ConfigTy, IREE::GPU::LoweringConfigAttr>) {
      return verifyGPUMatmulPipeline(op, loweringConfig, translationInfo);
    } else {
      return verifyGPUMatmulPipeline(op, loweringConfig, translationInfo,
                                     workgroupSize);
    }
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
  if (failed(verifyLoweringConfiguration<IREE::GPU::LoweringConfigAttr>(
          funcOp, translationInfo, workgroupSize.value()))) {
    return failure();
  }

  // Verify Codegen-specific configuration
  if (failed(verifyLoweringConfiguration<IREE::Codegen::LoweringConfigAttr>(
          funcOp, translationInfo, workgroupSize.value()))) {
    return failure();
  }

  return success();
}

void LLVMGPUSelectLoweringStrategyPass::runOnOperation() {
  auto moduleOp = getOperation();
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
