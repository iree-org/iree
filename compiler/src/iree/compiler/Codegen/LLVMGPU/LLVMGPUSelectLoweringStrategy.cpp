// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

namespace {
/// Selects a lowering strategy for taking a hal.executable.variant operation
/// to scalar/native-vector code.
class LLVMGPUSelectLoweringStrategyPass
    : public LLVMGPUSelectLoweringStrategyBase<
          LLVMGPUSelectLoweringStrategyPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // TODO(qedawkins): Once TransformStrategies is deprecated, drop the
    // unnecessary dialect registrations.
    // clang-format off
    registry
        .insert<IREE::Codegen::IREECodegenDialect,
                IREE::HAL::HALDialect,
                IREE::LinalgExt::IREELinalgExtDialect,
                linalg::LinalgDialect,
                gpu::GPUDialect,
                nvgpu::NVGPUDialect,
                pdl::PDLDialect,
                pdl_interp::PDLInterpDialect,
                scf::SCFDialect,
                tensor::TensorDialect,
                transform::TransformDialect,
                vector::VectorDialect>();
    // clang-format on
  }

  LLVMGPUSelectLoweringStrategyPass() = default;
  LLVMGPUSelectLoweringStrategyPass(
      const LLVMGPUSelectLoweringStrategyPass &pass) {};

  void runOnOperation() override;
};
} // namespace

/// Verify that valid configuration is set for all ops within the compiled
/// module.
template <typename F>
static LogicalResult
verifyLoweringConfiguration(FunctionOpInterface funcOp,
                            IREE::Codegen::TranslationInfoAttr translationInfo,
                            ArrayRef<int64_t> workgroupSize, F verificationFn) {
  auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    IREE::Codegen::LoweringConfigAttr loweringConfig = getLoweringConfig(op);
    if (!loweringConfig)
      return WalkResult::advance();
    return verificationFn(op, loweringConfig, translationInfo, workgroupSize);
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

  return verifyLoweringConfiguration(
      funcOp, translationInfo, workgroupSize.value(), verifyGPUMatmulPipeline);
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

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMGPUSelectLoweringStrategyPass() {
  return std::make_unique<LLVMGPUSelectLoweringStrategyPass>();
}

} // namespace mlir::iree_compiler
