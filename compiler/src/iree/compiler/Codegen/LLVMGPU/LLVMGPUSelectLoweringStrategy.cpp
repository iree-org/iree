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
      const LLVMGPUSelectLoweringStrategyPass &pass){};

  void runOnOperation() override;
};
} // namespace

/// Verify that valid configuration is set for all ops within the compiled
/// module.
template <typename F>
static LogicalResult
verifyLoweringConfiguration(ModuleOp module,
                            IREE::Codegen::TranslationInfoAttr translationInfo,
                            ArrayRef<int64_t> workgroupSize, F verificationFn) {
  auto walkResult = module.walk([&](Operation *op) -> WalkResult {
    IREE::Codegen::LoweringConfigAttr loweringConfig = getLoweringConfig(op);
    if (!loweringConfig)
      return WalkResult::advance();
    return verificationFn(op, loweringConfig, translationInfo, workgroupSize);
  });
  return failure(walkResult.wasInterrupted());
}

static LogicalResult
verifyEntryPoint(ModuleOp moduleOp,
                 IREE::Codegen::TranslationInfoAttr translationInfo,
                 IREE::HAL::ExecutableExportOp exportOp) {
  std::optional<mlir::ArrayAttr> workgroupSizeAttr =
      exportOp.getWorkgroupSize();

  if (workgroupSizeAttr.has_value()) {
    std::array<int64_t, 3> workgroupSizes;
    for (auto [index, attr] : llvm::enumerate(workgroupSizeAttr.value())) {
      workgroupSizes[index] = llvm::cast<IntegerAttr>(attr).getInt();
    }
    return verifyLoweringConfiguration(moduleOp, translationInfo,
                                       workgroupSizes, verifyGPUMatmulPipeline);
  }
  return success();
}

void LLVMGPUSelectLoweringStrategyPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();

  if (failed(initGPULaunchConfig(moduleOp))) {
    return signalPassFailure();
  }

  std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo =
      getIdenticalTranslationInfo(variantOp);
  if (!translationInfo) {
    moduleOp.emitOpError(
        "unhandled compilation of entry point functions with different "
        "translation info");
    return signalPassFailure();
  }

  // Verify the properties of each entry point based on the target pipeline.
  for (auto exportOp : variantOp.getExportOps()) {
    if (failed(verifyEntryPoint(moduleOp, translationInfo.value(), exportOp))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMGPUSelectLoweringStrategyPass() {
  return std::make_unique<LLVMGPUSelectLoweringStrategyPass>();
}

} // namespace mlir::iree_compiler
