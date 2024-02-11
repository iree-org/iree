// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

namespace {
using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

/// Lowers an IREE hal.executable.variant operation using a suitable pass
/// pipeline.
class ROCDLLowerExecutableTargetPass
    : public ROCDLLowerExecutableTargetBase<ROCDLLowerExecutableTargetPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::HAL::HALDialect, IREE::LinalgExt::IREELinalgExtDialect,
                gpu::GPUDialect, linalg::LinalgDialect, scf::SCFDialect,
                tensor::TensorDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();

    std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo =
        getIdenticalTranslationInfo(variantOp);
    if (!translationInfo) {
      variantOp.emitError(
          "unsupported entry point functions with different translation info");
      return signalPassFailure();
    }

    OpPassManager pipeline(variantOp.getOperationName());

    switch (translationInfo.value().getDispatchLoweringPassPipeline()) {
    case CodeGenPipeline::LLVMGPUWarpReduction:
      addGPUWarpReductionPassPipeline(pipeline);
      break;
    // If no pipeline specified, then nothing to do.
    case IREE::Codegen::DispatchLoweringPassPipeline::None:
      return;
    default:
      variantOp.emitOpError("unsupported pipeline on ROCDL target");
      return signalPassFailure();
    }

    if (failed(runPipeline(pipeline, variantOp))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createROCDLLowerExecutableTargetPass() {
  return std::make_unique<ROCDLLowerExecutableTargetPass>();
}

} // namespace mlir::iree_compiler
