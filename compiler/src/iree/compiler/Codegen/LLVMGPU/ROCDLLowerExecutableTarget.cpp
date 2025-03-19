// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLLOWEREXECUTABLETARGETPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {
using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

/// Lowers an IREE hal.executable.variant operation using a suitable pass
/// pipeline.
class ROCDLLowerExecutableTargetPass final
    : public impl::ROCDLLowerExecutableTargetPassBase<
          ROCDLLowerExecutableTargetPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::HAL::HALDialect, IREE::LinalgExt::IREELinalgExtDialect,
                IREE::GPU::IREEGPUDialect, amdgpu::AMDGPUDialect,
                bufferization::BufferizationDialect, gpu::GPUDialect,
                linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    IREE::Codegen::TranslationInfoAttr translationInfo =
        getTranslationInfo(funcOp);
    if (!translationInfo) {
      return;
    }

    IREE::GPU::GPUPipelineOptions pipelineOptions =
        IREE::GPU::getPipelineOptions(funcOp, translationInfo);

    std::optional<OpPassManager> maybePipeline =
        getFunctionOpInterfacePassManager(funcOp);
    if (!maybePipeline) {
      funcOp.emitOpError(
          "unhandled function-like container during executable lowering");
      return signalPassFailure();
    }
    OpPassManager &pipeline = maybePipeline.value();

    switch (translationInfo.getDispatchLoweringPassPipeline()) {
    case CodeGenPipeline::LLVMGPUBaseLowering:
      addGPUBaseLoweringPassPipeline(pipeline);
      break;
    case CodeGenPipeline::LLVMGPUWarpReduction:
      addGPUWarpReductionPassPipeline(pipeline);
      break;
    case CodeGenPipeline::LLVMGPUTileAndFuse:
      addGPUTileAndFusePassPipeline(pipeline, pipelineOptions);
      break;
    // If no pipeline specified, then nothing to do.
    case IREE::Codegen::DispatchLoweringPassPipeline::None:
      return;
    default:
      funcOp.emitOpError("unsupported pipeline on ROCDL target");
      return signalPassFailure();
    }

    if (failed(runPipeline(pipeline, funcOp))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
