// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Pass/PassManager.h"
//===---------------------------------------------------------------------===//
// Include pass headers per target device
//===---------------------------------------------------------------------===//
#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/Common/GPU/CommonGPUPasses.h"
#include "iree/compiler/Codegen/LLVMCPU/LLVMCPUPasses.h"
#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUPasses.h"
#include "iree/compiler/Codegen/SPIRV/SPIRVPasses.h"
#include "iree/compiler/Codegen/VMVX/VMVXPasses.h"
#include "iree/compiler/Codegen/WGSL/WGSLPasses.h"

namespace mlir {
namespace iree_compiler {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Passes.h.inc"
}  // namespace

void registerCodegenPasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<> LinalgLLVMPipeline(
      "iree-codegen-linalg-to-llvm-pipeline",
      "Runs the progressive lowering pipeline from Linalg to LLVM",
      [](OpPassManager &passManager) {
        buildLLVMCPUCodegenPassPipeline(passManager);
      });

  static PassPipelineRegistration<> LinalgNVVMPipeline(
      "iree-codegen-linalg-to-nvvm-pipeline",
      "Runs the progressive lowering pipeline from Linalg to NVVM",
      [](OpPassManager &passManager) {
        buildLLVMGPUTransformPassPipeline(passManager, false);
      });

  static PassPipelineRegistration<> LinalgROCDLPipeline(
      "iree-codegen-linalg-to-rocdl-pipeline",
      "Runs the progressive lowering pipeline from Linalg to ROCDL",
      [](OpPassManager &passManager) {
        buildLLVMGPUTransformPassPipeline(passManager, true);
      });

  static PassPipelineRegistration<> LinalgSPIRVPipeline(
      "iree-codegen-linalg-to-spirv-pipeline",
      "Runs the progressive lowering pipeline from linalg to SPIR-V",
      [](OpPassManager &passManager) {
        buildSPIRVCodegenPassPipeline(passManager, /*enableFastMath=*/false);
      });

  static PassPipelineRegistration<> LLVMCPULinkingPipeline(
      "iree-codegen-llvmcpu-linking-pipeline",
      "Runs the LLVMCPU HAL executable linking pipeline",
      [](OpPassManager &passManager) {
        buildLLVMCPULinkingPassPipeline(passManager);
      });

  static PassPipelineRegistration<> VMVXLinkingPipeline(
      "iree-codegen-vmvx-linking-pipeline",
      "Runs the VMVX HAL executable linking pipeline",
      [](OpPassManager &passManager) {
        buildVMVXLinkingPassPipeline(passManager);
      });
}

/// Hook to verify the lowering configuration and translation info for an
/// operation.
LogicalResult verifyLoweringConfiguration(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  switch (translationInfo.getDispatchLoweringPassPipeline()) {
    case IREE::Codegen::DispatchLoweringPassPipeline::Mmt4dTilingExpert:
      return verifyDoubleTilingExpertPassPipelineConfig(op, loweringConfig,
                                                        translationInfo);
    case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt:
      return verifyGPUMatmulPipeline(op, loweringConfig, translationInfo,
                                     workgroupSize);
    default:
      break;
  }
  return success();
}

void addCommonTargetExecutablePreprocessingPasses(OpPassManager &passManager) {
  passManager.addNestedPass<func::FuncOp>(createTypePropagationPass());
  passManager.addPass(createBubbleUpOrdinalOpsPass());
  passManager.addPass(createBufferizeCopyOnlyDispatchesPass());
  passManager.addNestedPass<func::FuncOp>(
      IREE::LinalgExt::createDecomposeSoftmaxPass());
  // Temporary solution to avoid large allocations due to softmax lowering.
  passManager.addNestedPass<func::FuncOp>(createRematerializeParallelOpsPass());
}

}  // namespace iree_compiler
}  // namespace mlir
