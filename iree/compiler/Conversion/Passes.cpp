// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Conversion/Passes.h.inc"
}  // namespace

void registerConversionPasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<LLVMTransformPassPipelineOptions>
      linalgLLVMVPipeline(
          "iree-codegen-linalg-to-llvm-pipeline",
          "Runs the progressive lowering pipeline from Linalg to LLVM",
          [](OpPassManager &passManager,
             const LLVMTransformPassPipelineOptions &options) {
            buildLLVMTransformPassPipeline(passManager, options);
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

  static PassPipelineRegistration<> linalgToSPIRVPipeline(
      "iree-codegen-linalg-to-spirv-pipeline",
      "Runs the progressive lowering pipeline from Linalg to SPIR-V",
      [](OpPassManager &passManager) {
        buildLinalgToSPIRVPassPipeline(passManager,
                                       SPIRVCodegenOptions::getFromCLOptions());
      });

  static PassPipelineRegistration<> hloToLinalgSPIRVPipeline(
      "iree-codegen-hlo-to-spirv-pipeline",
      "Runs the progressive lowering pipeline from XLA HLO to Linalg to "
      "SPIR-V",
      [](OpPassManager &passManager) {
        buildSPIRVTransformPassPipeline(
            passManager, SPIRVCodegenOptions::getFromCLOptions());
      });
}

}  // namespace iree_compiler
}  // namespace mlir
