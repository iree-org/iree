// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/Passes.h"

#include "iree/compiler/Dialect/Flow/Conversion/MeshToFlow/MeshToFlow.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/InputConversion/Common/Passes.h.inc" // IWYU pragma: export
} // namespace

void buildCommonInputConversionPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createIREEImportPublicPass());
  passManager.addPass(createImportMLProgramPass());
  passManager.addPass(createSanitizeModuleNamesPass());
  passManager.addPass(IREE::Flow::createConvertMeshToFlowPass());
}

void registerCommonInputConversionPasses() {
  // Generated passes.
  registerPasses();

  PassPipelineRegistration<> common(
      "iree-common-input-transformation-pipeline",
      "Runs the common input transformation pipeline",
      [](OpPassManager &passManager) {
        buildCommonInputConversionPassPipeline(passManager);
      });
}

} // namespace mlir::iree_compiler
