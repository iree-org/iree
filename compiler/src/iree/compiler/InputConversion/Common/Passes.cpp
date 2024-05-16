// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/Passes.h"

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::InputConversion {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/InputConversion/Common/Passes.h.inc" // IWYU pragma: export
} // namespace

void buildCommonInputConversionPassPipeline(
    OpPassManager &passManager, const TransformOptions &transformOptions) {
  passManager.addPass(createIREEImportPublicPass());
  passManager.addPass(createImportMLProgramPass());
  passManager.addPass(createSanitizeModuleNamesPass());

  // TODO: this pass should either live in InputConversion or be run in flow -
  // it's a mistake that it's here.
  passManager.addPass(IREE::Flow::createConvertMeshToFlowPass());

  // ML frontends have very uneven support for user-controlled types _and_ users
  // tend to use types not well suited for the work they are doing. These
  // demotions/promotions allow users to change the types after lowering out of
  // the frontends. It'll always be better to do this higher up in the stack
  // as these kind of blanket conversions have corner cases and potential
  // accuracy/precision losses beyond what the user may expect.
  if (transformOptions.options.demoteF64ToF32) {
    passManager.addPass(createDemoteF64ToF32Pass());
  }
  if (transformOptions.options.demoteF32ToF16) {
    passManager.addPass(createDemoteF32ToF16Pass());
  }
  if (transformOptions.options.promoteF16ToF32) {
    passManager.addPass(createPromoteF16ToF32Pass());
  }
  if (transformOptions.options.promoteBF16ToF32) {
    passManager.addPass(createPromoteBF16ToF32Pass());
  }
  if (transformOptions.options.demoteI64ToI32) {
    passManager.addPass(createDemoteI64ToI32Pass());
  }
}

void registerCommonInputConversionPasses() {
  // Generated passes.
  registerPasses();

  PassPipelineRegistration<TransformOptions> common(
      "iree-common-input-transformation-pipeline",
      "Runs the common input transformation pipeline",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildCommonInputConversionPassPipeline(passManager, transformOptions);
      });
}

} // namespace mlir::iree_compiler::InputConversion
