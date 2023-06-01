// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Inline/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {
namespace Inline {

using FunctionLikeNest = MultiOpNest<func::FuncOp, IREE::Util::InitializerOp>;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static void addCleanupPatterns(OpPassManager &passManager) {
  // Standard MLIR cleanup.
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(mlir::createCSEPass());

  FunctionLikeNest(passManager)
      // Simplify util.global accesses; this can help with data flow tracking as
      // redundant store-loads are removed.
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass);

  // Cleanup and canonicalization of util.global (and other util ops).
  passManager.addPass(IREE::Util::createApplyPatternsPass());
  passManager.addPass(IREE::Util::createFoldGlobalsPass());
  passManager.addPass(IREE::Util::createFuseGlobalsPass());
}

//===----------------------------------------------------------------------===//
// -iree-hal-inline-static-transformation-pipeline
//===----------------------------------------------------------------------===//

void buildHALInlineStaticTransformPassPipeline(
    OpPassManager &passManager, const TargetBackendRegistry &targetRegistry,
    const TargetOptions &targetOptions) {
  //----------------------------------------------------------------------------
  // Device assignment and interface materialization
  //----------------------------------------------------------------------------

  IREE::HAL::buildHALConfigurationPassPipeline(passManager, targetRegistry,
                                               targetOptions);

  //----------------------------------------------------------------------------
  // Executable translation
  //----------------------------------------------------------------------------

  // Translate each executable down to common MLIR dialects.
  passManager.addNestedPass<IREE::HAL::ExecutableOp>(
      IREE::HAL::createTranslateExecutablesPass(targetRegistry));

  // Inline the translated executable functions.
  // We preserve the executables for their metadata used during conversion.
  passManager.addPass(IREE::HAL::Inline::createInlineExecutablesPass());
  addCleanupPatterns(passManager);

  //----------------------------------------------------------------------------
  // Conversion
  //----------------------------------------------------------------------------

  // Convert from stream to hal_inline.
  passManager.addPass(IREE::HAL::Inline::createConversionPass());

  // Propagate buffer subranges across the program.
  passManager.addPass(IREE::Util::createPropagateSubrangesPass());

  //----------------------------------------------------------------------------
  // Cleanup and canonicalization
  //----------------------------------------------------------------------------

  addCleanupPatterns(passManager);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Modules/HAL/Inline/Transforms/Passes.h.inc"
}  // namespace

void registerHALInlinePasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<> transformPassPipeline(
      "iree-hal-inline-static-transformation-pipeline",
      "Runs the inline HAL dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildHALInlineStaticTransformPassPipeline(
            passManager, TargetBackendRegistry::getGlobal(),
            TargetOptions::FromFlags::get());
      });
}

}  // namespace Inline
}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
