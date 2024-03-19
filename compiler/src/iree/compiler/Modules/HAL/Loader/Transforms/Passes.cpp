// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::HAL::Loader {

using FunctionLikeNest =
    MultiOpNest<func::FuncOp, IREE::Util::InitializerOp, IREE::Util::FuncOp>;

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
// -iree-hal-inline-dynamic-transformation-pipeline
//===----------------------------------------------------------------------===//

void buildHALInlineDynamicTransformPassPipeline(
    OpPassManager &passManager, const TargetRegistry &targetRegistry,
    const TargetOptions &targetOptions) {
  //----------------------------------------------------------------------------
  // Device assignment and interface materialization
  //----------------------------------------------------------------------------

  IREE::HAL::buildHALConfigurationPassPipeline(passManager, targetRegistry,
                                               targetOptions);

  //----------------------------------------------------------------------------
  // Executable translation
  //----------------------------------------------------------------------------

  // Translate each executable variant to its target IR form.
  // It's extremely important this runs parallelized as it's where a large
  // majority of our compilation time lives (we invoke LLVM and lld and such).
  //
  // After this point the executables are opaque blobs and we cannot change
  // their interfaces.
  passManager.addNestedPass<IREE::HAL::ExecutableOp>(
      IREE::HAL::createConfigureExecutablesPass({targetRegistry}));
  passManager.addNestedPass<IREE::HAL::ExecutableOp>(
      IREE::HAL::createTranslateExecutablesPass({targetRegistry}));

  //----------------------------------------------------------------------------
  // Conversion
  //----------------------------------------------------------------------------

  // Convert from stream to hal_inline + hal_loader.
  passManager.addPass(IREE::HAL::Loader::createConversionPass());

  //----------------------------------------------------------------------------
  // Executable packing and runtime loading
  //----------------------------------------------------------------------------

  // Link executables together.
  passManager.addPass(IREE::HAL::createLinkExecutablesPass({targetRegistry}));

  // Resolve export ordinals from nested symbol references prior to
  // serialization.
  passManager.addPass(IREE::HAL::Loader::createResolveExportOrdinalsPass());

  // Serialize executables to their binary forms.
  passManager.addNestedPass<IREE::HAL::ExecutableOp>(
      IREE::HAL::createSerializeExecutablesPass(
          {&targetRegistry, targetOptions.debugLevel,
           targetOptions.executableIntermediatesPath,
           targetOptions.executableBinariesPath}));

  // NOTE: symbol DCE will destroy executable target contents.
  passManager.addPass(mlir::createSymbolDCEPass());

  // Materialize executable globals and initializers that load them.
  passManager.addPass(IREE::HAL::Loader::createMaterializeExecutablesPass());

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
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h.inc"
} // namespace

void registerHALLoaderPasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<> transformPassPipeline(
      "iree-hal-inline-dynamic-transformation-pipeline",
      "Runs the inline HAL executable loader dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildHALInlineDynamicTransformPassPipeline(
            passManager, TargetRegistry::getGlobal(),
            TargetOptions::FromFlags::get());
      });
}

} // namespace mlir::iree_compiler::IREE::HAL::Loader
