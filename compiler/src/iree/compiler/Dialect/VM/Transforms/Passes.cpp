// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

using FunctionLikeNest = MultiOpNest<func::FuncOp, IREE::Util::InitializerOp>;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static void addCleanupPatterns(OpPassManager &passManager) {
  // TODO(benvanik): run in a fixed-point iteration pipeline.

  // Standard MLIR cleanup.
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(mlir::createCSEPass());

  // Simplify util.global accesses; this can help with data flow tracking as
  // redundant store-loads are removed.
  FunctionLikeNest(passManager)
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass);

  // Cleanup and canonicalization of util.global (and other util ops).
  passManager.addPass(IREE::Util::createApplyPatternsPass());
  passManager.addPass(IREE::Util::createFoldGlobalsPass());
  passManager.addPass(IREE::Util::createFuseGlobalsPass());
}

//===----------------------------------------------------------------------===//
// -iree-vm-transformation-pipeline
//===----------------------------------------------------------------------===//

void buildVMTransformPassPipeline(OpPassManager &passManager,
                                  TargetOptions targetOptions) {
  // HACK: we'd really prefer not to inline here but most of the subsequent
  // passes are local only. We need to inline before we convert to the vm
  // dialect as they all operate on our various input dialects. If we could
  // control the inliner a bit more and not have it inline *everything* we could
  // do a more conservative inlining here and then the final one after lowering
  // to vm below.
  passManager.addPass(mlir::createInlinerPass());
  passManager.addPass(mlir::createSymbolDCEPass());

  FunctionLikeNest(passManager)
      .addPass(mlir::createSCFForLoopCanonicalizationPass);

  // This pass is sketchy as it can pessimize tight loops due to affine
  // treating all indices as signed and the unsigned conversion pass not being
  // able to handle that. The scf.for canonicalization does a decent job of
  // removing trivial loops above and this catches the rest. It inserts nasty
  // rem/div ops that we can never safely remove inside of the hot inner loop
  // and that sucks. We still have this here for now as the cost of the rem/div
  // are less than the cost of an additional loop that this could remove.
  passManager.addNestedPass<mlir::func::FuncOp>(createLoopCoalescingPass());

  FunctionLikeNest(passManager)
      .addPass(mlir::createLoopInvariantCodeMotionPass)
      .addPass(mlir::createConvertSCFToCFPass)
      .addPass(mlir::createLowerAffinePass)
      .addPass(mlir::arith::createArithUnsignedWhenEquivalentPass);

  // Propagate buffer subranges throughout the program - this should remove any
  // remaining subspans and give us a smaller surface area during conversion.
  passManager.addPass(IREE::Util::createPropagateSubrangesPass());
  addCleanupPatterns(passManager);

  // Convert std/util/etc -> VM, along with any other dialects implementing the
  // VM conversion dialect interface.
  passManager.addPass(createConversionPass(targetOptions));

  // Hoist globals and get the final set that need to be initialized.
  passManager.addNestedPass<IREE::VM::ModuleOp>(createHoistInlinedRodataPass());
  passManager.addNestedPass<IREE::VM::ModuleOp>(createDeduplicateRodataPass());
  addCleanupPatterns(passManager);
  passManager.addNestedPass<IREE::VM::ModuleOp>(createResolveRodataLoadsPass());

  // Catch any inlining opportunities we created during lowering.
  passManager.addPass(mlir::createInlinerPass());
  passManager.addPass(mlir::createSymbolDCEPass());

  // Create global initialization functions (__init/__deinit).
  addCleanupPatterns(passManager);
  passManager.addNestedPass<IREE::VM::ModuleOp>(
      createGlobalInitializationPass());

  // Ideally we'd run this as part of a fixed-point iteration: CSE may need the
  // canonicalizer to remove ops in order to get the equivalences it needs to
  // work while the canonicalizer may require CSE for patterns to kick in (like
  // "fold cmp if lhs=%0 && rhs=%0").
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(mlir::createCSEPass());
  passManager.addPass(mlir::createCanonicalizerPass());

  // Now that we've inlined/canonicalized/etc the initializers we can remove
  // them if they are empty to save a few bytes in the binary and avoid a
  // runtime initialization call.
  passManager.addNestedPass<IREE::VM::ModuleOp>(
      createDropEmptyModuleInitializersPass());

  if (targetOptions.optimizeForStackSize) {
    passManager.addNestedPass<IREE::VM::ModuleOp>(createSinkDefiningOpsPass());
  }
}

void registerVMTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-vm-transformation-pipeline",
      "Runs the full IREE VM dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildVMTransformPassPipeline(passManager,
                                     TargetOptions::FromFlags::get());
      });
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
