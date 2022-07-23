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
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
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
  passManager.addNestedPass<mlir::func::FuncOp>(createLoopCoalescingPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      createSCFForLoopCanonicalizationPass());
  FunctionLikeNest(passManager)
      .addPass(createLoopInvariantCodeMotionPass)
      .addPass(createConvertSCFToCFPass);

  // Propagate buffer subranges throughout the program - this should remove any
  // remaining subspans and give us a smaller surface area during conversion.
  passManager.addPass(IREE::Util::createPropagateSubrangesPass());
  addCleanupPatterns(passManager);

  // Convert std/util/etc -> VM, along with any other dialects implementing the
  // VM conversion dialect interface.
  passManager.addPass(createConversionPass(targetOptions));

  passManager.addNestedPass<IREE::VM::ModuleOp>(createHoistInlinedRodataPass());
  passManager.addNestedPass<IREE::VM::ModuleOp>(createDeduplicateRodataPass());
  passManager.addNestedPass<IREE::VM::ModuleOp>(
      createSinkGlobalBufferLoadsPass());
  passManager.addNestedPass<IREE::VM::ModuleOp>(
      createGlobalInitializationPass());

  passManager.addPass(createInlinerPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  // Now that we've inlined/canonicalized/etc the initializers we can remove
  // them if they are empty to save a few bytes in the binary and avoid a
  // runtime initialization call.
  passManager.addNestedPass<IREE::VM::ModuleOp>(
      createDropEmptyModuleInitializersPass());

  passManager.addPass(createSymbolDCEPass());
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
