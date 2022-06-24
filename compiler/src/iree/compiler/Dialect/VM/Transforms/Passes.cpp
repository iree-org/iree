// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
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

void buildVMTransformPassPipeline(OpPassManager &passManager,
                                  TargetOptions targetOptions) {
  passManager.addNestedPass<mlir::func::FuncOp>(createLoopCoalescingPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      createSCFForLoopCanonicalizationPass());
  FunctionLikeNest(passManager)
      .addPass(createLoopInvariantCodeMotionPass)
      .addPass(createConvertSCFToCFPass);

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(createConversionPass(targetOptions));

  passManager.addNestedPass<IREE::VM::ModuleOp>(createHoistInlinedRodataPass());
  passManager.addNestedPass<IREE::VM::ModuleOp>(createDeduplicateRodataPass());
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
