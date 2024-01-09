// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

void addCommonTargetExecutablePreprocessingPasses(
    OpPassManager &passManager, bool useDecomposeSoftmaxFusion) {
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(createTypePropagationPass());
  nestedModulePM.addPass(createBubbleUpOrdinalOpsPass());
  nestedModulePM.addPass(createBufferizeCopyOnlyDispatchesPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createDecomposeSoftmaxPass(useDecomposeSoftmaxFusion));
  passManager.addPass(createMaterializeUserConfigsPass());
}

//===---------------------------------------------------------------------===//
// Register Common Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Common/Passes.h.inc"
} // namespace

void registerCodegenCommonPasses() {
  // Generated.
  registerPasses();
}
} // namespace mlir::iree_compiler
