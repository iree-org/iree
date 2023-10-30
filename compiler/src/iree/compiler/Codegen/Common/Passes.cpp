// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {

void addCommonTargetExecutablePreprocessingPasses(OpPassManager &passManager) {
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(createTypePropagationPass());
  nestedModulePM.addPass(createBubbleUpOrdinalOpsPass());
  nestedModulePM.addPass(createBufferizeCopyOnlyDispatchesPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      IREE::LinalgExt::createDecomposeSoftmaxPass());
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
} // namespace iree_compiler
} // namespace mlir
