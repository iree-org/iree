// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Transforms/Passes.h"

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/VMVX/VMVXPasses.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {

// NOTE: this runs on the top-level program module containing all
// hal.executable ops.
void buildVMVXLinkingPassPipeline(OpPassManager &passManager) {
  // Link together executables. This may produce some IR duplication.
  passManager.addPass(createVMVXLinkExecutablesPass());

  // Cleanup IR duplication.
  passManager.addNestedPass<IREE::HAL::ExecutableOp>(
      mlir::createCanonicalizerPass());

  // Assign final executable constant ordinals.
  passManager.nest<IREE::HAL::ExecutableOp>()
      .addNestedPass<IREE::HAL::ExecutableVariantOp>(
          createVMVXAssignConstantOrdinalsPass());
}

}  // namespace iree_compiler
}  // namespace mlir
