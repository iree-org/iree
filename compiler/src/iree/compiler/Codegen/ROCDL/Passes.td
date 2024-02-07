// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_ROCDL_PASSES
#define IREE_CODEGEN_ROCDL_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ROCDL Passes (keep alphabetical)
//===----------------------------------------------------------------------===//

def ROCDLLowerExecutableTarget : Pass<
    "iree-rocdl-lower-executable-target",
    "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Lower an IREE hal.executable.variant op using a suitable "
                "pass pipeline";
  let constructor =
      "mlir::iree_compiler::createROCDLLowerExecutableTargetPass()";
}

def ROCDLSelectLoweringStrategy : Pass<
    "iree-rocdl-select-lowering-strategy",
    "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Select a suitable lowering strategy for an IREE "
                "hal.executable.variant op";
  let constructor =
      "mlir::iree_compiler::createROCDLSelectLoweringStrategyPass()";
}


#endif // IREE_CODEGEN_ROCDL_PASSES

