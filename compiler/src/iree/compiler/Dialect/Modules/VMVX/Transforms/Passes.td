// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_VMVX_PASSES
#define IREE_MODULES_VMVX_PASSES

include "mlir/Pass/PassBase.td"

def LowerLinalgMicrokernels :
    Pass<"iree-vmvx-lower-linalg-microkernels", ""> {
  let summary =
      "Lowers linalg ops to the VMVX microkernel library";
  let constructor = "mlir::iree_compiler::IREE::VMVX::createLowerLinalgMicrokernelsPass()";
}

def Conversion : Pass<"iree-vmvx-conversion", "mlir::ModuleOp"> {
  let summary = "Converts from various dialects to the VMVX dialect";
  let constructor = "mlir::iree_compiler::IREE::VMVX::createConversionPass()";
}

#endif  // IREE_MODULES_VMVX_PASSES
