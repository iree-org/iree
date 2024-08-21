// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_HAL_INLINE_PASSES
#define IREE_MODULES_HAL_INLINE_PASSES

include "mlir/Pass/PassBase.td"

def ConversionPass : Pass<"iree-hal-inline-conversion", "mlir::ModuleOp"> {
  let summary = "Converts from various dialects to the HAL inline dialect";
}

def InlineExecutablesPass : Pass<"iree-hal-inline-executables", "mlir::ModuleOp"> {
  let summary = "Inlines translated executable functions into the host program";
}

#endif  // IREE_MODULES_HAL_INLINE_PASSES
