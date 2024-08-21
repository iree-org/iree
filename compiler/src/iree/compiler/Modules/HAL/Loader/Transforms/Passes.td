// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_HAL_LOADER_PASSES
#define IREE_MODULES_HAL_LOADER_PASSES

include "mlir/Pass/PassBase.td"

def ConversionPass : Pass<"iree-hal-loader-conversion", "mlir::ModuleOp"> {
  let summary = "Converts from various dialects to the HAL loader dialect";
}

def MaterializeExecutablesPass : Pass<"iree-hal-loader-materialize-executables", "mlir::ModuleOp"> {
  let summary = "Materializes executable globals and loader code";
}

def ResolveExportOrdinalsPass : Pass<"iree-hal-loader-resolve-export-ordinals", "mlir::ModuleOp"> {
  let summary = "Resolves dispatch operation target export entry point ordinals";
}

#endif  // IREE_MODULES_HAL_LOADER_PASSES
