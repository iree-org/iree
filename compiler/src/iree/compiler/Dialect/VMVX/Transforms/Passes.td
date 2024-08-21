// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_VMVX_PASSES
#define IREE_MODULES_VMVX_PASSES

include "mlir/Pass/PassBase.td"

def ConversionPass : Pass<"iree-vmvx-conversion", "mlir::ModuleOp"> {
  let summary = "Converts from various dialects to the VMVX dialect";
}

def MaterializeConstantsPass : Pass<"iree-vmvx-materialize-constants", "mlir::ModuleOp"> {
  let summary = "Materializes executable constant global values";
}

def ResolveBufferDescriptorsPass : Pass<"iree-vmvx-resolve-buffer-descriptors", ""> {
  let summary = "Resolves any existing vmvx.get_buffer_descriptor ops";
}

#endif  // IREE_MODULES_VMVX_PASSES
