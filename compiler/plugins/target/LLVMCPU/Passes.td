// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PLUGINS_TARGET_LLVMCPU_PASSES
#define IREE_PLUGINS_TARGET_LLVMCPU_PASSES

include "mlir/Pass/PassBase.td"

def TestPropagateTargetAttrsPass :
    Pass<"iree-llvmcpu-target-test-propagate-target-attrs", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Test the utilities to propagate hal.target to individual functions";
  let dependentDialects = [
    "IREE::HAL::HALDialect",
    "::mlir::LLVM::LLVMDialect"
  ];
}

#endif // IREE_PLUGINS_TARGET_LLVMCPU_PASSES
