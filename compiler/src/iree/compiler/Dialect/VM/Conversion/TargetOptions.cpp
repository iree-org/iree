// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"

#include "llvm/Support/CommandLine.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::IREE::VM::TargetOptions);

namespace mlir::iree_compiler::IREE::VM {

void TargetOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory vmTargetOptionsCategory(
      "IREE VM target options");

  binder.opt<int>("iree-vm-target-index-bits", indexBits,
                  llvm::cl::desc("Bit width of index types."),
                  llvm::cl::cat(vmTargetOptionsCategory));
  binder.opt<bool>("iree-vm-target-extension-f32", f32Extension,
                   llvm::cl::desc("Support f32 target opcode extensions."),
                   llvm::cl::cat(vmTargetOptionsCategory));
  binder.opt<bool>("iree-vm-target-extension-f64", f64Extension,
                   llvm::cl::desc("Support f64 target opcode extensions."),
                   llvm::cl::cat(vmTargetOptionsCategory));
  binder.opt<bool>("iree-vm-target-truncate-unsupported-floats",
                   truncateUnsupportedFloats,
                   llvm::cl::desc("Truncate f64 to f32 when unsupported."),
                   llvm::cl::cat(vmTargetOptionsCategory));
  binder.opt<bool>(
      "iree-vm-target-optimize-for-stack-size", optimizeForStackSize,
      llvm::cl::desc(
          "Prefer optimizations that reduce VM stack usage over performance."),
      llvm::cl::cat(vmTargetOptionsCategory));
}

} // namespace mlir::iree_compiler::IREE::VM
