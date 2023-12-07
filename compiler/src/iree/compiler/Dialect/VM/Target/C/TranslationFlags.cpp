// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Target/C/TranslationFlags.h"

#include "llvm/Support/CommandLine.h"

namespace mlir::iree_compiler::IREE::VM {

static llvm::cl::opt<COutputFormat> outputFormatFlag{
    "iree-vm-c-module-output-format",
    llvm::cl::desc("Output format used to write the C module"),
    llvm::cl::init(COutputFormat::kCode),
    llvm::cl::values(
        clEnumValN(COutputFormat::kCode, "code", "C Code file"),
        clEnumValN(COutputFormat::kMlirText, "mlir-text",
                   "MLIR module file in the VM and EmitC dialects")),
};

static llvm::cl::opt<bool> optimizeFlag{
    "iree-vm-c-module-optimize",
    llvm::cl::desc(
        "Optimizes the VM module with CSE/inlining/etc prior to serialization"),
    llvm::cl::init(true),
};

static llvm::cl::opt<bool> stripDebugOpsFlag{
    "iree-vm-c-module-strip-debug-ops",
    llvm::cl::desc("Strips debug-only ops from the module"),
    llvm::cl::init(false),
};

CTargetOptions getCTargetOptionsFromFlags() {
  CTargetOptions targetOptions;
  targetOptions.outputFormat = outputFormatFlag;
  targetOptions.optimize = optimizeFlag;
  targetOptions.stripDebugOps = stripDebugOpsFlag;
  return targetOptions;
}

} // namespace mlir::iree_compiler::IREE::VM
