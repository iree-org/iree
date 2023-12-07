// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_C_CMODULETARGET_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_C_CMODULETARGET_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::VM {

// Defines the output format of the c module.
enum class COutputFormat {
  // C code.
  kCode,
  // MLIR text of the VM module mixed with emitc operations.
  kMlirText,
};

// Options that can be provided to c code translation.
struct CTargetOptions {
  // Format of the module written to the output stream.
  COutputFormat outputFormat = COutputFormat::kCode;

  // Run basic CSE/inlining/etc passes prior to serialization.
  bool optimize = true;

  // Strips vm ops with the VM_DebugOnly trait.
  bool stripDebugOps = false;
};

// Translates a vm.module to a c module.
//
// Exposed via the --iree-vm-ir-to-c-module translation.
LogicalResult translateModuleToC(IREE::VM::ModuleOp moduleOp,
                                 CTargetOptions targetOptions,
                                 llvm::raw_ostream &output);
LogicalResult translateModuleToC(mlir::ModuleOp outerModuleOp,
                                 CTargetOptions targetOptions,
                                 llvm::raw_ostream &output);

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_TARGET_C_CMODULETARGET_H_
