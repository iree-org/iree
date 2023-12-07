// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_UTILS_CALLINGCONVENTION_H_
#define IREE_COMPILER_DIALECT_VM_UTILS_CALLINGCONVENTION_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"

namespace mlir::iree_compiler::IREE::VM {

// Generates a string encoding the function type for defining the
// FunctionSignatureDef::calling_convention field for import functions.
//
// This differs from makeCallingConventionString in that it supports variadic
// arguments. Ideally we'd combine the two, but we only have this additional
// metadata on IREE::VM::ImportOp.
std::optional<std::string>
makeImportCallingConventionString(IREE::VM::ImportOp importOp);

// Generates a string encoding the function type for defining the
// FunctionSignatureDef::calling_convention field for internal/export functions.
std::optional<std::string> makeCallingConventionString(IREE::VM::FuncOp funcOp);

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_UTILS_CALLINGCONVENTION_H_
