// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_UTILS_TYPETABLE_H_
#define IREE_COMPILER_DIALECT_VM_UTILS_TYPETABLE_H_

#include <string>
#include <vector>

#include "iree/compiler/Dialect/VM/IR/VMOps.h"

namespace mlir::iree_compiler::IREE::VM {

struct TypeDef {
  Type type;
  std::string full_name;
};

// Finds all types in the module and builds a type table mapping the index in
// the vector to the type represented by the type ordinal.
std::vector<TypeDef> buildTypeTable(IREE::VM::ModuleOp moduleOp);

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_UTILS_TYPETABLE_H_
