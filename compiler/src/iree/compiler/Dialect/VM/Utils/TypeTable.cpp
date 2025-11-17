// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Utils/TypeTable.h"

namespace mlir::iree_compiler::IREE::VM {

// Finds all types in the module and builds a type table mapping the index in
// the vector to the type represented by the type ordinal.
std::vector<TypeDef> buildTypeTable(IREE::VM::ModuleOp moduleOp) {
  llvm::DenseMap<Type, std::string> typeMap;
  std::function<void(Type)> tryInsertType;
  tryInsertType = [&](Type type) {
    if (auto refPtrType = llvm::dyn_cast<IREE::VM::RefType>(type)) {
      type = refPtrType.getObjectType();
    }
    if (typeMap.count(type))
      return;
    std::string str;
    llvm::raw_string_ostream sstream(str);
    type.print(sstream);
    sstream.flush();
    typeMap.try_emplace(type, str);
    if (auto listType = llvm::dyn_cast<IREE::VM::ListType>(type)) {
      assert(listType.getElementType());
      tryInsertType(listType.getElementType());
    }
  };
  for (auto funcOp : moduleOp.getBlock().getOps<IREE::VM::FuncOp>()) {
    funcOp.walk([&](Operation *op) {
      for (auto type : op->getOperandTypes())
        tryInsertType(type);
      for (auto type : op->getResultTypes())
        tryInsertType(type);
    });
  }

  std::vector<TypeDef> table;
  table.reserve(typeMap.size());
  for (const auto &typeString : typeMap) {
    table.push_back(TypeDef{typeString.first, typeString.second});
  }
  llvm::stable_sort(
      table, +[](const TypeDef &lhs, const TypeDef &rhs) {
        // Always sort builtins above custom types.
        if (lhs.full_name[0] != '!' && rhs.full_name[0] == '!') {
          return true;
        } else if (lhs.full_name[0] == '!' && rhs.full_name[0] != '!') {
          return false;
        }
        return lhs.full_name.compare(rhs.full_name) < 0;
      });
  return table;
}

} // namespace mlir::iree_compiler::IREE::VM
