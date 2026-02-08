// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/BuiltinRegistry.h"

namespace mlir::iree_compiler {

void BuiltinRegistry::add(StringRef name, FunctionType type,
                          BodyBuilder bodyBuilder) {
  assert(!nameToIndex.count(name) && "duplicate builtin registration");
  unsigned index = builtins.size();
  builtins.push_back({name.str(), type, std::move(bodyBuilder),
                      /*funcOp=*/nullptr,
                      /*useCount=*/0});
  nameToIndex[builtins[index].name] = index;
}

void BuiltinRegistry::declareAll(Operation *moduleOp) {
  OpBuilder builder(moduleOp->getContext());
  auto &moduleBlock = moduleOp->getRegion(0).front();
  builder.setInsertionPointToStart(&moduleBlock);
  auto location = moduleOp->getLoc();
  for (auto &builtin : builtins) {
    auto funcOp =
        IREE::VM::FuncOp::create(builder, location, builtin.name, builtin.type);
    SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
    builtin.funcOp = funcOp;
  }
}

IREE::VM::FuncOp BuiltinRegistry::use(StringRef name) {
  auto iterator = nameToIndex.find(name);
  assert(iterator != nameToIndex.end() && "unknown builtin");
  auto &builtin = builtins[iterator->second];
  ++builtin.useCount;
  return builtin.funcOp;
}

void BuiltinRegistry::finalize() {
  for (auto &builtin : builtins) {
    if (builtin.useCount == 0) {
      builtin.funcOp->erase();
      builtin.funcOp = nullptr;
    } else {
      OpBuilder builder(builtin.funcOp);
      builtin.bodyBuilder(builtin.funcOp, builder);
    }
  }
}

} // namespace mlir::iree_compiler
