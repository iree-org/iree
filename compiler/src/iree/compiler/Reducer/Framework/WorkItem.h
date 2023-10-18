// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_REDUCER_WORK_ITEM_H
#define IREE_COMPILER_REDUCER_WORK_ITEM_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler::Reducer {

class WorkItem {
public:
  WorkItem(ModuleOp root) : root(root) {}

  const ModuleOp getModule() { return root; }
  const OpBuilder getBuilder() { return OpBuilder(root); }

  /// TODO(Groverkss): Ownership of module should be conveyed here via
  /// mlir::OwningOpReference<ModuleOp>.
  void replaceModule(ModuleOp newModule) {
    if (root)
      root->erase();
    root = newModule;
  }

  LogicalResult verify() { return root.verify(); }

  WorkItem clone();

private:
  ModuleOp root;
};

} // namespace mlir::iree_compiler::Reducer

#endif // IREE_COMPILER_REDUCER_WORK_ITEM_H
