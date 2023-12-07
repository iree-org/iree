// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/DropExcludedExports.h"

#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::VM {

class DropExcludedExportsPass
    : public PassWrapper<DropExcludedExportsPass,
                         OperationPass<IREE::VM::ModuleOp>> {
public:
  StringRef getArgument() const override {
    return "iree-vm-drop-excluded-exports";
  }

  StringRef getDescription() const override {
    return "Deletes exports if annotated with emitc.exclude.";
  }

  void runOnOperation() override {
    // Remove exports annotated with emitc.exclude.
    SmallVector<Operation *> opsToRemove;
    getOperation()->walk([&](IREE::VM::ExportOp exportOp) {
      Operation *op = exportOp.getOperation();
      if (op->hasAttr("emitc.exclude")) {
        opsToRemove.push_back(op);
      }
    });

    for (auto op : opsToRemove) {
      op->erase();
    }
  }
};

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createDropExcludedExportsPass() {
  return std::make_unique<DropExcludedExportsPass>();
}

static PassRegistration<DropExcludedExportsPass> pass;

} // namespace mlir::iree_compiler::IREE::VM
