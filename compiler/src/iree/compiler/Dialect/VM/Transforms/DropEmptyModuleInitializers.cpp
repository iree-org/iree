// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::VM {

namespace {

// Returns true if |funcOp| is empty.
static bool isFuncEmpty(IREE::VM::FuncOp funcOp) {
  return funcOp.empty() ||
         (&funcOp.front() == &funcOp.back() &&
          &funcOp.front().front() == funcOp.front().getTerminator());
}

} // namespace

class DropEmptyModuleInitializersPass
    : public PassWrapper<DropEmptyModuleInitializersPass,
                         OperationPass<IREE::VM::ModuleOp>> {
public:
  StringRef getArgument() const override {
    return "iree-vm-drop-empty-module-initializers";
  }

  StringRef getDescription() const override {
    return "Drops __init/__deinit functions that have no ops.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // Find all export ops so they are easier to remove.
    DenseMap<StringRef, IREE::VM::ExportOp> exportOps;
    for (auto exportOp : moduleOp.getOps<IREE::VM::ExportOp>()) {
      exportOps[exportOp.getExportName()] = exportOp;
    }

    // Check @__init:
    auto initFuncOp = symbolTable.lookup<IREE::VM::FuncOp>("__init");
    if (initFuncOp && isFuncEmpty(initFuncOp)) {
      auto exportOp = exportOps[initFuncOp.getName()];
      if (exportOp)
        exportOp.erase();
      initFuncOp.erase();
    }

    // Check @__deinit:
    auto deinitFuncOp = symbolTable.lookup<IREE::VM::FuncOp>("__deinit");
    if (deinitFuncOp && isFuncEmpty(deinitFuncOp)) {
      auto exportOp = exportOps[deinitFuncOp.getName()];
      if (exportOp)
        exportOp.erase();
      deinitFuncOp.erase();
    }
  }
};

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createDropEmptyModuleInitializersPass() {
  return std::make_unique<DropEmptyModuleInitializersPass>();
}

static PassRegistration<DropEmptyModuleInitializersPass> pass;

} // namespace mlir::iree_compiler::IREE::VM
