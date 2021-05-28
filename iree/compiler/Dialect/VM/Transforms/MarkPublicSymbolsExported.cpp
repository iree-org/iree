// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

class MarkPublicSymbolsExportedPass
    : public PassWrapper<MarkPublicSymbolsExportedPass,
                         OperationPass<mlir::ModuleOp>> {
 public:
  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<mlir::FuncOp>()) {
      if (funcOp.isPublic()) {
        funcOp->setAttr("iree.module.export", UnitAttr::get(&getContext()));
      }
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createMarkPublicSymbolsExportedPass() {
  return std::make_unique<MarkPublicSymbolsExportedPass>();
}

static PassRegistration<MarkPublicSymbolsExportedPass> pass(
    "iree-vm-mark-public-symbols-exported",
    "Sets public visiblity symbols to have the iree.module.export attribute");

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
