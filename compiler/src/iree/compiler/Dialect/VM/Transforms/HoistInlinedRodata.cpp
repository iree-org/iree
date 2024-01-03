// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::VM {

class HoistInlinedRodataPass
    : public PassWrapper<HoistInlinedRodataPass,
                         OperationPass<IREE::VM::ModuleOp>> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VM::VMDialect>();
  }

  StringRef getArgument() const override {
    return "iree-vm-hoist-inlined-rodata";
  }

  StringRef getDescription() const override {
    return "Hoists inline vm.rodata.inline values to module-level constant "
           "storage.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable moduleSymbolTable(moduleOp);

    // Find all inline byte buffers in the module.
    SmallVector<IREE::VM::RodataInlineOp> inlineOps;
    moduleOp.walk([&](IREE::VM::RodataInlineOp inlineOp) {
      inlineOps.push_back(inlineOp);
    });

    for (auto inlineOp : inlineOps) {
      auto *parentOp = findParentContainer(inlineOp);
      OpBuilder moduleBuilder(moduleOp.getContext());
      if (parentOp) {
        moduleBuilder.setInsertionPoint(parentOp);
      } else {
        moduleBuilder.setInsertionPointToStart(&moduleOp.getBlock());
      }
      auto rodataOp = moduleBuilder.create<IREE::VM::RodataOp>(
          inlineOp.getLoc(), inferConstantName(parentOp, inlineOp),
          inlineOp.getValue());
      if (auto alignmentAttr = inlineOp.getAlignmentAttr()) {
        rodataOp.setAlignmentAttr(alignmentAttr);
      }
      if (auto mimeTypeAttr = inlineOp.getMimeTypeAttr()) {
        rodataOp.setMimeTypeAttr(mimeTypeAttr);
      }
      moduleSymbolTable.insert(rodataOp);
      rodataOp.setPrivate();
      replaceInlineOpWithRodataRef(inlineOp, rodataOp);
    }
  }

private:
  Operation *findParentContainer(IREE::VM::RodataInlineOp inlineOp) {
    if (auto parentOp = inlineOp->getParentOfType<IREE::VM::InitializerOp>()) {
      return parentOp;
    } else if (auto parentOp = inlineOp->getParentOfType<IREE::VM::FuncOp>()) {
      return parentOp;
    }
    return nullptr;
  }

  std::string inferConstantName(Operation *parentOp,
                                IREE::VM::RodataInlineOp inlineOp) {
    if (auto nameAttr = inlineOp.getNameAttr()) {
      return nameAttr.str();
    }
    if (auto symbolOp = dyn_cast<SymbolOpInterface>(parentOp)) {
      return (symbolOp.getName() + "_const").str();
    }
    return "_const";
  }

  // Replaces a vm.rodata.inline op with a vm.const.ref.rodata op that
  // references the module-level |rodataOp|.
  void replaceInlineOpWithRodataRef(IREE::VM::RodataInlineOp inlineOp,
                                    IREE::VM::RodataOp rodataOp) {
    OpBuilder builder(inlineOp);
    auto refOp =
        builder.create<IREE::VM::ConstRefRodataOp>(inlineOp.getLoc(), rodataOp);
    inlineOp.replaceAllUsesWith(refOp.getValue());
    inlineOp.erase();
  }
};

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createHoistInlinedRodataPass() {
  return std::make_unique<HoistInlinedRodataPass>();
}

static PassRegistration<HoistInlinedRodataPass> pass;

} // namespace mlir::iree_compiler::IREE::VM
