// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

class HoistInlinedRodataPass
    : public PassWrapper<HoistInlinedRodataPass,
                         OperationPass<IREE::VM::ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREEDialect>();
    registry.insert<IREE::VM::VMDialect>();
  }

  StringRef getArgument() const override {
    return "iree-vm-hoist-inlined-rodata";
  }

  StringRef getDescription() const override {
    return "Hoists inline iree.byte_buffer values to module-level constant "
           "storage.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable moduleSymbolTable(moduleOp);

    // Find all inline byte buffers in the module.
    auto funcOps = llvm::to_vector<4>(moduleOp.getOps<IREE::VM::FuncOp>());
    for (auto funcOp : funcOps) {
      auto inlineOps =
          llvm::to_vector<4>(funcOp.getOps<IREE::VM::RodataInlineOp>());
      if (inlineOps.empty()) continue;

      OpBuilder moduleBuilder(moduleOp.getContext());
      moduleBuilder.setInsertionPoint(funcOp);
      for (auto inlineOp : inlineOps) {
        std::string name = inlineOp.name().hasValue()
                               ? inlineOp.name().getValue().str()
                               : (funcOp.getName() + "_const").str();
        auto rodataOp = OpBuilder(moduleOp.getContext())
                            .create<IREE::VM::RodataOp>(inlineOp.getLoc(), name,
                                                        inlineOp.value());
        if (inlineOp.alignmentAttr()) {
          rodataOp.alignmentAttr(inlineOp.alignmentAttr());
        }
        moduleSymbolTable.insert(rodataOp, moduleBuilder.getInsertionPoint());
        rodataOp.setPrivate();
        replaceInlineOpWithRodataRef(inlineOp, rodataOp);
      }
    }
  }

 private:
  // Replaces a vm.rodata.inline op with a vm.const.ref.rodata op that
  // references the module-level |rodataOp|.
  void replaceInlineOpWithRodataRef(IREE::VM::RodataInlineOp inlineOp,
                                    IREE::VM::RodataOp rodataOp) {
    OpBuilder builder(inlineOp);
    auto refOp =
        builder.create<IREE::VM::ConstRefRodataOp>(inlineOp.getLoc(), rodataOp);
    inlineOp.replaceAllUsesWith(refOp.value());
    inlineOp.erase();
  }
};

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createHoistInlinedRodataPass() {
  return std::make_unique<HoistInlinedRodataPass>();
}

static PassRegistration<HoistInlinedRodataPass> pass;

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
