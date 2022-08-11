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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

class SinkGlobalBufferLoadsPass
    : public PassWrapper<SinkGlobalBufferLoadsPass,
                         OperationPass<IREE::VM::ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VM::VMDialect>();
  }

  StringRef getArgument() const override {
    return "iree-vm-sink-global-buffer-loads";
  }

  StringRef getDescription() const override {
    return "Sinks global buffer references into loads.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTableCollection symbolTable;

    // Find all vm.global.store.ref ops in the module and note the ones that
    // meet our requirements:
    //   - Is in an initializer
    //   - Storing to an immutable global
    //   - Sourcing from a vm.const.ref.rodata or vm.const.ref.zero op
    //
    // We will set an info with a null initializerOp if there is a store but
    // an unrecognized source.
    struct GlobalInitInfo {
      IREE::VM::GlobalStoreRefOp storeOp;
      Operation *initializerOp;
    };
    DenseMap<IREE::VM::GlobalRefOp, GlobalInitInfo> globalInitInfos;
    moduleOp.walk([&](IREE::VM::GlobalStoreRefOp storeOp) {
      if (!storeOp->getParentOfType<IREE::VM::InitializerOp>()) {
        return;
      }
      // Only consider it a constant for a couple of cases.
      Operation *initializerOp = storeOp.getValue().getDefiningOp();
      if (initializerOp &&
          !llvm::isa<IREE::VM::ConstRefRodataOp, IREE::VM::ConstRefZeroOp>(
              initializerOp)) {
        initializerOp = nullptr;
      }
      auto globalOp =
          symbolTable.lookupNearestSymbolFrom<IREE::VM::GlobalRefOp>(
              storeOp->getParentOp(), storeOp.getGlobalAttr());
      if (globalOp) {
        if (!globalOp.isGlobalMutable()) {
          globalInitInfos[globalOp] = GlobalInitInfo{storeOp, initializerOp};
        }
      }
    });

    // Walk over all loads and update.
    moduleOp.walk([&](IREE::VM::GlobalLoadRefOp loadOp) {
      auto globalOp =
          symbolTable.lookupNearestSymbolFrom<IREE::VM::GlobalRefOp>(
              loadOp->getParentOp(), loadOp.getGlobalAttr());
      auto it = globalInitInfos.find(globalOp);
      if (it != globalInitInfos.end()) {
        auto &info = it->second;
        if (info.initializerOp) {
          // We are sourced from a constant. Clone/replace.
          OpBuilder builder(loadOp);
          Operation *newOp = builder.clone(*info.initializerOp);
          loadOp.replaceAllUsesWith(newOp);
          loadOp.erase();
        }
        return;
      }

      // Still here? If the global is immutable, we can replace with null.
      // (i.e. there is no initializing store)
      if (!globalOp.isGlobalMutable()) {
        OpBuilder builder(loadOp);
        Value zero = builder.create<IREE::VM::ConstRefZeroOp>(
            loadOp.getLoc(), loadOp.getResult().getType());
        loadOp.replaceAllUsesWith(zero);
        loadOp.erase();
      }
    });

    // Erase initializers no longer needed.
    for (auto it : globalInitInfos) {
      auto global = it.first;
      auto &info = it.second;
      if (info.initializerOp) {
        info.storeOp.erase();
        global.erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createSinkGlobalBufferLoadsPass() {
  return std::make_unique<SinkGlobalBufferLoadsPass>();
}

static PassRegistration<SinkGlobalBufferLoadsPass> pass;

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
