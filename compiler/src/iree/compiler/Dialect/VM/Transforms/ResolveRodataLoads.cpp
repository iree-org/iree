// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
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

// TODO(benvanik): replace this entire pass with generic IPO - the rodata refs
// are kind of constant like and should be trivial to inline, though they can't
// be ConstantLike and will need a new interface so that IPO can materialize
// ops. It's also possible we could use the dialect interface for materializing
// constants to do that, though.

// Returns the vm.rodata that is stored into the global.
// Returns nullptr if the rodata values stored differ across multiple stores.
static IREE::VM::RodataOp findUniformlyStoredRodata(
    Explorer &explorer, const Explorer::GlobalInfo *globalInfo) {
  // This will be the first op found; we'll use it to lookup the rodata.
  IREE::VM::RodataOp uniformRodataOp;
  for (auto storeOp : globalInfo->getStores()) {
    auto storedValue = storeOp.getStoredGlobalValue();
    if (explorer.walkDefiningOps(storedValue, [&](OpResult result) {
          if (auto refRodataOp = dyn_cast<IREE::VM::ConstRefRodataOp>(
                  result.getDefiningOp())) {
            if (!uniformRodataOp) {
              uniformRodataOp =
                  explorer.getSymbolTables()
                      .lookupNearestSymbolFrom<IREE::VM::RodataOp>(
                          refRodataOp, refRodataOp.getRodataAttr());
            } else if (refRodataOp.getRodata() != uniformRodataOp.getName()) {
              uniformRodataOp = nullptr;
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        }) == TraversalResult::INCOMPLETE) {
      // Unanalyzable.
      uniformRodataOp = nullptr;
    }
  }
  return uniformRodataOp;
}

// Performs inlining of vm.global.ref accesses to !vm.buffers that originate
// from vm.rodata ops. We check the stores to ensure they all point to the same
// vm.rodata and then rewrite all loads to use it.
static void processBufferGlobal(Explorer &explorer,
                                const Explorer::GlobalInfo *globalInfo,
                                DenseSet<Operation *> &deadOps) {
  // Ignore indirect/unanalyzable globals.
  if (globalInfo->isIndirect) return;
  // Ignore mutable globals, as they could be changed to various values.
  if (globalInfo->op.isGlobalMutable()) return;

  // If there are no stores to the global then it's always null.
  if (globalInfo->getStores().empty()) {
    for (auto loadOp : globalInfo->getLoads()) {
      OpBuilder builder(loadOp);
      auto loadedValue = loadOp.getLoadedGlobalValue();
      auto zeroRefOp = builder.create<IREE::VM::ConstRefZeroOp>(
          loadOp.getLoc(), loadedValue.getType());
      loadedValue.replaceAllUsesWith(zeroRefOp.getResult());
      deadOps.insert(loadOp);
    }
    return;
  }

  // Try to get the vm.rodata that is stored into the global uniformly across
  // the program (there may be multiple initializers or control flow that
  // determines the stored value).
  auto rodataOp = findUniformlyStoredRodata(explorer, globalInfo);
  if (!rodataOp) return;

  // All stores to the global are of the same rodata.
  // Replace all of the loads with direct references to the rodata and then
  // erase them.
  for (auto loadOp : globalInfo->getLoads()) {
    OpBuilder builder(loadOp);
    auto rodataRefOp =
        builder.create<IREE::VM::ConstRefRodataOp>(loadOp.getLoc(), rodataOp);
    auto loadedValue = loadOp.getLoadedGlobalValue();
    loadedValue.replaceAllUsesWith(rodataRefOp.getResult());
    deadOps.insert(loadOp);
  }

  // Remove the stores as they shouldn't be needed. This makes SymbolDCE easier.
  for (auto storeOp : globalInfo->getStores()) {
    deadOps.insert(storeOp);
  }
}

class ResolveRodataLoadsPass
    : public PassWrapper<ResolveRodataLoadsPass,
                         OperationPass<IREE::VM::ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VM::VMDialect>();
  }

  StringRef getArgument() const override {
    return "iree-vm-resolve-rodata-loads";
  }

  StringRef getDescription() const override {
    return "Resolves global loads of rodata ops to direct rodata references.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    Explorer explorer(moduleOp, TraversalAction::SHALLOW);
    explorer.setOpAction<IREE::VM::InitializerOp>(TraversalAction::RECURSE);
    explorer.setOpAction<IREE::VM::FuncOp>(TraversalAction::RECURSE);
    explorer.initialize();

    // Walk all !vm.buffer globals and process them (if possible).
    // Note that this pass mutates the module IR but only by dropping
    // loads/stores to the globals and leaves the globals for SymbolDCE.
    DenseSet<Operation *> deadOps;
    explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
      if (auto refType = llvm::dyn_cast<IREE::VM::RefType>(
              globalInfo->op.getGlobalType())) {
        if (llvm::isa<IREE::VM::BufferType>(refType.getObjectType())) {
          processBufferGlobal(explorer, globalInfo, deadOps);
        }
      }
    });

    // Erase all ops after we're done iterating them.
    for (auto *deadOp : deadOps) deadOp->erase();
  }
};

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createResolveRodataLoadsPass() {
  return std::make_unique<ResolveRodataLoadsPass>();
}

static PassRegistration<ResolveRodataLoadsPass> pass;

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
