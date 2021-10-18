// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <iterator>

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-util-fuse-globals"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {
namespace {

struct Global {
  size_t ordinal = 0;
  IREE::Util::GlobalOp op;
  bool isIndirect = false;
  SmallVector<IREE::Util::GlobalLoadOp> loadOps;
  SmallVector<IREE::Util::GlobalStoreOp> storeOps;

  bool isCandidate() {
    return !isIndirect && op.isPrivate() && !op->hasAttr("noinline");
  }
};

struct GlobalTable {
  mlir::ModuleOp moduleOp;
  SmallVector<StringRef> globalOrder;
  DenseMap<StringRef, Global> globalMap;

  size_t size() const { return globalOrder.size(); }

  explicit GlobalTable(mlir::ModuleOp moduleOp) : moduleOp(moduleOp) {
    rebuild();
  }

  void rebuild() {
    globalOrder.clear();
    globalMap.clear();
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOp>()) {
      globalMap[globalOp.getName()] = Global{globalOrder.size(), globalOp};
      globalOrder.push_back(globalOp.getName());
    }
    for (auto callableOp : moduleOp.getOps<CallableOpInterface>()) {
      callableOp.walk([&](Operation *op) {
        if (auto addressOp = dyn_cast<IREE::Util::GlobalAddressOp>(op)) {
          globalMap[addressOp.global()].isIndirect = true;
        } else if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOp>(op)) {
          globalMap[loadOp.global()].loadOps.push_back(loadOp);
        } else if (auto storeOp = dyn_cast<IREE::Util::GlobalStoreOp>(op)) {
          globalMap[storeOp.global()].storeOps.push_back(storeOp);
        }
      });
    }
  }
};

// Fuses globals that are always set to the same value into one.
//
// Example:
//  util.global mutable @a : i32
//  util.global mutable @b : i32
//  builtin.func @foo(%arg0: i32) {
//    util.global.store %arg0, @a : i32
//    util.global.store %arg0, @b : i32
// ->
//  util.global mutable @fused : i32
//  builtin.func @foo(%arg0: i32) {
//    util.global.store %arg0, @fused : i32
class FuseGlobalsPass
    : public PassWrapper<FuseGlobalsPass, OperationPass<mlir::ModuleOp>> {
 public:
  StringRef getArgument() const override { return "iree-util-fuse-globals"; }

  StringRef getDescription() const override {
    return "Fuses correlated globals together.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    GlobalTable globalTable(moduleOp);

    // Build a map of global symbol to a bitvector indicating which globals are
    // stored with the same values in all instances.
    // This is done by walking the values stored into globals and ANDing a
    // bitmask of the other globals stored with the same value.
    //
    // Note that we are only looking for stores within the same block - we
    // expect other canonicalizations to have moved stores into the same block
    // that are guaranteed to be on the same execution path.
    DenseMap<StringRef, llvm::BitVector> correlationMap;
    llvm::BitVector tempBits(globalTable.size());
    for (auto callableOp : moduleOp.getOps<CallableOpInterface>()) {
      for (auto &block : *callableOp.getCallableRegion()) {
        DenseMap<Value, SmallVector<IREE::Util::GlobalStoreOp>> valueStores;
        for (auto storeOp : block.getOps<IREE::Util::GlobalStoreOp>()) {
          auto &global = globalTable.globalMap[storeOp.global()];
          if (!global.isCandidate()) continue;
          valueStores[storeOp.value()].push_back(storeOp);
        }
        for (auto valueStore : valueStores) {
          tempBits.reset();
          for (auto storeOp : valueStore.second) {
            auto &global = globalTable.globalMap[storeOp.global()];
            tempBits.set(global.ordinal);
          }
          for (auto storeOp : valueStore.second) {
            auto entry = correlationMap.find(storeOp.global());
            if (entry == correlationMap.end()) {
              correlationMap.insert(std::make_pair(storeOp.global(), tempBits));
            } else {
              entry->second &= tempBits;
            }
          }
        }
      }
    }

    // Build equivalence classes for each global, giving us nice clustered sets.
    // We could probably fold this with the step above but my head hurts.
    llvm::EquivalenceClasses<StringRef> ec;
    for (auto it : correlationMap) {
      auto globalName = it.first;
      auto &correlationBits = it.second;
      auto &global = globalTable.globalMap[globalName];
      for (auto ordinal : correlationBits.set_bits()) {
        ec.unionSets(global.op.getName(), globalTable.globalOrder[ordinal]);
      }
    }

    // Build the sets of fusable globals. We use the equivalence classes we
    // built above to know which globals _should_ fuse, and this check lets us
    // filter out globals that _cannot_ fuse; such as when their initializers
    // differ.
    SmallVector<SmallVector<Global *>> fusableSets;
    for (auto it = ec.begin(), end = ec.end(); it != end; ++it) {
      if (!it->isLeader()) continue;  // Ignore non-leader sets.
      if (++ec.member_begin(it) == ec.member_end()) continue;  // size 1
      DenseMap<Attribute, SmallVector<Global *>> initialValueMap;
      for (auto mi = ec.member_begin(it); mi != ec.member_end(); ++mi) {
        Global &global = globalTable.globalMap[*mi];
        initialValueMap[global.op.initial_valueAttr()].push_back(&global);
      }
      for (auto it : initialValueMap) {
        fusableSets.push_back(std::move(it.second));
      }
    }

    // For each foldable set combine into a single global and update all uses.
    SymbolTable symbolTable(moduleOp);
    for (auto &fusableSet : fusableSets) {
      IREE::Util::GlobalOp baseGlobalOp = fusableSet.front()->op;

      // Build fused location from all of the globals.
      SmallVector<Location> locs;
      for (auto *global : fusableSet) {
        locs.push_back(global->op.getLoc());
        if (global->op->isBeforeInBlock(baseGlobalOp)) {
          baseGlobalOp = global->op;
        }
      }
      auto fusedLoc = FusedLoc::get(baseGlobalOp.getContext(), locs);

      // Update base global location.
      baseGlobalOp->setLoc(fusedLoc);

      // Replace all globals to point at the new one.
      auto baseGlobalNameAttr = FlatSymbolRefAttr::get(
          baseGlobalOp.getContext(), baseGlobalOp.getSymbolName());
      for (auto *global : fusableSet) {
        if (global->op == baseGlobalOp) continue;

        // Redirect all loads to the new fused global.
        for (auto loadOp : global->loadOps) {
          loadOp.globalAttr(baseGlobalNameAttr);
        }

        // Remove all stores to all variables but the base.
        for (auto storeOp : global->storeOps) {
          storeOp.erase();
        }

        global->op.erase();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createFuseGlobalsPass() {
  return std::make_unique<FuseGlobalsPass>();
}

static PassRegistration<FuseGlobalsPass> pass;

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
