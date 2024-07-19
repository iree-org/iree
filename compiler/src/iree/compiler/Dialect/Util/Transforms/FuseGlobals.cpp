// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <iterator>

#include "iree/compiler/Dialect/Util/Analysis/GlobalTable.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-util-fuse-globals"

namespace mlir::iree_compiler::IREE::Util {
namespace {

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     llvm::BitVector &bits) {
  for (unsigned i = 0; i < bits.size(); ++i) {
    os << (bits.test(i) ? "1" : "0");
  }
  return os;
}

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
class FuseGlobalsPass : public FuseGlobalsBase<FuseGlobalsPass> {
public:
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
      std::unique_ptr<AsmState> asmState;
      LLVM_DEBUG({
        asmState = std::make_unique<AsmState>(callableOp);
        llvm::dbgs() << "FuseGlobals: analyzing ";
        callableOp.print(llvm::dbgs(), *asmState);
        llvm::dbgs() << ":\n";
      });
      auto *region = callableOp.getCallableRegion();
      if (!region)
        continue;
      for (auto &block : *region) {
        DenseMap<Value, SmallVector<IREE::Util::GlobalStoreOpInterface>>
            valueStores;
        for (auto storeOp :
             block.getOps<IREE::Util::GlobalStoreOpInterface>()) {
          auto &global = globalTable.lookup(storeOp.getGlobalName());
          LLVM_DEBUG({
            llvm::dbgs() << " - store #" << global.ordinal << ": ";
            storeOp.print(llvm::dbgs(), *asmState);
            llvm::dbgs() << "; candidate=" << global.isCandidate() << "\n";
          });
          if (!global.isCandidate())
            continue;
          valueStores[storeOp.getStoredGlobalValue()].push_back(storeOp);
        }
        for (auto valueStore : valueStores) {
          LLVM_DEBUG({
            llvm::dbgs() << "= storing value ";
            valueStore.first.printAsOperand(llvm::dbgs(), *asmState);
            llvm::dbgs() << ":\n";
            for (auto storeOp : valueStore.second) {
              llvm::dbgs() << " => @" << storeOp.getGlobalName() << "\n";
            }
          });
          tempBits.reset();
          for (auto storeOp : valueStore.second) {
            auto &global = globalTable.lookup(storeOp.getGlobalName());
            tempBits.set(global.ordinal);
          }
          for (auto storeOp : valueStore.second) {
            auto entry = correlationMap.find(storeOp.getGlobalName());
            if (entry == correlationMap.end()) {
              correlationMap.insert(
                  std::make_pair(storeOp.getGlobalName(), tempBits));
            } else {
              entry->second &= tempBits;
            }
          }
        }
      }
    }

    // Resolve which globals are always set to the same value.
    // This ensures that if @a is set to @b that @b is also set to @a.
    // TODO(benvanik): find a better data structure that avoids the need for
    // this cleanup step. We should be able to do this during construction.
    for (auto it : correlationMap) {
      auto globalName = it.first;
      auto &correlationBits = it.second;
      auto &global = globalTable.lookup(globalName);
      llvm::BitVector tempBits = correlationBits;
      for (auto ordinal : correlationBits.set_bits()) {
        auto otherGlobalName = globalTable.lookupByOrdinal(ordinal);
        if (otherGlobalName == globalName) {
          continue;
        }
        auto &otherBits = correlationMap[otherGlobalName];
        if (!otherBits.test(global.ordinal)) {
          LLVM_DEBUG(llvm::dbgs() << "Fixup: " << globalName
                                  << " uncorrelated with " << otherGlobalName
                                  << ", masking off " << otherBits << "\n");
          tempBits.reset(otherBits);
        } else {
          tempBits &= otherBits;
        }
      }
      correlationMap[globalName] = tempBits;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "FuseGlobals correlation maps:\n";
      for (auto it : correlationMap) {
        auto globalName = it.first;
        auto &correlationBits = it.second;
        auto &global = globalTable.lookup(globalName);
        llvm::dbgs() << "= #" << global.ordinal << " " << global.getName()
                     << " = " << correlationBits << ":\n";
        for (auto ordinal : correlationBits.set_bits()) {
          llvm::dbgs() << "  => " << globalTable.lookupByOrdinal(ordinal)
                       << "\n";
        }
      }
    });

    // Build equivalence classes for each global, giving us nice clustered sets.
    // We could probably fold this with the step above but my head hurts.
    llvm::EquivalenceClasses<StringRef> ec;
    for (auto it : correlationMap) {
      auto globalName = it.first;
      auto &correlationBits = it.second;
      auto &global = globalTable.lookup(globalName);
      for (auto ordinal : correlationBits.set_bits()) {
        ec.unionSets(global.getName(), globalTable.lookupByOrdinal(ordinal));
      }
    }

    // Build the sets of fusable globals. We use the equivalence classes we
    // built above to know which globals _should_ fuse, and this check lets us
    // filter out globals that _cannot_ fuse; such as when their initializers
    // differ.
    SmallVector<SmallVector<Global *>> fusableSets;
    for (auto it = ec.begin(), end = ec.end(); it != end; ++it) {
      if (!it->isLeader()) {
        continue; // Ignore non-leader sets.
      }
      if (++ec.member_begin(it) == ec.member_end()) {
        continue; // size 1
      }
      DenseMap<Attribute, SmallVector<Global *>> initialValueMap;
      for (auto mi = ec.member_begin(it); mi != ec.member_end(); ++mi) {
        Global &global = globalTable.lookup(*mi);
        initialValueMap[global.op.getGlobalInitialValue()].push_back(&global);
      }
      for (auto it : initialValueMap) {
        fusableSets.push_back(std::move(it.second));
      }
    }

    // For each foldable set combine into a single global and update all uses.
    SymbolTable symbolTable(moduleOp);
    SmallVector<StringRef> deadGlobalNames;
    for (auto &fusableSet : fusableSets) {
      auto *baseGlobal = fusableSet.front();
      LLVM_DEBUG(llvm::dbgs()
                 << "Fusing " << fusableSet.size() << " globals into "
                 << baseGlobal->getName() << "\n");

      // Build fused location from all of the globals.
      SmallVector<Location> locs;
      for (auto *global : fusableSet) {
        locs.push_back(global->op.getLoc());
        if (global->ordinal < baseGlobal->ordinal) {
          baseGlobal = global;
        }
      }
      auto fusedLoc = FusedLoc::get(moduleOp.getContext(), locs);

      // Update base global location.
      IREE::Util::GlobalOpInterface baseGlobalOp = baseGlobal->op;
      baseGlobalOp->setLoc(fusedLoc);

      // Replace all globals to point at the new one.
      for (auto *global : fusableSet) {
        if (global->op == baseGlobalOp) {
          continue;
        }
        globalTable.renameGlobalUses(*global, *baseGlobal);
        deadGlobalNames.push_back(global->getName());
      }
    }
    for (auto globalName : deadGlobalNames) {
      globalTable.eraseGlobal(globalName);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createFuseGlobalsPass() {
  return std::make_unique<FuseGlobalsPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
