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
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-util-fold-globals"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {
namespace {

template <typename R>
static size_t count(R &&range) {
  return std::distance(range.begin(), range.end());
}

struct Global {
  size_t ordinal = 0;
  IREE::Util::GlobalOp op;
  bool isIndirect = false;
  SmallVector<IREE::Util::GlobalLoadOp> loadOps;
  SmallVector<IREE::Util::GlobalStoreOp> storeOps;
};

enum class GlobalAction {
  PRESERVE,
  UPDATE,
  DELETE,
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

  bool forEach(std::function<GlobalAction(Global &global)> fn) {
    bool didChange = false;
    for (size_t i = 0; i < size();) {
      auto globalName = globalOrder[i];
      auto action = fn(globalMap[globalName]);
      switch (action) {
        case GlobalAction::PRESERVE: {
          ++i;
          break;
        }
        case GlobalAction::UPDATE: {
          didChange |= true;
          ++i;
          break;
        }
        case GlobalAction::DELETE: {
          didChange |= true;
          auto &global = globalMap[globalName];
          assert(!global.op.isPublic() && "can't delete public globals");
          assert(global.loadOps.empty() && "must not be used");
          for (auto storeOp : global.storeOps) {
            storeOp.erase();
          }
          global.op.erase();
          globalMap.erase(globalName);
          globalOrder.erase(globalOrder.begin() + i);
          break;
        }
      }
    }
    return didChange;
  }
};

// Inlines constant stores into global initializers if always stored to the
// same value.
//
// Example:
//  util.global mutable @a : i32
//  builtin.func @foo() {
//    %c5 = arith.constant 5 : i32
//    util.global.store %c5, @a : i32
//  }
//  builtin.func @bar() {
//    %c5 = arith.constant 5 : i32
//    util.global.store %c5, @a : i32
//  }
// ->
//  util.global @a = 5 : i32
static bool inlineConstantGlobalStores(GlobalTable &globalTable) {
  return globalTable.forEach([&](Global &global) {
    if (global.isIndirect) return GlobalAction::PRESERVE;

    // Find the constant value used in all stores.
    // All stores must match the initial value of the global _or_ the global
    // must be uninitialized. A proper dataflow analysis would let us identify
    // the cases where there are loads before stores and the implicit 0 initial
    // value for uninitialized globals would be observable.
    Attribute constantValue = global.op.initial_valueAttr();
    for (auto storeOp : global.storeOps) {
      Attribute valueAttr;
      if (!matchPattern(storeOp.value(), m_Constant(&valueAttr))) {
        constantValue = {};
        break;
      }
      if (!constantValue) {
        // First found.
        constantValue = valueAttr;
      } else if (constantValue != valueAttr) {
        // Non-uniform.
        constantValue = {};
        break;
      }
    }
    if (!constantValue) return GlobalAction::PRESERVE;

    // Propagate constant into the initial value. Note that there may have been
    // a previous initial value that is being replaced.
    global.op.initial_valueAttr(constantValue);

    // Remove all of the stores.
    for (auto storeOp : global.storeOps) {
      storeOp.erase();
    }
    global.storeOps.clear();

    return GlobalAction::UPDATE;
  });
}

// Renames globals that are always chained through each other.
//
// Example:
//  util.global mutable @chained0 : i32
//  util.global mutable @chained1 : i32
//  builtin.func @foo() {
//    %0 = util.global.load @chained0 : i32
//    util.global.store %0, @chained1 : i32
// ->
//  util.global.mutable @chained0 : i32
static bool renameChainedGlobals(GlobalTable &globalTable) {
  return globalTable.forEach([&](Global &global) {
    if (global.isIndirect) return GlobalAction::PRESERVE;
    if (global.op.isPublic()) return GlobalAction::PRESERVE;

    // Find the other symbol this global is chained with by looking for uniform
    // stores. Note that we don't care about initializers.
    FlatSymbolRefAttr aliasName;
    for (auto storeOp : global.storeOps) {
      // Check to see if the stored value comes from another global.
      auto *definingOp = storeOp.value().getDefiningOp();
      if (auto loadOp =
              dyn_cast_or_null<IREE::Util::GlobalLoadOp>(definingOp)) {
        if (!aliasName) {
          aliasName = loadOp.globalAttr();
        } else if (aliasName != loadOp.globalAttr()) {
          aliasName = {};
          break;
        }
      } else {
        aliasName = {};
        break;
      }
    }
    if (!aliasName) return GlobalAction::PRESERVE;

    // Replace all loads from the global with the aliased global.
    auto &aliasGlobal = globalTable.globalMap[aliasName.getValue()];
    for (auto loadOp : global.loadOps) {
      loadOp.globalAttr(aliasName);
      aliasGlobal.loadOps.push_back(loadOp);
    }
    global.loadOps.clear();

    // Erase all stores to the global.
    for (auto storeOp : global.storeOps) {
      storeOp.erase();
    }
    global.storeOps.clear();

    return GlobalAction::DELETE;
  });
}

// Updates globals to be immutable if they are only stored during
// initialization.
//
// Example:
//  util.global mutable @a : i32
//  util.initializer {
//    %c5 = arith.constant 5 : i32
//    util.global.store %c5, @a : i32
// ->
//  util.global @a = 5 : i32
static bool updateGlobalImmutability(GlobalTable &globalTable) {
  return globalTable.forEach([&](Global &global) {
    if (global.isIndirect) return GlobalAction::PRESERVE;
    if (global.op.isPublic()) return GlobalAction::PRESERVE;
    if (!global.storeOps.empty()) return GlobalAction::PRESERVE;
    if (!global.op.isMutable()) return GlobalAction::PRESERVE;
    global.op.removeIs_mutableAttr();
    return GlobalAction::UPDATE;
  });
}

// Inlines constant global values that are known to not change.
static bool inlineConstantGlobalLoads(GlobalTable &globalTable) {
  return globalTable.forEach([&](Global &global) {
    if (global.isIndirect) return GlobalAction::PRESERVE;
    if (!global.storeOps.empty()) return GlobalAction::PRESERVE;
    if (global.op.isMutable()) return GlobalAction::PRESERVE;
    if (global.op->hasAttr("noinline")) return GlobalAction::PRESERVE;
    if (!global.op.initial_valueAttr()) return GlobalAction::PRESERVE;

    if (global.op.type().isa<IREE::Util::ReferenceTypeInterface>()) {
      // We only inline value types; reference types have meaning as globals.
      return GlobalAction::PRESERVE;
    }

    // Inline initial value into all loads.
    for (auto loadOp : global.loadOps) {
      OpBuilder builder(loadOp);
      auto constantOp = builder.create<arith::ConstantOp>(
          loadOp.getLoc(), loadOp.result().getType(),
          global.op.initial_valueAttr());
      loadOp.replaceAllUsesWith(constantOp.getResult());
      loadOp.erase();
    }
    global.loadOps.clear();

    // Only delete if private.
    return global.op.isPrivate() ? GlobalAction::DELETE : GlobalAction::UPDATE;
  });
}

// Erases all globals in the module that are never loaded.
// This differs from SymbolDCE in that initializations and storage operations
// are discarded.
static bool eraseUnusedGlobals(GlobalTable &globalTable) {
  return globalTable.forEach([&](Global &global) {
    if (global.isIndirect) return GlobalAction::PRESERVE;
    if (global.op.isPublic()) return GlobalAction::PRESERVE;
    if (global.loadOps.empty()) {
      // No loads; remove entirely.
      return GlobalAction::DELETE;
    }
    return GlobalAction::PRESERVE;
  });
}

// Deduplicates immutable globals with constant initial values.
// This is a simplified and safer version of the global fusion pass.
static bool deduplicateConstantGlobals(GlobalTable &globalTable) {
  // Build sets of all equivalent globals.
  llvm::EquivalenceClasses<StringRef> ec;
  DenseMap<Attribute, StringRef> leaderMap;
  for (auto globalIt : globalTable.globalMap) {
    auto &global = globalIt.getSecond();
    if (global.isIndirect || global.op.isPublic() ||
        !global.op.initial_value().hasValue() || !global.storeOps.empty()) {
      // Not eligible for deduplication.
      continue;
    }
    auto it =
        leaderMap.insert({global.op.initial_valueAttr(), global.op.getName()});
    if (it.second) {
      // Inserted new.
      ec.insert(global.op.getName());
    } else {
      // Existing global with the same value.
      ec.unionSets(global.op.getName(), it.first->second);
    }
  }

  bool didChange = false;
  for (auto it = ec.begin(), end = ec.end(); it != end; ++it) {
    if (!it->isLeader()) continue;  // Ignore non-leader sets.
    if (++ec.member_begin(it) == ec.member_end()) continue;
    IREE::Util::GlobalOp baseGlobalOp = globalTable.globalMap[it->getData()].op;

    // Build fused location from all of the globals.
    SmallVector<Location> locs;
    for (auto mi = ec.member_begin(it); mi != ec.member_end(); ++mi) {
      Global &global = globalTable.globalMap[*mi];
      locs.push_back(global.op.getLoc());
      if (global.op->isBeforeInBlock(baseGlobalOp)) {
        baseGlobalOp = global.op;
      }
    }
    auto fusedLoc = FusedLoc::get(baseGlobalOp.getContext(), locs);

    // Update base global location.
    baseGlobalOp->setLoc(fusedLoc);

    // Replace all other globals to point at the new one.
    auto baseGlobalNameAttr = FlatSymbolRefAttr::get(
        baseGlobalOp.getContext(), baseGlobalOp.getSymbolName());
    for (auto mi = ec.member_begin(it); mi != ec.member_end(); ++mi) {
      Global &global = globalTable.globalMap[*mi];
      if (global.op == baseGlobalOp) continue;
      for (auto loadOp : global.loadOps) {
        loadOp.globalAttr(baseGlobalNameAttr);
      }
      global.op.erase();
    }

    didChange |= true;
  }

  if (didChange) {
    // We could keep the table up to date to avoid the rebuilds by merging all
    // loads into the base global.
    globalTable.rebuild();
  }
  return didChange;
}

class FoldGlobalsPass
    : public PassWrapper<FoldGlobalsPass, OperationPass<mlir::ModuleOp>> {
 public:
  explicit FoldGlobalsPass() = default;
  FoldGlobalsPass(const FoldGlobalsPass &pass) {}

  StringRef getArgument() const override { return "iree-util-fold-globals"; }

  StringRef getDescription() const override {
    return "Folds duplicate globals and propagates constants.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::arith::ArithmeticDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    OwningRewritePatternList patterns(context);

    for (auto *dialect : context->getLoadedDialects()) {
      dialect->getCanonicalizationPatterns(patterns);
    }
    for (auto op : context->getRegisteredOperations()) {
      op.getCanonicalizationPatterns(patterns, context);
    }

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    auto moduleOp = getOperation();
    beforeFoldingGlobals = count(moduleOp.getOps<IREE::Util::GlobalOp>());
    for (int i = 0; i < 10; ++i) {
      // TODO(benvanik): determine if we need this expensive folding.
      if (failed(applyPatternsAndFoldGreedily(moduleOp, frozenPatterns))) {
        signalPassFailure();
      }

      GlobalTable globalTable(moduleOp);
      bool didChange = false;

      LLVM_DEBUG(llvm::dbgs() << "==== inlineConstantGlobalStores ====\n");
      if (inlineConstantGlobalStores(globalTable)) {
        LLVM_DEBUG(moduleOp.dump());
        didChange = true;
      }

      LLVM_DEBUG(llvm::dbgs() << "==== renameChainedGlobals ====\n");
      if (renameChainedGlobals(globalTable)) {
        LLVM_DEBUG(moduleOp.dump());
        didChange = true;
      }

      LLVM_DEBUG(llvm::dbgs() << "==== updateGlobalImmutability ====\n");
      if (updateGlobalImmutability(globalTable)) {
        LLVM_DEBUG(moduleOp.dump());
        didChange = true;
      }

      LLVM_DEBUG(llvm::dbgs() << "==== inlineConstantGlobalLoads ====\n");
      if (inlineConstantGlobalLoads(globalTable)) {
        LLVM_DEBUG(moduleOp.dump());
        didChange = true;
      }

      LLVM_DEBUG(llvm::dbgs() << "==== eraseUnusedGlobals ====\n");
      if (eraseUnusedGlobals(globalTable)) {
        LLVM_DEBUG(moduleOp.dump());
        didChange = true;
      }

      LLVM_DEBUG(llvm::dbgs() << "==== deduplicateConstantGlobals ====\n");
      if (deduplicateConstantGlobals(globalTable)) {
        LLVM_DEBUG(moduleOp.dump());
        didChange = true;
      }

      if (!didChange) break;
    }

    afterFoldingGlobals = count(moduleOp.getOps<IREE::Util::GlobalOp>());
  }

 private:
  Statistic beforeFoldingGlobals{this, "global ops before folding",
                                 "Number of util.global ops before folding"};
  Statistic afterFoldingGlobals{this, "global ops after folding",
                                "Number of util.global ops after folding"};
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createFoldGlobalsPass() {
  return std::make_unique<FoldGlobalsPass>();
}

static PassRegistration<FoldGlobalsPass> pass;

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
