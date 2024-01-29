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
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-util-fold-globals"

namespace mlir::iree_compiler::IREE::Util {
namespace {

template <typename R>
static size_t count(R &&range) {
  return std::distance(range.begin(), range.end());
}

struct Global {
  size_t ordinal = 0;
  IREE::Util::GlobalOpInterface op;
  bool isIndirect = false;
  SmallVector<IREE::Util::GlobalLoadOpInterface> loadOps;
  SmallVector<IREE::Util::GlobalStoreOpInterface> storeOps;

  bool isCandidate() { return !isIndirect && op.isGlobalPrivate(); }
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
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
      auto globalName = globalOp.getGlobalName();
      globalMap[globalName] = Global{globalOrder.size(), globalOp};
      globalOrder.push_back(globalName);
    }
    for (auto callableOp : moduleOp.getOps<CallableOpInterface>()) {
      callableOp.walk([&](Operation *op) {
        if (auto addressOp =
                dyn_cast<IREE::Util::GlobalAddressOpInterface>(op)) {
          globalMap[addressOp.getGlobalName()].isIndirect = true;
        } else if (auto loadOp =
                       dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
          globalMap[loadOp.getGlobalName()].loadOps.push_back(loadOp);
        } else if (auto storeOp =
                       dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
          globalMap[storeOp.getGlobalName()].storeOps.push_back(storeOp);
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
        assert(global.op.isGlobalPrivate() && "can't delete public globals");
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
    if (global.isIndirect)
      return GlobalAction::PRESERVE;

    // Find the constant value used in all stores.
    // All stores must match the initial value of the global _or_ the global
    // must be uninitialized. A proper dataflow analysis would let us identify
    // the cases where there are loads before stores and the implicit 0 initial
    // value for uninitialized globals would be observable.
    Attribute constantValue = global.op.getGlobalInitialValue();
    for (auto storeOp : global.storeOps) {
      Attribute valueAttr;
      if (!matchPattern(storeOp.getStoredGlobalValue(),
                        m_Constant(&valueAttr))) {
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
    if (!constantValue)
      return GlobalAction::PRESERVE;

    // Propagate constant into the initial value. Note that there may have been
    // a previous initial value that is being replaced.
    global.op.setGlobalInitialValue(constantValue);

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
    if (!global.isCandidate())
      return GlobalAction::PRESERVE;

    // Find the other symbol this global is chained with by looking for uniform
    // stores. Note that we don't care about initializers.
    FlatSymbolRefAttr aliasName;
    for (auto storeOp : global.storeOps) {
      // Check to see if the stored value comes from another global.
      auto *definingOp = storeOp.getStoredGlobalValue().getDefiningOp();
      if (auto loadOp =
              dyn_cast_or_null<IREE::Util::GlobalLoadOpInterface>(definingOp)) {
        if (!aliasName) {
          aliasName = loadOp.getGlobalAttr();
        } else if (aliasName != loadOp.getGlobalAttr()) {
          aliasName = {};
          break;
        }
      } else {
        aliasName = {};
        break;
      }
    }
    if (!aliasName)
      return GlobalAction::PRESERVE;

    // Replace all loads from the global with the aliased global.
    auto &aliasGlobal = globalTable.globalMap[aliasName.getValue()];
    for (auto loadOp : global.loadOps) {
      loadOp.setGlobalAttr(aliasName);
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
    if (!global.isCandidate())
      return GlobalAction::PRESERVE;
    if (!global.storeOps.empty())
      return GlobalAction::PRESERVE;
    if (!global.op.isGlobalMutable())
      return GlobalAction::PRESERVE;
    global.op.setGlobalMutable(false);
    return GlobalAction::UPDATE;
  });
}

// Tries to materialize a constant op for |attr| of |type|.
// Returns nullptr if the attribute could not be materialized as a constant.
static Value tryMaterializeConstant(Location loc, Type type, Attribute attr,
                                    OpBuilder &builder) {
  if (arith::ConstantOp::isBuildableWith(attr, type)) {
    // Common case fast-path.
    return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(attr));
  } else if (mlir::func::ConstantOp::isBuildableWith(attr, type)) {
    return builder.create<mlir::func::ConstantOp>(
        loc, type, llvm::cast<FlatSymbolRefAttr>(attr));
  }
  // Fallback that asks a dialect to materialize things. This may fail!
  auto *op = attr.getDialect().materializeConstant(builder, attr, type, loc);
  if (!op)
    return nullptr;
  return op->getResult(0);
}

// Inlines constant global values that are known to not change.
static bool inlineConstantGlobalLoads(GlobalTable &globalTable) {
  return globalTable.forEach([&](Global &global) {
    if (global.isIndirect)
      return GlobalAction::PRESERVE;
    if (!global.storeOps.empty())
      return GlobalAction::PRESERVE;
    if (global.op.isGlobalMutable())
      return GlobalAction::PRESERVE;
    if (!global.op.getGlobalInitialValue())
      return GlobalAction::PRESERVE;

    if (llvm::isa<IREE::Util::ReferenceTypeInterface>(
            global.op.getGlobalType())) {
      // We only inline value types; reference types have meaning as globals.
      return GlobalAction::PRESERVE;
    }

    // Inline initial value into all loads.
    auto inliningPolicy = global.op.getGlobalInliningPolicy();
    SmallVector<IREE::Util::GlobalLoadOpInterface> loadOps = global.loadOps;
    global.loadOps.clear();
    for (auto loadOp : loadOps) {
      if (inliningPolicy &&
          !inliningPolicy.isLegalToInline(loadOp, global.op)) {
        // Global not allowed to be inlined at this site so preserve the load.
        global.loadOps.push_back(loadOp);
        continue;
      }

      OpBuilder builder(loadOp);
      auto loadedValue = loadOp.getLoadedGlobalValue();
      auto constantValue =
          tryMaterializeConstant(loadOp.getLoc(), loadedValue.getType(),
                                 global.op.getGlobalInitialValue(), builder);
      if (!constantValue) {
        // Failed to materialize the constant at this site so preserve the load.
        global.loadOps.push_back(loadOp);
        continue;
      }
      loadedValue.replaceAllUsesWith(constantValue);
      loadOp.erase();
    }

    // If not all loads could be removed we need to preserve the global.
    if (!global.loadOps.empty())
      return GlobalAction::PRESERVE;

    // Only delete if private.
    return global.op.isGlobalPrivate() ? GlobalAction::DELETE
                                       : GlobalAction::UPDATE;
  });
}

// Erases all globals in the module that are never loaded.
// This differs from SymbolDCE in that initializations and storage operations
// are discarded.
static bool eraseUnusedGlobals(GlobalTable &globalTable) {
  return globalTable.forEach([&](Global &global) {
    if (!global.isCandidate())
      return GlobalAction::PRESERVE;
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
    if (!global.isCandidate())
      continue;
    if (!global.storeOps.empty()) {
      // Stores - not eligible for deduplication.
      continue;
    }
    if (!global.op.getGlobalInitialValue()) {
      // No initial value, not constant.
      continue;
    }
    auto it = leaderMap.insert(
        {global.op.getGlobalInitialValue(), global.op.getGlobalName()});
    if (it.second) {
      // Inserted new.
      ec.insert(global.op.getGlobalName());
    } else {
      // Existing global with the same value.
      ec.unionSets(global.op.getGlobalName(), it.first->second);
    }
  }

  bool didChange = false;
  for (auto it = ec.begin(), end = ec.end(); it != end; ++it) {
    if (!it->isLeader())
      continue; // Ignore non-leader sets.
    if (++ec.member_begin(it) == ec.member_end())
      continue;
    IREE::Util::GlobalOpInterface baseGlobalOp =
        globalTable.globalMap[it->getData()].op;

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
        baseGlobalOp.getContext(), baseGlobalOp.getGlobalName());
    for (auto mi = ec.member_begin(it); mi != ec.member_end(); ++mi) {
      Global &global = globalTable.globalMap[*mi];
      if (global.op == baseGlobalOp)
        continue;
      for (auto loadOp : global.loadOps) {
        loadOp.setGlobalAttr(baseGlobalNameAttr);
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

class FoldGlobalsPass : public FoldGlobalsBase<FoldGlobalsPass> {
public:
  explicit FoldGlobalsPass() = default;
  FoldGlobalsPass(const FoldGlobalsPass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);

    for (auto *dialect : context->getLoadedDialects()) {
      dialect->getCanonicalizationPatterns(patterns);
    }
    for (auto op : context->getRegisteredOperations()) {
      op.getCanonicalizationPatterns(patterns, context);
    }

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    auto moduleOp = getOperation();
    beforeFoldingGlobals =
        count(moduleOp.getOps<IREE::Util::GlobalOpInterface>());
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

      if (!didChange)
        break;
    }

    afterFoldingGlobals =
        count(moduleOp.getOps<IREE::Util::GlobalOpInterface>());
  }

private:
  Statistic beforeFoldingGlobals{this, "global ops before folding",
                                 "Number of util.global ops before folding"};
  Statistic afterFoldingGlobals{this, "global ops after folding",
                                "Number of util.global ops after folding"};
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createFoldGlobalsPass() {
  return std::make_unique<FoldGlobalsPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
