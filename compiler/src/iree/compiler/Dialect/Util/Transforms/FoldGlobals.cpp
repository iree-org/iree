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
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
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

#define GEN_PASS_DEF_FOLDGLOBALSPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

template <typename R>
static size_t count(R &&range) {
  return std::distance(range.begin(), range.end());
}

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
    if (global.isIndirect) {
      return GlobalAction::PRESERVE;
    }

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
    if (!constantValue || constantValue == global.op.getGlobalInitialValue()) {
      return GlobalAction::PRESERVE;
    }

    // Propagate constant into the initial value. Note that there may have been
    // a previous initial value that is being replaced.
    global.op.setGlobalInitialValue(constantValue);

    // Remove all of the stores.
    global.eraseStores();

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
    if (!global.isCandidate()) {
      return GlobalAction::PRESERVE;
    }

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
    if (!aliasName) {
      return GlobalAction::PRESERVE;
    }

    // Replace all loads from the global with the aliased global.
    auto &aliasGlobal = globalTable.lookup(aliasName.getValue());
    globalTable.renameGlobalUses(global, aliasGlobal);

    // Erase all stores to the global.
    global.eraseStores();

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
    if (!global.isCandidate()) {
      return GlobalAction::PRESERVE;
    } else if (!global.storeOps.empty() && !global.onlyInitialized) {
      return GlobalAction::PRESERVE;
    }
    bool didChangeAny = global.op.isGlobalMutable() != false;
    global.op.setGlobalMutable(false);
    for (auto loadOp : global.loadOps) {
      // NOTE: we don't set immutable on loads in initializers today.
      // We should be able to, though, with a bit better analysis.
      if (!loadOp->getParentOfType<IREE::Util::InitializerOpInterface>()) {
        if (!loadOp.isGlobalImmutable()) {
          loadOp.setGlobalImmutable(true);
          didChangeAny = true;
        }
      }
    }
    return didChangeAny ? GlobalAction::UPDATE : GlobalAction::PRESERVE;
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
  if (!op) {
    return nullptr;
  }
  return op->getResult(0);
}

// Inlines constant global values that are known to not change.
static bool inlineConstantGlobalLoads(GlobalTable &globalTable) {
  return globalTable.forEach([&](Global &global) {
    if (global.isIndirect) {
      return GlobalAction::PRESERVE;
    } else if (!global.storeOps.empty()) {
      return GlobalAction::PRESERVE;
    } else if (global.op.isGlobalMutable()) {
      return GlobalAction::PRESERVE;
    } else if (!global.op.getGlobalInitialValue()) {
      return GlobalAction::PRESERVE;
    }

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
    if (!global.loadOps.empty() || !global.referencingOps.empty()) {
      return GlobalAction::PRESERVE;
    }

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
    if (!global.canDCE()) {
      return GlobalAction::PRESERVE;
    }
    if (global.loadOps.empty() && global.referencingOps.empty()) {
      // No loads; remove entirely.
      return GlobalAction::DELETE;
    }
    return GlobalAction::PRESERVE;
  });
}

// Deduplicates immutable globals with constant initial values.
// This is a simplified and safer version of the global fusion pass.
static bool deduplicateConstantGlobals(GlobalTable &globalTable) {
  auto *context = globalTable.getContext();

  // Build sets of all equivalent globals.
  llvm::EquivalenceClasses<StringRef> ec;
  DenseMap<Attribute, StringRef> leaderMap;
  globalTable.forEach([&](Global &global) {
    if (!global.isCandidate()) {
      return GlobalAction::PRESERVE;
    } else if (!global.storeOps.empty()) {
      // Stores - not eligible for deduplication.
      return GlobalAction::PRESERVE;
    } else if (!global.op.getGlobalInitialValue()) {
      // No initial value, not constant.
      return GlobalAction::PRESERVE;
    }
    auto it = leaderMap.insert({
        ArrayAttr::get(
            context,
            {
                global.op.getGlobalInitialValue(),
                DictionaryAttr::get(
                    context, llvm::to_vector(global.op->getDialectAttrs())),
            }),
        global.getName(),
    });
    if (it.second) {
      // Inserted new.
      ec.insert(global.getName());
    } else {
      // Existing global with the same value.
      ec.unionSets(global.getName(), it.first->second);
    }
    return GlobalAction::PRESERVE;
  });

  SmallVector<StringRef> deadGlobalNames;
  for (auto it = ec.begin(), end = ec.end(); it != end; ++it) {
    if (!(*it)->isLeader()) {
      // Ignore non-leader sets.
      continue;
    } else if (++ec.member_begin(**it) == ec.member_end()) {
      continue;
    }
    auto *baseGlobal = &globalTable.lookup((*it)->getData());

    // Build fused location from all of the globals.
    SmallVector<Location> locs;
    for (auto mi = ec.member_begin(**it); mi != ec.member_end(); ++mi) {
      Global &global = globalTable.lookup(*mi);
      locs.push_back(global.op.getLoc());
      if (global.ordinal < baseGlobal->ordinal) {
        baseGlobal = &global;
      }
    }
    auto fusedLoc = FusedLoc::get(context, locs);

    // Update base global location.
    IREE::Util::GlobalOpInterface baseGlobalOp = baseGlobal->op;
    baseGlobalOp->setLoc(fusedLoc);

    // Replace all other globals to point at the new one.
    for (auto mi = ec.member_begin(**it); mi != ec.member_end(); ++mi) {
      Global &global = globalTable.lookup(*mi);
      if (global.op == baseGlobalOp) {
        continue;
      }
      globalTable.renameGlobalUses(global, *baseGlobal);
      deadGlobalNames.push_back(global.getName());
    }
  }
  if (deadGlobalNames.empty()) {
    return false; // no change
  }

  for (auto globalName : deadGlobalNames) {
    globalTable.eraseGlobal(globalName);
  }
  return true; // did change
}

struct FoldGlobalsPass : public impl::FoldGlobalsPassBase<FoldGlobalsPass> {
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
    GlobalTable globalTable(moduleOp);
    beforeFoldingGlobals = globalTable.size();
    for (int i = 0; i < 10; ++i) {
      // TODO(benvanik): determine if we need this expensive folding.
      if (failed(applyPatternsGreedily(moduleOp, frozenPatterns))) {
        signalPassFailure();
        return;
      }

      bool didChange = false;

      // Rebuild the global table after potential pattern changes.
      globalTable.rebuild();

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

      if (!didChange) {
        // No changes; complete fixed-point iteration.
        break;
      }
    }

    afterFoldingGlobals =
        count(moduleOp.getOps<IREE::Util::GlobalOpInterface>());
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
