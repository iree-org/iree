// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_COMBINEINITIALIZERSPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

// Returns true if |attr| can be materialized as a constant op.
static bool canMaterializeConstant(TypedAttr attr) {
  OpBuilder testBuilder(attr.getContext());
  auto *testOp = IREE::Util::materializeConstant(
      testBuilder, testBuilder.getUnknownLoc(), attr);
  const bool canMaterialize = testOp != nullptr;
  if (testOp) {
    testOp->erase();
  }
  return canMaterialize;
}

// Hoists all globals and initializers in |ops| to the first position in the
// module, maintaining their relative order. Sets the builder insertion point
// after the last hoisted operation.
static void hoistGlobalsAndInitializersToFirst(ArrayRef<Operation *> ops,
                                               OpBuilder &builder) {
  if (ops.empty()) {
    return;
  }

#ifndef NDEBUG
  // Sanity check all ops are in the correct order.
  Operation *firstOp = ops.front();
  for (auto *op : ops) {
    if (op->isBeforeInBlock(firstOp)) {
      firstOp = op;
    }
  }
  assert(firstOp == ops.front() && "expected ops to be in the correct order");
#endif // !NDEBUG

  // Move all ops to first position, maintaining relative order.
  Operation *insertionPointOp = ops.front();
  for (auto *op : ops) {
    if (op != insertionPointOp) {
      op->moveAfter(insertionPointOp);
      insertionPointOp = op;
    }
  }

  // Set builder insertion point after the last op.
  builder.setInsertionPointAfter(insertionPointOp);
}

// Collects all globals transitively accessed by an initializer, including
// through function calls. Uses a worklist algorithm to avoid stack overflow.
static SmallVector<Operation *>
collectTransitiveGlobalDeps(IREE::Util::InitializerOpInterface initOp,
                            SymbolTable &symbolTable) {
  // Start with the initializer's region and work depth-first.
  //
  // Note that we maintain discovery order as that'll be our final total global
  // order.
  llvm::SetVector<Operation *> globals;
  llvm::DenseSet<Operation *> visitedFuncs;
  SmallVector<Region *> worklist;
  worklist.push_back(&initOp.getInitializerRegion());
  while (!worklist.empty()) {
    Region *region = worklist.pop_back_val();
    region->walk([&](Operation *op) {
      // Collect direct global accesses.
      if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
        if (auto globalOp = symbolTable.lookup<IREE::Util::GlobalOpInterface>(
                loadOp.getGlobalName())) {
          globals.insert(globalOp.getOperation());
        }
      } else if (auto storeOp =
                     dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
        if (auto globalOp = symbolTable.lookup<IREE::Util::GlobalOpInterface>(
                storeOp.getGlobalName())) {
          globals.insert(globalOp.getOperation());
        }
      } else if (auto addrOp =
                     dyn_cast<IREE::Util::GlobalAddressOpInterface>(op)) {
        if (auto globalOp = symbolTable.lookup<IREE::Util::GlobalOpInterface>(
                addrOp.getGlobalName())) {
          globals.insert(globalOp.getOperation());
        }
      }

      // Handle transitive dependencies through function calls.
      if (auto callOp = dyn_cast<CallOpInterface>(op)) {
        auto callee = callOp.getCallableForCallee();
        if (auto symRef = dyn_cast<SymbolRefAttr>(callee)) {
          if (auto calleeOp = symbolTable.lookup<FunctionOpInterface>(
                  symRef.getRootReference())) {
            // Avoid infinite recursion by tracking visited functions.
            if (visitedFuncs.insert(calleeOp.getOperation()).second) {
              worklist.push_back(&calleeOp.getFunctionBody());
            }
          }
        }
      }
    });
  }

  return globals.takeVector();
}

struct CombineInitializersPass
    : public impl::CombineInitializersPassBase<CombineInitializersPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    // Gather all initialization points in module order.
    // This includes both util.initializer ops and util.global ops with initial
    // values.
    SmallVector<Operation *> initPoints;
    SmallVector<Location> locs;
    bool hasInitializers = false;
    for (auto &op : moduleOp.getOps()) {
      if (isa<IREE::Util::InitializerOpInterface>(&op)) {
        initPoints.push_back(&op);
        locs.push_back(op.getLoc());
        hasInitializers = true;
      } else if (auto globalOp = dyn_cast<IREE::Util::GlobalOpInterface>(&op)) {
        auto initialValue = globalOp.getGlobalInitialValue();
        if (initialValue && !isa<IREE::Util::UninitializedAttr>(initialValue)) {
          initPoints.push_back(&op);
          locs.push_back(op.getLoc());
        }
      }
    }

    // We only proceed if we have initializers - if we only have globals with
    // initial values the module is already in a stable state.
    if (!hasInitializers || initPoints.size() <= 1) {
      return;
    }

    // Simple symbol table for lookups.
    // We are otherwise performing local operations and do not need a full
    // GlobalTable analysis.
    SymbolTable symbolTable(moduleOp);

    // Process initialization points, potentially creating multiple combined
    // initializers if we encounter non-materializable constants that we can't
    // place inside our combined initializer. Initialization order of the
    // program will be preserved.
    InlinerInterface inlinerInterface(&getContext());
    SmallVector<IREE::Util::InitializerOp> createdInitializers;
    llvm::SetVector<Operation *> currentBatch;
    SmallVector<Location> currentLocs;
    auto finalizeBatch = [&]() {
      if (currentBatch.empty()) {
        return;
      }

      // Check if we have any actual initializers in this batch or just globals.
      SmallVector<Operation *> batchGlobalOps;
      SmallVector<Operation *> batchInitializerOps;
      for (auto *op : currentBatch) {
        if (isa<IREE::Util::GlobalOpInterface>(op)) {
          batchGlobalOps.push_back(op);
        } else if (isa<IREE::Util::InitializerOpInterface>(op)) {
          batchInitializerOps.push_back(op);
        }
      }

      // Only create a combined initializer if we have initializers to combine.
      // If we only have globals with initial values leave them as-is.
      if (batchInitializerOps.empty()) {
        currentBatch.clear();
        currentLocs.clear();
        return;
      }

      // Pull all globals in the batch to the first position to ensure they
      // are defined before the combined initializer that will initialize them.
      // This implements the "pull-up-to-first" principle: moving
      // globals/initializers upward is always safe (dependencies flow forward).
      //
      // Sort the batch by module order first to maintain relative ordering.
      SmallVector<Operation *> sortedBatch = currentBatch.takeVector();
      llvm::sort(sortedBatch, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });

      OpBuilder builder(&getContext());
      hoistGlobalsAndInitializersToFirst(sortedBatch, builder);

      // Create the new combined initializer after the globals.
      auto fusedLoc = currentLocs.size() == 1
                          ? currentLocs[0]
                          : FusedLoc::get(&getContext(), currentLocs);
      auto newOp = IREE::Util::InitializerOp::create(builder, fusedLoc);
      createdInitializers.push_back(newOp);
      builder.setInsertionPointToStart(newOp.addEntryBlock());

      // Process each initialization point in the current batch.
      for (auto *initPoint : sortedBatch) {
        if (auto initializerOp =
                dyn_cast<IREE::Util::InitializerOpInterface>(initPoint)) {
          // Inline existing initializer.
          if (failed(mlir::inlineRegion(
                  inlinerInterface, InlinerConfig{}.getCloneCallback(),
                  &initializerOp.getInitializerRegion(),
                  builder.getInsertionBlock(), builder.getInsertionPoint(),
                  /*inlinedOperands=*/ValueRange{},
                  /*resultsToReplace=*/ValueRange{}, /*inlineLoc=*/std::nullopt,
                  /*shouldCloneInlinedRegion=*/false))) {
            initializerOp.emitOpError()
                << "failed to inline into combined initializer";
            return signalPassFailure();
          }
          // After inlining, set insertion point to the end of the combined
          // initializer's block to continue appending operations.
          builder.setInsertionPointToEnd(&newOp.back());
        } else if (auto globalOp =
                       dyn_cast<IREE::Util::GlobalOpInterface>(initPoint)) {
          // Only materialize if this global has an initial value that can be
          // materialized. Globals added via transitive dependency collection
          // or non-materializable constants are left as-is.
          auto initialValue = globalOp.getGlobalInitialValue();
          if (initialValue &&
              !isa<IREE::Util::UninitializedAttr>(initialValue)) {
            auto typedAttr = cast<TypedAttr>(initialValue);

            // Skip non-materializable constants (e.g., #util.byte_pattern).
            if (!canMaterializeConstant(typedAttr)) {
              continue;
            }

            auto *constantOp = IREE::Util::materializeConstant(
                builder, initPoint->getLoc(), typedAttr);
            assert(constantOp && "unexpected materialization failure");

            // Store the materialized value to the global.
            globalOp.createStoreOp(initPoint->getLoc(),
                                   constantOp->getResult(0), builder);

            // Clear the initial value from the global.
            globalOp.setGlobalInitialValue(nullptr);
          }
        }
      }

      // Add the return terminator at the end of the initializer.
      IREE::Util::ReturnOp::create(builder, fusedLoc);

      currentBatch.clear();
      currentLocs.clear();
    };

    // Process each initialization point, collecting all into a single batch.
    // Non-materializable constants are included to preserve relative ordering
    // with related globals (e.g., resource+size pairs).
    for (auto *initPoint : initPoints) {
      // Add to current batch and collect any transitive dependencies for
      // initializers.
      if (auto initializerOp =
              dyn_cast<IREE::Util::InitializerOpInterface>(initPoint)) {
        // Collect all globals this initializer depends on (direct +
        // transitive). Warning that this could spider the whole program and we
        // currently don't cache partial results for functions called repeatedly
        // (as we tend not to have those... yet).
        auto dependentGlobals =
            collectTransitiveGlobalDeps(initializerOp, symbolTable);

        // Insert dependent globals BEFORE the initializer in the batch.
        // SetVector maintains insertion order and handles duplicates.
        currentBatch.insert(dependentGlobals.begin(), dependentGlobals.end());

        // Now add the initializer itself.
        currentBatch.insert(initPoint);
        currentLocs.push_back(initPoint->getLoc());
      } else {
        // Global with initial value - just add it (no materialization needed).
        currentBatch.insert(initPoint);
        currentLocs.push_back(initPoint->getLoc());
      }
    }

    // Finalize any remaining batch.
    finalizeBatch();

    // Erase original initializers (but not the globals).
    for (auto *initPoint : initPoints) {
      if (isa<IREE::Util::InitializerOpInterface>(initPoint)) {
        initPoint->erase();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
