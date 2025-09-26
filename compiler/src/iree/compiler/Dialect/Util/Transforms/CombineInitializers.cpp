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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"

#define DEBUG_TYPE "iree-util-combine-initializers"

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

class CombineInitializersPass
    : public impl::CombineInitializersPassBase<CombineInitializersPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

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

    // Process initialization points, potentially creating multiple combined
    // initializers if we encounter non-materializable constants that we can't
    // place inside our combined initializer. Initialization order of the
    // program will be preserved.
    InlinerInterface inlinerInterface(&getContext());
    SmallVector<IREE::Util::InitializerOp> createdInitializers;
    SmallVector<Operation *> currentBatch;
    SmallVector<Location> currentLocs;
    auto finalizeBatch = [&]() {
      if (currentBatch.empty()) {
        return;
      }

      // Check if we have any actual initializers in this batch or just globals.
      bool hasBatchInitializers = false;
      for (auto *op : currentBatch) {
        if (isa<IREE::Util::InitializerOpInterface>(op)) {
          hasBatchInitializers = true;
          break;
        }
      }

      // Only create a combined initializer if we have initializers to combine.
      // If we only have globals with initial values leave them as-is.
      if (!hasBatchInitializers) {
        currentBatch.clear();
        currentLocs.clear();
        return;
      }

      // Create the new combined initializer after the last op in the batch.
      OpBuilder builder(currentBatch.back());
      builder.setInsertionPointAfter(currentBatch.back());
      auto fusedLoc = currentLocs.size() == 1
                          ? currentLocs[0]
                          : FusedLoc::get(&getContext(), currentLocs);
      auto newOp = IREE::Util::InitializerOp::create(builder, fusedLoc);
      createdInitializers.push_back(newOp);
      builder.setInsertionPointToStart(newOp.addEntryBlock());

      // Process each initialization point in the current batch.
      for (auto *initPoint : currentBatch) {
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
          // Materialize the initial value and store it to the global.
          auto initialValue = globalOp.getGlobalInitialValue();
          auto typedAttr = cast<TypedAttr>(initialValue);
          auto *constantOp = IREE::Util::materializeConstant(
              builder, initPoint->getLoc(), typedAttr);

          // We should have already checked this won't fail in the main loop.
          assert(constantOp && "unexpected materialization failure");

          // Store the materialized value to the global.
          globalOp.createStoreOp(initPoint->getLoc(), constantOp->getResult(0),
                                 builder);

          // Clear the initial value from the global.
          globalOp.setGlobalInitialValue(nullptr);
        }
      }

      // Add the return terminator at the end of the initializer.
      IREE::Util::ReturnOp::create(builder, fusedLoc);

      currentBatch.clear();
      currentLocs.clear();
    };

    // Process each initialization point by creating batches split at
    // non-materializable constants.
    for (auto *initPoint : initPoints) {
      // Check if this is a global with a non-materializable initial value.
      bool canMaterialize = true;
      if (auto globalOp = dyn_cast<IREE::Util::GlobalOpInterface>(initPoint)) {
        auto initialValue = globalOp.getGlobalInitialValue();
        if (initialValue) {
          canMaterialize =
              canMaterializeConstant(cast<TypedAttr>(initialValue));
        }
      }

      if (!canMaterialize) {
        // Hit a non-materializable constant. Finalize the current batch
        // and skip this global (leave it with its initial value).
        finalizeBatch();
        LLVM_DEBUG(
            llvm::dbgs()
            << "skipping non-materializable global: "
            << cast<IREE::Util::GlobalOpInterface>(initPoint).getGlobalName()
            << "\n");
        continue;
      }

      // Add to current batch.
      currentBatch.push_back(initPoint);
      currentLocs.push_back(initPoint->getLoc());
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
