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
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/InliningUtils.h"

#define DEBUG_TYPE "iree-util-combine-initializers"

namespace mlir::iree_compiler::IREE::Util {
namespace {

class CombineInitializersPass
    : public CombineInitializersBase<CombineInitializersPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Gather all of the initializers in the module.
    // Build a fused loc from all initializers we are combining.
    SmallVector<IREE::Util::InitializerOp> initializerOps;
    SmallVector<Location> locs;
    for (auto initializerOp : moduleOp.getOps<IREE::Util::InitializerOp>()) {
      initializerOps.push_back(initializerOp);
      locs.push_back(initializerOp.getLoc());
    }
    if (initializerOps.size() <= 1)
      return;
    auto fusedLoc = FusedLoc::get(&getContext(), locs);

    // Make the new initializer op in the same location as the last initializer
    // we are combining - this ensures that module initialization order is
    // preserved.
    OpBuilder builder(initializerOps.back());
    auto newOp = builder.create<IREE::Util::InitializerOp>(fusedLoc);
    builder.setInsertionPointToStart(newOp.addEntryBlock());
    InlinerInterface inlinerInterface(&getContext());
    for (auto initializerOp : initializerOps) {
      if (failed(mlir::inlineRegion(
              inlinerInterface, &initializerOp.getBody(),
              builder.getInsertionBlock(), builder.getInsertionPoint(),
              /*inlinedOperands=*/ValueRange{},
              /*resultsToReplace=*/ValueRange{}, /*inlineLoc=*/std::nullopt,
              /*shouldCloneInlinedRegion=*/false))) {
        initializerOp.emitOpError()
            << "failed to inline into combined initializer";
        return signalPassFailure();
      }
      builder.setInsertionPointToEnd(&newOp.back());
      initializerOp.erase();
    }
    builder.create<IREE::Util::ReturnOp>(fusedLoc);
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createCombineInitializersPass() {
  return std::make_unique<CombineInitializersPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
