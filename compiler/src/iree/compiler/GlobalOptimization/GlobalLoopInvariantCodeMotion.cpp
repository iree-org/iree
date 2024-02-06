// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "llvm/Support/Debug.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

struct GlobalLoopInvariantCodeMotionPass
    : public GlobalLoopInvariantCodeMotionBase<
          GlobalLoopInvariantCodeMotionPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    SmallVector<LoopLikeOpInterface> worklist;
    funcOp.walk([&](LoopLikeOpInterface op) {
      worklist.push_back(op);
      return;
    });

    IRRewriter rewriter(context);
    for (auto loopOp : worklist) {
      llvm::SmallSetVector<Operation *, 8> toBeHoistedOps;
      moveLoopInvariantCode(
          loopOp.getLoopRegions(),
          [&](Value value, Region *region) {
            if (auto op = value.getDefiningOp()) {
              if (toBeHoistedOps.contains(op)) {
                return true;
              }
            }
            return loopOp.isDefinedOutsideOfLoop(value);
          },
          [&](Operation *op, Region *region) {
            return !toBeHoistedOps.contains(op) && isMemoryEffectFree(op) &&
                   isSpeculatable(op);
          },
          [&](Operation *op, Region *region) {
            toBeHoistedOps.insert(op);
            return;
          });
      if (toBeHoistedOps.empty()) {
        continue;
      }

      FailureOr<LoopLikeOpInterface> wrappedLoopOp =
          loopOp.replaceWithZeroTripCheck(rewriter);
      if (failed(wrappedLoopOp)) {
        continue;
      }

      for (auto op : toBeHoistedOps) {
        if (op->getParentOp() == wrappedLoopOp->getOperation()) {
          wrappedLoopOp->moveOutOfLoop(op);
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGlobalLoopInvariantCodeMotionPass() {
  return std::make_unique<GlobalLoopInvariantCodeMotionPass>();
}
} // namespace mlir::iree_compiler::GlobalOptimization
