// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "global-loop-invariant-code-motion"
#define LICM_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

// Check if the op and its producers are loop invariants and hoistable. Using
// `hoistableOps` to check if producers are hoistable. It expects the op's
// producers have been checked and the results exist in `hoistableOps`.
static bool isHoistableOp(LoopLikeOpInterface loopOp, Operation *op,
                          const llvm::SetVector<Operation *> &hoistableOps) {
  // Currently we don't handle implicit captures, so don't hoist ops with
  // regions.
  if (op->getNumRegions() > 0) {
    return false;
  }

  // Check if the op type is hoistable.
  if (!isa<tensor::EmptyOp, tensor::PackOp>(op)) {
    return false;
  }

  // Check if all producers are hoistable.
  for (OpOperand &operand : op->getOpOperands()) {
    Value value = operand.get();
    // Ignore values defined outside the loop.
    if (loopOp.isDefinedOutsideOfLoop(value))
      continue;

    Operation *producer = value.getDefiningOp();
    // If the producer is not an operation, can't hoist it.
    if (!producer) {
      return false;
    }
    if (!hoistableOps.contains(producer)) {
      return false;
    }
  }

  return true;
}

static LogicalResult hoistLoopInvariants(LoopLikeOpInterface loopOp,
                                         RewriterBase &rewriter) {
  // Find hoistable ops in the loop.
  llvm::SetVector<Operation *> hoistableOps;
  for (Region *region : loopOp.getLoopRegions()) {
    // Skip loops with multi-block regions to simplify op's dependency.
    if (!region->hasOneBlock())
      return failure();

    // Consider only the top-level ops in the region. The forward visiting in a
    // single block ensures we are check and add ops in topological order.
    for (Operation &op : region->front()) {
      if (isHoistableOp(loopOp, &op, hoistableOps)) {
        LLVM_DEBUG(LICM_DBGS() << "Found hoistable op: " << op << "\n");
        hoistableOps.insert(&op);
      }
    }
  }
  if (hoistableOps.empty())
    return success();

  // Wrap the loop in zero-trip-check so the hoisted ops will only run when the
  // loop condition is ever satisfied.
  // Assume the transformation only moves ops around in loop instead of clone
  // them.
  FailureOr<LoopLikeOpInterface> wrappedLoop =
      TypeSwitch<Operation *, FailureOr<LoopLikeOpInterface>>(
          loopOp.getOperation())
          .Case<scf::WhileOp>([&](scf::WhileOp op) {
            return scf::wrapWhileLoopInZeroTripCheck(op, rewriter);
          })
          .Default([&](Operation *op) { return failure(); });
  if (failed(wrappedLoop))
    return failure();

  // Hoist ops out of the loop in topological order.
  for (Operation *op : hoistableOps) {
    assert(wrappedLoop->getOperation()->isAncestor(op) &&
           "assume hoistable ops still stay in the loop after transformation");
    wrappedLoop->moveOutOfLoop(op);
  }
  return success();
}

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

struct GlobalLoopInvariantCodeMotionPass
    : public GlobalLoopInvariantCodeMotionBase<
          GlobalLoopInvariantCodeMotionPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    SmallVector<LoopLikeOpInterface> candidateLoops;
    // Candidate loops are visited in post-order so a loop invariant has chances
    // to move across multiple loop levels.
    funcOp.walk([&](LoopLikeOpInterface op) {
      // Check if the loop type is supported.
      if (isa<scf::WhileOp>(op))
        candidateLoops.push_back(op);
      return;
    });

    IRRewriter rewriter(context);
    for (auto loopOp : candidateLoops) {
      if (failed(hoistLoopInvariants(loopOp, rewriter)))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGlobalLoopInvariantCodeMotionPass() {
  return std::make_unique<GlobalLoopInvariantCodeMotionPass>();
}
} // namespace mlir::iree_compiler::GlobalOptimization
