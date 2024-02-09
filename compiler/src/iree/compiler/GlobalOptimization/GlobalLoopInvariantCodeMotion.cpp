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
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "global-loop-invariant-code-motion"
#define KD_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir::iree_compiler::GlobalOptimization {

// Check if the op is a leaf op we always want to hoist.
static bool isHoistableLeafOp(Operation *op) {
  // Currently it's limited to a small set of ops related to constant pack op.
  return isa<tensor::PackOp>(op);
}

// Check if the op is hoistable (but we might not want to hoist it alone).
static bool isHoistableOp(Operation *op) {
  // Currently it's limited to a small set of ops related to constant pack op.
  return op->hasTrait<OpTrait::ConstantLike>() || isa<tensor::EmptyOp>(op) ||
         isHoistableLeafOp(op);
}

// Check if the op and its producers are loop invariants and hoistable. Results
// are cached in hoistableOpMap to avoid repeated traversals.
static bool
checkHoistableTree(LoopLikeOpInterface loopOp, Operation *op,
                   llvm::SmallDenseMap<Operation *, bool> &hoistableOpMap) {
  // First check if the op has been analyzed.
  if (hoistableOpMap.contains(op))
    return hoistableOpMap[op];

  // Check if the nested ops and the op itself are hoistable. And check if the
  // producers of all operands are hoistable.
  auto walkFn = [&](Operation *walkOp) {
    // Check if the op is hoistable.
    if (!isHoistableOp(walkOp))
      return WalkResult::interrupt();

    WalkResult result = WalkResult::advance();
    // Check if the producers of operands are hoistable.
    for (OpOperand &operand : walkOp->getOpOperands()) {
      Value value = operand.get();
      // Ignore values defined in a nested region or outside the loop.
      if (walkOp->isAncestor(value.getParentRegion()->getParentOp()) ||
          loopOp.isDefinedOutsideOfLoop(value)) {
        continue;
      }

      Operation *producer = value.getDefiningOp();
      // If the value is not an operation, we don't hoist it.
      if (!producer || !checkHoistableTree(loopOp, producer, hoistableOpMap)) {
        result = WalkResult::interrupt();
        break;
      }
    }
    return result;
  };

  bool hoistable = !op->walk(walkFn).wasInterrupted();
  hoistableOpMap[op] = hoistable;

  LLVM_DEBUG(KD_DBGS() << (hoistable ? "Hoistable: " : "Non-hoistable: ") << *op
                       << "\n");
  return hoistable;
}

// Call `moveOutOfLoop` to hoist op and its producer out of the loop. Ops are
// hoisted in post-order to handle the dependencies (leaves of the producer tree
// are hoisted first).
static LogicalResult hoistProducerTree(Operation *op,
                                       LoopLikeOpInterface loopOp) {
  // Check if the op has been hoisted.
  if (!loopOp->isAncestor(op))
    return success();

  // Walk all producers and hoist them first.
  auto result = op->walk([&](Operation *walkOp) {
    for (OpOperand &operand : walkOp->getOpOperands()) {
      if (Operation *producer = operand.get().getDefiningOp()) {
        if (failed(hoistProducerTree(producer, loopOp)))
          return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  // Hoist the op itself.
  loopOp.moveOutOfLoop(op);
  return success();
}

static LogicalResult hoistLoopInvariants(LoopLikeOpInterface loopOp,
                                         RewriterBase &rewriter) {
  SmallVector<Operation *> opsToHoist;

  llvm::SmallDenseMap<Operation *, bool> hoistableOpMap;
  for (Region *region : loopOp.getLoopRegions()) {
    // Check top-level ops.
    for (Operation &op : region->getOps()) {
      if (!isHoistableLeafOp(&op))
        continue;
      if (checkHoistableTree(loopOp, &op, hoistableOpMap)) {
        LLVM_DEBUG(KD_DBGS() << "Found hoistable leaf: " << op << "\n");
        opsToHoist.push_back(&op);
      }
    }
  }
  if (opsToHoist.empty())
    return success();

  // Wrap the loop in zero-trip-check so the hoisted ops will only run when the
  // loop condition is ever satisfied.
  auto wrappedLoop =
      TypeSwitch<Operation *, FailureOr<LoopLikeOpInterface>>(
          loopOp.getOperation())
          .Case<scf::WhileOp>([&](scf::WhileOp op) {
            return scf::wrapWhileLoopInZeroTripCheck(op, rewriter);
          })
          .Default([&](Operation *op) { return failure(); });
  if (failed(wrappedLoop))
    return failure();

  // Hoist ops in order to handle the dependencies.
  for (Operation *op : opsToHoist) {
    if (failed(hoistProducerTree(op, wrappedLoop.value())))
      return failure();
  }

  return success();
}

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
