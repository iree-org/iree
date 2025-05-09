// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_REUSEALLOCATIONSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-reuse-allocations
//===----------------------------------------------------------------------===//

// Tries to replace the transient |allocaOp| with the operand of a prior
// deallocation on the same affinity and timeline. Only deallocations in the
// same block are checked.
//
// NOTE: assumes the alloca is uninitialized as the contents will be undefined.
//
// Returns true if the allocation was replaced.
static bool
tryReuseExistingAllocation(IREE::Stream::ResourceAllocaOp allocaOp) {
  // A walk is performed on the timeline defined by the timepoint SSA values.
  // Immediate allocations have no timeline and ones from
  auto awaitTimepoint = allocaOp.getAwaitTimepoint();
  if (!awaitTimepoint) {
    return false; // only timeline-based allocations are checked
  }
  auto awaitOp = dyn_cast_if_present<IREE::Stream::TimelineOpInterface>(
      awaitTimepoint.getDefiningOp());
  if (!awaitOp) {
    return false; // only timeline-based allocations are checked
  } else if (awaitOp->getBlock() != allocaOp->getBlock()) {
    return false; // only local analysis here
  }

  // TODO(benvanik): allow reuse across compatible affinities when in
  // unified memory mode (no NUMA). This information needs device analysis in a
  // way we don't have access to here yet. We could make it a flag to start
  // (ala the existing --iree-stream-resource-* flags).
  auto isCandidateDeallocaOp =
      [&](IREE::Stream::ResourceDeallocaOp deallocaOp) {
        return deallocaOp.getOperandSize() == allocaOp.getStorageSize() &&
               deallocaOp.getOperand().getType() ==
                   allocaOp.getResult().getType() &&
               deallocaOp.getAffinity() == allocaOp.getAffinity() &&
               deallocaOp->getBlock() == allocaOp->getBlock();
      };

  // Walk up timepoint joins and look for a dealloca that can be reused.
  // This is a local analysis and stops at the first op outside of the local
  // block.
  IREE::Stream::ResourceDeallocaOp chosenDeallocaOp;
  SmallVector<Operation *> worklist;
  worklist.push_back(awaitTimepoint.getDefiningOp());
  while (!worklist.empty()) {
    auto *workOp = worklist.pop_back_val();
    if (auto deallocaOp = dyn_cast<IREE::Stream::ResourceDeallocaOp>(workOp)) {
      if (isCandidateDeallocaOp(deallocaOp)) {
        chosenDeallocaOp = deallocaOp;
        break;
      }
    } else if (auto joinOp = dyn_cast<IREE::Stream::TimepointJoinOp>(workOp)) {
      for (auto joinTimepoint : joinOp.getAwaitTimepoints()) {
        if (auto *definingOp = joinTimepoint.getDefiningOp()) {
          if (definingOp->getBlock() == allocaOp->getBlock()) {
            worklist.push_back(definingOp);
          }
        }
      }
    }
  }
  if (!chosenDeallocaOp) {
    return false; // no candidate dealloca ops found on local timeline
  }

  // Replace the allocation with the previously deallocated resource and
  // erase the deallocation so it remains live. Users of the allocated
  // resource are updated to wait on whatever the deallocation was so the
  // resource is known to be available for reuse.
  Value availableTimepoint = chosenDeallocaOp.getAwaitTimepoint();
  if (!availableTimepoint) {
    OpBuilder builder(chosenDeallocaOp);
    availableTimepoint = builder.create<IREE::Stream::TimepointImmediateOp>(
        chosenDeallocaOp.getLoc());
  }
  allocaOp.replaceAllUsesWith(
      ValueRange{chosenDeallocaOp.getOperand(), availableTimepoint});
  allocaOp.erase();
  chosenDeallocaOp.replaceAllUsesWith(availableTimepoint);
  chosenDeallocaOp.erase();

  return true;
}

struct ReuseAllocationsPass
    : public IREE::Stream::impl::ReuseAllocationsPassBase<
          ReuseAllocationsPass> {
  void runOnOperation() override {
    auto parentOp = getOperation();
    if (!parentOp.getCallableRegion() ||
        parentOp.getCallableRegion()->empty()) {
      return;
    }

    // Traversal order defines whether earlier (pre-order) or later (post-order)
    // allocations are reused. It does not seem to make much of a difference
    // given the strict local behavior of the pass.
    bool didChange = false;
    for (auto allocaOp : llvm::make_early_inc_range(
             parentOp.getCallableRegion()
                 ->getOps<IREE::Stream::ResourceAllocaOp>())) {
      didChange = tryReuseExistingAllocation(allocaOp) || didChange;
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
