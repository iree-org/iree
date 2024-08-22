// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-emplace-allocations"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_EMPLACEALLOCATIONSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Emplacement
//===----------------------------------------------------------------------===//

static void
replaceUsesAndTransfer(Value oldValue, Value newValue,
                       IREE::Stream::AffinityAttr usageAffinityAttr) {
  assert(isa<IREE::Stream::ResourceType>(oldValue.getType()));
  assert(isa<IREE::Stream::ResourceType>(newValue.getType()));
  if (oldValue.getType() == newValue.getType()) {
    oldValue.replaceAllUsesWith(newValue);
    return;
  }
  OpBuilder builder(newValue.getContext());
  builder.setInsertionPointAfterValue(newValue);
  Value newValueSize = IREE::Util::SizeAwareTypeInterface::queryValueSize(
      newValue.getLoc(), newValue, builder);
  IREE::Stream::AffinityAttr sourceAffinity = usageAffinityAttr;
  IREE::Stream::AffinityAttr resultAffinity = usageAffinityAttr;
  Value transferValue = builder.create<IREE::Stream::AsyncTransferOp>(
      newValue.getLoc(), oldValue.getType(), newValue, newValueSize,
      newValueSize, sourceAffinity, resultAffinity);
  oldValue.replaceAllUsesWith(transferValue);
}

// TODO(#14566): multiple results with sparse ties don't work due to
// implicit operand/result ordering on the dispatch ops. Flow and stream
// dispatch ops and the executable entry points need to be reworked to
// remove the implicit ordering. For now we only emplace results until the
// first we can't then bail and leave them out-of-place.
static bool tryEmplaceDispatchOp(IREE::Stream::AsyncDispatchOp dispatchOp) {
  bool didChange = false;
  // Collect the Update ops and their corresponding resultIndex.
  SmallVector<std::tuple<IREE::Stream::AsyncUpdateOp, int>> updateOpDataVec;
  for (auto [resultIndex, result] : llvm::enumerate(dispatchOp.getResults())) {
    // Ignore results with multiple users. We could potentially place these but
    // that makes tracking much more complicated.
    if (!result.hasOneUse()) {
      // TODO(#14566): continue if sparse emplacement on multiple results.
      break;
    }
    // Ignore already-tied operands.
    // TODO(benvanik): update tied range if we want to place into a superset?
    auto operandIndex = dispatchOp.getTiedResultOperandIndex(resultIndex);
    if (operandIndex.has_value()) {
      // TODO(#14566): continue if sparse emplacement on multiple results.
      break;
    }

    // Find potential update to place the dispatch result into.
    Operation *userOp = *result.user_begin();
    auto updateOp = dyn_cast_or_null<IREE::Stream::AsyncUpdateOp>(userOp);
    if (!updateOp)
      break;
    if (updateOp.getUpdate() != result) {
      // TODO(#14566): continue if sparse emplacement on multiple results.
      break;
    }

    // Currently only allow exactly matching affinities.
    // TODO(multi-device): memory compatibility - if compatible then allow.
    if (updateOp.getAffinityAttr() != dispatchOp.getAffinityAttr()) {
      continue;
    }
    updateOpDataVec.emplace_back(updateOp, resultIndex);
  }
  // Sort the update ops in block order so that we dont accidentally move them
  // above the dispatch op in the next section which will cause a dominance
  // issue.
  llvm::sort(updateOpDataVec, [&](const auto &a, const auto &b) {
    return std::get<0>(a)->isBeforeInBlock(std::get<0>(b));
  });
  for (auto updateOpData : updateOpDataVec) {
    int resultIndex;
    IREE::Stream::AsyncUpdateOp updateOp;
    std::tie(updateOp, resultIndex) = updateOpData;
    Value targetResource = updateOp.getTarget();
    if (targetResource.getDefiningOp() == dispatchOp) {
      // NOTE: we may have already replaced the update target with one of our
      // results - if so we need to find the operand to capture tied to that
      // new result instead of our own new result (which would make a cycle).
      targetResource = dispatchOp.getTiedResultOperand(targetResource);
    }
    if (!targetResource) {
      // TODO(#14566): continue if sparse emplacement on multiple results.
      break;
    }
    Value targetResourceSize = updateOp.getTargetSize();
    Value targetOffset = updateOp.getTargetOffset();
    Value targetEnd = updateOp.getTargetEnd();
    Value targetLength = updateOp.getUpdateSize();
    Value targetResult = updateOp.getResult();
    Value targetResultSize = updateOp.getTargetSize();

    // Try to move all SSA values required into the appropriate place.
    // TODO(benvanik): undo this if there's a failure (or record/roll-back).
    if (!IREE::Util::tryMoveProducerBefore(updateOp.getUpdateSize(),
                                           dispatchOp) ||
        !IREE::Util::tryMoveProducerBefore(updateOp.getTargetSize(),
                                           dispatchOp) ||
        !IREE::Util::tryMoveProducerBefore(updateOp.getTargetOffset(),
                                           dispatchOp) ||
        !IREE::Util::tryMoveProducerBefore(updateOp.getTargetEnd(),
                                           dispatchOp) ||
        !IREE::Util::tryMoveProducerBefore(updateOp.getTarget(), dispatchOp)) {
      // Failed to move while keeping valid SSA dominance.
      // TODO(#14566): continue if sparse emplacement on multiple results.
      break;
    }

    // Add operand and tie the result.
    auto operandIndex = dispatchOp.getResourceOperands().size();
    dispatchOp.getResourceOperandsMutable().append(targetResource);
    dispatchOp.getResourceOperandSizesMutable().append(targetResourceSize);
    dispatchOp.getResourceOperandOffsetsMutable().append(targetOffset);
    dispatchOp.getResourceOperandEndsMutable().append(targetEnd);
    dispatchOp.getResourceOperandLengthsMutable().append(targetLength);
    dispatchOp.setTiedResultOperandIndex(resultIndex, operandIndex);

    // Update result size (requires this dance as [] is a no-op!).
    SmallVector<Value> resultSizes = dispatchOp.getResultSizes();
    resultSizes[resultIndex] = targetResultSize;
    dispatchOp.getResultSizesMutable().assign(resultSizes);

    // Replace users with the result of the dispatch op.
    replaceUsesAndTransfer(targetResult, updateOp.getUpdate(),
                           dispatchOp.getAffinityAttr());
    updateOp->erase();

    didChange = true;
  }
  return didChange;
}

// Emplaces allocations within |region|.
// Returns true if any allocations were elided by way of emplacement.
static bool emplaceAllocationsInRegion(Region &region) {
  bool didChange = false;
  for (auto &block : region.getBlocks()) {
    for (auto &op : block) {
      if (!op.hasTrait<OpTrait::IREE::Stream::AsyncPhaseOp>())
        continue;
      // TODO(benvanik): support placement for more ops e.g. copies/collectives.
      didChange = TypeSwitch<Operation *, bool>(&op)
                      // TODO(#11249): support in-place collective ops.
                      .Case<IREE::Stream::AsyncDispatchOp>(
                          [&](auto op) { return tryEmplaceDispatchOp(op); })
                      .Default(false) ||
                  didChange;
    }
  }
  return didChange;
}

//===----------------------------------------------------------------------===//
// --iree-stream-emplace-allocations
//===----------------------------------------------------------------------===//

struct EmplaceAllocationsPass
    : public IREE::Stream::impl::EmplaceAllocationsPassBase<
          EmplaceAllocationsPass> {
  void runOnOperation() override {
    bool didChange = false;
    getOperation()->walk([&](Region *region) {
      didChange = emplaceAllocationsInRegion(*region) || didChange;
    });
    // TODO(benvanik): run canonicalization patterns inline if anything changed.
    (void)didChange;
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
