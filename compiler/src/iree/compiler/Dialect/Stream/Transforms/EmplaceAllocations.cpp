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
#include "mlir/IR/AsmState.h"
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

// Returns true if |tiedOp| has any tied operands.
static bool hasTiedOperands(IREE::Util::TiedOpInterface tiedOp) {
  SmallVector<int64_t> tiedResultOperands;
  tiedOp.getAllTiedOperands(tiedResultOperands);
  return llvm::any_of(tiedResultOperands, [](int64_t index) {
    return index != IREE::Util::TiedOpInterface::kUntiedIndex;
  });
}

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

static bool tryEmplaceDispatchOp(IREE::Stream::AsyncDispatchOp dispatchOp,
                                 IndexSet &indexSet) {
  [[maybe_unused]] OpPrintingFlags printingFlags;
  LLVM_DEBUG(printingFlags = OpPrintingFlags()
                                 .elideLargeElementsAttrs()
                                 .assumeVerified()
                                 .skipRegions());

  // If the op has tied operands we have to bail; the dispatch binding mapping
  // is currently implicit and we can't tell where we should be inserting new
  // operands as we tie them. The dispatch op would need to store which bindings
  // in the target executable relate to which operand/result explicitly (and it
  // really should!) for us to handle that. We'd then change from appending new
  // operands to inserting at the appropriate location below.
  if (hasTiedOperands(dispatchOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  ! skipping dispatch, op already has tied operands\n");
    return false;
  }

  // Collect the update ops and their corresponding resultIndex.
  SmallVector<std::tuple<IREE::Stream::AsyncUpdateOp, int>> updateResultOps;
  for (auto [resultIndex, result] : llvm::enumerate(dispatchOp.getResults())) {
    // Ignore results with multiple users. We could potentially place these but
    // that makes tracking much more complicated.
    if (!result.hasOneUse()) {
      LLVM_DEBUG({
        llvm::dbgs() << "  ! skipping result #" << resultIndex
                     << " of dispatch ";
        result.printAsOperand(llvm::dbgs(), printingFlags);
        llvm::dbgs() << ": has multiple uses\n";
      });
      continue;
    }
    Operation *userOp = *result.user_begin();

    // Check if the user is an update op we can merge.
    auto updateOp = dyn_cast_or_null<IREE::Stream::AsyncUpdateOp>(userOp);
    if (!updateOp || updateOp.getUpdate() != result) {
      continue; // not relevant
    }

    // Currently only allow exactly matching affinities.
    // TODO(multi-device): memory compatibility - if compatible then allow.
    if (updateOp.getAffinityAttr() != dispatchOp.getAffinityAttr()) {
      LLVM_DEBUG({
        llvm::dbgs() << "  ! unable to emplace result #" << resultIndex
                     << " of dispatch ";
        result.printAsOperand(llvm::dbgs(), printingFlags);
        llvm::dbgs() << " into ";
        updateOp.print(llvm::dbgs(), printingFlags);
        llvm::dbgs() << ": affinities do not match\n";
        llvm::dbgs() << "    dispatch affinity: "
                     << dispatchOp.getAffinityAttr() << "\n";
        llvm::dbgs() << "    update affinity: " << updateOp.getAffinityAttr()
                     << "\n";
      });
      continue;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  + emplacing result #" << resultIndex
                   << " of dispatch ";
      result.printAsOperand(llvm::dbgs(), printingFlags);
      llvm::dbgs() << " into ";
      updateOp.print(llvm::dbgs(), printingFlags);
      llvm::dbgs() << "\n";
    });

    updateResultOps.emplace_back(updateOp, resultIndex);
  }
  if (updateResultOps.empty()) {
    // No potential updates found, early exit.
    return false;
  }

  // Convert the op to be fully emplaced (all results are tied allocas).
  // We may immediately replace some of these with transfer targets and that's
  // a waste of IR but we won't know exactly which ones we'll replace until we
  // are processing the results (as some values may be used by multiple operands
  // _and_ tied to multiple results, and that may only happen after we start
  // emplacing things).
  OpBuilder builder(dispatchOp);
  Value zeroOffset = indexSet.get(0);
  for (auto [resultIndex, result] : llvm::enumerate(dispatchOp.getResults())) {
    Value resultSize = dispatchOp.getResultSize(resultIndex);

    // TODO(multi-device): possibly perform analysis to pick an affinity based
    // on usage. Today we assume the resource affinity matches the execution op.
    auto allocaOp = builder.create<IREE::Stream::AsyncAllocaOp>(
        dispatchOp.getLoc(), result.getType(), resultSize,
        dispatchOp.getAffinityAttr());

    auto operandIndex = dispatchOp.getResourceOperands().size();
    dispatchOp.getResourceOperandsMutable().append(allocaOp.getResult());
    dispatchOp.getResourceOperandSizesMutable().append(resultSize);
    dispatchOp.getResourceOperandOffsetsMutable().append(zeroOffset);
    dispatchOp.getResourceOperandEndsMutable().append(resultSize);
    dispatchOp.getResourceOperandLengthsMutable().append(resultSize);
    dispatchOp.setTiedResultOperandIndex(resultIndex, operandIndex);
  }

  // Sort the update ops in block order so that we dont accidentally move them
  // above the dispatch op in the next section which will cause a dominance
  // issue.
  llvm::sort(updateResultOps, [&](const auto &a, const auto &b) {
    return std::get<0>(a)->isBeforeInBlock(std::get<0>(b));
  });

  // Try to prepare each candidate for use by the dispatch. If we can't then
  // we skip the particular emplacement.
  bool didChange = false;
  for (auto [updateOp, resultIndex] : updateResultOps) {
    Value targetResource = updateOp.getTarget();
    if (targetResource.getDefiningOp() == dispatchOp) {
      // NOTE: we may have already replaced the update target with one of our
      // results - if so we need to find the operand to capture tied to that
      // new result instead of our own new result (which would make a cycle).
      targetResource = dispatchOp.getTiedResultOperand(targetResource);
    } else if (!targetResource) {
      continue;
    }

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
        !IREE::Util::tryMoveProducerBefore(targetResource, dispatchOp)) {
      // Failed to move while keeping valid SSA dominance.
      LLVM_DEBUG({
        llvm::dbgs() << "  ! failed to move producers from update ";
        updateOp.print(llvm::dbgs(), printingFlags);
        llvm::dbgs() << " before dispatch with result #" << resultIndex << "\n";
      });
      continue;
    }

    Value targetResourceSize = updateOp.getTargetSize();
    Value targetOffset = updateOp.getTargetOffset();
    Value targetEnd = updateOp.getTargetEnd();
    Value targetLength = updateOp.getUpdateSize();
    Value targetResult = updateOp.getResult();
    Value targetResultSize = updateOp.getTargetSize();

    // The dispatch operands are mixed types but all resource-related lists are
    // only for resources. operandIndex is in the mixed domain and we have to
    // calculate the corresponding resource domain index.
    auto operandIndex = dispatchOp.getTiedResultOperandIndex(resultIndex);
    operandIndex =
        *operandIndex - dispatchOp.getTiedOperandsIndexAndLength().first;
    assert(operandIndex.has_value() && "should have been tied above");
    unsigned resourceIndex = 0;
    for (unsigned i = 0; i < *operandIndex; ++i) {
      resourceIndex += isa<IREE::Stream::ResourceType>(
                           dispatchOp.getResourceOperands()[i].getType())
                           ? 1
                           : 0;
    }

    // Replace the operand with the target range.
    Value previousResource = dispatchOp.getResourceOperands()[*operandIndex];
    dispatchOp.getResourceOperandsMutable()
        .slice(*operandIndex, 1)
        .assign(targetResource);
    dispatchOp.getResourceOperandSizesMutable()
        .slice(resourceIndex, 1)
        .assign(targetResourceSize);
    dispatchOp.getResourceOperandOffsetsMutable()
        .slice(resourceIndex, 1)
        .assign(targetOffset);
    dispatchOp.getResourceOperandEndsMutable()
        .slice(resourceIndex, 1)
        .assign(targetEnd);
    dispatchOp.getResourceOperandLengthsMutable()
        .slice(resourceIndex, 1)
        .assign(targetLength);
    dispatchOp.getResultSizesMutable()
        .slice(resultIndex, 1)
        .assign(targetResultSize);

    // Replace users with the result of the dispatch op.
    LLVM_DEBUG({
      llvm::dbgs() << "  * replacing uses of ";
      targetResult.printAsOperand(llvm::dbgs(), printingFlags);
      llvm::dbgs() << " with update source ";
      updateOp.print(llvm::dbgs(), printingFlags);
      llvm::dbgs() << "\n";
    });
    replaceUsesAndTransfer(targetResult, updateOp.getUpdate(),
                           dispatchOp.getAffinityAttr());
    updateOp->erase();

    // If the previously captured resource is no longer used then delete it.
    // This cleans up some of the allocas we speculatively add above.
    if (previousResource.use_empty()) {
      if (auto *definingOp = previousResource.getDefiningOp()) {
        definingOp->erase();
      }
    }

    didChange = true;
  }

  return didChange;
}

// Emplaces allocations within |region|.
// Returns true if any allocations were elided by way of emplacement.
static bool emplaceAllocationsInRegion(Region &region) {
  bool didChange = false;
  for (auto &block : region.getBlocks()) {
    IndexSet indexSet(region.getLoc(), OpBuilder::atBlockBegin(&block));
    for (auto &op : block) {
      if (op.hasTrait<OpTrait::IREE::Stream::AsyncPhaseOp>()) {
        didChange = TypeSwitch<Operation *, bool>(&op)
                        // TODO(#11249): support in-place collective ops.
                        .Case<IREE::Stream::AsyncDispatchOp>([&](auto op) {
                          return tryEmplaceDispatchOp(op, indexSet);
                        })
                        .Default(false) ||
                    didChange;
      }
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
