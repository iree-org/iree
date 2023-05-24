// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-emplace-allocations"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

//===----------------------------------------------------------------------===//
// Emplacement
//===----------------------------------------------------------------------===//

// TODO(benvanik): make an interface on relevant ops?
// Models a transfer operation between a source and a target.
struct Transfer {
  Value sourceResource;
  Value sourceResourceSize;
  Optional<Value> sourceOffset;
  Optional<Value> sourceEnd;
  Value targetResource;
  Value targetResourceSize;
  Value targetOffset;
  Value targetEnd;
  Value length;
  Value result;
};

static Transfer getTransfer(IREE::Stream::AsyncUpdateOp updateOp) {
  Transfer transfer;
  transfer.sourceResource = updateOp.getUpdate();
  transfer.sourceResourceSize = updateOp.getUpdateSize();
  transfer.targetResource = updateOp.getTarget();
  transfer.targetResourceSize = updateOp.getTargetSize();
  transfer.targetOffset = updateOp.getTargetOffset();
  transfer.targetEnd = updateOp.getTargetEnd();
  transfer.length = updateOp.getUpdateSize();
  transfer.result = updateOp.getResult();
  return transfer;
}

static Transfer getTransfer(IREE::Stream::AsyncCopyOp copyOp) {
  Transfer transfer;
  transfer.sourceResource = copyOp.getSource();
  transfer.sourceResourceSize = copyOp.getSourceSize();
  transfer.sourceOffset = copyOp.getSourceOffset();
  transfer.sourceEnd = copyOp.getSourceEnd();
  transfer.targetResource = copyOp.getTarget();
  transfer.targetResourceSize = copyOp.getTargetSize();
  transfer.targetOffset = copyOp.getTargetOffset();
  transfer.targetEnd = copyOp.getTargetEnd();
  transfer.length = copyOp.getLength();
  transfer.result = copyOp.getResult();
  return transfer;
}

// Updates |baseOffset| and |baseEnd| by adding the |targetOffset|.
// Returns {baseOffset + targetOffset, baseEnd += targetOffset}.
static std::pair<Value, Value> adjustRange(Value baseOffset, Value baseEnd,
                                           Value targetOffset,
                                           OpBuilder &builder) {
  return std::make_pair(
      builder.createOrFold<arith::AddIOp>(
          builder.getFusedLoc({baseOffset.getLoc(), targetOffset.getLoc()}),
          baseOffset, targetOffset),
      builder.createOrFold<arith::AddIOp>(
          builder.getFusedLoc({baseEnd.getLoc(), targetOffset.getLoc()}),
          baseEnd, targetOffset));
}

static void replaceUsesAndTransfer(Value oldValue, Value newValue) {
  assert(oldValue.getType().isa<IREE::Stream::ResourceType>());
  assert(newValue.getType().isa<IREE::Stream::ResourceType>());

  if (oldValue.getType() == newValue.getType()) {
    oldValue.replaceAllUsesWith(newValue);
    return;
  }

  OpBuilder builder(newValue.getContext());
  builder.setInsertionPointAfterValue(newValue);
  Value newValueSize = IREE::Util::SizeAwareTypeInterface::queryValueSize(
      newValue.getLoc(), newValue, builder);
  IREE::Stream::AffinityAttr sourceAffinity;
  IREE::Stream::AffinityAttr resultAffinity;
  Value transferValue = builder.create<IREE::Stream::AsyncTransferOp>(
      newValue.getLoc(), oldValue.getType(), newValue, newValueSize,
      newValueSize, sourceAffinity, resultAffinity);
  oldValue.replaceAllUsesWith(transferValue);
}

// Updates a dispatch operand to reference the transfer target.
// Expects that the transfer source is one or more of the operands.
static void applyDispatchOperandTransfer(
    IREE::Stream::AsyncDispatchOp dispatchOp, Transfer &transfer) {
  OpBuilder builder(dispatchOp);

  // Note that there may be multiple users and this particular op may use the
  // source multiple times with different ranges.
  SmallVector<unsigned> operandIndices;
  for (auto &use : transfer.sourceResource.getUses()) {
    if (use.getOwner() != dispatchOp) continue;
    operandIndices.push_back(use.getOperandNumber());
  }
  assert(!operandIndices.empty() &&
         "transfer source must be 1+ operands of the dispatch");

  // Swap each operand to point at the target resource and adjust the range by
  // factoring in the target offset.
  for (unsigned operandIndex : operandIndices) {
    LLVM_DEBUG({
      auto opFlags =
          OpPrintingFlags().elideLargeElementsAttrs().assumeVerified();
      llvm::dbgs() << "  + updating operand #" << operandIndex
                   << " of dispatch to transfer target\n";
      llvm::dbgs() << "    ";
      transfer.sourceResource.printAsOperand(llvm::dbgs(), opFlags);
      llvm::dbgs() << " -> ";
      transfer.targetResource.printAsOperand(llvm::dbgs(), opFlags);
      llvm::dbgs() << "[";
      transfer.targetOffset.printAsOperand(llvm::dbgs(), opFlags);
      llvm::dbgs() << " to ";
      transfer.targetEnd.printAsOperand(llvm::dbgs(), opFlags);
      llvm::dbgs() << " for ";
      transfer.length.printAsOperand(llvm::dbgs(), opFlags);
      llvm::dbgs() << "] = ";
      transfer.result.printAsOperand(llvm::dbgs(), opFlags);
      llvm::dbgs() << "\n";
    });
    dispatchOp.getResourceOperandsMutable()[operandIndex] =
        transfer.targetResource;
    dispatchOp.getResourceOperandSizesMutable()[operandIndex] =
        transfer.targetResourceSize;
    auto [newOffset, newEnd] =
        adjustRange(dispatchOp.getResourceOperandOffsetsMutable()[operandIndex],
                    dispatchOp.getResourceOperandEndsMutable()[operandIndex],
                    transfer.targetOffset, builder);
    dispatchOp.getResourceOperandOffsetsMutable()[operandIndex] = newOffset;
    dispatchOp.getResourceOperandEndsMutable()[operandIndex] = newEnd;
  }
}

// Updates a dispatch result to be stored in the transfer target.
// Expects that the transfer source is a result of the dispatch.
static void applyDispatchResultTransfer(
    IREE::Stream::AsyncDispatchOp dispatchOp, Transfer &transfer) {
  OpBuilder builder(dispatchOp);

  auto result = transfer.sourceResource.cast<OpResult>();
  assert(result && "transfer source must be a result of the dispatch");
  unsigned resultIndex = result.getResultNumber();

  LLVM_DEBUG({
    auto opFlags = OpPrintingFlags().elideLargeElementsAttrs().assumeVerified();
    llvm::dbgs() << "  * updating result #" << resultIndex << " of dispatch ";
    result.printAsOperand(llvm::dbgs(), opFlags);
    llvm::dbgs() << " to transfer target\n";
    llvm::dbgs() << "    ";
    transfer.sourceResource.printAsOperand(llvm::dbgs(), opFlags);
    llvm::dbgs() << " -> ";
    transfer.targetResource.printAsOperand(llvm::dbgs(), opFlags);
    llvm::dbgs() << "[";
    transfer.targetOffset.printAsOperand(llvm::dbgs(), opFlags);
    llvm::dbgs() << " to ";
    transfer.targetEnd.printAsOperand(llvm::dbgs(), opFlags);
    llvm::dbgs() << " for ";
    transfer.length.printAsOperand(llvm::dbgs(), opFlags);
    llvm::dbgs() << "] = ";
    transfer.result.printAsOperand(llvm::dbgs(), opFlags);
    llvm::dbgs() << "\n";
  });

  // Add operand and tie the result.
  int operandIndex = dispatchOp.getResourceOperands().size();
  dispatchOp.getResourceOperandsMutable().append(transfer.targetResource);
  dispatchOp.getResourceOperandSizesMutable().append(
      transfer.targetResourceSize);
  dispatchOp.getResourceOperandOffsetsMutable().append(transfer.targetOffset);
  dispatchOp.getResourceOperandEndsMutable().append(transfer.targetEnd);
  dispatchOp.getResourceOperandLengthsMutable().append(transfer.length);
  dispatchOp.setTiedResultOperandIndex(resultIndex, operandIndex);

  // Update result size (requires this dance as [] is a no-op!).
  SmallVector<Value> resultSizes = dispatchOp.getResultSizes();
  resultSizes[resultIndex] = transfer.targetResourceSize;
  dispatchOp.getResultSizesMutable().assign(resultSizes);
  result.setType(transfer.result.getType());

  // Replace users with the result of the dispatch op.
  replaceUsesAndTransfer(transfer.result, result);
}

// Returns true if |sourceOp| can be placed into storage used by |targetOp|.
static bool isPlacementCompatible(Operation *sourceOp, Transfer &transfer,
                                  DominanceInfo &domInfo) {
  // DO NOT SUBMIT
  // can the target storage (alloca/etc) be moved before?
  // if not then not compatible, still may be partially compatible with others
  //
  // what even does this mean?
  // ensure that value is usable on operands
  // disallow everything but dispatches today?
  return true;
}

// Tries to store allocations for operations in the target of a transfer.
// Any consumers of the result are changed to point to the newly updated
// subrange in the target resource.
//
// Requires:
//  - source of the transfer must be visible for analysis
//  - source of the transfer must be in the same execution scope
//  - no conflicting writes to the target resource
//
// Returns true if any IR changes were made.
static bool tryEmplaceFromTransfer(Operation *transferOp, Transfer transfer,
                                   DominanceInfo &domInfo) {
  // TODO(benvanik): figure out partial source transfers.
  if (transfer.sourceOffset.has_value()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "  ! skipping, source has an offset that isn't yet supported\n");
    return false;
  }

  // Must have visibility into the source of the transfer.
  // A more sophisticated analysis is required in order to emplace across block
  // or function boundaries.
  auto *sourceOp = transfer.sourceResource.getDefiningOp();
  if (!sourceOp) {
    LLVM_DEBUG(llvm::dbgs() << "  ! skipping, source comes from block arg\n");
    return false;
  }
  if (!isa<IREE::Stream::AsyncDispatchOp>(sourceOp)) {
    // TODO(benvanik): support tied results so we can do copies/collectives.
    LLVM_DEBUG(llvm::dbgs() << "  ! skipping, non-dispatch producers ("
                            << sourceOp->getName() << ") not yet supported\n");
    return false;
  }
  auto sourceTiedOp = dyn_cast<IREE::Util::TiedOpInterface>(sourceOp);
  if (sourceTiedOp) {
    if (sourceTiedOp.getTiedResultOperand(transfer.sourceResource)) {
      // TODO(benvanik): support tied results.
      LLVM_DEBUG(llvm::dbgs() << "  ! skipping, tied result in producer "
                              << sourceOp->getName() << " not yet supported\n");
      return false;
    }
  }

  // TODO(benvanik): better analysis using Explorer for walking across incoming
  // branches (emplace sources on all paths), out of nested regions (emplace
  // sources outside of an scf.if/for), across selects, tied operands, etc.
  // For now we require that all ops live in the same block and don't try to
  // walk through ties.

  if (IREE::Util::TiedOpInterface::hasAnyTiedUses(transfer.sourceResource)) {
    // TODO(benvanik): support propagating the change through; issue is that the
    // result size changes and propagating that to all subsequent ops requires
    // some non-trivial work (and may be impossible with the current IR). This
    // case is uncommon today but may get more important as more dispatches use
    // in-place operations.
    LLVM_DEBUG(llvm::dbgs() << "  ! skipping, tied uses of result by consumer "
                               "ops not yet supported");
    return false;
  }

  // Verify that all users of the transfer source are compatible.
  // They must (today) all be in the same execution scope and not have any
  // hazards with the target range.
  for (auto *userOp : transfer.sourceResource.getUsers()) {
    if (userOp == transferOp) continue;  // ignore the transfer under inspection
    if (!isPlacementCompatible(userOp, transfer, domInfo)) return false;
  }

  if (!IREE::Util::tryMoveProducerBefore(transfer.targetResource, sourceOp)) {
    LLVM_DEBUG({
      auto opFlags =
          OpPrintingFlags().elideLargeElementsAttrs().assumeVerified();
      llvm::dbgs() << "  ! failed to move target producer before source";
      if (auto targetOp = transfer.targetResource.getDefiningOp()) {
        llvm::dbgs() << " ";
        targetOp->print(llvm::dbgs(), opFlags);
      }
      llvm::dbgs() << "\n";
    });
    return false;
  }

  if (sourceTiedOp && transfer.targetResource.getDefiningOp() == sourceTiedOp) {
    // NOTE: we may have already replaced the update target with one of our
    // results - if so we need to find the operand to capture tied to that
    // new result instead of our own new result (which would make a cycle).
    transfer.targetResource =
        sourceTiedOp.getTiedResultOperand(transfer.targetResource);
  }

  // Emplace the original result into the target range.
  // TODO(benvanik): support tied results so we can do copies/collectives.
  TypeSwitch<Operation *>(sourceOp).Case<IREE::Stream::AsyncDispatchOp>(
      [&](auto sourceOp) { applyDispatchResultTransfer(sourceOp, transfer); });

  // Update all users to the new transfer target range.
  for (auto *userOp : transfer.sourceResource.getUsers()) {
    if (userOp == transferOp) continue;  // ignore the transfer under inspection
    // TODO(benvanik): support tied results so we can do copies/collectives.
    TypeSwitch<Operation *>(userOp).Case<IREE::Stream::AsyncDispatchOp>(
        [&](auto userOp) { applyDispatchOperandTransfer(userOp, transfer); });
  }

  // Erase the original transfer op now that it's been folded.
  if (transferOp->use_empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  x deleting original transfer op\n");
    transferOp->erase();
  } else {
    transferOp->emitWarning() << "op retained because it still has uses even "
                                 "after emplacing the transfer source\n";
  }

  return /*didChange=*/true;
}

// Tries to store allocations for operations in the target of an update.
//
// Example:
//  %update = stream.async.dispatch ... -> !stream.resource
//  %target1 = stream.async.update %update, %target0[...]
// ->
//  %target1 = stream.async.dispatch ..., %target0[...] -> %target0
static bool tryEmplaceFromUpdateOp(IREE::Stream::AsyncUpdateOp updateOp,
                                   DominanceInfo &domInfo) {
  LLVM_DEBUG({
    llvm::dbgs() << " tryEmplaceFromUpdateOp: ";
    updateOp.print(
        llvm::dbgs(),
        OpPrintingFlags().elideLargeElementsAttrs().assumeVerified());
    llvm::dbgs() << "\n";
  });
  return tryEmplaceFromTransfer(updateOp, getTransfer(updateOp), domInfo);
}

// Tries to store allocations for operations in the target of a copy.
//
// Example:
//  %source = stream.async.dispatch ... -> !stream.resource
//  %target1 = stream.async.copy %source[...], %target0[...]
// ->
//  %target1 = stream.async.dispatch ..., %target0[...] -> %target0
static bool tryEmplaceFromCopyOp(IREE::Stream::AsyncCopyOp copyOp,
                                 DominanceInfo &domInfo) {
  LLVM_DEBUG({
    llvm::dbgs() << " tryEmplaceFromCopyOp: ";
    copyOp.print(llvm::dbgs(),
                 OpPrintingFlags().elideLargeElementsAttrs().assumeVerified());
    llvm::dbgs() << "\n";
  });
  return tryEmplaceFromTransfer(copyOp, getTransfer(copyOp), domInfo);
}

// Emplaces allocations within |region|.
// Returns true if any allocations were elided by way of emplacement.
static bool emplaceAllocationsInRegion(Region &region) {
  LLVM_DEBUG(llvm::dbgs() << "emplaceAllocationsInRegion("
                          << region.getParentOp()->getName() << ")\n");
  DominanceInfo domInfo(region.getParentOp());
  bool didChange = false;
  for (auto &block : region.getBlocks()) {
    for (auto &op : llvm::make_early_inc_range(block)) {
      if (!op.hasTrait<OpTrait::IREE::Stream::AsyncPhaseOp>()) continue;
      // TODO(benvanik): support placement for more ops e.g. collectives.
      didChange = TypeSwitch<Operation *, bool>(&op)
                      .Case<IREE::Stream::AsyncUpdateOp>([&](auto op) {
                        return tryEmplaceFromUpdateOp(op, domInfo);
                      })
                      .Case<IREE::Stream::AsyncCopyOp>([&](auto op) {
                        return tryEmplaceFromCopyOp(op, domInfo);
                      })
                      .Default(false) ||
                  didChange;
    }
  }
  return didChange;
}

//===----------------------------------------------------------------------===//
// -iree-stream-emplace-allocations
//===----------------------------------------------------------------------===//

class EmplaceAllocationsPass
    : public EmplaceAllocationsBase<EmplaceAllocationsPass> {
 public:
  EmplaceAllocationsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    bool didChange = false;
    getOperation()->walk([&](Region *region) {
      didChange = emplaceAllocationsInRegion(*region) || didChange;
    });
    // TODO(benvanik): run canonicalization patterns inline if anything changed.
    (void)didChange;
  }
};

}  // namespace

std::unique_ptr<OperationPass<>> createEmplaceAllocationsPass() {
  return std::make_unique<EmplaceAllocationsPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
