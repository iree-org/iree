// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Utils/RegionOpUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-emplace-transients"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_EMPLACETRANSIENTSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-emplace-transients
//===----------------------------------------------------------------------===//

// Information about transient allocations associated with a storage buffer.
struct TransientResource {
  Value storage;
  Value storageSize;
  IREE::Stream::AffinityAttr affinity;
  SmallVector<IREE::Stream::ResourceTransientsOp> transientsOps;
  SmallVector<IREE::Stream::ResourceAllocaOp> allocaOps;
  SmallVector<IREE::Stream::ResourceDeallocaOp> deallocaOps;
};

// Represents a "slot" in the packed allocation.
// Multiple allocations can share a slot if they're mutually exclusive
// (e.g., in different branches of the same scf.if).
struct AllocationSlot {
  SmallVector<IREE::Stream::ResourceAllocaOp> allocations;
  Value conservativeSize; // max(all sizes in this slot)
  int64_t slotId;         // unique identifier for this slot
};

// Walks the timeline backwards from stream.resource.transients ops to gather
// all alloca/dealloca ops that feed into each unique storage buffer.
static LogicalResult
gatherTransientResources(FunctionOpInterface funcOp, Explorer &explorer,
                         SmallVector<TransientResource> &transientResources) {
  // Find all stream.resource.transients ops and group by storage buffer.
  SmallVector<IREE::Stream::ResourceTransientsOp> transientsOps;
  Value uniqueStorage;
  funcOp.walk([&](IREE::Stream::ResourceTransientsOp op) {
    transientsOps.push_back(op);

    // Validate storage buffer - must all be the same SSA value.
    Value storage = op.getStorage();
    if (!uniqueStorage) {
      uniqueStorage = storage;
      LLVM_DEBUG({
        LDBG() << "found storage buffer: ";
        storage.printAsOperand(llvm::dbgs(), explorer.getAsmState());
        llvm::dbgs() << "\n";
      });
    } else if (uniqueStorage != storage) {
      op.emitError("multiple unique storage buffers not supported - all "
                   "stream.resource.transients ops must use the same storage");
      return WalkResult::interrupt();
    }

    LLVM_DEBUG({
      LDBG() << "found transients op: ";
      op->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });
    return WalkResult::advance();
  });

  // Early exit if no transients found.
  if (transientsOps.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[emplace-transients] no transients ops found, skipping\n");
    return success();
  }

  LLVM_DEBUG(LDBG() << "found " << transientsOps.size()
                    << " transients op(s)\n");

  LLVM_DEBUG(LDBG() << "starting timeline traversal\n");

  // Create a transient resource entry for this storage buffer.
  TransientResource resource;
  resource.storage = uniqueStorage;
  resource.storageSize = transientsOps.front().getStorageSize();
  resource.affinity = transientsOps.front().getAffinityAttr();
  resource.transientsOps = transientsOps;

  // Seed worklist with await timepoints from all transients ops.
  DenseSet<Value> visitedTimepoints;
  SmallVector<Value> worklist;
  for (auto transientsOp : transientsOps) {
    if (auto awaitTp = transientsOp.getAwaitTimepoint()) {
      if (visitedTimepoints.insert(awaitTp).second) {
        worklist.push_back(awaitTp);
        LLVM_DEBUG({
          LDBG() << "seeding worklist with: ";
          awaitTp.printAsOperand(llvm::dbgs(), explorer.getAsmState());
          llvm::dbgs() << "\n";
        });
      }
    }
  }

  // Process worklist: use Explorer to walk defining ops and follow timeline
  // backwards.
  SetVector<IREE::Stream::ResourceAllocaOp> allocaOpsSet;
  SetVector<IREE::Stream::ResourceDeallocaOp> deallocaOpsSet;
  while (!worklist.empty()) {
    Value timepoint = worklist.pop_back_val();

    LLVM_DEBUG({
      LDBG() << "processing timepoint: ";
      timepoint.printAsOperand(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });

    // Use Explorer to walk all defining ops (handles block args, regions, etc).
    auto result = explorer.walkDefiningOps(
        timepoint,
        [&](OpResult definingResult) -> WalkResult {
          Operation *definingOp = definingResult.getOwner();

          LLVM_DEBUG({
            LDBG() << "  defining op: ";
            definingOp->print(llvm::dbgs(), explorer.getAsmState());
            llvm::dbgs() << "\n";
          });

          // Check if this is an alloca or dealloca op.
          // Only process transient lifetime resources.
          if (auto allocaOp =
                  dyn_cast<IREE::Stream::ResourceAllocaOp>(definingOp)) {
            auto resultType = llvm::cast<IREE::Stream::ResourceType>(
                allocaOp.getResult().getType());
            if (resultType.getLifetime() == IREE::Stream::Lifetime::Transient) {
              allocaOpsSet.insert(allocaOp);
              LLVM_DEBUG(
                  llvm::dbgs()
                  << "[emplace-transients]   found transient alloca op\n");
            }
          } else if (auto deallocaOp =
                         dyn_cast<IREE::Stream::ResourceDeallocaOp>(
                             definingOp)) {
            auto operandType = llvm::cast<IREE::Stream::ResourceType>(
                deallocaOp.getOperand().getType());
            if (operandType.getLifetime() ==
                IREE::Stream::Lifetime::Transient) {
              deallocaOpsSet.insert(deallocaOp);
              LLVM_DEBUG(
                  llvm::dbgs()
                  << "[emplace-transients]   found transient dealloca op\n");
            }
          }

          // If this is a timeline-aware op, add its await timepoints to
          // worklist.
          if (auto timelineOp =
                  dyn_cast<IREE::Stream::TimelineOpInterface>(definingOp)) {
            SmallVector<Value> awaitTimepoints =
                timelineOp.getAwaitTimepoints();
            if (!awaitTimepoints.empty()) {
              LLVM_DEBUG({
                llvm::dbgs()
                    << "[emplace-transients]   timeline-aware op with "
                    << awaitTimepoints.size() << " await timepoint(s)\n";
              });
              for (Value awaitTp : awaitTimepoints) {
                if (visitedTimepoints.insert(awaitTp).second) {
                  worklist.push_back(awaitTp);
                  LLVM_DEBUG({
                    llvm::dbgs()
                        << "[emplace-transients]     added to worklist: ";
                    awaitTp.printAsOperand(llvm::dbgs(),
                                           explorer.getAsmState());
                    llvm::dbgs() << "\n";
                  });
                } else {
                  LLVM_DEBUG({
                    llvm::dbgs()
                        << "[emplace-transients]     already visited: ";
                    awaitTp.printAsOperand(llvm::dbgs(),
                                           explorer.getAsmState());
                    llvm::dbgs() << "\n";
                  });
                }
              }
            }
          }

          return WalkResult::advance();
        },
        TraversalBehavior::DEFAULT);

    if (result == TraversalResult::INCOMPLETE) {
      // NOTE: we don't error here - incomplete traversal may just mean we
      // couldn't see through some indirect patterns, but we continue with
      // what we found.
      LLVM_DEBUG(
          llvm::dbgs()
          << "[emplace-transients]   incomplete traversal (indirect calls?)\n");
    }
  }

  // Extract vectors from SetVectors and reverse to restore program order
  // (backwards timeline walk discovers them in reverse program order).
  resource.allocaOps = allocaOpsSet.takeVector();
  std::reverse(resource.allocaOps.begin(), resource.allocaOps.end());
  resource.deallocaOps = deallocaOpsSet.takeVector();
  std::reverse(resource.deallocaOps.begin(), resource.deallocaOps.end());

  // Filter deallocas to only those that deallocate our specific allocas.
  // This prevents removing deallocas for other transient resources that
  // happen to be in the timeline but aren't being emplaced.
  DenseSet<Value> allocaResults;
  for (auto allocaOp : resource.allocaOps) {
    allocaResults.insert(allocaOp.getResult());
  }

  SmallVector<IREE::Stream::ResourceDeallocaOp> filteredDeallocaOps;
  for (auto deallocaOp : resource.deallocaOps) {
    if (allocaResults.contains(deallocaOp.getOperand())) {
      filteredDeallocaOps.push_back(deallocaOp);
    } else {
      LLVM_DEBUG({
        LDBG() << "  skipping dealloca for "
                  "non-emplaced transient: ";
        deallocaOp->print(llvm::dbgs(), explorer.getAsmState());
        llvm::dbgs() << "\n";
      });
    }
  }
  resource.deallocaOps = std::move(filteredDeallocaOps);

  LLVM_DEBUG({
    LDBG() << "timeline traversal complete: found " << resource.allocaOps.size()
           << " alloca(s) and " << resource.deallocaOps.size()
           << " dealloca(s)\n";
    for (auto allocaOp : resource.allocaOps) {
      LDBG() << "  alloca: ";
      allocaOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    }
    for (auto deallocaOp : resource.deallocaOps) {
      LDBG() << "  dealloca: ";
      deallocaOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    }
  });

  // Add this resource to the output list.
  transientResources.push_back(std::move(resource));
  return success();
}

// Returns true if an operation is pure (no memory effects, no calls).
static bool isOperationPureRecursively(Operation *op) {
  // Check for function calls.
  if (isa<CallOpInterface>(op)) {
    return false;
  }

  // Check for memory effects.
  // Note: isMemoryEffectFree automatically handles HasRecursiveMemoryEffects
  // and will check nested operations in regions.
  return mlir::isMemoryEffectFree(op);
}

// Finds the dominant insertion point for size computations by walking up the
// parent chain to find any RegionBranchOpInterface ops. Returns the operation
// before which size computations should be inserted to dominate all uses.
static Operation *findDominantInsertionPoint(Operation *startOp,
                                             Explorer &explorer) {
  Operation *insertionPoint = startOp;
  Operation *parentOp = insertionPoint->getParentOp();
  while (parentOp && !isa<FunctionOpInterface>(parentOp)) {
    // If parent implements RegionBranchOpInterface, insert before it to
    // dominate all control flow paths through the region.
    if (isa<RegionBranchOpInterface>(parentOp)) {
      insertionPoint = parentOp;
      LLVM_DEBUG({
        LDBG() << "op is nested in region branch "
                  "op, inserting before: ";
        insertionPoint->print(llvm::dbgs(), explorer.getAsmState());
        llvm::dbgs() << "\n";
      });
    }
    parentOp = parentOp->getParentOp();
  }
  return insertionPoint;
}

// Finds the tightest insertion point that dominates all allocas.
// Returns the operation before which we should insert pack/storage operations.
static Operation *
findDominantInsertionPoint(DominanceInfo &domInfo,
                           ArrayRef<IREE::Stream::ResourceAllocaOp> allocaOps,
                           FunctionOpInterface funcOp, Explorer &explorer) {
  SmallVector<Operation *> allocaOpPtrs;
  for (auto allocaOp : allocaOps) {
    allocaOpPtrs.push_back(allocaOp.getOperation());
  }

  // Find common dominator block of all allocas.
  Block *commonBlock = allocaOpPtrs.front()->getBlock();
  for (size_t i = 1; i < allocaOpPtrs.size(); ++i) {
    commonBlock = domInfo.findNearestCommonDominator(
        commonBlock, allocaOpPtrs[i]->getBlock());
    if (!commonBlock) {
      funcOp.emitError(
          "cannot find common dominator for allocas - internal error");
      return nullptr;
    }
  }

  // For each alloca, find its "anchor" operation in the common dominator block.
  // The anchor is either the alloca itself (if in common block) or the
  // operation in the common block that contains the alloca's region.
  // We want the EARLIEST anchor to get the tightest insertion point.
  Operation *earliestAnchor = nullptr;
  for (Operation *allocaOp : allocaOpPtrs) {
    // Walk up the parent chain until we reach an operation in the common block.
    Operation *anchor = allocaOp;
    while (anchor->getBlock() != commonBlock) {
      anchor = anchor->getParentOp();
      if (!anchor) {
        funcOp.emitError("alloca has no parent in common dominator block");
        return nullptr;
      }
    }

    // Track the earliest anchor.
    if (!earliestAnchor || anchor->isBeforeInBlock(earliestAnchor)) {
      earliestAnchor = anchor;
    }
  }

  LLVM_DEBUG({
    LDBG() << "found tightest insertion point: ";
    earliestAnchor->print(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << "\n";
  });

  return earliestAnchor;
}

// Represents a backward slice of operations needed to compute a size value.
// The slice stops at region boundaries - it includes all operations within
// the region needed to compute the size, plus any captured values from outside.
struct SizeComputationSlice {
  SmallVector<Operation *> opsToClone; // In topological order.
  SmallVector<Value> capturedValues;   // Values from outside region.
  Value rootSizeValue;                 // The size we're slicing from.

  SizeComputationSlice() = default;
  SizeComputationSlice(Value root) : rootSizeValue(root) {}
};

// Computes backward slice of size computation up to region boundary.
// Only includes pure operations (no side effects, no calls).
// Captures any values needed from outside the region.
static LogicalResult computeSizeSliceInRegion(Value sizeValue, Region *region,
                                              SizeComputationSlice &slice,
                                              Explorer &explorer) {
  slice.rootSizeValue = sizeValue;

  // If the size value is a block argument, it's captured from outside.
  if (auto blockArg = dyn_cast<BlockArgument>(sizeValue)) {
    slice.capturedValues.push_back(sizeValue);
    return success();
  }

  // Get the defining op - if it's outside the region, it's captured.
  Operation *definingOp = sizeValue.getDefiningOp();
  if (!definingOp) {
    return failure();
  }

  Region *definingRegion = definingOp->getParentRegion();
  LLVM_DEBUG({
    LDBG() << "computeSizeSliceInRegion for value: ";
    sizeValue.printAsOperand(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << "\n";
    LDBG() << "  definingOp parentRegion: " << definingRegion << "\n";
    LDBG() << "  target region: " << region << "\n";
    LDBG() << "  same? " << (definingRegion == region) << "\n";
    llvm::dbgs()
        << "[emplace-transients]   definingRegion is ancestor of target? "
        << definingRegion->isAncestor(region) << "\n";
  });

  // Check if the size value is accessible from the target region.
  // It's accessible if:
  // 1. Defined in the same region
  // 2. Defined in an ancestor region (outer scope)
  // If it's in a sibling or descendant region, we need to clone it.
  if (definingRegion == region) {
    // Same region - need to clone the computation.
    LLVM_DEBUG(llvm::dbgs()
               << "[emplace-transients]   in same region, will clone\n");
  } else if (definingRegion->isAncestor(region)) {
    // Defined in outer scope - already accessible, treat as captured.
    LLVM_DEBUG(LDBG() << "  defined in ancestor "
                         "region, treating as captured\n");
    slice.capturedValues.push_back(sizeValue);
    return success();
  } else {
    // Defined in sibling or other region - recursively compute slice there.
    // The hoistSizesConservatively function will merge slices from different
    // regions and clone all necessary operations to the insertion point.
    LLVM_DEBUG(llvm::dbgs()
               << "[emplace-transients]   defined in sibling region, computing "
                  "slice recursively\n");
    return computeSizeSliceInRegion(sizeValue, definingRegion, slice, explorer);
  }

  // Compute backward slice within the region using MLIR's slice analysis.
  BackwardSliceOptions options;
  options.omitBlockArguments = false; // We want to see captured values.
  options.inclusive = true;           // Include the root operation.

  // Filter: only include ops in the same region, and only pure ops.
  options.filter = [&](Operation *op) {
    // Must be in the target region.
    if (op->getParentRegion() != region) {
      return false;
    }
    // Must be pure (no side effects, no calls).
    if (!isOperationPureRecursively(op)) {
      return false;
    }
    return true;
  };

  // NOTE: getBackwardSlice may return failure, but that's okay - we'll just
  // have an empty slice which we handle below.
  llvm::SetVector<Operation *> sliceOps;
  [[maybe_unused]] LogicalResult sliceResult =
      getBackwardSlice(sizeValue, &sliceOps, options);

  // Ensure the defining op itself is in the slice if it's in the region.
  // getBackwardSlice might not include leaf ops (those with no dependencies).
  if (definingOp->getParentRegion() == region &&
      isOperationPureRecursively(definingOp)) {
    sliceOps.insert(definingOp);
  }

  LLVM_DEBUG({
    LDBG() << "  backward slice found " << sliceOps.size()
           << " op(s) for size value: ";
    sizeValue.printAsOperand(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << "\n";
    if (auto defOp = sizeValue.getDefiningOp()) {
      LDBG() << "  defined by: ";
      defOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    }
  });

  if (sliceOps.empty()) {
    // Empty slice - the size value must be captured from outside or a block
    // arg.
    LLVM_DEBUG(llvm::dbgs()
               << "[emplace-transients]   empty slice, treating as captured\n");
    return failure();
  }

  // Separate captured values from ops to clone.
  DenseSet<Value> capturedSet;
  for (Operation *op : sliceOps) {
    for (Value operand : op->getOperands()) {
      // If operand is from outside the region or not in our slice, it's
      // captured.
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (blockArg.getParentRegion() != region) {
          capturedSet.insert(operand);
        }
      } else if (Operation *operandOp = operand.getDefiningOp()) {
        if (!region->isAncestor(operandOp->getParentRegion()) ||
            !sliceOps.contains(operandOp)) {
          capturedSet.insert(operand);
        }
      }
    }
  }

  slice.opsToClone.assign(sliceOps.begin(), sliceOps.end());
  slice.capturedValues.assign(capturedSet.begin(), capturedSet.end());

  LLVM_DEBUG({
    LDBG() << "computed slice with " << slice.opsToClone.size() << " op(s) and "
           << slice.capturedValues.size() << " captured value(s)\n";
    for (Operation *op : slice.opsToClone) {
      LDBG() << "  op: ";
      op->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    }
    for (Value captured : slice.capturedValues) {
      LDBG() << "  captured: ";
      captured.printAsOperand(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    }
  });

  return success();
}

// Computes conservative maximum of multiple potential size values.
// Creates arith.maxui chain: maxui(maxui(a, b), c) for [a, b, c].
// Returns a single Value representing the maximum, or the single input if only
// one value is provided.
static Value computeConservativeMax(OpBuilder &builder, Location loc,
                                    ArrayRef<Value> potentialSizes) {
  assert(!potentialSizes.empty() && "must have at least one size");

  // Single value - no max needed.
  if (potentialSizes.size() == 1) {
    return potentialSizes[0];
  }

  // Build chain of maxui operations: max(max(a, b), c).
  Value currentMax = potentialSizes[0];
  for (auto potentialSize : potentialSizes) {
    currentMax =
        arith::MaxUIOp::create(builder, loc, currentMax, potentialSize);
  }

  return currentMax;
}

// Clones operations from a slice and returns a mapping from old values to new.
// Assumes operations are in topological order.
// Captured values are mapped to themselves (they're already hoisted and
// available).
static DenseMap<Value, Value> cloneSliceOps(ArrayRef<Operation *> opsToClone,
                                            ArrayRef<Value> capturedValues,
                                            OpBuilder &builder,
                                            Explorer &explorer) {
  // Map captured values to themselves (they're already defined and available).
  IRMapping mapping;
  for (Value captured : capturedValues) {
    mapping.map(captured, captured);
  }

  LLVM_DEBUG({
    LDBG() << "IRMapping has " << capturedValues.size()
           << " captured value(s)\n";
    for (Value captured : capturedValues) {
      LDBG() << "  mapped: ";
      captured.printAsOperand(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << " -> ";
      mapping.lookup(captured).printAsOperand(llvm::dbgs(),
                                              explorer.getAsmState());
      llvm::dbgs() << "\n";
    }
  });

  DenseMap<Value, Value> valueMap;
  for (Operation *op : opsToClone) {
    LLVM_DEBUG({
      LDBG() << "cloning op: ";
      op->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });

    // Clone the operation using the value mapping.
    Operation *clonedOp = builder.clone(*op, mapping);

    // Update mapping with results.
    for (auto [origResult, clonedResult] :
         llvm::zip(op->getResults(), clonedOp->getResults())) {
      mapping.map(origResult, clonedResult);
      valueMap[origResult] = clonedResult;
    }
  }
  return valueMap;
}

// Materializes size computations from slices at insertion point and computes
// conservative maximum. This is Tier 2: generic conservative approach.
// Clones all operations from all slices and inserts arith.maxui to compute
// the upper bound.
static FailureOr<Value>
hoistSizesConservatively(ArrayRef<SizeComputationSlice> slices,
                         Operation *insertionPoint, Explorer &explorer) {
  if (slices.empty()) {
    return failure();
  }

  LLVM_DEBUG(LDBG() << "hoisting sizes conservatively for " << slices.size()
                    << " slice(s)\n");

  OpBuilder builder(insertionPoint->getContext());
  builder.setInsertionPoint(insertionPoint);

  // Find the latest defining operation among all captured values.
  // We need to clone operations AFTER all captured values are defined.
  Operation *latestCapturedDefOp = nullptr;
  for (const auto &slice : slices) {
    for (Value capturedValue : slice.capturedValues) {
      if (auto definingOp = capturedValue.getDefiningOp()) {
        if (!latestCapturedDefOp ||
            latestCapturedDefOp->isBeforeInBlock(definingOp)) {
          latestCapturedDefOp = definingOp;
        }
      }
    }
  }

  // Set insertion point after all captured values are defined.
  if (latestCapturedDefOp) {
    builder.setInsertionPointAfter(latestCapturedDefOp);
    LLVM_DEBUG({
      LDBG() << "  setting insertion point after: ";
      latestCapturedDefOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });
  }

  // Collect all unique operations to clone across all slices.
  // Use SetVector to deduplicate (same op may appear in multiple slices).
  llvm::SetVector<Operation *> allOpsToClone;
  llvm::SetVector<Value> allCapturedValues;
  for (const auto &slice : slices) {
    allOpsToClone.insert(slice.opsToClone.begin(), slice.opsToClone.end());
    allCapturedValues.insert(slice.capturedValues.begin(),
                             slice.capturedValues.end());
  }
  LLVM_DEBUG({
    LDBG() << "  deduped: " << allOpsToClone.size()
           << " unique op(s) to clone, " << allCapturedValues.size()
           << " unique captured value(s)\n";
    for (Operation *op : allOpsToClone) {
      LDBG() << "    op to clone: ";
      op->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    }
    for (Value val : allCapturedValues) {
      LDBG() << "    captured value: ";
      val.printAsOperand(llvm::dbgs(), explorer.getAsmState());
      if (auto defOp = val.getDefiningOp()) {
        llvm::dbgs() << " (defined by: ";
        defOp->print(llvm::dbgs(), explorer.getAsmState());
        llvm::dbgs() << ")";
      }
      llvm::dbgs() << "\n";
    }
  });

  // Re-sort deduplicated operations in topological order.
  // SetVector insertion doesn't preserve topological order across slices.
  mlir::topologicalSort(allOpsToClone);

  // Clone all unique operations once.
  DenseMap<Value, Value> valueMap =
      cloneSliceOps(allOpsToClone.getArrayRef(),
                    allCapturedValues.getArrayRef(), builder, explorer);

  // Map each slice's root size value to its hoisted version.
  SmallVector<Value> hoistedSizes;
  for (const auto &slice : slices) {
    LLVM_DEBUG({
      LDBG() << "  root size value: ";
      slice.rootSizeValue.printAsOperand(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << ", valueMap.contains(root)="
                   << valueMap.contains(slice.rootSizeValue) << "\n";
    });
    Value hoistedSize;
    if (valueMap.contains(slice.rootSizeValue)) {
      // Root was cloned.
      hoistedSize = valueMap[slice.rootSizeValue];
      LLVM_DEBUG({
        LDBG() << "  using cloned value: ";
        hoistedSize.printAsOperand(llvm::dbgs(), explorer.getAsmState());
        llvm::dbgs() << "\n";
      });
    } else {
      // Root is a captured value - use it directly.
      hoistedSize = slice.rootSizeValue;
      LLVM_DEBUG({
        llvm::dbgs()
            << "[emplace-transients]   using original (captured) value: ";
        hoistedSize.printAsOperand(llvm::dbgs(), explorer.getAsmState());
        llvm::dbgs() << "\n";
      });
    }
    hoistedSizes.push_back(hoistedSize);
  }

  // Compute conservative max of all hoisted sizes.
  // We need to ensure the maxui is inserted after all values it depends on.
  // Find the latest defining operation among:
  // 1. All hoisted sizes (cloned operations' results)
  // 2. All captured values (used by cloned operations)
  Operation *latestDefiningOp = nullptr;

  // Check hoisted sizes.
  for (Value size : hoistedSizes) {
    if (auto definingOp = size.getDefiningOp()) {
      if (!latestDefiningOp || latestDefiningOp->isBeforeInBlock(definingOp)) {
        latestDefiningOp = definingOp;
      }
    }
  }

  // Check all captured values from all slices.
  for (const auto &slice : slices) {
    for (Value capturedValue : slice.capturedValues) {
      if (auto definingOp = capturedValue.getDefiningOp()) {
        if (!latestDefiningOp ||
            latestDefiningOp->isBeforeInBlock(definingOp)) {
          latestDefiningOp = definingOp;
        }
      }
    }
  }

  // Set insertion point after the latest defining op.
  if (latestDefiningOp) {
    builder.setInsertionPointAfter(latestDefiningOp);
  }

  Value conservativeMax =
      computeConservativeMax(builder, insertionPoint->getLoc(), hoistedSizes);
  LLVM_DEBUG({
    LDBG() << "  conservative max: ";
    conservativeMax.printAsOperand(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << "\n";
  });
  return conservativeMax;
}

// Computes allocation slots for packing. Allocations in mutually exclusive
// control flow branches (e.g., scf.if then vs else) share a slot and can
// reuse memory. Returns one slot per unique liveness interval.
static SmallVector<AllocationSlot>
computeAllocationSlots(ArrayRef<IREE::Stream::ResourceAllocaOp> allocaOps,
                       Explorer &explorer) {
  if (allocaOps.empty()) {
    return {};
  }

  LLVM_DEBUG(LDBG() << "computing allocation slots for " << allocaOps.size()
                    << " allocation(s)\n");

  // Group allocations by their containing region.
  // Use MapVector to ensure deterministic iteration order.
  llvm::MapVector<Region *, SmallVector<IREE::Stream::ResourceAllocaOp>>
      allocationsByRegion;
  for (auto allocaOp : allocaOps) {
    Region *region = allocaOp->getParentRegion();
    allocationsByRegion[region].push_back(allocaOp);
  }

  // Find groups of mutually exclusive regions (sibling regions of the same
  // RegionBranchOpInterface parent, like scf.if then/else branches).
  SmallVector<SmallVector<Region *>> exclusiveRegionGroups;
  DenseSet<Region *> processedRegions;
  DenseSet<Operation *> processedBranchOps;

  for (auto [region, allocations] : allocationsByRegion) {
    if (processedRegions.contains(region)) {
      continue; // Already grouped.
    }

    Operation *parentOp = region->getParentOp();

    // Check if parent is a RegionBranchOpInterface (scf.if, scf.case, etc).
    if (auto branchOp = dyn_cast<RegionBranchOpInterface>(parentOp)) {
      // Avoid processing the same branch op multiple times.
      if (processedBranchOps.contains(parentOp)) {
        continue;
      }
      processedBranchOps.insert(parentOp);

      // Collect all sibling regions under this parent.
      SmallVector<Region *> siblingRegions;
      for (auto &siblingRegion : parentOp->getRegions()) {
        // Only include regions that have allocations.
        if (allocationsByRegion.count(&siblingRegion)) {
          siblingRegions.push_back(&siblingRegion);
        }
      }

      if (siblingRegions.size() > 1) {
        // Multiple sibling regions with allocations - mutually exclusive.
        // Mark all regions in this group as processed before moving.
        for (Region *r : siblingRegions) {
          processedRegions.insert(r);
        }
        LLVM_DEBUG({
          LDBG() << "  found mutually exclusive "
                    "group with "
                 << siblingRegions.size() << " region(s) under: ";
          parentOp->print(llvm::dbgs(), explorer.getAsmState());
          llvm::dbgs() << "\n";
        });
        exclusiveRegionGroups.push_back(std::move(siblingRegions));
      }
      // If only one sibling has allocations, it will be handled in "remaining".
    }
    // Non-RegionBranchOpInterface regions will be handled in "remaining".
  }

  // Assign slot IDs based on mutual exclusivity.
  // Allocations in the same region get different slots (sequential).
  // Allocations in sibling regions at the same "position" share a slot.
  int64_t nextSlotId = 0;
  DenseMap<IREE::Stream::ResourceAllocaOp, int64_t> allocaToSlot;

  // Process exclusive region groups.
  for (auto &exclusiveGroup : exclusiveRegionGroups) {
    // Find maximum number of allocations in any region in this group.
    size_t maxAllocationsInRegion = 0;
    for (Region *region : exclusiveGroup) {
      maxAllocationsInRegion =
          std::max(maxAllocationsInRegion, allocationsByRegion[region].size());
    }

    LLVM_DEBUG({
      LDBG() << "  exclusive group needs " << maxAllocationsInRegion
             << " slot(s)\n";
    });

    // Allocate one slot per "position" across all exclusive regions.
    // Allocations at the same position in different regions share a slot.
    for (size_t position = 0; position < maxAllocationsInRegion; ++position) {
      int64_t slotId = nextSlotId++;

      for (Region *region : exclusiveGroup) {
        auto &allocations = allocationsByRegion[region];
        if (position < allocations.size()) {
          allocaToSlot[allocations[position]] = slotId;
          LLVM_DEBUG({
            LDBG() << "    assigning slot " << slotId
                   << " to alloca at position " << position << " in region: ";
            allocations[position]->print(llvm::dbgs(), explorer.getAsmState());
            llvm::dbgs() << "\n";
          });
        }
      }
    }
  }

  // Process remaining allocations (not in exclusive groups).
  // Each gets its own slot.
  for (auto [region, allocations] : allocationsByRegion) {
    if (processedRegions.contains(region)) {
      continue; // Already handled in exclusive group.
    }

    for (auto allocaOp : allocations) {
      int64_t slotId = nextSlotId++;
      allocaToSlot[allocaOp] = slotId;
      LLVM_DEBUG({
        LDBG() << "  assigning unique slot " << slotId
               << " to non-exclusive alloca: ";
        allocaOp->print(llvm::dbgs(), explorer.getAsmState());
        llvm::dbgs() << "\n";
      });
    }
  }

  // Build AllocationSlot structures.
  SmallVector<AllocationSlot> slots(nextSlotId);
  for (size_t i = 0; i < nextSlotId; ++i) {
    slots[i].slotId = i;
  }

  for (auto allocaOp : allocaOps) {
    auto it = allocaToSlot.find(allocaOp);
    assert(it != allocaToSlot.end() &&
           "all allocations must have been assigned a slot");
    int64_t slotId = it->second;
    assert(slotId >= 0 && slotId < nextSlotId &&
           "slot ID must be in valid range");
    slots[slotId].allocations.push_back(allocaOp);
  }

  LLVM_DEBUG({
    LDBG() << "computed " << slots.size() << " slot(s) for " << allocaOps.size()
           << " allocation(s)\n";
    for (auto &slot : slots) {
      LDBG() << "  slot " << slot.slotId << " contains "
             << slot.allocations.size() << " allocation(s)\n";
    }
  });

  return slots;
}

// Computes the conservative size for a single allocation slot.
// Returns the maximum size across all allocations in the slot.
static FailureOr<Value> computeConservativeSizeForSlot(
    ArrayRef<IREE::Stream::ResourceAllocaOp> allocations,
    Operation *insertionPoint, Explorer &explorer) {
  assert(!allocations.empty() && "slot should have at least one allocation");

  LLVM_DEBUG({
    LDBG() << "computing conservative size for slot "
              "with "
           << allocations.size() << " allocation(s)\n";
  });

  SmallVector<SizeComputationSlice> slices;
  for (auto allocaOp : allocations) {
    SizeComputationSlice slice;
    if (failed(computeSizeSliceInRegion(allocaOp.getStorageSize(),
                                        allocaOp->getParentRegion(), slice,
                                        explorer))) {
      return allocaOp.emitError(
          "cannot hoist transient size computation - contains non-pure "
          "operations or unsupported patterns");
    }
    slices.push_back(std::move(slice));
  }

  return hoistSizesConservatively(slices, insertionPoint, explorer);
}

// Creates a stream.resource.pack operation that computes offsets and total size
// for packing multiple transient allocations into a single slab.
// Returns the pack operation with results: (total_size, offset_0, offset_1,
// ...).
static IREE::Stream::ResourcePackOp
createPackedAllocation(OpBuilder &builder, Location loc,
                       ArrayRef<Value> sliceSizes,
                       IREE::Stream::AffinityAttr affinity) {
  assert(!sliceSizes.empty() && "should have sizes");

  auto indexType = builder.getIndexType();

  // Create lifetime intervals for all slices.
  // We use [0, 0] for all slices to indicate they all overlap temporally and
  // cannot alias (reuse memory). This is a conservative approach - future
  // optimization passes can refine these intervals based on actual lifetime
  // analysis.
  SmallVector<int64_t> lifetimeIntervals;
  for (size_t i = 0; i < sliceSizes.size(); ++i) {
    lifetimeIntervals.push_back(0); // start.
    lifetimeIntervals.push_back(0); // end.
  }

  // Create the pack operation with affinity from the transients op.
  SmallVector<Type> packedOffsetTypes(sliceSizes.size(), indexType);
  auto packOp = IREE::Stream::ResourcePackOp::create(
      builder, loc, indexType, packedOffsetTypes,
      /*offset=*/nullptr, builder.getIndexArrayAttr(lifetimeIntervals),
      sliceSizes, affinity);

  // Add the stream.experimental.transients attribute to mark this pack as
  // being used for external transient allocation. This is only used to aid
  // the MaterializeTransientSizeQueriesPass in finding these.
  packOp->setAttr("stream.experimental.transients", builder.getUnitAttr());

  return packOp;
}

// Replaces alloca ops with subviews of the backing resource at pack offsets.
// Maintains the timeline chain by forwarding each alloca's await timepoint to
// replace its result timepoint, preserving causality.
static void replaceAllocasWithSubviews(
    OpBuilder &builder, ArrayRef<IREE::Stream::ResourceAllocaOp> allocaOps,
    Value backingResource, Value packedTotalSize,
    const DenseMap<IREE::Stream::ResourceAllocaOp, Value> &allocaToOffset,
    const DenseMap<IREE::Stream::ResourceAllocaOp, Value> &allocaToSize,
    Explorer &explorer) {
  for (auto allocaOp : allocaOps) {
    auto offset = allocaToOffset.lookup(allocaOp);
    auto size = allocaToSize.lookup(allocaOp);
    assert(offset && size &&
           "all allocations must have offset and size mappings");

    // Set insertion point before the alloca so the subview replaces it.
    builder.setInsertionPoint(allocaOp);

    // Create subview at pack offset.
    auto subviewOp = IREE::Stream::ResourceSubviewOp::create(
        builder, allocaOp.getLoc(), backingResource, packedTotalSize, offset,
        size);
    LLVM_DEBUG({
      LDBG() << "replacing alloca with subview: ";
      subviewOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });

    // Replace alloca resource result with subview.
    allocaOp.getResult().replaceAllUsesWith(subviewOp.getResult());

    // Preserve timeline chain: replace alloca's result_timepoint with its
    // await_timepoint. This maintains causality - operations that awaited the
    // alloca now await whatever the alloca was waiting on.
    if (auto awaitTimepoint = allocaOp.getAwaitTimepoint()) {
      allocaOp.getResultTimepoint().replaceAllUsesWith(awaitTimepoint);
    } else {
      // If no await timepoint, create an immediate timepoint.
      // This case should be rare as allocas usually have await timepoints.
      auto immediateTp = IREE::Stream::TimepointImmediateOp::create(
          builder, allocaOp.getLoc());
      allocaOp.getResultTimepoint().replaceAllUsesWith(immediateTp.getResult());
    }

    // Erase the alloca.
    allocaOp.erase();
  }
}

// Hoists size computation values and their dependencies to the insertion point.
// Validates that hoisted operations are pure (no side effects, no calls).
// The dominantAllocaOp parameter specifies which alloca to use as the end point
// for validation - this should be the alloca that dominates all others.
// If skipValidation is true, validation is skipped (used for multiple
// branches).
static LogicalResult
hoistSizeComputations(FunctionOpInterface funcOp,
                      ArrayRef<IREE::Stream::ResourceAllocaOp> allocaOps,
                      Operation *insertionPoint,
                      IREE::Stream::ResourceAllocaOp dominantAllocaOp,
                      bool skipValidation, SmallVectorImpl<Value> &hoistedSizes,
                      Explorer &explorer) {
  if (allocaOps.empty()) {
    return success();
  }

  LLVM_DEBUG({
    LDBG() << "hoisting size computations to: ";
    insertionPoint->print(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << "\n";
  });

  // Collect all size SSA values from allocas.
  SmallVector<Value> allSizes;
  SmallVector<Operation *> sizeOps;
  for (auto allocaOp : allocaOps) {
    Value size = allocaOp.getStorageSize();
    allSizes.push_back(size);
    LLVM_DEBUG({
      LDBG() << "  size value: ";
      size.printAsOperand(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << " from alloca: ";
      allocaOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });

    // Track defining ops for hoisting (skip block args - they don't need
    // hoisting).
    if (auto definingOp = size.getDefiningOp()) {
      sizeOps.push_back(definingOp);
    }
  }

  // If all sizes are block arguments (e.g., function args), no hoisting needed.
  if (sizeOps.empty()) {
    LLVM_DEBUG(LDBG() << "all sizes are block args, "
                         "no hoisting needed\n");
    hoistedSizes = std::move(allSizes);
    return success();
  }

  // Hoist size computations and dependencies up to the dominating block.
  IRRewriter rewriter(insertionPoint->getContext());
  DominanceInfo domInfo(funcOp);
  LLVM_DEBUG(LDBG() << "calling moveOperandDefs with " << sizeOps.size()
                    << " size op(s)\n");
  if (failed(moveOperandDefs(rewriter, sizeOps, insertionPoint, domInfo))) {
    return funcOp.emitError(
        "failed to hoist transient size computations - circular dependency or "
        "dominance issue detected");
  }
  LLVM_DEBUG(
      llvm::dbgs()
      << "[emplace-transients] successfully hoisted size computations\n");

  // Validate hoisted operations are pure (no side effects, no calls).
  // Skip validation if allocas are in mutually exclusive branches.
  if (!skipValidation) {
    // Walk from insertion point to dominant alloca and check all ops in
    // between. If the insertion point is the dominant alloca, no ops were
    // hoisted, so skip validation.
    Operation *dominantAllocaOpPtr = dominantAllocaOp.getOperation();

    if (insertionPoint == dominantAllocaOpPtr) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[emplace-transients] insertion point is first "
                    "alloca, no validation needed\n");
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "[emplace-transients] validating hoisted ops from "
                    "insertion point to dominant alloca\n");
      Operation *currentOp = insertionPoint->getNextNode();
      while (currentOp && currentOp != dominantAllocaOpPtr) {
        LLVM_DEBUG({
          LDBG() << "  checking op: ";
          currentOp->print(llvm::dbgs(), explorer.getAsmState());
          llvm::dbgs() << "\n";
        });

        if (!isOperationPureRecursively(currentOp)) {
          return currentOp->emitError(
              "transient size computation contains non-hoistable operations "
              "(memory effects or calls not supported)");
        }

        currentOp = currentOp->getNextNode();
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[emplace-transients] validation passed for hoisted ops\n");
    }
  } else {
    LLVM_DEBUG(LDBG() << "skipping validation "
                         "(allocas in mutually exclusive branches)\n");
  }

  hoistedSizes = std::move(allSizes);
  return success();
}

// Groups allocations into slots and computes conservative maximum sizes.
// Uses backward slicing to extract size computations from control flow regions
// and clones them before the insertion point.
//
// For example, with allocations in scf.if branches:
//   scf.if %cond {
//     %size_a = arith.muli %x, %c4
//     alloca %size_a
//   } else {
//     %size_b = arith.addi %y, %c256
//     alloca %size_b
//   }
//
// This will:
// 1. Compute backward slices for size_a and size_b
// 2. Clone the arithmetic operations before the scf.if
// 3. Compute: %max = arith.maxui %size_a_cloned, %size_b_cloned
// 4. Use %max for all allocations
//
// Returns LogicalResult and outputs hoisted conservative sizes per allocation.
static LogicalResult computeConservativeSizesForSlots(
    FunctionOpInterface funcOp,
    ArrayRef<IREE::Stream::ResourceAllocaOp> allocaOps,
    Operation *insertionPoint, SmallVectorImpl<Value> &conservativeSizes,
    Explorer &explorer) {
  if (allocaOps.empty()) {
    return success();
  }

  LLVM_DEBUG(LDBG() << "computing conservative "
                       "sizes for "
                    << allocaOps.size() << " allocation(s)\n");

  // Check if all allocations are in the same block.
  // If so, we can use the simple path without slicing.
  bool allInSameBlock = true;
  Block *firstBlock = allocaOps.front()->getBlock();
  for (auto allocaOp : allocaOps) {
    if (allocaOp->getBlock() != firstBlock) {
      allInSameBlock = false;
      break;
    }
  }

  if (allInSameBlock) {
    LLVM_DEBUG(llvm::dbgs()
               << "[emplace-transients] all allocations in same block, using "
                  "simple size hoisting\n");
    // Simple case: just collect sizes and hoist them using moveOperandDefs.
    SmallVector<Value> sizes;
    SmallVector<Operation *> sizeOps;
    for (auto allocaOp : allocaOps) {
      Value size = allocaOp.getStorageSize();
      sizes.push_back(size);
      if (auto definingOp = size.getDefiningOp()) {
        sizeOps.push_back(definingOp);
      }
    }

    // Hoist size computations if needed.
    if (!sizeOps.empty()) {
      IRRewriter rewriter(funcOp.getContext());
      DominanceInfo domInfo(funcOp);
      rewriter.setInsertionPoint(insertionPoint);
      if (failed(moveOperandDefs(rewriter, sizeOps, insertionPoint, domInfo))) {
        return funcOp.emitError(
            "failed to hoist transient size computations - circular "
            "dependency or dominance issue detected");
      }
    }

    conservativeSizes = std::move(sizes);
    return success();
  }

  // Complex case: allocations are in different blocks (potentially mutually
  // exclusive branches). Use backward slicing and conservative hoisting.
  LLVM_DEBUG(llvm::dbgs()
             << "[emplace-transients] allocations in different blocks, using "
                "slice-based conservative hoisting\n");

  // Compute backward slices for each allocation's size.
  SmallVector<SizeComputationSlice> slices;
  for (auto allocaOp : allocaOps) {
    SizeComputationSlice slice;
    Region *allocaRegion = allocaOp->getParentRegion();
    Value sizeValue = allocaOp.getStorageSize();

    LLVM_DEBUG({
      LDBG() << "processing alloca: ";
      allocaOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
      LDBG() << "  alloca parentRegion: " << allocaRegion << "\n";
      LDBG() << "  size value: ";
      sizeValue.printAsOperand(llvm::dbgs(), explorer.getAsmState());
      if (auto sizeDefOp = sizeValue.getDefiningOp()) {
        llvm::dbgs() << " defined in region: " << sizeDefOp->getParentRegion();
      }
      llvm::dbgs() << "\n";
    });

    if (failed(computeSizeSliceInRegion(sizeValue, allocaRegion, slice,
                                        explorer))) {
      return allocaOp.emitError(
          "cannot hoist transient size computation - contains non-pure "
          "operations or unsupported patterns");
    }

    slices.push_back(std::move(slice));
  }

  // Use Tier 2: generic conservative hoisting with cloning and max.
  auto result = hoistSizesConservatively(slices, insertionPoint, explorer);
  if (failed(result)) {
    return funcOp.emitError("failed to hoist size computations conservatively");
  }

  // For Phase 1: use the conservative max for ALL allocations.
  // This is overly conservative but correct.
  // Future: detect slots and compute per-slot maxes for better memory usage.
  Value conservativeMax = *result;
  conservativeSizes.assign(allocaOps.size(), conservativeMax);

  LLVM_DEBUG({
    LDBG() << "using conservative max for all " << allocaOps.size()
           << " allocation(s): ";
    conservativeMax.printAsOperand(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << "\n";
  });

  return success();
}

// Processes a single transient resource by hoisting size computations and
// creating packed allocations.
static LogicalResult
processTransientResource(FunctionOpInterface funcOp,
                         TransientResource &transientResource,
                         Explorer &explorer) {
  if (transientResource.allocaOps.empty()) {
    // No allocations - create a trivial pack op with zero slices so that the
    // MaterializeTransientSizeQueriesPass can still generate a size query
    // function (which will return 0). This ensures consistent ABI behavior.
    LLVM_DEBUG(LDBG() << "no allocations found, "
                         "creating trivial zero-size pack\n");

    // Find the first transients op to use as insertion point.
    auto firstTransientsOp = transientResource.transientsOps.front();
    OpBuilder builder(firstTransientsOp);

    // Create a pack op with zero slices - it will fold to total_length = 0.
    auto indexType = builder.getIndexType();
    SmallVector<int64_t> lifetimeIntervals; // Empty for zero slices.
    SmallVector<Value> sliceSizes;          // Empty for zero slices.
    SmallVector<Type> packedOffsetTypes;    // Empty for zero slices.
    auto packOp = IREE::Stream::ResourcePackOp::create(
        builder, firstTransientsOp.getLoc(), indexType, packedOffsetTypes,
        /*offset=*/nullptr, builder.getIndexArrayAttr(lifetimeIntervals),
        sliceSizes, transientResource.affinity);

    // Add the stream.experimental.transients attribute so
    // MaterializeTransientSizeQueriesPass can find this pack op and generate
    // a size query function.
    packOp->setAttr("stream.experimental.transients", builder.getUnitAttr());

    LLVM_DEBUG({
      LDBG() << "created trivial pack op: ";
      packOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });

    // Remove the transients ops by forwarding their values.
    for (auto transientsOp : transientResource.transientsOps) {
      // Forward the result timepoint to the await timepoint.
      transientsOp.getResultTimepoint().replaceAllUsesWith(
          transientsOp.getAwaitTimepoint());
      // Forward the result resource to the input resource operand.
      transientsOp.getResult().replaceAllUsesWith(transientsOp.getResource());
      transientsOp.erase();
    }
    return success();
  }

  // Find the tightest insertion point that dominates all allocas.
  // This handles single allocas, sequential allocas, and allocas in different
  // branches (e.g., scf.if).
  DominanceInfo domInfo(funcOp);
  Operation *insertionPointOp = findDominantInsertionPoint(
      domInfo, transientResource.allocaOps, funcOp, explorer);

  if (!insertionPointOp) {
    return failure(); // Error already emitted.
  }

  // Compute allocation slots. Allocations in mutually exclusive control flow
  // branches share a slot and can reuse memory.
  SmallVector<AllocationSlot> slots =
      computeAllocationSlots(transientResource.allocaOps, explorer);
  assert(!slots.empty() && "should have at least one slot");

  // Compute conservative size for each slot.
  for (auto &slot : slots) {
    auto sizeResult = computeConservativeSizeForSlot(
        slot.allocations, insertionPointOp, explorer);
    if (failed(sizeResult)) {
      return failure();
    }
    slot.conservativeSize = *sizeResult;
  }

  // Build slot sizes array for pack operation.
  SmallVector<Value> slotSizes;
  for (auto &slot : slots) {
    slotSizes.push_back(slot.conservativeSize);
  }

  LLVM_DEBUG({
    LDBG() << "computed " << slots.size()
           << " slot(s) with conservative sizes:\n";
    for (auto &slot : slots) {
      LDBG() << "  slot " << slot.slotId << ": size ";
      slot.conservativeSize.printAsOperand(llvm::dbgs(),
                                           explorer.getAsmState());
      llvm::dbgs() << " (covers " << slot.allocations.size()
                   << " allocation(s))\n";
    }
  });

  // Create a stream.resource.pack with one slice per slot.
  // The pack operation computes offsets and total size for packing all
  // transient allocations into a single slab.
  // We must hoist size computations AND storage resource to before all allocas.
  IRRewriter rewriter(funcOp.getContext());
  rewriter.setInsertionPoint(insertionPointOp);

  // Collect all operations that need to be hoisted: slot sizes AND storage.
  // IMPORTANT: Add storage size BEFORE storage resource, as the resource
  // typically depends on the size (e.g., stream.tensor.import uses
  // hal.buffer.length result).
  SmallVector<Operation *> opsToHoist;
  for (auto &slot : slots) {
    if (auto definingOp = slot.conservativeSize.getDefiningOp()) {
      opsToHoist.push_back(definingOp);
    }
  }

  // Hoist storage size first (dependencies of storage resource).
  if (auto storageSizeDefOp = transientResource.storageSize.getDefiningOp()) {
    opsToHoist.push_back(storageSizeDefOp);
    LLVM_DEBUG({
      LDBG() << "  will hoist storage size: ";
      storageSizeDefOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });
  }

  // Then hoist storage resource (depends on storage size).
  if (auto storageDefOp = transientResource.storage.getDefiningOp()) {
    opsToHoist.push_back(storageDefOp);
    LLVM_DEBUG({
      LDBG() << "  will hoist storage resource: ";
      storageDefOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });
  }

  // Hoist all dependencies to the insertion point.
  if (!opsToHoist.empty()) {
    LLVM_DEBUG({
      LDBG() << "hoisting " << opsToHoist.size()
             << " operation(s) to insertion point: ";
      insertionPointOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
      for (size_t i = 0; i < opsToHoist.size(); ++i) {
        LDBG() << "  op[" << i << "] to hoist: ";
        opsToHoist[i]->print(llvm::dbgs(), explorer.getAsmState());
        llvm::dbgs() << "\n";
      }
    });
    if (failed(
            moveOperandDefs(rewriter, opsToHoist, insertionPointOp, domInfo))) {
      return funcOp.emitError(
          "failed to hoist transient size computations and storage resource - "
          "values may depend on operations defined after allocations");
    }
    LLVM_DEBUG(LDBG() << "hoisted operand definitions successfully\n");

    // moveOperandDefs only moves the DEPENDENCIES, not the operations
    // themselves. We need to explicitly move the operations now that their
    // dependencies are in place.
    for (Operation *op : opsToHoist) {
      LLVM_DEBUG({
        LDBG() << "  moving operation to insertion point: ";
        op->print(llvm::dbgs(), explorer.getAsmState());
        llvm::dbgs() << "\n";
      });
      op->moveBefore(insertionPointOp);
    }
    LLVM_DEBUG(LDBG() << "hoisting completed successfully\n");
  }

  auto packOp = createPackedAllocation(rewriter, insertionPointOp->getLoc(),
                                       slotSizes, transientResource.affinity);
  if (!packOp) {
    return funcOp.emitError(
        "failed to create pack operation for transient allocations");
  }
  LLVM_DEBUG({
    LDBG() << "created pack operation: ";
    packOp->print(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << "\n";
    LDBG() << "total packed size: ";
    packOp.getTotalLength().printAsOperand(llvm::dbgs(),
                                           explorer.getAsmState());
    llvm::dbgs() << "\n";
  });

  // Use the pack operation as stable insertion point (insertion point may have
  // been invalidated by hoisting).
  rewriter.setInsertionPointAfter(packOp);

  // Use external storage directly (already imported during HAL→Stream
  // conversion and hoisted above). After hoisting, we use the values directly
  // from transientResource - they now refer to the moved operations.
  Value storageResource = transientResource.storage;
  Value storageSize = transientResource.storageSize;
  Value packTotalSize = packOp.getTotalLength();
  LLVM_DEBUG({
    LDBG() << "using external storage: ";
    storageResource.printAsOperand(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << " with size: ";
    storageSize.printAsOperand(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << "\n";
    LDBG() << "pack total size: ";
    packTotalSize.printAsOperand(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << "\n";
  });

  // TODO(benvanik): add runtime validation that packed size fits in storage
  // size. This would require inserting a comparison op and conditional error
  // handling or a hal.buffer.assert, which we don't really have here. For now
  // we just trust the user and rely on runtime handling.

  // Create a subview of the user storage for the packed range.
  // This subview represents the portion of external storage used for
  // transients.
  Value zero = arith::ConstantIndexOp::create(rewriter, packOp.getLoc(), 0);
  auto storageSubviewOp = IREE::Stream::ResourceSubviewOp::create(
      rewriter, packOp.getLoc(), storageResource, storageSize, zero,
      packTotalSize);
  Value backingResource = storageSubviewOp.getResult();
  LLVM_DEBUG({
    LDBG() << "created storage subview: ";
    storageSubviewOp->print(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << "\n";
  });

  // Build mapping from allocations to their slot offsets and sizes.
  DenseMap<IREE::Stream::ResourceAllocaOp, Value> allocaToOffset;
  DenseMap<IREE::Stream::ResourceAllocaOp, Value> allocaToSize;
  SmallVector<Value> packedOffsets(packOp.getPackedOffsets());
  for (auto &slot : slots) {
    Value slotOffset = packedOffsets[slot.slotId];
    for (auto allocaOp : slot.allocations) {
      allocaToOffset[allocaOp] = slotOffset;
      // Each alloca uses its ORIGINAL size for the subview result type.
      // The conservative size is only for pack layout computation.
      allocaToSize[allocaOp] = allocaOp.getStorageSize();
    }
  }

  // Replace each alloca with a subview at its slot's offset.
  // All subviews are created from the storage subview (backingResource).
  replaceAllocasWithSubviews(rewriter, transientResource.allocaOps,
                             backingResource, packTotalSize, allocaToOffset,
                             allocaToSize, explorer);
  LLVM_DEBUG(LDBG() << "replaced " << transientResource.allocaOps.size()
                    << " alloca(s) with subviews\n");

  // Remove all dealloca ops (memory is externally managed).
  for (auto deallocaOp : transientResource.deallocaOps) {
    // Replace dealloca timepoint result with its await timepoint.
    deallocaOp.getResultTimepoint().replaceAllUsesWith(
        deallocaOp.getAwaitTimepoint());
    LLVM_DEBUG({
      LDBG() << "removing dealloca: ";
      deallocaOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });
    deallocaOp.erase();
  }
  LLVM_DEBUG(LDBG() << "removed " << transientResource.deallocaOps.size()
                    << " dealloca op(s)\n");

  // Remove all transients annotation ops.
  // These ops were just markers - now that we've emplaced the transients into
  // external storage, we can remove the annotations and forward their values.
  for (auto transientsOp : transientResource.transientsOps) {
    // Forward the result timepoint to the await timepoint.
    // This removes the annotation while preserving the timeline chain.
    if (auto awaitTimepoint = transientsOp.getAwaitTimepoint()) {
      transientsOp.getResultTimepoint().replaceAllUsesWith(awaitTimepoint);
    } else {
      // If no await timepoint, create an immediate timepoint.
      auto immediateTp = IREE::Stream::TimepointImmediateOp::create(
          rewriter, transientsOp.getLoc());
      transientsOp.getResultTimepoint().replaceAllUsesWith(
          immediateTp.getResult());
    }

    // Forward the result resource to the input resource operand.
    // The transients op was just passing through the resource value.
    transientsOp.getResult().replaceAllUsesWith(transientsOp.getResource());

    LLVM_DEBUG({
      LDBG() << "removing transients op: ";
      transientsOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });
    transientsOp.erase();
  }
  LLVM_DEBUG(LDBG() << "removed " << transientResource.transientsOps.size()
                    << " transients op(s)\n");

  return success();
}

static LogicalResult emplaceTransientsInFuncOp(FunctionOpInterface funcOp,
                                               Explorer &explorer) {
  LLVM_DEBUG({
    LDBG() << "processing function: ";
    funcOp->print(llvm::dbgs(), explorer.getAsmState());
    llvm::dbgs() << "\n";
  });

  // Validate that this is a public function.
  bool hasTransients = false;
  funcOp.walk([&](IREE::Stream::ResourceTransientsOp op) {
    hasTransients = true;
    return WalkResult::interrupt();
  });
  if (!hasTransients) {
    return success();
  }
  if (!funcOp.isPublic()) {
    return funcOp.emitError(
        "only public functions with stream.resource.transients are supported");
  }

  // Validate no function calls (not supported yet).
  bool hasCallOp = false;
  funcOp.walk([&](CallOpInterface callOp) {
    LLVM_DEBUG({
      LDBG() << "found unsupported call op: ";
      callOp->print(llvm::dbgs(), explorer.getAsmState());
      llvm::dbgs() << "\n";
    });
    hasCallOp = true;
    return WalkResult::interrupt();
  });
  if (hasCallOp) {
    return funcOp.emitError("function contains function calls, which are not "
                            "supported by EmplaceTransientsPass");
  }

  // Gather all transient resources by walking the timeline and early exit if no
  // transients found.
  SmallVector<TransientResource> transientResources;
  if (failed(gatherTransientResources(funcOp, explorer, transientResources))) {
    return failure();
  }
  if (transientResources.empty()) {
    return success();
  }

  // Process each transient resource independently.
  for (TransientResource &transientResource : transientResources) {
    if (failed(processTransientResource(funcOp, transientResource, explorer))) {
      return failure();
    }
  }

  return success();
}

struct EmplaceTransientsPass
    : public IREE::Stream::impl::EmplaceTransientsPassBase<
          EmplaceTransientsPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) {
      return;
    }

    Explorer explorer(moduleOp, TraversalAction::RECURSE);
    explorer.initialize();

    // NOTE: this early experimental pass is function local - when this is real
    // it will be running on the entire module to track across call edges.
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      if (failed(emplaceTransientsInFuncOp(funcOp, explorer))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
