// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-stream-automatic-reference-counting"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_AUTOMATICREFERENCECOUNTINGPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Local analysis
//===----------------------------------------------------------------------===//

// Analysis for timepoint coverage within the current recursive analysis scope
// (e.g., a function or nested region).
struct ScopedTimepointCoverage {
  // Must be provided if LLVM_DEBUG is enabled.
  AsmState *asmState;
  // A map of timepoint SSA values to indices within the coverage map.
  // This map accumulates timepoints from the current analysis scope, including
  // those from parent blocks/regions when the object is passed down.
  // Order is that of the appearance in the block but is not guaranteed and the
  // value should only be used for indexing into the map.
  DenseMap<Value, unsigned> timepoints;

  ScopedTimepointCoverage() = delete;
  ScopedTimepointCoverage(AsmState *asmState) : asmState(asmState) {}

  // A matrix of timepoints by index to bits indicating whether another
  // timepoint (column) covers the entry (row).
  //
  // Example:
  //
  // t0 -+--> t1 -+-> t3
  //      \-> t2 -/
  //            \---> t4
  //
  //     t0 t1 t2 t3 t4
  // t0  x
  // t1  x  x
  // t2  x     x
  // t3  x  x  x  x
  // t4  x     x     x
  //
  // "t0 covers t0" (identity)
  // "t1 covers t0" (covers(t0, t1) = true)
  // "t2 covers t0" (covers(t0, t2) = true)
  // "t3 covers t0, t1, t2"
  // "t4 covers t0, t1"
  SmallVector<llvm::BitVector> map;

  void reset() {
    timepoints.clear();
    map.clear();
  }

  // Adds an immediate |timepoint| that is always reached once defined.
  void add(Value timepoint) {
    getOrInsertIndex(timepoint);
    LLVM_DEBUG({
      llvm::dbgs() << "[arc] adding immediate timepoint ";
      timepoint.printAsOperand(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    });
  }

  // Adds coverage of |timepoint| by |predecessor|, indicating that if
  // |timepoint| is reached |predecessor| must have already been.
  void add(Value predecessor, Value timepoint) {
    auto predecessorIt = timepoints.find(predecessor);
    if (predecessorIt == timepoints.end()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[arc] coverage insertion for predecessor ";
        predecessor.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " over ";
        timepoint.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " failed: predecessor not local\n";
      });
      return;
    }
    const unsigned predecessorIndex = predecessorIt->second;
    const unsigned timepointIndex = getOrInsertIndex(timepoint);
    LLVM_DEBUG({
      llvm::dbgs() << "[arc] adding coverage of predecessor ";
      predecessor.printAsOperand(llvm::dbgs(), *asmState);
      llvm::dbgs() << " over ";
      timepoint.printAsOperand(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    });
    auto &timepointSet = map[timepointIndex];
    if (timepointSet.size() < predecessorIndex) {
      timepointSet.resize(predecessorIndex + 1);
    }
    timepointSet.set(predecessorIndex);

    // OR in the predecessor coverage bits.
    // NOTE: if this were a non-local analysis we would have to do this in two
    // steps (at least), as we may not know everything a predecessor covers on
    // the first iteration (since it may be covered across loop back edges, etc)
    // but in local analysis this is fine.
    auto &predecessorSet = map[predecessorIndex];
    timepointSet |= predecessorSet;
  }

  // Returns true if |predecessor| must have been reached by the time
  // |timepoint| is executed.
  bool covers(Value predecessor, Value timepoint) {
    auto predecessorIt = timepoints.find(predecessor);
    auto timepointIt = timepoints.find(timepoint);
    if (predecessorIt == timepoints.end()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[arc] coverage check for predecessor ";
        predecessor.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " over ";
        timepoint.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " failed: predecessor not local\n";
      });
      return false;
    } else if (timepointIt == timepoints.end()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[arc] coverage check for predecessor ";
        predecessor.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " over ";
        timepoint.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " failed: timepoint not local\n";
      });
      return false;
    }
    return map[timepointIt->second].test(predecessorIt->second);
  }

  // Returns the unique assigned index for |timepoint|, assigning one if needed.
  unsigned getOrInsertIndex(Value timepoint) {
    auto it = timepoints.find(timepoint);
    if (it != timepoints.end()) {
      return it->second;
    }
    unsigned index = timepoints.size();
    timepoints.insert({timepoint, index});
    map.resize(index + 1);
    auto &set = map[index];
    set.resize(index + 1);
    set.set(index); // identity
    return index;
  }
};

using TimepointSet = SmallPtrSet<Value, 4>;

// Manages a set of per-resource timeline uses.
// These are not the same as the last users of any particular SSA value as
// resources may be aliased and the timeline may be constructed out of order.
// In cases of forks in the timeline there may be multiple last uses.
struct LastUseSet {
  // Must be provided if LLVM_DEBUG is enabled.
  AsmState *asmState;
  // Timepoint coverage map.
  ScopedTimepointCoverage &coverage;
  // Resource base value to a set of signal timepoints from users.
  // Each set of timepoints is maintained as only those not covered by
  // others. Multiple values indicates a fork.
  DenseMap<Value, TimepointSet> map;
  // Map of resource values back to the root resource value they alias.
  // Each new alias added is always to the first root identified.
  DenseMap<Value, Value> aliases;
  // All base resources in the order they occur in the block.
  SmallVector<Value> baseResourceOrder;

  LastUseSet() = delete;
  LastUseSet(AsmState *asmState, ScopedTimepointCoverage &coverage)
      : asmState(asmState), coverage(coverage) {}

  // Calls |fn| for each base resource produced within the analysis scope with
  // the set of timepoints indicating when the last uses have completed. After
  // all timepoints in the set have been reached the resource may be
  // deallocated.
  void forEachResource(
      std::function<void(Value resource, TimepointSet &timepoints)> fn) {
    for (auto baseResource : llvm::reverse(baseResourceOrder)) {
      fn(baseResource, map[baseResource]);
    }
  }

  // Inserts a produced |resource| into the timepoint set ready when
  // |fromTimepoint| is reached. A new entry will be created for the
  // resource in the map.
  void produce(Value resource, Value fromTimepoint) {
    assert(!aliases.contains(resource) && "produced values must be unique");
    assert(!map.contains(resource) && "produced values must be unique");
    assert(fromTimepoint);
    map[resource].insert(fromTimepoint);
    baseResourceOrder.push_back(resource);
  }

  // Inserts a consumed |resource| that is used until |untilTimepoint|.
  // Any existing timepoints that are covered by the new timepoint will be
  // removed from the set.
  void consume(Value resource, Value untilTimepoint) {
    assert(untilTimepoint);

    // Dereference potentially aliased resource (we only maintain sets for
    // the base resources).
    Value baseResource = lookupResource(resource);

    LLVM_DEBUG({
      llvm::dbgs() << "[arc] marking consumption of ";
      baseResource.printAsOperand(llvm::dbgs(), *asmState);
      llvm::dbgs() << " until ";
      untilTimepoint.printAsOperand(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    });

    // Remove any predecessor timepoints to the last timeline use
    // provided based on the coverage map.
    auto &set = map[baseResource];
    set.remove_if([&](Value existingTimepoint) {
      const bool covers = coverage.covers(existingTimepoint, untilTimepoint);
      LLVM_DEBUG({
        if (covers) {
          llvm::dbgs() << "[arc] removing existing timepoint ";
          existingTimepoint.printAsOperand(llvm::dbgs(), *asmState);
          llvm::dbgs() << " as it covers ";
          untilTimepoint.printAsOperand(llvm::dbgs(), *asmState);
          llvm::dbgs() << "\n";
        }
      });
      return covers;
    });

    // Insert the new last use timepoint.
    set.insert(untilTimepoint);
  }

  // Inserts a |targetResource| tied to a |sourceResource|. Both resources
  // are considered equivalent and added to an alias map. This is
  // conceptually equivalent to:
  //   consume(sourceResource, timepoint);
  //   produce(targetResource, timepoint);
  void tie(Value sourceResource, Value targetResource, Value timepoint) {
    assert(timepoint);

    // Add an alias from source -> target.
    // Future references to the target should always return the base
    // (which may not be source if source itself is an alias).
    aliasResource(sourceResource, targetResource);

    // Mark the source (or base, if aliased) as in use until the timepoint
    // has been reached.
    consume(sourceResource, timepoint);
  }

  // Inserts an alias of |targetResource| to |sourceResource|.
  // If |sourceResource| is itself an alias the base resource will be
  // referenced instead.
  void aliasResource(Value sourceResource, Value targetResource) {
    Value baseResource = lookupResource(sourceResource);
    LLVM_DEBUG({
      llvm::dbgs() << "[arc] aliasing ";
      targetResource.printAsOperand(llvm::dbgs(), *asmState);
      llvm::dbgs() << " to base resource ";
      baseResource.printAsOperand(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    });
    aliases.insert({targetResource, baseResource});
  }

  // Returns the base resource for |resource| if it has been aliased or
  // |resource| itself.
  Value lookupResource(Value resource) {
    auto aliasIt = aliases.find(resource);
    if (aliasIt != aliases.end()) {
      return aliasIt->second;
    }
    Value baseResource =
        IREE::Util::TiedOpInterface::findTiedBaseValue(resource);
    aliases.insert({resource, baseResource});
    return baseResource;
  }
};

// Returns the last defined SSA value in the block in |timepoints| (textual
// order within the block). All timepoints must be in the same block.
static Value getLastTimepointInBlock(TimepointSet &timepoints) {
  if (timepoints.empty()) {
    return nullptr;
  } else if (timepoints.size() == 1) {
    return *timepoints.begin();
  }
  Value lastTimepoint;
  for (auto timepoint : timepoints) {
    if (!lastTimepoint) {
      lastTimepoint = timepoint;
    } else {
      auto *timepointOp = timepoint.getDefiningOp();
      auto *lastTimepointOp = lastTimepoint.getDefiningOp();
      if (!timepointOp) {
        continue; // block arg
      } else if (!lastTimepointOp) {
        lastTimepoint = timepoint; // last found was a block arg, this isn't
      } else if (lastTimepointOp->isBeforeInBlock(timepointOp)) {
        lastTimepoint = timepoint;
      }
    }
  }
  return lastTimepoint;
}

// Returns a FusedLoc with the location of all |timepoints| and the base |loc|.
static Location getFusedLocFromTimepoints(Location loc,
                                          TimepointSet &timepoints) {
  auto locs = llvm::map_to_vector(
      timepoints, [](Value timepoint) { return timepoint.getLoc(); });
  locs.push_back(loc);
  return FusedLoc::get(loc.getContext(), locs);
}

//===----------------------------------------------------------------------===//
// --iree-stream-automatic-reference-counting
//===----------------------------------------------------------------------===//

static StringRef getFuncName(FunctionOpInterface funcOp) {
  if (isa<IREE::Util::InitializerOp>(funcOp)) {
    return "(initializer)";
  } else {
    return funcOp.getName();
  }
}

// Conservatively marks all resources touched by an operation and its nested
// regions as indeterminate. Used as fallback for control flow we cannot
// analyze precisely.
static void markAllResourcesIndeterminateInOpAndRegions(
    Operation &op, LastUseSet &lastUseSet,
    DenseSet<Value> &indeterminateResources) {
  // Mark all currently tracked resources.
  lastUseSet.forEachResource([&](Value resource, TimepointSet &timepoints) {
    indeterminateResources.insert(resource);
  });

  // Walk nested operations to find all resources used within regions.
  op.walk([&](Operation *nestedOp) {
    for (auto operand : nestedOp->getOperands()) {
      if (isa<IREE::Stream::ResourceType>(operand.getType())) {
        Value baseResource = lastUseSet.lookupResource(operand);
        indeterminateResources.insert(baseResource);
      }
    }
    for (auto result : nestedOp->getResults()) {
      if (isa<IREE::Stream::ResourceType>(result.getType())) {
        Value baseResource = lastUseSet.lookupResource(result);
        indeterminateResources.insert(baseResource);
      }
    }
  });
}

// Forward declarations for mutual recursion.
static bool analyzeRegionBranchOp(RegionBranchOpInterface regionBranchOp,
                                  AsmState *asmState, LastUseSet &lastUseSet,
                                  ScopedTimepointCoverage &coverage,
                                  DenseSet<Value> &indeterminateResources,
                                  DenseSet<Value> &handledResources);

// Analyzes operations in a block, handling both timeline ops and nested
// control flow recursively.
static bool analyzeBlockOps(Block &block, AsmState *asmState,
                            LastUseSet &lastUseSet,
                            ScopedTimepointCoverage &coverage,
                            DenseSet<Value> &indeterminateResources,
                            DenseSet<Value> &handledResources) {
  for (Operation &op : block) {
    // Special case ops that are not timeline-aware but interoperate.
    if (auto immediateOp = dyn_cast<IREE::Stream::TimepointImmediateOp>(op)) {
      coverage.add(immediateOp.getResultTimepoint());
      continue;
    } else if (auto importOp = dyn_cast<IREE::Stream::TimepointImportOp>(op)) {
      coverage.add(importOp.getResultTimepoint());
      continue;
    }

    // Handle resource lifetime management ops.
    if (auto retainOp = dyn_cast<IREE::Stream::ResourceRetainOp>(op)) {
      handledResources.insert(lastUseSet.lookupResource(retainOp.getOperand()));
      continue;
    } else if (auto releaseOp = dyn_cast<IREE::Stream::ResourceReleaseOp>(op)) {
      handledResources.insert(
          lastUseSet.lookupResource(releaseOp.getOperand()));
      continue;
    }

    // Timeline ops are handled via the standard analysis.
    // IMPORTANT: Check TimelineOpInterface BEFORE RegionBranchOpInterface
    // because stream.cmd.execute implements both, and should be handled as a
    // timeline op.
    auto timelineOp = dyn_cast<IREE::Stream::TimelineOpInterface>(op);
    if (!timelineOp) {
      // Handle structured control flow recursively (scf.for, scf.if, etc.).
      if (auto regionBranchOp = dyn_cast<RegionBranchOpInterface>(op)) {
        if (!analyzeRegionBranchOp(regionBranchOp, asmState, lastUseSet,
                                   coverage, indeterminateResources,
                                   handledResources)) {
          // Unknown control flow - fallback to conservative marking.
          LLVM_DEBUG(
              llvm::dbgs()
              << "[arc] failed to analyze nested RegionBranchOpInterface "
              << op.getName() << "; using conservative fallback\n");
          markAllResourcesIndeterminateInOpAndRegions(op, lastUseSet,
                                                      indeterminateResources);
        }
        continue;
      }

      // Non-timeline, non-control-flow ops: mark resources indeterminate.
      for (auto operand : op.getOperands()) {
        if (isa<IREE::Stream::ResourceType>(operand.getType())) {
          indeterminateResources.insert(lastUseSet.lookupResource(operand));
        }
      }
      for (auto result : op.getResults()) {
        if (isa<IREE::Stream::ResourceType>(result.getType())) {
          indeterminateResources.insert(lastUseSet.lookupResource(result));
        }
      }
      continue;
    }

    // Process timeline op.
    Value resultTimepoint = timelineOp.getResultTimepoint();
    if (!resultTimepoint) {
      continue;
    }

    // Populate coverage.
    auto awaitTimepoints = timelineOp.getAwaitTimepoints();
    if (awaitTimepoints.empty()) {
      coverage.add(resultTimepoint);
    } else {
      for (Value awaitTimepoint : awaitTimepoints) {
        coverage.add(awaitTimepoint, resultTimepoint);
      }
    }

    // Check for explicitly indeterminate allocas.
    if (auto allocaOp = dyn_cast<IREE::Stream::ResourceAllocaOp>(op)) {
      if (allocaOp.getIndeterminateLifetime()) {
        indeterminateResources.insert(allocaOp.getResult());
      }
    }

    // Track existing deallocations.
    if (auto deallocaOp = dyn_cast<IREE::Stream::ResourceDeallocaOp>(op)) {
      handledResources.insert(
          lastUseSet.lookupResource(deallocaOp.getOperand()));
    }

    // Track resource consumption/production.
    auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
    for (auto operand : op.getOperands()) {
      if (isa<IREE::Stream::ResourceType>(operand.getType())) {
        lastUseSet.consume(operand, resultTimepoint);
      }
    }
    for (auto result : op.getResults()) {
      if (isa<IREE::Stream::ResourceType>(result.getType())) {
        Value operand = tiedOp ? tiedOp.getTiedResultOperand(result) : nullptr;
        if (operand) {
          lastUseSet.tie(operand, result, resultTimepoint);
        } else {
          lastUseSet.produce(result, resultTimepoint);
          // Mark non-alloca producers as indeterminate (#20817).
          if (!isa<IREE::Stream::ResourceAllocaOp>(op)) {
            indeterminateResources.insert(result);
          }
        }
      }
    }
  }
  return true;
}

// Inserts deallocations for all resources tracked in the LastUseSet.
// This is called after analysis is complete to insert deallocation operations.
static void insertDeallocations(LastUseSet &lastUseSet, AsmState *asmState,
                                DenseSet<Value> &indeterminateResources,
                                DenseSet<Value> &handledResources) {
  // Insert deallocations for all resources that we successfully analyzed.
  lastUseSet.forEachResource([&](Value resource, TimepointSet &timepoints) {
    assert(!timepoints.empty() && "all resources should have a timepoint");

    // Skip anything we could not analyze.
    Value baseResource = lastUseSet.lookupResource(resource);
    if (indeterminateResources.contains(baseResource)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[arc] skipping resource ";
        baseResource.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " marked as indeterminate\n";
      });
      return;
    } else if (handledResources.contains(baseResource)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[arc] skipping resource ";
        baseResource.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " marked as already handled\n";
      });
      return;
    }

    // Finds the last timepoint in the set (if >1) in SSA dominance order.
    Value lastTimepoint = getLastTimepointInBlock(timepoints);
    assert(lastTimepoint && "must have at least one timepoint");
    OpBuilder builder(lastTimepoint.getContext());
    builder.setInsertionPointAfterValue(lastTimepoint);

    // Try to grab a resource size or insert a query.
    // In almost all cases that this analysis can run we will have a
    // size-aware op that provides it.
    auto timepointsLoc =
        getFusedLocFromTimepoints(resource.getLoc(), timepoints);
    Value resourceSize = IREE::Util::SizeAwareTypeInterface::queryValueSize(
        timepointsLoc, resource, builder);

    // Lookup the affinity of the resource.
    // This should likely be a global affinity analysis but since we are
    // currently only processing locally we can assume this is only used when
    // we have an op local with an affinity assigned.
    IREE::Stream::AffinityAttr resourceAffinity;
    if (auto *definingOp = resource.getDefiningOp()) {
      resourceAffinity = IREE::Stream::AffinityAttr::lookup(definingOp);
    }
    UnitAttr preferOrigin =
        resourceAffinity ? UnitAttr{} : builder.getUnitAttr();

    if (timepoints.size() == 1) {
      // Single last user; the resource can have a deallocation directly
      // inserted as we have tracked both allocation and now deallocation to
      // single code points.
      LLVM_DEBUG({
        llvm::dbgs() << "[arc] inserting deallocation for ";
        resource.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " after timepoint ";
        lastTimepoint.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " directly\n";
      });
      auto deallocaOp = IREE::Stream::ResourceDeallocaOp::create(
          builder, timepointsLoc,
          builder.getType<IREE::Stream::TimepointType>(), resource,
          resourceSize, preferOrigin, lastTimepoint, resourceAffinity);
      lastTimepoint.replaceAllUsesExcept(deallocaOp.getResultTimepoint(),
                                         deallocaOp);
    } else if (timepoints.size() > 1) {
      // Multiple last users (fork); the resource still has a tracked
      // allocation and deallocation but there are multiple code points where
      // the deallocation may need to be inserted.
      //
      // Since this current analysis is local we can rely on SSA dominance to
      // find the last SSA value and insert a join on all timepoints there to
      // perform the deallocation, though this will cause extended lifetimes
      // in cases where scheduled timeline operations complete out of order.
      // We won't have correctness issues as all timepoints will be waited on.
      LLVM_DEBUG({
        llvm::dbgs() << "[arc] inserting forked deallocation for ";
        resource.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " after last SSA timepoint ";
        lastTimepoint.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " as a join\n";
      });
      auto joinOp = IREE::Stream::TimepointJoinOp::create(
          builder, timepointsLoc,
          builder.getType<IREE::Stream::TimepointType>(),
          llvm::map_to_vector(timepoints,
                              [](Value timepoint) { return timepoint; }));
      auto deallocaOp = IREE::Stream::ResourceDeallocaOp::create(
          builder, timepointsLoc,
          builder.getType<IREE::Stream::TimepointType>(), resource,
          resourceSize, preferOrigin, joinOp.getResultTimepoint(),
          resourceAffinity);
      lastTimepoint.replaceAllUsesExcept(deallocaOp.getResultTimepoint(),
                                         joinOp);
    }
  });
}

// Collects all timepoint results from an operation and creates a join if there
// are multiple. Returns the single timepoint or joined timepoint, or nullopt if
// no timepoint results exist.
static std::optional<Value> getOrJoinTimepointResults(Operation *op) {
  SmallVector<Value> resultTimepoints;
  for (Value result : op->getResults()) {
    if (isa<IREE::Stream::TimepointType>(result.getType())) {
      resultTimepoints.push_back(result);
    }
  }

  if (resultTimepoints.empty()) {
    return std::nullopt;
  }

  if (resultTimepoints.size() == 1) {
    return resultTimepoints[0];
  }

  // Multiple timepoint results - create a join.
  OpBuilder builder(op->getContext());
  builder.setInsertionPointAfter(op);
  auto joinOp = IREE::Stream::TimepointJoinOp::create(
      builder, op->getLoc(), builder.getType<IREE::Stream::TimepointType>(),
      resultTimepoints);
  LLVM_DEBUG({
    llvm::dbgs() << "[arc]   created join of " << resultTimepoints.size()
                 << " timepoint results\n";
  });
  return joinOp.getResultTimepoint();
}

// Extends the lifetime of captured resources (defined above a region but used
// within it) to a specified result timepoint.
//
// NOTE: This is a conservative over-approximation. Resources captured in a
// control flow region have their lifetimes extended to the region's result
// timepoint even if they are only used in one branch (scf.if) or only in the
// first iteration (scf.for). More precise per-iteration/per-branch tracking
// would require backward dataflow analysis which is not implemented.
static void extendCapturedResourceLifetimes(Region &region,
                                            Value resultTimepoint,
                                            LastUseSet &lastUseSet,
                                            AsmState *asmState) {
  SetVector<Value> capturedValues;
  getUsedValuesDefinedAbove(region, region, capturedValues);
  for (Value captured : capturedValues) {
    if (isa<IREE::Stream::ResourceType>(captured.getType())) {
      lastUseSet.consume(captured, resultTimepoint);
      LLVM_DEBUG({
        llvm::dbgs()
            << "[arc]   captured resource lifetime extended to result\n";
      });
    }
  }
}

// Analyzes scf.for loop with captured resource tracking.
static bool analyzeForLoop(scf::ForOp forOp, AsmState *asmState,
                           LastUseSet &lastUseSet,
                           ScopedTimepointCoverage &coverage,
                           DenseSet<Value> &indeterminateResources,
                           DenseSet<Value> &handledResources) {
  // scf.for doesn't implement TimelineOpInterface, but its result may be a
  // timepoint.
  std::optional<Value> loopResultTimepointOpt =
      getOrJoinTimepointResults(forOp);
  if (!loopResultTimepointOpt) {
    // No timepoint result, cannot track lifetimes through this loop.
    return false;
  }
  Value loopResultTimepoint = *loopResultTimepointOpt;

  // Register the loop result timepoint in the coverage map so that subsequent
  // consume() calls can recognize and prune timepoints dominated by this one.
  coverage.add(loopResultTimepoint);

  LLVM_DEBUG(
      { llvm::dbgs() << "[arc] recognized scf.for with timepoint result\n"; });

  // Step 1: Find captured resources (defined outside, used inside).
  // These need their lifetimes extended to the loop result in the parent block.
  extendCapturedResourceLifetimes(forOp.getRegion(), loopResultTimepoint,
                                  lastUseSet, asmState);

  // Step 2: Recursively analyze the loop body for local allocations.
  // Resources allocated and used entirely within the loop body should be
  // deallocated inside the loop body, UNLESS they are yielded out.
  for (Block &block : forOp.getRegion()) {
    // Create a fresh LastUseSet for analysis within this block.
    // This allows us to track and deallocate resources local to the loop body.
    LastUseSet blockLastUseSet(asmState, coverage);

    LLVM_DEBUG(llvm::dbgs() << "[arc]   analyzing loop body block\n");

    // Recursively analyze operations in this block.
    if (!analyzeBlockOps(block, asmState, blockLastUseSet, coverage,
                         indeterminateResources, handledResources)) {
      LLVM_DEBUG(llvm::dbgs() << "[arc]   failed to analyze loop body block\n");
      return false;
    }

    // Check if any resources are yielded out of this block.
    // Resources that escape via yield should NOT be deallocated locally.
    // Instead, they must be registered in the parent scope so deallocations
    // happen after the SCF operation completes.
    auto *terminator = block.getTerminator();
    if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
      for (auto [index, operand] : llvm::enumerate(yieldOp.getOperands())) {
        if (isa<IREE::Stream::ResourceType>(operand.getType())) {
          // Mark as handled to prevent local deallocation.
          handledResources.insert(operand);
          LLVM_DEBUG({
            llvm::dbgs() << "[arc]   resource ";
            operand.printAsOperand(llvm::dbgs(), *asmState);
            llvm::dbgs()
                << " yielded from loop body; preventing local deallocation\n";
          });

          // Register the corresponding scf.for result in the parent scope.
          // This ensures the resource gets deallocated after the loop even if
          // the result is not used (dropped).
          Value forResult = forOp.getResult(index);
          lastUseSet.produce(forResult, loopResultTimepoint);
          LLVM_DEBUG({
            llvm::dbgs() << "[arc]   registered loop result ";
            forResult.printAsOperand(llvm::dbgs(), *asmState);
            llvm::dbgs() << " in parent scope for deallocation\n";
          });
        }
      }
    }

    // Insert deallocations for resources local to this block (excluding yielded
    // ones).
    insertDeallocations(blockLastUseSet, asmState, indeterminateResources,
                        handledResources);
  }

  return true;
}

// Analyzes scf.if conditional with captured resource tracking.
static bool analyzeIfOp(scf::IfOp ifOp, AsmState *asmState,
                        LastUseSet &lastUseSet,
                        ScopedTimepointCoverage &coverage,
                        DenseSet<Value> &indeterminateResources,
                        DenseSet<Value> &handledResources) {
  // scf.if doesn't implement TimelineOpInterface, but its result may be a
  // timepoint.
  std::optional<Value> ifResultTpOpt = getOrJoinTimepointResults(ifOp);
  if (!ifResultTpOpt) {
    // No timepoint result, cannot track lifetimes through this conditional.
    return false;
  }
  Value ifResultTp = *ifResultTpOpt;

  // Register the if result timepoint in the coverage map.
  coverage.add(ifResultTp);

  LLVM_DEBUG(
      { llvm::dbgs() << "[arc] recognized scf.if with timepoint result\n"; });

  // Step 1: Find captured resources in both branches.
  // These need their lifetimes extended to the if result in the parent block.
  extendCapturedResourceLifetimes(ifOp.getThenRegion(), ifResultTp, lastUseSet,
                                  asmState);
  if (!ifOp.getElseRegion().empty()) {
    extendCapturedResourceLifetimes(ifOp.getElseRegion(), ifResultTp,
                                    lastUseSet, asmState);
  }

  // Step 2: Recursively analyze each branch for local allocations.
  // Resources allocated and used entirely within a branch should be
  // deallocated inside that branch, UNLESS they are yielded out.
  //
  // Track which if results have been registered to avoid duplicates.
  // Both branches may yield resources that map to the same result index.
  DenseSet<Value> registeredIfResults;
  auto analyzeRegion = [&](Region &region) -> bool {
    for (Block &block : region) {
      // Create a fresh LastUseSet for analysis within this block.
      LastUseSet blockLastUseSet(asmState, coverage);

      LLVM_DEBUG(llvm::dbgs()
                 << "[arc]   analyzing conditional branch block\n");

      // Recursively analyze operations in this block.
      if (!analyzeBlockOps(block, asmState, blockLastUseSet, coverage,
                           indeterminateResources, handledResources)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[arc]   failed to analyze conditional branch block\n");
        return false;
      }

      // Check if any resources are yielded out of this block.
      // Resources that escape via yield should NOT be deallocated locally.
      // Instead, they must be registered in the parent scope so deallocations
      // happen after the SCF operation completes.
      auto *terminator = block.getTerminator();
      if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
        for (auto [index, operand] : llvm::enumerate(yieldOp.getOperands())) {
          if (isa<IREE::Stream::ResourceType>(operand.getType())) {
            // Mark as handled to prevent local deallocation.
            handledResources.insert(operand);
            LLVM_DEBUG({
              llvm::dbgs() << "[arc]   resource ";
              operand.printAsOperand(llvm::dbgs(), *asmState);
              llvm::dbgs() << " yielded from conditional branch; preventing "
                              "local deallocation\n";
            });

            // Register the corresponding scf.if result in the parent scope.
            // Skip if already registered (e.g., from the other branch).
            Value ifResult = ifOp.getResult(index);
            if (!registeredIfResults.contains(ifResult)) {
              registeredIfResults.insert(ifResult);
              lastUseSet.produce(ifResult, ifResultTp);
              LLVM_DEBUG({
                llvm::dbgs() << "[arc]   registered if result ";
                ifResult.printAsOperand(llvm::dbgs(), *asmState);
                llvm::dbgs() << " in parent scope for deallocation\n";
              });
            }
          }
        }
      }

      // Insert deallocations for resources local to this block (excluding
      // yielded ones).
      insertDeallocations(blockLastUseSet, asmState, indeterminateResources,
                          handledResources);
    }
    return true;
  };

  if (!analyzeRegion(ifOp.getThenRegion())) {
    return false;
  }
  if (!ifOp.getElseRegion().empty() && !analyzeRegion(ifOp.getElseRegion())) {
    return false;
  }

  return true;
}

// Analyzes scf.while loop with captured resource tracking.
static bool analyzeWhileOp(scf::WhileOp whileOp, AsmState *asmState,
                           LastUseSet &lastUseSet,
                           ScopedTimepointCoverage &coverage,
                           DenseSet<Value> &indeterminateResources,
                           DenseSet<Value> &handledResources) {
  // scf.while has two regions: "before" (condition) and "after" (loop body).
  std::optional<Value> whileResultTimepointOpt =
      getOrJoinTimepointResults(whileOp);
  if (!whileResultTimepointOpt) {
    // No timepoint result, cannot track lifetimes through this loop.
    return false;
  }
  Value whileResultTimepoint = *whileResultTimepointOpt;

  // Register the while result timepoint in the coverage map.
  coverage.add(whileResultTimepoint);

  LLVM_DEBUG({
    llvm::dbgs() << "[arc] recognized scf.while with timepoint result\n";
  });

  // Step 1: Find captured resources (defined outside, used inside either
  // region).
  extendCapturedResourceLifetimes(whileOp.getBefore(), whileResultTimepoint,
                                  lastUseSet, asmState);
  extendCapturedResourceLifetimes(whileOp.getAfter(), whileResultTimepoint,
                                  lastUseSet, asmState);

  // Step 2: Recursively analyze both regions for local allocations.
  // For scf.while, we need to analyze both "before" and "after" regions.
  auto analyzeRegion = [&](Region &region) -> bool {
    for (Block &block : region) {
      LastUseSet blockLastUseSet(asmState, coverage);

      LLVM_DEBUG(llvm::dbgs() << "[arc]   analyzing while loop region block\n");

      if (!analyzeBlockOps(block, asmState, blockLastUseSet, coverage,
                           indeterminateResources, handledResources)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[arc]   failed to analyze while loop block\n");
        return false;
      }

      // Check for yielded resources.
      // Note: scf.while has TWO yield-like operations:
      // - scf.condition in "before" region: args become while results
      // - scf.yield in "after" region: args go back to "before" block args
      auto *terminator = block.getTerminator();
      if (auto conditionOp = dyn_cast<scf::ConditionOp>(terminator)) {
        // "before" region ends with scf.condition - args become while results.
        for (auto [index, operand] : llvm::enumerate(conditionOp.getArgs())) {
          if (isa<IREE::Stream::ResourceType>(operand.getType())) {
            handledResources.insert(operand);
            LLVM_DEBUG({
              llvm::dbgs() << "[arc]   resource ";
              operand.printAsOperand(llvm::dbgs(), *asmState);
              llvm::dbgs() << " yielded via scf.condition; "
                              "preventing local deallocation\n";
            });

            // Register the corresponding scf.while result in the parent scope.
            Value whileResult = whileOp.getResult(index);
            lastUseSet.produce(whileResult, whileResultTimepoint);
            LLVM_DEBUG({
              llvm::dbgs() << "[arc]   registered while result ";
              whileResult.printAsOperand(llvm::dbgs(), *asmState);
              llvm::dbgs() << " in parent scope for deallocation\n";
            });
          }
        }
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
        // "after" region ends with scf.yield.
        // These values go back to the "before" region, NOT to while results.
        // Mark as handled to prevent local deallocation (no parent produce).
        for (Value operand : yieldOp.getOperands()) {
          if (isa<IREE::Stream::ResourceType>(operand.getType())) {
            handledResources.insert(operand);
            LLVM_DEBUG({
              llvm::dbgs() << "[arc]   resource yielded from while body back "
                              "to loop; preventing local deallocation\n";
            });
          }
        }
      }

      insertDeallocations(blockLastUseSet, asmState, indeterminateResources,
                          handledResources);
    }
    return true;
  };

  if (!analyzeRegion(whileOp.getBefore())) {
    return false;
  }
  if (!analyzeRegion(whileOp.getAfter())) {
    return false;
  }

  return true;
}

// Dispatches to pattern-specific handlers for RegionBranchOpInterface ops.
static bool analyzeRegionBranchOp(RegionBranchOpInterface regionBranchOp,
                                  AsmState *asmState, LastUseSet &lastUseSet,
                                  ScopedTimepointCoverage &coverage,
                                  DenseSet<Value> &indeterminateResources,
                                  DenseSet<Value> &handledResources) {
  Operation *op = regionBranchOp.getOperation();

  // Dispatch to pattern-specific handlers using TypeSwitch.
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<scf::ForOp>([&](scf::ForOp forOp) {
        return analyzeForLoop(forOp, asmState, lastUseSet, coverage,
                              indeterminateResources, handledResources);
      })
      .Case<scf::IfOp>([&](scf::IfOp ifOp) {
        return analyzeIfOp(ifOp, asmState, lastUseSet, coverage,
                           indeterminateResources, handledResources);
      })
      .Case<scf::WhileOp>([&](scf::WhileOp whileOp) {
        return analyzeWhileOp(whileOp, asmState, lastUseSet, coverage,
                              indeterminateResources, handledResources);
      })
      .Default([](Operation *) {
        // Unknown RegionBranchOpInterface - cannot analyze.
        // This includes scf.parallel and scf.reduce which we don't use in IREE
        // today. If needed in the future, add explicit handlers for them.
        // Returning false triggers the conservative fallback in analyzeBlockOps
        // which marks all resources in the operation as indeterminate.
        // TODO(#12345): Add handlers for scf.parallel/scf.reduce if needed.
        return false;
      });
}

static void processFuncOp(FunctionOpInterface funcOp) {
  // Today we bail on unstructured control flow. Eventually we want to be able
  // to only support structured control flow in stream but today it can
  // sometimes leak through depending on how frontends lower ops.
  if (funcOp.getBlocks().size() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "[arc] skipping function @" << getFuncName(funcOp)
               << " as it contains unstructured control flow (>1 blocks)\n");
    return;
  }

  // NOTE: with the amount of IR this prints we must amortize the AsmState.
  // Do not print SSA values/ops without using it.
  std::unique_ptr<AsmState> asmState;
  LLVM_DEBUG({
    asmState = std::make_unique<AsmState>(
        funcOp, OpPrintingFlags().elideLargeElementsAttrs());
  });
  LLVM_DEBUG(llvm::dbgs() << "\n[arc] processing function @"
                          << getFuncName(funcOp) << ":\n");

  Liveness liveness(funcOp);
  DenseSet<Value> indeterminateResources;
  DenseSet<Value> handledResources;
  for (auto [blockIndex, block] : llvm::enumerate(funcOp.getBlocks())) {
    LLVM_DEBUG(llvm::dbgs() << "[arc] processing ^bb" << blockIndex << ":\n");
    ScopedTimepointCoverage coverage(asmState.get());
    LastUseSet lastUseSet(asmState.get(), coverage);

    // Add timepoint arguments as valid predecessors.
    for (auto arg : block.getArguments()) {
      if (isa<IREE::Stream::TimepointType>(arg.getType())) {
        coverage.add(arg);
      }
    }

    // Use liveness info to mark all live in/out resources as indeterminate
    // given the current limitations of this pass.
    auto *blockLiveness = liveness.getLiveness(&block);
    for (auto value : blockLiveness->in()) {
      if (isa<IREE::Stream::ResourceType>(value.getType())) {
        Value baseResource = lastUseSet.lookupResource(value);
        LLVM_DEBUG({
          llvm::dbgs() << "[arc] live-in resource ";
          baseResource.printAsOperand(llvm::dbgs(), *asmState);
          llvm::dbgs() << "; marking as indeterminate\n";
        });
        indeterminateResources.insert(baseResource);
      }
    }
    for (auto value : blockLiveness->out()) {
      if (isa<IREE::Stream::ResourceType>(value.getType())) {
        Value baseResource = lastUseSet.lookupResource(value);
        LLVM_DEBUG({
          llvm::dbgs() << "[arc] live-out resource ";
          baseResource.printAsOperand(llvm::dbgs(), *asmState);
          llvm::dbgs() << "; marking as indeterminate\n";
        });
        indeterminateResources.insert(baseResource);
      }
    }

    // Check for CallOpInterface before analyzing - we cannot handle calls yet.
    // TODO(benvanik): global analysis is required to know if calls are
    // side-effecting. We could annotate the calls as LLVM does so that we
    // could do local analysis and only pay attention to the operands/results.
    // util.call supports tied operands but does not have a way to associate
    // timepoints and this pass may never be able to work without that
    // information. The most common case of calls today is in external modules
    // not using `stream.cmd.call` and those are rare.
    for (auto &op : block) {
      if (isa<CallOpInterface>(op)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[arc] skipping function @" << getFuncName(funcOp)
                   << " as it contains call ops (" << op.getName() << ")\n");
        return;
      }
    }

    // Delegate block analysis to the shared helper.
    if (!analyzeBlockOps(block, asmState.get(), lastUseSet, coverage,
                         indeterminateResources, handledResources)) {
      LLVM_DEBUG(llvm::dbgs() << "[arc] failed to analyze function block\n");
      // The existing indeterminateResources/handledResources will prevent
      // deallocations for unanalyzable parts.
    }

    // Insert deallocations for all resources that we successfully analyzed.
    insertDeallocations(lastUseSet, asmState.get(), indeterminateResources,
                        handledResources);
  }

  LLVM_DEBUG(llvm::dbgs() << "\n");
}

// HACK: this implementation is currently very conservative and bails on most
// cases we cannot analyze locally. A larger DFX-based analysis that tracks
// resource usage through the entire program would allow us to deallocate
// replaced globals, reuse globals, and track program-local lifetime across
// functions/branches. This current implementation is only intended to help
// "v0" programs: those that mostly have a single function with a single block
// with no scf ops and no replaced globals. Hopefully this needs to be replaced
// soon - though knowing ML, it'll be 2030 by the time we have frontends that
// can generate a freaking if statement.
struct AutomaticReferenceCountingPass
    : public IREE::Stream::impl::AutomaticReferenceCountingPassBase<
          AutomaticReferenceCountingPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      if (funcOp.isExternal()) {
        continue;
      }
      processFuncOp(funcOp);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
