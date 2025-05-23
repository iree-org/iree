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
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-automatic-reference-counting"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_AUTOMATICREFERENCECOUNTINGPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Local analysis
//===----------------------------------------------------------------------===//

// Block-local analysis for timepoint coverage.
struct LocalTimepointCoverage {
  // Must be provided if LLVM_DEBUG is enabled.
  AsmState *asmState;
  // A map of timepoint SSA values to indices within the coverage map.
  // Values from other blocks are omitted. Order is that of the appearance
  // in the block but is not guaranteed and the value should only be used
  // for indexing into the map.
  DenseMap<Value, unsigned> timepoints;

  LocalTimepointCoverage() = delete;
  LocalTimepointCoverage(AsmState *asmState) : asmState(asmState) {}

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
  LocalTimepointCoverage &coverage;
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
  LastUseSet(AsmState *asmState, LocalTimepointCoverage &coverage)
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

// Returns the last defined SSA value in the block in |timepoints|.
// All timepoints must be in the same block.
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
    LocalTimepointCoverage coverage(asmState.get());
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

    for (auto &op : block) {
      // TODO(benvanik): global analysis is required to know if calls are
      // side-effecting. We could annotate the calls as LLVM does so that we
      // could do local analysis and only pay attention to the operands/results.
      // util.call supports tied operands but does not have a way to associate
      // timepoints and this pass may never be able to work without that
      // information. The most common case of calls today is in external modules
      // not using `stream.cmd.call` and those are rare.
      if (isa<CallOpInterface>(op)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[arc] skipping function @" << getFuncName(funcOp)
                   << " as it contains call ops (" << op.getName() << ")\n");
        return;
      }

      // TODO(benvanik): broader analysis or at least constrained handling of
      // scf. For now we bail if any ops from control flow dialects are found.
      // TODO(benvanik): maybe only skip the block containing the ops.
      if (op.getDialect()->getNamespace() == "scf") {
        LLVM_DEBUG(llvm::dbgs()
                   << "[arc] skipping function @" << getFuncName(funcOp)
                   << " block ^bb" << blockIndex
                   << " as it contains control flow ops (" << op.getName()
                   << ")\n");
        return;
      }

      // Special case ops that are not timeline-aware but interoperate.
      if (auto immediateOp = dyn_cast<IREE::Stream::TimepointImmediateOp>(op)) {
        coverage.add(immediateOp.getResultTimepoint());
        continue;
      } else if (auto importOp =
                     dyn_cast<IREE::Stream::TimepointImportOp>(op)) {
        coverage.add(importOp.getResultTimepoint());
        continue;
      }

      // Bail on processing any particular resource which has lifetime
      // management ops. They should not have been inserted yet and their
      // presence likely indicates we've already run the pass on this input.
      // We continue analysis for subsequent ops that may use different
      // resources so that we can handle other resources.
      if (auto retainOp = dyn_cast<IREE::Stream::ResourceRetainOp>(op)) {
        Value handledResource =
            lastUseSet.lookupResource(retainOp.getOperand());
        LLVM_DEBUG({
          llvm::dbgs() << "[arc] existing retain of ";
          handledResource.printAsOperand(llvm::dbgs(), *asmState);
          llvm::dbgs() << "; marking as handled\n";
        });
        handledResources.insert(handledResource);
        continue;
      } else if (auto releaseOp =
                     dyn_cast<IREE::Stream::ResourceReleaseOp>(op)) {
        Value handledResource =
            lastUseSet.lookupResource(releaseOp.getOperand());
        LLVM_DEBUG({
          llvm::dbgs() << "[arc] existing release of ";
          handledResource.printAsOperand(llvm::dbgs(), *asmState);
          llvm::dbgs() << "; marking as handled\n";
        });
        handledResources.insert(handledResource);
        continue;
      }

      // Analysis currently only works if all ops producing and consuming
      // resources are timeline ops. Any resource accessed by non-timeline ops
      // gets marked as indeterminate as the analysis does not know how they are
      // used on the timeline.
      auto timelineOp = dyn_cast<IREE::Stream::TimelineOpInterface>(op);
      if (!timelineOp) {
        for (auto operand : op.getOperands()) {
          if (isa<IREE::Stream::ResourceType>(operand.getType())) {
            Value baseResource = lastUseSet.lookupResource(operand);
            LLVM_DEBUG({
              llvm::dbgs() << "[arc] non-timeline use of operand ";
              baseResource.printAsOperand(llvm::dbgs(), *asmState);
              llvm::dbgs() << "; marking as indeterminate\n";
            });
            indeterminateResources.insert(baseResource);
          }
        }
        for (auto result : op.getResults()) {
          if (isa<IREE::Stream::ResourceType>(result.getType())) {
            Value baseResource = lastUseSet.lookupResource(result);
            LLVM_DEBUG({
              llvm::dbgs() << "[arc] non-timeline use of result ";
              baseResource.printAsOperand(llvm::dbgs(), *asmState);
              llvm::dbgs() << "; marking as indeterminate\n";
            });
            indeterminateResources.insert(baseResource);
          }
        }
        continue;
      }

      // Consumer-only timepoint ops (like stream.timepoint.await) block
      // propagation.
      Value resultTimepoint = timelineOp.getResultTimepoint();
      if (!resultTimepoint) {
        LLVM_DEBUG({
          llvm::dbgs() << "[arc] terminating timeline use by " << op.getName()
                       << "; stopping propagation\n";
        });
        continue;
      }

      // Populate coverage map for the declared timeline operation.
      auto awaitTimepoints = timelineOp.getAwaitTimepoints();
      if (awaitTimepoints.empty()) {
        coverage.add(resultTimepoint);
      } else {
        for (Value awaitTimepoint : timelineOp.getAwaitTimepoints()) {
          coverage.add(awaitTimepoint, resultTimepoint);
        }
      }

      // Alloca ops may have been assigned as indeterminate when created.
      if (auto allocaOp = dyn_cast<IREE::Stream::ResourceAllocaOp>(op)) {
        if (allocaOp.getIndeterminateLifetime()) {
          Value allocaResource = allocaOp.getResult();
          LLVM_DEBUG({
            llvm::dbgs() << "[arc] alloca producer explicitly states lifetime "
                            "is indeterminate for ";
            allocaResource.printAsOperand(llvm::dbgs(), *asmState);
            llvm::dbgs() << "; marking as indeterminate\n";
          });
          indeterminateResources.insert(allocaResource);
        }
      }

      // If a resource has a deallocation on it already then we cannot insert
      // another. This can arise when the pass is run twice or when an earlier
      // pass explicitly inserts deallocations to ensure they happen where they
      // want instead of relying on this analysis.
      if (auto deallocaOp = dyn_cast<IREE::Stream::ResourceDeallocaOp>(op)) {
        Value handledResource =
            lastUseSet.lookupResource(deallocaOp.getOperand());
        LLVM_DEBUG({
          llvm::dbgs() << "[arc] existing deallocation of ";
          handledResource.printAsOperand(llvm::dbgs(), *asmState);
          llvm::dbgs() << "; marking as handled\n";
        });
        handledResources.insert(handledResource);
      }

      // Track resources consumed/produced as part of the timeline operation.
      // This gives us the last timepoint(s) using the resources (not the last
      // SSA user). There may be multiple last timepoints if a fork occurs.
      //
      // Example:
      //   %r0, %t0 = alloca
      //   %t1 = exec %t0 => %r0
      //   %t2 = exec %t0 => %r0
      // The last timepoints of %r0 would be %t1 and %t2, indicating that after
      // %t1 and %t2 have both been reached (joined) the resource is no longer
      // live.
      auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
      for (auto operand : op.getOperands()) {
        if (isa<IREE::Stream::ResourceType>(operand.getType())) {
          lastUseSet.consume(operand, timelineOp.getResultTimepoint());
        }
      }
      for (auto result : op.getResults()) {
        if (isa<IREE::Stream::ResourceType>(result.getType())) {
          Value operand =
              tiedOp ? tiedOp.getTiedResultOperand(result) : nullptr;
          if (operand) {
            lastUseSet.tie(operand, result, timelineOp.getResultTimepoint());
          } else {
            lastUseSet.produce(result, timelineOp.getResultTimepoint());

            // TODO(#20817): parameter loads (and potentially other ops) may
            // cause far too many joins right now and will hit runtime errors
            // and performance issues. Until we have a way to partition joins we
            // need to avoid those. Parameter loads and other sources are things
            // we likely want to handle with retain/release instead of
            // deallocating anyway. Custom calls will also need a way to
            // indicate whether they are alloca-like or not so we just exclude
            // everything except alloca here.
            if (!isa<IREE::Stream::ResourceAllocaOp>(op)) {
              LLVM_DEBUG({
                llvm::dbgs() << "[arc] non-alloca producer "
                             << op.getName().getStringRef() << " of ";
                result.printAsOperand(llvm::dbgs(), *asmState);
                llvm::dbgs() << "; marking as indeterminate (#20817)\n";
              });
              indeterminateResources.insert(result);
            }
          }
        }
      }
    }

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
        auto deallocaOp = builder.create<IREE::Stream::ResourceDeallocaOp>(
            timepointsLoc, builder.getType<IREE::Stream::TimepointType>(),
            resource, resourceSize, preferOrigin, lastTimepoint,
            resourceAffinity);
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
        auto joinOp = builder.create<IREE::Stream::TimepointJoinOp>(
            timepointsLoc, builder.getType<IREE::Stream::TimepointType>(),
            llvm::map_to_vector(timepoints,
                                [](Value timepoint) { return timepoint; }));
        auto deallocaOp = builder.create<IREE::Stream::ResourceDeallocaOp>(
            timepointsLoc, builder.getType<IREE::Stream::TimepointType>(),
            resource, resourceSize, preferOrigin, joinOp.getResultTimepoint(),
            resourceAffinity);
        lastTimepoint.replaceAllUsesExcept(deallocaOp.getResultTimepoint(),
                                           joinOp);
      }
    });
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
    auto moduleOp = getOperation();
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
