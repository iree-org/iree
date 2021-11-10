// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-schedule-allocation"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

//===----------------------------------------------------------------------===//
// Alias analysis
//===----------------------------------------------------------------------===//

using ValueAliasingMap = llvm::MapVector<Value, SmallPtrSet<Value, 16>>;

// Builds a map of value aliases from aliasee to a set of aliasers.
// Only values that alias will be present in the map.
static ValueAliasingMap computeExecutionRegionValueAliases(
    IREE::Stream::AsyncExecuteOp executeOp) {
  auto *streamBlock = &executeOp.body().front();
  ValueAliasingMap valueAliases;

  auto propagateAlias = [&](Value streamValue, Value aliasedValue) {
    auto &baseSet = valueAliases[streamValue];
    baseSet.insert(aliasedValue);
    auto &aliasedSet = valueAliases[aliasedValue];
    baseSet.insert(aliasedSet.begin(), aliasedSet.end());
    aliasedSet.insert(streamValue);
  };

  // Start with outputs so that we handle tied values that may lead all the way
  // back up the chain to the stream inputs.
  auto tiedStreamOp =
      cast<IREE::Util::TiedOpInterface>(executeOp.getOperation());
  auto yieldOp = cast<IREE::Stream::YieldOp>(streamBlock->getTerminator());
  for (auto it : llvm::zip(executeOp.results(), yieldOp.operands())) {
    auto outerResult = std::get<0>(it);
    auto innerResult = std::get<1>(it);
    auto tiedOperandIndex =
        tiedStreamOp.getTiedResultOperandIndex(outerResult.getResultNumber());
    if (tiedOperandIndex.hasValue()) {
      auto arg = streamBlock->getArgument(tiedOperandIndex.getValue());
      propagateAlias(innerResult, arg);
    }
  }

  for (auto &op : *streamBlock) {
    // Tied results reuse their operand buffer.
    auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
    for (auto result : op.getResults()) {
      if (!result.getType().isa<IREE::Stream::ResourceType>()) continue;
      if (tiedOp) {
        auto tiedOperand = tiedOp.getTiedResultOperand(result);
        if (tiedOperand) {
          propagateAlias(result, tiedOperand);
        }
      }
    }
  }

  // Invert the value aliaser->aliasee map so that we have for any particular
  // value the list of all other values that alias it.
  for (auto it : valueAliases) {
    for (auto aliasee : it.second) {
      for (auto aliaser : it.second) {
        if (aliaser != aliasee) {
          valueAliases[aliasee].insert(aliaser);
        }
      }
    }
  }

  return valueAliases;
}

//===----------------------------------------------------------------------===//
// Liveness interval analysis
//===----------------------------------------------------------------------===//

static constexpr int LIVE_IN = INT_MIN;
static constexpr int LIVE_OUT = INT_MAX;
struct LivenessInterval {
  int start = 0;
  int end = 0;
  int ordinal = -1;  // unique per value
  Value value;
  bool operator<(const LivenessInterval &rhs) const {
    return ordinal < rhs.ordinal;
  }
};
using LivenessIntervalMap = DenseMap<Value, LivenessInterval>;
using LivenessIntervalList = SmallVector<LivenessInterval>;

// Computes the liveness intervals for each value in the execution region.
// Returns a closed range over an arbitrary operation ordering. The LIVE_IN and
// LIVE_OUT sentinels will be used to indicate values that are live-in and
// live-out to the execution region (captured input arguments and escaping
// output results).
//
// All values will have a range with aliased values sharing the union of their
// constituent ranges - including block arguments. Note that not all values will
// have buffers allocated to them - we are just tracking transitive SSA value
// lifetime.
static LivenessIntervalList computeExecutionRegionLivenessIntervals(
    IREE::Stream::AsyncExecuteOp executeOp,
    const ValueAliasingMap &valueAliases) {
  // Perform a liveness analysis on the execution region.
  // Fragments have a single block and as such the live-in/live-out block
  // information derived here applies to the entire stream region.
  assert(executeOp.body().getBlocks().size() == 1);
  auto *streamBlock = &executeOp.body().front();
  Liveness streamLiveness(executeOp);
  auto *livenessInfo = streamLiveness.getLiveness(streamBlock);

  // Operations don't allow us to get their already computed order so we make up
  // our own. We have a single block and thus the ordering is complete.
  DenseMap<Operation *, int> opOrdering;
  for (auto &op : *streamBlock) {
    opOrdering[&op] = opOrdering.size();
  }

  // Liveness doesn't track return values as live-outs so we do that here.
  SmallPtrSet<Value, 16> liveOuts;
  auto yieldOp = cast<IREE::Stream::YieldOp>(streamBlock->back());
  for (auto returnValue : yieldOp.operands()) {
    if (!returnValue.getType().isa<IREE::Stream::ResourceType>()) continue;
    liveOuts.insert(returnValue);
  }

  // Compute live-in intervals by hand as we won't catch them in the op walk
  // below since they are block arguments.
  LivenessIntervalMap valueIntervals;
  int ordinal = 0;
  for (Value value : streamBlock->getArguments()) {
    if (!value.getType().isa<IREE::Stream::ResourceType>()) continue;
    LivenessInterval interval;
    interval.start = LIVE_IN;
    if (liveOuts.contains(value)) {
      interval.end = LIVE_OUT;
    } else {
      auto *endOp = livenessInfo->getEndOperation(value, &streamBlock->front());
      interval.end = opOrdering[endOp];
    }
    interval.value = value;
    interval.ordinal = ++ordinal;
    valueIntervals[value] = interval;
  }

  // Compute ranges for all values independently (ignoring aliasing).
  for (auto &op : *streamBlock) {
    int start = opOrdering[&op];
    for (auto value : op.getResults()) {
      if (!value.getType().isa<IREE::Stream::ResourceType>()) continue;
      LivenessInterval interval;
      interval.start = start;
      if (liveOuts.contains(value)) {
        interval.end = LIVE_OUT;
      } else {
        interval.end = start;
        for (auto &use : value.getUses()) {
          interval.end = std::max(interval.end, opOrdering[use.getOwner()]);
        }
      }
      interval.value = value;
      interval.ordinal = ++ordinal;
      valueIntervals[value] = interval;
    }
  }

  // Walk the alias map and union intervals and propagate back.
  for (auto it : valueAliases) {
    auto &aliasee = it.first;
    auto &aliasers = it.second;
    auto &aliaseeInterval = valueIntervals[aliasee];
    assert(aliaseeInterval.ordinal != -1);
    int start = aliaseeInterval.start;
    int end = aliaseeInterval.end;
    for (auto aliaser : aliasers) {
      auto &aliaserInterval = valueIntervals[aliaser];
      assert(aliaserInterval.ordinal != -1);
      start = std::min(start, aliaserInterval.start);
      end = std::max(end, aliaserInterval.end);
    }
    aliaseeInterval.start = start;
    aliaseeInterval.end = end;
    for (auto aliaser : aliasers) {
      auto &aliaserInterval = valueIntervals[aliaser];
      aliaserInterval.start = start;
      aliaserInterval.end = end;
    }
  }

  // Sort all intervals by lifetime start. This makes the intervals easier to
  // read and deterministic across runs.
  SmallVector<LivenessInterval> sortedIntervals;
  sortedIntervals.reserve(valueIntervals.size());
  for (auto it : valueIntervals) {
    sortedIntervals.push_back(it.second);
  }
  llvm::stable_sort(sortedIntervals);
  return sortedIntervals;
}

//===----------------------------------------------------------------------===//
// Execution region allocation state
//===----------------------------------------------------------------------===//

struct ResourceRange {
  ResourceRange() = default;
  explicit ResourceRange(Value resource, Value resourceSize)
      : resource(resource), resourceSize(resourceSize) {}
  explicit ResourceRange(Value resource, Value resourceSize, Value offset,
                         Value length)
      : resource(resource),
        resourceSize(resourceSize),
        offset(offset),
        length(length) {}

  Value resource = nullptr;
  Value resourceSize = nullptr;
  Value offset = nullptr;
  Value length = nullptr;

  void print(llvm::raw_ostream &os, AsmState &asmState) {
    if (!resource) {
      os << "(null)";
      return;
    }
    resource.printAsOperand(os, asmState);
    os << "{";
    resourceSize.printAsOperand(os, asmState);
    os << "}";
    if (offset) {
      os << "[";
      offset.printAsOperand(os, asmState);
      os << " for ";
      length.printAsOperand(os, asmState);
      os << "]";
    }
  }
};

struct AllocationScope {
  explicit AllocationScope(IREE::Stream::AsyncExecuteOp rootAsyncOp)
      : rootOp(rootAsyncOp),
        valueAliases(computeExecutionRegionValueAliases(rootAsyncOp)) {}

  // Execution region being allocated.
  Operation *getRootOp() const { return rootOp; }

  // Aliasing map for the entire root op, indicating which values are tied.
  const ValueAliasingMap &getValueAliases() const { return valueAliases; }

  // TODO(benvanik): rework this so that we don't do a switcheroo right in the
  // middle of processing.
  void replaceRootOp(IREE::Stream::CmdExecuteOp newOp) {
    auto *oldOp = rootOp;
    rootOp = newOp;
    oldOp->erase();
  }

  // Returns a memoized ConstantIndexOp of |value|.
  Value lookupOrCreateIndex(int64_t value) {
    auto it = indexConstantMap.find(value);
    if (it != indexConstantMap.end()) return it->second;
    auto constantValue = OpBuilder(rootOp).createOrFold<arith::ConstantIndexOp>(
        rootOp->getLoc(), value);
    indexConstantMap.insert(std::make_pair(value, constantValue));
    return constantValue;
  }

  // Performs a memoized add (as many adds of offsets or lengths are redundant).
  Value add(Location loc, Value lhs, Value rhs) {
    // TODO(benvanik): memoize - if worth it. Needs profiling.
    auto result = OpBuilder(rootOp).createOrFold<arith::AddIOp>(loc, lhs, rhs);
    return result;
  }

  // Maps |resource| to the storage range defined by |resourceRange|.
  // All aliases of |resource| will also be mapped.
  void mapResourceRange(Value resource, ResourceRange resourceRange) {
    if (resourceRangeMap.count(resource)) return;

    if (!resourceRange.offset && !resourceRange.length) {
      resourceRange.offset = lookupOrCreateIndex(0);
      resourceRange.length = resourceRange.resourceSize;
    }

    resourceRangeMap.insert(std::make_pair(resource, resourceRange));
    LLVM_DEBUG({
      AsmState asmState(rootOp->getParentOp());
      llvm::dbgs() << " -> mapping ";
      resource.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << " = ";
      resourceRange.print(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    });

    // TODO(#5410): make alias propagation map through an indexing map for
    // slices/updates. Right now we assume all aliases are 1:1 full maps.
    for (auto alias : valueAliases[resource]) {
      resourceRangeMap.insert(std::make_pair(alias, resourceRange));
      LLVM_DEBUG({
        AsmState asmState(rootOp->getParentOp());
        llvm::dbgs() << "   = alias ";
        alias.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << " = ";
        resourceRange.print(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
    }
  }

  // Returns a storage range backing the given stream |resource|.
  ResourceRange lookupResourceRange(Value resource) const {
    auto it = resourceRangeMap.find(resource);
    LLVM_DEBUG({
      if (it == resourceRangeMap.end()) {
        AsmState asmState(rootOp->getParentOp());
        llvm::dbgs() << "!! storage not pre-allocated for resource ";
        resource.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << ":\n";
        for (auto mapping : resourceRangeMap) {
          llvm::dbgs() << "  * mapping ";
          mapping.first.printAsOperand(llvm::dbgs(), asmState);
          llvm::dbgs() << " -> ";
          mapping.second.print(llvm::dbgs(), asmState);
          llvm::dbgs() << "\n";
        }
      }
    });
    assert(it != resourceRangeMap.end() &&
           "storage not pre-allocated for resource");
    return it->second;
  }

  // Returns true if the given |resource| has a storage range mapped to it.
  bool hasResourceRange(Value resource) const {
    return resourceRangeMap.count(resource) != 0;
  }

  // Calls |callback| for |resource| and each value aliasing it.
  void forEachResourceAlias(Value resource,
                            std::function<void(Value)> callback) const {
    callback(resource);
    auto it = valueAliases.find(resource);
    if (it != valueAliases.end()) {
      for (auto alias : it->second) {
        callback(alias);
      }
    }
  }

 private:
  Operation *rootOp;

  // All values that have aliases mapped to a set of all of the values they
  // alias with. That two things alias does not imply the values can be treated
  // as equivalent: some values may be subranges of others.
  ValueAliasingMap valueAliases;

  // Index value -> std.constant index value.
  DenseMap<int64_t, Value> indexConstantMap;

  // Maps resource values inside the stream to a storage range.
  DenseMap<Value, ResourceRange> resourceRangeMap;
};

static LogicalResult applyResourceSubviewOp(
    IREE::Stream::ResourceSubviewOp asyncOp, AllocationScope &scope,
    OpBuilder builder) {
  // Allocation should have taken care of this by propagating the range.
  // By the time we walk to this op there should be no more users.
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncAllocaOp(IREE::Stream::AsyncAllocaOp asyncOp,
                                        AllocationScope &scope,
                                        OpBuilder builder) {
  // Allocation should have taken care of this and hoisted the alloc outside.
  // By the time we walk to this op there should be no more users.
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncConstantOp(IREE::Stream::AsyncConstantOp asyncOp,
                                          AllocationScope &scope,
                                          OpBuilder builder) {
  // Allocation should have taken care of this and hoisted the constant upload
  // outside. By the time we walk to this op there should be no more users.
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncSplatOp(IREE::Stream::AsyncSplatOp asyncOp,
                                       AllocationScope &scope,
                                       OpBuilder builder) {
  auto targetRange = scope.lookupResourceRange(asyncOp.result());
  builder.create<IREE::Stream::CmdFillOp>(
      asyncOp.getLoc(), targetRange.resource, targetRange.resourceSize,
      targetRange.offset, targetRange.length, asyncOp.value());
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncCloneOp(IREE::Stream::AsyncCloneOp asyncOp,
                                       AllocationScope &scope,
                                       OpBuilder builder) {
  auto sourceRange = scope.lookupResourceRange(asyncOp.source());
  auto targetRange = scope.lookupResourceRange(asyncOp.result());
  builder.create<IREE::Stream::CmdCopyOp>(
      asyncOp.getLoc(), sourceRange.resource, sourceRange.resourceSize,
      sourceRange.offset, targetRange.resource, targetRange.resourceSize,
      targetRange.offset, targetRange.length);
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncSliceOp(IREE::Stream::AsyncSliceOp asyncOp,
                                       AllocationScope &scope,
                                       OpBuilder builder) {
  // TODO(benvanik): place the allocation instead.
  auto sourceRange = scope.lookupResourceRange(asyncOp.source());
  auto sourceOffset =
      scope.add(asyncOp.getLoc(), sourceRange.offset, asyncOp.source_offset());
  auto targetRange = scope.lookupResourceRange(asyncOp.result());
  builder.create<IREE::Stream::CmdCopyOp>(
      asyncOp.getLoc(), sourceRange.resource, sourceRange.resourceSize,
      sourceOffset, targetRange.resource, targetRange.resourceSize,
      targetRange.offset, asyncOp.result_size());
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncFillOp(IREE::Stream::AsyncFillOp asyncOp,
                                      AllocationScope &scope,
                                      OpBuilder builder) {
  auto targetRange = scope.lookupResourceRange(asyncOp.result());
  auto targetOffset =
      scope.add(asyncOp.getLoc(), targetRange.offset, asyncOp.target_offset());
  builder.create<IREE::Stream::CmdFillOp>(
      asyncOp.getLoc(), targetRange.resource, targetRange.resourceSize,
      targetOffset, asyncOp.target_length(), asyncOp.value());
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncUpdateOp(IREE::Stream::AsyncUpdateOp asyncOp,
                                        AllocationScope &scope,
                                        OpBuilder builder) {
  // TODO(benvanik): place the allocation instead.
  auto sourceRange = scope.lookupResourceRange(asyncOp.update());
  auto sourceOffset = sourceRange.offset;
  auto targetRange = scope.lookupResourceRange(asyncOp.result());
  auto targetOffset =
      scope.add(asyncOp.getLoc(), targetRange.offset, asyncOp.target_offset());
  builder.create<IREE::Stream::CmdCopyOp>(
      asyncOp.getLoc(), sourceRange.resource, sourceRange.resourceSize,
      sourceOffset, targetRange.resource, targetRange.resourceSize,
      targetOffset, asyncOp.update_size());
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncCopyOp(IREE::Stream::AsyncCopyOp asyncOp,
                                      AllocationScope &scope,
                                      OpBuilder builder) {
  auto sourceRange = scope.lookupResourceRange(asyncOp.source());
  auto sourceOffset =
      scope.add(asyncOp.getLoc(), sourceRange.offset, asyncOp.source_offset());
  auto targetRange = scope.lookupResourceRange(asyncOp.result());
  auto targetOffset =
      scope.add(asyncOp.getLoc(), targetRange.offset, asyncOp.target_offset());
  builder.create<IREE::Stream::CmdCopyOp>(
      asyncOp.getLoc(), sourceRange.resource, sourceRange.resourceSize,
      sourceOffset, targetRange.resource, targetRange.resourceSize,
      targetOffset, asyncOp.length());
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncTransferOp(IREE::Stream::AsyncTransferOp asyncOp,
                                          AllocationScope &scope,
                                          OpBuilder builder) {
  // Lookup the affinity for where we are executing. This lets us determine if
  // this transfer is incoming or outgoing.
  auto isStaging = [](Value value) {
    return value.getType().cast<IREE::Stream::ResourceType>().getLifetime() ==
           IREE::Stream::Lifetime::Staging;
  };
  auto currentAffinityAttr = IREE::Stream::AffinityAttr::lookup(asyncOp);
  bool transferIn = asyncOp.source_affinityAttr() != currentAffinityAttr ||
                    isStaging(asyncOp.source());
  bool transferOut = asyncOp.result_affinityAttr() != currentAffinityAttr ||
                     isStaging(asyncOp.result());

  auto sourceRange = scope.lookupResourceRange(asyncOp.source());
  auto targetRange = scope.lookupResourceRange(asyncOp.result());

  // Incoming transfers need invalidation.
  if (transferIn) {
    builder.create<IREE::Stream::CmdInvalidateOp>(
        asyncOp.getLoc(), sourceRange.resource, sourceRange.resourceSize,
        sourceRange.offset, sourceRange.length, asyncOp.source_affinityAttr());
  }

  // Perform the copy.
  builder.create<IREE::Stream::CmdCopyOp>(
      asyncOp.getLoc(), sourceRange.resource, sourceRange.resourceSize,
      sourceRange.offset, targetRange.resource, targetRange.resourceSize,
      targetRange.offset, sourceRange.length);

  // Outgoing transfers need flushes.
  if (transferOut) {
    builder.create<IREE::Stream::CmdFlushOp>(
        asyncOp.getLoc(), targetRange.resource, targetRange.resourceSize,
        targetRange.offset, targetRange.length, asyncOp.result_affinityAttr());
  }

  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncDispatchOp(IREE::Stream::AsyncDispatchOp asyncOp,
                                          AllocationScope &scope,
                                          OpBuilder builder) {
  SmallVector<Value> newOperands;
  SmallVector<Value> newResources;
  SmallVector<Value> newResourceSizes;
  SmallVector<Value> newResourceOffsets;
  SmallVector<Value> newResourceLengths;
  SmallVector<Attribute> newResourceAccesses;

  for (auto it : llvm::enumerate(asyncOp.operands())) {
    auto operand = it.value();
    if (!operand.getType().isa<IREE::Stream::ResourceType>()) {
      // Primitive operand.
      newOperands.push_back(operand);
      continue;
    }

    // Read-only or read-write. Write-only are untied results below.
    unsigned operandIdx =
        asyncOp.getTiedOperandsIndexAndLength().first + it.index();
    auto accessBits = IREE::Stream::ResourceAccessBitfield::Read;
    if (asyncOp.isOperandTied(operandIdx)) {
      accessBits = accessBits | IREE::Stream::ResourceAccessBitfield::Write;
    }

    auto resourceRange = scope.lookupResourceRange(operand);
    auto resourceOffset = resourceRange.offset;
    auto resourceLength = asyncOp.getOperandSize(operandIdx);
    auto resourceAccess = IREE::Stream::ResourceAccessBitfieldAttr::get(
        builder.getContext(), accessBits);
    newResources.push_back(resourceRange.resource);
    newResourceSizes.push_back(resourceRange.resourceSize);
    newResourceOffsets.push_back(resourceOffset);
    newResourceLengths.push_back(resourceLength);
    newResourceAccesses.push_back(resourceAccess);
  }

  for (auto result : asyncOp.results()) {
    auto tiedOperand = asyncOp.getTiedResultOperand(result);
    if (tiedOperand) {
      // All tied results are handled above as read-write.
      continue;
    }
    auto resourceRange = scope.lookupResourceRange(result);
    auto resourceOffset = resourceRange.offset;
    auto resourceLength = asyncOp.getResultSize(result.getResultNumber());
    auto resourceAccess = IREE::Stream::ResourceAccessBitfieldAttr::get(
        builder.getContext(), IREE::Stream::ResourceAccessBitfield::Write);
    newResources.push_back(resourceRange.resource);
    newResourceSizes.push_back(resourceRange.resourceSize);
    newResourceOffsets.push_back(resourceOffset);
    newResourceLengths.push_back(resourceLength);
    newResourceAccesses.push_back(resourceAccess);
  }

  builder.create<IREE::Stream::CmdDispatchOp>(
      asyncOp.getLoc(), asyncOp.workgroup_count(), asyncOp.entry_point(),
      newOperands, newResources, newResourceSizes, newResourceOffsets,
      newResourceLengths, builder.getArrayAttr(newResourceAccesses));
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncAllocations(Region &region,
                                           AllocationScope &scope);

static LogicalResult applyAsyncConcurrentOp(
    IREE::Stream::AsyncConcurrentOp asyncOp, AllocationScope &scope,
    OpBuilder builder) {
  // Remove operands from the yield now that we aren't returning anything.
  // Must do this before we recurse so that the ops we are transforming have no
  // uses.
  auto yieldOp =
      cast<IREE::Stream::YieldOp>(asyncOp.body().front().getTerminator());
  yieldOp->eraseOperands(0, yieldOp.getNumOperands());

  // Recurse into the wave ops to process the async ops within.
  // Resources are captured inside of the region and need to be mapped back to
  // the parent scope where they will be captured after lowering into
  // stream.cmd.execute.
  if (failed(applyAsyncAllocations(asyncOp.body(), scope))) {
    return failure();
  }

  // Explicit variant has no args/operands
  auto &block = asyncOp.body().front();
  block.eraseArguments([&](auto arg) { return true; });

  // Rewrite wave op to remove results.
  auto newOp = builder.create<IREE::Stream::CmdConcurrentOp>(asyncOp.getLoc());
  newOp.body().takeBody(asyncOp.body());
  asyncOp.erase();
  return success();
}

// Converts async operations to explicit commands using the allocation mappings
// in |scope|. Upon successful return the region should have no more defined
// values.
static LogicalResult applyAsyncAllocations(Region &region,
                                           AllocationScope &scope) {
  // Walk the ops backwards so that we can delete them, freeing uses so that
  // producers can be deleted in turn.
  auto &block = region.getBlocks().front();
  auto ops = llvm::to_vector<4>(llvm::map_range(
      llvm::reverse(block), [&](Operation &op) { return &op; }));
  for (auto *op : ops) {
    if (op->hasTrait<OpTrait::IsTerminator>()) continue;
    if (failed(TypeSwitch<Operation *, LogicalResult>(op)
                   .Case([&](IREE::Stream::ResourceSubviewOp op) {
                     return applyResourceSubviewOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncAllocaOp op) {
                     return applyAsyncAllocaOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncConstantOp op) {
                     return applyAsyncConstantOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncSplatOp op) {
                     return applyAsyncSplatOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncCloneOp op) {
                     return applyAsyncCloneOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncSliceOp op) {
                     return applyAsyncSliceOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncFillOp op) {
                     return applyAsyncFillOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncUpdateOp op) {
                     return applyAsyncUpdateOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncCopyOp op) {
                     return applyAsyncCopyOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncTransferOp op) {
                     return applyAsyncTransferOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncDispatchOp op) {
                     return applyAsyncDispatchOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncConcurrentOp op) {
                     return applyAsyncConcurrentOp(op, scope, OpBuilder(op));
                   })
                   .Default(failure()))) {
      return region.getParentOp()->emitError()
             << "unhandled async op " << op->getName().getStringRef()
             << " during allocation (should have been removed earlier)";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Execution region local transient allocation
//===----------------------------------------------------------------------===//

struct TransientAllocation {
  // Timepoint that indicates availability of the allocation.
  // Execution must await on this value before using the memory.
  Value awaitTimepoint = nullptr;
  // Reservation storage of the total slab size.
  Value reservation = nullptr;
  // Size of the reservation storage.
  Value reservationSize = nullptr;
  // Entry block argument of the captured reservation.
  BlockArgument capturedArg = nullptr;
};

// Performs allocation for all local transients in the execution region (those
// !stream.resource<transient> values that don't escape). A new allocation op
// will be inserted using |externalBuilder| and mappings added to |scope|.
static llvm::Optional<TransientAllocation> allocateLocalTransients(
    IREE::Stream::AsyncExecuteOp executeOp, AllocationScope &scope,
    OpBuilder &externalBuilder) {
  // Track which values we've already reserved. This makes it easier to early-
  // exit on aliased values.
  SmallPtrSet<Value, 16> coveredValues;

  // Gather all of the transient values we need to allocate buffers for.
  SmallVector<Location> locs;
  SmallVector<Value> transientValues;
  SmallVector<int64_t> lifetimeIntervals;
  SmallVector<Value> dynamicSliceSizes;
  auto livenessIntervals = computeExecutionRegionLivenessIntervals(
      executeOp, scope.getValueAliases());
  for (auto valueInterval : livenessIntervals) {
    auto value = valueInterval.value;
    auto valueType = value.getType().dyn_cast<IREE::Stream::ResourceType>();
    if (!valueType) continue;

    // Only handle transient buffers (created/used/dropped within the stream).
    if (valueInterval.start == LIVE_IN || valueInterval.end == LIVE_OUT) {
      continue;
    }

    // Ignore covered values.
    if (scope.hasResourceRange(value) || coveredValues.contains(value)) {
      continue;
    }

    locs.push_back(value.getLoc());
    transientValues.push_back(value);
    lifetimeIntervals.push_back(valueInterval.start);
    lifetimeIntervals.push_back(valueInterval.end);

    // Compute the allocation size for the value.
    auto allocationSize = IREE::Util::SizeAwareTypeInterface::findSizeValue(
        value, executeOp->getBlock(), Block::iterator(executeOp));
    assert(allocationSize && "must have computable size");
    dynamicSliceSizes.push_back(allocationSize);

    // Mark as covered so we don't allocate it again.
    scope.forEachResourceAlias(
        value, [&](Value alias) { coveredValues.insert(alias); });
  }
  if (transientValues.empty()) {
    // No transients required.
    return llvm::None;
  }

  // Insert the stream.resource.pack op to compute the slices and total size of
  // the transient allocation required.
  auto fusedLoc = externalBuilder.getFusedLoc(locs);
  auto indexType = externalBuilder.getIndexType();
  SmallVector<Type> packedOffsetTypes(dynamicSliceSizes.size(), indexType);
  auto packOp = externalBuilder.create<IREE::Stream::ResourcePackOp>(
      fusedLoc, indexType, packedOffsetTypes,
      /*offset=*/nullptr, externalBuilder.getIndexArrayAttr(lifetimeIntervals),
      dynamicSliceSizes, executeOp.affinityAttr());

  // Allocate the transient storage for the entire packed slab.
  auto transientType = externalBuilder.getType<IREE::Stream::ResourceType>(
      IREE::Stream::Lifetime::Transient);
  auto timepointType = externalBuilder.getType<IREE::Stream::TimepointType>();
  auto allocaOp = externalBuilder.create<IREE::Stream::ResourceAllocaOp>(
      fusedLoc, transientType, timepointType, packOp.total_length(),
      executeOp.affinityAttr());
  TransientAllocation allocation;
  allocation.awaitTimepoint = allocaOp.result_timepoint();
  allocation.reservation = allocaOp.result();
  allocation.reservationSize = allocaOp.storage_size();
  allocation.capturedArg = executeOp.body().front().addArgument(
      allocation.reservation.getType(), allocation.reservation.getLoc());

  // Map values to their ranges within the slab.
  for (size_t i = 0; i < transientValues.size(); ++i) {
    auto value = transientValues[i];
    auto offset = packOp.packed_offsets()[i];
    auto length = packOp.dynamic_slice_sizes()[i];
    auto resourceRange = ResourceRange(
        allocation.capturedArg, allocation.reservationSize, offset, length);
    scope.mapResourceRange(value, resourceRange);
  }

  return allocation;
}

//===----------------------------------------------------------------------===//
// Constant allocation
//===----------------------------------------------------------------------===//

struct ConstantReservation {
  IREE::Stream::AsyncConstantOp constantOp;
  Value resource;
  Value resourceSize;
  // NOTE: may be nullptr if the constant is not used within the region.
  BlockArgument capturedArg;
};

struct ConstantAllocation {
  IREE::Stream::ResourceConstantsOp constantsOp;
  SmallVector<ConstantReservation> reservations;
};

// Returns true if |value| has one use and it is a stream.yield op.
static bool isOnlyUseYield(Value value) {
  for (auto *user : value.getUsers()) {
    if (!isa<IREE::Stream::YieldOp>(user)) return false;
  }
  return true;
}

// Extracts stream.async.constant ops from |executeOp| into their own dedicated
// stream.resource.constants upload op. The uploaded constants will be captured
// by the region for use within as if they had still existed in there.
static Optional<ConstantAllocation> extractConstants(
    IREE::Stream::AsyncExecuteOp executeOp, OpBuilder &externalBuilder) {
  // Gather all constant ops from the region, if any.
  auto constantOps =
      llvm::to_vector<4>(executeOp.getOps<IREE::Stream::AsyncConstantOp>());
  if (constantOps.empty()) return llvm::None;

  // Allocate a new constant upload op and insert a subview for each constant.
  SmallVector<Location> locs;
  SmallVector<Type> resultTypes;
  SmallVector<Attribute> initialValues;
  SmallVector<Value> resultSizes;
  for (auto constantOp : constantOps) {
    locs.push_back(constantOp.getLoc());
    resultTypes.push_back(constantOp.result().getType());
    initialValues.push_back(constantOp.value());
    resultSizes.push_back(constantOp.result_size());
  }
  auto timepointType = externalBuilder.getType<IREE::Stream::TimepointType>();
  ConstantAllocation allocation;
  allocation.constantsOp =
      externalBuilder.create<IREE::Stream::ResourceConstantsOp>(
          externalBuilder.getFusedLoc(locs), resultTypes, timepointType,
          externalBuilder.getArrayAttr(initialValues), resultSizes,
          executeOp.affinityAttr());

  // Remap original constants to reservations.
  auto &entryBlock = executeOp.body().front();
  for (auto it : llvm::enumerate(constantOps)) {
    unsigned idx = static_cast<unsigned>(it.index());
    auto constantOp = it.value();
    ConstantReservation reservation;
    reservation.constantOp = constantOp;

    // Grab the returned resource from the upload op. This will later likely
    // be subviewed but all we know here is what the subview result will be.
    reservation.resource = allocation.constantsOp.results()[idx];
    reservation.resourceSize = allocation.constantsOp.result_sizes()[idx];

    // Capture the subview resource and switch all uses of the internal op if
    // there are any. Otherwise, if it's just yield, we can avoid the capture
    // and implicit dependency.
    if (!constantOp.use_empty() && !isOnlyUseYield(constantOp.result())) {
      reservation.capturedArg =
          entryBlock.addArgument(constantOp.result().getType());
    }

    allocation.reservations.push_back(reservation);
  }
  return allocation;
}

//===----------------------------------------------------------------------===//
// Execution region result allocation
//===----------------------------------------------------------------------===//

struct ResultReservation {
  // Location of the result used for attribution.
  Location loc;
  // Original execution region result.
  Value result;
  // Resource type to allocate.
  IREE::Stream::ResourceType resultType;
  // Size of the result resource in bytes.
  Value resultSize;
  // Value passed to the stream.yield inside of the execution region.
  Value yieldValue;
};

struct ResultReservationSet {
  SmallVector<ResultReservation> reservations;
  // Locations of all reservations.
  SmallVector<Location> reservationLocs;
  // Types of all reservations.
  SmallVector<Type> reservationTypes;
  // Size of each resource reservation.
  SmallVector<Value> reservationSizes;
};

struct ResultAllocation {
  // Reservations bucketed by lifetime.
  SmallVector<ResultReservationSet> reservationSets;
};

// Produces parameters for one or more result allocations composed of an ordered
// set of |reservations| with matching lifetimes.
static ResultAllocation reserveResultAllocation(
    ArrayRef<ResultReservation> reservations) {
  // We want deterministic ordering of the allocations for each lifetime type
  // so we build them all here and then just nuke the ones we don't end up
  // using.
  SmallVector<ResultReservationSet> sets(
      IREE::Stream::getMaxEnumValForLifetime() + 1);
  for (auto &reservation : reservations) {
    auto &set =
        sets[static_cast<unsigned>(reservation.resultType.getLifetime())];
    set.reservationLocs.push_back(reservation.loc);
    set.reservationTypes.push_back(reservation.resultType);
    set.reservationSizes.push_back(reservation.resultSize);
    set.reservations.push_back(std::move(reservation));
  }

  // Remove unused sets. This does a bunch of moves and is really bad but eh.
  for (int i = sets.size() - 1; i >= 0; --i) {
    if (sets[i].reservations.empty()) {
      sets.erase(sets.begin() + i);
    }
  }
  return ResultAllocation{sets};
}

//===----------------------------------------------------------------------===//
// Execution region allocation
//===----------------------------------------------------------------------===//

// Walks the use-def chain to find a transitive escape of |seedValue| and
// returns the outer region result (if any).
static Value findTiedYieldResult(Value seedValue) {
  auto regionOp =
      cast<RegionBranchOpInterface>(seedValue.getParentRegion()->getParentOp());
  SmallVector<RegionSuccessor> regions;
  regionOp.getSuccessorRegions(0, regions);
  auto results = regions.front().getSuccessorInputs();
  SmallVector<Value> worklist;
  worklist.push_back(seedValue);
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    for (auto &use : value.getUses()) {
      if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(use.getOwner())) {
        worklist.append(tiedOp.getOperandTiedResults(use.getOperandNumber()));
      } else if (isa<IREE::Stream::YieldOp>(use.getOwner())) {
        // Escaping through a yield.
        return results[use.getOperandNumber()];
      }
    }
  }
  return {};
}

// TODO(benvanik): find a way to split this up. We could probably do this in
// several passes each time pulling out different resource types, however the
// analysis we perform needs to see the original form and getting a shared
// analysis that is durable across passes is difficult. We could at least not
// have a giant function in here. The split between constants and results (which
// includes variables) also creates suboptimal IR as with initialized variables
// we end up with two independent allocations that could otherwise be one.

// Performs allocation for all results and local region transients of the given
// |executeOp| region. IR will be inserted around the op in its parent block.
static LogicalResult allocateExecutionRegion(
    IREE::Stream::AsyncExecuteOp executeOp) {
  LLVM_DEBUG(llvm::dbgs() << "[[ Allocating execution region ]]\n");

  AllocationScope scope(executeOp);

  OpBuilder externalBuilder(executeOp);

  auto &entryBlock = executeOp.body().front();
  auto yieldOp = cast<IREE::Stream::YieldOp>(entryBlock.back());

  // We're going to allocate transients - if we do, we need to insert
  // corresponding deallocs. This tracks the (resource, resource_size) for
  // insertion below.
  SmallVector<std::pair<Value, Value>> pendingReleases;

  SmallVector<Value> newAwaitTimepoints;
  SmallVector<Value> newOperands;
  SmallVector<Value> newOperandSizes;
  SmallVector<std::pair<Value, Value>> resultReplacements;
  SetVector<Value> handledResults;
  if (executeOp.await_timepoint()) {
    newAwaitTimepoints.push_back(executeOp.await_timepoint());
  }
  llvm::append_range(newOperands, executeOp.operands());
  llvm::append_range(newOperandSizes, executeOp.operand_sizes());
  SmallVector<Value> joinTimepoints;

  // First find all constants and pull them out into a dedicated constant upload
  // op. We'll then capture the result and use that to initialize variables and
  // constants within the region. Note that this removes ops from the region and
  // as such we want to run it first before we go allocate transients.
  auto constantAllocation = extractConstants(executeOp, externalBuilder);
  if (constantAllocation.hasValue()) {
    bool anyCaptured = false;
    for (auto &reservation : constantAllocation->reservations) {
      if (reservation.capturedArg) {
        newOperands.push_back(reservation.resource);
        newOperandSizes.push_back(reservation.resourceSize);
        anyCaptured = true;
      }
      LLVM_DEBUG({
        AsmState asmState(executeOp->getParentOp());
        llvm::dbgs() << "    ";
        reservation.resource.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "{";
        reservation.resourceSize.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "}";
        if (reservation.capturedArg) {
          llvm::dbgs() << " captured as ";
          reservation.capturedArg.printAsOperand(llvm::dbgs(), asmState);
        }
        llvm::dbgs() << "\n";
      });
    }

    auto awaitTimepoint = constantAllocation->constantsOp.result_timepoint();
    if (anyCaptured) {
      // The execute region must depend on the constant upload as one or more
      // constants are used. All this code could be much more clever about
      // separating ones that have deps and ones that don't so that we minimize
      // the dependency chain.
      newAwaitTimepoints.push_back(awaitTimepoint);
      LLVM_DEBUG({
        AsmState asmState(executeOp->getParentOp());
        llvm::dbgs() << "  + adding await on dependent constant upload ";
        awaitTimepoint.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      for (auto &reservation : constantAllocation->reservations) {
        auto resourceRange =
            ResourceRange(reservation.capturedArg, reservation.resourceSize);
        scope.mapResourceRange(reservation.constantOp, resourceRange);
      }
    } else {
      // No deps within the execute op but we need to keep the dependency chain
      // valid: consumers of the execute op are expecting the constants to be
      // ready.
      joinTimepoints.push_back(awaitTimepoint);
      LLVM_DEBUG({
        AsmState asmState(executeOp->getParentOp());
        llvm::dbgs() << "  + adding join on non-dependent constant upload ";
        awaitTimepoint.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
    }

    // Replace results of escaping uploads with the upload values.
    for (auto &reservation : constantAllocation->reservations) {
      auto result = findTiedYieldResult(reservation.constantOp.result());
      if (!result) continue;
      result.replaceAllUsesWith(reservation.resource);
      handledResults.insert(result);
      LLVM_DEBUG({
        AsmState asmState(executeOp->getParentOp());
        llvm::dbgs() << "  = replacing result ";
        result.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << " with upload ";
        reservation.resource.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "  - no constants found\n");
  }

  // Compute an updated set of operands/results. After allocation all results
  // must be tied to operands; it's possible though that this is already the
  // case by construction.
  for (auto operand : llvm::enumerate(executeOp.operands())) {
    unsigned operandIdx =
        executeOp.getTiedOperandsIndexAndLength().first + operand.index();
    auto operandSize = executeOp.getOperandSize(operandIdx);
    auto arg = entryBlock.getArgument(operand.index());
    LLVM_DEBUG({
      AsmState asmState(executeOp->getParentOp());
      llvm::dbgs() << "  - tying argument ";
      arg.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << " = ";
      operand.value().printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    });
    auto resourceRange = ResourceRange(arg, operandSize);
    scope.mapResourceRange(arg, resourceRange);
  }
  SmallVector<ResultReservation> resultReservations;
  for (auto it : llvm::zip(executeOp.results(), executeOp.result_sizes())) {
    auto result = std::get<0>(it);
    auto resultSize = std::get<1>(it);
    auto resultType = result.getType().cast<IREE::Stream::ResourceType>();
    if (handledResults.contains(result)) {
      resultReplacements.push_back(std::make_pair(result, Value{}));
      continue;
    }

    // Find the internal op that defined the result.
    auto yieldValue = yieldOp.operands()[result.getResultNumber()];

    // Early-exit if we are tied to an operand and have storage already.
    auto tiedOperandIndex =
        executeOp.getTiedResultOperandIndex(result.getResultNumber());
    if (tiedOperandIndex.hasValue()) {
      // Already tied; no need to modify just map.
      auto tiedOperand = executeOp.getOperand(tiedOperandIndex.getValue());
      auto arg = entryBlock.getArgument(tiedOperandIndex.getValue());
      LLVM_DEBUG({
        AsmState asmState(executeOp->getParentOp());
        llvm::dbgs() << "  - tying operand ";
        tiedOperand.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << " = arg ";
        arg.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << " = ";
        result.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      scope.mapResourceRange(yieldValue, ResourceRange(arg, resultSize));
      resultReplacements.push_back(std::make_pair(result, tiedOperand));
      continue;
    }

    auto definingValue =
        IREE::Util::TiedOpInterface::findTiedBaseValue(yieldValue);
    auto *definingOp = definingValue.getDefiningOp();
    if (!definingOp) {
      // Directly returning an operand; this usually gets canonicalized away but
      // may be introduced by intermediate transformations.
      auto arg = definingValue.cast<BlockArgument>();
      auto operand = newOperands[arg.getArgNumber()];
      LLVM_DEBUG({
        AsmState asmState(executeOp->getParentOp());
        llvm::dbgs() << "  - passing through operand ";
        operand.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << " = arg ";
        arg.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << " = ";
        result.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      resultReplacements.push_back(std::make_pair(result, operand));
      continue;
    }

    // Queue up the allocation for packing.
    ResultReservation resultReservation = {
        definingOp->getLoc(), result, resultType, resultSize, yieldValue,
    };
    LLVM_DEBUG({
      AsmState asmState(executeOp->getParentOp());
      llvm::dbgs() << "  + queuing pending result reservation allocation for ";
      resultReservation.result.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    });
    resultReservations.push_back(resultReservation);
  }
  auto resultAllocation = reserveResultAllocation(resultReservations);
  for (auto &reservationSet : resultAllocation.reservationSets) {
    // Allocate and tie an operand to the result.
    auto allocOp = externalBuilder.create<IREE::Stream::ResourceAllocOp>(
        externalBuilder.getFusedLoc(reservationSet.reservationLocs),
        reservationSet.reservationTypes, reservationSet.reservationSizes,
        /*uninitialized=*/externalBuilder.getUnitAttr(),
        executeOp.affinityAttr());

    LLVM_DEBUG({
      AsmState asmState(executeOp->getParentOp());
      llvm::dbgs() << "  + alloc for result reservation set: ";
      allocOp.print(llvm::dbgs(), asmState);
      llvm::dbgs() << ":\n";
    });

    for (auto it : llvm::zip(reservationSet.reservations, allocOp.results())) {
      auto &reservation = std::get<0>(it);
      auto allocResult = std::get<1>(it);

      newOperands.push_back(allocResult);
      newOperandSizes.push_back(reservation.resultSize);
      resultReplacements.push_back(
          std::make_pair(reservation.result, allocResult));

      // Insert entry arg for the new operand tied all the way to the yield.
      auto arg =
          entryBlock.addArgument(reservation.resultType, reservation.loc);

      LLVM_DEBUG({
        AsmState asmState(executeOp->getParentOp());
        llvm::dbgs() << "    + adding entry arg for reservation ";
        reservation.result.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "{";
        reservation.resultSize.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "} from ";
        reservation.yieldValue.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << " as ";
        arg.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });

      // Map into scope, updating all aliases.
      auto resourceRange = ResourceRange(arg, reservation.resultSize);
      scope.mapResourceRange(reservation.yieldValue, resourceRange);
    }
  }

  // Allocate local transients that are scoped entirely within the region.
  // All locals are packed into a single slab and reserved as one.
  // Note that not all regions need transients.
  auto transientAllocation =
      allocateLocalTransients(executeOp, scope, externalBuilder);
  if (transientAllocation.hasValue()) {
    auto awaitTimepoint = transientAllocation->awaitTimepoint;
    auto reservation = transientAllocation->reservation;
    auto reservationSize = transientAllocation->reservationSize;
    newAwaitTimepoints.push_back(awaitTimepoint);
    newOperands.push_back(reservation);
    newOperandSizes.push_back(reservationSize);
    pendingReleases.push_back(std::make_pair(reservation, reservationSize));
    LLVM_DEBUG({
      AsmState asmState(executeOp->getParentOp());
      llvm::dbgs() << "  + adding await on transient alloca ";
      awaitTimepoint.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << " -> ";
      reservation.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << "{";
      reservationSize.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << "} captured as ";
      transientAllocation->capturedArg.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    });
  } else {
    LLVM_DEBUG(llvm::dbgs() << "  - no local transients found\n");
  }

  // If we have any waits then we attach them to the execution region.
  OpBuilder executeBuilder(executeOp);
  Value newAwaitTimepoint;
  if (newAwaitTimepoints.size() == 1) {
    newAwaitTimepoint = newAwaitTimepoints.front();
  } else if (newAwaitTimepoints.size() > 1) {
    newAwaitTimepoint =
        executeBuilder.createOrFold<IREE::Stream::TimepointJoinOp>(
            executeOp.getLoc(), newAwaitTimepoints.front().getType(),
            newAwaitTimepoints);
  }

  // Recreate the execution op with all the new arguments. Note that we drop
  // the results (besides the timepoint) as they are all aliased.
  auto newExecuteOp = executeBuilder.create<IREE::Stream::CmdExecuteOp>(
      executeOp.getLoc(), newAwaitTimepoint, newOperands, newOperandSizes);
  if (executeOp.affinity().hasValue()) {
    newExecuteOp.affinityAttr(executeOp.affinityAttr());
  }
  newExecuteOp.body().takeBody(executeOp.body());
  executeOp.result_timepoint().replaceAllUsesWith(
      newExecuteOp.result_timepoint());
  for (auto replacement : resultReplacements) {
    if (!replacement.second) continue;  // handled already
    replacement.first.replaceAllUsesWith(replacement.second);
  }
  scope.replaceRootOp(newExecuteOp);

  // Drop the operands on the yield op now that all are aliased.
  OpBuilder(yieldOp).create<IREE::Stream::YieldOp>(yieldOp.getLoc());
  yieldOp.erase();

  // Apply mappings from the parent execute op into all waves; as we allocate
  // and convert to stream.cmd.concurrent we are removing all wave
  // operands/results and taking whatever we established above as the true
  // ranges.
  newExecuteOp.body().walk<WalkOrder::PreOrder>(
      [&](IREE::Stream::AsyncConcurrentOp concurrentOp) {
        for (auto it : llvm::zip(concurrentOp.operands(),
                                 concurrentOp.body().getArguments())) {
          auto outerValue = std::get<0>(it);
          auto innerValue = std::get<1>(it);
          LLVM_DEBUG({
            AsmState asmState(newExecuteOp->getParentOp());
            llvm::dbgs() << "  = shady alias of wave operand ";
            outerValue.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << " to wave arg ";
            innerValue.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
          scope.mapResourceRange(innerValue,
                                 scope.lookupResourceRange(outerValue));
        }
        auto yieldOp = cast<IREE::Stream::YieldOp>(
            concurrentOp.body().front().getTerminator());
        for (auto it : llvm::zip(yieldOp.operands(), concurrentOp.results())) {
          auto innerValue = std::get<0>(it);
          auto outerValue = std::get<1>(it);
          LLVM_DEBUG({
            AsmState asmState(newExecuteOp->getParentOp());
            llvm::dbgs() << "  = shady alias of wave result ";
            innerValue.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << " to stream value ";
            outerValue.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
          scope.mapResourceRange(innerValue,
                                 scope.lookupResourceRange(outerValue));
        }
      });

  // Apply the scope to the region and convert ops.
  if (failed(applyAsyncAllocations(newExecuteOp.body(), scope))) {
    return newExecuteOp.emitError()
           << "failed to apply allocations/issue commands";
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Allocation of execution region complete:\n";
    newExecuteOp.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  OpBuilder builder(newExecuteOp);
  builder.setInsertionPointAfter(newExecuteOp);

  // Insert transient deallocations.
  SetVector<Operation *> executeTimepointUsers;
  for (auto &release : pendingReleases) {
    auto reservation = release.first;
    auto reservationSize = release.second;
    auto deallocaOp = builder.create<IREE::Stream::ResourceDeallocaOp>(
        reservation.getLoc(), reservation, reservationSize,
        newExecuteOp.result_timepoint(), newExecuteOp.affinityAttr());
    executeTimepointUsers.insert(deallocaOp);
  }

  // If we have any timepoints that we need to join with we do that now such
  // that the original timepoint dependency chain is preserved even if we make
  // local changes here.
  if (!joinTimepoints.empty()) {
    joinTimepoints.push_back(newExecuteOp.result_timepoint());
    auto fusedLoc = builder.getFusedLoc(llvm::to_vector<4>(llvm::map_range(
        joinTimepoints, [](auto timepoint) { return timepoint.getLoc(); })));
    auto joinOp = builder.create<IREE::Stream::TimepointJoinOp>(
        fusedLoc, newExecuteOp.result_timepoint().getType(), joinTimepoints);
    executeTimepointUsers.insert(joinOp);
    newExecuteOp.result_timepoint().replaceUsesWithIf(
        joinOp.result(), [&](OpOperand &operand) {
          return !executeTimepointUsers.contains(operand.getOwner());
        });
  }

  return success();
}

static LogicalResult convertAsyncLoadOp(IREE::Stream::AsyncLoadOp asyncOp) {
  auto newOp = OpBuilder(asyncOp).create<IREE::Stream::ResourceLoadOp>(
      asyncOp.getLoc(), asyncOp.result().getType(), asyncOp.source(),
      asyncOp.source_size(), asyncOp.source_offset());
  asyncOp.replaceAllUsesWith(newOp.result());
  asyncOp.erase();
  return success();
}

static LogicalResult convertAsyncStoreOp(IREE::Stream::AsyncStoreOp asyncOp) {
  auto newOp = OpBuilder(asyncOp).create<IREE::Stream::ResourceStoreOp>(
      asyncOp.getLoc(), asyncOp.target(), asyncOp.target_size(),
      asyncOp.target_offset(), asyncOp.value());
  asyncOp.replaceAllUsesWith(newOp.target());
  asyncOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// -iree-stream-schedule-allocation
//===----------------------------------------------------------------------===//

class ScheduleAllocationPass
    : public ScheduleAllocationBase<ScheduleAllocationPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::arith::ArithmeticDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto parentOp = dyn_cast<CallableOpInterface>(getOperation());
    if (!parentOp || !parentOp.getCallableRegion() ||
        parentOp.getCallableRegion()->empty()) {
      return;
    }

    for (auto &op :
         llvm::make_early_inc_range(parentOp.getCallableRegion()->getOps())) {
      if (failed(TypeSwitch<Operation *, LogicalResult>(&op)
                     .Case([&](IREE::Stream::AsyncExecuteOp op) {
                       return allocateExecutionRegion(op);
                     })
                     .Case([&](IREE::Stream::AsyncLoadOp op) {
                       return convertAsyncLoadOp(op);
                     })
                     .Case([&](IREE::Stream::AsyncStoreOp op) {
                       return convertAsyncStoreOp(op);
                     })
                     .Default(success()))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<>> createScheduleAllocationPass() {
  return std::make_unique<ScheduleAllocationPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
