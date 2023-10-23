// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
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

// Disjoint-set data structure holding non-overlapping sets of aliasing values.
class ValueAliasingSet {
public:
  void addAlias(Value aliasee, Value aliaser) {
    auto aliaseeWithId = getWithId(aliasee);
    auto aliaserWithId = getWithId(aliaser);
    valueAliasing.unionSets(aliaseeWithId, aliaserWithId);
  }

  SmallVector<SmallVector<Value>> getValueAliasSets() const {
    SmallVector<SmallVector<Value>> result;
    for (auto it = valueAliasing.begin(); it != valueAliasing.end(); ++it) {
      if (!it->isLeader())
        continue; // Ignore non-leader sets.
      auto &aliasSet = result.emplace_back();
      for (auto mi = valueAliasing.member_begin(it);
           mi != valueAliasing.member_end(); ++mi) {
        aliasSet.push_back(mi->value);
      }
    }
    return result;
  }

  auto getValueAliases(Value value) const {
    return llvm::make_filter_range(
        llvm::map_range(
            llvm::make_range(valueAliasing.findLeader(getWithId(value)),
                             valueAliasing.member_end()),
            NumberedValue::getValue),
        [=](Value aliaser) { return aliaser != value; });
  }

private:
  // EquivalenceClasses require ordering for value type to return deterministic
  // results, so we provide it by assigning id to all values added to the set.
  struct NumberedValue {
    Value value;
    int64_t id;

    static Value getValue(const NumberedValue &value) { return value.value; }
  };

  struct Comparator {
    int operator()(const NumberedValue &a, const NumberedValue &b) const {
      return a.id < b.id;
    }
  };

  NumberedValue getWithId(Value value) const {
    auto [iterator, inserted] = id.try_emplace(value, id.size());
    return {value, iterator->second};
  }

  mutable llvm::DenseMap<Value, int64_t> id;
  llvm::EquivalenceClasses<NumberedValue, Comparator> valueAliasing;
};

// Builds a map of value aliases from aliasee to a set of aliasers.
// Only values that alias will be present in the map. The map may contain
// values nested within the |regionOp|.
static void computeRegionValueAliases(Operation *regionOp,
                                      ValueAliasingSet &valueAliases) {
  auto *block = &regionOp->getRegion(0).front();

  // Filter out to only resource results - some regions may return additional
  // things like stream.async.execute returning a timepoint.
  auto resourceResults = llvm::to_vector_of<OpResult>(
      llvm::make_filter_range(regionOp->getResults(), [](OpResult result) {
        return llvm::isa<IREE::Stream::ResourceType>(result.getType());
      }));

  // Start with outputs so that we handle tied values that may lead all the way
  // back up the chain to the stream inputs.
  auto tiedStreamOp = cast<IREE::Util::TiedOpInterface>(regionOp);
  auto yieldOp = cast<IREE::Stream::YieldOp>(block->getTerminator());
  for (auto [outerResult, innerResult] :
       llvm::zip_equal(resourceResults, yieldOp.getResourceOperands())) {
    auto tiedOperandIndex =
        tiedStreamOp.getTiedResultOperandIndex(outerResult.getResultNumber());
    if (tiedOperandIndex.has_value()) {
      auto arg = block->getArgument(tiedOperandIndex.value());
      valueAliases.addAlias(innerResult, arg);
    }
  }

  for (auto &op : *block) {
    // Recurse into regions to build the alias map. This will introduce aliases
    // that are outside of the parent block scope but they should only be
    // present for in-place operations. If we want to allocate within the nested
    // scopes we'll need to fix up the subsequent liveness analysis code to
    // handle ops in differing scopes by walking parent chains.
    if (isa<mlir::RegionBranchOpInterface>(op)) {
      computeRegionValueAliases(&op, valueAliases);
    }

    // Tied results reuse their operand buffer.
    auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
    for (auto result : op.getResults()) {
      if (!llvm::isa<IREE::Stream::ResourceType>(result.getType()))
        continue;
      if (tiedOp) {
        auto tiedOperand = tiedOp.getTiedResultOperand(result);
        if (tiedOperand) {
          valueAliases.addAlias(result, tiedOperand);
        }
      }
    }
  }
}

// Builds a set of aliasing sets. Only values that alias will be present in the
// set. The set may contain values nested within the |executeOp|.
static ValueAliasingSet
computeExecutionRegionValueAliases(IREE::Stream::AsyncExecuteOp executeOp) {
  ValueAliasingSet valueAliases;
  computeRegionValueAliases(executeOp, valueAliases);
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
  int ordinal = -1; // unique per value
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
static LivenessIntervalList
computeExecutionRegionLivenessIntervals(IREE::Stream::AsyncExecuteOp executeOp,
                                        const ValueAliasingSet &valueAliases) {
  // Perform a liveness analysis on the execution region.
  // Fragments have a single block and as such the live-in/live-out block
  // information derived here applies to the entire stream region.
  assert(executeOp.getBody().getBlocks().size() == 1);
  auto *streamBlock = &executeOp.getBody().front();
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
  for (auto returnValue : yieldOp.getResourceOperands()) {
    if (!llvm::isa<IREE::Stream::ResourceType>(returnValue.getType()))
      continue;
    liveOuts.insert(returnValue);
  }

  // Compute live-in intervals by hand as we won't catch them in the op walk
  // below since they are block arguments.
  LivenessIntervalMap valueIntervals;
  int ordinal = 0;
  for (Value value : streamBlock->getArguments()) {
    if (!llvm::isa<IREE::Stream::ResourceType>(value.getType()))
      continue;
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
    if (auto concurrentOp = dyn_cast<IREE::Stream::AsyncConcurrentOp>(op)) {
      // HACK: allocation planning currently only works on the top-level
      // execute op but sometimes we need to allocate locals inside of
      // concurrent regions. The real fix here is to make allocation planning
      // handle arbitrary nesting but for now we do a quick walk through the
      // regions to see if there are any locals that need to be marked live for
      // the duration of the region.
      concurrentOp.walk([&](Operation *op) {
        for (auto value : op->getResults()) {
          if (!llvm::isa<IREE::Stream::ResourceType>(value.getType()))
            continue;
          if (!value.use_empty())
            continue;
          LivenessInterval interval;
          interval.start = start;
          interval.end = start;
          interval.value = value;
          interval.ordinal = -1;
          valueIntervals[value] = interval;
        }
      });
    }
    for (auto value : op.getResults()) {
      if (!llvm::isa<IREE::Stream::ResourceType>(value.getType()))
        continue;
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

  // Walk the alias set and union intervals and propagate back.
  for (auto &aliasSet : valueAliases.getValueAliasSets()) {

    // Return true if value is nested  somewhere within the current scope.
    auto isNested = [&](Value value) -> bool {
      return valueIntervals[value].ordinal == -1;
    };

    // We'd need to update this analysis to handle the nesting in order to
    // compute the ranges here but that's not (currently) required as all
    // allocated values roll up to the parent scope by way of the yields.
    if (llvm::all_of(aliasSet, isNested))
      continue;

    assert((llvm::all_of(aliasSet, isNested) ||
            llvm::none_of(aliasSet, isNested)) &&
           "nested values can't alias values in the current scope");

    auto &aliaseeInterval = valueIntervals[aliasSet.front()];
    int start = aliaseeInterval.start;
    int end = aliaseeInterval.end;

    auto aliasers = ArrayRef(aliasSet).drop_front(1);
    for (Value aliaser : aliasers) {
      auto &aliaserInterval = valueIntervals[aliaser];
      start = std::min(start, aliaserInterval.start);
      end = std::max(end, aliaserInterval.end);
    }

    // Propagate interval back to all values in the aliasing set.
    for (Value aliaser : aliasSet) {
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
    // Filter out values we couldn't analyze.
    if (it.second.value) {
      sortedIntervals.push_back(it.second);
    }
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
      : resource(resource), resourceSize(resourceSize), offset(offset),
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

static std::unique_ptr<AsmState> getRootAsmState(Operation *rootOp) {
  LLVM_DEBUG(return std::make_unique<AsmState>(rootOp));
  return nullptr;
}

struct AllocationScope {
  explicit AllocationScope(IREE::Stream::AsyncExecuteOp rootAsyncOp)
      : rootOp(rootAsyncOp),
        valueAliases(computeExecutionRegionValueAliases(rootAsyncOp)) {}

  // Execution region being allocated.
  Operation *getRootOp() const { return rootOp; }

  // Aliasing set for the entire root op, indicating which values are tied.
  const ValueAliasingSet &getValueAliases() const { return valueAliases; }

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
    if (it != indexConstantMap.end())
      return it->second;
    auto constantValue = OpBuilder(rootOp).createOrFold<arith::ConstantIndexOp>(
        rootOp->getLoc(), value);
    indexConstantMap.insert(std::make_pair(value, constantValue));
    return constantValue;
  }

  // Performs a memoized add (as many adds of offsets or lengths are redundant).
  Value add(Location loc, Value lhs, Value rhs) {
    // TODO(benvanik): memoize - if worth it. Needs profiling.
    if (matchPattern(lhs, m_Zero()))
      return rhs;
    if (matchPattern(rhs, m_Zero()))
      return lhs;
    auto result = OpBuilder(rootOp).createOrFold<arith::AddIOp>(loc, lhs, rhs);
    return result;
  }

  // Maps |resource| to the storage range defined by |resourceRange|.
  // All aliases of |resource| will also be mapped.
  void mapResourceRange(Value resource, ResourceRange resourceRange,
                        AsmState *asmState) {
    if (resourceRangeMap.count(resource))
      return;

    if (!resourceRange.offset && !resourceRange.length) {
      resourceRange.offset = lookupOrCreateIndex(0);
      resourceRange.length = resourceRange.resourceSize;
    }

    resourceRangeMap.insert(std::make_pair(resource, resourceRange));
    LLVM_DEBUG({
      llvm::dbgs() << " -> mapping ";
      resource.printAsOperand(llvm::dbgs(), *asmState);
      llvm::dbgs() << " = ";
      resourceRange.print(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    });

    // Propagate alias subranges when present.
    for (auto alias : valueAliases.getValueAliases(resource)) {
      ResourceRange aliasRange = resourceRange;
      if (auto subviewOp =
              IREE::Stream::ResourceSubviewOp::findSubviewOp(alias)) {
        aliasRange.resourceSize = subviewOp.getSubrangeResource();
        aliasRange.resourceSize = subviewOp.getSubrangeResourceSize();
        aliasRange.offset = subviewOp.getSubrangeOffset();
        aliasRange.length = subviewOp.getSubrangeLength();
      }
      resourceRangeMap.insert(std::make_pair(alias, aliasRange));
      LLVM_DEBUG({
        llvm::dbgs() << "   = alias ";
        alias.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " = ";
        aliasRange.print(llvm::dbgs(), *asmState);
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
        llvm::dbgs() << "\n";
        resource.getDefiningOp()->print(llvm::dbgs(), asmState);
        llvm::dbgs() << "\ncurrent mappings:\n";
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
    for (Value alias : valueAliases.getValueAliases(resource)) {
      callback(alias);
    }
  }

private:
  Operation *rootOp;

  // Disjoint-set of values that alias each other. That two things alias does
  // not imply the values can be treated as equivalent: some values may be
  // subranges of others.
  ValueAliasingSet valueAliases;

  // Index value -> std.constant index value.
  DenseMap<int64_t, Value> indexConstantMap;

  // Maps resource values inside the stream to a storage range.
  DenseMap<Value, ResourceRange> resourceRangeMap;
};

static LogicalResult
applyResourceSubviewOp(IREE::Stream::ResourceSubviewOp asyncOp,
                       AllocationScope &scope, OpBuilder builder) {
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
  auto targetRange = scope.lookupResourceRange(asyncOp.getResult());
  builder.create<IREE::Stream::CmdFillOp>(
      asyncOp.getLoc(), targetRange.resource, targetRange.resourceSize,
      targetRange.offset, targetRange.length, asyncOp.getValue());
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncCloneOp(IREE::Stream::AsyncCloneOp asyncOp,
                                       AllocationScope &scope,
                                       OpBuilder builder) {
  auto sourceRange = scope.lookupResourceRange(asyncOp.getSource());
  auto targetRange = scope.lookupResourceRange(asyncOp.getResult());
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
  auto sourceRange = scope.lookupResourceRange(asyncOp.getSource());
  auto sourceOffset = scope.add(asyncOp.getLoc(), sourceRange.offset,
                                asyncOp.getSourceOffset());
  auto targetRange = scope.lookupResourceRange(asyncOp.getResult());
  builder.create<IREE::Stream::CmdCopyOp>(
      asyncOp.getLoc(), sourceRange.resource, sourceRange.resourceSize,
      sourceOffset, targetRange.resource, targetRange.resourceSize,
      targetRange.offset, asyncOp.getResultSize());
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncFillOp(IREE::Stream::AsyncFillOp asyncOp,
                                      AllocationScope &scope,
                                      OpBuilder builder) {
  auto targetRange = scope.lookupResourceRange(asyncOp.getResult());
  auto targetOffset = scope.add(asyncOp.getLoc(), targetRange.offset,
                                asyncOp.getTargetOffset());
  builder.create<IREE::Stream::CmdFillOp>(
      asyncOp.getLoc(), targetRange.resource, targetRange.resourceSize,
      targetOffset, asyncOp.getTargetLength(), asyncOp.getValue());
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncUpdateOp(IREE::Stream::AsyncUpdateOp asyncOp,
                                        AllocationScope &scope,
                                        OpBuilder builder) {
  auto sourceRange = scope.lookupResourceRange(asyncOp.getUpdate());
  auto sourceOffset = sourceRange.offset;
  auto targetRange = scope.lookupResourceRange(asyncOp.getResult());
  auto targetOffset = scope.add(asyncOp.getLoc(), targetRange.offset,
                                asyncOp.getTargetOffset());
  builder.create<IREE::Stream::CmdCopyOp>(
      asyncOp.getLoc(), sourceRange.resource, sourceRange.resourceSize,
      sourceOffset, targetRange.resource, targetRange.resourceSize,
      targetOffset, asyncOp.getUpdateSize());
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncCopyOp(IREE::Stream::AsyncCopyOp asyncOp,
                                      AllocationScope &scope,
                                      OpBuilder builder) {
  auto sourceRange = scope.lookupResourceRange(asyncOp.getSource());
  auto sourceOffset = scope.add(asyncOp.getLoc(), sourceRange.offset,
                                asyncOp.getSourceOffset());
  auto targetRange = scope.lookupResourceRange(asyncOp.getResult());
  auto targetOffset = scope.add(asyncOp.getLoc(), targetRange.offset,
                                asyncOp.getTargetOffset());
  builder.create<IREE::Stream::CmdCopyOp>(
      asyncOp.getLoc(), sourceRange.resource, sourceRange.resourceSize,
      sourceOffset, targetRange.resource, targetRange.resourceSize,
      targetOffset, asyncOp.getLength());
  asyncOp.erase();
  return success();
}

static LogicalResult
applyAsyncCollectiveOp(IREE::Stream::AsyncCollectiveOp asyncOp,
                       AllocationScope &scope, OpBuilder builder) {
  SmallVector<Value> newResources;
  SmallVector<Value> newResourceSizes;
  SmallVector<Value> newResourceOffsets;
  SmallVector<Value> newResourceLengths;
  SmallVector<Attribute> newResourceAccesses;

  // TODO(#11249): support in-place collectives by a r/w resource? it may be
  // fine to leave them separate as then we get unique invalidation ranges.

  auto sourceRange = scope.lookupResourceRange(asyncOp.getSource());
  auto sourceOffset = scope.add(asyncOp.getLoc(), sourceRange.offset,
                                asyncOp.getSourceOffset());
  auto sourceLength = sourceRange.length;
  newResources.push_back(sourceRange.resource);
  newResourceSizes.push_back(sourceRange.resourceSize);
  newResourceOffsets.push_back(sourceOffset);
  newResourceLengths.push_back(sourceLength);
  newResourceAccesses.push_back(IREE::Stream::ResourceAccessBitfieldAttr::get(
      builder.getContext(), IREE::Stream::ResourceAccessBitfield::Read));

  auto targetRange = scope.lookupResourceRange(asyncOp.getResult());
  auto targetOffset = scope.add(asyncOp.getLoc(), targetRange.offset,
                                asyncOp.getTargetOffset());
  auto targetLength = targetRange.length;
  newResources.push_back(targetRange.resource);
  newResourceSizes.push_back(targetRange.resourceSize);
  newResourceOffsets.push_back(targetOffset);
  newResourceLengths.push_back(targetLength);
  newResourceAccesses.push_back(IREE::Stream::ResourceAccessBitfieldAttr::get(
      builder.getContext(), IREE::Stream::ResourceAccessBitfield::Write));

  builder.create<IREE::Stream::CmdCollectiveOp>(
      asyncOp.getLoc(), asyncOp.getOp(), asyncOp.getChannel(),
      asyncOp.getElementCount(), asyncOp.getParam(), newResources,
      newResourceSizes, newResourceOffsets, newResourceLengths,
      builder.getArrayAttr(newResourceAccesses));
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncTransferOp(IREE::Stream::AsyncTransferOp asyncOp,
                                          AllocationScope &scope,
                                          OpBuilder builder) {
  // Lookup the affinity for where we are executing. This lets us determine if
  // this transfer is incoming or outgoing.
  auto isStaging = [](Value value) {
    return llvm::cast<IREE::Stream::ResourceType>(value.getType())
               .getLifetime() == IREE::Stream::Lifetime::Staging;
  };
  auto currentAffinityAttr = IREE::Stream::AffinityAttr::lookup(asyncOp);
  bool transferIn = asyncOp.getSourceAffinityAttr() != currentAffinityAttr ||
                    isStaging(asyncOp.getSource());
  bool transferOut = asyncOp.getResultAffinityAttr() != currentAffinityAttr ||
                     isStaging(asyncOp.getResult());

  auto sourceRange = scope.lookupResourceRange(asyncOp.getSource());
  auto targetRange = scope.lookupResourceRange(asyncOp.getResult());

  // Incoming transfers need invalidation.
  if (transferIn) {
    builder.create<IREE::Stream::CmdInvalidateOp>(
        asyncOp.getLoc(), sourceRange.resource, sourceRange.resourceSize,
        sourceRange.offset, sourceRange.length,
        asyncOp.getSourceAffinityAttr());
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
        targetRange.offset, targetRange.length,
        asyncOp.getResultAffinityAttr());
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

  unsigned resourceIndex = 0;
  for (auto it : llvm::enumerate(asyncOp.getResourceOperands())) {
    auto operand = it.value();
    if (!llvm::isa<IREE::Stream::ResourceType>(operand.getType())) {
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
    auto resourceOffset =
        scope.add(asyncOp.getLoc(), resourceRange.offset,
                  asyncOp.getResourceOperandOffsets()[resourceIndex]);
    auto resourceLength = asyncOp.getResourceOperandLengths()[resourceIndex];
    auto resourceAccess = IREE::Stream::ResourceAccessBitfieldAttr::get(
        builder.getContext(), accessBits);
    newResources.push_back(resourceRange.resource);
    newResourceSizes.push_back(resourceRange.resourceSize);
    newResourceOffsets.push_back(resourceOffset);
    newResourceLengths.push_back(resourceLength);
    newResourceAccesses.push_back(resourceAccess);
    ++resourceIndex;
  }

  for (auto result : asyncOp.getResults()) {
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

  auto newOp = builder.create<IREE::Stream::CmdDispatchOp>(
      asyncOp.getLoc(), asyncOp.getWorkload(),
      builder.getArrayAttr({asyncOp.getEntryPoint()}), newOperands,
      newResources, newResourceSizes, newResourceOffsets, newResourceLengths,
      builder.getArrayAttr(newResourceAccesses));
  newOp->setDialectAttrs(asyncOp->getDialectAttrs());
  asyncOp.erase();
  return success();
}

// Converts an stream.async.func op into a stream.cmd.func op.
// Result resources are appended to the end of the operands and resources each
// get an offset and length of their subrange.
static void convertAsyncFuncOp(IREE::Stream::AsyncFuncOp asyncOp) {
  auto indexType = IndexType::get(asyncOp.getContext());
  auto oldFunctionType = asyncOp.getFunctionType();

  SmallVector<Type> newInputs;
  SmallVector<DictionaryAttr> newArgAttrs;
  for (auto [i, oldInput] : llvm::enumerate(oldFunctionType.getInputs())) {
    auto oldArgAttr = asyncOp.getArgAttrDict(i);
    if (llvm::isa<IREE::Stream::ResourceType>(oldInput)) {
      newInputs.push_back(oldInput); // resource
      newArgAttrs.push_back(oldArgAttr);
      newInputs.push_back(indexType); // offset
      newArgAttrs.push_back(nullptr);
      newInputs.push_back(indexType); // length
      newArgAttrs.push_back(nullptr);
    } else {
      newInputs.push_back(oldInput);
      newArgAttrs.push_back(oldArgAttr);
    }
  }

  SmallVector<Type> newResults;
  SmallVector<DictionaryAttr> newResultAttrs;
  for (auto [i, oldResult] : llvm::enumerate(oldFunctionType.getResults())) {
    auto oldResultAttr = asyncOp.getResultAttrDict(i);
    if (llvm::isa<IREE::Stream::ResourceType>(oldResult)) {
      if (asyncOp.isResultTied(i)) {
        // Tied results reuse the operands they are tied to.
        continue;
      }
      newInputs.push_back(oldResult); // resource
      newArgAttrs.push_back(oldResultAttr);
      newInputs.push_back(indexType); // offset
      newArgAttrs.push_back(nullptr);
      newInputs.push_back(indexType); // length
      newArgAttrs.push_back(nullptr);
    } else {
      newResults.push_back(oldResult);
      newResultAttrs.push_back(oldResultAttr);
    }
  }

  auto newFunctionType =
      FunctionType::get(asyncOp.getContext(), newInputs, newResults);

  OpBuilder builder(asyncOp);
  auto cmdOp = builder.create<IREE::Stream::CmdFuncOp>(
      asyncOp.getLoc(), asyncOp.getName(), newFunctionType, newArgAttrs,
      newResultAttrs);
  cmdOp->setDialectAttrs(asyncOp->getDialectAttrs());
  asyncOp.erase();
}

static LogicalResult applyAsyncCallOp(IREE::Stream::AsyncCallOp asyncOp,
                                      AllocationScope &scope,
                                      OpBuilder builder) {
  SmallVector<Value> newResourceOperands;
  SmallVector<Value> newResourceSizes;
  SmallVector<Value> newResourceOffsets;
  SmallVector<Value> newResourceLengths;
  SmallVector<Attribute> newResourceAccesses;
  SmallVector<Type> newResultTypes;

  unsigned resourceIndex = 0;
  for (auto [i, operand] : llvm::enumerate(asyncOp.getResourceOperands())) {
    if (!llvm::isa<IREE::Stream::ResourceType>(operand.getType())) {
      // Primitive operand.
      newResourceOperands.push_back(operand);
      continue;
    }

    // Read-only or read-write. Write-only are untied results below.
    unsigned operandIdx = asyncOp.getTiedOperandsIndexAndLength().first + i;
    auto accessBits = IREE::Stream::ResourceAccessBitfield::Read;
    if (asyncOp.isOperandTied(operandIdx)) {
      accessBits = accessBits | IREE::Stream::ResourceAccessBitfield::Write;
    }

    auto resourceRange = scope.lookupResourceRange(operand);
    auto resourceOffset =
        scope.add(asyncOp.getLoc(), resourceRange.offset,
                  asyncOp.getResourceOperandOffsets()[resourceIndex]);
    auto resourceLength = asyncOp.getResourceOperandLengths()[resourceIndex];
    auto resourceAccess = IREE::Stream::ResourceAccessBitfieldAttr::get(
        builder.getContext(), accessBits);
    newResourceOperands.push_back(resourceRange.resource);
    newResourceSizes.push_back(resourceRange.resourceSize);
    newResourceOffsets.push_back(resourceOffset);
    newResourceLengths.push_back(resourceLength);
    newResourceAccesses.push_back(resourceAccess);
    ++resourceIndex;
  }

  for (auto result : asyncOp.getResults()) {
    if (!llvm::isa<IREE::Stream::ResourceType>(result.getType())) {
      // Primitive result.
      newResultTypes.push_back(result.getType());
      continue;
    }

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
    newResourceOperands.push_back(resourceRange.resource);
    newResourceSizes.push_back(resourceRange.resourceSize);
    newResourceOffsets.push_back(resourceOffset);
    newResourceLengths.push_back(resourceLength);
    newResourceAccesses.push_back(resourceAccess);
  }

  auto newOp = builder.create<IREE::Stream::CmdCallOp>(
      asyncOp.getLoc(), newResultTypes, asyncOp.getCalleeAttr(),
      newResourceOperands, newResourceSizes, newResourceOffsets,
      newResourceLengths,
      /*result_sizes=*/ValueRange{},
      /*tied_operands=*/nullptr, builder.getArrayAttr(newResourceAccesses));
  newOp->setDialectAttrs(asyncOp->getDialectAttrs());
  asyncOp.erase();
  return success();
}

static LogicalResult applyAsyncAllocations(Region &region,
                                           AllocationScope &scope);

static LogicalResult
applyAsyncConcurrentOp(IREE::Stream::AsyncConcurrentOp asyncOp,
                       AllocationScope &scope, OpBuilder builder) {
  // Remove operands from the yield now that we aren't returning anything.
  // Must do this before we recurse so that the ops we are transforming have no
  // uses.
  auto yieldOp =
      cast<IREE::Stream::YieldOp>(asyncOp.getBody().front().getTerminator());
  yieldOp->eraseOperands(0, yieldOp.getNumOperands());

  // Recurse into the wave ops to process the async ops within.
  // Resources are captured inside of the region and need to be mapped back to
  // the parent scope where they will be captured after lowering into
  // stream.cmd.execute.
  if (failed(applyAsyncAllocations(asyncOp.getBody(), scope))) {
    return failure();
  }

  // Explicit variant has no args/operands
  auto &block = asyncOp.getBody().front();
  block.eraseArguments([&](auto arg) { return true; });

  // Rewrite wave op to remove results.
  auto newOp = builder.create<IREE::Stream::CmdConcurrentOp>(asyncOp.getLoc());
  newOp.getBody().takeBody(asyncOp.getBody());
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
  auto ops = llvm::map_to_vector(llvm::reverse(block),
                                 [&](Operation &op) { return &op; });
  for (auto *op : ops) {
    if (op->hasTrait<OpTrait::IsTerminator>())
      continue;
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
                   .Case([&](IREE::Stream::AsyncCollectiveOp op) {
                     return applyAsyncCollectiveOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncTransferOp op) {
                     return applyAsyncTransferOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncDispatchOp op) {
                     return applyAsyncDispatchOp(op, scope, OpBuilder(op));
                   })
                   .Case([&](IREE::Stream::AsyncCallOp op) {
                     return applyAsyncCallOp(op, scope, OpBuilder(op));
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
static std::optional<TransientAllocation>
allocateLocalTransients(IREE::Stream::AsyncExecuteOp executeOp,
                        AllocationScope &scope, OpBuilder &externalBuilder) {
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
    assert(value && "must have value for interval");
    auto valueType =
        llvm::dyn_cast<IREE::Stream::ResourceType>(value.getType());
    if (!valueType)
      continue;

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
    return std::nullopt;
  }

  // Insert the stream.resource.pack op to compute the slices and total size of
  // the transient allocation required.
  auto fusedLoc = externalBuilder.getFusedLoc(locs);
  auto indexType = externalBuilder.getIndexType();
  SmallVector<Type> packedOffsetTypes(dynamicSliceSizes.size(), indexType);
  auto packOp = externalBuilder.create<IREE::Stream::ResourcePackOp>(
      fusedLoc, indexType, packedOffsetTypes,
      /*offset=*/nullptr, externalBuilder.getIndexArrayAttr(lifetimeIntervals),
      dynamicSliceSizes, executeOp.getAffinityAttr());

  // Allocate the transient storage for the entire packed slab.
  auto transientType = externalBuilder.getType<IREE::Stream::ResourceType>(
      IREE::Stream::Lifetime::Transient);
  auto timepointType = externalBuilder.getType<IREE::Stream::TimepointType>();
  auto allocaOp = externalBuilder.create<IREE::Stream::ResourceAllocaOp>(
      fusedLoc, transientType, timepointType, packOp.getTotalLength(),
      executeOp.getAwaitTimepoint(), executeOp.getAffinityAttr());
  TransientAllocation allocation;
  allocation.awaitTimepoint = allocaOp.getResultTimepoint();
  allocation.reservation = allocaOp.getResult();
  allocation.reservationSize = allocaOp.getStorageSize();
  allocation.capturedArg = executeOp.getBody().front().addArgument(
      allocation.reservation.getType(), allocation.reservation.getLoc());

  // Map values to their ranges within the slab.
  auto asmState = getRootAsmState(executeOp->getParentOp());
  for (size_t i = 0; i < transientValues.size(); ++i) {
    auto value = transientValues[i];
    auto offset = packOp.getPackedOffsets()[i];
    auto length = packOp.getDynamicSliceSizes()[i];
    auto resourceRange = ResourceRange(
        allocation.capturedArg, allocation.reservationSize, offset, length);
    scope.mapResourceRange(value, resourceRange, asmState.get());
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
    if (!isa<IREE::Stream::YieldOp>(user))
      return false;
  }
  return true;
}

// Extracts stream.async.constant ops from |executeOp| into their own dedicated
// stream.resource.constants upload op. The uploaded constants will be captured
// by the region for use within as if they had still existed in there.
static std::optional<ConstantAllocation>
extractConstants(IREE::Stream::AsyncExecuteOp executeOp,
                 OpBuilder &externalBuilder) {
  // Gather all constant ops from the region, if any.
  auto constantOps =
      llvm::to_vector(executeOp.getOps<IREE::Stream::AsyncConstantOp>());
  if (constantOps.empty())
    return std::nullopt;

  // Allocate a new constant upload op and insert a subview for each constant.
  SmallVector<Location> locs;
  SmallVector<Type> resultTypes;
  SmallVector<Attribute> initialValues;
  SmallVector<Value> resultSizes;
  for (auto constantOp : constantOps) {
    locs.push_back(constantOp.getLoc());
    resultTypes.push_back(constantOp.getResult().getType());
    initialValues.push_back(constantOp.getValue());
    resultSizes.push_back(constantOp.getResultSize());
  }
  auto timepointType = externalBuilder.getType<IREE::Stream::TimepointType>();
  ConstantAllocation allocation;
  allocation.constantsOp =
      externalBuilder.create<IREE::Stream::ResourceConstantsOp>(
          externalBuilder.getFusedLoc(locs), resultTypes, timepointType,
          externalBuilder.getArrayAttr(initialValues), resultSizes,
          executeOp.getAffinityAttr());

  // Remap original constants to reservations.
  auto &entryBlock = executeOp.getBody().front();
  for (auto it : llvm::enumerate(constantOps)) {
    unsigned idx = static_cast<unsigned>(it.index());
    auto constantOp = it.value();
    ConstantReservation reservation;
    reservation.constantOp = constantOp;

    // Grab the returned resource from the upload op. This will later likely
    // be subviewed but all we know here is what the subview result will be.
    reservation.resource = allocation.constantsOp.getResults()[idx];
    reservation.resourceSize = allocation.constantsOp.getResultSizes()[idx];

    // Capture the subview resource and switch all uses of the internal op if
    // there are any. Otherwise, if it's just yield, we can avoid the capture
    // and implicit dependency.
    if (!constantOp.use_empty() && !isOnlyUseYield(constantOp.getResult())) {
      reservation.capturedArg = entryBlock.addArgument(
          constantOp.getResult().getType(), constantOp.getLoc());
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
static ResultAllocation
reserveResultAllocation(ArrayRef<ResultReservation> reservations) {
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
  regionOp.getSuccessorRegions(regionOp->getRegion(0), regions);
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

// Returns a reversed list of subrange operations that lead from an initial
// resource down a sequence to |derivedValue|. The first element in the list
// will be the last subview of |derivedValue| and the last element will be the
// first subview.
static SmallVector<IREE::Util::SubrangeOperand>
gatherSubranges(Value derivedValue) {
  SmallVector<IREE::Util::SubrangeOperand> subrangeStack;
  Value baseValue = derivedValue;
  while (auto definingOp = dyn_cast_or_null<IREE::Util::TiedOpInterface>(
             baseValue.getDefiningOp())) {
    auto tiedValue = definingOp.getTiedResultOperand(baseValue);
    if (!tiedValue)
      break;
    if (auto subrangeOp = dyn_cast<IREE::Util::SubrangeOpInterface>(
            definingOp.getOperation())) {
      if (subrangeOp.getSubrangeResource() == tiedValue) {
        subrangeStack.push_back(IREE::Util::SubrangeOperand{
            subrangeOp.getSubrangeResource(),
            subrangeOp.getSubrangeResourceSize(),
            subrangeOp.getSubrangeOffset(), subrangeOp.getSubrangeLength()});
      }
    }
    baseValue = tiedValue;
  }
  return subrangeStack;
}

// Returns a resource range for |resultValue| mapping to the base resource.
//
// Example:
//   %0 = resource
//   %1 = subview %0[%a for %a_length]
//   %2 = subview %0[%b for %b_length]
//   return %3 <- result
// -> range(%0[(%a + %b) for %b_length])
static ResourceRange deriveResourceRangeFromResult(Value resultValue,
                                                   Value resultSize,
                                                   OpBuilder &builder) {
  auto subranges = gatherSubranges(resultValue);
  if (subranges.empty())
    return ResourceRange(resultValue, resultSize);

  // TODO(benvanik): switch to affine.apply when fully supported.
  Value offset;
  for (auto subrange : llvm::reverse(subranges)) {
    if (offset) {
      offset = builder.createOrFold<arith::AddIOp>(resultValue.getLoc(), offset,
                                                   subrange.offset);
    } else {
      offset = subrange.offset;
    }
  }

  return ResourceRange(subranges.back().resource, subranges.back().resourceSize,
                       offset, resultSize);
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
static LogicalResult
allocateExecutionRegion(IREE::Stream::AsyncExecuteOp executeOp) {
  LLVM_DEBUG(llvm::dbgs() << "[[ Allocating execution region ]]\n");

  AllocationScope scope(executeOp);

  OpBuilder externalBuilder(executeOp);

  auto &entryBlock = executeOp.getBody().front();
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
  if (executeOp.getAwaitTimepoint()) {
    newAwaitTimepoints.push_back(executeOp.getAwaitTimepoint());
  }
  llvm::append_range(newOperands, executeOp.getResourceOperands());
  llvm::append_range(newOperandSizes, executeOp.getResourceOperandSizes());
  SmallVector<Value> joinTimepoints;

  // TODO(#11249): pre-scan region for collectives and handle in-place behavior
  // where send aliases recv. We probably want to do this early in case both
  // values are produced locally and escape the region.

  // First find all constants and pull them out into a dedicated constant upload
  // op. We'll then capture the result and use that to initialize variables and
  // constants within the region. Note that this removes ops from the region and
  // as such we want to run it first before we go allocate transients.
  auto constantAllocation = extractConstants(executeOp, externalBuilder);
  if (constantAllocation.has_value()) {
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

    auto awaitTimepoint = constantAllocation->constantsOp.getResultTimepoint();
    if (anyCaptured) {
      // The execute region must depend on the constant upload as one or more
      // constants are used. All this code could be much more clever about
      // separating ones that have deps and ones that don't so that we minimize
      // the dependency chain.
      newAwaitTimepoints.push_back(awaitTimepoint);
      auto asmState = getRootAsmState(executeOp->getParentOp());
      LLVM_DEBUG({
        llvm::dbgs() << "  + adding await on dependent constant upload ";
        awaitTimepoint.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << "\n";
      });
      for (auto &reservation : constantAllocation->reservations) {
        auto resourceRange =
            ResourceRange(reservation.capturedArg, reservation.resourceSize);
        scope.mapResourceRange(reservation.constantOp, resourceRange,
                               asmState.get());
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
      auto result = findTiedYieldResult(reservation.constantOp.getResult());
      if (!result)
        continue;
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
  auto asmState = getRootAsmState(executeOp->getParentOp());
  for (auto operand : llvm::enumerate(executeOp.getResourceOperands())) {
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
    scope.mapResourceRange(arg, resourceRange, asmState.get());
  }
  SmallVector<ResultReservation> resultReservations;
  for (auto [result, resultSize] :
       llvm::zip_equal(executeOp.getResults(), executeOp.getResultSizes())) {
    auto resultType = llvm::cast<IREE::Stream::ResourceType>(result.getType());
    if (handledResults.contains(result)) {
      resultReplacements.push_back(std::make_pair(result, Value{}));
      continue;
    }

    // Find the internal op that defined the result.
    auto yieldValue = yieldOp.getResourceOperands()[result.getResultNumber()];

    // Early-exit if we are tied to an operand and have storage already.
    auto tiedOperandIndex =
        executeOp.getTiedResultOperandIndex(result.getResultNumber());
    if (tiedOperandIndex.has_value()) {
      // Already tied; no need to modify just map.
      auto tiedOperand = executeOp.getOperand(tiedOperandIndex.value());
      LLVM_DEBUG({
        auto arg = entryBlock.getArgument(tiedOperandIndex.value());
        AsmState asmState(executeOp->getParentOp());
        llvm::dbgs() << "  - tying operand ";
        tiedOperand.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << " = arg ";
        arg.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << " = ";
        result.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
        auto subrangeStack = gatherSubranges(yieldValue);
        if (!subrangeStack.empty()) {
          llvm::dbgs() << "    -> subranges:\n";
          for (auto subrange : llvm::reverse(subrangeStack)) {
            llvm::dbgs() << "       ";
            subrange.resource.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << "{";
            subrange.resourceSize.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << "}[";
            subrange.offset.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << " for ";
            subrange.length.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << "]\n";
          }
        }
      });
      auto resourceRange = deriveResourceRangeFromResult(yieldValue, resultSize,
                                                         externalBuilder);
      scope.mapResourceRange(yieldValue, resourceRange, asmState.get());
      if (resourceRange.offset) {
        auto resultSubviewOp =
            externalBuilder.create<IREE::Stream::ResourceSubviewOp>(
                yieldValue.getLoc(), tiedOperand, resourceRange.resourceSize,
                resourceRange.offset, resourceRange.length);
        resultReplacements.push_back(
            std::make_pair(result, resultSubviewOp.getResult()));
      } else {
        resultReplacements.push_back(std::make_pair(result, tiedOperand));
      }
      continue;
    }

    auto definingValue =
        IREE::Util::TiedOpInterface::findTiedBaseValue(yieldValue);
    auto *definingOp = definingValue.getDefiningOp();
    if (!definingOp) {
      // Directly returning an operand; this usually gets canonicalized away but
      // may be introduced by intermediate transformations.
      auto arg = llvm::cast<BlockArgument>(definingValue);
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
    auto timepointType = externalBuilder.getType<IREE::Stream::TimepointType>();
    auto [allocaOp, suballocations] =
        IREE::Stream::ResourceAllocaOp::createSuballocations(
            timepointType, reservationSet.reservationTypes.front(),
            reservationSet.reservationLocs, reservationSet.reservationSizes,
            executeOp.getAwaitTimepoint(), executeOp.getAffinityAttr(),
            externalBuilder);
    newAwaitTimepoints.push_back(allocaOp.getResultTimepoint());

    auto asmState = getRootAsmState(executeOp->getParentOp());
    LLVM_DEBUG({
      llvm::dbgs() << "  + alloc for result reservation set: ";
      allocaOp.print(llvm::dbgs(), *asmState);
      llvm::dbgs() << ":\n";
    });

    for (auto [reservation, suballocation] :
         llvm::zip_equal(reservationSet.reservations, suballocations)) {
      newOperands.push_back(suballocation);
      newOperandSizes.push_back(reservation.resultSize);
      resultReplacements.push_back(
          std::make_pair(reservation.result, suballocation));

      // Insert entry arg for the new operand tied all the way to the yield.
      auto arg =
          entryBlock.addArgument(reservation.resultType, reservation.loc);

      LLVM_DEBUG({
        llvm::dbgs() << "    + adding entry arg for reservation ";
        reservation.result.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << "{";
        reservation.resultSize.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << "} from ";
        reservation.yieldValue.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << " as ";
        arg.printAsOperand(llvm::dbgs(), *asmState);
        llvm::dbgs() << "\n";
      });

      // Map into scope, updating all aliases.
      auto resourceRange = ResourceRange(arg, reservation.resultSize);
      scope.mapResourceRange(reservation.yieldValue, resourceRange,
                             asmState.get());
    }
  }

  // Allocate local transients that are scoped entirely within the region.
  // All locals are packed into a single slab and reserved as one.
  // Note that not all regions need transients.
  auto transientAllocation =
      allocateLocalTransients(executeOp, scope, externalBuilder);
  if (transientAllocation.has_value()) {
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
  if (executeOp.getAffinity().has_value()) {
    newExecuteOp.setAffinityAttr(executeOp.getAffinityAttr());
  }
  newExecuteOp.getBody().takeBody(executeOp.getBody());
  executeOp.getResultTimepoint().replaceAllUsesWith(
      newExecuteOp.getResultTimepoint());
  for (auto replacement : resultReplacements) {
    if (!replacement.second)
      continue; // handled already
    LLVM_DEBUG({
      AsmState asmState(newExecuteOp->getParentOp());
      llvm::dbgs() << "  == replacing region result ";
      replacement.first.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << " -> ";
      replacement.second.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    });
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
  asmState = getRootAsmState(newExecuteOp->getParentOp());
  newExecuteOp.getBody().walk<WalkOrder::PreOrder>(
      [&](IREE::Stream::AsyncConcurrentOp concurrentOp) {
        for (auto [outerValue, innerValue] :
             llvm::zip_equal(concurrentOp.getResourceOperands(),
                             concurrentOp.getBody().getArguments())) {
          LLVM_DEBUG({
            llvm::dbgs() << "  = shady alias of wave operand ";
            outerValue.printAsOperand(llvm::dbgs(), *asmState);
            llvm::dbgs() << " to wave arg ";
            innerValue.printAsOperand(llvm::dbgs(), *asmState);
            llvm::dbgs() << "\n";
          });
          scope.mapResourceRange(innerValue,
                                 scope.lookupResourceRange(outerValue),
                                 asmState.get());
        }
        auto yieldOp = cast<IREE::Stream::YieldOp>(
            concurrentOp.getBody().front().getTerminator());
        for (auto [innerValue, outerValue] : llvm::zip_equal(
                 yieldOp.getResourceOperands(), concurrentOp.getResults())) {
          LLVM_DEBUG({
            llvm::dbgs() << "  = shady alias of wave result ";
            innerValue.printAsOperand(llvm::dbgs(), *asmState);
            llvm::dbgs() << " to stream value ";
            outerValue.printAsOperand(llvm::dbgs(), *asmState);
            llvm::dbgs() << "\n";
          });
          scope.mapResourceRange(innerValue,
                                 scope.lookupResourceRange(outerValue),
                                 asmState.get());
        }
      });

  // Apply the scope to the region and convert ops.
  if (failed(applyAsyncAllocations(newExecuteOp.getBody(), scope))) {
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
        newExecuteOp.getResultTimepoint(), newExecuteOp.getAffinityAttr());
    joinTimepoints.push_back(deallocaOp.getResultTimepoint());
    executeTimepointUsers.insert(deallocaOp);
  }

  // If we have any timepoints that we need to join with we do that now such
  // that the original timepoint dependency chain is preserved even if we make
  // local changes here.
  if (!joinTimepoints.empty()) {
    joinTimepoints.push_back(newExecuteOp.getResultTimepoint());
    auto fusedLoc = builder.getFusedLoc(llvm::map_to_vector(
        joinTimepoints, [](auto timepoint) { return timepoint.getLoc(); }));
    auto joinOp = builder.create<IREE::Stream::TimepointJoinOp>(
        fusedLoc, newExecuteOp.getResultTimepoint().getType(), joinTimepoints);
    executeTimepointUsers.insert(joinOp);
    newExecuteOp.getResultTimepoint().replaceUsesWithIf(
        joinOp.getResultTimepoint(), [&](OpOperand &operand) {
          return !executeTimepointUsers.contains(operand.getOwner());
        });
  }

  return success();
}

static LogicalResult convertAsyncLoadOp(IREE::Stream::AsyncLoadOp asyncOp) {
  auto newOp = OpBuilder(asyncOp).create<IREE::Stream::ResourceLoadOp>(
      asyncOp.getLoc(), asyncOp.getResult().getType(), asyncOp.getSource(),
      asyncOp.getSourceSize(), asyncOp.getSourceOffset());
  asyncOp.replaceAllUsesWith(newOp.getResult());
  asyncOp.erase();
  return success();
}

static LogicalResult convertAsyncStoreOp(IREE::Stream::AsyncStoreOp asyncOp) {
  auto newOp = OpBuilder(asyncOp).create<IREE::Stream::ResourceStoreOp>(
      asyncOp.getLoc(), asyncOp.getTarget(), asyncOp.getTargetSize(),
      asyncOp.getTargetOffset(), asyncOp.getValue());
  asyncOp.replaceAllUsesWith(newOp.getTarget());
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
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    for (auto &parentOp : llvm::make_early_inc_range(moduleOp.getOps())) {
      if (auto asyncFuncOp = dyn_cast<IREE::Stream::AsyncFuncOp>(parentOp)) {
        convertAsyncFuncOp(asyncFuncOp);
        continue;
      }
      auto callableOp = dyn_cast<CallableOpInterface>(parentOp);
      if (!callableOp || !callableOp.getCallableRegion() ||
          callableOp.getCallableRegion()->empty()) {
        continue;
      }
      for (auto &op : llvm::make_early_inc_range(
               callableOp.getCallableRegion()->getOps())) {
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
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createScheduleAllocationPass() {
  return std::make_unique<ScheduleAllocationPass>();
}

} // namespace Stream
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
