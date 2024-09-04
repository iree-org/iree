// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-reuse-shared-memory-allocs"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUREUSESHAREDMEMORYALLOCSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

// Group of Alloc operations that have overlapping liveness ranges.
using AliasGroup = SmallVector<Operation *>;
// A pair of operations representing the first and last use of a buffer in a
// top level block.
using LivenessRange = std::pair<Operation *, Operation *>;

/// Helper to compute the liveness range of an allocation. The liveness range
/// is defined by the first and last use in the `parentRegion`. More
/// specifically, the liveness range is the range in the `parentRegion` between
/// the first and last operations that belong to `parentRegion` and contain a
/// use of the alloc (or subview of the alloc) within their child regions.
static LivenessRange getLivenessRange(memref::AllocOp alloc,
                                      DominanceInfo &dominanceInfo,
                                      Region &parentRegion) {
  Operation *begin = nullptr, *end = nullptr;
  SmallVector<Operation *> workList;
  workList.push_back(alloc);
  while (!workList.empty()) {
    Operation *op = workList.pop_back_val();
    for (auto user : op->getUsers()) {
      if (isa<ViewLikeOpInterface, arith::SelectOp>(user)) {
        workList.push_back(user);
        continue;
      }
      Operation *userAncestor = parentRegion.findAncestorOpInRegion(*user);
      if (!begin && !end) {
        begin = end = userAncestor;
        continue;
      }
      if (dominanceInfo.dominates(end, userAncestor)) {
        end = userAncestor;
      }
      if (dominanceInfo.dominates(userAncestor, begin)) {
        begin = userAncestor;
      }
    }
  }
  return {begin, end};
}

/// Compute the liveness range within the single block of the `rootOp` for each
/// shared memory allocation. For this to succeed the `rootOp` must have the
/// following conditions:
///
/// 1. `rootOp` must have only a single region with a single block.
/// 2. `rootOp` must not have control flow.
/// 3. `rootOp` must have at least 2 shared memory allocations.
static LogicalResult populateLivenessRanges(
    Operation *rootOp, DenseMap<Operation *, LivenessRange> &livenessMap,
    SmallVector<memref::AllocOp> &allocs, DominanceInfo &dominanceInfo) {
  if (rootOp->getNumRegions() != 1) {
    return failure();
  }
  if (!rootOp->getRegion(0).hasOneBlock()) {
    return failure();
  }
  Block &rootBlock = rootOp->getRegion(0).getBlocks().front();
  if (llvm::any_of(rootBlock.getOperations(), [](Operation &op) {
        return isa_and_present<cf::ControlFlowDialect>(op.getDialect());
      })) {
    return failure();
  }

  // Collect all shared memory allocations.
  SmallVector<memref::AllocOp> sharedMemAllocs;
  for (auto alloc : rootBlock.getOps<memref::AllocOp>()) {
    if (hasSharedMemoryAddressSpace(alloc.getType())) {
      sharedMemAllocs.push_back(alloc);
    }
  }
  if (sharedMemAllocs.size() < 2) {
    return failure();
  }

  for (memref::AllocOp alloc : sharedMemAllocs) {
    LivenessRange livenessRange =
        getLivenessRange(alloc, dominanceInfo, rootOp->getRegion(0));
    livenessMap.insert(std::make_pair(alloc, livenessRange));
    allocs.push_back(alloc);
  }
  return success();
}

/// Helper to determine if 2 liveness ranges overlap. The ranges overlap if
/// the `begin` of one of the ranges falls on or between the `begin` and `end`
/// of the other range.
static bool livenessRangesOverlap(LivenessRange range, LivenessRange other,
                                  DominanceInfo &dominanceInfo,
                                  Block *entryBlock) {
  return (dominanceInfo.dominates(range.first, other.first) &&
          dominanceInfo.dominates(other.first, range.second)) ||
         (dominanceInfo.dominates(other.first, range.first) &&
          dominanceInfo.dominates(range.first, other.second));
}

/// Returns the sets of allocations that have overlapping liveness ranges.
/// Each set will not have any overlapping liveness ranges with any other set.
/// This greedily fuses allocs into the same group if the range between their
/// first and last uses overlap.
///
/// Example:
/// ```
/// %alloc0 = memref.alloc()
/// %alloc1 = memref.alloc()
/// %alloc2 = memref.alloc()
/// %w0 = memref.store %val0, %alloc0
/// %w1 = memref.store %val1, %alloc1
/// %r0 = memref.load %alloc0
/// %w2 = memref.store %val2, %alloc2
/// %r1 = memref.load %alloc1
/// %r2 = memref.load %alloc2
/// ```
/// The alias group will be {%alloc0, %alloc1, %alloc2}, since `%alloc0`
/// overlaps with `%alloc1` and `%alloc1` overlaps with `%alloc2`.
static SmallVector<AliasGroup>
computeAliasGroups(DenseMap<Operation *, LivenessRange> livenessMap,
                   SmallVector<memref::AllocOp> allocs,
                   DominanceInfo &dominanceInfo, Block *entryBlock) {
  SmallVector<AliasGroup> aliasGroups;
  for (memref::AllocOp alloc : allocs) {
    LivenessRange livenessRange = livenessMap[alloc];
    // Check for overlapping liveness with other alias groups. Combine all
    // groups that overlap with the `alloc`.
    SmallVector<AliasGroup> overlappingGroups, newAliasGroups;
    for (AliasGroup group : aliasGroups) {
      if (llvm::any_of(group, [&](Operation *alias) {
            return livenessRangesOverlap(livenessRange, livenessMap[alias],
                                         dominanceInfo, entryBlock);
          })) {
        overlappingGroups.push_back(group);
      } else {
        newAliasGroups.push_back(group);
      }
    }
    AliasGroup combinedGroup;
    for (AliasGroup group : overlappingGroups) {
      combinedGroup.append(group);
    }
    combinedGroup.push_back(alloc);
    newAliasGroups.push_back(combinedGroup);
    aliasGroups = newAliasGroups;
  }
  return aliasGroups;
}

/// Get all subviews of the `op` recursively. Child subviews come before their
/// parent views in the set of `views`.
static void getAllSubviews(Operation *op, SetVector<Operation *> &views) {
  for (Operation *user : op->getUsers()) {
    if (isa<ViewLikeOpInterface, arith::SelectOp>(user)) {
      if (user->getUsers().empty()) {
        return;
      }
      getAllSubviews(user, views);
      views.insert(user);
    }
  }
}

/// Compute the combined LivenessRange of the `aliasGroup`. The resulting
/// LivenessRange will span the LivenessRange of all aliases in the group.
static LivenessRange
getAliasGroupLivenessRange(AliasGroup group, DominanceInfo &dominanceInfo,
                           DenseMap<Operation *, LivenessRange> livenessMap) {
  Operation *begin = nullptr, *end = nullptr;
  for (Operation *alias : group) {
    LivenessRange liveness = livenessMap[alias];
    if (!begin || dominanceInfo.dominates(liveness.first, begin)) {
      begin = liveness.first;
    }
    if (!end || dominanceInfo.dominates(end, liveness.second)) {
      end = liveness.second;
    }
  }
  return {begin, end};
}

namespace {
struct GPUReuseSharedMemoryAllocsPass final
    : impl::GPUReuseSharedMemoryAllocsPassBase<GPUReuseSharedMemoryAllocsPass> {
  using GPUReuseSharedMemoryAllocsPassBase::GPUReuseSharedMemoryAllocsPassBase;
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // Map for storing the liveness range of each buffer. Because this pass only
    // considers functions with a single block and no control flow, this can
    // just be a pair of operations representing the first and last use of the
    // buffer.
    DenseMap<Operation *, LivenessRange> livenessMap;
    SmallVector<memref::AllocOp> allocs;

    DominanceInfo dominanceInfo(funcOp);
    // If the funcOp does not meet the conditions for the analysis, do nothing.
    if (failed(populateLivenessRanges(funcOp, livenessMap, allocs,
                                      dominanceInfo))) {
      return;
    }

    // Vector of aliasGroups with non-overlapping liveness ranges.
    Block *entryBlock = &(*funcOp.getBlocks().begin());
    SmallVector<AliasGroup> aliasGroups =
        computeAliasGroups(livenessMap, allocs, dominanceInfo, entryBlock);

    // Nothing to reuse if there is only a single alias group.
    if (aliasGroups.size() < 2) {
      return;
    }

    // We may need to add extra barriers to make sure we are done writing or
    // reading from the previous alias group before starting a new one.
    SmallVector<LivenessRange> aliasGroupLivenessRanges;
    for (SmallVector<Operation *> group : aliasGroups) {
      aliasGroupLivenessRanges.push_back(
          getAliasGroupLivenessRange(group, dominanceInfo, livenessMap));
    }
    // Add a barrier at the start of the LivenessRange of any AliasGroup that
    // has a preceding AliasGroup.
    OpBuilder builder(funcOp->getContext());
    for (size_t groupIdx = 0; groupIdx < aliasGroups.size(); groupIdx++) {
      OpBuilder::InsertionGuard guard(builder);
      LivenessRange liveness = aliasGroupLivenessRanges[groupIdx];
      for (size_t otherIdx = 0; otherIdx < aliasGroups.size(); otherIdx++) {
        if (otherIdx == groupIdx) {
          continue;
        }
        LivenessRange otherLiveness = aliasGroupLivenessRanges[otherIdx];
        // Add a barrier if the `otherLiveness` comes before `liveness`.
        if (dominanceInfo.dominates(otherLiveness.first, liveness.first)) {
          builder.setInsertionPoint(liveness.first);
          builder.create<gpu::BarrierOp>(liveness.first->getLoc());
          break;
        }
      }
    }

    // Pack all the allocations into one i8 alloc.
    packAllocs(builder, funcOp, aliasGroups);
  }
};
} // namespace

} // namespace mlir::iree_compiler
