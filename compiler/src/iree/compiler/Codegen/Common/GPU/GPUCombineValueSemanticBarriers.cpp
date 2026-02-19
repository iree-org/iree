// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// This file implements the pass to combine multiple `iree_gpu.value_barrier`
// and `iree_gpu.barrier_region` ops.
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-codegen-gpu-combine-value-semantic-barriers"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCOMBINEVALUESEMANTICBARRIERSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Move the given backward slice before the given barrier.
/// The backward slice should not have any operations which are before the
/// barrier or the barrier itself.
static void moveBackwardSliceBeforeBarrier(RewriterBase &rewriter,
                                           llvm::SetVector<Operation *> &slice,
                                           Operation *leadingBarrier) {
  // Sort operations to be moved topologically.
  slice = topologicalSort(slice);

  // It is always valid (w.r.t. dominance) to move topologically sorted
  // operations in a backward slice which come after the insertion point, to
  // before the insertion point. This is because:
  //  - Since we are operating on a backward slice, producers to every operation
  //  in the slice are already in the slice, and will be moved behind the
  //  insertion point.
  //  - Any consumers will still remain after the operation, as we are only
  //  moving the operation before.
  for (Operation *sliceOp : slice) {
    rewriter.moveOpBefore(sliceOp, leadingBarrier);
  }
}

/// Move the slice after the given barrier.
/// The forward slice should not have any operations which are after the
/// barrier or the barrier itself.
static void moveForwardSliceAfterBarrier(RewriterBase &rewriter,
                                         llvm::SetVector<Operation *> &slice,
                                         Operation *trailingBarrier) {
  // Sort operations to be moved topologically.
  slice = topologicalSort(slice);

  // It is always valid (w.r.t. dominance) to move topologically sorted
  // operations in a forward slice which come before the insertion point, to
  // after the insertion point. This is because:
  //  - Since we are operating on a forward slice, consumers to every operation
  //  in the slice are already in the slice, and will be moved after the
  //  insertion point,
  //  - Any producers will still remain before the operation, as we are only
  //  moving the operation after.
  for (Operation *sliceOp : llvm::reverse(slice)) {
    rewriter.moveOpAfter(sliceOp, trailingBarrier);
  }
}

/// Given two barriers in the same block, use backward/forward slice analysis
/// to move their dependencies before the leading barrier and their consumers
/// after the trailing barrier, making them safe to combine. Returns failure if
/// the barriers form a dependency chain.
///
/// On entry, barrierA may be before or after barrierB; on exit, barrierA is
/// guaranteed to be before barrierB (swapped if necessary).
static LogicalResult enforceBarrierOrdering(RewriterBase &rewriter,
                                            Operation *&barrierA,
                                            Operation *&barrierB) {
  if (barrierB->isBeforeInBlock(barrierA)) {
    std::swap(barrierA, barrierB);
  }

  assert(barrierA->getBlock() == barrierB->getBlock());
  Block *block = barrierA->getBlock();

  auto sliceFilterBackward = [&block, &barrierA](Operation *candidate) -> bool {
    if (candidate->getBlock() != block) {
      return false;
    }
    if (candidate == block->getTerminator()) {
      return false;
    }
    if (candidate->isBeforeInBlock(barrierA)) {
      return false;
    }
    return true;
  };

  // Find the combined backward slice of barrierA and barrierB and try
  // to move it before barrierA (before both the barriers).
  BackwardSliceOptions bOptions;
  bOptions.omitUsesFromAbove = false;
  bOptions.filter = sliceFilterBackward;
  SetVector<Operation *> backwardSliceA;
  SetVector<Operation *> backwardSliceB;
  [[maybe_unused]] LogicalResult resultA =
      getBackwardSlice(barrierA, &backwardSliceA, bOptions);
  assert(resultA.succeeded());
  [[maybe_unused]] LogicalResult resultB =
      getBackwardSlice(barrierB, &backwardSliceB, bOptions);
  assert(resultB.succeeded());
  backwardSliceA.insert(backwardSliceB.begin(), backwardSliceB.end());
  // If the first barrier is contained in the combined backward slice of both
  // barriers, the barriers form a chain and cannot be combined.
  if (backwardSliceA.contains(barrierA)) {
    return failure();
  }
  moveBackwardSliceBeforeBarrier(rewriter, backwardSliceA, barrierA);

  auto sliceFilterForward = [&block, &barrierB](Operation *candidate) -> bool {
    if (candidate->getBlock() != block) {
      return false;
    }
    if (candidate == block->getTerminator()) {
      return false;
    }
    if (barrierB->isBeforeInBlock(candidate)) {
      return false;
    }
    return true;
  };

  // Find the combined forward slice of barrierA and barrierB and try to
  // move it after barrierB (after both the barriers).
  ForwardSliceOptions fOptions;
  fOptions.filter = sliceFilterForward;
  SetVector<Operation *> forwardSliceA;
  SetVector<Operation *> forwardSliceB;
  getForwardSlice(barrierA, &forwardSliceA, fOptions);
  getForwardSlice(barrierB, &forwardSliceB, fOptions);
  forwardSliceA.insert(forwardSliceB.begin(), forwardSliceB.end());
  // If the second barrier is contained in the combined forward slice of both
  // barriers, the barriers form a chain and cannot be combined.
  if (forwardSliceA.contains(barrierA)) {
    return failure();
  }
  moveForwardSliceAfterBarrier(rewriter, forwardSliceA, barrierB);

  return success();
}

/// Given two value_barrier ops, combine them into a single value_barrier.
static FailureOr<IREE::GPU::ValueBarrierOp>
combineValueBarrierPair(RewriterBase &rewriter,
                        IREE::GPU::ValueBarrierOp barrierA,
                        IREE::GPU::ValueBarrierOp barrierB) {
  // Both barriers need to have either tensor semantics or vector semantics.
  if (barrierA.hasTensorSemantics() != barrierB.hasTensorSemantics()) {
    return failure();
  }

  Operation *opA = barrierA;
  Operation *opB = barrierB;
  if (failed(enforceBarrierOrdering(rewriter, opA, opB))) {
    return failure();
  }
  barrierA = cast<IREE::GPU::ValueBarrierOp>(opA);
  barrierB = cast<IREE::GPU::ValueBarrierOp>(opB);

  // We add the new barrier after both the barriers (it is always better
  // to sink barriers).
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(barrierB);

  SmallVector<Value> barrierOperands;
  barrierOperands.append(barrierA.getOperands().begin(),
                         barrierA.getOperands().end());
  barrierOperands.append(barrierB.getOperands().begin(),
                         barrierB.getOperands().end());

  auto combinedBarrierOp = IREE::GPU::ValueBarrierOp::create(
      rewriter, barrierB.getLoc(), barrierOperands);

  int numOperandsA = barrierA.getNumOperands();
  int numOperandsB = barrierB.getNumOperands();
  rewriter.replaceOp(barrierA,
                     combinedBarrierOp->getResults().slice(0, numOperandsA));
  rewriter.replaceOp(barrierB, combinedBarrierOp->getResults().slice(
                                   numOperandsA, numOperandsB));

  return combinedBarrierOp;
}

/// Given two barrier_region ops, combine them into a single barrier_region.
static FailureOr<IREE::GPU::BarrierRegionOp>
combineBarrierRegionPair(RewriterBase &rewriter,
                         IREE::GPU::BarrierRegionOp barrierA,
                         IREE::GPU::BarrierRegionOp barrierB) {
  Operation *opA = barrierA;
  Operation *opB = barrierB;
  if (failed(enforceBarrierOrdering(rewriter, opA, opB))) {
    return failure();
  }
  barrierA = cast<IREE::GPU::BarrierRegionOp>(opA);
  barrierB = cast<IREE::GPU::BarrierRegionOp>(opB);

  Location fusedLoc =
      rewriter.getFusedLoc({barrierA.getLoc(), barrierB.getLoc()});

  // Get the combined operands, result types, and yielded values.
  SmallVector<Value> combinedOperands = barrierA.getInputs();
  combinedOperands.append(barrierB.getInputs().begin(),
                          barrierB.getInputs().end());
  SmallVector<Type> combinedTypes(barrierA->getResultTypes());
  combinedTypes.append(barrierB->getResultTypes().begin(),
                       barrierB->getResultTypes().end());

  auto aYield = cast<IREE::GPU::YieldOp>(barrierA.getBody()->getTerminator());
  auto bYield = cast<IREE::GPU::YieldOp>(barrierB.getBody()->getTerminator());
  SmallVector<Value> combinedYields = aYield.getValues();
  combinedYields.append(bYield.getValues().begin(), bYield.getValues().end());

  // Create the combined barrier after barrierB (it is always better to sink
  // barriers).
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(barrierB);

  auto combinedBarrierOp = IREE::GPU::BarrierRegionOp::create(
      rewriter, fusedLoc, combinedTypes, combinedOperands);

  MutableArrayRef<BlockArgument> barrierABbArgReplacements =
      combinedBarrierOp.getBody()->getArguments().take_front(
          barrierA->getNumOperands());
  MutableArrayRef<BlockArgument> barrierBBbArgReplacements =
      combinedBarrierOp.getBody()->getArguments().take_back(
          barrierB->getNumOperands());

  // Merge the bodies of the old barriers into the new one.
  rewriter.mergeBlocks(barrierA.getBody(), combinedBarrierOp.getBody(),
                       barrierABbArgReplacements);
  rewriter.mergeBlocks(barrierB.getBody(), combinedBarrierOp.getBody(),
                       barrierBBbArgReplacements);

  // Erase the old terminators and create a new one with the concatenated
  // yielded values.
  rewriter.eraseOp(aYield);
  rewriter.eraseOp(bYield);

  rewriter.setInsertionPointToEnd(combinedBarrierOp.getBody());
  IREE::GPU::YieldOp::create(rewriter, fusedLoc, combinedYields);

  // Replace all uses of the previous barriers with the new combined barrier.
  int numResultsA = barrierA.getNumResults();
  int numResultsB = barrierB.getNumResults();
  rewriter.replaceOp(barrierA,
                     combinedBarrierOp->getResults().slice(0, numResultsA));
  rewriter.replaceOp(barrierB, combinedBarrierOp->getResults().slice(
                                   numResultsA, numResultsB));

  return combinedBarrierOp;
}

/// Try to combine all same-type barriers in a block using O(n^2) pairwise
/// iteration.
template <typename BarrierOp, typename CombineFn>
static void combineBarriersInBlock(RewriterBase &rewriter, Block *block,
                                   CombineFn combineFn) {
  SmallVector<BarrierOp> barriers;
  for (Operation &op : block->getOperations()) {
    if (auto barrier = dyn_cast<BarrierOp>(op)) {
      barriers.push_back(barrier);
    }
  }

  // We iterate over all pairs. This could be optimized to O(n) to take
  // into account deletions, but we do the simplest thing for now.
  int numBarriers = barriers.size();
  for (int i = 0; i < numBarriers; ++i) {
    if (!barriers[i]) {
      continue;
    }

    for (int j = i + 1; j < numBarriers; ++j) {
      if (!barriers[j]) {
        continue;
      }

      auto combined = combineFn(rewriter, barriers[i], barriers[j]);
      if (succeeded(combined)) {
        barriers[i] = combined.value();
        barriers[j] = nullptr;
      }
    }
  }
}

struct GPUCombineValueSemanticBarriersPass final
    : impl::GPUCombineValueSemanticBarriersPassBase<
          GPUCombineValueSemanticBarriersPass> {

  void runOnOperation() override {
    // Walk the operation to get all blocks that have barriers. We restrict
    // ourselves to blocks, because the order of operations in a block is easy
    // to determine.
    SmallVector<Block *> blocks;
    getOperation()->walk([&blocks](Block *block) {
      if (llvm::any_of(block->getOperations(), [](Operation &op) {
            return isa<IREE::GPU::ValueBarrierOp, IREE::GPU::BarrierRegionOp>(
                op);
          })) {
        blocks.push_back(block);
      }
    });

    IRRewriter rewriter(&getContext());
    for (auto *block : blocks) {
      combineBarriersInBlock<IREE::GPU::ValueBarrierOp>(
          rewriter, block, combineValueBarrierPair);
      combineBarriersInBlock<IREE::GPU::BarrierRegionOp>(
          rewriter, block, combineBarrierRegionPair);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
