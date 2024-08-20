// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// This file implements the pass to combine multiple `iree_gpu.value_barrier`
// ops.
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-codegen-gpu-combine-value-barriers"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCOMBINEVALUEBARRIERSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Move the given backward slice before the given operation.
static LogicalResult
moveBackwardSliceBeforeOp(RewriterBase &rewriter,
                          llvm::SetVector<Operation *> &slice,
                          Operation *insertionPoint) {
  SetVector<Operation *> toMove;
  // First remove all operations that are already before the insertion point.
  for (Operation *op : slice) {
    if (op == insertionPoint) {
      // A slice containing the insertion point itself cannot be moved
      // behind the insertion point.
      return failure();
    }

    if (insertionPoint->isBeforeInBlock(op)) {
      toMove.insert(op);
    }
  }

  // Sort operations to be moved topologically.
  toMove = topologicalSort(toMove);

  // It is always valid (w.r.t. dominance) to move topologically sorted
  // operations in a backward slice which come after the insertion point, to
  // before the insertion point. This is because:
  //  - Since we are operating on a backward slice, producers to every operation
  //  in the slice already in the slice, and will be moved behind the insertion
  //  point.
  //  - Any consumers will still remain after the operation, as we are only
  //  moving the operation before.
  for (Operation *sliceOp : toMove) {
    rewriter.moveOpBefore(sliceOp, insertionPoint);
  }

  return success();
}

/// Move the slice after the given operation.
static LogicalResult
moveForwardSliceAfterOp(RewriterBase &rewriter,
                        llvm::SetVector<Operation *> &slice,
                        Operation *insertionPoint) {
  SetVector<Operation *> toMove;
  // First remove all operations that are already after the insertion point.
  for (Operation *op : slice) {
    if (op == insertionPoint) {
      // A slice containing the insertion point itself cannot be moved
      // behind the insertion point.
      return failure();
    }

    if (op->isBeforeInBlock(insertionPoint)) {
      toMove.insert(op);
    }
  }

  // Sort operations to be moved topologically.
  toMove = topologicalSort(toMove);

  // It is always valid (w.r.t. dominance) to move topologically sorted
  // operations in a forward slice which come before the insertion point, to
  // after the insertion point. This is because:
  //  - Since we are operating on a forward slice, consumers to every operation
  //  in the slice are already in the slice, and will be moved after the
  //  insertion point,
  //  - Any producers will still remain before the operation, as we are only
  //  moving the operation after.
  for (Operation *sliceOp : llvm::reverse(toMove)) {
    rewriter.moveOpAfter(sliceOp, insertionPoint);
  }

  return success();
}

/// Combine all value barriers into a single value barrier.
static LogicalResult
combineValueBarrierOps(RewriterBase &rewriter, Location loc,
                       ArrayRef<IREE::GPU::ValueBarrierOp> valueBarriers) {
  if (valueBarriers.size() <= 1) {
    return success();
  }
  SmallVector<Value> barrierOperands;
  for (auto barrierOp : valueBarriers) {
    barrierOperands.append(barrierOp.getInputs().begin(),
                           barrierOp.getInputs().end());
  }
  auto combinedBarrierOp =
      rewriter.create<IREE::GPU::ValueBarrierOp>(loc, barrierOperands);

  // Replace all uses of the previous barrier with new barrier.
  int resultNumber = 0;
  for (auto barrierOp : valueBarriers) {
    int numResults = barrierOp.getNumResults();
    rewriter.replaceOp(barrierOp, combinedBarrierOp->getResults().slice(
                                      resultNumber, numResults));
    resultNumber += numResults;
  }
  return success();
}

/// Given two barriers, barrierA and barrierB, combine them into a single
/// barrier.
static FailureOr<IREE::GPU::ValueBarrierOp>
combineValueBarrierPair(RewriterBase &rewriter,
                        IREE::GPU::ValueBarrierOp barrierA,
                        IREE::GPU::ValueBarrierOp barrierB) {
  // Both barriers need to have either tensor semantics or vector semantics.
  if (barrierA.hasTensorSemantics() && !barrierB.hasTensorSemantics()) {
    return failure();
  }
  if (!barrierA.hasTensorSemantics() && barrierB.hasTensorSemantics()) {
    return failure();
  }

  // We assume barrierA is always before barrierB.
  if (barrierB->isBeforeInBlock(barrierA)) {
    std::swap(barrierA, barrierB);
  }

  // barrierA and barrierB are in the same block.
  Block *block = barrierA->getBlock();

  auto sliceFilter = [&block](Operation *candidate) -> bool {
    if (candidate->getBlock() != block) {
      return false;
    }
    if (candidate == block->getTerminator()) {
      // Do not move the terminator.
      return false;
    }
    return true;
  };

  // Find the combined backward slice of barrierA and barrierB and try
  // to move it before barrierA (before both the barriers).
  BackwardSliceOptions bOptions;
  bOptions.filter = sliceFilter;
  SetVector<Operation *> backwardSliceA;
  SetVector<Operation *> backwardSliceB;
  getBackwardSlice(barrierA, &backwardSliceA, bOptions);
  getBackwardSlice(barrierB, &backwardSliceB, bOptions);
  backwardSliceA.insert(backwardSliceB.begin(), backwardSliceB.end());
  // Move the backward slice before barrierA.
  if (failed(moveBackwardSliceBeforeOp(rewriter, backwardSliceA, barrierA))) {
    return failure();
  }

  // Find the combined forward slice of barrierA and barrierB and try to
  // move it after barrierB (after both the barriers).
  ForwardSliceOptions fOptions;
  fOptions.filter = sliceFilter;
  SetVector<Operation *> forwardSliceA;
  SetVector<Operation *> forwardSliceB;
  getForwardSlice(barrierA, &forwardSliceA, fOptions);
  getForwardSlice(barrierB, &forwardSliceB, fOptions);
  forwardSliceA.insert(forwardSliceB.begin(), forwardSliceB.end());
  // Move the forward slice after barrierB.
  if (failed(moveForwardSliceAfterOp(rewriter, forwardSliceA, barrierB))) {
    return failure();
  }

  // We add the new barrier after both the barriers (it is always better
  // to sink barriers).
  rewriter.setInsertionPointAfter(barrierB);

  SmallVector<Value> barrierOperands;
  barrierOperands.append(barrierA.getOperands().begin(),
                         barrierA.getOperands().end());
  barrierOperands.append(barrierB.getOperands().begin(),
                         barrierB.getOperands().end());

  auto combinedBarrierOp = rewriter.create<IREE::GPU::ValueBarrierOp>(
      barrierB.getLoc(), barrierOperands);

  int numOperandsA = barrierA.getNumOperands();
  int numOperandsB = barrierB.getNumOperands();
  rewriter.replaceOp(barrierA,
                     combinedBarrierOp->getResults().slice(0, numOperandsA));
  rewriter.replaceOp(barrierB, combinedBarrierOp->getResults().slice(
                                   numOperandsA, numOperandsB));

  return combinedBarrierOp;
}

static void combineValueBarriersInBlock(RewriterBase &rewriter, Block *block) {
  SmallVector<IREE::GPU::ValueBarrierOp> barriers;
  for (Operation &op : block->getOperations()) {
    if (auto barrier = dyn_cast<IREE::GPU::ValueBarrierOp>(op)) {
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

      FailureOr<IREE::GPU::ValueBarrierOp> combined =
          combineValueBarrierPair(rewriter, barriers[i], barriers[j]);
      if (succeeded(combined)) {
        barriers[i] = combined.value();
        barriers[j] = nullptr;
      }
    }
  }
}

struct GPUCombineValueBarriersPass final
    : impl::GPUCombineValueBarriersPassBase<GPUCombineValueBarriersPass> {

  void runOnOperation() override {
    // Walk the operation to get all blocks that have value barriers. We
    // restrict ourselves to blocks, because the order of operations in a block
    // is easy to determine.
    SmallVector<Block *> blocks;
    getOperation()->walk([&blocks](Block *block) {
      if (llvm::any_of(block->getOperations(),
                       llvm::IsaPred<IREE::GPU::ValueBarrierOp>)) {
        blocks.push_back(block);
      }
    });

    IRRewriter rewriter(&getContext());
    for (auto *block : blocks) {
      combineValueBarriersInBlock(rewriter, block);
    }

    return;
  }
};

} // namespace

} // namespace mlir::iree_compiler
