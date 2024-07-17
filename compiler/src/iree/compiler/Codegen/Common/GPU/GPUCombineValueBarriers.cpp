// Copyright 2022 The IREE Authors
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
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-codegen-gpu-combine-value-barriers"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCOMBINEVALUEBARRIERSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Move the topologically sorted slice before the given operation.
static void moveSliceBeforeOp(RewriterBase &rewriter,
                              llvm::SetVector<Operation *> &slice,
                              Operation *insertionPoint) {
  for (auto sliceOp : slice) {
    rewriter.moveOpBefore(sliceOp, insertionPoint);
  }
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

static LogicalResult
combineValueBarriersInBackwardSliceOf(RewriterBase &rewriter, Operation *op) {
  // 1. First compute the backward slice which contains ops in the same basic
  // block
  //    but the `value_barrier` ops are treated as stopping points for the
  //    slice.
  BackwardSliceOptions options;
  SmallVector<IREE::GPU::ValueBarrierOp> valueBarriers;
  options.filter = [&](Operation *candidate) {
    if (candidate->getBlock() != op->getBlock()) {
      return false;
    }
    if (auto valueBarrierOp = dyn_cast<IREE::GPU::ValueBarrierOp>(candidate)) {
      valueBarriers.push_back(valueBarrierOp);
      return false;
    }
    return true;
  };
  llvm::SetVector<Operation *> slice;
  mlir::getBackwardSlice(op, &slice, options);

  // Return early with no value barrier ops.
  if (valueBarriers.size() <= 1) {
    return success();
  }

  // 2. The slice is already topologically sorted. Just move them before the
  // op.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  moveSliceBeforeOp(rewriter, slice, op);

  // 3. At this point we should be able to combine all the value barriers into
  // a single value barrier before the first op of the slice that has been
  // moved.
  if (!slice.empty()) {
    rewriter.setInsertionPoint(slice.front());
  }
  return combineValueBarrierOps(rewriter, op->getLoc(), valueBarriers);
}

/// Combine `value_barrier`s at the ends of blocsk
static LogicalResult combineTrailingValueBarriers(RewriterBase &rewriter,
                                                  Block *block) {
  auto terminator = block->getTerminator();
  return combineValueBarriersInBackwardSliceOf(rewriter, terminator);
}

/// Combine `value_barriers` at the beginning of blocks.
static LogicalResult combineLeadingValueBarriers(RewriterBase &rewriter,
                                                 Block *block) {
  // 1. Find all the leading `value_barrier`s in the block
  //    whose slice does not contain any other `value_barrier`
  SmallVector<IREE::GPU::ValueBarrierOp> candidateValueBarriers;
  Operation *insertionPoint = nullptr;
  llvm::SetVector<Operation *> sliceUnion;
  for (Operation &op : *block) {
    auto valueBarrierOp = dyn_cast<IREE::GPU::ValueBarrierOp>(&op);
    if (!valueBarrierOp) {
      continue;
    }

    // Check the slice in the block to see it has no other `value_barrier`s.
    BackwardSliceOptions options;
    bool hasValueBarrierInSlice = false;
    options.filter = [&](Operation *candidate) {
      if (candidate->getBlock() != op.getBlock()) {
        return false;
      }
      if ((candidate != &op) && isa<IREE::GPU::ValueBarrierOp>(candidate)) {
        hasValueBarrierInSlice = true;
        return false;
      }
      return true;
    };
    llvm::SetVector<Operation *> currSlice;
    mlir::getBackwardSlice(&op, &currSlice, options);

    if (hasValueBarrierInSlice) {
      continue;
    }
    candidateValueBarriers.push_back(valueBarrierOp);
    sliceUnion.insert(currSlice.begin(), currSlice.end());
    if (!insertionPoint) {
      // The first barrier is the insertion point.
      // All dependent ops will be moved before this.
      insertionPoint = &op;
    }
  }

  if (candidateValueBarriers.size() <= 1) {
    return success();
  }

  // 2. Topologically sort the union of slices.
  mlir::topologicalSort(sliceUnion);

  // 3. Move the operations before the insertion point;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(insertionPoint);
  moveSliceBeforeOp(rewriter, sliceUnion, insertionPoint);

  // 4. Now combine the value barrier ops.
  return combineValueBarrierOps(rewriter, block->getParentOp()->getLoc(),
                                candidateValueBarriers);
}

/// Main dispatch method to combine value barries in region.
static LogicalResult combineValueBarriersInRegion(RewriterBase &rewriter,
                                                  Region *region) {
  if (region->getBlocks().size() != 1) {
    return region->getParentOp()->emitOpError(
        "expected region with single block");
  }

  Block *block = &region->front();
  // Note: the order of the following steps shouldnt matter.
  // 1. Combine the value barriers "at the ends of blocks".
  if (failed(combineTrailingValueBarriers(rewriter, block))) {
    return failure();
  }

  // 2. Combine the value barriers "at the begining of blocks"
  if (failed(combineLeadingValueBarriers(rewriter, block))) {
    return failure();
  }

  // TODO: We could also combine value barries in middle of operations
  // by anchoring on operations.
  return success();
}

struct GPUCombineValueBarriersPass final
    : impl::GPUCombineValueBarriersPassBase<GPUCombineValueBarriersPass> {

  void runOnOperation() override {
    auto operation = getOperation();

    SetVector<Region *> barrierRegions;
    // 1. Walk the operation to get all regions that have value barriers
    // (restrict to operations with single block);
    operation->walk([&](IREE::GPU::ValueBarrierOp valueBarrierOp) {
      Region *parentRegion = valueBarrierOp->getBlock()->getParent();
      if (parentRegion->getBlocks().size() == 1) {
        barrierRegions.insert(parentRegion);
      }
    });

    IRRewriter rewriter(&getContext());
    for (Region *region : barrierRegions) {
      if (failed(combineValueBarriersInRegion(rewriter, region))) {
        region->getParentOp()->emitOpError(
            "failed to combined value barriers for this op");
        return signalPassFailure();
      }
    }

    return;
  }
};

} // namespace

} // namespace mlir::iree_compiler
