// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/RegionOpUtils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler {

LogicalResult moveOperandDefs(RewriterBase &rewriter,
                              ArrayRef<Operation *> operations,
                              Operation *insertionPoint,
                              DominanceInfo &dominanceInfo,
                              ArrayRef<Operation *> ignoreOperations) {
  BackwardSliceOptions options;
  llvm::DenseSet<Operation *> ignoreOperationsSet;
  ignoreOperationsSet.insert(ignoreOperations.begin(), ignoreOperations.end());
  options.filter = [&](Operation *op) {
    return !dominanceInfo.properlyDominates(op, insertionPoint) &&
           !ignoreOperationsSet.contains(op);
  };
  // Set inclusive to true cause the slice is computed from the operand, and
  // we want to include the defining op (which is the point here)
  options.omitUsesFromAbove = false;
  options.inclusive = true;

  llvm::SetVector<Operation *> slice;
  for (auto op : operations) {
    for (auto operand : op->getOperands()) {
      // If operand is the insertion point, there is nothing to move.
      if (operand.getDefiningOp() == insertionPoint) {
        continue;
      }
      [[maybe_unused]] LogicalResult result =
          getBackwardSlice(operand, &slice, options);
      assert(result.succeeded());
    }
    auto regions = op->getRegions();
    if (regions.empty()) {
      continue;
    }
    llvm::SetVector<Value> capturedVals;
    mlir::getUsedValuesDefinedAbove(regions, capturedVals);
    for (auto value : capturedVals) {
      // If operand is the insertion point, there is nothing to move.
      if (value.getDefiningOp() == insertionPoint) {
        continue;
      }
      [[maybe_unused]] LogicalResult result =
          getBackwardSlice(value, &slice, options);
      assert(result.succeeded());
    }
  }

  if (slice.contains(insertionPoint)) {
    return failure();
  }

  mlir::topologicalSort(slice);
  for (auto op : slice) {
    rewriter.moveOpBefore(op, insertionPoint);
  }
  return success();
}

} // namespace mlir::iree_compiler
