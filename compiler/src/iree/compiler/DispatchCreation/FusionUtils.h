// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- FusionUtils.h --- Utility functions used in fusion ---------------===//
//
// Utility functions to decide of ops are fusable or not, etc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::DispatchCreation {

/// Return true of the producer and consumer of `operand` are fusable
/// using elementwise op fusion transformation.
bool areFusableAsElementwiseOps(MLIRContext *context, OpOperand *operand,
                                bool fuseMultiReduction);

/// Check that a given operation is "horizontal" to the group. The operation
/// is horizontal if the program slice of the operation (from op back to seedOp)
/// does not contain any op from the group.
bool isHorizontalToGroup(Operation *op, ArrayRef<Operation *> currGroup,
                         const DominanceInfo &dominanceInfo, Operation *seedOp);

/// Moves the operands and transitive defs for each op in `operations` directly
/// after `insertionPoint`. Note: this does not check if it is legal to move the
/// operands.
template <typename T>
static LogicalResult
moveOperandDefs(RewriterBase &rewriter, ArrayRef<T> operations,
                Operation *insertionPoint, const DominanceInfo &dominanceInfo,
                ArrayRef<linalg::LinalgOp> ignoreOperations = {}) {
  BackwardSliceOptions options;
  options.omitUsesFromAbove = false;
  llvm::DenseSet<Operation *> ignoreOperationsSet;
  ignoreOperationsSet.insert(ignoreOperations.begin(), ignoreOperations.end());
  options.filter = [&](Operation *op) {
    return !dominanceInfo.properlyDominates(op, insertionPoint) &&
           !ignoreOperationsSet.contains(op);
  };
  // Set inclusive to true cause the slice is computed from the operand, and
  // we want to include the defining op (which is the point here)
  options.inclusive = true;

  llvm::SetVector<Operation *> slice;
  for (auto op : operations) {
    assert(insertionPoint->getBlock() == op->getBlock());
    for (auto operand : op->getOperands()) {
      getBackwardSlice(operand, &slice, options);
    }
  }

  mlir::topologicalSort(slice);
  for (auto op : slice) {
    rewriter.moveOpBefore(op, insertionPoint);
  }
  return success();
}

} // namespace mlir::iree_compiler::DispatchCreation
