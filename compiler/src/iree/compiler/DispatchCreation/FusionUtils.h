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

#include "mlir/IR/Operation.h"

namespace mlir {
class DominanceInfo;
} // namespace mlir

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

} // namespace mlir::iree_compiler::DispatchCreation
