// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include <numeric>

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

//===----------------------------------------------------------------------===//
// LayoutConflictResolutionOp
//===----------------------------------------------------------------------===//

LogicalResult validateLayout(Operation *op, StringRef label,
                             VectorLayoutInterface layout,
                             ArrayRef<int64_t> inputShape) {
  if (!layout.isValidLayout(inputShape)) {
    return op->emitError(
        "The " + label +
        " layout shape cannot be distributed over the given vector shape.");
  }
  return success();
}

// Validate that the desired layout has the same shape as the input.
LogicalResult LayoutConflictResolutionOp::verify() {
  Operation *op = getOperation();
  ArrayRef<int64_t> inputShape =
      cast<VectorType>(getInput().getType()).getShape();
  if (succeeded(validateLayout(op, "source", getSourceLayout(), inputShape)))
    return validateLayout(op, "desired", getDesiredLayout(), inputShape);
  return failure();
}

// clang-format off
#define GET_OP_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.cpp.inc" // IWYU pragma: keep
// clang-format: on
