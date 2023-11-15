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
namespace IREE = mlir::iree_compiler::IREE;

//===----------------------------------------------------------------------===//
// LayoutConflictResolutionOp
//===----------------------------------------------------------------------===//

// Validate that the desired layout has the same shape as the input.
LogicalResult LayoutConflictResolutionOp::verify() {
  Operation *op = getOperation();
  LayoutAttr layout = getDesiredLayout();
  ArrayRef<int64_t> inputShape =
      cast<VectorType>(getInput().getType()).getShape();
  for (auto perDimLayout : llvm::enumerate(layout.getLayouts())) {
    ArrayRef<int64_t> shape = perDimLayout.value().getShapes();
    int64_t computedShape =
        std::reduce(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
    int64_t expectedShape = inputShape[perDimLayout.index()];
    if (computedShape != expectedShape) {
      return op->emitError("The layout shape does not match the input shape. "
                           "Expected shape to be ")
             << std::to_string(expectedShape) << ", got "
             << std::to_string(computedShape);
    }
  }
  return success();
}

// clang-format off
#define GET_OP_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.cpp.inc" // IWYU pragma: keep
// clang-format: on
