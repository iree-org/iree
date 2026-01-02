// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTOPINTERFACES_H_
#define IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTOPINTERFACES_H_

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::TensorExt {

class SparseCastOpInterface;
// Interface verification method to verify the sparse op satisfies
// interface constraints.
LogicalResult verifySparseCastOpInterface(SparseCastOpInterface sparseOp);

} // namespace mlir::iree_compiler::IREE::TensorExt

// clang-format off: must be included after all LLVM/MLIR headers
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOpInterfaces.h.inc" // IWYU pragma: keep
// clang-format on: must be included after all LLVM/MLIR headers

namespace mlir::iree_compiler::IREE::TensorExt {

/// If a `Range` is defined using the result of an operation that implements the
/// `SparseCastOpInterface`, the operation needs to be used to resolve this
/// range (either to an estimated range or generate code that iterates over the
/// exact sparse range). This struct holds information of the sparse operation
/// that has to be used for the resolution, and the result dimension that
/// defines the range.
struct SparseRangeResolver {
  SparseCastOpInterface sparseOp;
  int64_t resultDim;
};

/// For a given Range retrieve the SparseRangeResolver if it is defined by a
/// sparse operation.
std::optional<SparseRangeResolver> getSparseRangeResolver(Range range);

/// For a list of Ranges retrieve the SparseRangeResolvers if they are defined
/// by sparse operations. The returned vector has the same size as the input
/// ranges. If a resolver cannot be found for any of the ranges, and empty
/// resolver is returned at that position.
SmallVector<SparseRangeResolver>
getSparseRangeResolvers(ArrayRef<Range> ranges);

} // namespace mlir::iree_compiler::IREE::TensorExt

#endif // IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTOPINTERFACES_H_
