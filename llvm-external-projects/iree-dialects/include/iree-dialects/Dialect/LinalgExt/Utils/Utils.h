// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_UTILS_UTILS_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_UTILS_UTILS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`.
Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim);

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`. If the shape is constant, returns the shape as an `IntegerAttr`.
OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim);
SmallVector<OpFoldResult> getDims(OpBuilder &builder, Location loc, Value v);

/// Returns a vector that interchanges `elements` starting at offset `offset`
/// based on the indexes in `interchangeVector`.
template <typename T>
SmallVector<T> interchange(ArrayRef<T> elements,
                           ArrayRef<int64_t> interchangeVector,
                           int offset = 0) {
  SmallVector<T> vec = llvm::to_vector(elements);
  for (auto en : llvm::enumerate(interchangeVector)) {
    vec[en.index() + offset] = elements[en.value() + offset];
  }
  return vec;
}
template <typename T>
SmallVector<T> undoInterchange(ArrayRef<T> elements,
                               ArrayRef<int64_t> interchangeVector,
                               int offset = 0) {
  SmallVector<T> vec = llvm::to_vector(elements);
  for (auto en : llvm::enumerate(interchangeVector)) {
    vec[en.value() + offset] = elements[en.index() + offset];
  }
  return vec;
}

/// Returns the `interchangeVector` based on `dimsPos`.
SmallVector<int64_t> computeInterchangeFromDimPos(ArrayRef<int64_t> dimsPos,
                                                  int64_t rank);

/// Converts a 2D float array to a constant value. The 2D array is stored as
/// a 1D row-major array in `val` and has shape `rows` x `cols`.
Value createValueFrom2DConstant(const float *val, int64_t rows, int64_t cols,
                                Location loc, RewriterBase &rewriter);

// Converts OpFoldResults to int64_t shape entries, unconditionally mapping all
// Value's to kDynamic, even if they are arith.constant values.
SmallVector<int64_t> asShapeWithAnyValueAsDynamic(ArrayRef<OpFoldResult> ofrs);

enum class Permutation {
  NCHW_TO_NHWC,
  NHWC_TO_NCHW,
  TTNHWC_TO_TTNCHW,
  TTNCHW_TO_TTNHWC,
};

// Permutes the elements of a SmallVector depending on the permutation specified
template <Permutation P, typename T>
static void permute(SmallVectorImpl<T> &vector) {
  switch (P) {
  case Permutation::NCHW_TO_NHWC:
    std::rotate(vector.begin() + 1, vector.begin() + 2, vector.end());
    break;
  case Permutation::NHWC_TO_NCHW:
    std::rotate(vector.rbegin(), vector.rbegin() + 1, vector.rend() - 1);
    break;
  case Permutation::TTNCHW_TO_TTNHWC:
    std::rotate(vector.begin() + 3, vector.begin() + 4, vector.end());
    break;
  case Permutation::TTNHWC_TO_TTNCHW:
    std::rotate(vector.rbegin(), vector.rbegin() + 1, vector.rend() - 3);
    break;
  default:
    break;
  }
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
#endif // IREE_DIALECTS_DIALECT_LINALGEXT_UTILS_UTILS_H_
