// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_UTILS_UTILS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_UTILS_UTILS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`.
Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim);

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`. If the shape is constant, returns the shape as an `IntegerAttr`.
OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim);
SmallVector<OpFoldResult> getDims(OpBuilder &builder, Location loc, Value v);

/// Returns a `memref.subview` or a `tensor.extract_slice` based on the type of
/// `src`.
Value getSlice(OpBuilder &b, Location loc, Value src,
               ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
               ArrayRef<OpFoldResult> strides);

/// Returns a `memref.cast` or `tensor.cast` based on the type of `src`.
Value castValue(OpBuilder &builder, Location loc, Value src, ShapedType type);

/// Returns a vector that interchanges `elements` starting at offset `offset`
/// based on the indexes in `interchangeVector`.
template <typename T>
SmallVector<T> interchange(ArrayRef<T> elements,
                           ArrayRef<int64_t> interchangeVector,
                           int offset = 0) {
  SmallVector<T> vec = llvm::to_vector(elements);
  for (auto [idx, val] : llvm::enumerate(interchangeVector)) {
    vec[idx + offset] = elements[val + offset];
  }
  return vec;
}
template <typename T>
SmallVector<T> undoInterchange(ArrayRef<T> elements,
                               ArrayRef<int64_t> interchangeVector,
                               int offset = 0) {
  SmallVector<T> vec = llvm::to_vector(elements);
  for (auto [idx, val] : llvm::enumerate(interchangeVector)) {
    vec[val + offset] = elements[idx + offset];
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
  FCHW_TO_HWCF,
  HWC_TO_CHW,
  CHW_TO_HWC,
  TTFC_TO_TTCF,
};

// Permutes the elements of a SmallVector depending on the permutation specified
template <Permutation P, typename T>
static void permute(SmallVectorImpl<T> &vector) {
  switch (P) {
  case Permutation::FCHW_TO_HWCF:
    std::rotate(vector.begin(), vector.begin() + 2, vector.end()); // to HWFC
    std::rotate(vector.rbegin(), vector.rbegin() + 1, vector.rend() - 2);
    break;
  case Permutation::CHW_TO_HWC:
    std::rotate(vector.rbegin(), vector.rbegin() + 2, vector.rbegin() + 3);
    break;
  case Permutation::HWC_TO_CHW:
    std::rotate(vector.rbegin(), vector.rbegin() + 1, vector.rbegin() + 3);
    break;
  case Permutation::TTFC_TO_TTCF:
    std::rotate(vector.rbegin(), vector.rbegin() + 1, vector.rend() - 2);
    break;
  default:
    break;
  }
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
#endif // IREE_COMPILER_DIALECT_LINALGEXT_UTILS_UTILS_H_
