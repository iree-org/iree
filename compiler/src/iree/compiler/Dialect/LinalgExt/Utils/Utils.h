// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_UTILS_UTILS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_UTILS_UTILS_H_

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
struct Range;
}; // namespace mlir

namespace mlir::iree_compiler::IREE::LinalgExt {

// Helper method to add 2 OpFoldResult inputs with affine.apply.
OpFoldResult addOfrs(OpBuilder &builder, Location loc, OpFoldResult a,
                     OpFoldResult b);

// Helper method to multiply 2 OpFoldResult inputs with affine.apply.
OpFoldResult mulOfrs(OpBuilder &builder, Location loc, OpFoldResult a,
                     OpFoldResult b);

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`.
Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim);

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`. If the shape is constant, returns the shape as an `IntegerAttr`.
OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim);
SmallVector<OpFoldResult> getDims(OpBuilder &builder, Location loc, Value v);

/// Returns a `memref.subview` or a `tensor.extract_slice` based on the type of
/// `src`.
Operation *getSlice(OpBuilder &b, Location loc, Value src,
                    ArrayRef<Range> slice);
Operation *getSlice(OpBuilder &b, Location loc, Value src,
                    ArrayRef<OpFoldResult> offsets,
                    ArrayRef<OpFoldResult> sizes,
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
  NCHW_TO_NHWC,
  NHWC_TO_NCHW,
  FCHW_TO_HWCF,
  TTNHWC_TO_TTNCHW,
  TTNCHW_TO_TTNHWC,
  TTFC_TO_TTCF,
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
  case Permutation::FCHW_TO_HWCF:
    std::rotate(vector.begin(), vector.begin() + 2, vector.end()); // to HWFC
    std::rotate(vector.rbegin(), vector.rbegin() + 1, vector.rend() - 2);
    break;
  case Permutation::TTNCHW_TO_TTNHWC:
    std::rotate(vector.begin() + 3, vector.begin() + 4, vector.end());
    break;
  case Permutation::TTNHWC_TO_TTNCHW:
    std::rotate(vector.rbegin(), vector.rbegin() + 1, vector.rend() - 3);
    break;
  case Permutation::TTFC_TO_TTCF:
    std::rotate(vector.rbegin(), vector.rbegin() + 1, vector.rend() - 2);
    break;
  default:
    break;
  }
}

/// Return dim expresssions that can be used as replacements in map that
/// contains `numSymbols` symbols. The new dim expressions have positions
/// `numDims, numDims + 1, numDims + 2, ...., numDims + numSymbols - 1`.
SmallVector<AffineExpr> getDimExprsForSymbols(MLIRContext *context,
                                              unsigned numDims,
                                              unsigned numSymbols);

/// Convert all symbols in the map to dim expressions, such that the new dim
/// expressions have positions `numDims, numDims + 1, numDims + 2, ...., numDims
/// + numSymbols - 1`.
AffineMap convertDimsToSymbols(AffineMap map, unsigned numDims,
                               unsigned numSymbols,
                               SmallVector<AffineExpr> &symbolReplacements);
SmallVector<AffineMap>
convertDimsToSymbols(ArrayRef<AffineMap> maps, unsigned numDims,
                     unsigned numSymbols,
                     SmallVector<AffineExpr> &symbolReplacements);
SmallVector<AffineMap> convertDimsToSymbols(MLIRContext *context,
                                            ArrayRef<AffineMap> map,
                                            unsigned numDims,
                                            unsigned numSymbols);

/// Returns the indexing maps array for a convolution operation with IGEMM
/// indexing. The resulting indexing maps should represent the indexing of some
/// contraction that computes the equivalent IGEMM matmul of the convolution.
FailureOr<SmallVector<AffineMap>>
getIGEMMContractionIndexingMaps(linalg::LinalgOp linalgOp);

/// Returns the loop bounds of a convolution op with IGEMM indexing. This
/// function assumes the same ordering of dimensions as
/// getIGEMMContractionIndexingMaps;
FailureOr<SmallVector<int64_t>> getIGEMMLoopBounds(linalg::LinalgOp linalgOp);

/// Returns the operand list for a convolution with IGEMM indexing. This is
/// used to determine which inputs are the lhs and rhs, since depending on the
/// layout, the order can be different (e.g., NCHW has the lhs and rhs swapped).
FailureOr<SmallVector<Value>> getIGEMMOperands(linalg::LinalgOp linalgOp);

/// Returns true if the operation increases bitwidths of tensors.
/// This function checks that the genericOp:
/// 1. Has only one output.
/// 2. Has all parallel loops.
/// 3. Compared to the element type of the input with highest rank,
///    the output element type has a higher bitwidth.
bool isBitExtendOp(Operation *op);

/// Returns true if the operation decreases bitwidths of tensors.
/// This function checks that the genericOp:
/// 1. Has only one output.
/// 2. Has all parallel loops.
/// 3. Compared to the element type of the input with highest rank,
///    the output element type has a lower bitwidth.
bool isBitTruncateOp(Operation *op);

/// Returns true if the operation is a BroadcastOp or a GenericOp performing
/// a broadcast.
/// This function checks that the genericOp:
///     1. Has a single input and output.
///     2. Has all parallel loops.
///     3. Has an identity output map.
///     4. Has a projected permutation input map.
///     5. The input map has fewer results than the output map.
///     6. Has a body with only a linalg.yield op.
bool isBroadcastingOp(linalg::LinalgOp op);

/// Returns true if the operation is a `linalg.generic` that is similar in
/// effect to a gather.
/// This function checks that the genericOp:
///     1. Has a single input and output.
///     2. Has all parallel loops.
///     2. `linalg.yield` consumes the result of a `tensor.extract_slice`
bool isGatherlikeOp(Operation *op);

} // namespace mlir::iree_compiler::IREE::LinalgExt
#endif // IREE_COMPILER_DIALECT_LINALGEXT_UTILS_UTILS_H_
