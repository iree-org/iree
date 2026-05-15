// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_

// TODO(Max191): Move to Utils/ once the IR/ -> Utils/ dependency cycle is
// resolved. Im2colUtils.{h,cpp} are transformation helpers consumed by
// Transforms/ and IR/AggregatedOpInterfaceImpl, not op definitions or dialect
// infrastructure, but Utils/ cannot currently depend on IR/ types like
// Im2colOp.

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Holds the computed source indices for an im2col operation at a given
/// output position. These indices describe where to read from the input tensor.
struct Im2colSourceIndices {
  /// Full set of offsets into the input tensor, one per input dimension.
  /// When padding is present, these are in the padded coordinate space.
  SmallVector<OpFoldResult> sliceOffsets;
  /// Sizes for each input dimension (1 except for the vectorized dim).
  SmallVector<OpFoldResult> sliceSizes;
};

/// Compute source (input tensor) indices for a given im2col output position.
///
/// Given the loop induction variables representing the current output position,
/// compute the corresponding offsets and sizes into the input tensor. Uses the
/// unified offsets + output_sizes attributes to delinearize each output dim
/// independently, then maps to input coordinates via strides, dilations, and
/// input_k_perm.
///
/// Shared by both the decomposition and vectorization paths.
Im2colSourceIndices computeIm2colSourceIndices(OpBuilder &b, Location loc,
                                               Im2colOp im2colOp,
                                               ArrayRef<Value> ivs,
                                               OpFoldResult innerTileSize);

/// Compute the valid read size along the vectorized (innermost input) dimension
/// for im2col decomposition and vectorization.
///
/// Given source indices in the padded coordinate space, determines how many
/// elements can be validly read from the input tensor. Accounts for:
///   - Input padding (pad_low/pad_high on the convolution input)
///   - Output padding (output_pad_low/output_pad_high on the output)
///   - Out-of-bounds status of non-vectorized dimensions
///
/// When any non-vectorized dimension is out-of-bounds, returns 0.
///
/// Callers are responsible for computing their own read offsets from the
/// source indices (e.g., by subtracting pad_low and clamping).
///
/// \p outputIVs are the loop IVs for output dimensions (in actual tensor dim
/// order). Used for output-side bounds checking when output padding is present.
Value computeIm2colValidSize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                             const Im2colSourceIndices &srcIndices,
                             OpFoldResult innerTileSize,
                             ArrayRef<Value> outputIVs,
                             std::optional<int64_t> vecOutputDim);

/// Choose which output dimension to vectorize for an im2col op.
/// Returns the output dimension index, or std::nullopt if no dimension can be
/// vectorized (in which case scalar unrolling should be used).
///
/// \p offsets are the per-output-dim offsets from the im2col op's attributes.
/// For K output dims, the offset of the specific dim being considered is used
/// directly for the contiguity check (no linearization needed).
std::optional<int64_t> chooseDimToVectorize(OpBuilder &b, Location loc,
                                            Im2colOp im2colOp,
                                            ArrayRef<Range> iterationDomain,
                                            ArrayRef<OpFoldResult> offsets);

/// Compute vector tile sizes for an im2col op. Returns a vector of tile sizes
/// with one entry per output dimension. The vectorizable dimension (if any)
/// gets its full iteration size; all other dimensions get 1. Returns nullopt
/// if no vectorizable dimension is found (e.g. no contiguous slice exists).
std::optional<SmallVector<int64_t>>
computeIm2colVectorTileSizes(OpBuilder &b, Im2colOp im2colOp);

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_
