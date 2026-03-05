// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: Move Im2colUtils.{h,cpp} from IR/ to Utils/. These are transformation
// helpers consumed by Transforms/ and IR/AggregatedOpInterfaceImpl, not op
// definitions or dialect infrastructure.

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_

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

/// Holds the computed padding bounds for im2col decomposition/vectorization.
/// For each input dimension, provides the adjusted (unpadded-space) offset and
/// the clamped valid size within the input bounds.
struct Im2colPaddingBounds {
  /// Adjusted offsets: srcIndices.sliceOffsets - padLow, clamped to [0, dim-1].
  /// Used as read offsets into the unpadded input tensor.
  SmallVector<OpFoldResult> readOffsets;
  /// Valid size along the vectorized dimension, accounting for all spatial
  /// bounds. When any non-vectorized spatial dim is out-of-bounds, this is 0.
  Value validSize;
  /// Low-side pad amount along the vectorized dimension (max(-adjusted, 0)).
  /// Non-zero when padding extends before the input start along the vec dim.
  Value vecLowPadAmt;
};

/// Compute padding bounds for im2col decomposition and vectorization.
///
/// The padding bounds are determined by when the load/slice for a given output
/// position falls outside of the input tensor bounds. This encompasses both
/// input padding (explicit pad_low/pad_high on the convolution input) and
/// output padding (when an output position maps to coordinates outside the
/// input, e.g. due to strided convolutions or output alignment padding).
///
/// Given source indices in the padded coordinate space, compute:
///   1. Adjusted read offsets in the unpadded input space (clamped to bounds)
///   2. Valid size along the vectorized dim, incorporating out-of-bounds checks
///      for all non-vectorized spatial dims
///
/// Only \p padLow is needed; high-side bounds are captured implicitly by the
/// `inputExtent - readStart` computation in the valid size calculation.
///
/// Shared by both decomposition (uses validSize for extract_slice + pad) and
/// vectorization (uses bounds to create masks).
Im2colPaddingBounds
computeIm2colPaddingBounds(OpBuilder &b, Location loc, Im2colOp im2colOp,
                           const Im2colSourceIndices &srcIndices,
                           ArrayRef<OpFoldResult> inputSizes,
                           ArrayRef<OpFoldResult> padLow,
                           OpFoldResult innerTileSize,
                           ArrayRef<Value> outputIVs,
                           ArrayRef<OpFoldResult> outputOffsets,
                           std::optional<int64_t> vecOutputDim);

/// Returns true if the im2col op has output-side padding, i.e. any output
/// tensor dimension is larger than the product of its output_sizes. This
/// happens after FoldOutputPadIntoIm2col enlarges the output for alignment.
bool hasOutputPadding(Im2colOp im2colOp);

/// Compute the valid (unpadded) size for each canonical output dimension.
/// Returns the product of output_sizes inner dims for each output dim.
/// When FoldOutputPadIntoIm2col enlarges the output tensor, output_sizes
/// remains unchanged, so this product gives the pre-padding valid region.
SmallVector<OpFoldResult>
computeOutputValidSizes(OpBuilder &b, Location loc, Im2colOp im2colOp);

/// Choose which output dimension to vectorize for an im2col op.
/// Returns the output dimension index, or std::nullopt if no dimension can be
/// vectorized (in which case scalar unrolling should be used).
///
/// \p offsets are the per-output-dim offsets from the im2col op's attributes.
/// For K output dims, the offset of the specific dim being considered is used
/// directly for the contiguity check (no linearization needed).
std::optional<int64_t>
chooseDimToVectorize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                     ArrayRef<Range> iterationDomain,
                     ArrayRef<OpFoldResult> inputSizes,
                     ArrayRef<OpFoldResult> offsets);

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_
