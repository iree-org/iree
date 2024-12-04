// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#define DEBUG_TYPE "iree-codegen-common-transforms"

namespace mlir::iree_compiler {

/// Pattern to convert `tensor.extract_slice(tensor.expand_shape)` to
/// `tensor.expand_shape(tensor.extract_slice)`.
static LogicalResult
swapExpandShapeWithSlice(RewriterBase &rewriter,
                         tensor::ExpandShapeOp expandShapeOp,
                         tensor::ExtractSliceOp sliceOp) {
  SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();

  if (sliceOp.getResultType().getRank() != sizes.size()) {
    return rewriter.notifyMatchFailure(sliceOp,
                                       "unimplemented: rank reducing slice");
  }

  // Helper variables and function for accumulating the new offset and length
  // values.
  Location loc = expandShapeOp->getLoc();
  AffineExpr d0, d1, d2;
  bindDims(rewriter.getContext(), d0, d1, d2);
  // Multiply two integers.
  auto mul = [&](OpFoldResult v1, OpFoldResult v2) {
    auto mulMap = AffineMap::get(2, 0, {d0 * d1});
    return affine::makeComposedFoldedAffineApply(rewriter, loc, mulMap,
                                                 {v1, v2});
  };
  auto mulAdd = [&](OpFoldResult v1, OpFoldResult v2, OpFoldResult v3) {
    auto mulMap = AffineMap::get(3, 0, {d0 * d1 + d2});
    return affine::makeComposedFoldedAffineApply(rewriter, loc, mulMap,
                                                 {v1, v2, v3});
  };

  SmallVector<OpFoldResult> outputShape =
      getMixedValues(expandShapeOp.getStaticOutputShape(),
                     expandShapeOp.getOutputShape(), rewriter);

  auto isZeroOffsetAndFullSize = [](OpFoldResult offset, OpFoldResult sliceSize,
                                    OpFoldResult size) {
    if (!isConstantIntValue(offset, 0))
      return false;
    FailureOr<bool> maybeEqual =
        ValueBoundsConstraintSet::areEqual(sliceSize, size);
    return llvm::succeeded(maybeEqual) && maybeEqual.value();
  };

  // First verify that this is a full slice of the expanded tensor.
  for (const ReassociationIndices &indices :
       expandShapeOp.getReassociationIndices()) {
    int64_t i = 0;
    int64_t e = indices.size();
    // Find the first expanded dim after the first dim with non-unit extracted
    // size.
    for (; i < e; ++i) {
      if (!isConstantIntValue(sizes[indices[i]], 1)) {
        // +1 to skip the first non-unit size dim.
        i++;
        break;
      }
    }

    // Verify that all subsequent dimensions extract the full size of the
    // source tensor.
    for (; i < e; ++i) {
      int64_t expandedDim = indices[i];
      if (!isZeroOffsetAndFullSize(offsets[expandedDim], sizes[expandedDim],
                                   outputShape[expandedDim])) {
        return rewriter.notifyMatchFailure(
            sliceOp, "Not a contiguous slice of the expanded tensor.");
      }
    }
  }

  // Compute new offsets, lengths, and strides.
  SmallVector<OpFoldResult> newOffsets, newLengths, newStrides;
  for (const ReassociationIndices &indices :
       expandShapeOp.getReassociationIndices()) {
    OpFoldResult newOffset = rewriter.getIndexAttr(0);
    OpFoldResult newSize = rewriter.getIndexAttr(1);

    int64_t i = 0;
    int64_t e = indices.size();
    // Offset = cumulative product of leading unit extracted dims.
    for (; i < e; ++i) {
      int64_t expandedDim = indices[i];
      if (!isConstantIntValue(sizes[expandedDim], 1))
        break;

      newOffset =
          mulAdd(newOffset, outputShape[expandedDim], offsets[expandedDim]);
    }

    if (i != e) {
      int64_t expandedDim = indices[i];
      newOffset =
          mulAdd(newOffset, outputShape[expandedDim], offsets[expandedDim]);
      newSize = sizes[expandedDim];
      i++;
    }

    for (; i < e; ++i) {
      OpFoldResult fullSize = outputShape[indices[i]];
      newOffset = mul(newOffset, fullSize);
      newSize = mul(newSize, fullSize);
    }

    newOffsets.push_back(newOffset);
    newLengths.push_back(newSize);

    // Only unit stride supported.
    newStrides.push_back(rewriter.getIndexAttr(1));
  }

  // The shape of the result can be obtained from the sizes passed in.
  SmallVector<Value> dynDims;
  SmallVector<int64_t> shape;
  dispatchIndexOpFoldResults(sizes, dynDims, shape);
  RankedTensorType resultType = RankedTensorType::get(
      shape, expandShapeOp.getResultType().getElementType());

  // Create a new ExtractSliceOp and ExpandShapeOp.
  Value newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, expandShapeOp.getSrc(), newOffsets, newLengths, newStrides);
  auto newExpandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
      loc, resultType, newSliceOp, expandShapeOp.getReassociationIndices(),
      sizes);
  rewriter.replaceOp(sliceOp, newExpandShapeOp);
  return success();
}

namespace {

/// tensor.empty does not define any tensor contents, so an unpadded pack
/// can be folded away.
struct SwapExpandShapeWithSlicePattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto expandOp = sliceOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp) {
      return failure();
    }

    if (!sliceOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "unsupported: non-unit stride");
    }

    return swapExpandShapeWithSlice(rewriter, expandOp, sliceOp);
  }
};

} // namespace

void populateSwapExtractWithExpandPattern(RewritePatternSet &patterns) {
  patterns.add<SwapExpandShapeWithSlicePattern>(patterns.getContext());
}

} // namespace mlir::iree_compiler
