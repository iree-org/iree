// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"

namespace mlir::iree_compiler::IREE::Flow {

/// Gets the list of non-static values from a list of `OpFoldResult`.
static SmallVector<Value>
getDynamicValues(ArrayRef<OpFoldResult> valueOrAttrList) {
  SmallVector<Value> dynamicDims;
  for (auto valueOrAttr : valueOrAttrList) {
    if (auto value = valueOrAttr.dyn_cast<Value>()) {
      dynamicDims.push_back(value);
    }
  }
  return dynamicDims;
}

/// Get shape of the tensor given the sizes as a list of `OpFoldResult`.
static SmallVector<int64_t>
getShapeFromSizes(ArrayRef<OpFoldResult> valueOrAttrList) {
  return llvm::map_to_vector(
      valueOrAttrList, [&](OpFoldResult valueOrAttr) -> int64_t {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          return llvm::cast<IntegerAttr>(attr).getInt();
        }
        return ShapedType::kDynamic;
      });
}

// This is not a complete heuristic for whether a particular index value
// depends on something expected to require a host-device sync to use, but
// works for most inputs we can expect today.
static bool producedByValueExtract(OpFoldResult index) {
  Value indexVal = dyn_cast<Value>(index);
  if (!indexVal) {
    return false;
  }

  BackwardSliceOptions options;
  bool hasExtract = false;
  options.inclusive = false;
  options.omitBlockArguments = true;
  options.filter = [&](Operation *op) {
    if (isa<tensor::ExtractOp, TensorLoadOp>(op)) {
      hasExtract = true;
      return false;
    }
    return true;
  };

  // Get the backward slice of the index.
  SetVector<Operation *> backwardSlice;
  getBackwardSlice(indexVal, &backwardSlice, options);
  return hasExtract;
}

static bool isOffsetSizeAndStrideMappableToFlow(ArrayRef<OpFoldResult> offsets,
                                                ArrayRef<OpFoldResult> sizes,
                                                ArrayRef<OpFoldResult> strides,
                                                ArrayRef<int64_t> baseShape) {
  if (offsets.size() != baseShape.size()) {
    // Unhanded rank-reducing case.
    return false;
  }
  auto getVal = [](OpFoldResult valueOrAttr, int64_t dynamicVal) -> int64_t {
    auto attr = valueOrAttr.dyn_cast<Attribute>();
    return attr ? llvm::cast<IntegerAttr>(attr).getInt() : dynamicVal;
  };
  /// To ensure contiguity, start from the least significant dimension. As long
  /// as the inner slices are "full slices", the current slice can be any offset
  /// and size. If the inner slices are not "full slices", the current slice
  /// must be of size 1. All strides must be one.

  bool fullSlices = true;
  for (size_t dim = offsets.size(); dim > 0; dim--) {
    OpFoldResult offset = offsets[dim - 1];
    OpFoldResult size = sizes[dim - 1];
    OpFoldResult stride = strides[dim - 1];

    // The offsets and sizes dont have to be static for all dimensions. When
    // `fullSlices` is true, the offset and sizes can be dynamic. But in some
    // cases, the dynamic offset/size value is obtained by computing from
    // another tensor which lives on the device. To avoid host-round tripping
    // try to infer when a value is extracted from a tensor.
    if (producedByValueExtract(offset) || producedByValueExtract(stride) ||
        producedByValueExtract(size)) {
      return false;
    }

    int64_t staticOffset = getVal(offset, ShapedType::kDynamic);
    int64_t staticSize = getVal(size, ShapedType::kDynamic);
    int64_t staticStride = getVal(stride, ShapedType::kDynamic);

    if (staticStride != 1)
      return false;

    if (fullSlices == false) {
      if (staticSize != 1)
        return false;
    } else {
      // TODO: Use ValueBoundsAnalysis to check whether two dynamic values
      // are equal.
      if (!(staticOffset == 0 && !ShapedType::isDynamic(staticSize) &&
            !ShapedType::isDynamic(baseShape[dim - 1]) &&
            staticSize == baseShape[dim - 1])) {
        fullSlices = false;
      }
    }
  }
  return true;
}

LogicalResult
convertInsertSliceOpToFlowUpdateOp(RewriterBase &rewriter,
                                   tensor::InsertSliceOp insertOp) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(insertOp);

  if (insertOp->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
    return failure();
  }

  SmallVector<OpFoldResult> offsets = insertOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = insertOp.getMixedSizes();
  SmallVector<OpFoldResult> strides = insertOp.getMixedStrides();
  ArrayRef<int64_t> dstShape = insertOp.getType().getShape();
  if (!isOffsetSizeAndStrideMappableToFlow(offsets, sizes, strides, dstShape)) {
    return failure();
  }

  Location loc = insertOp.getLoc();
  auto sourceDynamicDims = getDynamicValues(sizes);
  Value source = insertOp.getSource();
  ShapedType sourceType = insertOp.getSourceType();
  ShapedType destType = insertOp.getType();

  // Handle rank-reduced version.
  if (sourceType.getRank() < destType.getRank()) {
    // Get the un-rank-reduced shape of the source.
    auto unreducedShape = getShapeFromSizes(sizes);
    sourceType =
        RankedTensorType::get(unreducedShape, sourceType.getElementType());
    source = rewriter.create<IREE::Flow::TensorReshapeOp>(
        loc, sourceType, source, sourceDynamicDims, sourceDynamicDims);
  }

  auto offsetVals = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                    insertOp.getMixedOffsets());
  Value dest = insertOp.getDest();
  auto destDynamicDims = tensor::createDynamicDimValues(rewriter, loc, dest);
  rewriter.replaceOpWithNewOp<TensorUpdateOp>(insertOp, insertOp.getType(),
                                              dest, destDynamicDims, offsetVals,
                                              source, sourceDynamicDims);
  return success();
}

LogicalResult
convertExtractSliceOpToFlowSliceOp(RewriterBase &rewriter,
                                   tensor::ExtractSliceOp sliceOp) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(sliceOp);

  if (sliceOp->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
    return failure();
  }

  SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();
  SmallVector<OpFoldResult> strides = sliceOp.getMixedStrides();
  ArrayRef<int64_t> srcShape = sliceOp.getSourceType().getShape();
  if (!isOffsetSizeAndStrideMappableToFlow(offsets, sizes, strides, srcShape)) {
    return failure();
  }

  Location loc = sliceOp.getLoc();

  ShapedType sourceType = sliceOp.getSourceType();
  ShapedType resultType = sliceOp.getType();

  // Handle rank reduced version.
  if (resultType.getRank() < sourceType.getRank()) {
    // Get the un-rank-reduced shape of the result.
    auto unreducedShape = getShapeFromSizes(sizes);
    resultType =
        RankedTensorType::get(unreducedShape, sourceType.getElementType());
  }

  auto offsetVals = getValueOrCreateConstantIndexOp(rewriter, loc, offsets);
  auto sizeVals = getValueOrCreateConstantIndexOp(rewriter, loc, sizes);
  auto sourceDynamicDims =
      tensor::createDynamicDimValues(rewriter, loc, sliceOp.getSource());
  auto resultDynamicDims = getDynamicValues(sizes);
  Value replacement = rewriter.create<TensorSliceOp>(
      loc, resultType, sliceOp.getSource(), sourceDynamicDims, offsetVals,
      sizeVals, resultDynamicDims);
  if (resultType.getRank() > sliceOp.getType().getRank()) {
    replacement = rewriter.create<IREE::Flow::TensorReshapeOp>(
        loc, sliceOp.getType(), replacement, resultDynamicDims,
        resultDynamicDims);
  }
  rewriter.replaceOp(sliceOp, replacement);
  return success();
}

} // namespace mlir::iree_compiler::IREE::Flow
