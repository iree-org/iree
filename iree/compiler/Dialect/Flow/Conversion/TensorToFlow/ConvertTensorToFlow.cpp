// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/ConvertTensorToFlow.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

/// An operation that uses `offsets`, `sizes` and `strides` (i.e. implements the
/// `OffsetSizeAndStrideInterface`) can be mapped to flow operations that
/// eventually map to DMA operations if the offsets/sizes/strides represent a
/// contiguous memory.
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
    return attr ? attr.cast<IntegerAttr>().getInt() : dynamicVal;
  };
  /// To ensure contiguity, start from the least signficant dimension. As long
  /// as the inner slices are "full slices", the current slice can be any offset
  /// and size. If the inner slices are not "full slices", the current slice
  /// must be of size 1. All strides must be one.

  bool fullSlices = true;
  for (size_t dim = offsets.size(); dim > 0; dim--) {
    int64_t staticOffset =
        getVal(offsets[dim - 1], ShapedType::kDynamicStrideOrOffset);
    int64_t staticSize = getVal(sizes[dim - 1], ShapedType::kDynamicSize);
    int64_t staticStride =
        getVal(strides[dim - 1], ShapedType::kDynamicStrideOrOffset);

    if (staticStride != 1) return false;
    // The offsets and sizes dont have to be static for all dimensions. When
    // `fullSlices` is true, the offset and sizes can be dynamic. But many
    // cases, the dynamic offset/size value is obtained by computing from
    // another tensor which lives on the device. To avoid host-round tripping
    // enforce that offset/size is also static.
    if (staticSize == ShapedType::kDynamicSize) return false;
    if (staticOffset == ShapedType::kDynamicStrideOrOffset) return false;

    if (fullSlices == false) {
      if (staticSize != 1) return false;
    } else {
      if (!(staticOffset == 0 && staticSize != ShapedType::kDynamicSize &&
            baseShape[dim - 1] != ShapedType::kDynamicSize &&
            staticSize == baseShape[dim - 1]))
        fullSlices = false;
    }
  }
  return true;
}

/// Returns the `Value`s for a list of `OpFoldResult` by generating std.constant
/// ops for the static values.
static SmallVector<Value, 4> getAsValues(
    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> valueOrAttrList) {
  SmallVector<Value, 4> values;
  for (auto valueOrAttr : valueOrAttrList) {
    if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
      values.push_back(
          b.create<ConstantIndexOp>(loc, attr.cast<IntegerAttr>().getInt()));
    } else {
      values.push_back(valueOrAttr.get<Value>());
    }
  }
  return values;
}

/// Gets the list of non-static values from a list of `OpFoldResult`.
static SmallVector<Value, 4> getDynamicValues(
    ArrayRef<OpFoldResult> valueOrAttrList) {
  SmallVector<Value, 4> dynamicDims;
  for (auto valueOrAttr : valueOrAttrList) {
    if (auto value = valueOrAttr.dyn_cast<Value>()) {
      dynamicDims.push_back(value);
    }
  }
  return dynamicDims;
}

/// Get shape of the tensor given the sizes as a list of `OpFoldResult`.
static SmallVector<int64_t, 4> getShapeFromSizes(
    ArrayRef<OpFoldResult> valueOrAttrList) {
  return llvm::to_vector<4>(llvm::map_range(
      valueOrAttrList, [&](OpFoldResult valueOrAttr) -> int64_t {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          return attr.cast<IntegerAttr>().getInt();
        }
        return ShapedType::kDynamicSize;
      }));
}

/// Generates `memref.dim` operations to get the dynamic sizes of a value `v`.
static SmallVector<Value, 4> getDynamicDimValues(OpBuilder &b, Location loc,
                                                 Value v) {
  SmallVector<Value, 4> dynamicDims;
  for (auto dim : llvm::enumerate(v.getType().cast<ShapedType>().getShape())) {
    if (dim.value() != ShapedType::kDynamicSize) continue;
    dynamicDims.push_back(b.createOrFold<tensor::DimOp>(loc, v, dim.index()));
  }
  return dynamicDims;
}

/// Convert subtensor insert operation flow.tensor.update where possible.
struct SubTensorInsertToTensorUpdate
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (insertOp->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    SmallVector<OpFoldResult, 4> offsets = insertOp.getMixedOffsets();
    SmallVector<OpFoldResult, 4> sizes = insertOp.getMixedSizes();
    SmallVector<OpFoldResult, 4> strides = insertOp.getMixedStrides();
    ArrayRef<int64_t> dstShape = insertOp.getType().getShape();
    if (!isOffsetSizeAndStrideMappableToFlow(offsets, sizes, strides,
                                             dstShape)) {
      return failure();
    }
    Location loc = insertOp.getLoc();
    auto sourceDynamicDims = getDynamicValues(sizes);
    Value source = insertOp.source();
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

    auto offsetVals = getAsValues(rewriter, loc, offsets);
    Value dest = insertOp.dest();
    auto destDynamicDims = getDynamicDimValues(rewriter, loc, dest);
    rewriter.replaceOpWithNewOp<TensorUpdateOp>(
        insertOp, insertOp.getType(), dest, destDynamicDims, offsetVals, source,
        sourceDynamicDims, nullptr);
    return success();
  }
};

/// Convert subtensor operation to flow.tensor.slice where possible.
struct SubTensorToTensorSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (sliceOp->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    SmallVector<OpFoldResult, 4> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult, 4> sizes = sliceOp.getMixedSizes();
    SmallVector<OpFoldResult, 4> strides = sliceOp.getMixedStrides();
    ArrayRef<int64_t> srcShape = sliceOp.getSourceType().getShape();
    if (!isOffsetSizeAndStrideMappableToFlow(offsets, sizes, strides,
                                             srcShape)) {
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

    auto offsetVals = getAsValues(rewriter, loc, offsets);
    auto sizeVals = getAsValues(rewriter, loc, sizes);
    auto sourceDynamicDims =
        getDynamicDimValues(rewriter, loc, sliceOp.source());
    auto resultDynamicDims = getDynamicValues(sizes);
    Value replacement = rewriter.create<TensorSliceOp>(
        loc, resultType, sliceOp.source(), sourceDynamicDims, offsetVals,
        sizeVals, resultDynamicDims);
    if (resultType.getRank() > sliceOp.getType().getRank()) {
      replacement = rewriter.create<IREE::Flow::TensorReshapeOp>(
          loc, sliceOp.getType(), replacement, resultDynamicDims,
          resultDynamicDims);
    }
    rewriter.replaceOp(sliceOp, replacement);
    return success();
  }
};

struct ConvertTensorCastPattern : public OpConversionPattern<tensor::CastOp> {
  using OpConversionPattern<tensor::CastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::CastOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = operands.front();
    ShapedType inputType = input.getType().dyn_cast<ShapedType>();
    ShapedType resultType =
        typeConverter->convertType(op.getType()).dyn_cast_or_null<ShapedType>();
    if (!inputType || !resultType || !inputType.hasRank() ||
        !resultType.hasRank()) {
      return rewriter.notifyMatchFailure(op, "not ranked shaped types");
    }
    // This should not happen, except in the context of type conversion.
    if (inputType.getRank() != resultType.getRank()) {
      return rewriter.notifyMatchFailure(op, "mismatched rank");
    }

    // Resolve dims to the most specific value.
    int rank = inputType.getRank();
    SmallVector<Value> dimSizes(rank);
    auto resolveDimSize = [&](int position) -> Value {
      if (!dimSizes[position]) {
        // Find the most specific.
        if (!inputType.isDynamicDim(position) ||
            !resultType.isDynamicDim(position)) {
          // Static dim.
          int64_t dimSize = !inputType.isDynamicDim(position)
                                ? inputType.getDimSize(position)
                                : resultType.getDimSize(position);
          dimSizes[position] = rewriter.create<ConstantIndexOp>(loc, dimSize);
        } else {
          // Dynamic dim.
          dimSizes[position] =
              rewriter.create<tensor::DimOp>(loc, input, position);
        }
      }

      return dimSizes[position];
    };

    SmallVector<Value> sourceDynamicDims;
    SmallVector<Value> targetDynamicDims;
    for (int i = 0; i < rank; i++) {
      if (inputType.isDynamicDim(i)) {
        sourceDynamicDims.push_back(resolveDimSize(i));
      }
      if (resultType.isDynamicDim(i)) {
        targetDynamicDims.push_back(resolveDimSize(i));
      }
    }

    // TODO: Decide if this needs to be replaced with a flow.tensor.cast
    // See https://github.com/google/iree/issues/6418
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorReshapeOp>(
        op, resultType, input, sourceDynamicDims, targetDynamicDims);

    return success();
  }
};

struct ConvertTensorFromElementsPattern
    : public OpConversionPattern<tensor::FromElementsOp> {
  using OpConversionPattern<tensor::FromElementsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::FromElementsOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    if (shouldBeConverted(op)) {
      SmallVector<Value> dimSizes(1);
      dimSizes[0] = rewriter.create<ConstantIndexOp>(loc, 1);
      rewriter.replaceOpWithNewOp<IREE::Flow::TensorSplatOp>(
          op, op.getType(), operands.front(), dimSizes);
    }

    return success();
  }

  // TODO: This pattern was mainly added to iron out some kinks specific to
  // detensoring (see: https://github.com/google/iree/issues/1159). Do we need
  // to expand this check for other uses?
  static bool shouldBeConverted(tensor::FromElementsOp op) {
    return op.getType().getDimSize(0) == 1;
  }
};

}  // namespace

void populateTensorToFlowPatterns(MLIRContext *context,
                                  OwningRewritePatternList &patterns) {
  patterns.insert<SubTensorInsertToTensorUpdate, SubTensorToTensorSlice>(
      context);
}

void setupTensorToFlowLegality(MLIRContext *context,
                               ConversionTarget &conversionTarget,
                               TypeConverter &typeConverter) {
  conversionTarget.addIllegalOp<tensor::CastOp>();
  conversionTarget.addDynamicallyLegalOp<tensor::FromElementsOp>(
      [](tensor::FromElementsOp op) {
        return !ConvertTensorFromElementsPattern::shouldBeConverted(op);
      });

  conversionTarget.addLegalDialect<StandardOpsDialect>();
  conversionTarget.addLegalDialect<tensor::TensorDialect>();
  conversionTarget.addLegalDialect<IREE::Flow::FlowDialect>();
}

void populateConvertTensorToFlowPatterns(MLIRContext *context,
                                         OwningRewritePatternList &patterns,
                                         TypeConverter &typeConverter) {
  patterns.insert<ConvertTensorCastPattern, ConvertTensorFromElementsPattern>(
      typeConverter, context);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
