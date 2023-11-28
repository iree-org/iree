// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO convolution ops to Linalg dialect.

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo-iree/Conversion/LegalizeToLinalgUtils.h"
#include "stablehlo-iree/Conversion/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {
/// Apply dilation and padding to the input of a convolution.
Value applyConvolutionPadding(Location loc, Value input,
                              DenseIntElementsAttr padding,
                              DenseIntElementsAttr lhsDilation,
                              llvm::ArrayRef<int64_t> dimMappings,
                              OpBuilder &rewriter) {
  if ((!padding || isSplatValue(padding, 0)) &&
      (!lhsDilation || isSplatValue(lhsDilation, 1))) {
    return input;
  }

  auto inputType = cast<ShapedType>(input.getType());
  int64_t rank = inputType.getRank();

  // Translate window padding into low/high padding.
  SmallVector<int64_t, 8> padLow(rank, 0);
  SmallVector<int64_t, 8> padHigh(rank, 0);
  if (padding) {
    // The padding attribute contains two values per dimension, but excludes the
    // batch and feature dimensions.
    assert(rank * 2 == padding.size() + 4 &&
           "There should be 2 padding values per dimension, i.e low and high.");
    for (int64_t i : llvm::seq<int64_t>(0, padding.size() / 2)) {
      int64_t dim = dimMappings[i];
      padLow[dim] = padding.getValues<int64_t>()[i * 2];
      padHigh[dim] = padding.getValues<int64_t>()[i * 2 + 1];
    }
  }

  // Translate input dilation into interior padding.
  SmallVector<int64_t, 8> padInterior(rank, 0);
  if (lhsDilation) {
    assert(rank == lhsDilation.size() + 2);
    for (int64_t i : llvm::seq<int64_t>(0, lhsDilation.size())) {
      int64_t dim = dimMappings[i];
      padInterior[dim] = lhsDilation.getValues<int64_t>()[i] - 1;
    }
  }

  IntegerType indexType = rewriter.getIntegerType(64);
  auto attrType = RankedTensorType::get({rank}, indexType);

  Value zero;
  if (auto complexType = dyn_cast<ComplexType>(inputType.getElementType())) {
    auto zeroElement = rewriter.getZeroAttr(complexType.getElementType());
    auto zeroAttr = rewriter.getArrayAttr({zeroElement, zeroElement});
    zero = rewriter.create<complex::ConstantOp>(loc, complexType, zeroAttr);
    zero = rewriter.create<tensor::FromElementsOp>(
        loc, RankedTensorType::get({}, complexType), zero);
  } else {
    zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(
                 RankedTensorType::get({}, inputType.getElementType())));
  }

  return rewriter.create<mlir::stablehlo::PadOp>(
      loc, input, zero, DenseIntElementsAttr::get(attrType, padLow),
      DenseIntElementsAttr::get(attrType, padHigh),
      DenseIntElementsAttr::get(attrType, padInterior));
}

/// If the ConvolutionOp has a window reversal, applies it to the filter.
Value applyConvolutionReversal(Location loc, OpBuilder &b,
                               mlir::stablehlo::ConvolutionOp op,
                               Value filter) {
  std::optional reversals = op.getWindowReversal();
  if (!reversals.has_value()) {
    return filter;
  }
  llvm::SmallVector<int64_t> reversedDims;
  for (auto [idx, reversed] :
       llvm::enumerate(reversals.value().getValues<bool>())) {
    if (reversed) {
      reversedDims.push_back(
          op.getDimensionNumbers().getKernelSpatialDimensions()[idx]);
    }
  }

  return b.create<mlir::stablehlo::ReverseOp>(
      loc, filter,
      mlir::DenseIntElementsAttr::get(
          RankedTensorType::get(reversedDims.size(), b.getI64Type()),
          reversedDims));
}

/// Returns true if the given `dimensionNumbers` from a stablehlo.convolution op
/// follows a canonical form:
///
/// * Input dimensions have order: (batch_count, spatial_dims,
///   input_channel_count).
/// * Filter dimensions have order: (spatial_dims, input_channel_count,
///   output_channel_count).
/// * Output dimensions have order: (batch_count, spatial_dims,
///   output_channel_count).
bool hasCanonicalDimensionNumbers(
    mlir::stablehlo::ConvDimensionNumbersAttr dimensionNumbers) {
  const int64_t inputSpatialRank =
      dimensionNumbers.getInputSpatialDimensions().size();
  // The dimensions for input should follow the order of
  // batch_count, spatial_dims..., input_feature_count.
  if (dimensionNumbers.getInputBatchDimension() != 0 ||
      dimensionNumbers.getInputFeatureDimension() != (inputSpatialRank + 1)) {
    return false;
  }

  const int64_t kernelSpatialRank =
      dimensionNumbers.getKernelSpatialDimensions().size();
  // The dimensions for filter should follow the order of
  // spatial_dims..., input_feature_count, num_output_feature_count.
  if (dimensionNumbers.getKernelInputFeatureDimension() != kernelSpatialRank ||
      dimensionNumbers.getKernelOutputFeatureDimension() !=
          (kernelSpatialRank + 1)) {
    return false;
  }

  const int64_t outputSpatialRank =
      dimensionNumbers.getOutputSpatialDimensions().size();
  // The dimensions for output should follow the order of
  // batch_count, spatial_dims.., output_feature_count.
  if (dimensionNumbers.getOutputBatchDimension() != 0 ||
      dimensionNumbers.getOutputFeatureDimension() != (outputSpatialRank + 1)) {
    return false;
  }

  if (inputSpatialRank != outputSpatialRank ||
      inputSpatialRank != kernelSpatialRank) {
    return false;
  }

  const int64_t *inputSpatialDim =
      dimensionNumbers.getInputSpatialDimensions().data();
  const int64_t *kernelSpatialDim =
      dimensionNumbers.getKernelSpatialDimensions().data();
  const int64_t *outputSpatialDim =
      dimensionNumbers.getOutputSpatialDimensions().data();
  // Check spatial dims are ordered correctly.
  for (int64_t i = 0; i < inputSpatialRank; ++i) {
    const int64_t dim = i + 1;
    if ((*inputSpatialDim++) != dim || (*outputSpatialDim++) != dim ||
        (*kernelSpatialDim++) != i) {
      return false;
    }
  }

  return true;
}

/// Converts stablehlo.conv operation to linalg named op. This only covers
/// normal convolution cases. The op must have canonical dimension numbers.
/// Depthwise convolution and pointwise convolution are not handled in the
/// conversion.
struct NormalConvolutionOpConversion final
    : OpConversionPattern<mlir::stablehlo::ConvolutionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!hasCanonicalDimensionNumbers(op.getDimensionNumbers())) {
      return failure();
    }
    if (op.getFeatureGroupCount() != 1u)
      return failure();
    if (op.getBatchGroupCount() != 1u)
      return failure();

    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    Value filter = adaptor.getRhs();
    filter = applyConvolutionReversal(loc, rewriter, op, filter);
    auto resultType = dyn_cast_or_null<ShapedType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }

    int64_t rank = resultType.getRank();

    // Immediately emit an EmptyOp for output tensors with zero dimension.
    if (llvm::is_contained(resultType.getShape(), 0)) {
      rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, resultType.getShape(),
                                                   resultType.getElementType());
      return success();
    }

    // The output shape is N spatial_dims F.
    SmallVector<Value, 8> dynSizes;
    if (resultType.isDynamicDim(0)) {
      dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, input, 0));
    }
    for (int64_t i = 1, e = rank - 1; i < e; ++i) {
      if (resultType.isDynamicDim(i)) {
        return rewriter.notifyMatchFailure(
            op, "expected output spatial dims to be static shapes");
      }
    }
    if (resultType.isDynamicDim(rank - 1)) {
      dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, filter, rank - 1));
    }
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(), dynSizes);
    Value zeroTensor = fillTensorWithZeros(rewriter, loc, emptyTensor);
    linalg::LinalgOp res;
    Attribute strides = op.getWindowStridesAttr();
    Attribute dilations = op.getRhsDilationAttr();

    // Apply padding and input dilation.
    llvm::SmallVector<int64_t> spatialDimMapping(rank - 2);
    std::iota(spatialDimMapping.begin(), spatialDimMapping.end(), 1);
    input = applyConvolutionPadding(loc, input, op.getPaddingAttr(),
                                    op.getLhsDilationAttr(), spatialDimMapping,
                                    rewriter);

    switch (rank) {
    case 2: {
      res = rewriter.create<linalg::MatmulOp>(
          loc, resultType, ValueRange{input, filter}, ValueRange{zeroTensor},
          linalg::getPrunedAttributeList(op));
      break;
    }
    case 3: {
      res = rewriter.create<linalg::Conv1DNwcWcfOp>(
          loc, resultType, ValueRange{input, filter}, ValueRange{zeroTensor},
          strides, dilations, linalg::getPrunedAttributeList(op));
      break;
    }
    case 4: {
      res = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
          loc, resultType, ValueRange{input, filter}, ValueRange{zeroTensor},
          strides, dilations, linalg::getPrunedAttributeList(op));
      break;
    }
    case 5: {
      res = rewriter.create<linalg::Conv3DNdhwcDhwcfOp>(
          loc, resultType, ValueRange{input, filter}, ValueRange{zeroTensor},
          strides, dilations, linalg::getPrunedAttributeList(op));
      break;
    }
    default: {
      return rewriter.notifyMatchFailure(op, "expected 1/2/3D conv op");
    }
    }
    rewriter.replaceOp(op, res.getOperation()->getResults());
    return success();
  }
};

/// Handles all possible inputs for the mlir::stablehlo::ConvolutionOp
struct ConvolutionOpGeneralConversion final
    : OpConversionPattern<mlir::stablehlo::ConvolutionOp> {
  using OpConversionPattern::OpConversionPattern;

  /// This lowering proceeds with the following steps:
  /// 1. Handle padding and dilation of the input
  /// 2. Handle padding and dilation of the window
  /// 3. Handle reversal of the window
  /// 4. If feature_group_count != 1:
  ///    - Reshape the input feature dimension, kernel output feature dimension,
  ///      and output feature dimension.
  ///    - Create the AffineExpr for the new dimension
  ///    - Conceptually, this splits the input feature and both output feature
  ///      dimensions and computes sets of convolutions with these partial views
  ///      of the values as if they were multiple convolutions combined in a
  ///      batch.
  /// 5: If batch_group_count != 1:
  ///    - Reshape the input batch dimension, kernel output feature dimension,
  ///      and output feature dimension.
  ///    - Create the AffineExpr for the new dimension
  ///    - Conceptually, this splits the input batch and both output feature
  ///      dimensions and computes sets of convolutions with these partial views
  ///      of the values as if they were multiple convolutions combined in a
  ///      batch.
  /// 6. For all dimensions not newly created by a reshape, create the
  ///    appropriate parallel and reduction dimensions to create a convolution.
  /// 7. Create the linalg.generic that computes the multiply-add
  /// 8. Reshape the output to the original shape if it was reshaped by the
  ///    feature or group count attributes.
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    auto resultType = dyn_cast_or_null<ShapedType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }

    auto reshapedResultShape = resultType.getShape().vec();
    if (!resultType.hasStaticShape())
      return failure();

    // Immediately emit an EmptyOp for output tensors with zero dimension.
    if (llvm::is_contained(reshapedResultShape, 0)) {
      rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, reshapedResultShape,
                                                   resultType.getElementType());
      return success();
    }

    mlir::stablehlo::ConvDimensionNumbersAttr dimensionNumbers =
        op.getDimensionNumbers();
    int64_t inputBatchDimension = dimensionNumbers.getInputBatchDimension();
    int64_t inputFeatureDimension = dimensionNumbers.getInputFeatureDimension();
    ArrayRef<int64_t> inputSpatialDimensions =
        dimensionNumbers.getInputSpatialDimensions();

    int64_t kernelInputFeatureDimension =
        dimensionNumbers.getKernelInputFeatureDimension();
    int64_t kernelOutputFeatureDimension =
        dimensionNumbers.getKernelOutputFeatureDimension();
    ArrayRef<int64_t> kernelSpatialDimensions =
        dimensionNumbers.getKernelSpatialDimensions();

    int64_t outputBatchDimension = dimensionNumbers.getOutputBatchDimension();
    int64_t outputFeatureDimension =
        dimensionNumbers.getOutputFeatureDimension();
    ArrayRef<int64_t> outputSpatialDimensions =
        dimensionNumbers.getOutputSpatialDimensions();

    size_t featureGroupCount = op.getFeatureGroupCount();
    size_t batchGroupCount = op.getBatchGroupCount();

    if (op.getFeatureGroupCount() != 1 && op.getBatchGroupCount() != 1) {
      return rewriter.notifyMatchFailure(
          op, "only one of feature and batch group counts can be non-one");
    }

    // Decompose the convolution into an initial padding
    Value modifiedLhs = applyConvolutionPadding(
        op.getLoc(), adaptor.getLhs(), adaptor.getPaddingAttr(),
        adaptor.getLhsDilationAttr(),
        op.getDimensionNumbers().getInputSpatialDimensions(), rewriter);
    Value modifiedRhs = applyConvolutionPadding(
        op.getLoc(), adaptor.getRhs(), nullptr, adaptor.getRhsDilationAttr(),
        op.getDimensionNumbers().getKernelSpatialDimensions(), rewriter);
    modifiedRhs = applyConvolutionReversal(loc, rewriter, op, modifiedRhs);

    // Non-one values for feature or batch group counts will result in reshaped
    // inputs and outputs. These mappings are used to keep track of the the new
    // index after reshaping has possibly inserted new dimensions.
    auto paddedLhsType = cast<ShapedType>(modifiedLhs.getType());
    auto paddedRhsType = cast<ShapedType>(modifiedRhs.getType());
    SmallVector<int64_t> lhsIndexMapping(paddedLhsType.getRank());
    std::iota(lhsIndexMapping.begin(), lhsIndexMapping.end(), 0);
    SmallVector<int64_t> rhsIndexMapping(paddedRhsType.getRank());
    std::iota(rhsIndexMapping.begin(), rhsIndexMapping.end(), 0);
    SmallVector<int64_t> resultIndexMapping(resultType.getRank());
    std::iota(resultIndexMapping.begin(), resultIndexMapping.end(), 0);
    auto updateDimMappingFromOffset =
        [](llvm::SmallVectorImpl<int64_t> &mapping, int64_t offset) {
          for (auto &mappingElt : llvm::drop_begin(mapping, offset)) {
            mappingElt += 1;
          }
        };

    // The rest of this code prepares the inputs and a single linalg::GenericOp
    // to execute the convolution. The final linalg::GenericOp will be iterated
    // through based on the following eventual maps.
    SmallVector<AffineExpr, 2> srcExprs(paddedLhsType.getRank());
    SmallVector<AffineExpr, 2> windowExprs(paddedRhsType.getRank());
    SmallVector<AffineExpr, 2> dstExprs(reshapedResultShape.size());
    int64_t nextDim = 0;
    int64_t rank = resultType.getRank();

    auto reshapeShapeVector = [](llvm::ArrayRef<int64_t> oldShape,
                                 llvm::SmallVectorImpl<int64_t> &newShape,
                                 int64_t reshapedDim, int64_t factor) {
      newShape.reserve(oldShape.size() + 1);
      for (int64_t i : llvm::seq<int64_t>(0, oldShape.size())) {
        if (i == reshapedDim) {
          newShape.push_back(factor);
          newShape.push_back(oldShape[reshapedDim] / factor);
        } else {
          newShape.push_back(oldShape[i]);
        }
      }
    };

    // If batch or feature count groupings exist, represent this through
    // reshaping the input to have an additional dimension that these groupings
    // exist along, and reduce in that dimension
    SmallVector<utils::IteratorType, 3> iterationLoops;
    if (featureGroupCount != 1) {
      AffineExpr parallelDim = mlir::getAffineDimExpr(nextDim++, ctx);
      iterationLoops.push_back(utils::IteratorType::parallel);
      // Reshape LHS
      {
        srcExprs.insert(srcExprs.begin() + inputFeatureDimension, parallelDim);
        auto prevDimsRef = paddedLhsType.getShape();
        llvm::SmallVector<int64_t> newShape;
        reshapeShapeVector(prevDimsRef, newShape, inputFeatureDimension,
                           featureGroupCount);
        updateDimMappingFromOffset(lhsIndexMapping, inputFeatureDimension);
        modifiedLhs = rewriter.create<mlir::stablehlo::ReshapeOp>(
            loc,
            RankedTensorType::get(newShape, paddedLhsType.getElementType()),
            modifiedLhs);
      }

      // Reshape RHS
      {
        windowExprs.insert(windowExprs.begin() + kernelOutputFeatureDimension,
                           parallelDim);
        auto prevDimsRef = paddedRhsType.getShape();
        llvm::SmallVector<int64_t> newShape;
        reshapeShapeVector(prevDimsRef, newShape, kernelOutputFeatureDimension,
                           featureGroupCount);
        updateDimMappingFromOffset(rhsIndexMapping,
                                   kernelOutputFeatureDimension);
        modifiedRhs = rewriter.create<mlir::stablehlo::ReshapeOp>(
            loc,
            RankedTensorType::get(newShape, paddedRhsType.getElementType()),
            modifiedRhs);
      }
      // Prepare reshaped output shape
      {
        dstExprs.insert(dstExprs.begin() + outputFeatureDimension, parallelDim);
        updateDimMappingFromOffset(resultIndexMapping, outputFeatureDimension);
        reshapedResultShape.insert(reshapedResultShape.begin() +
                                       outputFeatureDimension,
                                   featureGroupCount);
        reshapedResultShape[outputFeatureDimension + 1] /= featureGroupCount;
      }
    }

    if (batchGroupCount != 1) {
      iterationLoops.push_back(utils::IteratorType::parallel);
      AffineExpr parallelDim = mlir::getAffineDimExpr(nextDim++, ctx);
      // Reshape LHS
      {
        srcExprs.insert(srcExprs.begin() + inputBatchDimension, parallelDim);
        ArrayRef<int64_t> prevDimsRef = paddedLhsType.getShape();
        llvm::SmallVector<int64_t> newShape;
        reshapeShapeVector(prevDimsRef, newShape, inputBatchDimension,
                           batchGroupCount);
        updateDimMappingFromOffset(lhsIndexMapping, inputBatchDimension);
        modifiedLhs = rewriter.create<mlir::stablehlo::ReshapeOp>(
            op.getLoc(),
            RankedTensorType::get(newShape, paddedLhsType.getElementType()),
            modifiedLhs);
      }

      // Reshape RHS
      {
        windowExprs.insert(windowExprs.begin() + kernelOutputFeatureDimension,
                           parallelDim);
        ArrayRef<int64_t> prevDimsRef = paddedRhsType.getShape();
        llvm::SmallVector<int64_t> newShape;
        reshapeShapeVector(prevDimsRef, newShape, kernelOutputFeatureDimension,
                           batchGroupCount);
        updateDimMappingFromOffset(rhsIndexMapping,
                                   kernelOutputFeatureDimension);
        modifiedRhs = rewriter.create<mlir::stablehlo::ReshapeOp>(
            op.getLoc(),
            RankedTensorType::get(newShape, paddedRhsType.getElementType()),
            modifiedRhs);
      }
      // Prepare reshaped output shape
      {
        int64_t outputFeatureDim = resultIndexMapping[outputFeatureDimension];
        dstExprs.insert(dstExprs.begin() + outputFeatureDim, parallelDim);
        updateDimMappingFromOffset(resultIndexMapping, outputFeatureDimension);
        reshapedResultShape.insert(
            reshapedResultShape.begin() + outputFeatureDim, batchGroupCount);
        reshapedResultShape[outputFeatureDim + 1] /= batchGroupCount;
      }
    }

    // Handle input feature dimension
    {
      iterationLoops.push_back(utils::IteratorType::reduction);
      AffineExpr inputFeatureDim = mlir::getAffineDimExpr(nextDim++, ctx);
      srcExprs[lhsIndexMapping[inputFeatureDimension]] = inputFeatureDim;
      windowExprs[rhsIndexMapping[kernelInputFeatureDimension]] =
          inputFeatureDim;
    }

    // Handle output feature dimension
    {
      iterationLoops.push_back(utils::IteratorType::parallel);
      AffineExpr outputFeatureDim = mlir::getAffineDimExpr(nextDim++, ctx);
      dstExprs[resultIndexMapping[outputFeatureDimension]] = outputFeatureDim;
      windowExprs[rhsIndexMapping[kernelOutputFeatureDimension]] =
          outputFeatureDim;
    }

    // Handle spatial Dimensions
    int64_t numSpatialDims = rank - 2;
    for (int64_t i = 0; i < numSpatialDims; ++i) {
      iterationLoops.push_back(utils::IteratorType::parallel);
      iterationLoops.push_back(utils::IteratorType::reduction);
      AffineExpr dim0 = mlir::getAffineDimExpr(nextDim++, ctx);
      AffineExpr dim1 = mlir::getAffineDimExpr(nextDim++, ctx);

      AffineExpr stride = dim0;
      if (op.getWindowStrides().has_value())
        stride = stride * op.getWindowStrides().value().getValues<int64_t>()[i];
      AffineExpr srcExpr = stride + dim1;

      srcExprs[lhsIndexMapping[inputSpatialDimensions[i]]] = srcExpr;
      dstExprs[resultIndexMapping[outputSpatialDimensions[i]]] = dim0;
      windowExprs[rhsIndexMapping[kernelSpatialDimensions[i]]] = dim1;
    }

    // Handle batch dimension
    {
      iterationLoops.push_back(utils::IteratorType::parallel);
      AffineExpr batchDim = mlir::getAffineDimExpr(nextDim++, ctx);

      srcExprs[lhsIndexMapping[inputBatchDimension]] = batchDim;
      dstExprs[resultIndexMapping[outputBatchDimension]] = batchDim;
    }

    // Finally, create the computation
    auto inferredMaps =
        AffineMap::inferFromExprList({srcExprs, windowExprs, dstExprs});

    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, reshapedResultShape, resultType.getElementType());
    Value zeroTensor = fillTensorWithZeros(rewriter, loc, emptyTensor);

    Value convolved =
        rewriter
            .create<linalg::GenericOp>(
                loc,
                /*resultTensors=*/
                llvm::ArrayRef<Type>(zeroTensor.getType()),
                /*inputs=*/
                llvm::ArrayRef<Value>({modifiedLhs, modifiedRhs}),
                /*outputs=*/llvm::ArrayRef<Value>(zeroTensor), inferredMaps,
                iterationLoops,
                /*bodyBuild=*/
                [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange) {
                  ImplicitLocOpBuilder builder(nestedLoc, nestedBuilder);
                  linalg::Conv2DOp::regionBuilder(
                      builder, *builder.getInsertionBlock(), {});
                },
                linalg::getPrunedAttributeList(op))
            .getResult(0);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, resultType,
                                                            convolved);

    return success();
  }
};

/// Converts stablehlo.convolution operation to
/// linalg.depthwise_conv_2d_input_nhwc_filter_hwcf op or
/// depthwise_conv_2d_input_nhwc_filter_hwc op.
struct DepthwiseConvolutionOpConversion final
    : OpConversionPattern<mlir::stablehlo::ConvolutionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getBatchGroupCount() != 1)
      return failure();
    // Fall into the normal convolution cases.
    if (op.getFeatureGroupCount() == 1)
      return failure();

    const mlir::stablehlo::ConvDimensionNumbersAttr &dimensionNumbers =
        op.getDimensionNumbers();
    const int64_t spatialRank =
        dimensionNumbers.getInputSpatialDimensions().size();
    if (spatialRank == 0 || spatialRank > 3) {
      return rewriter.notifyMatchFailure(op, "only support up to 3D for now");
    }

    // Make sure that this is depthwise convolution.
    int64_t inputFeatureDim = dimensionNumbers.getInputFeatureDimension();
    int64_t inputFeatureCount =
        cast<ShapedType>(op.getLhs().getType()).getDimSize(inputFeatureDim);
    if (static_cast<int64_t>(op.getFeatureGroupCount()) != inputFeatureCount) {
      return rewriter.notifyMatchFailure(op, "not depth-wise convolution");
    }

    // Make sure that this convolution has a canonical form.
    if (!hasCanonicalDimensionNumbers(dimensionNumbers)) {
      return rewriter.notifyMatchFailure(op, "does not have canonical form");
    }

    Attribute windowStrides;
    if (op.getWindowStrides()) {
      windowStrides = op.getWindowStrides().value();
    } else {
      windowStrides = SplatElementsAttr::get(
          VectorType::get({spatialRank}, rewriter.getI64Type()),
          rewriter.getI64IntegerAttr(1));
    }

    Attribute rhsDilation;
    if (op.getRhsDilation()) {
      rhsDilation = op.getRhsDilation().value();
    } else {
      rhsDilation = SplatElementsAttr::get(
          VectorType::get({spatialRank}, rewriter.getI64Type()),
          rewriter.getI64IntegerAttr(1));
    }

    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    Value filter = adaptor.getRhs();
    auto resultType = dyn_cast_or_null<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }
    if (!resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected output has static shapes");
    }

    // Immediately emit an EmptyOp for output tensors with zero dimension.
    if (llvm::is_contained(resultType.getShape(), 0)) {
      rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, resultType.getShape(),
                                                   resultType.getElementType());
      return success();
    }

    // Apply padding and input dilation.
    llvm::SmallVector<int64_t> spatialDimMapping(spatialRank);
    std::iota(spatialDimMapping.begin(), spatialDimMapping.end(), 1);
    input = applyConvolutionPadding(loc, input, op.getPaddingAttr(),
                                    op.getLhsDilationAttr(), spatialDimMapping,
                                    rewriter);

    auto filterDims =
        llvm::to_vector(cast<ShapedType>(op.getRhs().getType()).getShape());

    auto getReassociationIndicesToCollapseLastTwoDims = [](Value v) {
      SmallVector<ReassociationIndices> reassociations;
      int64_t rank = cast<ShapedType>(v.getType()).getRank();
      for (int64_t i = 0; i < rank - 1; ++i)
        reassociations.emplace_back(1, i);
      reassociations.back().push_back(rank - 1);
      return reassociations;
    };

    int64_t kernelInputFeatureDimension =
        dimensionNumbers.getKernelInputFeatureDimension();
    int64_t kernelOutputFeatureDimension =
        dimensionNumbers.getKernelOutputFeatureDimension();
    if (filterDims[kernelInputFeatureDimension] *
            filterDims[kernelOutputFeatureDimension] !=
        static_cast<int64_t>(op.getFeatureGroupCount())) {
      // For cases where channel multiplier != 1

      // Reshaping filter shape
      //   [filter_height, filter_width, 1, kernel-output-feature].
      // to
      //   [filter_height, filter_width, feature_group_count,
      //      kernel-output-feature/feature_group_count ]
      SmallVector<int64_t> reshapedFilterDims;
      reshapedFilterDims.assign(filterDims.begin(), filterDims.end());
      Value reshapedFilter = filter;
      if (filterDims[kernelInputFeatureDimension] == 1) {
        reshapedFilterDims[kernelInputFeatureDimension] =
            op.getFeatureGroupCount();
        reshapedFilterDims[kernelOutputFeatureDimension] /=
            op.getFeatureGroupCount();
        auto reshapedFilterType = RankedTensorType::get(
            reshapedFilterDims,
            cast<ShapedType>(op.getRhs().getType()).getElementType());

        reshapedFilter = rewriter.create<mlir::stablehlo::ReshapeOp>(
            loc, reshapedFilterType, filter);
      }

      ArrayRef<int64_t> outputDims = resultType.getShape();
      int64_t channelMultiplier = reshapedFilterDims.back();
      SmallVector<int64_t> reshapedOutputDims;
      reshapedOutputDims.assign(outputDims.begin(), outputDims.end());
      reshapedOutputDims.push_back(channelMultiplier);
      reshapedOutputDims[reshapedOutputDims.size() - 2] /= channelMultiplier;

      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, reshapedOutputDims, resultType.getElementType());
      Value zeroTensor = fillTensorWithZeros(rewriter, loc, emptyTensor);

      auto reshapedOutputType = RankedTensorType::get(
          reshapedOutputDims, resultType.getElementType());
      Value conv;
      switch (spatialRank) {
      case 1: {
        conv =
            rewriter
                .create<linalg::DepthwiseConv1DNwcWcmOp>(
                    loc, reshapedOutputType, ValueRange{input, reshapedFilter},
                    ValueRange{zeroTensor}, windowStrides, rhsDilation,
                    linalg::getPrunedAttributeList(op))
                .getResult(0);
        break;
      }
      case 2: {
        conv =
            rewriter
                .create<linalg::DepthwiseConv2DNhwcHwcmOp>(
                    loc, reshapedOutputType, ValueRange{input, reshapedFilter},
                    ValueRange{zeroTensor}, windowStrides, rhsDilation,
                    linalg::getPrunedAttributeList(op))
                .getResult(0);
        break;
      }
      case 3: {
        conv =
            rewriter
                .create<linalg::DepthwiseConv3DNdhwcDhwcmOp>(
                    loc, reshapedOutputType, ValueRange{input, reshapedFilter},
                    ValueRange{zeroTensor}, windowStrides, rhsDilation,
                    linalg::getPrunedAttributeList(op))
                .getResult(0);
        break;
      }
      default:
        llvm_unreachable("Unhandled case");
      }

      // Create a Linalg reshape op that converts the output from 5 dimensions
      // into 4 dimensions (by collapsing the last two dimensions). This is
      // needed because linalg.depthwise_conv_2d_input_nhwc_filter_hwcf returns
      // 5 dimensions for the output.
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, resultType, conv,
          getReassociationIndicesToCollapseLastTwoDims(conv));
    } else {
      // For cases where channel multiplier == 1
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), resultType.getElementType());
      Value zeroTensor = fillTensorWithZeros(rewriter, loc, emptyTensor);

      // Create a Linalg reshape op that converts the filter from 4 dimensions
      // into 3 dimensions (by droping the unit dimension). This is needed
      // because linalg.depthwise_conv_2d_input_nhwc_filter_hwc expects 3
      // dimensions for the filter.

      filterDims[filterDims.size() - 2] =
          static_cast<int64_t>(op.getFeatureGroupCount());
      filterDims.pop_back();

      RankedTensorType filterShape =
          RankedTensorType::get(filterDims, op.getType().getElementType());

      Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
          loc, filterShape, filter,
          getReassociationIndicesToCollapseLastTwoDims(filter));

      switch (spatialRank) {
      case 1:
        rewriter.replaceOpWithNewOp<linalg::DepthwiseConv1DNwcWcOp>(
            op, resultType, ValueRange{input, reshapedFilter},
            ValueRange{zeroTensor}, windowStrides, rhsDilation,
            linalg::getPrunedAttributeList(op));
        break;
      case 2:
        rewriter.replaceOpWithNewOp<linalg::DepthwiseConv2DNhwcHwcOp>(
            op, resultType, ValueRange{input, reshapedFilter},
            ValueRange{zeroTensor}, windowStrides, rhsDilation,
            linalg::getPrunedAttributeList(op));
        break;
      case 3:
        rewriter.replaceOpWithNewOp<linalg::DepthwiseConv3DNdhwcDhwcOp>(
            op, resultType, ValueRange{input, reshapedFilter},
            ValueRange{zeroTensor}, windowStrides, rhsDilation,
            linalg::getPrunedAttributeList(op));
        break;
      }
    }

    return success();
  }
};

} // namespace

namespace detail {
void populateStableHloConvolutionToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns) {
  // Ensure specialized patterns are higher priority than their generic
  // versions.
  patterns
      ->add<NormalConvolutionOpConversion, DepthwiseConvolutionOpConversion>(
          typeConverter, context, PatternBenefit(2));

  patterns->add<ConvolutionOpGeneralConversion>(typeConverter, context);
}
} // namespace detail
} // namespace mlir::iree_compiler::stablehlo
