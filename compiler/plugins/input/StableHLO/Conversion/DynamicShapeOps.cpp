// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowerings for the StableHLO ops whose shapes are known only at runtime.

#include "compiler/plugins/input/StableHLO/Conversion/LegalizeToLinalgUtils.h"
#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

Value getEmptyTensorFor(OpBuilder &builder, Location loc,
                        RankedTensorType resultType, Operation *op,
                        ValueRange operands) {
  SmallVector<Value> sizes;
  if (!resultType.hasStaticShape()) {
    SmallVector<Value> shapes;
    if (failed(cast<InferShapedTypeOpInterface>(op).reifyReturnTypeShapes(
            builder, operands, shapes))) {
      return {};
    }
    for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
      if (resultType.isDynamicDim(dim)) {
        Value index = arith::ConstantIndexOp::create(builder, loc, dim);
        sizes.push_back(
            tensor::ExtractOp::create(builder, loc, shapes[0], index));
      }
    }
  }
  if (sparse_tensor::getSparseTensorEncoding(resultType)) {
    return bufferization::AllocTensorOp::create(builder, loc, resultType, sizes,
                                                Value(), IntegerAttr());
  }
  return tensor::EmptyOp::create(builder, loc, resultType, sizes);
}

Value fillTensorWithZeros(OpBuilder &builder, Location loc, Value tensor) {
  Type elementType = cast<RankedTensorType>(tensor.getType()).getElementType();
  Value zero;
  if (auto complexType = dyn_cast<ComplexType>(elementType)) {
    auto zeroElement = builder.getZeroAttr(complexType.getElementType());
    zero = complex::ConstantOp::create(
        builder, loc, complexType,
        builder.getArrayAttr({zeroElement, zeroElement}));
  } else {
    zero = arith::ConstantOp::create(builder, loc,
                                     builder.getZeroAttr(elementType));
  }
  return linalg::FillOp::create(builder, loc, zero, tensor).result();
}

// Copied verbatim from StablehloLegalizeToLinalg.cpp, where it is file-local.
Value extractIndexFromTensor(OpBuilder &builder, Location loc, Value tensor,
                             ShapedType originalType,
                             ArrayRef<Value> tensorIndex = {}) {
  Value extracted =
      tensor::ExtractOp::create(builder, loc, tensor, tensorIndex);
  if (extracted.getType().isIndex()) {
    return extracted;
  }
  return originalType.getElementType().isUnsignedInteger()
             ? builder.createOrFold<arith::IndexCastUIOp>(
                   loc, builder.getIndexType(), extracted)
             : builder.createOrFold<arith::IndexCastOp>(
                   loc, builder.getIndexType(), extracted);
}

Value extractIndex(OpBuilder &b, Location loc, Value shapeTensor, int64_t i) {
  Value index = arith::ConstantIndexOp::create(b, loc, i);
  Value element = tensor::ExtractOp::create(b, loc, shapeTensor, index);
  if (element.getType().isIndex()) {
    return element;
  }
  return arith::IndexCastOp::create(b, loc, b.getIndexType(), element);
}

// tensor.reshape takes the shape tensor as-is, and Flow lowers it.
struct DynamicReshapeOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicReshapeOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(
        op, resultType, adaptor.getOperand(), adaptor.getOutputShape());
    return success();
  }
};

struct DynamicPadGeometry {
  SmallVector<Value> scratchDynamicDims, resultDynamicDims;
  SmallVector<OpFoldResult> insertSizes, insertOffsets, insertStrides;
  SmallVector<OpFoldResult> extractOffsets, extractSizes;
  Value nonempty;
};

DynamicPadGeometry
computeDynamicPadGeometry(OpBuilder &rewriter, Location loc,
                          mlir::stablehlo::DynamicPadOp::Adaptor adaptor,
                          RankedTensorType resultType) {
  int64_t rank = resultType.getRank();
  DynamicPadGeometry geometry;
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  geometry.nonempty = rewriter.createOrFold<arith::ConstantIntOp>(loc, 1, 1);
  geometry.insertSizes =
      tensor::getMixedSizes(rewriter, loc, adaptor.getOperand());
  for (int64_t i = 0; i < rank; ++i) {
    Value dim =
        getValueOrCreateConstantIndexOp(rewriter, loc, geometry.insertSizes[i]);
    Value low = extractIndex(rewriter, loc, adaptor.getEdgePaddingLow(), i);
    Value high = extractIndex(rewriter, loc, adaptor.getEdgePaddingHigh(), i);
    Value interior =
        extractIndex(rewriter, loc, adaptor.getInteriorPadding(), i);

    Value lowPos = arith::MaxSIOp::create(rewriter, loc, low, zero);
    Value highPos = arith::MaxSIOp::create(rewriter, loc, high, zero);
    Value lowNeg = arith::MaxSIOp::create(
        rewriter, loc, arith::SubIOp::create(rewriter, loc, zero, low), zero);

    Value dimMinusOne = arith::SubIOp::create(rewriter, loc, dim, one);
    Value clampedDimMinusOne =
        arith::MaxSIOp::create(rewriter, loc, dimMinusOne, zero);
    Value interiorTotal =
        arith::MulIOp::create(rewriter, loc, clampedDimMinusOne, interior);
    Value dimAndInterior =
        arith::AddIOp::create(rewriter, loc, dim, interiorTotal);

    // The scratch tensor's size depends on lowPos/highPos, which are
    // runtime values even where the result type's dim is static, so every
    // dim of the scratch tensor is dynamic.
    Value dimAndLowPos =
        arith::AddIOp::create(rewriter, loc, dimAndInterior, lowPos);
    Value filledDim =
        arith::AddIOp::create(rewriter, loc, dimAndLowPos, highPos);
    geometry.scratchDynamicDims.push_back(filledDim);

    if (resultType.isDynamicDim(i)) {
      Value dimAndLow =
          arith::AddIOp::create(rewriter, loc, dimAndInterior, low);
      Value resultDim = arith::AddIOp::create(rewriter, loc, dimAndLow, high);
      geometry.extractSizes.push_back(resultDim);
      geometry.resultDynamicDims.push_back(resultDim);
    } else {
      geometry.extractSizes.push_back(
          rewriter.getIndexAttr(resultType.getDimSize(i)));
    }

    geometry.insertOffsets.push_back(lowPos);
    geometry.insertStrides.push_back(
        arith::AddIOp::create(rewriter, loc, interior, one).getResult());
    geometry.extractOffsets.push_back(lowNeg);
    Value size = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                 geometry.extractSizes.back());
    Value positive = rewriter.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, size, zero);
    geometry.nonempty =
        rewriter.createOrFold<arith::AndIOp>(loc, geometry.nonempty, positive);
  }
  return geometry;
}

Value createPaddedTensor(OpBuilder &rewriter, Location loc, Value operand,
                         Value paddingValue, RankedTensorType resultType,
                         const DynamicPadGeometry &geometry) {
  int64_t rank = resultType.getRank();
  SmallVector<int64_t> scratchShape(rank, ShapedType::kDynamic);
  Value empty = tensor::EmptyOp::create(rewriter, loc, scratchShape,
                                        resultType.getElementType(),
                                        geometry.scratchDynamicDims);
  Value filled =
      linalg::FillOp::create(rewriter, loc, paddingValue, empty).result();
  Value inserted = tensor::InsertSliceOp::create(
                       rewriter, loc, operand, filled, geometry.insertOffsets,
                       geometry.insertSizes, geometry.insertStrides)
                       .getResult();
  // Negative edge padding crops, which the insert cannot express.
  SmallVector<OpFoldResult> extractStrides(rank, rewriter.getIndexAttr(1));
  return tensor::ExtractSliceOp::create(rewriter, loc, resultType, inserted,
                                        geometry.extractOffsets,
                                        geometry.extractSizes, extractStrides);
}

// tensor.pad has no interior padding, and whether the interior amounts are
// zero is unknown here, so the strided insert covers every case.
struct DynamicPadOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicPadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicPadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    auto operandType =
        dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
    if (!resultType || !operandType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }

    Value paddingValue = rewriter.createOrFold<tensor::ExtractOp>(
        loc, adaptor.getPaddingValue());

    auto geometry =
        computeDynamicPadGeometry(rewriter, loc, adaptor, resultType);
    // Empty crops need no scratch buffer. In particular, runtime convolution
    // padding can remove an arbitrarily large dilated input entirely.
    auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{resultType},
                                  geometry.nonempty,
                                  /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value result = createPaddedTensor(rewriter, loc, adaptor.getOperand(),
                                        paddingValue, resultType, geometry);
      scf::YieldOp::create(rewriter, loc, result);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      Value resultEmpty = tensor::EmptyOp::create(rewriter, loc, resultType,
                                                  geometry.resultDynamicDims);
      scf::YieldOp::create(rewriter, loc, resultEmpty);
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorTieShapeOp>(
        op, resultType, ifOp.getResult(0), geometry.resultDynamicDims);
    return success();
  }
};

struct ConvolutionWindow {
  SmallVector<int64_t> strides, lhsDilations, rhsDilations, padding;
};

ConvolutionWindow getConvolutionWindow(mlir::stablehlo::ConvolutionOp op) {
  int64_t spatialRank =
      op.getDimensionNumbers().getInputSpatialDimensions().size();
  ConvolutionWindow window{SmallVector<int64_t>(spatialRank, 1),
                           SmallVector<int64_t>(spatialRank, 1),
                           SmallVector<int64_t>(spatialRank, 1),
                           SmallVector<int64_t>(2 * spatialRank, 0)};
  if (op.getWindowStrides()) {
    window.strides = llvm::to_vector(*op.getWindowStrides());
  }
  if (op.getLhsDilation()) {
    window.lhsDilations = llvm::to_vector(*op.getLhsDilation());
  }
  if (op.getRhsDilation()) {
    window.rhsDilations = llvm::to_vector(*op.getRhsDilation());
  }
  if (op.getPadding()) {
    window.padding = llvm::to_vector(op.getPadding()->getValues<int64_t>());
  }
  return window;
}

struct ConvolutionResultShape {
  SmallVector<Value> dynamicDims;
  Value nonempty;
};

ConvolutionResultShape
computeConvolutionResultShape(OpBuilder &rewriter,
                              mlir::stablehlo::ConvolutionOp op, Value input,
                              Value filter, RankedTensorType resultType,
                              const ConvolutionWindow &window) {
  Location loc = op.getLoc();
  int64_t rank = resultType.getRank();
  auto constant = [&](int64_t value) -> Value {
    return rewriter.createOrFold<arith::ConstantIndexOp>(loc, value);
  };
  Value zero = constant(0);
  Value one = constant(1);
  // An empty input/kernel stays empty under dilation (StableHLO C25).
  auto dilate = [&](Value size, int64_t dilation) -> Value {
    if (dilation == 1) {
      return size;
    }
    Value empty = rewriter.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, size, zero);
    Value span = rewriter.createOrFold<arith::SubIOp>(loc, size, one);
    span = rewriter.createOrFold<arith::MulIOp>(loc, span, constant(dilation));
    Value dilated = rewriter.createOrFold<arith::AddIOp>(loc, span, one);
    return rewriter.createOrFold<arith::SelectOp>(loc, empty, zero, dilated);
  };

  ConvolutionResultShape shape;
  shape.nonempty = rewriter.createOrFold<arith::ConstantIntOp>(loc, 1, 1);
  for (int64_t d = 0; d < rank; ++d) {
    Value size;
    if (!resultType.isDynamicDim(d)) {
      size = constant(resultType.getDimSize(d));
    } else if (d == 0) {
      size = rewriter.createOrFold<tensor::DimOp>(loc, input, 0);
      size = rewriter.createOrFold<arith::DivUIOp>(
          loc, size, constant(op.getBatchGroupCount()));
    } else if (d == rank - 1) {
      size = rewriter.createOrFold<tensor::DimOp>(loc, filter, rank - 1);
    } else {
      int64_t s = d - 1;
      Value inputSize = rewriter.createOrFold<tensor::DimOp>(loc, input, d);
      Value kernelSize = rewriter.createOrFold<tensor::DimOp>(loc, filter, s);
      Value padded = dilate(inputSize, window.lhsDilations[s]);
      padded = rewriter.createOrFold<arith::AddIOp>(
          loc, padded, constant(window.padding[2 * s]));
      padded = rewriter.createOrFold<arith::AddIOp>(
          loc, padded, constant(window.padding[2 * s + 1]));
      Value dilatedKernel = dilate(kernelSize, window.rhsDilations[s]);
      Value span =
          rewriter.createOrFold<arith::SubIOp>(loc, padded, dilatedKernel);
      // For a fitting window span is nonnegative, so truncation is floor.
      // The select discards this count when the window does not fit.
      Value count = rewriter.createOrFold<arith::DivSIOp>(
          loc, span, constant(window.strides[s]));
      count = rewriter.createOrFold<arith::AddIOp>(loc, count, one);
      Value fits = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, span, zero);
      Value hasInput = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, padded, zero);
      fits = rewriter.createOrFold<arith::AndIOp>(loc, fits, hasInput);
      size = rewriter.createOrFold<arith::SelectOp>(loc, fits, count, zero);
    }
    if (resultType.isDynamicDim(d)) {
      shape.dynamicDims.push_back(size);
    }
    Value positive = rewriter.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, size, zero);
    shape.nonempty =
        rewriter.createOrFold<arith::AndIOp>(loc, shape.nonempty, positive);
  }
  return shape;
}

Value padConvolutionInput(OpBuilder &rewriter, Location loc, Value input,
                          const ConvolutionWindow &window) {
  int64_t rank = cast<RankedTensorType>(input.getType()).getRank();
  if (llvm::any_of(window.padding, [](int64_t v) { return v != 0; }) ||
      llvm::any_of(window.lhsDilations, [](int64_t v) { return v != 1; })) {
    SmallVector<int64_t> lows(rank, 0), highs(rank, 0), interiors(rank, 0);
    for (int64_t i = 0; i < rank - 2; ++i) {
      lows[i + 1] = window.padding[2 * i];
      highs[i + 1] = window.padding[2 * i + 1];
      interiors[i + 1] = window.lhsDilations[i] - 1;
    }
    auto shapeType = RankedTensorType::get({rank}, rewriter.getI64Type());
    auto shapeConstant = [&](ArrayRef<int64_t> values) -> Value {
      return arith::ConstantOp::create(
          rewriter, loc, DenseIntElementsAttr::get(shapeType, values));
    };
    auto inputType = cast<RankedTensorType>(input.getType());
    SmallVector<int64_t> paddedShape(inputType.getShape());
    for (int64_t i = 1; i < rank - 1; ++i) {
      paddedShape[i] = ShapedType::kDynamic;
    }
    Value scalarEmpty = tensor::EmptyOp::create(
        rewriter, loc, ArrayRef<int64_t>{}, inputType.getElementType());
    Value padValue = fillTensorWithZeros(rewriter, loc, scalarEmpty);
    return mlir::stablehlo::DynamicPadOp::create(
        rewriter, loc,
        RankedTensorType::get(paddedShape, inputType.getElementType()), input,
        padValue, shapeConstant(lows), shapeConstant(highs),
        shapeConstant(interiors));
  }
  return input;
}

Value splitConvolutionGroupDimension(OpBuilder &rewriter, Location loc,
                                     Value value, int64_t dimension,
                                     int64_t groups) {
  auto type = cast<RankedTensorType>(value.getType());
  int64_t rank = type.getRank();
  SmallVector<int64_t> shape(type.getShape());
  if (!ShapedType::isDynamic(shape[dimension])) {
    shape[dimension] /= groups;
  }
  shape.insert(shape.begin() + dimension, groups);
  SmallVector<ReassociationIndices> reassociation;
  for (int64_t d = 0; d < rank; ++d) {
    if (d < dimension) {
      reassociation.push_back({d});
    } else if (d == dimension) {
      reassociation.push_back({d, d + 1});
    } else {
      reassociation.push_back({d + 1});
    }
  }
  return tensor::ExpandShapeOp::create(
      rewriter, loc, RankedTensorType::get(shape, type.getElementType()), value,
      reassociation);
}

Value createGroupedConvolution(OpBuilder &rewriter,
                               mlir::stablehlo::ConvolutionOp op, Value input,
                               Value filter, Value init,
                               RankedTensorType resultType,
                               const ConvolutionWindow &window) {
  Location loc = op.getLoc();
  int64_t rank = resultType.getRank();
  // Split the grouped dimension instead of unrolling one convolution per
  // group. This keeps the IR size independent of the number of channels.
  int64_t groups = std::max(op.getFeatureGroupCount(), op.getBatchGroupCount());
  input = splitConvolutionGroupDimension(
      rewriter, loc, input, op.getFeatureGroupCount() != 1 ? rank - 1 : 0,
      groups);
  filter =
      splitConvolutionGroupDimension(rewriter, loc, filter, rank - 1, groups);
  Value groupedInit =
      splitConvolutionGroupDimension(rewriter, loc, init, rank - 1, groups);
  // Parallel loops: batch, output spatial dimensions, group, output
  // channel within the group. Reduction loops: kernel spatial, channel.
  int64_t loops = 2 * rank;
  auto dim = [&](int64_t d) { return rewriter.getAffineDimExpr(d); };
  SmallVector<AffineExpr> inputMap, filterMap, outputMap;
  if (op.getBatchGroupCount() != 1) {
    inputMap.push_back(dim(rank - 1));
  }
  inputMap.push_back(dim(0));
  for (int64_t i = 0; i < rank - 2; ++i) {
    inputMap.push_back(dim(i + 1) * window.strides[i] +
                       dim(rank + 1 + i) * window.rhsDilations[i]);
    filterMap.push_back(dim(rank + 1 + i));
  }
  if (op.getFeatureGroupCount() != 1) {
    inputMap.push_back(dim(rank - 1));
  }
  inputMap.push_back(dim(loops - 1));
  filterMap.append({dim(loops - 1), dim(rank - 1), dim(rank)});
  for (int64_t i = 0; i <= rank; ++i) {
    outputMap.push_back(dim(i));
  }
  SmallVector<AffineMap> maps;
  for (auto &exprs : {inputMap, filterMap, outputMap}) {
    maps.push_back(AffineMap::get(loops, 0, exprs, rewriter.getContext()));
  }
  SmallVector<utils::IteratorType> iterators(loops,
                                             utils::IteratorType::reduction);
  std::fill_n(iterators.begin(), rank + 1, utils::IteratorType::parallel);
  Value convolved =
      linalg::GenericOp::create(
          rewriter, loc, TypeRange{groupedInit.getType()},
          ValueRange{input, filter}, ValueRange{groupedInit}, maps, iterators,
          [&](OpBuilder &builder, Location nestedLoc, ValueRange) {
            ImplicitLocOpBuilder nested(nestedLoc, builder);
            linalg::Conv2DOp::regionBuilder(nested, *nested.getInsertionBlock(),
                                            {}, {});
          },
          linalg::getPrunedAttributeList(op))
          .getResult(0);
  SmallVector<ReassociationIndices> collapse;
  for (int64_t i = 0; i < rank - 1; ++i) {
    collapse.push_back({i});
  }
  collapse.push_back({rank - 1, rank});
  return tensor::CollapseShapeOp::create(rewriter, loc, resultType, convolved,
                                         collapse);
}

Value createConvolution(OpBuilder &rewriter, mlir::stablehlo::ConvolutionOp op,
                        Value input, Value filter, Value init,
                        RankedTensorType resultType,
                        const ConvolutionWindow &window) {
  Location loc = op.getLoc();
  int64_t rank = resultType.getRank();
  bool grouped = op.getFeatureGroupCount() != 1 || op.getBatchGroupCount() != 1;
  auto strideAttr = rewriter.getI64TensorAttr(window.strides);
  auto dilationAttr = rewriter.getI64TensorAttr(window.rhsDilations);
  if (grouped) {
    return createGroupedConvolution(rewriter, op, input, filter, init,
                                    resultType, window);
  }
  if (rank == 3) {
    return linalg::Conv1DNwcWcfOp::create(
               rewriter, loc, resultType, ValueRange{input, filter},
               ValueRange{init}, strideAttr, dilationAttr,
               linalg::getPrunedAttributeList(op))
        .getResult(0);
  }
  if (rank == 4) {
    return linalg::Conv2DNhwcHwcfOp::create(
               rewriter, loc, resultType, ValueRange{input, filter},
               ValueRange{init}, strideAttr, dilationAttr,
               linalg::getPrunedAttributeList(op))
        .getResult(0);
  }
  return linalg::Conv3DNdhwcDhwcfOp::create(
             rewriter, loc, resultType, ValueRange{input, filter},
             ValueRange{init}, strideAttr, dilationAttr,
             linalg::getPrunedAttributeList(op))
      .getResult(0);
}

// The upstream named convolution lowering computes dynamic batch/channel sizes
// but requires static spatial results. Compute those sizes from StableHLO C25.
// Grouped convolutions additionally split the group dimensions for Linalg.
struct DynamicConvolutionOpConversion final
    : OpConversionPattern<mlir::stablehlo::ConvolutionOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked result");
    }
    int64_t rank = resultType.getRank();
    if (rank < 3 || rank > 5) {
      return rewriter.notifyMatchFailure(op,
                                         "expected 1D, 2D or 3D convolution");
    }
    auto dims = op.getDimensionNumbers();
    bool grouped =
        op.getFeatureGroupCount() != 1 || op.getBatchGroupCount() != 1;
    if ((!grouped || resultType.hasStaticShape()) &&
        !llvm::any_of(dims.getOutputSpatialDimensions(),
                      [&](int64_t i) { return resultType.isDynamicDim(i); })) {
      return rewriter.notifyMatchFailure(
          op, "upstream handles static spatial results");
    }
    if (auto reversal = op.getWindowReversal();
        reversal && llvm::is_contained(*reversal, true)) {
      return rewriter.notifyMatchFailure(
          op, "expected preprocessed window reversal");
    }
    // The preprocessing pass puts convolutions into this layout.
    if (dims.getInputBatchDimension() != 0 ||
        dims.getOutputBatchDimension() != 0 ||
        dims.getInputFeatureDimension() != rank - 1 ||
        dims.getOutputFeatureDimension() != rank - 1 ||
        dims.getKernelInputFeatureDimension() != rank - 2 ||
        dims.getKernelOutputFeatureDimension() != rank - 1) {
      return rewriter.notifyMatchFailure(
          op, "expected canonical dimension numbers");
    }
    for (int64_t i = 0; i < rank - 2; ++i) {
      if (dims.getInputSpatialDimensions()[i] != i + 1 ||
          dims.getOutputSpatialDimensions()[i] != i + 1 ||
          dims.getKernelSpatialDimensions()[i] != i) {
        return rewriter.notifyMatchFailure(
            op, "expected canonical spatial dimensions");
      }
    }

    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    Value filter = adaptor.getRhs();
    auto window = getConvolutionWindow(op);
    auto shape = computeConvolutionResultShape(rewriter, op, input, filter,
                                               resultType, window);
    Value empty =
        tensor::EmptyOp::create(rewriter, loc, resultType, shape.dynamicDims);
    Value init = fillTensorWithZeros(rewriter, loc, empty);

    // Avoid materializing a negatively-sized crop (or running a convolution)
    // when no window fits. The shape calculation above still follows C25.
    auto ifOp =
        scf::IfOp::create(rewriter, loc, TypeRange{resultType}, shape.nonempty,
                          /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      input = padConvolutionInput(rewriter, loc, input, window);
      Value result = createConvolution(rewriter, op, input, filter, init,
                                       resultType, window);
      scf::YieldOp::create(rewriter, loc, result);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      scf::YieldOp::create(rewriter, loc, init);
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorTieShapeOp>(
        op, resultType, ifOp.getResult(0), shape.dynamicDims);
    return success();
  }
};

// Adapted from StablehloLegalizeToLinalg.cpp's GatherConversion.
// The operand rank and result shape suffice without constant slice sizes.
struct DynamicGatherOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicGatherOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicGatherOp gatherOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = gatherOp.getLoc();
    Value startIndices = adaptor.getStartIndices();
    Value operand = adaptor.getOperand();
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(gatherOp.getType());
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto startIndicesType = dyn_cast<RankedTensorType>(startIndices.getType());
    if (!resultType || !operandType || !startIndicesType) {
      return rewriter.notifyMatchFailure(gatherOp, "unranked operands");
    }
    int64_t resultRank = resultType.getRank();
    int64_t operandRank = operandType.getRank();

    auto dims = gatherOp.getDimensionNumbers();
    int64_t indexVectorDim = dims.getIndexVectorDim();
    ArrayRef<int64_t> offsetDims = dims.getOffsetDims();
    ArrayRef<int64_t> collapsedSliceDims = dims.getCollapsedSliceDims();
    ArrayRef<int64_t> operandBatchingDims = dims.getOperandBatchingDims();
    ArrayRef<int64_t> startIndicesBatchingDims =
        dims.getStartIndicesBatchingDims();
    ArrayRef<int64_t> startIndexMap = dims.getStartIndexMap();

    SmallVector<Value> constants;
    for (int64_t i = 0, e = std::max({resultRank, operandRank, int64_t{2}});
         i < e; ++i) {
      constants.push_back(arith::ConstantIndexOp::create(rewriter, loc, i));
    }

    Value emptyOp = getEmptyTensorFor(rewriter, loc, resultType, gatherOp,
                                      adaptor.getOperands());
    if (!emptyOp) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "could not reify result shape");
    }

    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{}, /*outputs=*/emptyOp,
        SmallVector<AffineMap>{rewriter.getMultiDimIdentityMap(resultRank)},
        getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &b, Location nestedLoc, ValueRange) {
          SmallVector<Value> linalgIndices, gatherIndex;
          for (int64_t dim = 0; dim < resultRank; ++dim) {
            Value index = linalg::IndexOp::create(b, nestedLoc, dim);
            linalgIndices.push_back(index);
            if (!llvm::is_contained(offsetDims, dim)) {
              gatherIndex.push_back(index);
            }
          }

          SmallVector<Value> indexFromStartIndices;
          for (size_t i = 0, e = startIndexMap.size(); i != e; ++i) {
            SmallVector<Value> gCombine(gatherIndex);
            if (indexVectorDim != startIndicesType.getRank()) {
              gCombine.insert(gCombine.begin() + indexVectorDim, constants[i]);
            }
            indexFromStartIndices.push_back(extractIndexFromTensor(
                b, nestedLoc, startIndices,
                gatherOp.getStartIndices().getType(), gCombine));
          }

          SmallVector<Value> remappedIndexFromIndices(operandRank,
                                                      constants[0]);
          for (auto [idx, value] : llvm::enumerate(startIndexMap)) {
            remappedIndexFromIndices[value] = indexFromStartIndices[idx];
          }

          SmallVector<Value> indexFromBatching(operandRank, constants[0]);
          for (auto [operandDim, indicesDim] :
               llvm::zip_equal(operandBatchingDims, startIndicesBatchingDims)) {
            indexFromBatching[operandDim] =
                gatherIndex[indicesDim - (indicesDim < indexVectorDim ? 0 : 1)];
          }

          auto isCollapsedOrBatching = [&](int64_t dim) {
            return llvm::is_contained(collapsedSliceDims, dim) ||
                   llvm::is_contained(operandBatchingDims, dim);
          };
          SmallVector<unsigned> remappedOffsetDims;
          for (int64_t i = 0; i < operandRank; ++i) {
            if (!isCollapsedOrBatching(i)) {
              remappedOffsetDims.push_back(static_cast<unsigned>(i));
            }
          }

          // Clamp start indices to [0, operand_dim - slice_size]; the slice
          // size is the matching result dim, or 1 for a collapsed dim.
          for (int i = 0, operandIndexDim = 0; i < operandRank; ++i) {
            Value outputDimSize = constants[1];
            if (!isCollapsedOrBatching(i)) {
              outputDimSize = b.createOrFold<tensor::DimOp>(
                  nestedLoc, emptyOp, offsetDims[operandIndexDim++]);
            }
            if (remappedIndexFromIndices[i] == constants[0]) {
              continue;
            }
            Value operandDimSize =
                b.createOrFold<tensor::DimOp>(nestedLoc, operand, i);
            Value largestValidIndex = b.createOrFold<arith::SubIOp>(
                nestedLoc, operandDimSize, outputDimSize);
            remappedIndexFromIndices[i] = arith::MinSIOp::create(
                b, nestedLoc,
                arith::MaxSIOp::create(b, nestedLoc, constants[0],
                                       remappedIndexFromIndices[i]),
                largestValidIndex);
          }

          SmallVector<Value> indexFromOffset(operandRank, constants[0]);
          for (auto [remappedOffsetDim, offsetDim] :
               llvm::zip_equal(remappedOffsetDims, offsetDims)) {
            indexFromOffset[remappedOffsetDim] = linalgIndices[offsetDim];
          }

          SmallVector<Value> combinedIndex;
          for (int64_t i = 0; i < operandRank; ++i) {
            combinedIndex.push_back(b.createOrFold<arith::AddIOp>(
                nestedLoc, b.getIndexType(),
                b.createOrFold<arith::AddIOp>(nestedLoc, b.getIndexType(),
                                              remappedIndexFromIndices[i],
                                              indexFromBatching[i]),
                indexFromOffset[i]));
          }
          Value element =
              tensor::ExtractOp::create(b, nestedLoc, operand, combinedIndex);
          linalg::YieldOp::create(b, nestedLoc, element);
        },
        linalg::getPrunedAttributeList(gatherOp));

    rewriter.replaceOp(gatherOp, linalgOp.getResults());
    return success();
  }
};

// An affine indexing map cannot branch on a runtime size, so when nothing
// says whether an operand dim expands, gather each element.
struct DynamicBroadcastInDimGatherConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = adaptor.getOperand();
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!operandType || !resultType) {
      return rewriter.notifyMatchFailure(op, "unranked");
    }

    // Step in only when some operand dim is dynamic and unannotated; upstream
    // handles the rest.
    ArrayRef<int64_t> bcastDims = op.getBroadcastDimensions();
    SmallVector<std::optional<bool>> expanding(operandType.getRank());
    for (auto [idx, dim] : llvm::enumerate(operandType.getShape())) {
      if (!ShapedType::isDynamic(dim)) {
        expanding[idx] = (dim == 1);
      }
    }
    if (auto known = op.getKnownExpandingDimensions()) {
      for (int64_t i : *known) {
        expanding[i] = true;
      }
    }
    if (auto known = op.getKnownNonexpandingDimensions()) {
      for (int64_t i : *known) {
        expanding[i] = false;
      }
    }
    if (llvm::all_of(expanding, [](auto v) { return v.has_value(); })) {
      return rewriter.notifyMatchFailure(op, "expansion is decidable");
    }

    Value empty =
        getEmptyTensorFor(rewriter, loc, resultType, op, adaptor.getOperands());
    if (!empty) {
      return rewriter.notifyMatchFailure(op, "could not reify result shape");
    }
    int64_t resultRank = resultType.getRank();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    SmallVector<Value> isExpanding(operandType.getRank());
    for (int64_t i = 0, e = operandType.getRank(); i < e; ++i) {
      if (expanding[i].has_value()) {
        continue;
      }
      Value dim = rewriter.createOrFold<tensor::DimOp>(loc, operand, i);
      isExpanding[i] = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, dim, one);
    }

    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{}, /*outputs=*/empty,
        SmallVector<AffineMap>{rewriter.getMultiDimIdentityMap(resultRank)},
        getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &b, Location nestedLoc, ValueRange) {
          SmallVector<Value> index;
          for (auto [operandDim, resultDim] : llvm::enumerate(bcastDims)) {
            if (expanding[operandDim] == true) {
              index.push_back(zero);
              continue;
            }
            Value resultIndex =
                linalg::IndexOp::create(b, nestedLoc, resultDim);
            index.push_back(expanding[operandDim].has_value()
                                ? resultIndex
                                : arith::SelectOp::create(
                                      b, nestedLoc, isExpanding[operandDim],
                                      zero, resultIndex));
          }
          Value element =
              tensor::ExtractOp::create(b, nestedLoc, operand, index);
          linalg::YieldOp::create(b, nestedLoc, element);
        },
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

} // namespace

void populateDynamicShapeConversionPatterns(MLIRContext *context,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet *patterns) {
  // Prefer these lowerings for runtime shapes over upstream fallbacks.
  patterns->add<DynamicReshapeOpConversion, DynamicPadOpConversion,
                DynamicConvolutionOpConversion, DynamicGatherOpConversion,
                DynamicBroadcastInDimGatherConversion>(typeConverter, context,
                                                       PatternBenefit{1000});
}

} // namespace mlir::iree_compiler::stablehlo
