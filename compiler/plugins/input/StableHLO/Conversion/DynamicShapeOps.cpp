// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowerings for the StableHLO ops whose shapes are known only at runtime.

#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/LegalizeToLinalgUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

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
    int64_t rank = operandType.getRank();

    Value paddingValue = rewriter.createOrFold<tensor::ExtractOp>(
        loc, adaptor.getPaddingValue());

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value> dynamicDims, resultDynamicDims;
    Value nonempty = rewriter.createOrFold<arith::ConstantIntOp>(loc, 1, 1);
    SmallVector<OpFoldResult> insertSizes =
        tensor::getMixedSizes(rewriter, loc, adaptor.getOperand());
    SmallVector<OpFoldResult> insertOffsets, insertStrides;
    SmallVector<OpFoldResult> extractOffsets, extractSizes;
    for (int64_t i = 0; i < rank; ++i) {
      Value dim =
          getValueOrCreateConstantIndexOp(rewriter, loc, insertSizes[i]);
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
      dynamicDims.push_back(filledDim);

      if (resultType.isDynamicDim(i)) {
        Value dimAndLow =
            arith::AddIOp::create(rewriter, loc, dimAndInterior, low);
        Value resultDim = arith::AddIOp::create(rewriter, loc, dimAndLow, high);
        extractSizes.push_back(resultDim);
        resultDynamicDims.push_back(resultDim);
      } else {
        extractSizes.push_back(rewriter.getIndexAttr(resultType.getDimSize(i)));
      }

      insertOffsets.push_back(lowPos);
      insertStrides.push_back(
          arith::AddIOp::create(rewriter, loc, interior, one).getResult());
      extractOffsets.push_back(lowNeg);
      Value size =
          getValueOrCreateConstantIndexOp(rewriter, loc, extractSizes.back());
      Value positive = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, size, zero);
      nonempty = rewriter.createOrFold<arith::AndIOp>(loc, nonempty, positive);
    }
    // Empty crops need no scratch buffer. In particular, runtime convolution
    // padding can remove an arbitrarily large dilated input entirely.
    auto ifOp =
        scf::IfOp::create(rewriter, loc, TypeRange{resultType}, nonempty,
                          /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      SmallVector<int64_t> scratchShape(rank, ShapedType::kDynamic);
      Value empty =
          tensor::EmptyOp::create(rewriter, loc, scratchShape,
                                  resultType.getElementType(), dynamicDims);
      Value filled =
          linalg::FillOp::create(rewriter, loc, paddingValue, empty).result();
      Value inserted = tensor::InsertSliceOp::create(
                           rewriter, loc, adaptor.getOperand(), filled,
                           insertOffsets, insertSizes, insertStrides)
                           .getResult();
      // Negative edge padding crops, which the insert cannot express.
      SmallVector<OpFoldResult> extractStrides(rank, rewriter.getIndexAttr(1));
      Value result = tensor::ExtractSliceOp::create(
          rewriter, loc, resultType, inserted, extractOffsets, extractSizes,
          extractStrides);
      scf::YieldOp::create(rewriter, loc, result);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      Value resultEmpty =
          tensor::EmptyOp::create(rewriter, loc, resultType, resultDynamicDims);
      scf::YieldOp::create(rewriter, loc, resultEmpty);
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorTieShapeOp>(
        op, resultType, ifOp.getResult(0), resultDynamicDims);
    return success();
  }
};

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
    SmallVector<int64_t> strides(rank - 2, 1), lhsDilations(rank - 2, 1),
        rhsDilations(rank - 2, 1), padding(2 * (rank - 2), 0);
    if (op.getWindowStrides()) {
      strides = llvm::to_vector(*op.getWindowStrides());
    }
    if (op.getLhsDilation()) {
      lhsDilations = llvm::to_vector(*op.getLhsDilation());
    }
    if (op.getRhsDilation()) {
      rhsDilations = llvm::to_vector(*op.getRhsDilation());
    }
    if (op.getPadding()) {
      padding = llvm::to_vector(op.getPadding()->getValues<int64_t>());
    }
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
      span =
          rewriter.createOrFold<arith::MulIOp>(loc, span, constant(dilation));
      Value dilated = rewriter.createOrFold<arith::AddIOp>(loc, span, one);
      return rewriter.createOrFold<arith::SelectOp>(loc, empty, zero, dilated);
    };

    SmallVector<Value> dynamicDims;
    Value nonempty = rewriter.createOrFold<arith::ConstantIntOp>(loc, 1, 1);
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
        Value padded = dilate(inputSize, lhsDilations[s]);
        padded = rewriter.createOrFold<arith::AddIOp>(loc, padded,
                                                      constant(padding[2 * s]));
        padded = rewriter.createOrFold<arith::AddIOp>(
            loc, padded, constant(padding[2 * s + 1]));
        Value window = dilate(kernelSize, rhsDilations[s]);
        Value span = rewriter.createOrFold<arith::SubIOp>(loc, padded, window);
        // For a fitting window span is nonnegative, so truncation is floor.
        // The select discards this count when the window does not fit.
        Value count = rewriter.createOrFold<arith::DivSIOp>(
            loc, span, constant(strides[s]));
        count = rewriter.createOrFold<arith::AddIOp>(loc, count, one);
        Value fits = rewriter.createOrFold<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, span, zero);
        Value hasInput = rewriter.createOrFold<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, padded, zero);
        fits = rewriter.createOrFold<arith::AndIOp>(loc, fits, hasInput);
        size = rewriter.createOrFold<arith::SelectOp>(loc, fits, count, zero);
      }
      if (resultType.isDynamicDim(d)) {
        dynamicDims.push_back(size);
      }
      Value positive = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, size, zero);
      nonempty = rewriter.createOrFold<arith::AndIOp>(loc, nonempty, positive);
    }
    Value empty =
        tensor::EmptyOp::create(rewriter, loc, resultType, dynamicDims);
    Value init = mlir::stablehlo::fillTensorWithZeros(rewriter, loc, empty);

    // Avoid materializing a negatively-sized crop (or running a convolution)
    // when no window fits. The shape calculation above still follows C25.
    auto ifOp =
        scf::IfOp::create(rewriter, loc, TypeRange{resultType}, nonempty,
                          /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      if (llvm::any_of(padding, [](int64_t v) { return v != 0; }) ||
          llvm::any_of(lhsDilations, [](int64_t v) { return v != 1; })) {
        SmallVector<int64_t> lows(rank, 0), highs(rank, 0), interiors(rank, 0);
        for (int64_t i = 0; i < rank - 2; ++i) {
          lows[i + 1] = padding[2 * i];
          highs[i + 1] = padding[2 * i + 1];
          interiors[i + 1] = lhsDilations[i] - 1;
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
        Value padValue =
            mlir::stablehlo::fillTensorWithZeros(rewriter, loc, scalarEmpty);
        input = mlir::stablehlo::DynamicPadOp::create(
            rewriter, loc,
            RankedTensorType::get(paddedShape, inputType.getElementType()),
            input, padValue, shapeConstant(lows), shapeConstant(highs),
            shapeConstant(interiors));
      }
      auto strideAttr = rewriter.getI64TensorAttr(strides);
      auto dilationAttr = rewriter.getI64TensorAttr(rhsDilations);
      Value result;
      if (grouped) {
        // Split the grouped dimension instead of unrolling one convolution per
        // group. This keeps the IR size independent of the number of channels.
        int64_t groups =
            std::max(op.getFeatureGroupCount(), op.getBatchGroupCount());
        auto split = [&](Value value, int64_t dimension) -> Value {
          auto type = cast<RankedTensorType>(value.getType());
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
              rewriter, loc,
              RankedTensorType::get(shape, type.getElementType()), value,
              reassociation);
        };
        input = split(input, op.getFeatureGroupCount() != 1 ? rank - 1 : 0);
        filter = split(filter, rank - 1);
        Value groupedInit = split(init, rank - 1);
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
          inputMap.push_back(dim(i + 1) * strides[i] +
                             dim(rank + 1 + i) * rhsDilations[i]);
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
          maps.push_back(
              AffineMap::get(loops, 0, exprs, rewriter.getContext()));
        }
        SmallVector<utils::IteratorType> iterators(
            loops, utils::IteratorType::reduction);
        std::fill_n(iterators.begin(), rank + 1, utils::IteratorType::parallel);
        Value convolved =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{groupedInit.getType()},
                ValueRange{input, filter}, ValueRange{groupedInit}, maps,
                iterators,
                [&](OpBuilder &builder, Location nestedLoc, ValueRange) {
                  ImplicitLocOpBuilder nested(nestedLoc, builder);
                  linalg::Conv2DOp::regionBuilder(
                      nested, *nested.getInsertionBlock(), {}, {});
                },
                linalg::getPrunedAttributeList(op))
                .getResult(0);
        SmallVector<ReassociationIndices> collapse;
        for (int64_t i = 0; i < rank - 1; ++i) {
          collapse.push_back({i});
        }
        collapse.push_back({rank - 1, rank});
        result = tensor::CollapseShapeOp::create(rewriter, loc, resultType,
                                                 convolved, collapse);
      } else if (rank == 3) {
        result = linalg::Conv1DNwcWcfOp::create(
                     rewriter, loc, resultType, ValueRange{input, filter},
                     ValueRange{init}, strideAttr, dilationAttr,
                     linalg::getPrunedAttributeList(op))
                     .getResult(0);
      } else if (rank == 4) {
        result = linalg::Conv2DNhwcHwcfOp::create(
                     rewriter, loc, resultType, ValueRange{input, filter},
                     ValueRange{init}, strideAttr, dilationAttr,
                     linalg::getPrunedAttributeList(op))
                     .getResult(0);
      } else {
        result = linalg::Conv3DNdhwcDhwcfOp::create(
                     rewriter, loc, resultType, ValueRange{input, filter},
                     ValueRange{init}, strideAttr, dilationAttr,
                     linalg::getPrunedAttributeList(op))
                     .getResult(0);
      }
      scf::YieldOp::create(rewriter, loc, result);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      scf::YieldOp::create(rewriter, loc, init);
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorTieShapeOp>(
        op, resultType, ifOp.getResult(0), dynamicDims);
    return success();
  }
};

} // namespace

void populateDynamicShapeConversionPatterns(MLIRContext *context,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet *patterns) {
  // Prefer these lowerings for runtime shapes over upstream fallbacks.
  patterns->add<DynamicReshapeOpConversion, DynamicPadOpConversion,
                DynamicConvolutionOpConversion>(typeConverter, context,
                                                PatternBenefit{1000});
}

} // namespace mlir::iree_compiler::stablehlo
