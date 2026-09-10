// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowerings for the StableHLO ops whose shapes are known only at runtime.

#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

// Reads element `i` of a 1-D shape operand as an index.
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
    SmallVector<Value> dynamicDims;
    SmallVector<OpFoldResult> insertOffsets, insertSizes, insertStrides;
    SmallVector<OpFoldResult> extractOffsets, extractSizes;
    for (int64_t i = 0; i < rank; ++i) {
      Value dim = tensor::DimOp::create(rewriter, loc, adaptor.getOperand(), i);
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
      } else {
        extractSizes.push_back(rewriter.getIndexAttr(resultType.getDimSize(i)));
      }

      insertOffsets.push_back(lowPos);
      insertSizes.push_back(
          tensor::getMixedSize(rewriter, loc, adaptor.getOperand(), i));
      insertStrides.push_back(
          arith::AddIOp::create(rewriter, loc, interior, one).getResult());
      extractOffsets.push_back(lowNeg);
    }

    SmallVector<int64_t> scratchShape(rank, ShapedType::kDynamic);
    Value empty = tensor::EmptyOp::create(
        rewriter, loc, scratchShape, resultType.getElementType(), dynamicDims);
    Value filled =
        linalg::FillOp::create(rewriter, loc, paddingValue, empty).result();
    Value inserted = tensor::InsertSliceOp::create(
                         rewriter, loc, adaptor.getOperand(), filled,
                         insertOffsets, insertSizes, insertStrides)
                         .getResult();

    // Negative edge padding crops, which the insert cannot express.
    SmallVector<OpFoldResult> extractStrides(rank, rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultType, inserted, extractOffsets, extractSizes, extractStrides);
    return success();
  }
};

} // namespace

void populateDynamicShapeConversionPatterns(MLIRContext *context,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet *patterns) {
  // Higher benefit than the upstream patterns, which decline these forms.
  patterns->add<DynamicReshapeOpConversion, DynamicPadOpConversion>(
      typeConverter, context, PatternBenefit{1000});
}

} // namespace mlir::iree_compiler::stablehlo
