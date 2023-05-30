// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Inline/Conversion/HALToHALInline/Patterns.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineDialect.h"
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct BufferSubspanOpPattern
    : public OpConversionPattern<IREE::HAL::BufferSubspanOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferSubspanOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto bufferType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferSubspanOp>(
        op, bufferType, adaptor.getSourceBuffer(), adaptor.getSourceOffset(),
        adaptor.getLength());
    return success();
  }
};

struct BufferLengthOpPattern
    : public OpConversionPattern<IREE::HAL::BufferLengthOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferLengthOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sizeType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferLengthOp>(
        op, sizeType, adaptor.getBuffer());
    return success();
  }
};

struct BufferLoadOpPattern
    : public OpConversionPattern<IREE::HAL::BufferLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferLoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value storageBuffer =
        rewriter.createOrFold<IREE::HAL::Inline::BufferStorageOp>(
            op.getLoc(), adaptor.getSourceBuffer());
    Value storageSize = rewriter.create<IREE::HAL::Inline::BufferLengthOp>(
        op.getLoc(), adaptor.getSourceBuffer());
    auto loadType = getTypeConverter()->convertType(op.getResult().getType());
    auto elementSize =
        rewriter.createOrFold<IREE::Util::SizeOfOp>(op.getLoc(), loadType);
    rewriter.replaceOpWithNewOp<IREE::Util::BufferLoadOp>(
        op, loadType, storageBuffer, storageSize, adaptor.getSourceOffset(),
        elementSize);
    return success();
  }
};

struct BufferStoreOpPattern
    : public OpConversionPattern<IREE::HAL::BufferStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferStoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value storageBuffer =
        rewriter.createOrFold<IREE::HAL::Inline::BufferStorageOp>(
            op.getLoc(), adaptor.getTargetBuffer());
    Value storageSize = rewriter.create<IREE::HAL::Inline::BufferLengthOp>(
        op.getLoc(), adaptor.getTargetBuffer());
    auto elementSize = rewriter.createOrFold<IREE::Util::SizeOfOp>(
        op.getLoc(), adaptor.getValue().getType());
    rewriter.replaceOpWithNewOp<IREE::Util::BufferStoreOp>(
        op, adaptor.getValue(), storageBuffer, storageSize,
        adaptor.getTargetOffset(), elementSize);
    return success();
  }
};

struct BufferViewCreateOpPattern
    : public OpConversionPattern<IREE::HAL::BufferViewCreateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferViewCreateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferViewCreateOp>(
        op, adaptor.getSourceBuffer(), adaptor.getSourceOffset(),
        adaptor.getSourceLength(), adaptor.getElementType(),
        adaptor.getEncodingType(), adaptor.getShape());
    return success();
  }
};

struct BufferViewBufferOpPattern
    : public OpConversionPattern<IREE::HAL::BufferViewBufferOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferViewBufferOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferViewBufferOp>(
        op, rewriter.getType<IREE::HAL::BufferType>(), adaptor.getBufferView());
    return success();
  }
};

struct BufferViewAssertOpPattern
    : public OpConversionPattern<IREE::HAL::BufferViewAssertOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferViewAssertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferViewAssertOp>(
        op, adaptor.getBufferView(), adaptor.getMessage(),
        adaptor.getElementType(), adaptor.getEncodingType(),
        adaptor.getShape());
    return success();
  }
};

struct BufferViewElementTypeOpPattern
    : public OpConversionPattern<IREE::HAL::BufferViewElementTypeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferViewElementTypeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferViewElementTypeOp>(
        op, op.getResult().getType(), adaptor.getBufferView());
    return success();
  }
};

struct BufferViewEncodingTypeOpPattern
    : public OpConversionPattern<IREE::HAL::BufferViewEncodingTypeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferViewEncodingTypeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferViewEncodingTypeOp>(
        op, op.getResult().getType(), adaptor.getBufferView());
    return success();
  }
};

struct BufferViewRankOpPattern
    : public OpConversionPattern<IREE::HAL::BufferViewRankOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferViewRankOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferViewRankOp>(
        op, op.getResult().getType(), adaptor.getBufferView());
    return success();
  }
};

struct BufferViewDimOpPattern
    : public OpConversionPattern<IREE::HAL::BufferViewDimOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferViewDimOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferViewDimOp>(
        op, op.getResult().getType(), adaptor.getBufferView(),
        adaptor.getIndexAttr());
    return success();
  }
};

struct BufferViewTraceOpPattern
    : public OpConversionPattern<IREE::HAL::BufferViewTraceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::BufferViewTraceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferViewTraceOp>(
        op, adaptor.getKeyAttr(), adaptor.getOperands());
    return success();
  }
};

}  // namespace

void populateHALToHALInlinePatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  typeConverter.addConversion([](IREE::HAL::BufferType type) { return type; });
  typeConverter.addConversion(
      [](IREE::HAL::BufferViewType type) { return type; });

  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, IREE::Util::BufferType type, ValueRange inputs,
         Location loc) -> Value {
        assert(inputs.size() == 1);
        if (llvm::isa<IREE::HAL::BufferType>(inputs[0].getType())) {
          return builder.createOrFold<IREE::HAL::Inline::BufferStorageOp>(
              loc, inputs[0]);
        } else {
          emitError(loc) << "unsupported HAL inline target materialization: "
                         << inputs[0].getType();
          return nullptr;
        }
      });

  patterns.insert<BufferSubspanOpPattern>(typeConverter, context);
  patterns.insert<BufferLengthOpPattern>(typeConverter, context);
  patterns.insert<BufferLoadOpPattern>(typeConverter, context);
  patterns.insert<BufferStoreOpPattern>(typeConverter, context);

  patterns.insert<BufferViewCreateOpPattern>(typeConverter, context);
  patterns.insert<BufferViewAssertOpPattern>(typeConverter, context);
  patterns.insert<BufferViewBufferOpPattern>(typeConverter, context);
  patterns.insert<BufferViewElementTypeOpPattern>(typeConverter, context);
  patterns.insert<BufferViewEncodingTypeOpPattern>(typeConverter, context);
  patterns.insert<BufferViewRankOpPattern>(typeConverter, context);
  patterns.insert<BufferViewDimOpPattern>(typeConverter, context);
  patterns.insert<BufferViewTraceOpPattern>(typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
