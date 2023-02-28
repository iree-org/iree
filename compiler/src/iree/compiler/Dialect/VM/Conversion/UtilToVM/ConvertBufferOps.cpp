// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/Conversion/UtilToVM/ConvertUtilToVM.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

static Value castToI64(Value value, OpBuilder &builder) {
  if (value.getType().isInteger(64)) return value;
  return builder.createOrFold<IREE::VM::ExtI32I64UOp>(
      value.getLoc(), builder.getI64Type(), value);
}

static Value castToIndex(Value value, OpBuilder &builder) {
  if (value.getType().isIndex()) return value;
  return builder.createOrFold<arith::IndexCastOp>(
      value.getLoc(), builder.getIndexType(), value);
}

struct BufferConstantOpConversion
    : public OpConversionPattern<IREE::Util::BufferConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::BufferConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto alignmentAttr = op.getAlignmentAttr();
    if (alignmentAttr) {
      alignmentAttr = rewriter.getI64IntegerAttr(alignmentAttr.getInt());
    }
    rewriter.replaceOpWithNewOp<IREE::VM::RodataInlineOp>(
        op,
        IREE::VM::RefType::get(
            IREE::VM::BufferType::get(rewriter.getContext())),
        op.getNameAttr(), op.getValue(), alignmentAttr, op.getMimeTypeAttr());
    return success();
  }
};

struct BufferAllocOpConversion
    : public OpConversionPattern<IREE::Util::BufferAllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::BufferAllocOp allocOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(#9165): support alignment for vm.buffer.alloc. So far we ignore the
    // alignment attribute when lowering the op to VM dialect.
    (void)adaptor.getAlignment();
    auto resultType =
        getTypeConverter()->convertType(allocOp.getResult().getType());
    rewriter.replaceOpWithNewOp<IREE::VM::BufferAllocOp>(
        allocOp, resultType, castToI64(adaptor.getStorageSize(), rewriter));
    return success();
  }
};

struct BufferDeallocOpConversion
    : public OpConversionPattern<IREE::Util::BufferDeallocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::BufferDeallocOp deallocOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // No-op today. We could make this force a dealloc of the underlying storage
    // or have a vm.hint.reset or something to force a drop of the reference.
    rewriter.eraseOp(deallocOp);
    return success();
  }
};

// Expands util.buffer.slice -> vm.buffer.alloc + vm.buffer.copy.
// We could have a vm.buffer.slice op if we wanted; today there's nothing we'd
// do in the runtime besides this.
struct BufferSliceOpConversion
    : public OpConversionPattern<IREE::Util::BufferSliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::BufferSliceOp sliceOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(#9165): support alignment for vm.buffer.alloc. So far we ignore the
    // alignment attribute when lowering the op to VM dialect.
    (void)adaptor.getAlignment();
    auto resultType =
        getTypeConverter()->convertType(sliceOp.getResult().getType());
    auto sliceLength = castToI64(adaptor.getResultSize(), rewriter);
    Value newBuffer = rewriter.create<IREE::VM::BufferAllocOp>(
        sliceOp.getLoc(), resultType, sliceLength);
    Value zero = rewriter.create<IREE::VM::ConstI64ZeroOp>(sliceOp.getLoc());
    rewriter.create<IREE::VM::BufferCopyOp>(
        sliceOp.getLoc(), adaptor.getSource(),
        castToI64(adaptor.getSourceOffset(), rewriter), newBuffer, zero,
        sliceLength);
    rewriter.replaceOp(sliceOp, newBuffer);
    return success();
  }
};

struct BufferSizeOpConversion
    : public OpConversionPattern<IREE::Util::BufferSizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::BufferSizeOp sizeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value size = rewriter.create<IREE::VM::BufferLengthOp>(
        sizeOp.getLoc(), rewriter.getI64Type(), adaptor.getOperand());
    rewriter.replaceOp(sizeOp, castToIndex(size, rewriter));
    return success();
  }
};

struct BufferCopyOpConversion
    : public OpConversionPattern<IREE::Util::BufferCopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::BufferCopyOp copyOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::BufferCopyOp>(
        copyOp, adaptor.getSource(),
        castToI64(adaptor.getSourceOffset(), rewriter), adaptor.getTarget(),
        castToI64(adaptor.getTargetOffset(), rewriter),
        castToI64(adaptor.getLength(), rewriter));
    return success();
  }
};

struct BufferCompareOpConversion
    : public OpConversionPattern<IREE::Util::BufferCompareOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::BufferCompareOp compareOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        getTypeConverter()->convertType(compareOp.getResult().getType());
    rewriter.replaceOpWithNewOp<IREE::VM::BufferCompareOp>(
        compareOp, resultType, adaptor.getLhs(),
        castToI64(adaptor.getLhsOffset(), rewriter), adaptor.getRhs(),
        castToI64(adaptor.getRhsOffset(), rewriter),
        castToI64(adaptor.getLength(), rewriter));
    return success();
  }
};

static Value unscaleOffset(Location loc, Value offset, int64_t scale,
                           OpBuilder &builder) {
  if (scale == 1) return offset;
  return builder.createOrFold<IREE::VM::DivI64SOp>(
      loc, offset.getType(), offset,
      builder.create<IREE::VM::ConstI64Op>(loc, scale));
}

struct BufferFillOpConversion
    : public OpConversionPattern<IREE::Util::BufferFillOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::BufferFillOp fillOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto oldType = fillOp.getPattern().getType();
    auto newType = adaptor.getPattern().getType();
    if (oldType.isa<IndexType>()) {
      // Use the actual converted type for IndexType.
      oldType = newType;
    }
    auto byteOffset = castToI64(adaptor.getTargetOffset(), rewriter);
    auto byteLength = castToI64(adaptor.getLength(), rewriter);
    int64_t elementSize = IREE::Util::getRoundedElementByteWidth(oldType);
    auto elementOffset =
        unscaleOffset(fillOp.getLoc(), byteOffset, elementSize, rewriter);
    auto elementLength =
        unscaleOffset(fillOp.getLoc(), byteLength, elementSize, rewriter);
    auto pattern = adaptor.getPattern();
    if (auto integerType = oldType.dyn_cast<IntegerType>()) {
      if (integerType.isInteger(1) || integerType.isInteger(8)) {
        rewriter.replaceOpWithNewOp<IREE::VM::BufferFillI8Op>(
            fillOp, adaptor.getTarget(), byteOffset, byteLength, pattern);
      } else if (integerType.isInteger(16)) {
        rewriter.replaceOpWithNewOp<IREE::VM::BufferFillI16Op>(
            fillOp, adaptor.getTarget(), elementOffset, elementLength, pattern);
      } else if (integerType.isInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::BufferFillI32Op>(
            fillOp, adaptor.getTarget(), elementOffset, elementLength, pattern);
      } else if (integerType.isInteger(64)) {
        rewriter.replaceOpWithNewOp<IREE::VM::BufferFillI64Op>(
            fillOp, adaptor.getTarget(), elementOffset, elementLength, pattern);
      } else {
        return rewriter.notifyMatchFailure(
            fillOp, "invalid integer buffer element type");
      }
    } else if (oldType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferFillF32Op>(
          fillOp, adaptor.getTarget(), elementOffset, elementLength, pattern);
    } else if (oldType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferFillF64Op>(
          fillOp, adaptor.getTarget(), elementOffset, elementLength, pattern);
    } else {
      return rewriter.notifyMatchFailure(fillOp,
                                         "invalid float buffer element type");
    }
    return success();
  }
};

struct BufferLoadOpConversion
    : public OpConversionPattern<IREE::Util::BufferLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::BufferLoadOp loadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto oldType = loadOp.getResult().getType();
    auto newType = getTypeConverter()->convertType(oldType);
    if (oldType.isa<IndexType>()) {
      oldType = newType;
    }
    auto byteOffset = castToI64(adaptor.getSourceOffset(), rewriter);
    int64_t elementSize = IREE::Util::getRoundedElementByteWidth(oldType);
    auto elementOffset =
        unscaleOffset(loadOp.getLoc(), byteOffset, elementSize, rewriter);
    if (auto integerType = oldType.dyn_cast<IntegerType>()) {
      if (integerType.isInteger(1) || integerType.isInteger(8)) {
        if (integerType.isSigned() || integerType.isSignless()) {
          rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI8SOp>(
              loadOp, newType, adaptor.getSource(), byteOffset);
        } else {
          rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI8UOp>(
              loadOp, newType, adaptor.getSource(), byteOffset);
        }
      } else if (integerType.isInteger(16)) {
        if (integerType.isSigned() || integerType.isSignless()) {
          rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI16SOp>(
              loadOp, newType, adaptor.getSource(), elementOffset);
        } else {
          rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI16UOp>(
              loadOp, newType, adaptor.getSource(), elementOffset);
        }
      } else if (integerType.isInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI32Op>(
            loadOp, newType, adaptor.getSource(), elementOffset);
      } else if (integerType.isInteger(64)) {
        rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI64Op>(
            loadOp, newType, adaptor.getSource(), elementOffset);
      } else {
        return rewriter.notifyMatchFailure(
            loadOp, "invalid integer buffer element type");
      }
    } else if (oldType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadF32Op>(
          loadOp, newType, adaptor.getSource(), elementOffset);
    } else if (oldType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadF64Op>(
          loadOp, newType, adaptor.getSource(), elementOffset);
    } else {
      return rewriter.notifyMatchFailure(loadOp,
                                         "invalid float buffer element type");
    }
    return success();
  }
};

struct BufferStoreOpConversion
    : public OpConversionPattern<IREE::Util::BufferStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::BufferStoreOp storeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto oldType = storeOp.getSource().getType();
    auto newType = adaptor.getSource().getType();
    if (oldType.isa<IndexType>()) {
      oldType = newType;
    }
    auto byteOffset = castToI64(adaptor.getTargetOffset(), rewriter);
    int64_t elementSize = IREE::Util::getRoundedElementByteWidth(oldType);
    auto elementOffset =
        unscaleOffset(storeOp.getLoc(), byteOffset, elementSize, rewriter);
    if (oldType.isInteger(1) || oldType.isInteger(8)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreI8Op>(
          storeOp, adaptor.getTarget(), byteOffset, adaptor.getSource());
    } else if (oldType.isInteger(16)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreI16Op>(
          storeOp, adaptor.getTarget(), elementOffset, adaptor.getSource());
    } else if (oldType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreI32Op>(
          storeOp, adaptor.getTarget(), elementOffset, adaptor.getSource());
    } else if (oldType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreI64Op>(
          storeOp, adaptor.getTarget(), elementOffset, adaptor.getSource());
    } else if (oldType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreF32Op>(
          storeOp, adaptor.getTarget(), elementOffset, adaptor.getSource());
    } else if (oldType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreF64Op>(
          storeOp, adaptor.getTarget(), elementOffset, adaptor.getSource());
    } else {
      return rewriter.notifyMatchFailure(storeOp,
                                         "invalid buffer element type");
    }
    return success();
  }
};

}  // namespace

void populateUtilBufferToVMPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  typeConverter.addConversion(
      [](IREE::Util::BufferType type) -> Optional<Type> {
        return IREE::VM::RefType::get(
            IREE::VM::BufferType::get(type.getContext()));
      });

  // TODO(benvanik): some way to handle subspans if they survive. For today we
  // require they are all removed by propagation. This won't be the case if
  // buffer subspans are returned across the ABI boundary.
  conversionTarget.addIllegalOp<IREE::Util::BufferStorageOp>();
  conversionTarget.addIllegalOp<IREE::Util::BufferSubspanOp>();

  conversionTarget
      .addIllegalOp<IREE::Util::BufferConstantOp, IREE::Util::BufferAllocOp,
                    IREE::Util::BufferDeallocOp, IREE::Util::BufferSliceOp,
                    IREE::Util::BufferSizeOp, IREE::Util::BufferCopyOp,
                    IREE::Util::BufferCompareOp, IREE::Util::BufferFillOp,
                    IREE::Util::BufferLoadOp, IREE::Util::BufferStoreOp>();

  patterns.insert<BufferConstantOpConversion>(typeConverter, context);
  patterns.insert<BufferAllocOpConversion>(typeConverter, context);
  patterns.insert<BufferDeallocOpConversion>(typeConverter, context);
  patterns.insert<BufferSliceOpConversion>(typeConverter, context);
  patterns.insert<BufferSizeOpConversion>(typeConverter, context);
  patterns.insert<BufferCopyOpConversion>(typeConverter, context);
  patterns.insert<BufferCompareOpConversion>(typeConverter, context);
  patterns.insert<BufferFillOpConversion>(typeConverter, context);
  patterns.insert<BufferLoadOpConversion>(typeConverter, context);
  patterns.insert<BufferStoreOpConversion>(typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
