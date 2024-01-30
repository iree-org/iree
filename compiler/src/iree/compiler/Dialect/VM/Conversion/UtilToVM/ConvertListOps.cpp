// Copyright 2019 The IREE Authors
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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

static Value castToI32(Value value, OpBuilder &builder) {
  if (value.getType().isInteger(32))
    return value;
  return builder.createOrFold<IREE::VM::TruncI64I32Op>(
      value.getLoc(), builder.getI32Type(), value);
}

static Value castToIndex(Value value, OpBuilder &builder) {
  if (value.getType().isIndex())
    return value;
  return builder.createOrFold<arith::IndexCastOp>(
      value.getLoc(), builder.getIndexType(), value);
}

class ListCreateOpConversion
    : public OpConversionPattern<IREE::Util::ListCreateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::ListCreateOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value initialCapacity = adaptor.getInitialCapacity();
    if (initialCapacity) {
      initialCapacity = castToI32(initialCapacity, rewriter);
    } else {
      initialCapacity = rewriter.create<IREE::VM::ConstI32Op>(
          srcOp.getLoc(), rewriter.getI32IntegerAttr(0));
    }
    rewriter.replaceOpWithNewOp<IREE::VM::ListAllocOp>(
        srcOp, typeConverter->convertType(srcOp.getResult().getType()),
        initialCapacity);
    return success();
  }
};

class ListSizeOpConversion
    : public OpConversionPattern<IREE::Util::ListSizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::ListSizeOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value size = rewriter.create<IREE::VM::ListSizeOp>(
        srcOp.getLoc(), rewriter.getI32Type(), adaptor.getList());
    rewriter.replaceOp(srcOp, castToIndex(size, rewriter));
    return success();
  }
};

class ListResizeOpConversion
    : public OpConversionPattern<IREE::Util::ListResizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::ListResizeOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ListResizeOp>(
        srcOp, adaptor.getList(), castToI32(adaptor.getNewSize(), rewriter));
    return success();
  }
};

class ListGetOpConversion : public OpConversionPattern<IREE::Util::ListGetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::ListGetOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto index = castToI32(adaptor.getIndex(), rewriter);
    auto resultType = typeConverter->convertType(srcOp.getResult().getType());
    if (resultType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetI32Op>(
          srcOp, resultType, adaptor.getList(), index);
    } else if (resultType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetI64Op>(
          srcOp, resultType, adaptor.getList(), index);
    } else if (resultType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetF32Op>(
          srcOp, resultType, adaptor.getList(), index);
    } else if (resultType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetF64Op>(
          srcOp, resultType, adaptor.getList(), index);
    } else if (!resultType.isIntOrIndexOrFloat()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetRefOp>(
          srcOp, resultType, adaptor.getList(), index);
    } else {
      return srcOp.emitError() << "unsupported list element type in the VM";
    }
    return success();
  }
};

class ListSetOpConversion : public OpConversionPattern<IREE::Util::ListSetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::ListSetOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto index = castToI32(adaptor.getIndex(), rewriter);
    auto valueType = adaptor.getValue().getType();
    if (valueType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetI32Op>(
          srcOp, adaptor.getList(), index, adaptor.getValue());
    } else if (valueType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetI64Op>(
          srcOp, adaptor.getList(), index, adaptor.getValue());
    } else if (valueType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetF32Op>(
          srcOp, adaptor.getList(), index, adaptor.getValue());
    } else if (valueType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetF64Op>(
          srcOp, adaptor.getList(), index, adaptor.getValue());
    } else if (!valueType.isIntOrIndexOrFloat()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetRefOp>(
          srcOp, adaptor.getList(), index, adaptor.getValue());
    } else {
      return srcOp.emitError() << "unsupported list element type in the VM";
    }
    return success();
  }
};

} // namespace

void populateUtilListToVMPatterns(MLIRContext *context,
                                  ConversionTarget &conversionTarget,
                                  TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  typeConverter.addConversion(
      [&typeConverter](IREE::Util::ListType type) -> std::optional<Type> {
        Type elementType;
        if (llvm::isa<IREE::Util::ObjectType>(type.getElementType()) ||
            llvm::isa<IREE::Util::VariantType>(type.getElementType())) {
          elementType = IREE::VM::OpaqueType::get(type.getContext());
        } else {
          elementType = typeConverter.convertType(type.getElementType());
        }
        if (!elementType)
          return std::nullopt;
        return IREE::VM::RefType::get(IREE::VM::ListType::get(elementType));
      });

  conversionTarget.addIllegalOp<
      IREE::Util::ListCreateOp, IREE::Util::ListSizeOp,
      IREE::Util::ListResizeOp, IREE::Util::ListGetOp, IREE::Util::ListSetOp>();

  patterns
      .insert<ListCreateOpConversion, ListSizeOpConversion,
              ListResizeOpConversion, ListGetOpConversion, ListSetOpConversion>(
          typeConverter, context);
}

} // namespace mlir::iree_compiler
