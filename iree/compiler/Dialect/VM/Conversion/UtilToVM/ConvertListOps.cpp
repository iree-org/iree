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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class ListCreateOpConversion
    : public OpConversionPattern<IREE::Util::ListCreateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::ListCreateOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value initialCapacity = adaptor.initial_capacity();
    if (!initialCapacity) {
      initialCapacity = rewriter.create<IREE::VM::ConstI32Op>(
          srcOp.getLoc(), rewriter.getI32IntegerAttr(0));
    }
    rewriter.replaceOpWithNewOp<IREE::VM::ListAllocOp>(
        srcOp, typeConverter->convertType(srcOp.result().getType()),
        initialCapacity);
    return success();
  }
};

class ListSizeOpConversion
    : public OpConversionPattern<IREE::Util::ListSizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::ListSizeOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ListSizeOp>(
        srcOp, typeConverter->convertType(srcOp.result().getType()),
        adaptor.list());
    return success();
  }
};

class ListResizeOpConversion
    : public OpConversionPattern<IREE::Util::ListResizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::ListResizeOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ListResizeOp>(srcOp, adaptor.list(),
                                                        adaptor.new_size());
    return success();
  }
};

class ListGetOpConversion : public OpConversionPattern<IREE::Util::ListGetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::ListGetOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = typeConverter->convertType(srcOp.result().getType());
    if (resultType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetI32Op>(
          srcOp, resultType, adaptor.list(), adaptor.index());
    } else if (resultType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetI64Op>(
          srcOp, resultType, adaptor.list(), adaptor.index());
    } else if (resultType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetF32Op>(
          srcOp, resultType, adaptor.list(), adaptor.index());
    } else if (resultType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetF64Op>(
          srcOp, resultType, adaptor.list(), adaptor.index());
    } else if (!resultType.isIntOrIndexOrFloat()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetRefOp>(
          srcOp, resultType, adaptor.list(), adaptor.index());
    } else {
      return srcOp.emitError() << "unsupported list element type in the VM";
    }
    return success();
  }
};

class ListSetOpConversion : public OpConversionPattern<IREE::Util::ListSetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::ListSetOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto valueType = adaptor.value().getType();
    if (valueType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetI32Op>(
          srcOp, adaptor.list(), adaptor.index(), adaptor.value());
    } else if (valueType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetI64Op>(
          srcOp, adaptor.list(), adaptor.index(), adaptor.value());
    } else if (valueType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetF32Op>(
          srcOp, adaptor.list(), adaptor.index(), adaptor.value());
    } else if (valueType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetF64Op>(
          srcOp, adaptor.list(), adaptor.index(), adaptor.value());
    } else if (!valueType.isIntOrIndexOrFloat()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetRefOp>(
          srcOp, adaptor.list(), adaptor.index(), adaptor.value());
    } else {
      return srcOp.emitError() << "unsupported list element type in the VM";
    }
    return success();
  }
};

}  // namespace

void populateUtilListToVMPatterns(MLIRContext *context,
                                  ConversionTarget &conversionTarget,
                                  TypeConverter &typeConverter,
                                  OwningRewritePatternList &patterns) {
  typeConverter.addConversion(
      [&typeConverter](IREE::Util::ListType type) -> Optional<Type> {
        Type elementType;
        if (type.getElementType().isa<IREE::Util::VariantType>()) {
          elementType = IREE::VM::OpaqueType::get(type.getContext());
        } else {
          elementType = typeConverter.convertType(type.getElementType());
        }
        if (!elementType) return llvm::None;
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

}  // namespace iree_compiler
}  // namespace mlir
