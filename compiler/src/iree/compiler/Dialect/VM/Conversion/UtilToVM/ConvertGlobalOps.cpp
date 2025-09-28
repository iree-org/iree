// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct InitializerOpConversion
    : public OpConversionPattern<IREE::Util::InitializerOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::Util::InitializerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = IREE::VM::InitializerOp::create(rewriter, op.getLoc());
    rewriter.cloneRegionBefore(op.getBody(), newOp.getBody(),
                               newOp.getBody().begin());

    // Tell the rewriter to convert the region signature.
    const TypeConverter &typeConverter = *getTypeConverter();
    TypeConverter::SignatureConversion signatureConversion(0);
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), typeConverter,
                                           &signatureConversion))) {
      return rewriter.notifyMatchFailure(op, "failed to convert region types");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<IREE::Util::ReturnOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::Util::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ReturnOp>(op);
    return success();
  }
};

struct GlobalOpConversion : public OpConversionPattern<IREE::Util::GlobalOp> {
  TypeConverter &typeConverter;
  GlobalOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *newOp = nullptr;
    auto convertedType = typeConverter.convertType(op.getType());
    const bool isInitialized =
        op.getInitialValueAttr() &&
        !isa<IREE::Util::UninitializedAttr>(op.getInitialValueAttr());
    if (llvm::isa<IREE::VM::RefType>(convertedType) ||
        IREE::VM::RefType::isCompatible(convertedType)) {
      newOp = rewriter.replaceOpWithNewOp<IREE::VM::GlobalRefOp>(
          op, op.getSymName(), op.getIsMutable(), convertedType,
          llvm::to_vector(op->getDialectAttrs()));
    } else if (convertedType.isInteger(32)) {
      std::optional<TypedAttr> convertedValue = std::nullopt;
      if (isInitialized) {
        convertedValue = rewriter.getI32IntegerAttr(static_cast<int32_t>(
            llvm::cast<IntegerAttr>(op.getInitialValue().value()).getInt()));
      }
      newOp = rewriter.replaceOpWithNewOp<IREE::VM::GlobalI32Op>(
          op, op.getSymName(), op.getIsMutable(), convertedType, convertedValue,
          llvm::to_vector(op->getDialectAttrs()));
    } else if (convertedType.isInteger(64)) {
      std::optional<TypedAttr> convertedValue = std::nullopt;
      if (isInitialized) {
        convertedValue = rewriter.getI64IntegerAttr(
            llvm::cast<IntegerAttr>(op.getInitialValue().value()).getInt());
      }
      newOp = rewriter.replaceOpWithNewOp<IREE::VM::GlobalI64Op>(
          op, op.getSymName(), op.getIsMutable(), convertedType, convertedValue,
          llvm::to_vector(op->getDialectAttrs()));
    } else if (convertedType.isF32()) {
      std::optional<TypedAttr> convertedValue = std::nullopt;
      if (isInitialized) {
        convertedValue = rewriter.getF32FloatAttr(static_cast<float>(
            llvm::cast<FloatAttr>(op.getInitialValue().value())
                .getValueAsDouble()));
      }
      newOp = rewriter.replaceOpWithNewOp<IREE::VM::GlobalF32Op>(
          op, op.getSymName(), op.getIsMutable(), convertedType, convertedValue,
          llvm::to_vector(op->getDialectAttrs()));
    } else if (convertedType.isF64()) {
      std::optional<TypedAttr> convertedValue = std::nullopt;
      if (isInitialized) {
        convertedValue = rewriter.getF64FloatAttr(
            llvm::cast<FloatAttr>(op.getInitialValue().value())
                .getValueAsDouble());
      }
      newOp = rewriter.replaceOpWithNewOp<IREE::VM::GlobalF64Op>(
          op, op.getSymName(), op.getIsMutable(), convertedType, convertedValue,
          llvm::to_vector(op->getDialectAttrs()));
    } else {
      return op.emitOpError("unsupported global type");
    }

    // New global carries the same visibility as the original.
    cast<SymbolOpInterface>(newOp).setVisibility(op.getVisibility());

    return success();
  }
};

struct GlobalAddressOpConversion
    : public OpConversionPattern<IREE::Util::GlobalAddressOp> {
  TypeConverter &typeConverter;
  GlobalAddressOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalAddressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::GlobalAddressOp>(
        op, typeConverter.convertType(op.getType()), op.getGlobalAttr(),
        op.getIsImmutableAttr());
    return success();
  }
};

struct GlobalLoadOpConversion
    : public OpConversionPattern<IREE::Util::GlobalLoadOp> {
  TypeConverter &typeConverter;
  GlobalLoadOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = op.getType();
    auto convertedType = typeConverter.convertType(operandType);
    if (IREE::VM::RefType::isCompatible(operandType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadRefOp>(
          op, convertedType, op.getGlobalAttr(), adaptor.getIsImmutableAttr());
    } else if (convertedType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadI32Op>(
          op, convertedType, op.getGlobalAttr(), adaptor.getIsImmutableAttr());
    } else if (convertedType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadI64Op>(
          op, convertedType, op.getGlobalAttr(), adaptor.getIsImmutableAttr());
    } else if (convertedType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadF32Op>(
          op, convertedType, op.getGlobalAttr(), adaptor.getIsImmutableAttr());
    } else if (convertedType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadF64Op>(
          op, convertedType, op.getGlobalAttr(), adaptor.getIsImmutableAttr());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled global type");
    }
    return success();
  }
};

struct GlobalLoadIndirectOpConversion
    : public OpConversionPattern<IREE::Util::GlobalLoadIndirectOp> {
  TypeConverter &typeConverter;
  GlobalLoadIndirectOpConversion(MLIRContext *context,
                                 TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalLoadIndirectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = op.getType();
    auto convertedType = typeConverter.convertType(operandType);
    if (IREE::VM::RefType::isCompatible(operandType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectRefOp>(
          op, convertedType, adaptor.getGlobal(), adaptor.getIsImmutableAttr());
    } else if (convertedType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectI32Op>(
          op, convertedType, adaptor.getGlobal(), adaptor.getIsImmutableAttr());
    } else if (convertedType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectI64Op>(
          op, convertedType, adaptor.getGlobal(), adaptor.getIsImmutableAttr());
    } else if (convertedType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectF32Op>(
          op, convertedType, adaptor.getGlobal(), adaptor.getIsImmutableAttr());
    } else if (convertedType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectF64Op>(
          op, convertedType, adaptor.getGlobal(), adaptor.getIsImmutableAttr());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled global type");
    }
    return success();
  }
};

struct GlobalStoreOpConversion
    : public OpConversionPattern<IREE::Util::GlobalStoreOp> {
  GlobalStoreOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = adaptor.getValue().getType();
    if (llvm::isa<IREE::VM::RefType>(operandType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreRefOp>(
          op, adaptor.getValue(), op.getGlobal());
    } else if (operandType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreI32Op>(
          op, adaptor.getValue(), op.getGlobal());
    } else if (operandType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreI64Op>(
          op, adaptor.getValue(), op.getGlobal());
    } else if (operandType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreF32Op>(
          op, adaptor.getValue(), op.getGlobal());
    } else if (operandType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreF64Op>(
          op, adaptor.getValue(), op.getGlobal());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled global type");
    }
    return success();
  }
};

struct GlobalStoreIndirectOpConversion
    : public OpConversionPattern<IREE::Util::GlobalStoreIndirectOp> {
  GlobalStoreIndirectOpConversion(MLIRContext *context,
                                  TypeConverter &typeConverter)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalStoreIndirectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = adaptor.getValue().getType();
    if (llvm::isa<IREE::VM::RefType>(operandType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectRefOp>(
          op, adaptor.getValue(), adaptor.getGlobal());
    } else if (operandType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectI32Op>(
          op, adaptor.getValue(), adaptor.getGlobal());
    } else if (operandType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectI64Op>(
          op, adaptor.getValue(), adaptor.getGlobal());
    } else if (operandType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectF32Op>(
          op, adaptor.getValue(), adaptor.getGlobal());
    } else if (operandType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectF64Op>(
          op, adaptor.getValue(), adaptor.getGlobal());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled global type");
    }
    return success();
  }
};

} // namespace

void populateUtilGlobalToVMPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  conversionTarget
      .addIllegalOp<IREE::Util::InitializerOp, IREE::Util::ReturnOp>();
  patterns.insert<InitializerOpConversion, ReturnOpConversion>(typeConverter,
                                                               context);

  conversionTarget.addIllegalOp<
      IREE::Util::GlobalOp, IREE::Util::GlobalAddressOp,
      IREE::Util::GlobalLoadOp, IREE::Util::GlobalLoadIndirectOp,
      IREE::Util::GlobalStoreOp, IREE::Util::GlobalStoreIndirectOp>();
  patterns.insert<GlobalOpConversion, GlobalAddressOpConversion,
                  GlobalLoadOpConversion, GlobalLoadIndirectOpConversion,
                  GlobalStoreOpConversion, GlobalStoreIndirectOpConversion>(
      context, typeConverter);
}

} // namespace mlir::iree_compiler
