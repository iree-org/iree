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

namespace mlir {
namespace iree_compiler {
namespace {

class GlobalOpConversion : public OpConversionPattern<IREE::Util::GlobalOp> {
 public:
  GlobalOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto convertedType = typeConverter.convertType(op.type());
    if (convertedType.isa<IREE::VM::RefType>() ||
        IREE::VM::RefType::isCompatible(convertedType)) {
      auto newOp = rewriter.replaceOpWithNewOp<IREE::VM::GlobalRefOp>(
          op, op.sym_name(), op.is_mutable(), convertedType, op.initializer(),
          op.initial_value(), llvm::to_vector<4>(op->getDialectAttrs()));
      newOp.setVisibility(op.getVisibility());
      return success();
    } else if (convertedType.isInteger(32)) {
      auto convertedValue =
          op.initial_value().hasValue()
              ? rewriter.getI32IntegerAttr(static_cast<int32_t>(
                    op.initial_value().getValue().cast<IntegerAttr>().getInt()))
              : Attribute{};
      auto newOp = rewriter.replaceOpWithNewOp<IREE::VM::GlobalI32Op>(
          op, op.sym_name(), op.is_mutable(), convertedType, op.initializer(),
          convertedValue, llvm::to_vector<4>(op->getDialectAttrs()));
      newOp.setVisibility(op.getVisibility());
      return success();
    } else if (convertedType.isInteger(64)) {
      auto convertedValue =
          op.initial_value().hasValue()
              ? rewriter.getI64IntegerAttr(
                    op.initial_value().getValue().cast<IntegerAttr>().getInt())
              : Attribute{};
      auto newOp = rewriter.replaceOpWithNewOp<IREE::VM::GlobalI64Op>(
          op, op.sym_name(), op.is_mutable(), convertedType, op.initializer(),
          convertedValue, llvm::to_vector<4>(op->getDialectAttrs()));
      newOp.setVisibility(op.getVisibility());
      return success();
    } else if (convertedType.isF32()) {
      auto convertedValue = op.initial_value().hasValue()
                                ? rewriter.getF32FloatAttr(static_cast<float>(
                                      op.initial_value()
                                          .getValue()
                                          .cast<FloatAttr>()
                                          .getValueAsDouble()))
                                : Attribute{};
      auto newOp = rewriter.replaceOpWithNewOp<IREE::VM::GlobalF32Op>(
          op, op.sym_name(), op.is_mutable(), convertedType, op.initializer(),
          convertedValue, llvm::to_vector<4>(op->getDialectAttrs()));
      newOp.setVisibility(op.getVisibility());
      return success();
    } else if (convertedType.isF64()) {
      auto convertedValue =
          op.initial_value().hasValue()
              ? rewriter.getF64FloatAttr(op.initial_value()
                                             .getValue()
                                             .cast<FloatAttr>()
                                             .getValueAsDouble())
              : Attribute{};
      auto newOp = rewriter.replaceOpWithNewOp<IREE::VM::GlobalF64Op>(
          op, op.sym_name(), op.is_mutable(), convertedType, op.initializer(),
          convertedValue, llvm::to_vector<4>(op->getDialectAttrs()));
      newOp.setVisibility(op.getVisibility());
      return success();
    }
    return op.emitOpError("unsupported global type");
  }

 private:
  TypeConverter &typeConverter;
};

class GlobalAddressOpConversion
    : public OpConversionPattern<IREE::Util::GlobalAddressOp> {
 public:
  GlobalAddressOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalAddressOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::GlobalAddressOp>(
        op, typeConverter.convertType(op.getType()), op.global());
    return success();
  }

 private:
  TypeConverter &typeConverter;
};

class GlobalLoadOpConversion
    : public OpConversionPattern<IREE::Util::GlobalLoadOp> {
 public:
  GlobalLoadOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalLoadOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto operandType = op.getType();
    auto convertedType = typeConverter.convertType(operandType);
    if (IREE::VM::RefType::isCompatible(operandType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadRefOp>(op, convertedType,
                                                             op.global());
    } else if (convertedType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadI32Op>(op, convertedType,
                                                             op.global());
    } else if (convertedType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadI64Op>(op, convertedType,
                                                             op.global());
    } else if (convertedType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadF32Op>(op, convertedType,
                                                             op.global());
    } else if (convertedType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadF64Op>(op, convertedType,
                                                             op.global());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled global type");
    }
    return success();
  }

 private:
  TypeConverter &typeConverter;
};

class GlobalLoadIndirectOpConversion
    : public OpConversionPattern<IREE::Util::GlobalLoadIndirectOp> {
 public:
  GlobalLoadIndirectOpConversion(MLIRContext *context,
                                 TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalLoadIndirectOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto operandType = op.getType();
    auto convertedType = typeConverter.convertType(operandType);
    if (IREE::VM::RefType::isCompatible(operandType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectRefOp>(
          op, convertedType, op.global());
    } else if (convertedType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectI32Op>(
          op, convertedType, op.global());
    } else if (convertedType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectI64Op>(
          op, convertedType, op.global());
    } else if (convertedType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectF32Op>(
          op, convertedType, op.global());
    } else if (convertedType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectF64Op>(
          op, convertedType, op.global());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled global type");
    }
    return success();
  }

 private:
  TypeConverter &typeConverter;
};

class GlobalStoreOpConversion
    : public OpConversionPattern<IREE::Util::GlobalStoreOp> {
 public:
  GlobalStoreOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalStoreOp op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Util::GlobalStoreOp::Adaptor operands(newOperands);
    auto operandType = operands.value().getType();
    if (operandType.isa<IREE::VM::RefType>()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreRefOp>(
          op, operands.value(), op.global());
    } else if (operandType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreI32Op>(
          op, operands.value(), op.global());
    } else if (operandType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreI64Op>(
          op, operands.value(), op.global());
    } else if (operandType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreF32Op>(
          op, operands.value(), op.global());
    } else if (operandType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreF64Op>(
          op, operands.value(), op.global());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled global type");
    }
    return success();
  }
};

class GlobalStoreIndirectOpConversion
    : public OpConversionPattern<IREE::Util::GlobalStoreIndirectOp> {
 public:
  GlobalStoreIndirectOpConversion(MLIRContext *context,
                                  TypeConverter &typeConverter)
      : OpConversionPattern(context) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalStoreIndirectOp op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Util::GlobalStoreIndirectOp::Adaptor operands(newOperands);
    auto operandType = operands.value().getType();
    if (operandType.isa<IREE::VM::RefType>()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectRefOp>(
          op, operands.value(), op.global());
    } else if (operandType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectI32Op>(
          op, operands.value(), op.global());
    } else if (operandType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectI64Op>(
          op, operands.value(), op.global());
    } else if (operandType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectF32Op>(
          op, operands.value(), op.global());
    } else if (operandType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectF64Op>(
          op, operands.value(), op.global());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled global type");
    }
    return success();
  }
};

}  // namespace

void populateUtilGlobalToVMPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    OwningRewritePatternList &patterns) {
  conversionTarget.addIllegalOp<
      IREE::Util::GlobalOp, IREE::Util::GlobalAddressOp,
      IREE::Util::GlobalLoadOp, IREE::Util::GlobalLoadIndirectOp,
      IREE::Util::GlobalStoreOp, IREE::Util::GlobalStoreIndirectOp>();
  patterns.insert<GlobalOpConversion, GlobalAddressOpConversion,
                  GlobalLoadOpConversion, GlobalLoadIndirectOpConversion,
                  GlobalStoreOpConversion, GlobalStoreIndirectOpConversion>(
      context, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
