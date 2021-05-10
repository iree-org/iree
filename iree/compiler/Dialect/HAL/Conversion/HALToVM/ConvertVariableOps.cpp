// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class VariableOpConversion : public OpConversionPattern<IREE::HAL::VariableOp> {
 public:
  VariableOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::VariableOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto convertedType = typeConverter.convertType(op.type());
    if (convertedType.isa<IREE::VM::RefType>() ||
        IREE::VM::RefType::isCompatible(convertedType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalRefOp>(
          op, op.sym_name(), op.is_mutable(), convertedType, op.initializer(),
          op.initial_value(), llvm::to_vector<4>(op->getDialectAttrs()));
      return success();
    } else if (convertedType.isInteger(32)) {
      auto convertedValue =
          op.initial_value().hasValue()
              ? rewriter.getI32IntegerAttr(static_cast<int32_t>(
                    op.initial_value().getValue().cast<IntegerAttr>().getInt()))
              : Attribute{};
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalI32Op>(
          op, op.sym_name(), op.is_mutable(), convertedType, op.initializer(),
          convertedValue, llvm::to_vector<4>(op->getDialectAttrs()));
      return success();
    } else if (convertedType.isInteger(64)) {
      auto convertedValue =
          op.initial_value().hasValue()
              ? rewriter.getI64IntegerAttr(
                    op.initial_value().getValue().cast<IntegerAttr>().getInt())
              : Attribute{};
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalI64Op>(
          op, op.sym_name(), op.is_mutable(), convertedType, op.initializer(),
          convertedValue, llvm::to_vector<4>(op->getDialectAttrs()));
      return success();
    } else if (convertedType.isF32()) {
      auto convertedValue = op.initial_value().hasValue()
                                ? rewriter.getF32FloatAttr(static_cast<float>(
                                      op.initial_value()
                                          .getValue()
                                          .cast<FloatAttr>()
                                          .getValueAsDouble()))
                                : Attribute{};
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalF32Op>(
          op, op.sym_name(), op.is_mutable(), convertedType, op.initializer(),
          convertedValue, llvm::to_vector<4>(op->getDialectAttrs()));
      return success();
    } else if (convertedType.isF64()) {
      auto convertedValue =
          op.initial_value().hasValue()
              ? rewriter.getF64FloatAttr(op.initial_value()
                                             .getValue()
                                             .cast<FloatAttr>()
                                             .getValueAsDouble())
              : Attribute{};
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalF64Op>(
          op, op.sym_name(), op.is_mutable(), convertedType, op.initializer(),
          convertedValue, llvm::to_vector<4>(op->getDialectAttrs()));
      return success();
    }
    return op.emitOpError("unsupported variable type");
  }

 private:
  TypeConverter &typeConverter;
};

class VariableAddressOpConversion
    : public OpConversionPattern<IREE::HAL::VariableAddressOp> {
 public:
  VariableAddressOpConversion(MLIRContext *context,
                              TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::VariableAddressOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::GlobalAddressOp>(
        op, typeConverter.convertType(op.getType()), op.variable());
    return success();
  }

 private:
  TypeConverter &typeConverter;
};

class VariableLoadOpConversion
    : public OpConversionPattern<IREE::HAL::VariableLoadOp> {
 public:
  VariableLoadOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::VariableLoadOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto operandType = op.getType();
    auto convertedType = typeConverter.convertType(operandType);
    if (IREE::VM::RefType::isCompatible(operandType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadRefOp>(op, convertedType,
                                                             op.variable());
    } else if (convertedType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadI32Op>(op, convertedType,
                                                             op.variable());
    } else if (convertedType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadI64Op>(op, convertedType,
                                                             op.variable());
    } else if (convertedType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadF32Op>(op, convertedType,
                                                             op.variable());
    } else if (convertedType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadF64Op>(op, convertedType,
                                                             op.variable());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled variable type");
    }
    return success();
  }

 private:
  TypeConverter &typeConverter;
};

class VariableLoadIndirectOpConversion
    : public OpConversionPattern<IREE::HAL::VariableLoadIndirectOp> {
 public:
  VariableLoadIndirectOpConversion(MLIRContext *context,
                                   TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::VariableLoadIndirectOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto operandType = op.getType();
    auto convertedType = typeConverter.convertType(operandType);
    if (IREE::VM::RefType::isCompatible(operandType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectRefOp>(
          op, convertedType, op.variable());
    } else if (convertedType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectI32Op>(
          op, convertedType, op.variable());
    } else if (convertedType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectI64Op>(
          op, convertedType, op.variable());
    } else if (convertedType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectF32Op>(
          op, convertedType, op.variable());
    } else if (convertedType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectF64Op>(
          op, convertedType, op.variable());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled variable type");
    }
    return success();
  }

 private:
  TypeConverter &typeConverter;
};

class VariableStoreOpConversion
    : public OpConversionPattern<IREE::HAL::VariableStoreOp> {
 public:
  VariableStoreOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::VariableStoreOp op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::VariableStoreOp::Adaptor operands(newOperands);
    auto operandType = operands.value().getType();
    if (operandType.isa<IREE::VM::RefType>()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreRefOp>(
          op, operands.value(), op.variable());
    } else if (operandType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreI32Op>(
          op, operands.value(), op.variable());
    } else if (operandType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreI64Op>(
          op, operands.value(), op.variable());
    } else if (operandType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreF32Op>(
          op, operands.value(), op.variable());
    } else if (operandType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreF64Op>(
          op, operands.value(), op.variable());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled variable type");
    }
    return success();
  }
};

class VariableStoreIndirectOpConversion
    : public OpConversionPattern<IREE::HAL::VariableStoreIndirectOp> {
 public:
  VariableStoreIndirectOpConversion(MLIRContext *context,
                                    TypeConverter &typeConverter)
      : OpConversionPattern(context) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::VariableStoreIndirectOp op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::VariableStoreIndirectOp::Adaptor operands(newOperands);
    auto operandType = operands.value().getType();
    if (operandType.isa<IREE::VM::RefType>()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectRefOp>(
          op, operands.value(), op.variable());
    } else if (operandType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectI32Op>(
          op, operands.value(), op.variable());
    } else if (operandType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectI64Op>(
          op, operands.value(), op.variable());
    } else if (operandType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectF32Op>(
          op, operands.value(), op.variable());
    } else if (operandType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectF64Op>(
          op, operands.value(), op.variable());
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled variable type");
    }
    return success();
  }
};

}  // namespace

void populateHALVariableToVMPatterns(MLIRContext *context,
                                     SymbolTable &importSymbols,
                                     TypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns) {
  patterns.insert<VariableOpConversion, VariableAddressOpConversion,
                  VariableLoadOpConversion, VariableLoadIndirectOpConversion,
                  VariableStoreOpConversion, VariableStoreIndirectOpConversion>(
      context, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
