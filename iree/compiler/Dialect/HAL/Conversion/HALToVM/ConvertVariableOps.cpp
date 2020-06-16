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
          op.initial_value(), llvm::to_vector<4>(op.getDialectAttrs()));
      return success();
    } else if (convertedType.isSignlessIntOrIndex()) {
      // TODO(benvanik): support other types.
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalI32Op>(
          op, op.sym_name(), op.is_mutable(), convertedType, op.initializer(),
          op.initial_value(), llvm::to_vector<4>(op.getDialectAttrs()));
      return success();
    }
    return failure();
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
    if (IREE::VM::RefType::isCompatible(op.getType())) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadRefOp>(
          op, typeConverter.convertType(op.getType()), op.variable());
    } else {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadI32Op>(
          op, typeConverter.convertType(op.getType()), op.variable());
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
      IREE::HAL::VariableLoadIndirectOp op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::VariableLoadIndirectOp::Adaptor operands(newOperands);
    if (IREE::VM::RefType::isCompatible(op.getType())) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectRefOp>(
          op, typeConverter.convertType(op.getType()), operands.variable());
    } else {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadIndirectI32Op>(
          op, typeConverter.convertType(op.getType()), operands.variable());
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
    if (operands.value().getType().isa<IREE::VM::RefType>()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreRefOp>(
          op, operands.value(), op.variable());
    } else {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreI32Op>(
          op, operands.value(), op.variable());
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
    if (operands.value().getType().isa<IREE::VM::RefType>()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectRefOp>(
          op, operands.value(), operands.variable());
    } else {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreIndirectI32Op>(
          op, operands.value(), operands.variable());
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
