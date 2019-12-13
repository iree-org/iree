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

  PatternMatchResult matchAndRewrite(
      IREE::HAL::VariableOp op, llvm::ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (op.type().isa<IREE::RefPtrType>()) {
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalRefOp>(
          op, op.sym_name(), op.is_mutable(),
          typeConverter.convertType(op.type()), op.initializer(),
          op.initial_value(), llvm::to_vector<4>(op.getDialectAttrs()));
      return matchSuccess();
    } else if (op.type().isIntOrIndex()) {
      // TODO(benvanik): support other types.
      rewriter.replaceOpWithNewOp<IREE::VM::GlobalI32Op>(
          op, op.sym_name(), op.is_mutable(),
          typeConverter.convertType(op.type()), op.initializer(),
          op.initial_value(), llvm::to_vector<4>(op.getDialectAttrs()));
      return matchSuccess();
    }
    return matchFailure();
  }

 private:
  TypeConverter &typeConverter;
};

class VariableLoadOpConversion
    : public OpConversionPattern<IREE::HAL::VariableLoadOp> {
 public:
  VariableLoadOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  PatternMatchResult matchAndRewrite(
      IREE::HAL::VariableLoadOp op, llvm::ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::GlobalLoadRefOp>(
        op, typeConverter.convertType(op.getType()), op.variable());
    return matchSuccess();
  }

 private:
  TypeConverter &typeConverter;
};

class VariableStoreOpConversion
    : public OpConversionPattern<IREE::HAL::VariableStoreOp> {
 public:
  VariableStoreOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context) {}

  PatternMatchResult matchAndRewrite(
      IREE::HAL::VariableStoreOp op, llvm::ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::GlobalStoreRefOp>(op, op.variable(),
                                                            operands[0]);
    return matchSuccess();
  }
};

}  // namespace

void populateHALVariableToVMPatterns(MLIRContext *context,
                                     SymbolTable &importSymbols,
                                     TypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns) {
  patterns.insert<VariableOpConversion, VariableLoadOpConversion,
                  VariableStoreOpConversion>(context, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
