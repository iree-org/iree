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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Conversion/FlowToHAL/ConvertFlowToHAL.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class VariableOpConversion
    : public OpConversionPattern<IREE::Flow::VariableOp> {
 public:
  VariableOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  PatternMatchResult matchAndRewrite(
      IREE::Flow::VariableOp variableOp, llvm::ArrayRef<Value *> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): multiple converted type results to multiple variables.
    rewriter.replaceOpWithNewOp<IREE::HAL::VariableOp>(
        variableOp, variableOp.sym_name(), variableOp.is_mutable(),
        converter.convertType(variableOp.type()), variableOp.initializer(),
        variableOp.initial_value(),
        llvm::to_vector<4>(variableOp.getDialectAttrs()));
    return matchSuccess();
  }

 private:
  TypeConverter &converter;
};

class VariableLoadOpConversion
    : public OpConversionPattern<IREE::Flow::VariableLoadOp> {
 public:
  VariableLoadOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  PatternMatchResult matchAndRewrite(
      IREE::Flow::VariableLoadOp loadOp, llvm::ArrayRef<Value *> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): multiple converted type results to multiple variables.
    rewriter.replaceOpWithNewOp<IREE::HAL::VariableLoadOp>(
        loadOp, converter.convertType(loadOp.result()->getType()),
        rewriter.getSymbolRefAttr(loadOp.variable()));
    return matchSuccess();
  }

 private:
  TypeConverter &converter;
};

class VariableStoreOpConversion
    : public OpConversionPattern<IREE::Flow::VariableStoreOp> {
 public:
  VariableStoreOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  PatternMatchResult matchAndRewrite(
      IREE::Flow::VariableStoreOp storeOp, llvm::ArrayRef<Value *> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::VariableStoreOpOperandAdaptor operands(newOperands);
    // TODO(benvanik): multiple converted type results to multiple variables.
    rewriter.replaceOpWithNewOp<IREE::HAL::VariableStoreOp>(
        storeOp, operands.value(),
        rewriter.getSymbolRefAttr(storeOp.variable()));
    return matchSuccess();
  }
};

}  // namespace

void populateFlowVariableToHALPatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns,
                                       TypeConverter &converter) {
  patterns.insert<VariableOpConversion, VariableLoadOpConversion,
                  VariableStoreOpConversion>(context, converter);
}

}  // namespace iree_compiler
}  // namespace mlir
