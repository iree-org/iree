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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

static FuncOp createInitializerFromImmediate(
    IREE::Flow::VariableOp variableOp, ElementsAttr immediateElements,
    ConversionPatternRewriter &rewriter) {
  auto loc = variableOp.getLoc();
  auto initializerType = FunctionType::get({}, {immediateElements.getType()},
                                           rewriter.getContext());
  // TODO(b/145839814): It is presently possible to collide with user
  // provided symbols and it seems like it shouldn't be.
  auto uniqueName = (Twine("__") + variableOp.getName() + "_initializer").str();
  auto initializerFuncOp =
      rewriter.create<FuncOp>(variableOp.getLoc(), uniqueName, initializerType,
                              ArrayRef<NamedAttribute>{});
  auto *entryBlock = initializerFuncOp.addEntryBlock();
  rewriter.setInsertionPointToEnd(entryBlock);

  // Create const and return ops.
  auto constValue = rewriter.create<ConstantOp>(loc, immediateElements);
  rewriter.create<mlir::ReturnOp>(loc, constValue.getResult());
  return initializerFuncOp;
}

class VariableOpConversion
    : public OpConversionPattern<IREE::Flow::VariableOp> {
 public:
  VariableOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      IREE::Flow::VariableOp variableOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): multiple converted type results to multiple variables.
    Optional<StringRef> initializer = variableOp.initializer();
    Optional<Attribute> initialValue = variableOp.initial_value();

    // Hoist any immediate initial_value elements to an initializer function
    // that returns it. This will then be converted by the framework to
    // an appropriate HAL Buffer-based initializer.
    if (auto initialValueElements =
            variableOp.initial_valueAttr().dyn_cast_or_null<ElementsAttr>()) {
      auto initializerFunc = createInitializerFromImmediate(
          variableOp, initialValueElements, rewriter);
      initializer = initializerFunc.getName();
      initialValue = llvm::None;
    }

    rewriter.setInsertionPoint(variableOp);
    auto newOp = rewriter.create<IREE::HAL::VariableOp>(
        variableOp.getLoc(), variableOp.sym_name(), variableOp.is_mutable(),
        converter.convertType(variableOp.type()), initializer, initialValue,
        llvm::to_vector<4>(variableOp.getDialectAttrs()));
    SymbolTable::setSymbolVisibility(
        newOp, SymbolTable::getSymbolVisibility(variableOp));
    rewriter.replaceOp(variableOp, {});
    return success();
  }

 private:
  TypeConverter &converter;
};

class VariableAddressOpConversion
    : public OpConversionPattern<IREE::Flow::VariableAddressOp> {
 public:
  VariableAddressOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      IREE::Flow::VariableAddressOp addressOp,
      llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): multiple converted type results to multiple variables.
    rewriter.replaceOpWithNewOp<IREE::HAL::VariableAddressOp>(
        addressOp, converter.convertType(addressOp.result().getType()),
        addressOp.variable());
    return success();
  }

 private:
  TypeConverter &converter;
};

class VariableLoadOpConversion
    : public OpConversionPattern<IREE::Flow::VariableLoadOp> {
 public:
  VariableLoadOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      IREE::Flow::VariableLoadOp loadOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): multiple converted type results to multiple variables.
    rewriter.replaceOpWithNewOp<IREE::HAL::VariableLoadOp>(
        loadOp, converter.convertType(loadOp.result().getType()),
        rewriter.getSymbolRefAttr(loadOp.variable()));
    return success();
  }

 private:
  TypeConverter &converter;
};

class VariableLoadIndirectOpConversion
    : public OpConversionPattern<IREE::Flow::VariableLoadIndirectOp> {
 public:
  VariableLoadIndirectOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      IREE::Flow::VariableLoadIndirectOp loadOp,
      llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): multiple converted type results to multiple variables.
    rewriter.replaceOpWithNewOp<IREE::HAL::VariableLoadIndirectOp>(
        loadOp, converter.convertType(loadOp.result().getType()),
        loadOp.variable());
    return success();
  }

 private:
  TypeConverter &converter;
};

class VariableStoreOpConversion
    : public OpConversionPattern<IREE::Flow::VariableStoreOp> {
 public:
  VariableStoreOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      IREE::Flow::VariableStoreOp storeOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::VariableStoreOp::Adaptor operands(newOperands);
    // TODO(benvanik): multiple converted type results to multiple variables.
    rewriter.replaceOpWithNewOp<IREE::HAL::VariableStoreOp>(
        storeOp, operands.value(),
        rewriter.getSymbolRefAttr(storeOp.variable()));
    return success();
  }
};

class VariableStoreIndirectOpConversion
    : public OpConversionPattern<IREE::Flow::VariableStoreIndirectOp> {
 public:
  VariableStoreIndirectOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      IREE::Flow::VariableStoreIndirectOp storeOp,
      llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::VariableStoreIndirectOp::Adaptor operands(newOperands);
    // TODO(benvanik): multiple converted type results to multiple variables.
    rewriter.replaceOpWithNewOp<IREE::HAL::VariableStoreIndirectOp>(
        storeOp, operands.value(), storeOp.variable());
    return success();
  }
};

}  // namespace

void populateFlowVariableToHALPatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns,
                                       TypeConverter &converter) {
  patterns.insert<VariableOpConversion, VariableAddressOpConversion,
                  VariableLoadOpConversion, VariableLoadIndirectOpConversion,
                  VariableStoreOpConversion, VariableStoreIndirectOpConversion>(
      context, converter);
}

}  // namespace iree_compiler
}  // namespace mlir
