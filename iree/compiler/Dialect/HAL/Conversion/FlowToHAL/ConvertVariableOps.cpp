// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Conversion/FlowToHAL/ConvertFlowToHAL.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

static FuncOp createInitializerFromImmediate(
    IREE::Flow::VariableOp variableOp, ElementsAttr immediateElements,
    ConversionPatternRewriter &rewriter) {
  auto loc = variableOp.getLoc();
  auto initializerType = FunctionType::get(rewriter.getContext(), {},
                                           {immediateElements.getType()});
  // TODO(b/145839814): It is presently possible to collide with user
  // provided symbols and it seems like it shouldn't be.
  auto uniqueName = (Twine("__") + variableOp.getName() + "_initializer").str();
  auto initializerFuncOp =
      rewriter.create<FuncOp>(variableOp.getLoc(), uniqueName, initializerType);
  rewriter.createBlock(&initializerFuncOp.getBody(), initializerFuncOp.begin(),
                       initializerType.getInputs());

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
      rewriter.setInsertionPointAfter(variableOp);
      auto initializerFunc = createInitializerFromImmediate(
          variableOp, initialValueElements, rewriter);
      initializer = initializerFunc.getName();
      initialValue = llvm::None;
    }

    rewriter.setInsertionPoint(variableOp);
    auto newOp = rewriter.create<IREE::HAL::VariableOp>(
        variableOp.getLoc(), variableOp.sym_name(), variableOp.is_mutable(),
        converter.convertType(variableOp.type()), initializer, initialValue,
        llvm::to_vector<4>(variableOp->getDialectAttrs()));
    newOp.setVisibility(variableOp.getVisibility());
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

namespace {

Value implicitCastVariableStore(Location loc, Value storeValue,
                                Type variableType,
                                ConversionPatternRewriter &rewriter) {
  Type storeType = storeValue.getType();

  // A limited number of implicit conversions on store are allowed.
  if (variableType != storeType) {
    if (storeType.isa<IREE::HAL::BufferViewType>() &&
        variableType.isa<IREE::HAL::BufferType>()) {
      return rewriter.create<IREE::HAL::BufferViewBufferOp>(loc, variableType,
                                                            storeValue);
    } else {
      return nullptr;
    }
  }
  return storeValue;
}

}  // namespace

class VariableStoreOpConversion
    : public OpConversionPattern<IREE::Flow::VariableStoreOp> {
 public:
  VariableStoreOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      IREE::Flow::VariableStoreOp storeOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::VariableStoreOp::Adaptor operands(newOperands);
    auto *variableOp =
        SymbolTable::lookupNearestSymbolFrom(storeOp, storeOp.variable());
    if (!variableOp) return failure();

    Type variableType = getVariableType(variableOp);
    if (!variableType) {
      return rewriter.notifyMatchFailure(storeOp, "illegal variable op type");
    }
    Value storeValue = implicitCastVariableStore(
        storeOp.getLoc(), operands.value(), variableType, rewriter);
    if (!storeValue) {
      return rewriter.notifyMatchFailure(storeOp,
                                         "mismatched store and variable type");
    }
    // TODO(benvanik): multiple converted type results to multiple variables.
    rewriter.replaceOpWithNewOp<IREE::HAL::VariableStoreOp>(
        storeOp, storeValue, rewriter.getSymbolRefAttr(storeOp.variable()));
    return success();
  }

  Type getVariableType(Operation *variableOp) const {
    if (auto halVariableOp = dyn_cast<IREE::HAL::VariableOp>(variableOp)) {
      return halVariableOp.type();
    } else if (auto flowVariableOp =
                   dyn_cast<IREE::Flow::VariableOp>(variableOp)) {
      // If the variable referent is not in dominance order at the module level,
      // it may not have been converted yet. So get the unconverted op and
      // convert its type to allow variables and uses in any order.
      return converter.convertType(flowVariableOp.type());
    }

    return nullptr;
  }

  TypeConverter &converter;
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

    Type variableType = operands.variable()
                            .getType()
                            .cast<IREE::Util::PtrType>()
                            .getTargetType();
    Value storeValue = implicitCastVariableStore(
        storeOp.getLoc(), operands.value(), variableType, rewriter);
    if (!storeValue) {
      return rewriter.notifyMatchFailure(storeOp,
                                         "mismatched store and variable type");
    }
    // TODO(benvanik): multiple converted type results to multiple variables.
    rewriter.replaceOpWithNewOp<IREE::HAL::VariableStoreIndirectOp>(
        storeOp, storeValue, storeOp.variable());
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
