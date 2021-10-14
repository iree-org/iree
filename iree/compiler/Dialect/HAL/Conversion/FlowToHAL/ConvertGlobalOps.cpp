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

static void createInitializerFromImmediate(
    IREE::Util::GlobalOp globalOp, ElementsAttr immediateElements,
    ConversionPatternRewriter &rewriter) {
  auto loc = globalOp.getLoc();
  auto initializerOp =
      rewriter.create<IREE::Util::InitializerOp>(globalOp.getLoc());
  rewriter.setInsertionPointToStart(initializerOp.addEntryBlock());

  // Create const and store ops.
  auto constValue = rewriter.create<arith::ConstantOp>(loc, immediateElements);
  rewriter.create<IREE::Util::GlobalStoreOp>(loc, constValue.getResult(),
                                             globalOp.getName());

  rewriter.create<IREE::Util::InitializerReturnOp>(loc);
}

class GlobalOpConversion : public OpConversionPattern<IREE::Util::GlobalOp> {
 public:
  GlobalOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalOp globalOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): multiple converted type results to multiple globals.
    Optional<Attribute> initialValue = globalOp.initial_value();

    // Hoist any immediate initial_value elements to an initializer function
    // that returns it. This will then be converted by the framework to
    // an appropriate HAL Buffer-based initializer.
    if (auto initialValueElements =
            globalOp.initial_valueAttr().dyn_cast_or_null<ElementsAttr>()) {
      auto ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointAfter(globalOp);
      createInitializerFromImmediate(globalOp, initialValueElements, rewriter);
      rewriter.restoreInsertionPoint(ip);
      initialValue = llvm::None;
    }

    rewriter.setInsertionPoint(globalOp);
    auto newOp = rewriter.create<IREE::Util::GlobalOp>(
        globalOp.getLoc(), globalOp.sym_name(), globalOp.is_mutable(),
        converter.convertType(globalOp.type()), initialValue,
        llvm::to_vector<4>(globalOp->getDialectAttrs()));
    newOp.setVisibility(globalOp.getVisibility());
    rewriter.replaceOp(globalOp, {});
    return success();
  }

 private:
  TypeConverter &converter;
};

class GlobalAddressOpConversion
    : public OpConversionPattern<IREE::Util::GlobalAddressOp> {
 public:
  GlobalAddressOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalAddressOp addressOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): multiple converted type results to multiple globals.
    rewriter.replaceOpWithNewOp<IREE::Util::GlobalAddressOp>(
        addressOp, converter.convertType(addressOp.result().getType()),
        addressOp.global());
    return success();
  }

 private:
  TypeConverter &converter;
};

class GlobalLoadOpConversion
    : public OpConversionPattern<IREE::Util::GlobalLoadOp> {
 public:
  GlobalLoadOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalLoadOp loadOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): multiple converted type results to multiple globals.
    rewriter.replaceOpWithNewOp<IREE::Util::GlobalLoadOp>(
        loadOp, converter.convertType(loadOp.result().getType()),
        SymbolRefAttr::get(rewriter.getContext(), loadOp.global()));
    return success();
  }

 private:
  TypeConverter &converter;
};

class GlobalLoadIndirectOpConversion
    : public OpConversionPattern<IREE::Util::GlobalLoadIndirectOp> {
 public:
  GlobalLoadIndirectOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalLoadIndirectOp loadOp,
      llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Util::GlobalLoadIndirectOp::Adaptor operands(newOperands);
    // TODO(benvanik): multiple converted type results to multiple globals.
    rewriter.replaceOpWithNewOp<IREE::Util::GlobalLoadIndirectOp>(
        loadOp, converter.convertType(loadOp.result().getType()),
        operands.global());
    return success();
  }

 private:
  TypeConverter &converter;
};

namespace {

Value implicitCastGlobalStore(Location loc, Value storeValue, Type globalType,
                              ConversionPatternRewriter &rewriter) {
  Type storeType = storeValue.getType();

  // A limited number of implicit conversions on store are allowed.
  if (globalType != storeType) {
    if (storeType.isa<IREE::HAL::BufferViewType>() &&
        globalType.isa<IREE::HAL::BufferType>()) {
      return rewriter.create<IREE::HAL::BufferViewBufferOp>(loc, globalType,
                                                            storeValue);
    } else {
      return nullptr;
    }
  }
  return storeValue;
}

}  // namespace

class GlobalStoreOpConversion
    : public OpConversionPattern<IREE::Util::GlobalStoreOp> {
 public:
  GlobalStoreOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalStoreOp storeOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Util::GlobalStoreOp::Adaptor operands(newOperands);
    auto globalOp = storeOp.getGlobalOp();
    if (!globalOp) return failure();

    Type globalType = converter.convertType(globalOp.type());
    if (!globalType) {
      return rewriter.notifyMatchFailure(storeOp, "illegal global op type");
    }
    Value storeValue = implicitCastGlobalStore(
        storeOp.getLoc(), operands.value(), globalType, rewriter);
    if (!storeValue) {
      return rewriter.notifyMatchFailure(storeOp,
                                         "mismatched store and global type");
    }
    // TODO(benvanik): multiple converted type results to multiple globals.
    rewriter.replaceOpWithNewOp<IREE::Util::GlobalStoreOp>(
        storeOp, storeValue,
        SymbolRefAttr::get(rewriter.getContext(), storeOp.global()));
    return success();
  }

  TypeConverter &converter;
};

class GlobalStoreIndirectOpConversion
    : public OpConversionPattern<IREE::Util::GlobalStoreIndirectOp> {
 public:
  GlobalStoreIndirectOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      IREE::Util::GlobalStoreIndirectOp storeOp,
      llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Util::GlobalStoreIndirectOp::Adaptor operands(newOperands);

    Type globalType =
        operands.global().getType().cast<IREE::Util::PtrType>().getTargetType();
    Value storeValue = implicitCastGlobalStore(
        storeOp.getLoc(), operands.value(), globalType, rewriter);
    if (!storeValue) {
      return rewriter.notifyMatchFailure(storeOp,
                                         "mismatched store and global type");
    }

    // TODO(benvanik): multiple converted type results to multiple globals.
    rewriter.replaceOpWithNewOp<IREE::Util::GlobalStoreIndirectOp>(
        storeOp, storeValue, operands.global());
    return success();
  }
};

}  // namespace

void populateFlowGlobalToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &converter) {
  patterns.insert<GlobalOpConversion, GlobalAddressOpConversion,
                  GlobalLoadOpConversion, GlobalLoadIndirectOpConversion,
                  GlobalStoreOpConversion, GlobalStoreIndirectOpConversion>(
      context, converter);
}

}  // namespace iree_compiler
}  // namespace mlir
