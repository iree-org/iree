// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/StandardToHAL/ConvertStandardToHAL.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/SCF/SCF.h"
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

class FuncOpSignatureConversion : public OpConversionPattern<mlir::FuncOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::FuncOp funcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();

    // Convert the input signature types.
    // TODO(benvanik): dynamic shapes by passing in tensor dynamic dims.
    auto originalType = funcOp.getType();
    TypeConverter::SignatureConversion newSignature(
        originalType.getNumInputs());
    for (auto argType : llvm::enumerate(originalType.getInputs())) {
      if (failed(typeConverter.convertSignatureArg(
              argType.index(), argType.value(), newSignature))) {
        return failure();
      }
    }
    SmallVector<Type, 4> newResultTypes;
    if (failed(typeConverter.convertTypes(originalType.getResults(),
                                          newResultTypes))) {
      return failure();
    }

    // Replace function.
    auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
    newFuncOp.getBlocks().clear();
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    newFuncOp.setType(rewriter.getFunctionType(newSignature.getConvertedTypes(),
                                               newResultTypes));
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter,
                                           &newSignature))) {
      return failure();
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

class CallOpConversion : public OpConversionPattern<mlir::CallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::CallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 4> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return rewriter.notifyMatchFailure(op, "unable to convert result types");
    }
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, resultTypes, op.getCallee(),
                                              adaptor.operands());
    return success();
  }
};

class BranchOpConversion : public OpConversionPattern<mlir::BranchOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::BranchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::BranchOp>(op, op.getDest(),
                                                adaptor.getDestOperands());
    return success();
  }
};

class CondBranchOpConversion : public OpConversionPattern<mlir::CondBranchOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::CondBranchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<mlir::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::ReturnOp returnOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(returnOp, adaptor.operands());
    return success();
  }
};

class SelectOpConversion : public OpConversionPattern<mlir::arith::SelectOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::arith::SelectOp selectOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        selectOp, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return success();
  }
};

struct ConvertIfOp : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      scf::IfOp ifOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultTypes = llvm::to_vector<4>(llvm::map_range(
        ifOp.getResultTypes(),
        [&](Type type) { return getTypeConverter()->convertType(type); }));
    auto newOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), resultTypes,
                                            adaptor.getCondition(),
                                            ifOp.elseBlock() != nullptr);
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.eraseBlock(&newOp.getThenRegion().front());
    if (ifOp.elseBlock()) {
      rewriter.inlineRegionBefore(ifOp.getElseRegion(), newOp.getElseRegion(),
                                  newOp.getElseRegion().end());
      rewriter.eraseBlock(&newOp.getElseRegion().front());
    }
    rewriter.replaceOp(ifOp, newOp.getResults());
    return success();
  }
};

struct ConvertYieldOp : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      scf::YieldOp yieldOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, adaptor.getResults());
    return success();
  }
};

}  // namespace

void populateStandardStructuralToHALPatterns(MLIRContext *context,
                                             ConversionTarget &conversionTarget,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  conversionTarget.addLegalOp<mlir::ModuleOp>();

  // We need to rewrite certain types on operands/results so use the default
  // dynamic legality checker to force any ops using such types to run through
  // our patterns.
  conversionTarget.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getType()) &&
           typeConverter.isLegal(&op.getBody());
  });

  patterns
      .insert<FuncOpSignatureConversion, CallOpConversion, BranchOpConversion,
              CondBranchOpConversion, ReturnOpConversion, SelectOpConversion>(
          typeConverter, context);

  // TODO(benvanik): move to general utils conversion.
  addGenericLegalOp<scf::IfOp>(conversionTarget, typeConverter);
  addGenericLegalOp<scf::YieldOp>(conversionTarget, typeConverter);
  patterns.insert<ConvertIfOp, ConvertYieldOp>(typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
