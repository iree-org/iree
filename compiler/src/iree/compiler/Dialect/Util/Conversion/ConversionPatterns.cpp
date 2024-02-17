// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

void populateUtilConversionPatterns(MLIRContext *context,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  patterns
      .insert<GenericConvertTypesPattern<IREE::Util::OptimizationBarrierOp>>(
          typeConverter, context);

  typeConverter.addConversion([&](IREE::Util::PtrType type,
                                  SmallVectorImpl<Type> &results) {
    SmallVector<Type> targetTypes;
    if (failed(typeConverter.convertType(type.getTargetType(), targetTypes))) {
      return failure();
    }
    results.reserve(targetTypes.size());
    for (auto targetType : targetTypes) {
      results.push_back(IREE::Util::PtrType::get(targetType));
    }
    return success();
  });

  typeConverter.addConversion([&](IREE::Util::ListType type) {
    auto elementType = typeConverter.convertType(type.getElementType());
    return IREE::Util::ListType::get(elementType);
  });
  patterns.insert<GenericConvertTypesPattern<IREE::Util::ListCreateOp>,
                  GenericConvertTypesPattern<IREE::Util::ListGetOp>,
                  GenericConvertTypesPattern<IREE::Util::ListSetOp>>(
      typeConverter, context);
}

void populateUtilConversionPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  addGenericLegalOp<IREE::Util::OptimizationBarrierOp>(conversionTarget,
                                                       typeConverter);
  addGenericLegalOp<IREE::Util::ListCreateOp>(conversionTarget, typeConverter);
  addGenericLegalOp<IREE::Util::ListGetOp>(conversionTarget, typeConverter);
  addGenericLegalOp<IREE::Util::ListSetOp>(conversionTarget, typeConverter);

  populateUtilConversionPatterns(context, typeConverter, patterns);
}

//===----------------------------------------------------------------------===//
// Structural op conversion
//===----------------------------------------------------------------------===//

namespace {

struct ConvertInitializerOp
    : public OpConversionPattern<IREE::Util::InitializerOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::InitializerOp initializerOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();
    rewriter.startOpModification(initializerOp);
    if (failed(rewriter.convertRegionTypes(&initializerOp.getBody(),
                                           typeConverter))) {
      rewriter.cancelOpModification(initializerOp);
      return rewriter.notifyMatchFailure(initializerOp,
                                         "failed to convert region types");
    }
    rewriter.finalizeOpModification(initializerOp);
    return success();
  }
};

struct ConvertFuncOp : public OpConversionPattern<IREE::Util::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();

    // Convert the input signature types.
    // TODO(benvanik): dynamic shapes by passing in tensor dynamic dims.
    auto originalType = funcOp.getFunctionType();
    TypeConverter::SignatureConversion newSignature(
        originalType.getNumInputs());
    for (auto argType : llvm::enumerate(originalType.getInputs())) {
      if (failed(typeConverter.convertSignatureArg(
              argType.index(), argType.value(), newSignature))) {
        return rewriter.notifyMatchFailure(funcOp,
                                           "failed to convert arg type");
      }
    }
    SmallVector<Type> newResultTypes;
    if (failed(typeConverter.convertTypes(originalType.getResults(),
                                          newResultTypes))) {
      return rewriter.notifyMatchFailure(funcOp,
                                         "failed to convert result type");
    }

    // Replace function.
    auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
    newFuncOp.getBlocks().clear();
    rewriter.inlineRegionBefore(funcOp.getFunctionBody(),
                                newFuncOp.getFunctionBody(), newFuncOp.end());
    newFuncOp.setType(rewriter.getFunctionType(newSignature.getConvertedTypes(),
                                               newResultTypes));
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getFunctionBody(),
                                           typeConverter, &newSignature))) {
      return rewriter.notifyMatchFailure(funcOp,
                                         "failed to convert region types");
    }
    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct ConvertCallOp : public OpConversionPattern<IREE::Util::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return rewriter.notifyMatchFailure(op, "unable to convert result types");
    }
    auto newOp = rewriter.replaceOpWithNewOp<IREE::Util::CallOp>(
        op, resultTypes, op.getCallee(), adaptor.getOperands(),
        adaptor.getTiedOperandsAttr());
    newOp->setDialectAttrs(op->getDialectAttrs());
    return success();
  }
};

struct ConvertReturnOp : public OpConversionPattern<IREE::Util::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Util::ReturnOp>(returnOp,
                                                      adaptor.getOperands());
    return success();
  }
};

struct ConvertFuncFuncOp : public OpConversionPattern<mlir::func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();

    // Convert the input signature types.
    // TODO(benvanik): dynamic shapes by passing in tensor dynamic dims.
    auto originalType = funcOp.getFunctionType();
    TypeConverter::SignatureConversion newSignature(
        originalType.getNumInputs());
    for (auto argType : llvm::enumerate(originalType.getInputs())) {
      if (failed(typeConverter.convertSignatureArg(
              argType.index(), argType.value(), newSignature))) {
        return rewriter.notifyMatchFailure(funcOp,
                                           "failed to convert arg type");
      }
    }
    SmallVector<Type> newResultTypes;
    if (failed(typeConverter.convertTypes(originalType.getResults(),
                                          newResultTypes))) {
      return rewriter.notifyMatchFailure(funcOp,
                                         "failed to convert result type");
    }

    // Replace function.
    auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
    newFuncOp.getBlocks().clear();
    rewriter.inlineRegionBefore(funcOp.getFunctionBody(),
                                newFuncOp.getFunctionBody(), newFuncOp.end());
    newFuncOp.setType(rewriter.getFunctionType(newSignature.getConvertedTypes(),
                                               newResultTypes));
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getFunctionBody(),
                                           typeConverter, &newSignature))) {
      return rewriter.notifyMatchFailure(funcOp,
                                         "failed to convert region types");
    }
    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct ConvertFuncCallOp : public OpConversionPattern<mlir::func::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return rewriter.notifyMatchFailure(op, "unable to convert result types");
    }
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, resultTypes, op.getCallee(), adaptor.getOperands());
    return success();
  }
};

struct ConvertFuncReturnOp : public OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(returnOp,
                                                      adaptor.getOperands());
    return success();
  }
};

struct ConvertBranchOp : public OpConversionPattern<mlir::cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest(),
                                                    adaptor.getDestOperands());
    return success();
  }
};

struct ConvertCondBranchOp
    : public OpConversionPattern<mlir::cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    return success();
  }
};

struct ConvertSwitchOp : public OpConversionPattern<mlir::cf::SwitchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::SwitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::SwitchOp>(
        op, adaptor.getFlag(), op.getDefaultDestination(),
        adaptor.getDefaultOperands(), op.getCaseValuesAttr(),
        op.getCaseDestinations(), adaptor.getCaseOperands());
    return success();
  }
};

struct ConvertSelectOp : public OpConversionPattern<mlir::arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::arith::SelectOp selectOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        selectOp, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return success();
  }
};

struct ConvertIfOp : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp ifOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTypes =
        llvm::map_to_vector(ifOp.getResultTypes(), [&](Type type) {
          return getTypeConverter()->convertType(type);
        });
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
  LogicalResult
  matchAndRewrite(scf::YieldOp yieldOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, adaptor.getResults());
    return success();
  }
};

} // namespace

void populateGenericStructuralConversionPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  conversionTarget.addLegalOp<mlir::ModuleOp>();

  // We need to rewrite certain types on operands/results so use the default
  // dynamic legality checker to force any ops using such types to run through
  // our patterns.

  conversionTarget.addDynamicallyLegalOp<IREE::Util::InitializerOp>(
      [&](IREE::Util::InitializerOp op) {
        return typeConverter.isLegal(&op.getBody());
      });
  conversionTarget.addDynamicallyLegalOp<IREE::Util::FuncOp>(
      [&](IREE::Util::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody());
      });
  addGenericLegalOp<IREE::Util::CallOp>(conversionTarget, typeConverter);
  addGenericLegalOp<IREE::Util::ReturnOp>(conversionTarget, typeConverter);
  patterns.insert<ConvertInitializerOp, ConvertFuncOp, ConvertCallOp,
                  ConvertReturnOp>(typeConverter, context);

  conversionTarget.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  addGenericLegalOp<func::CallOp>(conversionTarget, typeConverter);
  addGenericLegalOp<func::ReturnOp>(conversionTarget, typeConverter);
  patterns.insert<ConvertFuncFuncOp, ConvertFuncCallOp, ConvertFuncReturnOp>(
      typeConverter, context);

  addGenericLegalOp<cf::BranchOp>(conversionTarget, typeConverter);
  addGenericLegalOp<cf::CondBranchOp>(conversionTarget, typeConverter);
  addGenericLegalOp<cf::SwitchOp>(conversionTarget, typeConverter);
  patterns.insert<ConvertBranchOp, ConvertCondBranchOp, ConvertSwitchOp>(
      typeConverter, context);

  addGenericLegalOp<arith::SelectOp>(conversionTarget, typeConverter);
  patterns.insert<ConvertSelectOp>(typeConverter, context);

  addGenericLegalOp<scf::IfOp>(conversionTarget, typeConverter);
  addGenericLegalOp<scf::YieldOp>(conversionTarget, typeConverter);
  patterns.insert<ConvertIfOp, ConvertYieldOp>(typeConverter, context);
}

} // namespace mlir::iree_compiler
