// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Conversion/FuncToUtil/Patterns.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct FuncFuncOpPattern : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType srcFuncType = srcOp.getFunctionType();
    TypeConverter::SignatureConversion signatureConversion(
        srcOp.getNumArguments());

    // Convert function arguments.
    for (unsigned i = 0, e = srcFuncType.getNumInputs(); i < e; ++i) {
      if (failed(getTypeConverter()->convertSignatureArg(
              i, srcFuncType.getInput(i), signatureConversion))) {
        return rewriter.notifyMatchFailure(srcOp, "argument failed to convert");
      }
    }

    // Convert function results.
    SmallVector<Type, 1> convertedResultTypes;
    if (failed(getTypeConverter()->convertTypes(srcFuncType.getResults(),
                                                convertedResultTypes))) {
      return rewriter.notifyMatchFailure(srcOp, "results failed to convert");
    }

    // Build tied operands index mapping results back to operands.
    SmallVector<int64_t> tiedOperands;
    bool anyTiedOperands = false;
    for (unsigned i = 0; i < srcFuncType.getNumResults(); ++i) {
      auto tiedAttr =
          srcOp.getResultAttrOfType<IntegerAttr>(i, "iree.abi.tied");
      if (tiedAttr) {
        tiedOperands.push_back(tiedAttr.getInt());
      } else {
        tiedOperands.push_back(-1);
      }
    }
    auto tiedOperandsAttr = anyTiedOperands
                                ? rewriter.getIndexArrayAttr(tiedOperands)
                                : ArrayAttr{};

    // Create new function with converted argument and result types.
    // Note that attributes are dropped. Consider preserving some if needed.
    auto newFuncType = mlir::FunctionType::get(
        srcOp.getContext(), signatureConversion.getConvertedTypes(),
        convertedResultTypes);
    auto newFuncOp = rewriter.create<IREE::Util::FuncOp>(
        srcOp.getLoc(), srcOp.getName(), newFuncType, tiedOperandsAttr);
    newFuncOp.setSymVisibilityAttr(srcOp.getSymVisibilityAttr());
    rewriter.inlineRegionBefore(srcOp.getBody(), newFuncOp.getFunctionBody(),
                                newFuncOp.end());

    // Handle defacto attrs to specialized ones.
    if (srcOp->hasAttr("noinline")) {
      newFuncOp.setInliningPolicyAttr(
          rewriter.getAttr<IREE::Util::InlineNeverAttr>());
    }

    // Allowlist of function attributes to retain when importing funcs.
    constexpr const char *kRetainedAttributes[] = {
        // Ends up in serialized modules.
        "iree.reflection",
        // Controls placement.
        "stream.affinity",
        // VM interop.
        "vm.fallback",
        "vm.signature",
        "vm.version",
        // Overrides.
        // TODO(benvanik): add a util.func structured attr ala inlining policy.
        "nosideeffects",
    };
    auto retainedAttributes = ArrayRef<const char *>(
        kRetainedAttributes,
        sizeof(kRetainedAttributes) / sizeof(kRetainedAttributes[0]));
    for (auto retainAttrName : retainedAttributes) {
      StringRef attrName(retainAttrName);
      Attribute attr = srcOp->getAttr(attrName);
      if (attr)
        newFuncOp->setAttr(attrName, attr);
    }

    // Copy all arg/result attrs. We could filter these.
    if (auto argAttrs = srcOp.getAllArgAttrs()) {
      newFuncOp.setAllArgAttrs(argAttrs);
    }
    if (auto resultAttrs = srcOp.getAllResultAttrs()) {
      newFuncOp.setAllResultAttrs(resultAttrs);
    }

    // Tell the rewriter to convert the region signature.
    const TypeConverter &typeConverter = *getTypeConverter();
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getFunctionBody(),
                                           typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    rewriter.eraseOp(srcOp);
    return success();
  }
};

struct FuncCallOpPattern : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::CallOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 1> resultTypes;
    if (failed(getTypeConverter()->convertTypes(srcOp.getResultTypes(),
                                                resultTypes))) {
      return rewriter.notifyMatchFailure(srcOp, "results failed to convert");
    }
    auto tiedOperandsAttr =
        srcOp->getAttrOfType<ArrayAttr>("iree.abi.tied_operands");
    rewriter.replaceOpWithNewOp<IREE::Util::CallOp>(
        srcOp, resultTypes, srcOp.getCallee(), adaptor.getOperands(),
        tiedOperandsAttr, srcOp.getArgAttrsAttr(), srcOp.getResAttrsAttr());
    return success();
  }
};

struct FuncReturnOpPattern : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Util::ReturnOp>(srcOp,
                                                      adaptor.getOperands());
    return success();
  }
};

} // namespace

void populateFuncToUtilPatterns(MLIRContext *context,
                                ConversionTarget &conversionTarget,
                                TypeConverter &typeConverter,
                                RewritePatternSet &patterns,
                                mlir::ModuleOp rootModuleOp) {
  // Allow the func dialect within nested modules but not in the top-level
  // one that represents the host program.
  conversionTarget.addDynamicallyLegalDialect<func::FuncDialect>(
      [=](Operation *op) -> std::optional<bool> {
        return op->getParentOfType<mlir::ModuleOp>() != rootModuleOp;
      });

  patterns.insert<FuncFuncOpPattern>(typeConverter, context);
  patterns.insert<FuncCallOpPattern>(typeConverter, context);
  patterns.insert<FuncReturnOpPattern>(typeConverter, context);
}

} // namespace mlir::iree_compiler
