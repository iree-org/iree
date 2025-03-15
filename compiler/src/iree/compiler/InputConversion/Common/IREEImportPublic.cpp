// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::InputConversion {

#define GEN_PASS_DEF_IREEIMPORTPUBLICPASS
#include "iree/compiler/InputConversion/Common/Passes.h.inc"

namespace {

class IREEImportPublicPass final
    : public impl::IREEImportPublicPassBase<IREEImportPublicPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, IREE::HAL::HALDialect,
                    IREE::Util::UtilDialect, mlir::func::FuncDialect,
                    mlir::arith::ArithDialect>();
  }
  void runOnOperation() override;
};

class IREETypeConverter : public TypeConverter {
public:
  IREETypeConverter();
};

//===----------------------------------------------------------------------===//
// Func dialect -> Util patterns
//===----------------------------------------------------------------------===//

class FuncFuncOpPattern : public OpConversionPattern<func::FuncOp> {
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
        "iree.reflection", "stream.affinity", "vm.fallback",
        "vm.signature",    "vm.version",      "nosideeffects",
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

class FuncCallOpPattern : public OpConversionPattern<func::CallOp> {
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

class FuncReturnOpPattern : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Util::ReturnOp>(srcOp,
                                                      adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Generic conversion
//===----------------------------------------------------------------------===//

// Matches any op and generically converts types. Matches with benefit 0.
class GenericTypeConvert : public ConversionPattern {
public:
  GenericTypeConvert(TypeConverter &converter, MLIRContext *context,
                     PatternBenefit benefit)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), benefit, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<NamedAttribute> newAttr;
    llvm::append_range(newAttr, op->getAttrs());
    llvm::SmallVector<Type> newResults;
    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttr, op->getSuccessors());
    for (Region &r : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion->getArgumentTypes(), result);
      rewriter.applySignatureConversion(&newRegion->front(), result);
    }
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

IREETypeConverter::IREETypeConverter() {
  addConversion([](Type t) { return t; });
}

void IREEImportPublicPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  ConversionTarget target(getContext());
  target.addLegalDialect<IREE::Flow::FlowDialect>();
  target.addLegalDialect<IREE::HAL::HALDialect>();
  target.addLegalDialect<IREE::Util::UtilDialect>();

  IREETypeConverter typeConverter;
  auto isLegallyTypedOp = [&](Operation *op) -> bool {
    for (Type type : op->getResultTypes()) {
      if (!typeConverter.isLegal(type))
        return false;
    }
    for (Type type : op->getOperandTypes()) {
      if (!typeConverter.isLegal(type))
        return false;
    }
    return true;
  };
  target.markUnknownOpDynamicallyLegal(isLegallyTypedOp);

  PatternBenefit specific_benefit = 100;
  patterns.insert<GenericTypeConvert>(typeConverter, &getContext(), 0);

  target.addDynamicallyLegalDialect<func::FuncDialect>(
      [&](Operation *op) -> std::optional<bool> {
        // Allow the func dialect within nested modules but not in the top-level
        // one that represents the host program.
        return op->getParentOfType<mlir::ModuleOp>() != getOperation();
      });
  patterns.insert<FuncFuncOpPattern>(typeConverter, &getContext(),
                                     specific_benefit);
  patterns.insert<FuncCallOpPattern>(typeConverter, &getContext(),
                                     specific_benefit);
  patterns.insert<FuncReturnOpPattern>(typeConverter, &getContext(),
                                       specific_benefit);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace mlir::iree_compiler::InputConversion
