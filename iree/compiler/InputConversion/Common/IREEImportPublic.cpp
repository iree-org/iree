// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Allowlist of function attributes to retain when importing funcs.
constexpr const char *kRetainedAttributes[] = {
    "iree.abi",
    "iree.reflection",
    "sym_visibility",
    "noinline",
};

struct IREEImportPublicPass
    : public IREEImportPublicBase<IREEImportPublicPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Input::IREEInputDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, IREE::Util::UtilDialect,
                    mlir::func::FuncDialect, mlir::arith::ArithmeticDialect>();
  }
  void runOnOperation() override;
};

class IREETypeConverter : public TypeConverter {
 public:
  IREETypeConverter();
};

// Generic 1:1 conversion pattern which effectively just renames an op.
// It does not support regions or ops with successors.
class OneToOneConverionPattern : public ConversionPattern {
 public:
  OneToOneConverionPattern(TypeConverter &converter, StringRef srcName,
                           StringRef targetName, MLIRContext *context,
                           PatternBenefit benefit)
      : ConversionPattern(converter, srcName, benefit, context),
        targetName(targetName) {}
  LogicalResult matchAndRewrite(
      Operation *srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(srcOp->getResultTypes(),
                                           resultTypes))) {
      return srcOp->emitError()
             << "could not convert result types to IREE internal types";
    }

    OperationState state(srcOp->getLoc(), targetName, operands, resultTypes,
                         srcOp->getAttrs());
    Operation *targetOp = rewriter.createOperation(state);
    rewriter.replaceOp(srcOp, targetOp->getResults());
    return success();
  }

 private:
  StringRef targetName;
};

class BufferViewToTensorPattern
    : public OpConversionPattern<IREE::Input::BufferViewToTensorOp> {
  using OpConversionPattern<
      IREE::Input::BufferViewToTensorOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Input::BufferViewToTensorOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TensorType resultType = typeConverter->convertType(srcOp.target().getType())
                                .dyn_cast_or_null<TensorType>();
    if (!resultType) return failure();
    if (adaptor.target_dims().empty() && !resultType.hasStaticShape()) {
      // For the input dialect, we allow ops that don't have their dims
      // specified and we reify them here with the specific builder that does
      // the work.
      rewriter.replaceOpWithNewOp<IREE::HAL::TensorImportOp>(srcOp, resultType,
                                                             adaptor.source());
    } else {
      // Dynamic dims explicitly provided (or wrong, in which case the verifier
      // will get it).
      rewriter.replaceOpWithNewOp<IREE::HAL::TensorImportOp>(
          srcOp, resultType, adaptor.source(), TypeAttr::get(resultType),
          adaptor.target_dims());
    }
    return success();
  }
};

class TensorToBufferViewPattern
    : public OpConversionPattern<IREE::Input::TensorToBufferViewOp> {
  using OpConversionPattern<
      IREE::Input::TensorToBufferViewOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Input::TensorToBufferViewOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(srcOp.target().getType());
    TensorType sourceType = adaptor.source().getType().dyn_cast<TensorType>();
    if (!resultType || !sourceType) return failure();
    if (adaptor.source_dims().empty() && !sourceType.hasStaticShape()) {
      // For the input dialect, we allow ops that don't have their dims
      // specified and we reify them here with the specific builder that does
      // the work.
      rewriter.replaceOpWithNewOp<IREE::HAL::TensorExportOp>(srcOp, resultType,
                                                             adaptor.source());
    } else {
      // Dynamic dims explicitly provided (or wrong, in which case the verifier
      // will get it).
      rewriter.replaceOpWithNewOp<IREE::HAL::TensorExportOp>(
          srcOp, resultType, adaptor.source(),
          TypeAttr::get(adaptor.source().getType()), adaptor.source_dims(),
          /*target_storage=*/nullptr);
    }
    return success();
  }
};

class BuiltinFuncOpPattern : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      FuncOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FunctionType srcFuncType = srcOp.getType();
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

    // Create new function with converted argument and result types.
    // Note that attributes are dropped. Consider preserving some if needed.
    auto newFuncType = mlir::FunctionType::get(
        srcOp.getContext(), signatureConversion.getConvertedTypes(),
        convertedResultTypes);
    auto newFuncOp =
        rewriter.create<FuncOp>(srcOp.getLoc(), srcOp.getName(), newFuncType);
    rewriter.inlineRegionBefore(srcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Retain function attributes in the allowlist.
    auto retainedAttributes = ArrayRef<const char *>(
        kRetainedAttributes,
        sizeof(kRetainedAttributes) / sizeof(kRetainedAttributes[0]));
    for (auto retainAttrName : retainedAttributes) {
      StringRef attrName(retainAttrName);
      Attribute attr = srcOp->getAttr(attrName);
      if (attr) {
        newFuncOp->setAttr(attrName, attr);
      }
    }

    // Tell the rewriter to convert the region signature.
    TypeConverter &typeConverter = *getTypeConverter();
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    rewriter.replaceOp(srcOp, llvm::None);
    return success();
  }
};

class GlobalOpPattern : public OpConversionPattern<IREE::Input::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Input::GlobalOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type newType = typeConverter->convertType(srcOp.type());
    if (!newType) return failure();
    auto globalOp = rewriter.replaceOpWithNewOp<IREE::Util::GlobalOp>(
        srcOp, srcOp.getName(), srcOp.is_mutable(), newType,
        srcOp.initial_value());
    globalOp.setVisibility(srcOp.getVisibility());
    if (srcOp.initializer().hasValue()) {
      auto initializerOp =
          rewriter.create<IREE::Util::InitializerOp>(srcOp.getLoc());
      auto ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(initializerOp.addEntryBlock());
      auto callOp = rewriter.create<mlir::func::CallOp>(
          srcOp.getLoc(), srcOp.initializerAttr(), TypeRange{newType});
      rewriter.create<IREE::Util::GlobalStoreOp>(
          srcOp.getLoc(), callOp.getResult(0), srcOp.getName());
      rewriter.create<IREE::Util::InitializerReturnOp>(srcOp.getLoc());
      rewriter.restoreInsertionPoint(ip);
    }
    return success();
  }
};

// Matches any op and generically converts types. Matches with benefit 0.
class GenericTypeConvert : public ConversionPattern {
 public:
  GenericTypeConvert(TypeConverter &converter, MLIRContext *context,
                     PatternBenefit benefit)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), benefit, context) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<NamedAttribute, 4> newAttr;
    llvm::append_range(newAttr, op->getAttrs());
    llvm::SmallVector<Type, 4> newResults;
    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttr, op->getSuccessors());
    for (Region &r : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion->getArgumentTypes(), result);
      rewriter.applySignatureConversion(newRegion, result);
    }
    Operation *newOp = rewriter.createOperation(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

}  // namespace

IREETypeConverter::IREETypeConverter() {
  addConversion([](Type t) { return t; });
  addConversion([=](IREE::Input::BufferViewType t) {
    return IREE::HAL::BufferViewType::get(t.getContext());
  });
  addConversion([=](IREE::Input::ListType t) -> IREE::Util::ListType {
    auto subType = convertType(t.getElementType());
    if (!subType) return nullptr;
    return IREE::Util::ListType::get(subType);
  });
  addConversion([=](IREE::Input::PtrType t) -> IREE::Util::PtrType {
    auto subType = convertType(t.getTargetType());
    if (!subType) return nullptr;
    return IREE::Util::PtrType::get(subType);
  });
  addConversion([](IREE::Input::VariantType t) {
    return IREE::Util::VariantType::get(t.getContext());
  });
}

void IREEImportPublicPass::runOnOperation() {
  auto &context = getContext();
  RewritePatternSet patterns(&getContext());
  ConversionTarget target(getContext());
  target.addLegalDialect<IREE::Flow::FlowDialect>();
  target.addLegalDialect<IREE::HAL::HALDialect>();
  target.addLegalDialect<IREE::Util::UtilDialect>();
  target.addIllegalDialect<IREE::Input::IREEInputDialect>();

  auto ireeDialect = context.getOrLoadDialect<IREE::Input::IREEInputDialect>();
  auto isIllegalType = [&](Type t) {
    return t.getDialect().getTypeID() == ireeDialect->getTypeID();
  };
  auto isLegallyTypedOp = [&](Operation *op) -> bool {
    for (Type type : op->getResultTypes()) {
      if (isIllegalType(type)) return false;
    }
    for (Type type : op->getOperandTypes()) {
      if (isIllegalType(type)) return false;
    }
    return true;
  };

  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
    for (Type type : funcOp.getType().getInputs()) {
      if (isIllegalType(type)) return false;
    }
    for (Type type : funcOp.getType().getResults()) {
      if (isIllegalType(type)) return false;
    }
    for (Block &block : funcOp.body()) {
      for (Type type : block.getArgumentTypes()) {
        if (isIllegalType(type)) return false;
      }
    }
    return true;
  });
  target.markUnknownOpDynamicallyLegal(isLegallyTypedOp);

  IREETypeConverter typeConverter;
  PatternBenefit specific_benefit = 100;
  patterns.insert<GenericTypeConvert>(typeConverter, &getContext(), 0);
  patterns.insert<BuiltinFuncOpPattern>(typeConverter, &getContext(),
                                        specific_benefit);
  patterns.insert<BufferViewToTensorPattern>(typeConverter, &getContext(),
                                             specific_benefit);
  patterns.insert<TensorToBufferViewPattern>(typeConverter, &getContext(),
                                             specific_benefit);
  patterns.insert<GlobalOpPattern>(typeConverter, &getContext(), 0);

#define ONETOONE(SrcOpTy, TargetOpTy)             \
  patterns.insert<OneToOneConverionPattern>(      \
      typeConverter, SrcOpTy::getOperationName(), \
      TargetOpTy::getOperationName(), &getContext(), specific_benefit)

  ONETOONE(IREE::Input::BufferViewRankOp, IREE::HAL::BufferViewRankOp);
  ONETOONE(IREE::Input::BufferViewDimOp, IREE::HAL::BufferViewDimOp);
  ONETOONE(IREE::Input::ListCreateOp, IREE::Util::ListCreateOp);
  ONETOONE(IREE::Input::ListSizeOp, IREE::Util::ListSizeOp);
  ONETOONE(IREE::Input::ListResizeOp, IREE::Util::ListResizeOp);
  ONETOONE(IREE::Input::ListGetOp, IREE::Util::ListGetOp);
  ONETOONE(IREE::Input::ListSetOp, IREE::Util::ListSetOp);
  ONETOONE(IREE::Input::NullOp, IREE::Util::NullOp);
  ONETOONE(IREE::Input::TensorCloneOp, IREE::Flow::TensorCloneOp);
  ONETOONE(IREE::Input::TensorLoadOp, IREE::Flow::TensorLoadOp);
  ONETOONE(IREE::Input::TensorReshapeOp, IREE::Flow::TensorReshapeOp);
  ONETOONE(IREE::Input::TensorSliceOp, IREE::Flow::TensorSliceOp);
  ONETOONE(IREE::Input::TensorSplatOp, IREE::Flow::TensorSplatOp);
  ONETOONE(IREE::Input::TensorStoreOp, IREE::Flow::TensorStoreOp);
  ONETOONE(IREE::Input::TensorUpdateOp, IREE::Flow::TensorUpdateOp);
  ONETOONE(IREE::Input::TensorTraceOp, IREE::Flow::TensorTraceOp);
  ONETOONE(IREE::Input::GlobalAddressOp, IREE::Util::GlobalAddressOp);
  ONETOONE(IREE::Input::GlobalLoadOp, IREE::Util::GlobalLoadOp);
  ONETOONE(IREE::Input::GlobalLoadIndirectOp, IREE::Util::GlobalLoadIndirectOp);
  ONETOONE(IREE::Input::GlobalStoreOp, IREE::Util::GlobalStoreOp);
  ONETOONE(IREE::Input::GlobalStoreIndirectOp,
           IREE::Util::GlobalStoreIndirectOp);

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createIREEImportPublicPass() {
  return std::make_unique<IREEImportPublicPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
