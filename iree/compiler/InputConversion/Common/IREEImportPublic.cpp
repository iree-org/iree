// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/IREE/IREEDialect.h"
#include "iree-dialects/Dialect/IREE/IREEOps.h"
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
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace IREEPublic = mlir::iree;

namespace mlir {
namespace iree_compiler {

namespace {

// Allowlist of function attributes to retain when importing funcs.
constexpr const char *kRetainedAttributes[] = {
    "iree.reflection",
    "sym_visibility",
    "noinline",
};

struct IREEImportPublicPass
    : public IREEImportPublicBase<IREEImportPublicPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::iree::IREEDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, IREE::Util::UtilDialect>();
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
    : public OpConversionPattern<IREEPublic::BufferViewToTensorOp> {
  using OpConversionPattern<
      IREEPublic::BufferViewToTensorOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREEPublic::BufferViewToTensorOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREEPublic::BufferViewToTensorOpAdaptor adaptor(operands);
    Type resultType = typeConverter->convertType(srcOp.target().getType());
    if (!resultType) return failure();
    rewriter.replaceOpWithNewOp<IREE::HAL::TensorCastOp>(
        srcOp, resultType, adaptor.source(), adaptor.target_dims());
    return success();
  }
};

class TensorToBufferViewPattern
    : public OpConversionPattern<IREEPublic::TensorToBufferViewOp> {
  using OpConversionPattern<
      IREEPublic::TensorToBufferViewOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREEPublic::TensorToBufferViewOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREEPublic::TensorToBufferViewOpAdaptor adaptor(operands);
    Type resultType = typeConverter->convertType(srcOp.target().getType());
    if (!resultType) return failure();
    rewriter.replaceOpWithNewOp<IREE::HAL::TensorCastOp>(
        srcOp, resultType, adaptor.source(), adaptor.source_dims());
    return success();
  }
};

class BuiltinFuncOpPattern : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      FuncOp srcOp, ArrayRef<Value> operands,
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
  addConversion([=](IREEPublic::BufferViewType t) {
    return IREE::HAL::BufferViewType::get(t.getContext());
  });
  addConversion([=](IREEPublic::ListType t) -> IREE::Util::ListType {
    auto subType = convertType(t.getElementType());
    if (!subType) return nullptr;
    return IREE::Util::ListType::get(subType);
  });
  addConversion([=](IREEPublic::PtrType t) -> IREE::Util::PtrType {
    auto subType = convertType(t.getTargetType());
    if (!subType) return nullptr;
    return IREE::Util::PtrType::get(subType);
  });
  addConversion([](IREEPublic::VariantType t) {
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
  target.addIllegalDialect<IREEPublic::IREEDialect>();

  auto ireeDialect = context.getOrLoadDialect<IREEPublic::IREEDialect>();
  auto isIllegalType = [&](Type t) {
    return t.getDialect().getTypeID() == ireeDialect->getTypeID();
  };

  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
    for (Type type : funcOp.getType().getInputs()) {
      if (isIllegalType(type)) return false;
    }
    for (Type type : funcOp.getType().getResults()) {
      if (isIllegalType(type)) return false;
    }
    return true;
  });

  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    for (Type type : op->getResultTypes()) {
      if (isIllegalType(type)) return false;
    }
    for (Type type : op->getOperandTypes()) {
      if (isIllegalType(type)) return false;
    }
    return true;
  });

  IREETypeConverter typeConverter;
  PatternBenefit specific_benefit = 100;
  patterns.insert<GenericTypeConvert>(typeConverter, &getContext(), 0);
  patterns.insert<BuiltinFuncOpPattern>(typeConverter, &getContext(),
                                        specific_benefit);
  patterns.insert<BufferViewToTensorPattern>(typeConverter, &getContext(),
                                             specific_benefit);
  patterns.insert<TensorToBufferViewPattern>(typeConverter, &getContext(),
                                             specific_benefit);

#define ONETOONE(SrcOpTy, TargetOpTy)             \
  patterns.insert<OneToOneConverionPattern>(      \
      typeConverter, SrcOpTy::getOperationName(), \
      TargetOpTy::getOperationName(), &getContext(), specific_benefit)

  ONETOONE(IREEPublic::BufferViewRankOp, IREE::HAL::BufferViewRankOp);
  ONETOONE(IREEPublic::BufferViewDimOp, IREE::HAL::BufferViewDimOp);
  ONETOONE(IREEPublic::ListCreateOp, IREE::Util::ListCreateOp);
  ONETOONE(IREEPublic::ListSizeOp, IREE::Util::ListSizeOp);
  ONETOONE(IREEPublic::ListResizeOp, IREE::Util::ListResizeOp);
  ONETOONE(IREEPublic::ListGetOp, IREE::Util::ListGetOp);
  ONETOONE(IREEPublic::ListSetOp, IREE::Util::ListSetOp);
  ONETOONE(IREEPublic::NullOp, IREE::Util::NullOp);
  ONETOONE(IREEPublic::TensorCloneOp, IREE::Flow::TensorCloneOp);
  ONETOONE(IREEPublic::TensorLoadOp, IREE::Flow::TensorLoadOp);
  ONETOONE(IREEPublic::TensorReshapeOp, IREE::Flow::TensorReshapeOp);
  ONETOONE(IREEPublic::TensorSliceOp, IREE::Flow::TensorSliceOp);
  ONETOONE(IREEPublic::TensorSplatOp, IREE::Flow::TensorSplatOp);
  ONETOONE(IREEPublic::TensorStoreOp, IREE::Flow::TensorStoreOp);
  ONETOONE(IREEPublic::TensorUpdateOp, IREE::Flow::TensorUpdateOp);
  ONETOONE(IREEPublic::TensorTraceOp, IREE::Flow::TensorTraceOp);
  ONETOONE(IREEPublic::GlobalOp, IREE::Util::GlobalOp);
  ONETOONE(IREEPublic::GlobalAddressOp, IREE::Util::GlobalAddressOp);
  ONETOONE(IREEPublic::GlobalLoadOp, IREE::Util::GlobalLoadOp);
  ONETOONE(IREEPublic::GlobalLoadIndirectOp, IREE::Util::GlobalLoadIndirectOp);
  ONETOONE(IREEPublic::GlobalStoreOp, IREE::Util::GlobalStoreOp);
  ONETOONE(IREEPublic::GlobalStoreIndirectOp,
           IREE::Util::GlobalStoreIndirectOp);

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createIREEImportPublicPass() {
  return std::make_unique<IREEImportPublicPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
