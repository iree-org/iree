// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
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
    registry.insert<IREE::Input::IREEInputDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, IREE::Util::UtilDialect,
                    mlir::func::FuncDialect, mlir::arith::ArithDialect>();
  }
  void runOnOperation() override;
};

class IREETypeConverter : public TypeConverter {
public:
  IREETypeConverter();
};

//===----------------------------------------------------------------------===//
// Attributes conversion from IREE::Input to IREE::HAL.
//===----------------------------------------------------------------------===//

template <typename To, typename From, typename Converter>
static SmallVector<To> convertAttributes(ArrayRef<From> src, Converter fn) {
  SmallVector<To> result;
  llvm::transform(src, std::back_inserter(result), fn);
  return result;
}

template <typename To, typename From, typename Converter>
static ArrayAttr convertArrayAttribute(ArrayAttr src, Converter fn) {
  SmallVector<Attribute> result;
  for (auto attr : src) {
    if (auto arr = dyn_cast<ArrayAttr>(attr)) {
      result.push_back(convertArrayAttribute<To, From, Converter>(arr, fn));
    } else {
      result.push_back(fn(cast<From>(attr)));
    }
  }
  return ArrayAttr::get(src.getContext(), result);
}

static IREE::HAL::DescriptorType
convertDescriptorType(IREE::Input::DescriptorType src) {
  switch (src) {
  case IREE::Input::DescriptorType::StorageBuffer:
    return IREE::HAL::DescriptorType::StorageBuffer;
  case IREE::Input::DescriptorType::UniformBuffer:
    return IREE::HAL::DescriptorType::UniformBuffer;
  default:
    llvm_unreachable("Unexpected descriptor type");
  }
}

static IREE::HAL::DescriptorFlags
convertDescriptorFlags(std::optional<IREE::Input::DescriptorFlags> src) {
  if (!src.has_value())
    return IREE::HAL::DescriptorFlags::None;
  switch (*src) {
  default:
  case IREE::Input::DescriptorFlags::None:
    return IREE::HAL::DescriptorFlags::None;
  case IREE::Input::DescriptorFlags::ReadOnly:
    return IREE::HAL::DescriptorFlags::ReadOnly;
  }
}

static IREE::HAL::PipelineBindingAttr
convertPipelineBinding(IREE::Input::PipelineBindingAttr src) {
  return IREE::HAL::PipelineBindingAttr::get(
      src.getContext(), convertDescriptorType(src.getType()),
      convertDescriptorFlags(src.getFlags()));
}

static std::optional<IREE::HAL::PipelineLayoutFlags> convertPipelineLayoutFlags(
    std::optional<IREE::Input::PipelineLayoutFlags> src) {
  if (!src.has_value())
    return std::nullopt;
  switch (*src) {
  case IREE::Input::PipelineLayoutFlags::None:
    return IREE::HAL::PipelineLayoutFlags::None;
  case IREE::Input::PipelineLayoutFlags::Indirect:
    return IREE::HAL::PipelineLayoutFlags::Indirect;
  default:
    return std::nullopt;
  }
}

static IREE::HAL::PipelineLayoutAttr
convertPipelineLayout(IREE::Input::PipelineLayoutAttr src) {
  return IREE::HAL::PipelineLayoutAttr::get(
      src.getContext(),
      convertAttributes<IREE::HAL::PipelineBindingAttr>(src.getBindings(),
                                                        convertPipelineBinding),
      src.getConstants(), convertPipelineLayoutFlags(src.getFlags()));
}

static IREE::HAL::ExecutableObjectAttr
convertExecutableObject(IREE::Input::ExecutableObjectAttr src) {
  return IREE::HAL::ExecutableObjectAttr::get(
      src.getContext(), src.getPath(),
      dyn_cast_if_present<IREE::Util::SerializableAttrInterface>(
          src.getData()));
}

static IREE::HAL::ExecutableTargetAttr
convertExecutableTarget(IREE::Input::ExecutableTargetAttr src) {
  return IREE::HAL::ExecutableTargetAttr::get(src.getContext(),
                                              src.getBackend(), src.getFormat(),
                                              src.getConfiguration());
}

static IREE::HAL::ExecutableObjectsAttr
convertExecutableObjects(IREE::Input::ExecutableObjectsAttr src) {
  return IREE::HAL::ExecutableObjectsAttr::get(
      src.getContext(),
      convertArrayAttribute<IREE::HAL::ExecutableTargetAttr,
                            IREE::Input::ExecutableTargetAttr>(
          src.getTargets(), convertExecutableTarget),
      convertArrayAttribute<IREE::HAL::ExecutableObjectAttr,
                            IREE::Input::ExecutableObjectAttr>(
          src.getTargetObjects(), convertExecutableObject));
}

//===----------------------------------------------------------------------===//
// Generic 1:1 conversion pattern which effectively just renames an op.
// It does not support regions or ops with successors.
//===----------------------------------------------------------------------===//

class OneToOneConverionPattern : public ConversionPattern {
public:
  OneToOneConverionPattern(TypeConverter &converter, StringRef srcName,
                           StringRef targetName, MLIRContext *context,
                           PatternBenefit benefit)
      : ConversionPattern(converter, srcName, benefit, context),
        targetName(targetName) {}
  LogicalResult
  matchAndRewrite(Operation *srcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(srcOp->getResultTypes(),
                                           resultTypes))) {
      return srcOp->emitError()
             << "could not convert result types to IREE internal types";
    }

    OperationState state(srcOp->getLoc(), targetName, operands, resultTypes,
                         srcOp->getAttrs());
    Operation *targetOp = rewriter.create(state);
    rewriter.replaceOp(srcOp, targetOp->getResults());
    return success();
  }

private:
  StringRef targetName;
};

//===----------------------------------------------------------------------===//
// Tensor operations conversion patterns
//===----------------------------------------------------------------------===//

class TensorImportPattern
    : public OpConversionPattern<IREE::Input::TensorImportOp> {
  using OpConversionPattern<IREE::Input::TensorImportOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Input::TensorImportOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorType resultType = llvm::dyn_cast_if_present<TensorType>(
        typeConverter->convertType(srcOp.getTarget().getType()));
    if (!resultType)
      return failure();
    if (adaptor.getTargetDims().empty() && !resultType.hasStaticShape()) {
      // For the input dialect, we allow ops that don't have their dims
      // specified and we reify them here with the specific builder that does
      // the work.
      rewriter.replaceOpWithNewOp<IREE::HAL::TensorImportOp>(
          srcOp, resultType, adaptor.getSource(), TypeAttr::get(resultType),
          /*name=*/nullptr,
          /*affinity=*/nullptr);
    } else {
      // Dynamic dims explicitly provided (or wrong, in which case the verifier
      // will get it).
      rewriter.replaceOpWithNewOp<IREE::HAL::TensorImportOp>(
          srcOp, resultType, adaptor.getSource(), TypeAttr::get(resultType),
          adaptor.getTargetDims(), /*wait_fence=*/Value{}, /*name=*/nullptr,
          /*affinity=*/nullptr);
    }
    return success();
  }
};

class TensorExportPattern
    : public OpConversionPattern<IREE::Input::TensorExportOp> {
  using OpConversionPattern<IREE::Input::TensorExportOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Input::TensorExportOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(srcOp.getTarget().getType());
    auto sourceType = llvm::dyn_cast<TensorType>(adaptor.getSource().getType());
    if (!resultType || !sourceType)
      return failure();
    if (adaptor.getSourceDims().empty() && !sourceType.hasStaticShape()) {
      // For the input dialect, we allow ops that don't have their dims
      // specified and we reify them here with the specific builder that does
      // the work.
      rewriter.replaceOpWithNewOp<IREE::HAL::TensorExportOp>(
          srcOp, resultType, adaptor.getSource(),
          TypeAttr::get(adaptor.getSource().getType()), /*name=*/nullptr,
          /*affinity=*/nullptr);
    } else {
      // Dynamic dims explicitly provided (or wrong, in which case the verifier
      // will get it).
      rewriter.replaceOpWithNewOp<IREE::HAL::TensorExportOp>(
          srcOp, resultType, adaptor.getSource(),
          TypeAttr::get(adaptor.getSource().getType()), adaptor.getSourceDims(),
          /*name=*/nullptr,
          /*affinity=*/nullptr);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Executable source conversion patterns
//===----------------------------------------------------------------------===//

class ExecutableSourcePattern
    : public OpConversionPattern<IREE::Input::ExecutableSourceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Input::ExecutableSourceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto halSource = rewriter.create<IREE::HAL::ExecutableSourceOp>(
        srcOp.getLoc(), srcOp.getSymVisibilityAttr(), srcOp.getSymNameAttr(),
        convertExecutableObjects(srcOp.getObjectsAttr()));
    rewriter.inlineRegionBefore(srcOp.getBody(), halSource.getBody(),
                                halSource.getBody().end());
    rewriter.eraseOp(srcOp);
    return success();
  }
};

class ExecutableExportPattern
    : public OpConversionPattern<IREE::Input::ExecutableExportOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Input::ExecutableExportOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::ExecutableExportOp>(
        srcOp, srcOp.getSymNameAttr(), srcOp.getOrdinalAttr(),
        convertPipelineLayout(srcOp.getLayout()), srcOp.getWorkgroupSizeAttr(),
        srcOp.getSubgroupSizeAttr(), srcOp.getWorkgroupLocalMemoryAttr());
    return success();
  }
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

class GlobalOpPattern : public OpConversionPattern<IREE::Input::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Input::GlobalOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newType = typeConverter->convertType(srcOp.getType());
    if (!newType)
      return failure();
    auto globalOp = rewriter.replaceOpWithNewOp<IREE::Util::GlobalOp>(
        srcOp, srcOp.getName(), srcOp.getIsMutable(), newType,
        srcOp.getInitialValue());
    globalOp.setVisibility(srcOp.getVisibility());
    if (srcOp.getInitializer().has_value()) {
      auto initializerOp =
          rewriter.create<IREE::Util::InitializerOp>(srcOp.getLoc());
      auto ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(initializerOp.addEntryBlock());
      auto callOp = rewriter.create<mlir::func::CallOp>(
          srcOp.getLoc(), srcOp.getInitializerAttr(), TypeRange{newType});
      rewriter.create<IREE::Util::GlobalStoreOp>(
          srcOp.getLoc(), callOp.getResult(0), srcOp.getName());
      rewriter.create<IREE::Util::ReturnOp>(srcOp.getLoc());
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
  addConversion([=](IREE::Input::BufferType t) {
    return IREE::HAL::BufferType::get(t.getContext());
  });
  addConversion([=](IREE::Input::BufferViewType t) {
    return IREE::HAL::BufferViewType::get(t.getContext());
  });
  addConversion([=](IREE::Input::ByteBufferType t) {
    return IREE::Util::BufferType::get(t.getContext());
  });
  addConversion([=](IREE::Input::ListType t) -> IREE::Util::ListType {
    auto subType = convertType(t.getElementType());
    if (!subType)
      return nullptr;
    return IREE::Util::ListType::get(subType);
  });
  addConversion([=](IREE::Input::PtrType t) -> IREE::Util::PtrType {
    auto subType = convertType(t.getTargetType());
    if (!subType)
      return nullptr;
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
      if (isIllegalType(type))
        return false;
    }
    for (Type type : op->getOperandTypes()) {
      if (isIllegalType(type))
        return false;
    }
    return true;
  };
  target.markUnknownOpDynamicallyLegal(isLegallyTypedOp);

  IREETypeConverter typeConverter;
  PatternBenefit specific_benefit = 100;
  patterns.insert<GenericTypeConvert>(typeConverter, &getContext(), 0);
  patterns.insert<GlobalOpPattern>(typeConverter, &getContext(), 0);
  patterns.insert<TensorExportPattern, TensorImportPattern>(
      typeConverter, &getContext(), specific_benefit);
  patterns.insert<ExecutableSourcePattern, ExecutableExportPattern>(
      typeConverter, &getContext(), specific_benefit);

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

#define ONE_TO_ONE(SrcOpTy, TargetOpTy)                                        \
  patterns.insert<OneToOneConverionPattern>(                                   \
      typeConverter, SrcOpTy::getOperationName(),                              \
      TargetOpTy::getOperationName(), &getContext(), specific_benefit)

  ONE_TO_ONE(IREE::Input::BufferSubspanOp, IREE::HAL::BufferSubspanOp);
  ONE_TO_ONE(IREE::Input::BufferViewCreateOp, IREE::HAL::BufferViewCreateOp);
  ONE_TO_ONE(IREE::Input::BufferViewRankOp, IREE::HAL::BufferViewRankOp);
  ONE_TO_ONE(IREE::Input::BufferViewDimOp, IREE::HAL::BufferViewDimOp);
  ONE_TO_ONE(IREE::Input::ByteBufferConstantOp, IREE::Util::BufferConstantOp);
  ONE_TO_ONE(IREE::Input::ListCreateOp, IREE::Util::ListCreateOp);
  ONE_TO_ONE(IREE::Input::ListSizeOp, IREE::Util::ListSizeOp);
  ONE_TO_ONE(IREE::Input::ListResizeOp, IREE::Util::ListResizeOp);
  ONE_TO_ONE(IREE::Input::ListGetOp, IREE::Util::ListGetOp);
  ONE_TO_ONE(IREE::Input::ListSetOp, IREE::Util::ListSetOp);
  ONE_TO_ONE(IREE::Input::NullOp, IREE::Util::NullOp);
  ONE_TO_ONE(IREE::Input::TensorCloneOp, IREE::Flow::TensorCloneOp);
  ONE_TO_ONE(IREE::Input::TensorLoadOp, IREE::Flow::TensorLoadOp);
  ONE_TO_ONE(IREE::Input::TensorReshapeOp, IREE::Flow::TensorReshapeOp);
  ONE_TO_ONE(IREE::Input::TensorBitCastOp, IREE::Flow::TensorBitCastOp);
  ONE_TO_ONE(IREE::Input::TensorSliceOp, IREE::Flow::TensorSliceOp);
  ONE_TO_ONE(IREE::Input::TensorSplatOp, IREE::Flow::TensorSplatOp);
  ONE_TO_ONE(IREE::Input::TensorStoreOp, IREE::Flow::TensorStoreOp);
  ONE_TO_ONE(IREE::Input::TensorUpdateOp, IREE::Flow::TensorUpdateOp);
  ONE_TO_ONE(IREE::Input::TensorTraceOp, IREE::Flow::TensorTraceOp);
  ONE_TO_ONE(IREE::Input::DispatchOp, IREE::Flow::DispatchOp);
  ONE_TO_ONE(IREE::Input::GlobalAddressOp, IREE::Util::GlobalAddressOp);
  ONE_TO_ONE(IREE::Input::GlobalLoadOp, IREE::Util::GlobalLoadOp);
  ONE_TO_ONE(IREE::Input::GlobalLoadIndirectOp,
             IREE::Util::GlobalLoadIndirectOp);
  ONE_TO_ONE(IREE::Input::GlobalStoreOp, IREE::Util::GlobalStoreOp);
  ONE_TO_ONE(IREE::Input::GlobalStoreIndirectOp,
             IREE::Util::GlobalStoreIndirectOp);
  ONE_TO_ONE(IREE::Input::OptimizationBarrierOp,
             IREE::Util::OptimizationBarrierOp);
  ONE_TO_ONE(IREE::Input::ExecutableSourceEndOp,
             IREE::HAL::ExecutableSourceEndOp);

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace mlir::iree_compiler::InputConversion
