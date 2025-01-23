// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements IREE-specific logic for lowering StableHLO dialect to
// IREE dialects: Linalg, Arith, Math, Tensor, Util, ML Program, etc.

#include "compiler/plugins/input/StableHLO/Conversion/LegalizeToLinalgUtils.h"
#include "compiler/plugins/input/StableHLO/Conversion/PassDetail.h"
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Rewriters.h"
#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/ConversionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/conversions/linalg/transforms/TypeConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOIREEINPUTDIALECTS
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

// Returns true if all attributes in the given dictionary are valid for IREE
// input dialects.
static bool isValidFuncAttr(DictionaryAttr attrs) {
  // TODO: switch to using a dialect-based exclusion list or some other way that
  // is not a big string table.
  for (auto attr : attrs) {
    if (attr.getName() == "tf.aliasing_output")
      return false;
  }
  return true;
}

// Adds iree.abi.encoding attributes for arguments and results when they have
// had their type changed during conversion.
static void setFuncEncodings(func::FuncOp funcOp, FunctionType oldFuncType,
                             FunctionType newFuncType) {
  auto encodingName = StringAttr::get(funcOp.getContext(), "iree.abi.encoding");
  for (auto [i, oldType, newType] :
       llvm::enumerate(oldFuncType.getInputs(), newFuncType.getInputs())) {
    if (oldType != newType)
      funcOp.setArgAttr(i, encodingName, TypeAttr::get(oldType));
  }
  for (auto [i, oldType, newType] :
       llvm::enumerate(oldFuncType.getResults(), newFuncType.getResults())) {
    if (oldType != newType)
      funcOp.setResultAttr(i, encodingName, TypeAttr::get(oldType));
  }
}

// Rewrites attributes on the function from ones coming from HLO-based frontends
// to the IREE supported versions.
static void rewriteFuncAttrs(func::FuncOp funcOp) {
  auto *context = funcOp.getContext();
  auto indexType = IndexType::get(context);
  auto abiOutputName = StringAttr::get(context, "iree.abi.output");
  auto aliasingOutputName = StringAttr::get(context, "tf.aliasing_output");
  auto rewriteAttrs = [&](DictionaryAttr &allAttrs) {
    SmallVector<NamedAttribute> newAttrs;
    newAttrs.reserve(allAttrs.size());
    for (auto attr : allAttrs) {
      if (attr.getName() == aliasingOutputName) {
        newAttrs.push_back({
            abiOutputName,
            IntegerAttr::get(indexType,
                             llvm::cast<IntegerAttr>(attr.getValue()).getInt()),
        });
      } else {
        newAttrs.push_back(attr);
      }
    }
    allAttrs = DictionaryAttr::get(context, newAttrs);
  };
  SmallVector<DictionaryAttr> argAttrs;
  funcOp.getAllArgAttrs(argAttrs);
  llvm::for_each(argAttrs, rewriteAttrs);
  funcOp.setAllArgAttrs(argAttrs);
  SmallVector<DictionaryAttr> resultAttrs;
  funcOp.getAllResultAttrs(resultAttrs);
  llvm::for_each(resultAttrs, rewriteAttrs);
  funcOp.setAllResultAttrs(resultAttrs);
}

// We need to convert func ops in order to convert types.
struct BuiltinFuncOpPattern final : OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType srcFuncType = srcOp.getFunctionType();
    TypeConverter::SignatureConversion signatureConversion(
        srcOp.getNumArguments());

    // Convert function arguments.
    for (auto [idx, inputTy] : llvm::enumerate(srcFuncType.getInputs())) {
      if (failed(getTypeConverter()->convertSignatureArg(
              idx, inputTy, signatureConversion))) {
        return rewriter.notifyMatchFailure(srcOp, "argument failed to convert");
      }
    }

    // Convert function results.
    SmallVector<Type> convertedResultTypes;
    if (failed(getTypeConverter()->convertTypes(srcFuncType.getResults(),
                                                convertedResultTypes))) {
      return rewriter.notifyMatchFailure(srcOp, "results failed to convert");
    }

    // Create new function with converted argument and result types.
    auto oldFuncType = srcOp.getFunctionType();
    auto newFuncType = mlir::FunctionType::get(
        srcOp.getContext(), signatureConversion.getConvertedTypes(),
        convertedResultTypes);

    // Update the function in place.
    rewriter.startOpModification(srcOp);
    srcOp.setType(newFuncType);
    rewriteFuncAttrs(srcOp);
    setFuncEncodings(srcOp, oldFuncType, newFuncType);

    // Tell the rewriter to convert the region signature.
    const TypeConverter &typeConverter = *getTypeConverter();
    if (failed(rewriter.convertRegionTypes(&srcOp.getBody(), typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    rewriter.finalizeOpModification(srcOp);
    return success();
  }
};

struct TensorEmptyPattern final : OpConversionPattern<tensor::EmptyOp> {
  using OpConversionPattern<tensor::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto oldType = cast<ShapedType>(op.getType());
    auto newType = getTypeConverter()->convertType(oldType);
    if (newType == oldType)
      return failure();

    if (!newType)
      return rewriter.notifyMatchFailure(op, "result type conversion failed");

    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        op, oldType.getShape(),
        getTypeConverter()->convertType(oldType.getElementType()),
        op.getDynamicSizes());
    return success();
  }
};

struct GlobalOpPattern final : OpConversionPattern<ml_program::GlobalOp> {
  using OpConversionPattern<ml_program::GlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ml_program::GlobalOp globalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type oldType = globalOp.getType();
    Type newType = getTypeConverter()->convertType(oldType);
    if (newType == oldType)
      return failure();
    if (!newType) {
      return rewriter.notifyMatchFailure(globalOp,
                                         "result type conversion failed");
    }
    rewriter.modifyOpInPlace(globalOp, [&]() {
      globalOp.setType(newType);
      if (Attribute oldValue = globalOp.getValueAttr()) {
        globalOp.setValueAttr(
            convertAttribute(globalOp.getLoc(), oldValue, *getTypeConverter()));
      }
    });
    return success();
  }
};

struct OptimizationBarrierOpConversion final
    : OpConversionPattern<mlir::stablehlo::OptimizationBarrierOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::OptimizationBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> outputs;
    for (Value operand : adaptor.getOperands()) {
      outputs.push_back(
          rewriter
              .create<IREE::Util::OptimizationBarrierOp>(op.getLoc(), operand)
              .getResult(0));
    }
    rewriter.replaceOp(op, outputs);
    return success();
  }
};

template <typename T>
struct GenericTypeConvert final : ConversionPattern {
  GenericTypeConvert(StringRef rootName, TypeConverter &converter,
                     MLIRContext *context, PatternBenefit benefit = 0)
      : ConversionPattern(converter, rootName, benefit, context) {}

  GenericTypeConvert(TypeConverter &converter, MLIRContext *context,
                     PatternBenefit benefit = 0)
      : ConversionPattern(converter, T::getOperationName(), benefit, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<NamedAttribute> newAttr;
    llvm::append_range(newAttr, op->getAttrs());

    llvm::SmallVector<Type> newResults;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResults))) {
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }

    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttr, op->getSuccessors());
    for (Region &r : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      if (failed(getTypeConverter()->convertSignatureArgs(
              newRegion->getArgumentTypes(), result))) {
        return rewriter.notifyMatchFailure(op,
                                           "argument type conversion failed");
      }
      rewriter.applySignatureConversion(&newRegion->front(), result);
    }
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

Value scalarToTensor(OpBuilder &builder, Type /*type*/, ValueRange inputs,
                     Location loc) {
  assert(inputs.size() == 1);
  if (isa<ShapedType>(inputs.front().getType())) {
    return Value();
  }
  return builder
      .create<tensor::FromElementsOp>(
          loc, RankedTensorType::get({}, inputs.front().getType()),
          inputs.front())
      .getResult();
}

// Strips attributes from common StableHLO frontends (JAX, TF, etc) that are not
// used after conversion into the IREE input dialects. Leaving these attributes
// is confusing as they can become inconsistent during subsequent conversions or
// leak frontend details lower into the pipeline than should be allowed.
static void stripFrontendAttrs(mlir::ModuleOp moduleOp) {
  auto isAttrFiltered = [](NamedAttribute attr) {
    auto fullName = attr.getName().getValue();
    return fullName.starts_with("mhlo.") || fullName.starts_with("jax.") ||
           fullName.starts_with("tf.");
  };
  auto filterOpAttrs = [&](Operation *op) {
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : op->getDialectAttrs()) {
      if (!isAttrFiltered(attr))
        newAttrs.push_back(attr);
    }
    op->setDialectAttrs(newAttrs);
  };
  auto filterAttrDicts = [&](ArrayAttr allOldAttrs,
                             SmallVectorImpl<DictionaryAttr> &newAttrs) {
    if (!allOldAttrs)
      return false;
    for (auto oldAttrs : allOldAttrs.getAsRange<DictionaryAttr>()) {
      SmallVector<NamedAttribute> preservedAttrs;
      preservedAttrs.reserve(oldAttrs.size());
      for (auto attr : oldAttrs) {
        if (!isAttrFiltered(attr))
          preservedAttrs.push_back(attr);
      }
      newAttrs.push_back(
          DictionaryAttr::get(allOldAttrs.getContext(), preservedAttrs));
    }
    return true;
  };
  filterOpAttrs(moduleOp);
  for (auto callableOp : moduleOp.getOps<mlir::CallableOpInterface>()) {
    filterOpAttrs(callableOp);
    if (auto funcOp = dyn_cast<func::FuncOp>(callableOp.getOperation())) {
      SmallVector<DictionaryAttr> newArgAttrs;
      if (filterAttrDicts(funcOp.getAllArgAttrs(), newArgAttrs)) {
        funcOp.setAllArgAttrs(newArgAttrs);
      }
      SmallVector<DictionaryAttr> newResultAttrs;
      if (filterAttrDicts(funcOp.getAllResultAttrs(), newResultAttrs)) {
        funcOp.setAllResultAttrs(newResultAttrs);
      }
    }
  }
}

struct ConvertStableHloToIreeInputDialects final
    : impl::ConvertStableHloToIreeInputDialectsBase<
          ConvertStableHloToIreeInputDialects> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        IREE::Flow::FlowDialect, IREE::Util::UtilDialect, linalg::LinalgDialect,
        arith::ArithDialect, tensor::TensorDialect, shape::ShapeDialect,
        math::MathDialect, memref::MemRefDialect, complex::ComplexDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    std::unique_ptr<TypeConverter> typeConverter =
        std::make_unique<::mlir::stablehlo::LinalgTypeConverter>();
    typeConverter->addArgumentMaterialization(scalarToTensor);
    typeConverter->addSourceMaterialization(scalarToTensor);
    typeConverter->addTargetMaterialization(scalarToTensor);

    // Run stablehlo canonicalization patterns with a high benefit to avoid some
    // expensive expansions.
    populateCanonicalizationPatterns(context, &patterns, /*benefit=*/1024);

    populateStableHloCollectivesConversionPatterns(context, *typeConverter,
                                                   &patterns);

    // TODO(#12678): Handle remaining complex ops.

    // TODO(*): expose patterns that do this much better from
    // iree/compiler/Dialect/Util/Transforms/ConvertPrimitiveType.cpp

    // Structural patterns (functions, cfg, terminators).
    patterns.add<BuiltinFuncOpPattern>(*typeConverter, context);
    patterns.add<GlobalOpPattern, TensorEmptyPattern>(*typeConverter, context);
    patterns.add<OptimizationBarrierOpConversion>(*typeConverter, context);

    patterns.add<
        GenericTypeConvert<cf::CondBranchOp>, GenericTypeConvert<cf::BranchOp>,
        GenericTypeConvert<func::ReturnOp>, GenericTypeConvert<func::ReturnOp>,
        GenericTypeConvert<func::CallOp>,
        GenericTypeConvert<ml_program::GlobalLoadOp>,
        GenericTypeConvert<ml_program::GlobalLoadConstOp>,
        GenericTypeConvert<ml_program::GlobalStoreOp>,
        GenericTypeConvert<scf::ForOp>, GenericTypeConvert<scf::IfOp>,
        GenericTypeConvert<scf::YieldOp>, GenericTypeConvert<scf::ConditionOp>,
        GenericTypeConvert<scf::WhileOp>,
        GenericTypeConvert<tensor::FromElementsOp>,
        GenericTypeConvert<tensor::CollapseShapeOp>,
        GenericTypeConvert<tensor::ExpandShapeOp>,
        GenericTypeConvert<arith::IndexCastUIOp>,
        GenericTypeConvert<arith::SelectOp>>(*typeConverter, context);

    ConversionTarget target(*context);
    auto isIllegalType = [&](Type t) { return !typeConverter->isLegal(t); };
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

    target.addIllegalDialect<mlir::chlo::ChloDialect>();
    target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();

    // Functions must have legal types.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp funcOp) {
      if (auto attrs = funcOp.getAllArgAttrs()) {
        if (!llvm::all_of(attrs.getAsRange<DictionaryAttr>(),
                          isValidFuncAttr)) {
          return false;
        }
      }
      if (auto attrs = funcOp.getAllResultAttrs()) {
        if (!llvm::all_of(attrs.getAsRange<DictionaryAttr>(),
                          isValidFuncAttr)) {
          return false;
        }
      }
      for (Type type : funcOp.getFunctionType().getInputs()) {
        if (isIllegalType(type))
          return false;
      }
      for (Type type : funcOp.getFunctionType().getResults()) {
        if (isIllegalType(type))
          return false;
      }
      for (Block &block : funcOp.getFunctionBody()) {
        for (Type type : block.getArgumentTypes()) {
          if (isIllegalType(type))
            return false;
        }
      }
      return true;
    });
    target.addDynamicallyLegalOp<ml_program::GlobalOp>(
        [&](ml_program::GlobalOp op) {
          return typeConverter->isLegal(op.getType());
        });

    target.addDynamicallyLegalOp<tensor::EmptyOp>([&](tensor::EmptyOp op) {
      return typeConverter->isLegal(op.getType());
    });

    // Let the rest fall through.
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<IREE::LinalgExt::IREELinalgExtDialect>();
    target.markUnknownOpDynamicallyLegal(isLegallyTypedOp);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }

    {
      // Apply the patterns to remove unused operands and results.
      RewritePatternSet removeUnusedOperandsResultsPatterns(context);
      linalg::populateEraseUnusedOperandsAndResultsPatterns(
          removeUnusedOperandsResultsPatterns);
      if (failed(applyPatternsGreedily(
              getOperation(),
              std::move(removeUnusedOperandsResultsPatterns)))) {
        return signalPassFailure();
      }
    }

    // Drop module/function attributes now that they are no longer required.
    stripFrontendAttrs(getOperation());
  }
};

} // namespace

} // namespace mlir::iree_compiler::stablehlo
