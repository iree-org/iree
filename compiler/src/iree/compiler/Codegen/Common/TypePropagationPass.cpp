// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- TypePropagationPass.cpp -------------------------------------------===//
//
// The dispatch regions passed to the backends legalizes the bitwidth of
// element types used for the input/output buffers. To avoid illegal load/stores
// within the dispatch, the type needs to be propagated to avoid having tensors
// of illegal bitwidths.
//
// This pass uses the dialect conversion framework to propagate the types,
// - All ops are marked dynamically illegal if their operands/result uses
//   unsupported element type.
// - A generic pattern is added to legalize all such ops that triggers on every
//   operation.
//   - For operations with illegal result types, it creates a new
//     operations with legalized return types.
//   - This pattern uses the generic operation creation methods to be
//     op-agnostic.
// - For ops that need specifc handling, patterns are added with higher benefit,
//   so that they trigger first during legalization.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/ElementPackingUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TYPEPROPAGATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

/// Insert instructions to convert from one element type to another.
static Value convertElementType(OpBuilder &b, Location loc, Type targetType,
                                Value source) {
  Type sourceType = source.getType();
  if (sourceType == targetType)
    return source;
  if (llvm::isa<IntegerType>(sourceType) &&
      llvm::isa<IntegerType>(targetType)) {
    unsigned sourceBitWidth = sourceType.getIntOrFloatBitWidth();
    unsigned destBitWidth = targetType.getIntOrFloatBitWidth();
    if (sourceBitWidth > destBitWidth) {
      return b.create<arith::TruncIOp>(loc, targetType, source);
    } else {
      return b.create<arith::ExtUIOp>(loc, targetType, source);
    }
  }
  return nullptr;
}

/// Legalizes the given type. If the type is already legal, returns
/// std::nullopt.
static std::optional<Type> getLegalizedType(Type t) {
  if (auto shapedType = llvm::dyn_cast<RankedTensorType>(t)) {
    Type elementType = shapedType.getElementType();
    std::optional<Type> legalizedElementType =
        legalizeStorageElementType(elementType);
    if (!legalizedElementType)
      return std::nullopt;
    return RankedTensorType::get(shapedType.getShape(),
                                 legalizedElementType.value(),
                                 shapedType.getEncoding());
  }
  return std::nullopt;
}

namespace {

/// Materialize
Value materializeAsConvertElementType(OpBuilder &builder, Type type,
                                      ValueRange inputs, Location loc) {
  assert(inputs.size() == 1 && "expected exactly one input");
  return convertElementType(builder, loc, type, inputs[0]);
}

/// Type converter to use for type propagation.
struct TypePropagationTypeConverter : public TypeConverter {
  TypePropagationTypeConverter() {
    addConversion([](Type t) {
      auto convertedType = getLegalizedType(t);
      if (!convertedType)
        return t;
      return convertedType.value();
    });
  }
};

/// Base class for patterns that handle individual operations.
template <typename T>
struct TypePropagationPattern : public OpConversionPattern<T> {
  TypePropagationPattern(TypePropagationTypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<T>(typeConverter, context, 100) {}
};

/// Type conversion for arith.constant operands.
struct ConstantOpTypeConversion
    : public TypePropagationPattern<arith::ConstantOp> {
  using TypePropagationPattern<arith::ConstantOp>::TypePropagationPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp constantOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto attr = llvm::cast<ElementsAttr>(constantOp.getValue());
    auto attrType = llvm::dyn_cast<ShapedType>(attr.getType());
    if (!attrType) {
      return rewriter.notifyMatchFailure(
          constantOp, "expected attribute type to be shaped type");
    }
    std::optional<Type> legalizedElementType =
        legalizeStorageElementType(attrType.getElementType());
    if (!legalizedElementType) {
      return rewriter.notifyMatchFailure(constantOp,
                                         "cannot legalize elementType");
    }
    if (!legalizedElementType->isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          constantOp, "expected legalized type to be integer or float type");
    }
    SmallVector<APInt> legalizedValues;
    unsigned numElements = attr.isSplat() ? 1 : attr.getNumElements();
    legalizedValues.reserve(numElements);
    unsigned bitWidth = legalizedElementType->getIntOrFloatBitWidth();
    for (auto value : attr.getValues<APInt>()) {
      legalizedValues.emplace_back(bitWidth, value.getZExtValue());
    }
    auto newAttrType = RankedTensorType::get(attrType.getShape(),
                                             legalizedElementType.value());
    auto newAttr = DenseElementsAttr::get(newAttrType, legalizedValues);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, newAttrType,
                                                   newAttr);
    return success();
  }
};

/// Type conversion for Linalg named op. Interface patterns are not used
/// here cause the region of the operation cannot be cloned. Instead create
/// a new operation with the operands of the correct type.
template <typename OpTy>
struct NamedOpTypePropagation : public TypePropagationPattern<OpTy> {
  using TypePropagationPattern<OpTy>::TypePropagationPattern;

  LogicalResult
  matchAndRewrite(OpTy namedOp, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type> resultTypes;
    resultTypes.reserve(namedOp->getNumResults());
    for (auto resultType : namedOp->getResultTypes()) {
      Type legalizedType = this->getTypeConverter()->convertType(resultType);
      resultTypes.push_back(legalizedType);
    }
    rewriter.replaceOpWithNewOp<OpTy>(namedOp, resultTypes, adaptor.getInputs(),
                                      adaptor.getOutputs(),
                                      linalg::getPrunedAttributeList(namedOp));
    return success();
  }
};

/// Propagates the type for `linalg.generic` operation.
/// - Convert operands whose type has changed.
/// - Convert corresponding basic block argument type and introduce element
/// conversion ops to get back the original type.
/// - Convert the result type if the `outs` operand has changed.
struct GenericOpTypePropagation
    : public TypePropagationPattern<linalg::GenericOp> {
  using TypePropagationPattern<linalg::GenericOp>::TypePropagationPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp genericOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallSetVector<unsigned, 8> modifiedOperandIndex;
    SmallVector<Type> resultTypes;

    // 1. Check if any of the operands needs to be legalized.
    for (auto [index, operand] : llvm::enumerate(genericOp->getOpOperands())) {
      Type operandType = operand.get().getType();
      Type legalizedType = this->getTypeConverter()->convertType(operandType);
      if (operandType != legalizedType) {
        modifiedOperandIndex.insert(index);
      }
      // If the operand is an `outs` tensor, its type needs to be changed.
      if (genericOp.isDpsInit(&operand)) {
        resultTypes.push_back(legalizedType);
      }
    }

    // 2. If there are no operands modified, just return failure.
    assert(!modifiedOperandIndex.empty() &&
           "unexpected all types legal within conversion pattern");

    // 3. Create a clone of the operation without cloning its regions.
    auto linalgOp = cast<linalg::LinalgOp>(genericOp.getOperation());
    auto modifiedOp = cast<linalg::LinalgOp>(mlir::cloneWithoutRegions(
        rewriter, linalgOp, resultTypes, adaptor.getOperands()));

    if (genericOp->getNumRegions() != 1) {
      return genericOp.emitOpError("unhanled linalg op with numRegions != 1");
    }

    // 4. Inline the region from the original operation into the new
    // operation.
    rewriter.inlineRegionBefore(genericOp->getRegions().front(),
                                modifiedOp->getRegions().front(),
                                modifiedOp->getRegions().front().begin());
    Region &modifiedOpRegion = modifiedOp->getRegions().front();

    // 5. Convert the signature of the region to use the corresponding element
    // type.
    TypeConverter::SignatureConversion signatureConverter(
        modifiedOpRegion.getNumArguments());
    for (auto [index, arg] : llvm::enumerate(modifiedOpRegion.getArguments())) {
      Type argType = arg.getType();
      if (!modifiedOperandIndex.count(index)) {
        signatureConverter.addInputs(index, argType);
        continue;
      }
      std::optional<Type> legalizedArgType =
          legalizeStorageElementType(argType);
      if (!legalizedArgType) {
        return genericOp.emitOpError("failed to get legalized type for arg ")
               << index;
      }
      signatureConverter.addInputs(index, legalizedArgType.value());
    }
    rewriter.applySignatureConversion(&modifiedOpRegion.front(),
                                      signatureConverter, getTypeConverter());

    // 6. Introduce scalar conversion operations to convert back to the
    // original scalar type.
    {
      OpBuilder::InsertionGuard g(rewriter);
      Block *entryBlock = modifiedOp.getBlock();

      // 6b. If any of the operands modified were outputs, the yield values
      // need to be modified as well.
      Operation *yieldOp = entryBlock->getTerminator();
      rewriter.setInsertionPoint(yieldOp);
      bool modifyYield = false;
      SmallVector<Value> yieldOperands(yieldOp->operand_begin(),
                                       yieldOp->operand_end());
      for (auto modifiedOperandIndex : modifiedOperandIndex) {
        OpOperand *modifiedOpOperand =
            &modifiedOp->getOpOperand(modifiedOperandIndex);
        if (modifiedOp.isDpsInit(modifiedOpOperand)) {
          modifyYield = true;
          OpOperand *yieldOperand =
              modifiedOp.getMatchingYieldValue(modifiedOpOperand);
          std::optional<Type> legalizedType =
              legalizeStorageElementType(yieldOperand->get().getType());
          if (!legalizedType) {
            return genericOp.emitOpError(
                "failed to get legalized type for yield value");
          }
          yieldOperands[yieldOperand->getOperandNumber()] =
              convertElementType(rewriter, yieldOp->getLoc(),
                                 legalizedType.value(), yieldOperand->get());
        }
      }
      if (modifyYield) {
        rewriter.replaceOpWithNewOp<linalg::YieldOp>(yieldOp, yieldOperands);
      }
    }

    rewriter.replaceOp(genericOp, modifiedOp->getResults());
    return success();
  }
};

/// Legalizes `linalg.fill` operation.
struct LinalgFillTypePropagation
    : public TypePropagationPattern<linalg::FillOp> {
  using TypePropagationPattern<linalg::FillOp>::TypePropagationPattern;

  LogicalResult
  matchAndRewrite(linalg::FillOp fillOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value value = adaptor.getInputs().front();
    std::optional<Type> legalizedElementType =
        legalizeStorageElementType(value.getType());
    if (!legalizedElementType) {
      return fillOp.emitOpError("failed to get legalized type for value");
    }
    Value legalizedValue = convertElementType(
        rewriter, fillOp->getLoc(), legalizedElementType.value(), value);
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        fillOp, ValueRange{legalizedValue}, ValueRange{adaptor.getOutputs()});
    return success();
  }
};

/// Pattern to legalize `tensor.extract` operations.
struct TensorExtractTypePropagation
    : public TypePropagationPattern<tensor::ExtractOp> {
  using TypePropagationPattern<tensor::ExtractOp>::TypePropagationPattern;

  LogicalResult
  matchAndRewrite(tensor::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = extractOp.getLoc();
    Value newExtract = rewriter.create<tensor::ExtractOp>(
        loc, adaptor.getTensor(), adaptor.getIndices());
    Value replacement = convertElementType(
        rewriter, loc, extractOp.getResult().getType(), newExtract);
    rewriter.replaceOp(extractOp, replacement);
    return success();
  }
};

/// Pattern to legalize `iree_linalg_ext.scatter` operations.
struct IREELinalgExtScatterTypePropagation
    : TypePropagationPattern<IREE::LinalgExt::ScatterOp> {
  using TypePropagationPattern<
      IREE::LinalgExt::ScatterOp>::TypePropagationPattern;
  LogicalResult
  matchAndRewrite(IREE::LinalgExt::ScatterOp scatterOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto opOperands = scatterOp->getOpOperands();
    Type inputType = opOperands[0].get().getType();
    Type legalizedInputType = this->getTypeConverter()->convertType(inputType);

    if (inputType == legalizedInputType) {
      return scatterOp.emitOpError(
          "unexpected all types legal within conversion pattern");
    }

    Type resultType = opOperands[2].get().getType();
    Type legalizedResultType =
        this->getTypeConverter()->convertType(resultType);

    // Create a clone of the operation without cloning its regions.
    auto modifiedOp =
        cast<IREE::LinalgExt::ScatterOp>(mlir::cloneWithoutRegions(
            rewriter, scatterOp, {legalizedResultType}, adaptor.getOperands()));

    // Inline the region from the original operation into the new operation.
    rewriter.inlineRegionBefore(scatterOp->getRegions().front(),
                                modifiedOp->getRegions().front(),
                                modifiedOp->getRegions().front().begin());
    Region &modifiedOpRegion = modifiedOp->getRegions().front();

    // Convert the signature of the region to use the corresponding element
    // type.
    TypeConverter::SignatureConversion signatureConverter(
        modifiedOpRegion.getNumArguments());
    Type argType = modifiedOpRegion.getArguments()[0].getType();
    std::optional<Type> legalizedArgType = legalizeStorageElementType(argType);
    if (!legalizedArgType) {
      return scatterOp.emitOpError("failed to get legalized type for argument");
    }
    signatureConverter.addInputs(0, legalizedArgType.value());
    signatureConverter.addInputs(1, legalizedArgType.value());
    rewriter.applySignatureConversion(&modifiedOpRegion.front(),
                                      signatureConverter, getTypeConverter());

    {
      // Introduce scalar conversion operations to convert back to the original
      // scalar type.
      OpBuilder::InsertionGuard g(rewriter);
      Block *entryBlock = &modifiedOp->getRegion(0).getBlocks().front();

      // If the output is of an illegal type, the yield value needs to be
      // modified
      auto yieldOp = entryBlock->getTerminator();

      rewriter.setInsertionPoint(yieldOp);
      OpOperand *modifiedOpOperand = &yieldOp->getOpOperand(0);

      auto yieldOperand = convertElementType(rewriter, yieldOp->getLoc(),
                                             legalizedArgType.value(),
                                             modifiedOpOperand->get());

      rewriter.replaceOpWithNewOp<IREE::LinalgExt::YieldOp>(yieldOp,
                                                            yieldOperand);
    }
    rewriter.replaceOp(scatterOp, modifiedOp->getResults());
    return success();
  }
};

/// Pattern to legalize `iree_linalg_ext.sort` operations.
struct IREELinalgExtSortTypePropagation
    : TypePropagationPattern<IREE::LinalgExt::SortOp> {
  using TypePropagationPattern<IREE::LinalgExt::SortOp>::TypePropagationPattern;
  LogicalResult
  matchAndRewrite(IREE::LinalgExt::SortOp sortOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type> legalizedResultTypes;
    for (Type resultType : sortOp->getResultTypes()) {
      Type legalizedType = this->getTypeConverter()->convertType(resultType);
      legalizedResultTypes.push_back(legalizedType);
    }

    // Create a clone of the operation without cloning its regions.
    auto modifiedOp = cast<IREE::LinalgExt::SortOp>(mlir::cloneWithoutRegions(
        rewriter, sortOp, {legalizedResultTypes}, adaptor.getOperands()));

    // Inline the region from the original operation into the new operation.
    rewriter.inlineRegionBefore(sortOp->getRegions().front(),
                                modifiedOp->getRegions().front(),
                                modifiedOp->getRegions().front().begin());
    Region &modifiedOpRegion = modifiedOp->getRegions().front();

    // Convert the signature of the region to use the corresponding element
    // type.
    TypeConverter::SignatureConversion signatureConverter(
        modifiedOpRegion.getNumArguments());
    for (auto [index, arg] : llvm::enumerate(modifiedOpRegion.getArguments())) {
      std::optional<Type> legalizedArgType =
          legalizeStorageElementType(arg.getType());
      if (!legalizedArgType) {
        return sortOp.emitOpError("failed to get legalized type for argument");
      }
      signatureConverter.addInputs(index, legalizedArgType.value());
    }
    rewriter.applySignatureConversion(&modifiedOpRegion.front(),
                                      signatureConverter, getTypeConverter());
    rewriter.replaceOp(sortOp, modifiedOp->getResults());
    return success();
  }
};

/// Simple rewrite pattern that just forwards the source as the result if the
/// result type is not legal (but source type is)
template <typename OpTy>
struct ForwardSourceType : public TypePropagationPattern<OpTy> {
  using TypePropagationPattern<OpTy>::TypePropagationPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op->getNumResults() != 1 || adaptor.getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "unhandled op with multiple operands/results");
    }
    Value input = adaptor.getOperands()[0];
    rewriter.replaceOp(op, input);
    return success();
  }
};

/// Rewrite pattern to replace the element type (if it is not legal) with the
/// legal element type.
struct LegalizeResultElementType : public ConversionPattern {
  LegalizeResultElementType(TypePropagationTypeConverter &typeConverter,
                            MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> convertedOperands,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    SmallVector<Type> resultTypes;
    for (Type resultType : op->getResultTypes()) {
      Type legalizedType = this->typeConverter->convertType(resultType);
      resultTypes.push_back(legalizedType);
    }
    OperationState state(loc, op->getName(), convertedOperands, resultTypes,
                         op->getAttrs());
    for (unsigned i = 0, e = op->getNumRegions(); i != e; ++i) {
      state.addRegion();
    }
    for (auto successor : op->getSuccessors()) {
      state.addSuccessors(successor);
    }
    Operation *newOp = rewriter.create(state);

    // Move all the regions from the old op to the new op and legalize its
    // signature.
    for (const auto &[index, region] : llvm::enumerate(op->getRegions())) {
      Region &newOpRegion = newOp->getRegion(index);
      rewriter.inlineRegionBefore(region, newOpRegion, newOpRegion.begin());
      TypeConverter::SignatureConversion signatureConverter(
          newOpRegion.getNumArguments());
      bool doSignatureConversion = false;
      for (const auto &[argIndex, arg] :
           llvm::enumerate(newOpRegion.getArguments())) {
        Type argType = arg.getType();
        Type legalizedType = this->typeConverter->convertType(argType);
        signatureConverter.addInputs(argIndex, legalizedType);
        doSignatureConversion |= argType != legalizedType;
      }
      if (doSignatureConversion) {
        rewriter.applySignatureConversion(&newOpRegion.front(),
                                          signatureConverter);
      }
    }
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// Rewrite pattern for converting the signature of all basic blocks in the
// top-level operation.
template <typename OpTy>
struct LegalizeBasicBlocks : public TypePropagationPattern<OpTy> {
  using TypePropagationPattern<OpTy>::TypePropagationPattern;

  LogicalResult
  matchAndRewrite(OpTy funcOp, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    FailureOr<Block *> newEntry =
        rewriter.convertRegionTypes(&funcOp.getBody(), *this->typeConverter);
    if (failed(newEntry)) {
      return rewriter.notifyMatchFailure(funcOp,
                                         "failed to convert region types");
    }
    rewriter.modifyOpInPlace(funcOp, []() {});
    return success();
  }
};

struct TypePropagationPass final
    : impl::TypePropagationPassBase<TypePropagationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    TypePropagationTypeConverter typeConverter;
    typeConverter.addSourceMaterialization(materializeAsConvertElementType);
    typeConverter.addTargetMaterialization(materializeAsConvertElementType);

    patterns.insert<
        ConstantOpTypeConversion, ForwardSourceType<arith::ExtUIOp>,
        ForwardSourceType<arith::TruncIOp>, GenericOpTypePropagation,
        LinalgFillTypePropagation, LegalizeBasicBlocks<func::FuncOp>,
        LegalizeResultElementType,
        NamedOpTypePropagation<linalg::BatchMatmulOp>,
        NamedOpTypePropagation<linalg::MatmulOp>,
        NamedOpTypePropagation<linalg::MatvecOp>,
        NamedOpTypePropagation<linalg::DotOp>, TensorExtractTypePropagation,
        IREELinalgExtScatterTypePropagation, IREELinalgExtSortTypePropagation>(
        typeConverter, context);

    ConversionTarget target(*context);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp funcOp) {
      Region &body = funcOp.getBody();
      for (Block &block : body) {
        for (auto arg : block.getArguments()) {
          Type argType = arg.getType();
          Type convertedArgType = typeConverter.convertType(argType);
          if (convertedArgType != argType) {
            return false;
          }
        }
      }
      return true;
    });
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        Type operandType = operand.getType();
        if (operandType != typeConverter.convertType(operandType)) {
          return false;
        }
      }
      for (auto result : op->getResults()) {
        Type resultType = result.getType();
        if (resultType != typeConverter.convertType(resultType)) {
          return false;
        }
      }
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
