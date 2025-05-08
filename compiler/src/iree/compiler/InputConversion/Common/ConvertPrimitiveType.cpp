// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/Utils/ConversionUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::InputConversion {

#define GEN_PASS_DEF_DEMOTEF32TOF16PASS
#define GEN_PASS_DEF_DEMOTEF64TOF32PASS
#define GEN_PASS_DEF_DEMOTEI64TOI32PASS
#define GEN_PASS_DEF_PROMOTEBF16TOF32PASS
#define GEN_PASS_DEF_PROMOTEF16TOF32PASS
#include "iree/compiler/InputConversion/Common/Passes.h.inc"

namespace {

Value convertRankedFloat(OpBuilder &builder, Type type, ValueRange inputs,
                         Location loc) {
  Type eTy = getElementTypeOrSelf(type);
  Type inputETy = getElementTypeOrSelf(inputs[0].getType());
  if (!llvm::isa<FloatType>(getElementTypeOrSelf(type)))
    return nullptr;

  if (inputETy.getIntOrFloatBitWidth() > eTy.getIntOrFloatBitWidth()) {
    return builder.create<arith::TruncFOp>(loc, type, inputs[0]);
  }

  if (inputETy.getIntOrFloatBitWidth() < eTy.getIntOrFloatBitWidth()) {
    return builder.create<arith::ExtFOp>(loc, type, inputs[0]);
  }

  return nullptr;
};

Value convertRankedInteger(OpBuilder &builder, Type type, ValueRange inputs,
                           Location loc) {
  Type eTy = getElementTypeOrSelf(type);
  Type inputETy = getElementTypeOrSelf(inputs[0].getType());
  if (!llvm::isa<FloatType>(getElementTypeOrSelf(type)))
    return nullptr;
  bool isUnsigned = eTy.isUnsignedInteger();

  int64_t inBitwidth = inputETy.getIntOrFloatBitWidth();
  int64_t outBitwidth = eTy.getIntOrFloatBitWidth();

  if (inBitwidth > outBitwidth) {
    return builder.create<arith::TruncIOp>(loc, type, inputs[0]);
  }

  if (inBitwidth < outBitwidth && isUnsigned) {
    return builder.create<arith::ExtUIOp>(loc, type, inputs[0]);
  }

  if (inBitwidth < outBitwidth && !isUnsigned) {
    return builder.create<arith::ExtSIOp>(loc, type, inputs[0]);
  }

  return nullptr;
};

// Converts from |SourceType| to |TargetType|.
template <typename SourceType, typename TargetType>
struct PrimitiveTypeConverter : public TypeConverter {
  explicit PrimitiveTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([&](SourceType type) -> Type {
      if (!isSourceType(type))
        return type;
      return getTargetType(type);
    });
    addConversion([&](ComplexType type) {
      return ComplexType::get(convertType(type.getElementType()));
    });
    addConversion([&](RankedTensorType type) {
      return RankedTensorType::get(type.getShape(),
                                   convertType(type.getElementType()),
                                   type.getEncoding());
    });
    addConversion([&](VectorType type) {
      return VectorType::get(type.getShape(),
                             convertType(type.getElementType()));
    });
    addConversion([&](IREE::Util::PtrType ptrType) {
      return IREE::Util::PtrType::get(convertType(ptrType.getTargetType()));
    });
  }

  virtual ~PrimitiveTypeConverter() = default;

  // Returns true if |type| matches the expected source type.
  // Subclasses can override to restrict their conversion to specific subtypes.
  virtual bool isSourceType(SourceType type) { return true; }

  // Returns the newly converted type of |type|.
  // Subclasses can override to pass additional type parameters.
  virtual Type getTargetType(SourceType type) = 0;
};

template <typename SourceType, typename TargetType>
struct FloatTypeConverter
    : public PrimitiveTypeConverter<SourceType, TargetType> {
  explicit FloatTypeConverter() {
    this->addArgumentMaterialization(convertRankedFloat);
    this->addSourceMaterialization(convertRankedFloat);
    this->addTargetMaterialization(convertRankedFloat);
  }
};

template <typename SourceType, typename TargetType>
struct IntegerTypeConverter
    : public PrimitiveTypeConverter<SourceType, TargetType> {
  explicit IntegerTypeConverter() {
    this->addArgumentMaterialization(convertRankedInteger);
    this->addSourceMaterialization(convertRankedInteger);
    this->addTargetMaterialization(convertRankedInteger);
  }
};

// Tries to completely convert a generic Operation.
// This will process attributes, result types, and nested regions.
struct GenericTypeConversionPattern : public ConversionPattern {
  GenericTypeConversionPattern(MLIRContext *context,
                               TypeConverter &typeConverter)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 0, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert attributes only if this is a constant-like op.
    // This is because some ops use typed attributes for structural information
    // - like linalg ops using i64 for dimension indices - and if we converted
    // them all the ops would become invalid. This may still be too broad,
    // though, if some constant ops include attributes with both the type we
    // want to convert and structural information in the same type.
    llvm::SmallVector<NamedAttribute> newAttrs;
    if (op->hasTrait<OpTrait::ConstantLike>()) {
      for (auto attr : op->getAttrs()) {
        auto newAttr = convertAttribute(op->getLoc(), attr.getValue(),
                                        *getTypeConverter());
        newAttrs.push_back(NamedAttribute(attr.getName(), newAttr));
      }
    } else {
      newAttrs.append(op->getAttrs().begin(), op->getAttrs().end());
    }

    llvm::SmallVector<Type> newResults;
    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);

    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttrs, op->getSuccessors());

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

struct GlobalOpConversionPattern
    : public OpInterfaceConversionPattern<IREE::Util::GlobalOpInterface> {
  GlobalOpConversionPattern(MLIRContext *context, TypeConverter &typeConverter)
      : OpInterfaceConversionPattern(typeConverter, context) {}
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<NamedAttribute> newAttrs;
    for (auto attr : op->getAttrs()) {
      auto newAttr =
          convertAttribute(op->getLoc(), attr.getValue(), *getTypeConverter());
      newAttrs.push_back(NamedAttribute(attr.getName(), newAttr));
    }
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         {}, newAttrs, op->getSuccessors());
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

template <typename OpTy, typename TypeTy,
          typename OperandToResultWidthLegalityRelation>
struct ConvertTypeSensitiveArithCastOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    auto operandType =
        this->getTypeConverter()->convertType(op.getOperand().getType());

    auto resultEType = cast<TypeTy>(getElementTypeOrSelf(resultType));
    auto operandEType = cast<TypeTy>(getElementTypeOrSelf(operandType));
    // If post-conversion, the types would be equal, then the op becomes a
    // no-op. Note that the op does not itself allow such a configuration, so we
    // have to catch this before creating the new op.
    if (resultEType == operandEType) {
      rewriter.replaceOp(op, adaptor.getOperands()[0]);
      return success();
    }
    // If after conversion the op becomes invalid, but not same-type (which we
    // can fold above), then bail out.
    // TODO: In some cases, we can repair the situation here, but for integer
    // truncation, we don't know whether we should invert with signed or
    // unsigned extension.
    if (!OperandToResultWidthLegalityRelation()(operandEType.getWidth(),
                                                resultEType.getWidth())) {
      return rewriter.notifyMatchFailure(op, "invalid width combination");
    }
    rewriter.replaceOpWithNewOp<OpTy>(op, resultType, op.getOperand());
    return success();
  }
};

template <typename Base, typename Converter>
struct ConvertTypesPass : public Base {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &this->getContext();

    // Scan the module to detect external functions with types that would be
    // converted. This pass cannot be used with them.
    auto moduleOp = this->getOperation();
    for (auto funcOp : moduleOp.template getOps<mlir::FunctionOpInterface>()) {
      if (funcOp.isExternal() &&
          !typeConverter.isSignatureLegal(
              cast<FunctionType>(funcOp.getFunctionType()))) {
        funcOp.emitError()
            << "external functions with types that are being demoted are not "
               "allowed; do not use the pass or manually convert the function "
               "signature as required prior to running it";
        return this->signalPassFailure();
      }
    }

    RewritePatternSet patterns(context);
    patterns.insert<GenericTypeConversionPattern>(context, typeConverter);
    patterns.insert<GlobalOpConversionPattern>(context, typeConverter);
    patterns.insert<ConvertTypeSensitiveArithCastOp<arith::TruncFOp, FloatType,
                                                    std::greater<unsigned>>>(
        typeConverter, context);
    patterns.insert<ConvertTypeSensitiveArithCastOp<arith::ExtFOp, FloatType,
                                                    std::less<unsigned>>>(
        typeConverter, context);
    patterns.insert<ConvertTypeSensitiveArithCastOp<
        arith::TruncIOp, IntegerType, std::less<unsigned>>>(typeConverter,
                                                            context);
    patterns.insert<ConvertTypeSensitiveArithCastOp<arith::ExtUIOp, IntegerType,
                                                    std::less<unsigned>>>(
        typeConverter, context);
    patterns.insert<ConvertTypeSensitiveArithCastOp<arith::ExtSIOp, IntegerType,
                                                    std::less<unsigned>>>(
        typeConverter, context);
    ConversionTarget target(*context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<IREE::Util::InitializerOp>(
        patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<IREE::Util::FuncOp>(
        patterns, typeConverter);

    // Operations are legal if they don't contain any illegal type.
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto globalOp = dyn_cast<IREE::Util::GlobalOpInterface>(op)) {
        return typeConverter.isLegal(globalOp.getGlobalType());
      } else if (auto funcOp = dyn_cast<mlir::FunctionOpInterface>(op)) {
        for (Type type : funcOp.getArgumentTypes()) {
          if (!typeConverter.isLegal(type))
            return false;
        }
        for (Type type : funcOp.getResultTypes()) {
          if (!typeConverter.isLegal(type))
            return false;
        }
      }
      for (Type type : op->getResultTypes()) {
        if (!typeConverter.isLegal(type))
          return false;
      }
      for (Type type : op->getOperandTypes()) {
        if (!typeConverter.isLegal(type))
          return false;
      }
      for (auto &region : op->getRegions()) {
        if (!typeConverter.isLegal(&region))
          return false;
      }
      return true;
    });

    // Note that this will fail if we can't convert any types.
    if (failed(applyFullConversion(this->getOperation(), target,
                                   std::move(patterns)))) {
      return this->signalPassFailure();
    }
  }

  Converter typeConverter;
};
} // namespace

namespace {
struct DemoteI64ToI32Converter
    : public PrimitiveTypeConverter<IntegerType, IntegerType> {
  bool isSourceType(IntegerType type) override { return type.isInteger(64); }
  Type getTargetType(IntegerType type) override {
    return IntegerType::get(type.getContext(), 32, type.getSignedness());
  }
};
class DemoteI64ToI32Pass final
    : public ConvertTypesPass<impl::DemoteI64ToI32PassBase<DemoteI64ToI32Pass>,
                              DemoteI64ToI32Converter> {};
} // namespace

namespace {
struct DemoteF32ToF16Converter
    : public PrimitiveTypeConverter<Float32Type, Float16Type> {
  Type getTargetType(Float32Type type) override {
    return Float16Type::get(type.getContext());
  }
};
class DemoteF32ToF16Pass final
    : public ConvertTypesPass<impl::DemoteF32ToF16PassBase<DemoteF32ToF16Pass>,
                              DemoteF32ToF16Converter> {};
} // namespace

namespace {
struct PromoteF16ToF32Converter
    : public PrimitiveTypeConverter<Float16Type, Float32Type> {
  Type getTargetType(Float16Type type) override {
    return Float32Type::get(type.getContext());
  }
};
class PromoteF16ToF32Pass final
    : public ConvertTypesPass<
          impl::PromoteF16ToF32PassBase<PromoteF16ToF32Pass>,
          PromoteF16ToF32Converter> {};
} // namespace

namespace {
struct PromoteBF16ToF32Converter
    : public FloatTypeConverter<BFloat16Type, Float32Type> {
  Type getTargetType(BFloat16Type type) override {
    return Float32Type::get(type.getContext());
  }
};
class PromoteBF16ToF32Pass final
    : public ConvertTypesPass<
          impl::PromoteBF16ToF32PassBase<PromoteBF16ToF32Pass>,
          PromoteBF16ToF32Converter> {};
} // namespace

namespace {
struct DemoteF64ToF32Converter
    : public PrimitiveTypeConverter<Float64Type, Float32Type> {
  Type getTargetType(Float64Type type) override {
    return Float32Type::get(type.getContext());
  }
};
class DemoteF64ToF32Pass final
    : public ConvertTypesPass<impl::DemoteF64ToF32PassBase<DemoteF64ToF32Pass>,
                              DemoteF64ToF32Converter> {};
} // namespace

} // namespace mlir::iree_compiler::InputConversion
