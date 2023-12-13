// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file implements a pass to emulate 16-bit brain float arithmetic
// operations with float 32 equivalents.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <utility>

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Utils/ConversionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

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
                             convertType(type.getElementType()),
                             type.getScalableDims());
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
    if (op->hasTrait<OpTrait::ConstantLike>() ||
        isa<IREE::Util::GlobalOpInterface>(op)) {
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
      rewriter.applySignatureConversion(newRegion, result);
    }

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// Handles constructing the appropriate ext/trunc operations that depend on
// element type. This could be for floating point, signed integers, and
// unsigned integer values.
template <typename OpTy, typename TypeTy,
          typename OperandToResultWidthLegalityRelation>
struct ConvertTypeSensitiveArithCastOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    auto operandType = adaptor.getIn().getType();

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

// This is required due to https://github.com/openxla/iree/issues/13891.
// We are not able to execute `arith.extf` or `arith.truncf` on a scalar
// vector type. To handle materializing `vector<bf16>` to `vector<f32>`
// we should propagate into the source operation by constant propagation
// instead.
template <typename SrcOp>
class PropagateCastF : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    auto operand = op.getOperand();
    auto ty = dyn_cast<VectorType>(operand.getType());
    auto resultTy = dyn_cast<VectorType>(op.getType());

    if (!ty || ty.getRank() != 0) {
      return rewriter.notifyMatchFailure(op, "Not casting from vector-scalar");
    }

    mlir::ElementsAttr vectorCst;
    if (!matchPattern(operand, m_Constant(&vectorCst))) {
      return failure();
    }

    mlir::FloatAttr val = vectorCst.getSplatValue<mlir::FloatAttr>();
    auto newVal = FloatAttr::get(resultTy.getElementType(),
                                 val.getValue().convertToDouble());
    auto vectorVal = DenseElementsAttr::get(resultTy, newVal);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultTy, vectorVal);
    return success();
  }
};

// Converts BF16s to F32s.
struct PromoteBF16ToF32Converter
    : public FloatTypeConverter<BFloat16Type, Float32Type> {
  Type getTargetType(BFloat16Type type) override {
    return Float32Type::get(type.getContext());
  }
};

struct ConvertBf16ArithToF32Pass
    : public ConvertBf16ArithToF32Base<ConvertBf16ArithToF32Pass> {
  using ConvertBf16ArithToF32Base::ConvertBf16ArithToF32Base;
  void runOnOperation() override {
    MLIRContext *context = &this->getContext();
    RewritePatternSet patterns(context);
    patterns.insert<GenericTypeConversionPattern>(context, typeConverter);
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
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });

    auto checkOp = [&](Operation *op) {
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
    };

    // Operations are legal if they don't contain any illegal type.
    target.addDynamicallyLegalDialect<arith::ArithDialect>(checkOp);
    target.addDynamicallyLegalDialect<math::MathDialect>(checkOp);

    // Some arithmetic operations exist in the vector dialect.
    target.addDynamicallyLegalOp<vector::FMAOp, vector::ReductionOp,
                                 vector::MultiDimReductionOp, vector::MaskOp,
                                 vector::MatmulOp, vector::OuterProductOp, vector::YieldOp>(
        checkOp);

    // Some ops are always legal.
    target.addLegalOp<arith::BitcastOp>();

    if (failed(applyFullConversion(this->getOperation(), target,
                                   std::move(patterns)))) {
      return this->signalPassFailure();
    }

    // This is due to arith.extf and arith.truncf validation failing on
    // rank-0 vectors. These can only be generated by arith.constant so
    // in these cases we just propagate the type.
    RewritePatternSet cleanupPatterns(context);
    cleanupPatterns
        .insert<PropagateCastF<arith::TruncFOp>, PropagateCastF<arith::ExtFOp>>(
            context);
    if (applyPatternsAndFoldGreedily(this->getOperation(),
                                     std::move(cleanupPatterns))
            .failed()) {
      return this->signalPassFailure();
    }
  }

  PromoteBF16ToF32Converter typeConverter;
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertBf16ArithToF32Pass() {
  return std::make_unique<ConvertBf16ArithToF32Pass>();
}

} // namespace mlir::iree_compiler
