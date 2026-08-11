//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/FP8Lowering.h"

#include <cstdint>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {
namespace {

struct FP8Format {
  uint32_t exponentBits;
  uint32_t mantissaBits;
  uint32_t exponentBias;
  bool hasInfinity;
};

static Type getTypeWithElementType(Type type, Type elementType) {
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return VectorType::get(vectorType.getShape(), elementType,
                           vectorType.getScalableDims());
  }
  return elementType;
}

static Value createIntegerConstant(ConversionPatternRewriter &rewriter,
                                   Location loc, Type type, uint32_t value) {
  auto elementType = cast<IntegerType>(getElementTypeOrSelf(type));
  auto elementAttr = rewriter.getIntegerAttr(elementType, value);
  Attribute attr = elementAttr;
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    attr = SplatElementsAttr::get(shapedType, elementAttr);
  }
  return LLVM::ConstantOp::create(rewriter, loc, type, attr);
}

static uint32_t getSubnormalF32Bits(uint32_t mantissa,
                                    const FP8Format &format) {
  if (mantissa == 0) {
    return 0;
  }

  uint32_t leadingBit = 0;
  for (uint32_t value = mantissa; value > 1; value >>= 1) {
    ++leadingBit;
  }
  int32_t exponent = 1 - static_cast<int32_t>(format.exponentBias) -
                     static_cast<int32_t>(format.mantissaBits) +
                     static_cast<int32_t>(leadingBit);
  uint32_t fraction =
      (mantissa - (1u << leadingBit)) << (23 - leadingBit);
  return (static_cast<uint32_t>(exponent + 127) << 23) | fraction;
}

static Value createFP8ToF32Bits(ConversionPatternRewriter &rewriter,
                                Location loc, Value input,
                                const FP8Format &format) {
  Type i32Type = getTypeWithElementType(input.getType(), rewriter.getI32Type());
  Value bits = LLVM::ZExtOp::create(rewriter, loc, i32Type, input);

  Value sign = LLVM::LShrOp::create(
      rewriter, loc, i32Type, bits,
      createIntegerConstant(rewriter, loc, i32Type, 7));
  sign = LLVM::ShlOp::create(rewriter, loc, i32Type, sign,
                             createIntegerConstant(rewriter, loc, i32Type, 31));

  Value exponent = LLVM::LShrOp::create(
      rewriter, loc, i32Type, bits,
      createIntegerConstant(rewriter, loc, i32Type, format.mantissaBits));
  uint32_t exponentMask = (1u << format.exponentBits) - 1;
  exponent = LLVM::AndOp::create(
      rewriter, loc, i32Type, exponent,
      createIntegerConstant(rewriter, loc, i32Type, exponentMask));

  uint32_t mantissaMask = (1u << format.mantissaBits) - 1;
  Value mantissa = LLVM::AndOp::create(
      rewriter, loc, i32Type, bits,
      createIntegerConstant(rewriter, loc, i32Type, mantissaMask));

  Value exponentF32 = LLVM::AddOp::create(
      rewriter, loc, i32Type, exponent,
      createIntegerConstant(rewriter, loc, i32Type,
                            127 - format.exponentBias));
  exponentF32 = LLVM::ShlOp::create(
      rewriter, loc, i32Type, exponentF32,
      createIntegerConstant(rewriter, loc, i32Type, 23));
  Value mantissaF32 = LLVM::ShlOp::create(
      rewriter, loc, i32Type, mantissa,
      createIntegerConstant(rewriter, loc, i32Type, 23 - format.mantissaBits));
  Value normal = LLVM::OrOp::create(
      rewriter, loc, i32Type, sign,
      LLVM::OrOp::create(rewriter, loc, i32Type, exponentF32, mantissaF32));

  // Build the (small) subnormal lookup with selects. This avoids relying on a
  // target-specific exponent-scaling instruction and works for scalar and
  // vector values alike.
  Value subnormal = sign;
  for (uint32_t value = 1; value <= mantissaMask; ++value) {
    Value valueBits = LLVM::OrOp::create(
        rewriter, loc, i32Type, sign,
        createIntegerConstant(rewriter, loc, i32Type,
                              getSubnormalF32Bits(value, format)));
    Value isValue = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::eq, mantissa,
        createIntegerConstant(rewriter, loc, i32Type, value));
    subnormal = LLVM::SelectOp::create(rewriter, loc, isValue, valueBits,
                                       subnormal);
  }

  Value isExponentZero = LLVM::ICmpOp::create(
      rewriter, loc, LLVM::ICmpPredicate::eq, exponent,
      createIntegerConstant(rewriter, loc, i32Type, 0));
  Value result = LLVM::SelectOp::create(rewriter, loc, isExponentZero,
                                        subnormal, normal);

  Value isMaxExponent = LLVM::ICmpOp::create(
      rewriter, loc, LLVM::ICmpPredicate::eq, exponent,
      createIntegerConstant(rewriter, loc, i32Type, exponentMask));
  if (format.hasInfinity) {
    Value infinity = LLVM::OrOp::create(
        rewriter, loc, i32Type, sign,
        createIntegerConstant(rewriter, loc, i32Type, 0x7f800000));
    Value nan = LLVM::OrOp::create(
        rewriter, loc, i32Type, infinity,
        LLVM::ShlOp::create(
            rewriter, loc, i32Type, mantissa,
            createIntegerConstant(rewriter, loc, i32Type,
                                  23 - format.mantissaBits)));
    Value isMantissaZero = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::eq, mantissa,
        createIntegerConstant(rewriter, loc, i32Type, 0));
    Value special = LLVM::SelectOp::create(rewriter, loc, isMantissaZero,
                                           infinity, nan);
    result = LLVM::SelectOp::create(rewriter, loc, isMaxExponent, special,
                                    result);
  } else {
    Value isNaN = LLVM::AndOp::create(
        rewriter, loc, isMaxExponent.getType(), isMaxExponent,
        LLVM::ICmpOp::create(
            rewriter, loc, LLVM::ICmpPredicate::eq, mantissa,
            createIntegerConstant(rewriter, loc, i32Type, mantissaMask)));
    Value nan = LLVM::OrOp::create(
        rewriter, loc, i32Type, sign,
        createIntegerConstant(rewriter, loc, i32Type, 0x7fc00000));
    result = LLVM::SelectOp::create(rewriter, loc, isNaN, nan, result);
  }

  return result;
}

struct LowerFP8ExtFOp final : OpConversionPattern<arith::ExtFOp> {
  LowerFP8ExtFOp(const LLVMTypeConverter &typeConverter,
                 MLIRContext *context)
      : OpConversionPattern(typeConverter, context, PatternBenefit(2)) {}

  LogicalResult
  matchAndRewrite(arith::ExtFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type sourceType = getElementTypeOrSelf(op.getIn().getType());
    Type resultType = getElementTypeOrSelf(op.getOut().getType());
    if (!resultType.isF32()) {
      return failure();
    }

    FP8Format format;
    if (isa<Float8E4M3FNType>(sourceType)) {
      format = {/*exponentBits=*/4, /*mantissaBits=*/3,
                /*exponentBias=*/7, /*hasInfinity=*/false};
    } else if (isa<Float8E5M2Type>(sourceType)) {
      format = {/*exponentBits=*/5, /*mantissaBits=*/2,
                /*exponentBias=*/15, /*hasInfinity=*/true};
    } else {
      return failure();
    }

    Type convertedResultType = getTypeConverter()->convertType(op.getType());
    if (!convertedResultType) {
      return failure();
    }
    Value resultBits = createFP8ToF32Bits(rewriter, op.getLoc(),
                                           adaptor.getIn(), format);
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, convertedResultType,
                                                 resultBits);
    return success();
  }
};

} // namespace

void populateFP8ToNVVMConversionPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  patterns.add<LowerFP8ExtFOp>(typeConverter, patterns.getContext());
}

} // namespace mlir::iree_compiler
