// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------- ConvertUnsupportedFloatArithPass.cpp ----------------===//
//
//   Emulate arith and vector floating point operations that use float types
//   which are unspported on a target by inserting extf/truncf pairs around all
//   such operations in order to produce arithmetic that can be performed while
//   preserving the original rounding behavior.
//
//===---------------------------------------------------------------------===//

#include <type_traits>

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-convert-unsupported-float-arith"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTUNSUPPORTEDFLOATARITHPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// SFINAE (i.e., substitution failure is not an error) helper to detect if
/// T::get(MLIRContext*) is valid.
template <typename T, typename = void>
struct HasContextGet : std::false_type {};
template <typename T>
struct HasContextGet<
    T, std::void_t<decltype(T::get(std::declval<MLIRContext *>()))>>
    : std::true_type {};

/// Helpers to append types to a vector if they are f8 types.
template <typename T>
void maybeAppendType(MLIRContext *ctx, SmallVectorImpl<Type> &types) {
  if constexpr (HasContextGet<T>::value) {
    Type t = T::get(ctx);
    if (isa<FloatType>(t) && t.getIntOrFloatBitWidth() == 8) {
      types.push_back(t);
    }
  }
}
template <typename... Ts>
void appendTypesIf(MLIRContext *ctx, SmallVectorImpl<Type> &types) {
  (maybeAppendType<Ts>(ctx, types), ...);
}

//===----------------------------------------------------------------------===//
// Float type format parameters
//===----------------------------------------------------------------------===//

/// Parameters describing a floating-point format for conversion.
struct FloatTypeInfo {
  unsigned expBits;     // Number of exponent bits.
  unsigned mantBits;    // Number of mantissa bits (excluding implicit bit).
  int bias;             // Exponent bias.
  bool hasInf;          // Whether the format supports infinity.
  bool hasNegZero;      // Whether the format supports negative zero.
  unsigned nanEncoding; // Bit pattern for canonical NaN.
  unsigned infEncoding; // Bit pattern for +Inf (0 if no Inf support).
};

/// Returns format info for a float type by querying APFloat semantics. It
/// assumes that the type can always be casted to FloatType.
static FloatTypeInfo getFloatTypeInfo(Type type) {
  auto floatType = cast<FloatType>(type);
  const llvm::fltSemantics &sem = floatType.getFloatSemantics();
  FloatTypeInfo info;

  // Derive format parameters from APFloat semantics.
  // Precision includes the implicit bit, so mantissa bits = precision - 1.
  info.mantBits = llvm::APFloat::semanticsPrecision(sem) - 1;
  unsigned totalBits = llvm::APFloat::semanticsSizeInBits(sem);
  // Total bits = 1 (sign) + expBits + mantBits.
  info.expBits = totalBits - 1 - info.mantBits;

  // Bias = 1 - minExponent. This works for both IEEE and FNUZ formats.
  // For f32: minExp = -126, so bias = 1 - (-126) = 127.
  // For f8E5M2FNUZ: minExp = -15, so bias = 1 - (-15) = 16.
  int minExp = llvm::APFloat::semanticsMinExponent(sem);
  info.bias = 1 - minExp;

  info.hasInf = llvm::APFloat::semanticsHasInf(sem);
  // Check for negative zero support by seeing if getZero with negative=true
  // produces a negative zero (vs NaN in FNUZ types where 0x80 encodes NaN).
  llvm::APFloat negZero = llvm::APFloat::getZero(sem, /*Negative=*/true);
  info.hasNegZero = negZero.isZero() && negZero.isNegative();

  // Compute NaN and Inf encodings based on format type.
  unsigned maxExpCode = (1u << info.expBits) - 1;
  unsigned mantMask = (1u << info.mantBits) - 1;

  if (!info.hasNegZero && !info.hasInf) {
    // FNUZ types: NaN = 0x80 (sign bit set, all else zero), no Inf.
    info.nanEncoding = 1u << (totalBits - 1);
    info.infEncoding = 0;
  } else if (info.hasInf) {
    // IEEE types: NaN = all exp bits + some mant bits, Inf = all exp bits.
    info.infEncoding = maxExpCode << info.mantBits;
    info.nanEncoding = info.infEncoding | mantMask;
  } else {
    // FN types (like f8E4M3FN): No Inf, but has NaN at max exp with mant != 0.
    info.infEncoding = 0;
    info.nanEncoding = (maxExpCode << info.mantBits) | mantMask;
  }

  return info;
}

//===----------------------------------------------------------------------===//
// Helper for float emulation patterns
//===----------------------------------------------------------------------===//

/// Helper class for emulating float conversions using integer bit manipulation.
/// Handles both scalar and vector types uniformly.
class FloatEmulationHelper {
public:
  FloatEmulationHelper(RewriterBase &rewriter, Location loc, Type type)
      : rewriter(rewriter), loc(loc), vecType(dyn_cast<VectorType>(type)) {
    Type i32ScalarType = rewriter.getI32Type();
    Type i8ScalarType = rewriter.getI8Type();
    i32Type = vecType ? VectorType::get(vecType.getShape(), i32ScalarType)
                      : i32ScalarType;
    i8Type = vecType ? VectorType::get(vecType.getShape(), i8ScalarType)
                     : i8ScalarType;
  }

  /// Creates an i32 constant, splatted if working with vectors.
  Value createI32Const(int64_t value) {
    auto attr = rewriter.getIntegerAttr(rewriter.getI32Type(), value);
    if (vecType) {
      auto splatAttr = SplatElementsAttr::get(cast<ShapedType>(i32Type), attr);
      return rewriter.createOrFold<arith::ConstantOp>(loc, i32Type, splatAttr);
    }
    return rewriter.createOrFold<arith::ConstantOp>(loc, i32Type, attr);
  }

  /// Extracts sign, exponent, and mantissa from an i32 value.
  /// Returns {sign, biasedExp, mantissa}.
  std::tuple<Value, Value, Value> extractFields(Value i32Val,
                                                const FloatTypeInfo &info) {
    Value cExpShift = createI32Const(info.mantBits);
    Value cSignShift = createI32Const(info.expBits + info.mantBits);
    Value cExpMask = createI32Const((1 << info.expBits) - 1);
    Value cMantMask = createI32Const((1 << info.mantBits) - 1);

    Value sign = arith::ShRUIOp::create(rewriter, loc, i32Val, cSignShift);
    Value biasedExp = arith::ShRUIOp::create(rewriter, loc, i32Val, cExpShift);
    biasedExp = arith::AndIOp::create(rewriter, loc, biasedExp, cExpMask);
    Value mant = arith::AndIOp::create(rewriter, loc, i32Val, cMantMask);

    return {sign, biasedExp, mant};
  }

  /// Packs sign, exponent, and mantissa into an i32 value.
  Value packFields(Value sign, Value biasedExp, Value mant,
                   const FloatTypeInfo &info) {
    Value cExpShift = createI32Const(info.mantBits);
    Value cSignShift = createI32Const(info.expBits + info.mantBits);

    Value signShifted = arith::ShLIOp::create(rewriter, loc, sign, cSignShift);
    Value expShifted =
        arith::ShLIOp::create(rewriter, loc, biasedExp, cExpShift);
    Value result = arith::OrIOp::create(rewriter, loc, signShifted, expShifted);
    return arith::OrIOp::create(rewriter, loc, result, mant);
  }

  /// Checks if the value represents zero (exp == 0 && mant == 0).
  Value isZero(Value biasedExp, Value mant) {
    Value c0 = createI32Const(0);
    Value expIsZero = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, biasedExp, c0);
    Value mantIsZero = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, mant, c0);
    return arith::AndIOp::create(rewriter, loc, expIsZero, mantIsZero);
  }

  Type getI32Type() const { return i32Type; }
  Type getI8Type() const { return i8Type; }

private:
  RewriterBase &rewriter;
  Location loc;
  VectorType vecType;
  Type i32Type;
  Type i8Type;
};

//===----------------------------------------------------------------------===//
// TruncF to small float emulation pattern
//===----------------------------------------------------------------------===//

/// Emulates arith.truncf from f32 to fp8 using integer bit manipulation.
/// This is needed because LLVM doesn't support fptrunc to fp8 types.
///
/// The conversion handles three categories of fp8 formats:
///
/// 1. FNUZ types (f8E5M2FNUZ, f8E4M3FNUZ): No Inf, no negative zero.
///    - NaN is encoded as 0x80 (sign=1, exp=0, mant=0).
///    - Overflow (finite values too large) produces NaN.
///    - Zero is always positive (0x00).
///
/// 2. IEEE types (f8E5M2): Has Inf and negative zero.
///    - NaN is encoded as exp=max with non-zero mantissa.
///    - Inf is encoded as exp=max with zero mantissa.
///    - Overflow produces signed Inf (preserves sign).
///    - Zero preserves sign (+0.0 or -0.0).
///
/// 3. FN types (f8E4M3FN): No Inf, but has negative zero.
///    - NaN is encoded at a specific bit pattern (e.g., 0x7F).
///    - Overflow produces NaN.
///    - Zero preserves sign (+0.0 or -0.0).
///
/// Special value handling priority (highest to lowest):
///   1. Source NaN -> destination NaN (must propagate).
///   2. Source Inf -> destination Inf (IEEE) or NaN (FNUZ/FN).
///   3. Source zero -> destination zero (preserve sign if supported).
///   4. Overflow -> Inf (IEEE) or NaN (FNUZ/FN).
///   5. Underflow -> zero.
///
/// Denormal handling:
///   This implementation uses flush-to-zero (FTZ) semantics for denormals.
///   When the adjusted exponent is <= 0 (underflow), the result is flushed to
///   zero rather than producing a denormal fp8 value. This is a known
///   limitation that simplifies the implementation. Proper denormal support
///   would require additional logic to compute the subnormal mantissa shift.
///
/// Rounding:
///   Uses round-half-away-from-zero (adding the round bit), not IEEE 754
///   round-half-to-even (banker's rounding). This is a simplification that
///   may cause minor differences compared to strict IEEE 754 implementations.
struct TruncFToFP8 final : public OpRewritePattern<arith::TruncFOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();
    Type inputType = op.getIn().getType();
    Type resultElemType = getElementTypeOrSelf(resultType);
    Type inputElemType = getElementTypeOrSelf(inputType);

    // TODO(#23105): handle other fp types, e.g., fp4.
    if (!isa<Float32Type>(inputElemType) ||
        resultElemType.getIntOrFloatBitWidth() != 8) {
      return failure();
    }

    FloatTypeInfo srcInfo = getFloatTypeInfo(inputElemType);
    FloatTypeInfo dstInfo = getFloatTypeInfo(resultElemType);
    Location loc = op.getLoc();
    FloatEmulationHelper helper(rewriter, loc, resultType);

    // Calculate format-specific constants.
    int biasDiff = srcInfo.bias - dstInfo.bias;
    int mantShift = srcInfo.mantBits - dstInfo.mantBits;
    int dstMantMask = (1 << dstInfo.mantBits) - 1;
    int roundBit = 1 << (mantShift - 1);

    // For IEEE types, max normal exponent is (2^expBits - 2) because
    // (2^expBits - 1) encodes Inf/NaN. For FNUZ types, all exponent codes
    // except 0 (with sign=1 for NaN) are valid normal values.
    int dstMaxNormalExp = dstInfo.hasInf ? (1 << dstInfo.expBits) - 2
                                         : (1 << dstInfo.expBits) - 1;

    Value c0 = helper.createI32Const(0);
    Value c1 = helper.createI32Const(1);
    Value cBiasDiff = helper.createI32Const(biasDiff);
    Value cMantShift = helper.createI32Const(mantShift);
    Value cDstMaxNormalExp = helper.createI32Const(dstMaxNormalExp);
    Value cDstMantMask = helper.createI32Const(dstMantMask);
    Value cRoundBit = helper.createI32Const(roundBit);
    Value cMantOverflowBit = helper.createI32Const(1 << srcInfo.mantBits);
    Value cSrcExpMask = helper.createI32Const((1 << srcInfo.expBits) - 1);
    Value cNaN = helper.createI32Const(dstInfo.nanEncoding);
    // Overflow produces Inf for IEEE types, NaN for FNUZ/FN types.
    Value cOverflow = helper.createI32Const(
        dstInfo.hasInf ? dstInfo.infEncoding : dstInfo.nanEncoding);

    // Bitcast f32 to i32 and extract fields.
    Value i32Val = arith::BitcastOp::create(rewriter, loc, helper.getI32Type(),
                                            op.getIn());
    auto [sign, biasedExpSrc, mantSrc] = helper.extractFields(i32Val, srcInfo);

    // Check for NaN/Inf in source (exponent == max).
    Value srcIsNanOrInf = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, biasedExpSrc, cSrcExpMask);
    // Distinguish NaN vs Inf in source: NaN has non-zero mantissa.
    Value srcIsNan = arith::AndIOp::create(
        rewriter, loc, srcIsNanOrInf,
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, mantSrc,
                              c0));
    Value srcIsInf = arith::AndIOp::create(
        rewriter, loc, srcIsNanOrInf,
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, mantSrc,
                              c0));

    // Check for zero.
    Value isZeroVal = helper.isZero(biasedExpSrc, mantSrc);

    // Adjust exponent bias.
    Value biasedExpDst =
        arith::SubIOp::create(rewriter, loc, biasedExpSrc, cBiasDiff);

    // Check for overflow (exp > dstMaxNormalExp) and underflow (exp <= 0).
    Value isOverflow =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                              biasedExpDst, cDstMaxNormalExp);
    Value isUnderflow = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sle, biasedExpDst, c0);

    // Round mantissa and handle mantissa overflow (carry into exponent).
    Value mantRounded =
        arith::AddIOp::create(rewriter, loc, mantSrc, cRoundBit);
    Value mantOverflow =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                              mantRounded, cMantOverflowBit);

    Value expIncr = arith::AddIOp::create(rewriter, loc, biasedExpDst, c1);
    biasedExpDst = arith::SelectOp::create(rewriter, loc, mantOverflow, expIncr,
                                           biasedExpDst);
    mantRounded =
        arith::SelectOp::create(rewriter, loc, mantOverflow, c0, mantRounded);

    // Re-check overflow after potential exponent increment from rounding.
    isOverflow = arith::OrIOp::create(
        rewriter, loc, isOverflow,
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                              biasedExpDst, cDstMaxNormalExp));

    // Shift mantissa to destination width.
    Value mantDst =
        arith::ShRUIOp::create(rewriter, loc, mantRounded, cMantShift);
    mantDst = arith::AndIOp::create(rewriter, loc, mantDst, cDstMantMask);

    // Pack result and handle special cases.
    Value result = helper.packFields(sign, biasedExpDst, mantDst, dstInfo);

    // Compute sign bit position for destination format.
    int dstSignBitPos = dstInfo.expBits + dstInfo.mantBits;
    Value cSignShift = helper.createI32Const(dstSignBitPos);
    Value signBit = arith::ShLIOp::create(rewriter, loc, sign, cSignShift);

    // For IEEE types with Inf, compute signed Inf (preserve sign on overflow).
    // For FNUZ/FN types, overflow/Inf always produces NaN (unsigned).
    Value signedOverflow = cOverflow;
    if (dstInfo.hasInf) {
      signedOverflow = arith::OrIOp::create(rewriter, loc, cOverflow, signBit);
    }

    // For types with negative zero support, compute signed zero.
    // FNUZ types don't have negative zero, so zero is always positive.
    Value zeroResult = c0;
    if (dstInfo.hasNegZero) {
      zeroResult = signBit;
    }

    // Select order matters: later selects take precedence over earlier ones.
    // This is critical because multiple conditions can be true simultaneously.
    // For example, f32 NaN has exp=255, so after bias adjustment (255-112=143),
    // it exceeds dstMaxNormalExp (30), making both srcIsNan AND overflow true.
    // If overflow were checked after srcIsNan, NaN would incorrectly become
    // Inf.
    //
    // Order from lowest to highest priority:
    // 1. Overflow/underflow for normal values.
    // 2. Zero input (preserve sign for IEEE types).
    // 3. Source Inf (takes precedence over overflow, since Inf also overflows).
    // 4. Source NaN (highest priority, must always propagate).
    result = arith::SelectOp::create(rewriter, loc, isOverflow, signedOverflow,
                                     result);
    result = arith::SelectOp::create(rewriter, loc, isUnderflow, c0, result);
    result =
        arith::SelectOp::create(rewriter, loc, isZeroVal, zeroResult, result);
    result = arith::SelectOp::create(rewriter, loc, srcIsInf, signedOverflow,
                                     result);
    result = arith::SelectOp::create(rewriter, loc, srcIsNan, cNaN, result);

    // Truncate to i8 and bitcast to fp8.
    result = arith::TruncIOp::create(rewriter, loc, helper.getI8Type(), result);
    result = arith::BitcastOp::create(rewriter, loc, resultType, result);
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ExtF from small float emulation pattern
//===----------------------------------------------------------------------===//

/// Emulates arith.extf from fp8 to f32 using integer bit manipulation.
/// This is needed because LLVM doesn't support fpext from fp8 types.
///
/// The conversion handles three categories of fp8 formats:
///
/// 1. FNUZ types (f8E5M2FNUZ, f8E4M3FNUZ): No Inf, no negative zero.
///    - NaN is detected by exact bit pattern match (0x80).
///    - Zero (exp=0, mant=0) is always positive in f32.
///    - Note: FNUZ NaN (0x80) has exp=0, mant=0, so NaN check must take
///      precedence over zero check.
///
/// 2. IEEE types (f8E5M2): Has Inf and negative zero.
///    - NaN is detected as exp=max && mant!=0 -> f32 NaN (0x7FC00000).
///    - Inf is detected as exp=max && mant==0 -> f32 signed Inf.
///    - Zero preserves sign (+0.0 or -0.0).
///
/// 3. FN types (f8E4M3FN): No Inf, but has negative zero.
///    - NaN is detected by exact bit pattern match (e.g., 0x7F).
///    - Other exp=max values are valid normal numbers (no Inf).
///    - Zero preserves sign (+0.0 or -0.0).
///
/// Special value handling priority (highest to lowest):
///   1. Source NaN -> f32 NaN (0x7FC00000).
///   2. Source Inf -> f32 signed Inf (IEEE types only).
///   3. Source zero -> f32 zero (preserve sign if supported).
///
/// Denormal handling:
///   This implementation does NOT correctly handle denormal fp8 inputs
///   (exp=0, mant!=0). Denormal values are treated as if they were normal
///   values with biased exponent 0, which produces incorrect (larger) results.
///   This is a known limitation. When round-tripping through this emulation,
///   TruncFToFP8 uses FTZ so denormals won't be produced. However, fp8 data
///   from external sources (e.g., model weights) containing denormals will not
///   be converted correctly. Proper denormal support would require detecting
///   exp=0 && mant!=0 and computing the appropriate f32 exponent/mantissa.
struct ExtFFromFP8 final : public OpRewritePattern<arith::ExtFOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();
    Type inputType = op.getIn().getType();
    Type resultElemType = getElementTypeOrSelf(resultType);
    Type inputElemType = getElementTypeOrSelf(inputType);

    // TODO(#23105): handle other fp types, e.g., fp4.
    if (inputElemType.getIntOrFloatBitWidth() != 8 ||
        !isa<Float32Type>(resultElemType)) {
      return failure();
    }

    FloatTypeInfo srcInfo = getFloatTypeInfo(inputElemType);
    FloatTypeInfo dstInfo = getFloatTypeInfo(resultElemType);
    Location loc = op.getLoc();
    FloatEmulationHelper helper(rewriter, loc, resultType);

    // Calculate format-specific constants.
    int biasDiff = dstInfo.bias - srcInfo.bias;
    int mantShift = dstInfo.mantBits - srcInfo.mantBits;
    int srcMaxExp = (1 << srcInfo.expBits) - 1;

    Value c0 = helper.createI32Const(0);
    Value cBiasDiff = helper.createI32Const(biasDiff);
    Value cMantShift = helper.createI32Const(mantShift);
    Value cSrcMaxExp = helper.createI32Const(srcMaxExp);
    Value cF32NaN = helper.createI32Const(0x7FC00000);
    Value cF32Inf = helper.createI32Const(0x7F800000);
    Value cSignBit = helper.createI32Const(0x80000000);

    // Bitcast fp8 to i8, extend to i32, and extract fields.
    Value i8Val =
        arith::BitcastOp::create(rewriter, loc, helper.getI8Type(), op.getIn());
    Value i32Val =
        arith::ExtUIOp::create(rewriter, loc, helper.getI32Type(), i8Val);
    auto [sign, biasedExpSrc, mantSrc] = helper.extractFields(i32Val, srcInfo);

    // Detect special values based on format type.
    Value isNaN;
    Value isInf;
    if (!srcInfo.hasNegZero && !srcInfo.hasInf) {
      // FNUZ: NaN = 0x80 (sign=1, exp=0, mant=0), no Inf.
      Value cFNUZNaN = helper.createI32Const(srcInfo.nanEncoding);
      isNaN = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                    i32Val, cFNUZNaN);
    } else if (srcInfo.hasInf) {
      // IEEE types: NaN = exp==max && mant!=0, Inf = exp==max && mant==0.
      Value expIsMax = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, biasedExpSrc, cSrcMaxExp);
      Value mantIsZero = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, mantSrc, c0);
      Value mantIsNonZero = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ne, mantSrc, c0);
      isNaN = arith::AndIOp::create(rewriter, loc, expIsMax, mantIsNonZero);
      isInf = arith::AndIOp::create(rewriter, loc, expIsMax, mantIsZero);
    } else {
      // FN types (like f8E4M3FN): No Inf, NaN only at specific encoding.
      // NaN = exp==max && mant==maxMant (e.g., 0x7F for f8E4M3FN).
      // Other exp==max values with mant!=maxMant are valid normal numbers.
      Value cNaNEncoding = helper.createI32Const(srcInfo.nanEncoding);
      isNaN = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                    i32Val, cNaNEncoding);
    }

    // Check for zero.
    Value isZeroVal = helper.isZero(biasedExpSrc, mantSrc);

    // Adjust exponent bias and shift mantissa.
    Value biasedExpDst =
        arith::AddIOp::create(rewriter, loc, biasedExpSrc, cBiasDiff);
    Value mantDst = arith::ShLIOp::create(rewriter, loc, mantSrc, cMantShift);

    // Pack result and handle special cases.
    Value result = helper.packFields(sign, biasedExpDst, mantDst, dstInfo);

    // For types with negative zero support, compute signed zero in f32.
    // f32 -0.0 = 0x80000000 (sign bit at position 31).
    Value zeroResult = c0;
    if (srcInfo.hasNegZero) {
      Value cF32SignShift = helper.createI32Const(31);
      Value f32SignBit =
          arith::ShLIOp::create(rewriter, loc, sign, cF32SignShift);
      zeroResult = f32SignBit;
    }

    // Handle zero first (before NaN check for non-FNUZ types).
    // For FNUZ, zero check must come after NaN check since NaN (0x80) has
    // exp=0, mant=0 which looks like zero except for sign bit.
    result =
        arith::SelectOp::create(rewriter, loc, isZeroVal, zeroResult, result);

    // Handle Inf (IEEE types only): preserve sign.
    if (srcInfo.hasInf) {
      Value signedInf = arith::SelectOp::create(
          rewriter, loc,
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, sign,
                                c0),
          arith::OrIOp::create(rewriter, loc, cF32Inf, cSignBit), cF32Inf);
      result = arith::SelectOp::create(rewriter, loc, isInf, signedInf, result);
    }

    // Handle NaN last so it takes precedence over zero for FNUZ types. Then
    // bitcast it back to f32.
    result = arith::SelectOp::create(rewriter, loc, isNaN, cF32NaN, result);
    result = arith::BitcastOp::create(rewriter, loc, resultType, result);
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct ConvertUnsupportedFloatArithPass final
    : public impl::ConvertUnsupportedFloatArithPassBase<
          ConvertUnsupportedFloatArithPass> {
  void runOnOperation() override;
  using Base::Base;
};

} // namespace

static void populateCPUSourceAndTargetType(MLIRContext *ctx, Operation *op,
                                           SmallVectorImpl<Type> &sourceTypes,
                                           Type &targetType) {
  // For CPU backend, we emulate all the fp8 types to fp32.
  appendTypesIf<
#define GET_TYPEDEF_LIST
#include "mlir/IR/BuiltinTypes.cpp.inc"
      >(ctx, sourceTypes);
  targetType = Float32Type::get(ctx);
}

// Populates source and target conversion types based on the target
// architecture.
// TODO(pashu123): Refine the patterns based on the target arch.
static void populateGPUSourceAndTargetType(MLIRContext *ctx, Operation *op,
                                           SmallVectorImpl<Type> &sourceTypes,
                                           Type &targetType) {
  auto gpuAttr = getGPUTargetAttr(op);
  if (!gpuAttr) {
    return;
  }
  StringRef chipset = gpuAttr.getArch();
  FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
  if (failed(maybeChipset)) {
    LDBG() << "Invalid chip name";
    return;
  }
  constexpr amdgpu::Chipset kGfx942{9, 4, 2};
  constexpr amdgpu::Chipset kGfx950{9, 5, 0};
  constexpr amdgpu::Chipset kGfx10{10, 0, 0};
  constexpr amdgpu::Chipset kGfx12{12, 0, 0};
  // Add source and target conversion types for gfx94{*} series.
  if (*maybeChipset >= kGfx942 && *maybeChipset <= kGfx950) {
    sourceTypes.insert(sourceTypes.end(), {Float8E4M3FNUZType::get(ctx),
                                           Float8E5M2FNUZType::get(ctx)});
    targetType = Float32Type::get(ctx);
  }
  // gfx950 and gfx12+ support OCP FP8 conversions
  if (*maybeChipset >= kGfx12 ||
      (*maybeChipset <= kGfx10 && *maybeChipset >= kGfx950)) {
    // TODO(kdrewnia): On gfx950, add fp4 or fp6 here maybe.
    // TODO(kdrewnia): After checking for instruction avaiability, turn
    // the target type down to f16.
    sourceTypes.push_back(Float8E4M3FNType::get(ctx));
    sourceTypes.push_back(Float8E5M2Type::get(ctx));
    targetType = Float32Type::get(ctx);
  }
  return;
}

void ConvertUnsupportedFloatArithPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FunctionOpInterface funcOp = getOperation();
  SmallVector<Type> sourceTypes;
  Type targetType = nullptr;

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  bool isCPU = isLLVMCPUBackend(targetAttr);
  if (isCPU) {
    populateCPUSourceAndTargetType(context, funcOp, sourceTypes, targetType);
  } else if (isROCMBackend(targetAttr)) {
    populateGPUSourceAndTargetType(context, funcOp, sourceTypes, targetType);
  } else {
    LDBG() << "backend does not require float emulation";
    return;
  }

  if (sourceTypes.empty() || !targetType) {
    LDBG() << "no source or target type specified, float emulation will do "
              "nothing";
    return;
  }

  if (llvm::is_contained(sourceTypes, targetType)) {
    funcOp->emitError() << " target type cannot be an unsupported source type";
    return signalPassFailure();
  }

  // Apply the standard float emulation patterns. This inserts extf/truncf pairs
  // around unsupported float operations.
  {
    TypeConverter converter;
    arith::populateEmulateUnsupportedFloatsConversions(converter, sourceTypes,
                                                       targetType);
    RewritePatternSet patterns(context);
    arith::populateEmulateUnsupportedFloatsPatterns(patterns, converter);
    ConversionTarget target(*context);
    arith::populateEmulateUnsupportedFloatsLegality(target, converter);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // For CPU, emulate extf/truncf to/from small float types using integer ops.
  // This is needed because LLVM doesn't support fpext/fptrunc for fp8 types;
  // the fp8 types eventually get lowered to i8 in LLVM IR.
  if (isCPU) {
    RewritePatternSet emulationPatterns(context);
    emulationPatterns.add<TruncFToFP8, ExtFFromFP8>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(emulationPatterns)))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
