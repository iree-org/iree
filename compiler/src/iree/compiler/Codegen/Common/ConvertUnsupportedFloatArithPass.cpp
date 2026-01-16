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

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
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

/// Detector for T::get(MLIRContext*) used by llvm::is_detected.
template <typename T>
using hasContextGet = decltype(T::get(std::declval<MLIRContext *>()));

/// Helpers to append types to a vector if they are f8 types.
template <typename T>
static void maybeAppendType(MLIRContext *ctx, SmallVectorImpl<Type> &types) {
  if constexpr (llvm::is_detected<hasContextGet, T>::value) {
    Type t = T::get(ctx);
    if (isa<FloatType>(t) && t.getIntOrFloatBitWidth() == 8) {
      types.push_back(t);
    }
  }
}
template <typename... Ts>
static void appendTypesIf(MLIRContext *ctx, SmallVectorImpl<Type> &types) {
  (maybeAppendType<Ts>(ctx, types), ...);
}

//===----------------------------------------------------------------------===//
// Floating-point format constants and type info
//===----------------------------------------------------------------------===//
//
// This follows the same approach as IREE's
// runtime/src/iree/base/internal/math.h for floating-point conversions. The
// compiler emits equivalent logic using MLIR arith ops instead of C control
// flow.
//
// IEEE 754 floating-point format: [sign | exponent | mantissa]
//   - Sign: 1 bit (0 = positive, 1 = negative)
//   - Exponent: biased (stored = actual + bias)
//   - Mantissa: fractional bits with implicit leading 1 for normal values
//
// Special values:
//   - Zero: exp=0, mantissa=0 (signed zero if format supports it)
//   - Denormal: exp=0, mantissa!=0, value = mantissa * 2^(1-bias-mantissa_bits)
//   - Inf: exp=max, mantissa=0 (IEEE types only)
//   - NaN: exp=max, mantissa!=0 (IEEE), or special encoding (FNUZ: 0x80)

// F32 format constants (IEEE 754 binary32).
constexpr int kF32MantBits = 23;
constexpr int kF32Bias = 127;
constexpr int kF32MantMask = (1 << kF32MantBits) - 1;
constexpr int kF32MaxExp = 0xFF; // All exponent bits set (255).

/// Parameters describing a floating-point format for conversion.
/// Derived from APFloat semantics at runtime to support all MLIR float types.
struct FloatTypeInfo {
  int expBits;       // Number of exponent bits.
  int mantBits;      // Number of mantissa bits (excluding implicit leading 1).
  int bias;          // Exponent bias: stored_exp = actual_exp + bias.
  bool hasInf;       // Format supports infinity (exp=max, mantissa=0).
  bool hasNan;       // Format supports NaN.
  bool hasNegZero;   // Format supports negative zero.
  bool nanAsNegZero; // NaN encoded as 0x80 (FNUZ types).
  unsigned signMask; // Bit mask for sign bit.
  unsigned expMask;  // Bit mask for exponent field.
  unsigned mantMask; // Bit mask for mantissa field.
  unsigned nanEncoding; // Bit pattern for canonical NaN.
  unsigned infEncoding; // Bit pattern for +Inf (0 if no Inf).
  unsigned maxFinite;   // Bit pattern for max finite value.
};

/// Returns format info for a float type by querying APFloat semantics.
static FloatTypeInfo getFloatTypeInfo(Type type) {
  auto floatType = cast<FloatType>(type);
  const llvm::fltSemantics &sem = floatType.getFloatSemantics();
  FloatTypeInfo info;

  // Derive format parameters from APFloat semantics.
  info.mantBits = llvm::APFloat::semanticsPrecision(sem) - 1;
  unsigned totalBits = llvm::APFloat::semanticsSizeInBits(sem);
  info.expBits = totalBits - 1 - info.mantBits;

  // Compute masks.
  info.signMask = 1u << (info.expBits + info.mantBits);
  info.mantMask = (1u << info.mantBits) - 1;
  info.expMask = ((1u << info.expBits) - 1) << info.mantBits;

  // Bias from APFloat semantics.
  int minExp = llvm::APFloat::semanticsMinExponent(sem);
  info.bias = 1 - minExp;

  info.hasInf = llvm::APFloat::semanticsHasInf(sem);
  info.hasNan = llvm::APFloat::semanticsHasNaN(sem);

  // Check for FNUZ types where NaN is encoded as negative zero (0x80).
  // These types have no negative zero and no infinity.
  llvm::APFloat negZero = llvm::APFloat::getZero(sem, /*Negative=*/true);
  info.hasNegZero = negZero.isZero() && negZero.isNegative();
  info.nanAsNegZero = !info.hasNegZero && !info.hasInf && info.hasNan;

  // Compute NaN and Inf encodings.
  unsigned maxExpCode = (1u << info.expBits) - 1;
  if (info.nanAsNegZero) {
    // FNUZ types: NaN = 0x80 (sign bit set, all else zero).
    info.nanEncoding = info.signMask;
  } else {
    // IEEE and FN types: NaN = all exp bits + some mantissa bits.
    info.nanEncoding = info.expMask | info.mantMask;
  }

  // Inf encoding: only meaningful for types with infinity.
  info.infEncoding = info.hasInf ? info.expMask : 0;

  // Max finite value: for types with infinity, it's exp=(max-1), mantissa=all
  // 1s. For types without infinity, max exp is valid, so exp=max, mantissa=all
  // 1s (except for FN types where max mantissa is NaN).
  if (info.hasInf) {
    info.maxFinite = ((maxExpCode - 1) << info.mantBits) | info.mantMask;
  } else if (info.hasNan && !info.nanAsNegZero) {
    // FN types: max exp is valid but max mantissa is NaN.
    info.maxFinite = info.expMask | (info.mantMask - 1);
  } else {
    // FNUZ or no-NaN types: all bit patterns except NaN are valid.
    info.maxFinite = info.expMask | info.mantMask;
  }

  return info;
}

//===----------------------------------------------------------------------===//
// Helper for float emulation patterns
//===----------------------------------------------------------------------===//

/// Extracted components from an f32 value stored as i32 bits.
struct F32Fields {
  Value sign;      // Sign bit (0 or 1) shifted to bit 0.
  Value biasedExp; // Biased exponent (8 bits).
  Value mantissa;  // Mantissa (23 bits, no implicit leading 1).
};

/// Helper class for emulating float conversions using integer bit manipulation.
/// Handles both scalar and vector types uniformly.
class FloatEmulationHelper {
public:
  FloatEmulationHelper(RewriterBase &rewriter, Location loc, Type type)
      : rewriter(rewriter), loc(loc), vecType(dyn_cast<VectorType>(type)) {
    Type i32ScalarType = rewriter.getI32Type();
    Type i8ScalarType = rewriter.getI8Type();
    Type f32ScalarType = rewriter.getF32Type();
    i32Type = vecType ? VectorType::get(vecType.getShape(), i32ScalarType)
                      : i32ScalarType;
    i8Type = vecType ? VectorType::get(vecType.getShape(), i8ScalarType)
                     : i8ScalarType;
    f32Type = vecType ? VectorType::get(vecType.getShape(), f32ScalarType)
                      : f32ScalarType;
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

  /// Creates an f32 constant, splatted if working with vectors.
  Value createF32Const(float value) {
    auto attr = rewriter.getF32FloatAttr(value);
    if (vecType) {
      auto splatAttr = SplatElementsAttr::get(cast<ShapedType>(f32Type), attr);
      return rewriter.createOrFold<arith::ConstantOp>(loc, f32Type, splatAttr);
    }
    return rewriter.createOrFold<arith::ConstantOp>(loc, f32Type, attr);
  }

  /// Extracts sign, exponent, and mantissa from an f32 value (as i32 bits).
  /// F32 format: 1 sign bit, 8 exponent bits, 23 mantissa bits.
  F32Fields extractF32Fields(Value i32Val) {
    Value cMantBits = createI32Const(kF32MantBits);
    Value cExpMask = createI32Const(kF32MaxExp);
    Value cMantMask = createI32Const(kF32MantMask);
    Value cSignShift = createI32Const(31);

    F32Fields fields;
    fields.sign = arith::ShRUIOp::create(rewriter, loc, i32Val, cSignShift);
    fields.biasedExp = arith::AndIOp::create(
        rewriter, loc, arith::ShRUIOp::create(rewriter, loc, i32Val, cMantBits),
        cExpMask);
    fields.mantissa = arith::AndIOp::create(rewriter, loc, i32Val, cMantMask);

    return fields;
  }

  /// Adds a bias to input for round-to-nearest-even before right-shifting.
  /// This matches runtime/src/iree/base/internal/math.h's bias_to_nearest_even:
  ///   even_bit = 1 << shift_amount
  ///   odd_bit = even_bit >> 1
  ///   bias = (input & even_bit) ? odd_bit : (odd_bit - 1)
  ///   return input + bias
  ///
  /// The caller should right-shift the result by shift_amount after this call.
  Value biasForRoundToNearestEven(Value input, Value shiftAmount) {
    Value c0 = createI32Const(0);
    Value c1 = createI32Const(1);
    Value evenBit = arith::ShLIOp::create(rewriter, loc, c1, shiftAmount);
    Value oddBit = arith::ShRUIOp::create(rewriter, loc, evenBit, c1);
    Value oddBitMinus1 = arith::SubIOp::create(rewriter, loc, oddBit, c1);
    Value hasEvenBit = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne,
        arith::AndIOp::create(rewriter, loc, input, evenBit), c0);
    Value bias = arith::SelectOp::create(rewriter, loc, hasEvenBit, oddBit,
                                         oddBitMinus1);
    return arith::AddIOp::create(rewriter, loc, input, bias);
  }

  Type getI32Type() const { return i32Type; }
  Type getI8Type() const { return i8Type; }
  Type getF32Type() const { return f32Type; }

private:
  RewriterBase &rewriter;
  Location loc;
  VectorType vecType;
  Type i32Type;
  Type i8Type;
  Type f32Type;
};

//===----------------------------------------------------------------------===//
// TruncF to small float emulation pattern
//===----------------------------------------------------------------------===//

/// Emulates arith.truncf from f32 to fp8 using integer bit manipulation.
/// This implementation follows IREE's runtime/src/iree/base/internal/math.h
/// (specifically iree_math_truncate_f32_to_bits_rounding_to_nearest_even).
///
/// Features:
///   - Round-to-nearest-even (IEEE 754 default rounding mode).
///   - Proper denormal/subnormal generation for underflow cases.
///   - Correct handling of all fp8 format variants (IEEE, FN, FNUZ).
///
/// The conversion handles three categories of fp8 formats:
///
/// 1. FNUZ types (f8E5M2FNUZ, f8E4M3FNUZ): No Inf, no negative zero.
///    - NaN is encoded as 0x80 (sign=1, exp=0, mantissa=0).
///    - Overflow produces NaN; zero is always positive.
///
/// 2. IEEE-like types (f8E5M2): Has Inf and negative zero.
///    - Standard IEEE-like encoding with Inf at max exponent.
///
/// 3. FN types (f8E4M3FN): No Inf, but has negative zero.
///    - Max exponent values are valid finite numbers (except NaN encoding).
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

    FloatTypeInfo dstInfo = getFloatTypeInfo(resultElemType);
    Location loc = op.getLoc();
    FloatEmulationHelper helper(rewriter, loc, resultType);

    // Constants for destination format.
    int dstMaxBiasedExp = (1 << dstInfo.expBits) - 1;
    // Max exponent for normal values (excludes Inf/NaN encoding if applicable).
    int dstMaxNormalBiasedExp =
        dstInfo.hasInf ? dstMaxBiasedExp - 1 : dstMaxBiasedExp;
    int mantShift = kF32MantBits - dstInfo.mantBits;

    // Create constants.
    Value c0 = helper.createI32Const(0);
    Value c1 = helper.createI32Const(1);
    Value cF32MantBits = helper.createI32Const(kF32MantBits);
    Value cF32MantMask = helper.createI32Const(kF32MantMask);
    Value cF32MaxExp = helper.createI32Const(kF32MaxExp);
    Value cF32Bias = helper.createI32Const(kF32Bias);
    Value cDstBias = helper.createI32Const(dstInfo.bias);
    Value cDstMantBits = helper.createI32Const(dstInfo.mantBits);
    Value cDstExpShift = cDstMantBits;
    Value cDstSignShift =
        helper.createI32Const(dstInfo.expBits + dstInfo.mantBits);
    Value cDstMantMask = helper.createI32Const(dstInfo.mantMask);
    Value cDstMaxNormalExp = helper.createI32Const(dstMaxNormalBiasedExp);
    Value cMantShift = helper.createI32Const(mantShift);
    Value cNaN = helper.createI32Const(dstInfo.nanEncoding);
    Value cDstSignMask = helper.createI32Const(dstInfo.signMask);

    // Bitcast f32 to i32 and extract fields.
    Value i32Val = arith::BitcastOp::create(rewriter, loc, helper.getI32Type(),
                                            op.getIn());
    F32Fields f32Fields = helper.extractF32Fields(i32Val);

    // Compute destination sign.
    Value dstSign =
        arith::ShLIOp::create(rewriter, loc, f32Fields.sign, cDstSignShift);

    // Check for NaN/Inf in source.
    Value srcIsNanOrInf =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                              f32Fields.biasedExp, cF32MaxExp);
    Value srcMantIsZero = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, f32Fields.mantissa, c0);
    Value srcIsInf =
        arith::AndIOp::create(rewriter, loc, srcIsNanOrInf, srcMantIsZero);
    // Derive srcIsNan from srcIsInf to avoid redundant comparison.
    // srcIsNan = srcIsNanOrInf && !srcMantIsZero = srcIsNanOrInf && !srcIsInf
    // Since srcIsNanOrInf is true when either is true, and srcIsInf requires
    // srcMantIsZero, we can XOR: srcIsNan = srcIsNanOrInf XOR srcIsInf
    Value srcIsNan =
        arith::XOrIOp::create(rewriter, loc, srcIsNanOrInf, srcIsInf);

    // Check for zero or subnormal in source (exp == 0).
    Value srcExpIsZero = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, f32Fields.biasedExp, c0);

    // Compute arithmetic exponent (unbiased).
    Value arithmeticExp =
        arith::SubIOp::create(rewriter, loc, f32Fields.biasedExp, cF32Bias);

    // Check overflow: arithmetic_exp > max_normal_exp - dst_bias + dst_bias
    // Simplified: biased_dst_exp > max_normal_exp
    Value biasedDstExp =
        arith::AddIOp::create(rewriter, loc, arithmeticExp, cDstBias);
    Value isOverflow =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                              biasedDstExp, cDstMaxNormalExp);

    // Check underflow: biased_dst_exp <= 0 means subnormal or zero.
    Value isUnderflow = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sle, biasedDstExp, c0);

    // Check if rounding caused mantissa overflow (carry into exponent).
    Value biasedMant =
        helper.biasForRoundToNearestEven(f32Fields.mantissa, cMantShift);
    Value mantOverflowed = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ugt, biasedMant, cF32MantMask);
    biasedDstExp = arith::SelectOp::create(
        rewriter, loc, mantOverflowed,
        arith::AddIOp::create(rewriter, loc, biasedDstExp, c1), biasedDstExp);
    biasedMant =
        arith::SelectOp::create(rewriter, loc, mantOverflowed, c0, biasedMant);

    // Re-check overflow after rounding increment.
    isOverflow = arith::OrIOp::create(
        rewriter, loc, isOverflow,
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                              biasedDstExp, cDstMaxNormalExp));

    // Shift mantissa to destination width.
    Value dstMant =
        arith::ShRUIOp::create(rewriter, loc, biasedMant, cMantShift);
    dstMant = arith::AndIOp::create(rewriter, loc, dstMant, cDstMantMask);

    // Pack normal result.
    Value dstExp =
        arith::ShLIOp::create(rewriter, loc, biasedDstExp, cDstExpShift);
    Value normalResult = arith::OrIOp::create(
        rewriter, loc, arith::OrIOp::create(rewriter, loc, dstSign, dstExp),
        dstMant);

    // Underflow case: generate subnormal or zero.
    // For subnormals, dst_exp = 0 and mantissa encodes the value.
    // shift_amount = f32_mant_bits - dst_mant_bits - arithmetic_exp + (1 -
    // dst_bias)
    Value dstSubnormalExp = helper.createI32Const(1 - dstInfo.bias);
    Value shiftAmount = arith::SubIOp::create(
        rewriter, loc,
        arith::SubIOp::create(rewriter, loc, cF32MantBits, cDstMantBits),
        arith::SubIOp::create(rewriter, loc, arithmeticExp, dstSubnormalExp));

    // Add implicit leading 1 to f32 mantissa for the shift.
    Value cImplicitBit = helper.createI32Const(1 << kF32MantBits);
    Value effectiveMant =
        arith::OrIOp::create(rewriter, loc, f32Fields.mantissa, cImplicitBit);

    // Compute round-to-nearest-even for subnormal.
    Value subnormalMantRounded =
        helper.biasForRoundToNearestEven(effectiveMant, shiftAmount);

    // Shift to get subnormal mantissa.
    Value subnormalMant = arith::ShRUIOp::create(
        rewriter, loc, subnormalMantRounded, shiftAmount);
    subnormalMant =
        arith::AndIOp::create(rewriter, loc, subnormalMant, cDstMantMask);

    // Subnormal result has exp=0.
    Value subnormalResult =
        arith::OrIOp::create(rewriter, loc, dstSign, subnormalMant);

    // Check if shift is too large (complete underflow to zero).
    Value shiftTooLarge = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sgt, shiftAmount,
        helper.createI32Const(kF32MantBits + 1));
    subnormalResult = arith::SelectOp::create(
        rewriter, loc, shiftTooLarge,
        dstInfo.nanAsNegZero ? c0 : dstSign, // Zero (signed if supported)
        subnormalResult);

    // Select cascade for final result.
    //
    // Unlike runtime code which uses early returns, SSA form requires computing
    // all paths and selecting between them. The ORDER of selects matters:
    // later selects override earlier ones. We order from lowest to highest
    // priority so the final select (NaN) always wins.
    //
    // Priority (lowest to highest):
    //   1. Normal/subnormal computation (base case)
    //   2. Source zero/subnormal -> zero
    //   3. Negative zero correction (FNUZ only, must be before NaN handling.)
    //   4. Overflow -> Inf or NaN
    //   5. Source Inf -> Inf or NaN
    //   6. Source NaN -> NaN (highest priority, always wins)
    Value result = arith::SelectOp::create(rewriter, loc, isUnderflow,
                                           subnormalResult, normalResult);

    // F32 subnormals (exp=0) become zero in fp8 (much smaller than fp8 min).
    Value zeroResult = dstInfo.nanAsNegZero ? c0 : dstSign;
    result = arith::SelectOp::create(rewriter, loc, srcExpIsZero, zeroResult,
                                     result);

    // FNUZ: Negative zero (0x80) must become positive zero (0x00).
    // CRITICAL: This must happen BEFORE NaN/Inf/overflow handling.
    // For FNUZ types, 0x80 is the NaN encoding, not negative zero.
    // If we did this after, we would incorrectly convert NaN to zero.
    if (dstInfo.nanAsNegZero) {
      Value resultIsNegZero = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, result, cDstSignMask);
      result =
          arith::SelectOp::create(rewriter, loc, resultIsNegZero, c0, result);
    }

    // Overflow and source Inf both map to the same result:
    // Inf (IEEE) or NaN (FN/FNUZ) or saturate to max finite (no Inf/NaN).
    Value infOrOverflowResult;
    if (dstInfo.hasInf) {
      infOrOverflowResult = arith::OrIOp::create(
          rewriter, loc, dstSign, helper.createI32Const(dstInfo.infEncoding));
    } else if (dstInfo.hasNan) {
      infOrOverflowResult = cNaN;
    } else {
      // No Inf, no NaN: saturate to max finite.
      infOrOverflowResult = arith::OrIOp::create(
          rewriter, loc, dstSign, helper.createI32Const(dstInfo.maxFinite));
    }
    result = arith::SelectOp::create(rewriter, loc, isOverflow,
                                     infOrOverflowResult, result);
    result = arith::SelectOp::create(rewriter, loc, srcIsInf,
                                     infOrOverflowResult, result);

    // Handle source NaN last so it takes precedence.
    Value nanResult;
    if (dstInfo.hasNan) {
      nanResult = cNaN;
    } else {
      nanResult = c0; // No NaN encoding: convert to +0.
    }
    result =
        arith::SelectOp::create(rewriter, loc, srcIsNan, nanResult, result);

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
/// This implementation follows IREE's runtime/src/iree/base/internal/math.h
/// (specifically iree_math_make_f32_from_bits).
///
/// For normal values: adjust exponent bias and shift mantissa.
///
/// For denormals (exp=0, mantissa!=0):
///   value = mantissa * 2^(1 - bias - mantissa_bits)
/// We implement this using uitofp + mulf with a precomputed scale factor,
/// which is simpler and more general than enumerating all possible values.
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
    Location loc = op.getLoc();
    FloatEmulationHelper helper(rewriter, loc, resultType);

    // Conversion parameters.
    int srcMaxExp = (1 << srcInfo.expBits) - 1; // All exp bits set.
    int biasDiff =
        kF32Bias - srcInfo.bias; // Bias adjustment for normal values.
    int mantShift =
        kF32MantBits - srcInfo.mantBits; // Left shift to align mantissa.

    Value c0 = helper.createI32Const(0);
    Value cBiasDiff = helper.createI32Const(biasDiff);
    Value cMantShift = helper.createI32Const(mantShift);
    Value cSrcMaxExp = helper.createI32Const(srcMaxExp);
    Value cF32NaN = helper.createI32Const(0x7FC00000); // Canonical quiet NaN.
    Value cF32Inf = helper.createI32Const(0x7F800000); // +Infinity.
    Value cF32MantBits = helper.createI32Const(kF32MantBits);

    // Bitcast fp8 to i8, extend to i32, and extract fields.
    Value i8Val =
        arith::BitcastOp::create(rewriter, loc, helper.getI8Type(), op.getIn());
    Value i32Val =
        arith::ExtUIOp::create(rewriter, loc, helper.getI32Type(), i8Val);

    // Extract fields.
    Value cSrcMantBits = helper.createI32Const(srcInfo.mantBits);
    Value cSrcExpMask = helper.createI32Const((1 << srcInfo.expBits) - 1);
    Value cSrcMantMask = helper.createI32Const(srcInfo.mantMask);
    Value cSrcSignShift =
        helper.createI32Const(srcInfo.expBits + srcInfo.mantBits);

    Value sign = arith::ShRUIOp::create(rewriter, loc, i32Val, cSrcSignShift);
    Value biasedExpSrc = arith::AndIOp::create(
        rewriter, loc,
        arith::ShRUIOp::create(rewriter, loc, i32Val, cSrcMantBits),
        cSrcExpMask);
    Value mantSrc = arith::AndIOp::create(rewriter, loc, i32Val, cSrcMantMask);

    // Compute f32 sign bit.
    Value f32Sign =
        arith::ShLIOp::create(rewriter, loc, sign, helper.createI32Const(31));

    // Precompute mantissa comparisons (used by multiple checks below).
    Value mantIsZero = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, mantSrc, c0);
    Value mantIsNonZero = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, mantSrc, c0);

    // Detect special values based on format type.
    Value isNaN;
    Value isInf; // Only used for IEEE types (hasInf=true).
    if (srcInfo.nanAsNegZero) {
      // FNUZ: NaN = 0x80 (sign=1, exp=0, mantissa=0), no Inf.
      Value cFNUZNaN = helper.createI32Const(srcInfo.nanEncoding);
      isNaN = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                    i32Val, cFNUZNaN);
    } else if (srcInfo.hasInf) {
      // IEEE types: NaN = exp==max && mantissa!=0, Inf = exp==max &&
      // mantissa==0.
      Value expIsMax = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, biasedExpSrc, cSrcMaxExp);
      isNaN = arith::AndIOp::create(rewriter, loc, expIsMax, mantIsNonZero);
      isInf = arith::AndIOp::create(rewriter, loc, expIsMax, mantIsZero);
    } else {
      // FN types: NaN only at specific encoding, no Inf.
      Value cNaNEncoding = helper.createI32Const(srcInfo.nanEncoding);
      isNaN = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                    i32Val, cNaNEncoding);
    }

    // Check for zero (exp=0, mantissa=0).
    Value expIsZero = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, biasedExpSrc, c0);
    Value isZero = arith::AndIOp::create(rewriter, loc, expIsZero, mantIsZero);

    // Check for denormal (exp=0, mantissa!=0).
    Value isDenormal =
        arith::AndIOp::create(rewriter, loc, expIsZero, mantIsNonZero);

    // Normal value conversion:
    // f32_exp = src_exp + bias_diff, f32_mant = src_mant << mant_shift
    Value normalF32Exp =
        arith::AddIOp::create(rewriter, loc, biasedExpSrc, cBiasDiff);
    Value normalF32Mant =
        arith::ShLIOp::create(rewriter, loc, mantSrc, cMantShift);
    Value normalResult = arith::OrIOp::create(
        rewriter, loc, f32Sign,
        arith::OrIOp::create(
            rewriter, loc,
            arith::ShLIOp::create(rewriter, loc, normalF32Exp, cF32MantBits),
            normalF32Mant));

    // For denormals: value = mantissa * 2^(1 - bias - mant_bits)
    // Use mulf with a precomputed scale factor instead of enumerating values.
    Value mantF32 =
        arith::UIToFPOp::create(rewriter, loc, helper.getF32Type(), mantSrc);
    float scaleValue = std::ldexp(1.0f, 1 - srcInfo.bias - srcInfo.mantBits);
    Value scale = helper.createF32Const(scaleValue);
    Value denormalF32 = arith::MulFOp::create(rewriter, loc, mantF32, scale);
    // Apply sign: bitcast to i32, OR with sign bit.
    Value denormalI32 = arith::BitcastOp::create(
        rewriter, loc, helper.getI32Type(), denormalF32);
    Value denormalResult =
        arith::OrIOp::create(rewriter, loc, f32Sign, denormalI32);

    // Select cascade for final result (same ordering logic as TruncF).
    // Later selects override earlier ones. NaN must be last (highest priority).
    Value result = arith::SelectOp::create(rewriter, loc, isDenormal,
                                           denormalResult, normalResult);

    // Zero: use signed zero if format supports negative zero, else +0.
    Value zeroResult = srcInfo.hasNegZero ? f32Sign : c0;
    result = arith::SelectOp::create(rewriter, loc, isZero, zeroResult, result);

    // Inf (IEEE types only): preserve sign.
    if (srcInfo.hasInf) {
      Value signedInf = arith::OrIOp::create(rewriter, loc, f32Sign, cF32Inf);
      result = arith::SelectOp::create(rewriter, loc, isInf, signedInf, result);
    }

    // NaN: always canonical quiet NaN (sign bit ignored).
    // Must be last to take precedence over zero for FNUZ (where 0x80 is NaN).
    result = arith::SelectOp::create(rewriter, loc, isNaN, cF32NaN, result);

    // Bitcast to f32.
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
  // For CPU backend, we emulate all the fp8 types to fp32. This supports any
  // future new fp8 types automatically.
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
