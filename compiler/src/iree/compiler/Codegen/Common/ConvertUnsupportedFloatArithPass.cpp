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
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-convert-unsupported-float-arith"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTUNSUPPORTEDFLOATARITHPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Detector for T::get(MLIRContext*) used by llvm::is_detected.
template <typename T>
using hasContextGet = decltype(T::get(std::declval<MLIRContext *>()));

/// Helpers to append types to a vector if they are small float types (fp4/fp8).
template <typename T>
static void maybeAppendSmallFloatType(MLIRContext *ctx,
                                      SmallVectorImpl<Type> &types) {
  if constexpr (llvm::is_detected<hasContextGet, T>::value) {
    Type t = T::get(ctx);
    if (isa<FloatType>(t)) {
      unsigned bitWidth = t.getIntOrFloatBitWidth();
      if (bitWidth == 4 || bitWidth == 8) {
        types.push_back(t);
      }
    }
  }
}

template <typename... Ts>
static void appendSmallFloatTypes(MLIRContext *ctx,
                                  SmallVectorImpl<Type> &types) {
  (maybeAppendSmallFloatType<Ts>(ctx, types), ...);
}

//===----------------------------------------------------------------------===//
// Helper for float emulation patterns
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
//   - NaN: exp=max, mantissa!=0 (IEEE), or sign bit only (FNUZ)

// F32 format constants (IEEE 754 binary32).
constexpr int kF32MantBits = 23;
constexpr int kF32Bias = 127;

/// Extracted components from an f32 value stored as i32 bits.
struct F32Fields {
  Value sign;      // Sign bit (0 or 1) shifted to bit 0.
  Value biasedExp; // Biased exponent (8 bits).
  Value mantissa;  // Mantissa (23 bits, no implicit leading 1).
};

/// Helper class for emulating small float (e.g., fp4, fp8) conversions to/from
/// f32 using integer bit manipulation. Handles both scalar and vector types.
///
/// Takes a small float type (scalar or vector), queries its semantics via
/// APFloat, and provides methods that return Value constants for the format
/// parameters.
class FloatEmulationHelper {
public:
  /// Constructor for use with small float types (fp4, fp8).
  /// The smallFloatBitWidth parameter determines the packed integer type.
  FloatEmulationHelper(RewriterBase &rewriter, Location loc, Type type,
                       unsigned smallFloatBitWidth)
      : rewriter(rewriter), loc(loc), vecType(dyn_cast<VectorType>(type)),
        sem(cast<FloatType>(getElementTypeOrSelf(type)).getFloatSemantics()) {
    // Setup scalar and vector types for i32, small int (i4/i8), f32.
    Type i32Scalar = rewriter.getI32Type();
    Type smallIntScalar = rewriter.getIntegerType(smallFloatBitWidth);
    Type f32Scalar = rewriter.getF32Type();
    i32Type =
        vecType ? VectorType::get(vecType.getShape(), i32Scalar) : i32Scalar;
    smallIntType = vecType ? VectorType::get(vecType.getShape(), smallIntScalar)
                           : smallIntScalar;
    f32Type =
        vecType ? VectorType::get(vecType.getShape(), f32Scalar) : f32Scalar;

    // Derive format parameters from APFloat semantics.
    smallMantBits = llvm::APFloat::semanticsPrecision(sem) - 1;
    int totalBits = llvm::APFloat::semanticsSizeInBits(sem);
    smallExpBits = totalBits - 1 - smallMantBits;
    smallBias = 1 - llvm::APFloat::semanticsMinExponent(sem);

    // Query format capabilities.
    smallHasInf = llvm::APFloat::semanticsHasInf(sem);
    smallHasNan = llvm::APFloat::semanticsHasNaN(sem);

    // Check for FNUZ types where NaN is encoded as sign bit only
    // (e.g., 0x80 for fp8, 0x8 for fp4). These types have no negative zero
    // and no infinity.
    if (llvm::APFloat::semanticsHasZero(sem)) {
      llvm::APFloat negZero = llvm::APFloat::getZero(sem, /*Negative=*/true);
      smallHasNegZero = negZero.isZero() && negZero.isNegative();
    } else {
      smallHasNegZero = false;
    }
    smallNanIsNegZero = !smallHasNegZero && !smallHasInf && smallHasNan;
  }

  //===--------------------------------------------------------------------===//
  // Generic constant creation
  //===--------------------------------------------------------------------===//

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

  //===--------------------------------------------------------------------===//
  // F32 format constants (as Value)
  //===--------------------------------------------------------------------===//

  Value getF32MantBitsConst() { return createI32Const(kF32MantBits); }
  Value getF32BiasConst() { return createI32Const(kF32Bias); }
  Value getF32MantMaskConst() {
    return createI32Const((1 << kF32MantBits) - 1);
  }
  Value getF32MaxExpConst() { return createI32Const(0xFF); }
  Value getF32ImplicitBitConst() { return createI32Const(1 << kF32MantBits); }
  Value getF32NaNConst() { return createI32Const(0x7FC00000); }
  Value getF32InfConst() { return createI32Const(0x7F800000); }

  //===--------------------------------------------------------------------===//
  // Small float format constants (as Value)
  //===--------------------------------------------------------------------===//

  Value getSmallMantBitsConst() { return createI32Const(smallMantBits); }
  Value getSmallBiasConst() { return createI32Const(smallBias); }
  Value getSmallSignShiftConst() {
    return createI32Const(smallExpBits + smallMantBits);
  }
  Value getSmallMantMaskConst() {
    return createI32Const((1u << smallMantBits) - 1);
  }
  Value getSmallExpMaskConst() {
    return createI32Const(((1u << smallExpBits) - 1) << smallMantBits);
  }
  Value getSmallSignMaskConst() {
    return createI32Const(1u << (smallExpBits + smallMantBits));
  }
  Value getSmallMaxExpConst() {
    return createI32Const((1 << smallExpBits) - 1);
  }
  Value getSmallMaxNormalExpConst() {
    int maxExp = (1 << smallExpBits) - 1;
    return createI32Const(smallHasInf ? maxExp - 1 : maxExp);
  }

  /// Returns the mantissa shift between f32 and small float.
  Value getMantShiftConst() {
    return createI32Const(kF32MantBits - smallMantBits);
  }

  /// Returns the bias difference (f32_bias - small_bias).
  Value getBiasDiffConst() { return createI32Const(kF32Bias - smallBias); }

  /// Returns the subnormal exponent constant (1 - bias).
  Value getSubnormalExpConst() { return createI32Const(1 - smallBias); }

  /// Returns the NaN encoding for the small float type.
  Value getNaNEncodingConst() {
    if (smallNanIsNegZero) {
      // FNUZ types: NaN = sign bit only (e.g., 0x80 for fp8, 0x8 for fp4).
      return getSmallSignMaskConst();
    }
    // IEEE and FN types: NaN = all exp bits + some mantissa bits.
    unsigned expMask = ((1u << smallExpBits) - 1) << smallMantBits;
    unsigned mantMask = (1u << smallMantBits) - 1;
    return createI32Const(expMask | mantMask);
  }

  /// Returns the Inf encoding for the small float type (0 if no Inf support).
  Value getInfEncodingConst() {
    if (!smallHasInf) {
      return createI32Const(0);
    }
    return getSmallExpMaskConst();
  }

  /// Returns the max finite value encoding for the small float type.
  Value getMaxFiniteConst() {
    unsigned maxExpCode = (1u << smallExpBits) - 1;
    unsigned mantMask = (1u << smallMantBits) - 1;
    unsigned expMask = ((1u << smallExpBits) - 1) << smallMantBits;

    if (smallHasInf) {
      return createI32Const(((maxExpCode - 1) << smallMantBits) | mantMask);
    }
    if (smallHasNan && !smallNanIsNegZero) {
      // FN types: max exp is valid but max mantissa is NaN.
      return createI32Const(expMask | (mantMask - 1));
    }
    // FNUZ or no-NaN types: all bit patterns except NaN are valid.
    return createI32Const(expMask | mantMask);
  }

  /// Returns the denormal scale factor for extf: 2^(1 - bias - mantBits).
  Value getDenormalScaleConst() {
    float scale = std::ldexp(1.0f, 1 - smallBias - smallMantBits);
    return createF32Const(scale);
  }

  //===--------------------------------------------------------------------===//
  // Format capability queries
  //===--------------------------------------------------------------------===//

  bool hasInf() const { return smallHasInf; }
  bool hasNan() const { return smallHasNan; }
  bool hasNegZero() const { return smallHasNegZero; }
  bool isNanEncodedAsNegZero() const { return smallNanIsNegZero; }

  //===--------------------------------------------------------------------===//
  // F32 field extraction
  //===--------------------------------------------------------------------===//

  /// Extracts sign, exponent, and mantissa from an f32 value (as i32 bits).
  F32Fields extractF32Fields(Value i32Val) {
    Value cMantBits = getF32MantBitsConst();
    Value cExpMask = getF32MaxExpConst();
    Value cMantMask = getF32MantMaskConst();
    Value cSignShift = createI32Const(31);

    F32Fields fields;
    fields.sign = arith::ShRUIOp::create(rewriter, loc, i32Val, cSignShift);
    fields.biasedExp = arith::AndIOp::create(
        rewriter, loc, arith::ShRUIOp::create(rewriter, loc, i32Val, cMantBits),
        cExpMask);
    fields.mantissa = arith::AndIOp::create(rewriter, loc, i32Val, cMantMask);

    return fields;
  }

  //===--------------------------------------------------------------------===//
  // Rounding support
  //===--------------------------------------------------------------------===//

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

  //===--------------------------------------------------------------------===//
  // Type accessors
  //===--------------------------------------------------------------------===//

  Type getI32Type() const { return i32Type; }
  Type getSmallIntType() const { return smallIntType; }
  Type getF32Type() const { return f32Type; }

private:
  RewriterBase &rewriter;
  Location loc;
  VectorType vecType;
  const llvm::fltSemantics &sem;

  // Derived types for scalar/vector operations.
  Type i32Type;
  Type smallIntType;
  Type f32Type;

  // Small float format parameters (derived from semantics).
  int smallExpBits;
  int smallMantBits;
  int smallBias;
  bool smallHasInf;
  bool smallHasNan;
  bool smallHasNegZero;
  bool smallNanIsNegZero;
};

//===----------------------------------------------------------------------===//
// TruncF to small float emulation pattern
//===----------------------------------------------------------------------===//

/// Emulates arith.truncf from f32 to small floats (fp4, fp8) using integer bit
/// manipulation. This implementation follows IREE's
/// runtime/src/iree/base/internal/math.h
/// (specifically iree_math_truncate_f32_to_bits_rounding_to_nearest_even).
///
/// Features:
///   - Round-to-nearest-even (IEEE 754 default rounding mode).
///   - Proper denormal/subnormal generation for underflow cases.
///   - Correct handling of all format variants (IEEE, FN, FNUZ).
///
/// Supported format categories:
///
/// 1. FNUZ types (f8E5M2FNUZ, f8E4M3FNUZ): No Inf, no negative zero.
///    - NaN is encoded as 0x80 (sign=1, exp=0, mantissa=0).
///    - Overflow produces NaN; zero is always positive.
///
/// 2. IEEE types (f8E5M2): Has Inf and negative zero.
///    - Standard IEEE-like encoding with Inf at max exponent.
///
/// 3. FN types (f8E4M3FN, f4E2M1FN): No Inf, but may have negative zero.
///    - Max exponent values are valid finite numbers (except NaN encoding).
///    - f4E2M1FN has no NaN and no negative zero.
struct TruncFToSmallFloat final : OpRewritePattern<arith::TruncFOp> {
  TruncFToSmallFloat(MLIRContext *ctx, ArrayRef<Type> sourceTypes)
      : OpRewritePattern(ctx),
        sourceTypes(sourceTypes.begin(), sourceTypes.end()) {}

  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();
    Type resultElemType = getElementTypeOrSelf(resultType);
    Type inputElemType = getElementTypeOrSelf(op.getIn().getType());

    unsigned resultBitWidth = resultElemType.getIntOrFloatBitWidth();
    if (!isa<Float32Type>(inputElemType) ||
        (resultBitWidth != 4 && resultBitWidth != 8)) {
      return failure();
    }

    // Only match types that are in our source types list.
    if (!llvm::is_contained(sourceTypes, resultElemType)) {
      return failure();
    }

    Location loc = op.getLoc();
    FloatEmulationHelper helper(rewriter, loc, resultType, resultBitWidth);

    // Get constants from helper.
    Value c0 = helper.createI32Const(0);
    Value c1 = helper.createI32Const(1);
    Value cF32MantBits = helper.getF32MantBitsConst();
    Value cF32MantMask = helper.getF32MantMaskConst();
    Value cF32MaxExp = helper.getF32MaxExpConst();
    Value cF32Bias = helper.getF32BiasConst();
    Value cDstBias = helper.getSmallBiasConst();
    Value cDstMantBits = helper.getSmallMantBitsConst();
    Value cDstExpShift = cDstMantBits;
    Value cDstSignShift = helper.getSmallSignShiftConst();
    Value cDstMantMask = helper.getSmallMantMaskConst();
    Value cDstMaxNormalExp = helper.getSmallMaxNormalExpConst();
    Value cMantShift = helper.getMantShiftConst();
    Value cNaN = helper.getNaNEncodingConst();
    Value cDstSignMask = helper.getSmallSignMaskConst();

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
    // srcIsNan = srcIsNanOrInf XOR srcIsInf
    Value srcIsNan =
        arith::XOrIOp::create(rewriter, loc, srcIsNanOrInf, srcIsInf);

    // Check for zero or subnormal in source (exp == 0).
    Value srcExpIsZero = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, f32Fields.biasedExp, c0);

    // Compute arithmetic exponent (unbiased).
    Value arithmeticExp =
        arith::SubIOp::create(rewriter, loc, f32Fields.biasedExp, cF32Bias);

    // Check overflow: biased_dst_exp > max_normal_exp.
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
    // shift_amount = f32_mant_bits - dst_mant_bits - arithmetic_exp + (1 -
    // dst_bias)
    Value dstSubnormalExp = helper.getSubnormalExpConst();
    Value shiftAmount = arith::SubIOp::create(
        rewriter, loc,
        arith::SubIOp::create(rewriter, loc, cF32MantBits, cDstMantBits),
        arith::SubIOp::create(rewriter, loc, arithmeticExp, dstSubnormalExp));

    // Add implicit leading 1 to f32 mantissa for the shift.
    Value cImplicitBit = helper.getF32ImplicitBitConst();
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
    // Zero (signed if supported).
    Value zeroValue = helper.isNanEncodedAsNegZero() ? c0 : dstSign;
    subnormalResult = arith::SelectOp::create(rewriter, loc, shiftTooLarge,
                                              zeroValue, subnormalResult);

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

    // F32 subnormals (exp=0) become zero in small floats (much smaller than
    // min).
    Value zeroResult = helper.isNanEncodedAsNegZero() ? c0 : dstSign;
    result = arith::SelectOp::create(rewriter, loc, srcExpIsZero, zeroResult,
                                     result);

    // FNUZ: Negative zero (sign bit only, e.g., 0x80 for fp8, 0x8 for fp4)
    // must become positive zero. CRITICAL: This must happen BEFORE
    // NaN/Inf/overflow handling. For FNUZ types, the sign-bit-only pattern
    // is the NaN encoding, not negative zero. If we did this after, we would
    // incorrectly convert NaN to zero.
    if (helper.isNanEncodedAsNegZero()) {
      Value resultIsNegZero = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, result, cDstSignMask);
      result =
          arith::SelectOp::create(rewriter, loc, resultIsNegZero, c0, result);
    }

    // Overflow and source Inf both map to the same result:
    // Inf (IEEE) or NaN (FN/FNUZ) or saturate to max finite (no Inf/NaN).
    Value infOrOverflowResult;
    if (helper.hasInf()) {
      infOrOverflowResult = arith::OrIOp::create(rewriter, loc, dstSign,
                                                 helper.getInfEncodingConst());
    } else if (helper.hasNan()) {
      infOrOverflowResult = cNaN;
    } else {
      // No Inf, no NaN: saturate to max finite.
      infOrOverflowResult = arith::OrIOp::create(rewriter, loc, dstSign,
                                                 helper.getMaxFiniteConst());
    }
    result = arith::SelectOp::create(rewriter, loc, isOverflow,
                                     infOrOverflowResult, result);
    result = arith::SelectOp::create(rewriter, loc, srcIsInf,
                                     infOrOverflowResult, result);

    // Handle source NaN last so it takes precedence.
    Value nanResult = helper.hasNan() ? cNaN : c0;
    result =
        arith::SelectOp::create(rewriter, loc, srcIsNan, nanResult, result);

    // Truncate to small int type and bitcast to small float.
    result = arith::TruncIOp::create(rewriter, loc, helper.getSmallIntType(),
                                     result);
    result = arith::BitcastOp::create(rewriter, loc, resultType, result);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  SmallVector<Type> sourceTypes;
};

//===----------------------------------------------------------------------===//
// ExtF from small float emulation pattern
//===----------------------------------------------------------------------===//

/// Emulates arith.extf from small floats (fp4, fp8) to f32 using integer bit
/// manipulation. This implementation follows IREE's
/// runtime/src/iree/base/internal/math.h
/// (specifically iree_math_make_f32_from_bits).
///
/// For normal values: adjust exponent bias and shift mantissa.
///
/// For denormals (exp=0, mantissa!=0):
///   value = mantissa * 2^(1 - bias - mantissa_bits)
/// We implement this using uitofp + mulf with a precomputed scale factor,
/// which is simpler and more general than enumerating all possible values.
struct ExtFFromSmallFloat final : OpRewritePattern<arith::ExtFOp> {
  ExtFFromSmallFloat(MLIRContext *ctx, ArrayRef<Type> sourceTypes)
      : OpRewritePattern(ctx),
        sourceTypes(sourceTypes.begin(), sourceTypes.end()) {}

  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();
    Type inputType = op.getIn().getType();
    Type resultElemType = getElementTypeOrSelf(resultType);
    Type inputElemType = getElementTypeOrSelf(inputType);

    unsigned inputBitWidth = inputElemType.getIntOrFloatBitWidth();
    if ((inputBitWidth != 4 && inputBitWidth != 8) ||
        !isa<Float32Type>(resultElemType)) {
      return failure();
    }

    // Only match types that are in our source types list.
    if (!llvm::is_contained(sourceTypes, inputElemType)) {
      return failure();
    }

    Location loc = op.getLoc();
    FloatEmulationHelper helper(rewriter, loc, inputType, inputBitWidth);

    // Get constants from helper.
    Value c0 = helper.createI32Const(0);
    Value cBiasDiff = helper.getBiasDiffConst();
    Value cMantShift = helper.getMantShiftConst();
    Value cSrcMaxExp = helper.getSmallMaxExpConst();
    Value cF32NaN = helper.getF32NaNConst();
    Value cF32Inf = helper.getF32InfConst();
    Value cF32MantBits = helper.getF32MantBitsConst();

    // Bitcast small float to small int, extend to i32, and extract fields.
    Value smallIntVal = arith::BitcastOp::create(
        rewriter, loc, helper.getSmallIntType(), op.getIn());
    Value i32Val =
        arith::ExtUIOp::create(rewriter, loc, helper.getI32Type(), smallIntVal);

    // Extract fields from small float.
    Value cSrcMantBits = helper.getSmallMantBitsConst();
    Value cSrcExpMask = helper.getSmallMaxExpConst();
    Value cSrcMantMask = helper.getSmallMantMaskConst();
    Value cSrcSignShift = helper.getSmallSignShiftConst();

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
    if (!helper.hasNan()) {
      // Types without NaN (e.g., f4E2M1FN): isNaN is always false.
      // We use a constant false comparison that will be optimized away.
      isNaN = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, c0,
                                    c0);
    } else if (helper.isNanEncodedAsNegZero()) {
      // FNUZ: NaN = sign bit only (sign=1, exp=0, mantissa=0), no Inf.
      Value cFNUZNaN = helper.getNaNEncodingConst();
      isNaN = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                    i32Val, cFNUZNaN);
    } else if (helper.hasInf()) {
      // IEEE types: NaN = exp==max && mantissa!=0, Inf = exp==max &&
      // mantissa==0.
      Value expIsMax = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, biasedExpSrc, cSrcMaxExp);
      isNaN = arith::AndIOp::create(rewriter, loc, expIsMax, mantIsNonZero);
      isInf = arith::AndIOp::create(rewriter, loc, expIsMax, mantIsZero);
    } else {
      // FN types with NaN: NaN only at specific encoding, no Inf.
      Value cNaNEncoding = helper.getNaNEncodingConst();
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
    Value scale = helper.getDenormalScaleConst();
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
    Value zeroResult = helper.hasNegZero() ? f32Sign : c0;
    result = arith::SelectOp::create(rewriter, loc, isZero, zeroResult, result);

    // Inf (IEEE types only): preserve sign.
    if (helper.hasInf()) {
      Value signedInf = arith::OrIOp::create(rewriter, loc, f32Sign, cF32Inf);
      result = arith::SelectOp::create(rewriter, loc, isInf, signedInf, result);
    }

    // NaN: always canonical quiet NaN (sign bit ignored).
    // Must be last to take precedence over zero for FNUZ (where sign bit is
    // NaN).
    result = arith::SelectOp::create(rewriter, loc, isNaN, cF32NaN, result);

    // Bitcast to f32.
    result = arith::BitcastOp::create(rewriter, loc, resultType, result);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  SmallVector<Type> sourceTypes;
};

struct ConvertUnsupportedFloatArithPass final
    : public impl::ConvertUnsupportedFloatArithPassBase<
          ConvertUnsupportedFloatArithPass> {
  void runOnOperation() override;
  using Base::Base;
};

/// Returns the types that need extf/truncf emulation (bit manipulation) for
/// the given GPU target. Types with hardware conversion instructions are
/// excluded since ArithToAMDGPU patterns in ConvertToROCDL handle those.
///
/// Note: ALL small float types need arithmetic emulation (wrapping addf/mulf
/// with extf/truncf) because no GPU has native fp4/fp8 arithmetic instructions.
/// This function only determines which types need SOFTWARE conversion.
static SmallVector<Type>
getTypesNeedingConversionEmulationForGPU(MLIRContext *context,
                                         IREE::GPU::TargetAttr gpuAttr) {
  SmallVector<Type> types;
  appendSmallFloatTypes<
#define GET_TYPEDEF_LIST
#include "mlir/IR/BuiltinTypes.cpp.inc"
      >(context, types);

  // Remove types that have hardware conversion support on this chip.
  StringRef chipset = gpuAttr.getArch();
  FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
  if (failed(maybeChipset)) {
    LDBG() << "Invalid chip name";
    return types;
  }
  constexpr amdgpu::Chipset kGfx942{9, 4, 2};
  constexpr amdgpu::Chipset kGfx950{9, 5, 0};
  if (*maybeChipset >= kGfx942 && *maybeChipset < kGfx950) {
    // gfx942 has hardware conversion for FNUZ types.
    llvm::erase(types, Float8E4M3FNUZType::get(context));
    llvm::erase(types, Float8E5M2FNUZType::get(context));
  }
  if (amdgpu::hasOcpFp8(*maybeChipset)) {
    // gfx950+ and gfx12+ have hardware conversion for OCP types.
    llvm::erase(types, Float8E4M3FNType::get(context));
    llvm::erase(types, Float8E5M2Type::get(context));
    llvm::erase(types, Float4E2M1FNType::get(context));
  }
  return types;
}

} // namespace

void ConvertUnsupportedFloatArithPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FunctionOpInterface funcOp = getOperation();
  Type targetType = Float32Type::get(context);

  // All small float types (fp4, fp8) need arithmetic emulation unless the
  // hardware has native arithmetic instructions for these types. Operations
  // like addf/mulf usually have to be wrapped with extf/truncf to compute in
  // f32.
  SmallVector<Type> allSmallFloatTypes;
  appendSmallFloatTypes<
#define GET_TYPEDEF_LIST
#include "mlir/IR/BuiltinTypes.cpp.inc"
      >(context, allSmallFloatTypes);

  // Apply the standard float emulation patterns. This inserts extf/truncf pairs
  // around unsupported float operations.
  {
    TypeConverter converter;
    arith::populateEmulateUnsupportedFloatsConversions(
        converter, allSmallFloatTypes, targetType);
    RewritePatternSet patterns(context);
    arith::populateEmulateUnsupportedFloatsPatterns(patterns, converter);
    ConversionTarget target(*context);
    arith::populateEmulateUnsupportedFloatsLegality(target, converter);

    // Mark scaling ops as legal - they have their own expansion patterns in
    // arith::populateExpandScalingExtTruncPatterns that run in later passes.
    // We don't want to insert extf/truncf pairs around them.
    target.addLegalOp<arith::ScalingExtFOp, arith::ScalingTruncFOp>();

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // Determine which types need software conversion emulation. By default, all
  // the small float types need the emulation. For GPU targets, some types may
  // have hardware conversion support and can be skipped.
  SmallVector<Type> typesNeedingConversionEmulation = allSmallFloatTypes;
  if (auto gpuAttr = getGPUTargetAttr(funcOp)) {
    typesNeedingConversionEmulation =
        getTypesNeedingConversionEmulationForGPU(context, gpuAttr);
  }

  // Emulate extf/truncf to/from small float types using integer bit
  // manipulation. Only for types without hardware conversion support.
  // This is gated by the enableExtTruncEmulation flag.
  if (enableExtTruncEmulation && !typesNeedingConversionEmulation.empty()) {
    RewritePatternSet emulationPatterns(context);
    emulationPatterns.add<TruncFToSmallFloat, ExtFFromSmallFloat>(
        context, typesNeedingConversionEmulation);
    walkAndApplyPatterns(funcOp, std::move(emulationPatterns));
  }
}

} // namespace mlir::iree_compiler
