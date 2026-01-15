// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_MAP_STABLEHLO_TO_SCALAR_OP_H
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_MAP_STABLEHLO_TO_SCALAR_OP_H

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {
namespace impl {

// A struct to map StableHloBinaryOpTy type to the corresponding floating-point
// and integer scalar operation types.
template <typename StableHloBinaryOpTy>
struct StableHloToScalarOp {
  using FOp = void;
  using IOp = void;
  using UOp = void;
  using COp = void;
};

template <>
struct StableHloToScalarOp<stablehlo::AddOp> {
  using FOp = ::mlir::arith::AddFOp;
  using IOp = ::mlir::arith::AddIOp;
  using UOp = ::mlir::arith::AddIOp;
  using COp = ::mlir::complex::AddOp;
};
template <>
struct StableHloToScalarOp<stablehlo::AndOp> {
  using IOp = ::mlir::arith::AndIOp;
  using UOp = ::mlir::arith::AndIOp;
};
template <>
struct StableHloToScalarOp<stablehlo::CbrtOp> {
  using FOp = ::mlir::math::CbrtOp;
};
template <>
struct StableHloToScalarOp<stablehlo::CompareOp> {
  using FOp = ::mlir::arith::CmpFOp;
  using IOp = ::mlir::arith::CmpIOp;
  using UOp = ::mlir::arith::CmpIOp;
};
template <>
struct StableHloToScalarOp<stablehlo::CeilOp> {
  using FOp = ::mlir::math::CeilOp;
};
template <>
struct StableHloToScalarOp<stablehlo::ClzOp> {
  using IOp = ::mlir::math::CountLeadingZerosOp;
  using UOp = ::mlir::math::CountLeadingZerosOp;
};
template <>
struct StableHloToScalarOp<stablehlo::CosineOp> {
  using FOp = ::mlir::math::CosOp;
  using COp = ::mlir::complex::CosOp;
};
template <>
struct StableHloToScalarOp<stablehlo::ExpOp> {
  using FOp = ::mlir::math::ExpOp;
  using COp = ::mlir::complex::ExpOp;
};
template <>
struct StableHloToScalarOp<stablehlo::Expm1Op> {
  using FOp = ::mlir::math::ExpM1Op;
  using COp = ::mlir::complex::Expm1Op;
};
template <>
struct StableHloToScalarOp<stablehlo::FloorOp> {
  using FOp = ::mlir::math::FloorOp;
};
template <>
struct StableHloToScalarOp<stablehlo::LogOp> {
  using FOp = ::mlir::math::LogOp;
  using COp = ::mlir::complex::LogOp;
};
template <>
struct StableHloToScalarOp<stablehlo::Log1pOp> {
  using FOp = ::mlir::math::Log1pOp;
  using COp = ::mlir::complex::Log1pOp;
};
template <>
struct StableHloToScalarOp<stablehlo::MulOp> {
  using FOp = ::mlir::arith::MulFOp;
  using IOp = ::mlir::arith::MulIOp;
  using UOp = ::mlir::arith::MulIOp;
  using COp = ::mlir::complex::MulOp;
};
template <>
struct StableHloToScalarOp<stablehlo::OrOp> {
  using IOp = ::mlir::arith::OrIOp;
  using UOp = ::mlir::arith::OrIOp;
};
template <>
struct StableHloToScalarOp<stablehlo::PopulationCountOp> {
  using IOp = ::mlir::math::CtPopOp;
  using UOp = ::mlir::math::CtPopOp;
};
template <>
struct StableHloToScalarOp<stablehlo::RsqrtOp> {
  using FOp = ::mlir::math::RsqrtOp;
  using COp = ::mlir::complex::RsqrtOp;
};
template <>
struct StableHloToScalarOp<stablehlo::RoundNearestEvenOp> {
  using FOp = ::mlir::math::RoundEvenOp;
};
template <>
struct StableHloToScalarOp<stablehlo::RoundOp> {
  using FOp = ::mlir::math::RoundOp;
};
template <>
struct StableHloToScalarOp<stablehlo::SubtractOp> {
  using FOp = ::mlir::arith::SubFOp;
  using IOp = ::mlir::arith::SubIOp;
  using UOp = ::mlir::arith::SubIOp;
  using COp = ::mlir::complex::SubOp;
};
template <>
struct StableHloToScalarOp<stablehlo::SqrtOp> {
  using FOp = ::mlir::math::SqrtOp;
  using COp = ::mlir::complex::SqrtOp;
};
template <>
struct StableHloToScalarOp<stablehlo::SineOp> {
  using FOp = ::mlir::math::SinOp;
  using COp = ::mlir::complex::SinOp;
};
// FIXME(Jakub)
/*
template <>
struct StableHloToScalarOp<stablehlo::TanOp> {
  using FOp = ::mlir::math::TanOp;
  using COp = ::mlir::complex::TanOp;
};
*/
template <>
struct StableHloToScalarOp<stablehlo::Atan2Op> {
  using FOp = ::mlir::math::Atan2Op;
  using COp = ::mlir::complex::Atan2Op;
};
template <>
struct StableHloToScalarOp<stablehlo::TanhOp> {
  using FOp = ::mlir::math::TanhOp;
  using COp = ::mlir::complex::TanhOp;
};
template <>
struct StableHloToScalarOp<stablehlo::XorOp> {
  using IOp = ::mlir::arith::XOrIOp;
  using UOp = ::mlir::arith::XOrIOp;
};

// Alias for the map from StableHLO binary op type to STD floating-point op
// type.
template <typename StableHloOp>
using ScalarFOp = typename StableHloToScalarOp<StableHloOp>::FOp;
// Alias for the map from StableHLO binary op type to STD signed integer op
// type.
template <typename StableHloOp>
using ScalarIOp = typename StableHloToScalarOp<StableHloOp>::IOp;
// Alias for the map from StableHLO binary op type to STD unsigned integer op
// type.
template <typename StableHloOp>
using ScalarUOp = typename StableHloToScalarOp<StableHloOp>::UOp;
// Alias for the map from StableHLO binary op type to STD complex op type.
template <typename StableHloOp>
using ScalarCOp = typename StableHloToScalarOp<StableHloOp>::COp;

template <typename... Args>
struct MapStableHloOpToScalarOpImpl {
  Value operator()(Location /*loc*/, ArrayRef<Type> /*ResultTypes*/,
                   ArrayRef<Type> /*argTypes*/, ValueRange /*args*/,
                   OpBuilder * /*b*/) {
    return nullptr;
  }
};

template <typename StdScalarOp>
struct MapStableHloOpToScalarOpImpl<StdScalarOp> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> /*argTypes*/, ValueRange args, OpBuilder *b) {
    return StdScalarOp::create(*b, loc, resultTypes, args,
                               ArrayRef<NamedAttribute>{});
  }
};

template <typename SupportedType, typename StdScalarOp, typename... Args>
struct MapStableHloOpToScalarOpImpl<SupportedType, StdScalarOp, Args...> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args, OpBuilder *b) {
    Type elementType = getElementTypeOrSelf(argTypes.front());
    if (SupportedType{}(elementType)) {
      return StdScalarOp::create(*b, loc, resultTypes, args,
                                 ArrayRef<NamedAttribute>{});
    }
    return MapStableHloOpToScalarOpImpl<Args...>{}(loc, resultTypes, argTypes,
                                                   args, b);
  }
};

template <typename SupportedType, typename... Args>
struct MapStableHloOpToScalarOpImpl<SupportedType, void, Args...> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args, OpBuilder *b) {
    return MapStableHloOpToScalarOpImpl<Args...>{}(loc, resultTypes, argTypes,
                                                   args, b);
  }
};

struct IsAnyIntegerType {
  bool operator()(Type t) { return isa<IntegerType>(t); }
};

struct IsSignedIntegerType {
  bool operator()(Type t) {
    // Pretend that signless is signed. This will change eventually.
    return isa<IntegerType>(t) && !t.isUnsignedInteger() &&
           !t.isSignlessInteger(1);
  }
};

struct IsUnsignedIntegerType {
  bool operator()(Type t) {
    return t.isUnsignedInteger() || t.isSignlessInteger(1);
  }
};

struct IsFloatType {
  bool operator()(Type t) { return isa<FloatType>(t); }
};

struct IsComplexType {
  bool operator()(Type t) { return isa<ComplexType>(t); }
};

template <template <typename T> class MapTy, typename OpTy,
          typename PredTy = llvm::is_detected<MapTy, OpTy>>
struct MapableIf {
  using type = void;
};
template <template <typename T> class MapTy, typename OpTy>
struct MapableIf<MapTy, OpTy, std::true_type> {
  using type = MapTy<OpTy>;
};

// Inserts the computation that corresponds to the body of the loop for lowered
// StableHLO unary/binary op. Returns the value for the result.
template <typename StableHloOpTy>
inline Value mapStableHloOpToStdScalarOp(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    typename StableHloOpTy::Adaptor adaptor, OpBuilder *b) {
  using ScalarIOpOrVoid = typename MapableIf<ScalarIOp, StableHloOpTy>::type;
  using ScalarUOpOrVoid = typename MapableIf<ScalarUOp, StableHloOpTy>::type;
  using ScalarFOpOrVoid = typename MapableIf<ScalarFOp, StableHloOpTy>::type;
  using ScalarCOpOrVoid = typename MapableIf<ScalarCOp, StableHloOpTy>::type;
  return MapStableHloOpToScalarOpImpl<IsSignedIntegerType, ScalarIOpOrVoid,
                                      IsUnsignedIntegerType, ScalarUOpOrVoid,
                                      IsFloatType, ScalarFOpOrVoid,
                                      IsComplexType, ScalarCOpOrVoid>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), b);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::AbsOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::AbsOp::Adaptor adaptor, OpBuilder *b) {
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (isa<FloatType>(elementType)) {
    return MapStableHloOpToScalarOpImpl<IsFloatType, ::mlir::math::AbsFOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);
  }
  if (isa<ComplexType>(elementType)) {
    return MapStableHloOpToScalarOpImpl<IsComplexType,
                                        ::mlir::complex::AbsOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);
  }
  if (elementType.isSignlessInteger() || elementType.isSignedInteger()) {
    // lmhlo.abs(x, result) ->  result = select((x > 0), x, sub(0, x))
    Value lhs = adaptor.getOperand();
    Value zeroIntval =
        arith::ConstantOp::create(*b, loc, b->getZeroAttr(lhs.getType()));
    auto lhsGtZero = ScalarIOp<CompareOp>::create(
        *b, loc, arith::CmpIPredicate::sge, lhs, zeroIntval);
    auto negVal =
        ScalarIOp<stablehlo::SubtractOp>::create(*b, loc, zeroIntval, lhs);
    return mlir::arith::SelectOp::create(*b, loc, lhsGtZero, lhs, negVal);
  }
  return nullptr;
}

// Return a constant for v of type t, splat if t is a vector type.
inline Value getConstantOrSplat(OpBuilder *b, Location loc, Type t,
                                Attribute v) {
  if (VectorType vecType = dyn_cast<VectorType>(t)) {
    v = SplatElementsAttr::get(vecType, v);
  }
  return arith::ConstantOp::create(*b, loc, t, cast<TypedAttr>(v));
}

template <typename PredicateType>
inline std::optional<PredicateType>
getCmpPredicate(stablehlo::ComparisonDirection, bool) {
  return std::nullopt;
}

template <>
inline std::optional<arith::CmpFPredicate>
getCmpPredicate<arith::CmpFPredicate>(
    stablehlo::ComparisonDirection comparisonDirection, bool isSigned) {
  assert(isSigned && "cannot have an unsigned float!");
  return llvm::StringSwitch<std::optional<arith::CmpFPredicate>>(
             stringifyComparisonDirection(comparisonDirection))
      .Case("EQ", arith::CmpFPredicate::OEQ)
      .Case("NE", arith::CmpFPredicate::UNE)
      .Case("GE", arith::CmpFPredicate::OGE)
      .Case("GT", arith::CmpFPredicate::OGT)
      .Case("LE", arith::CmpFPredicate::OLE)
      .Case("LT", arith::CmpFPredicate::OLT)
      .Default(std::nullopt);
}

template <>
inline std::optional<arith::CmpIPredicate>
getCmpPredicate<arith::CmpIPredicate>(
    stablehlo::ComparisonDirection comparisonDirection, bool isSigned) {
  return llvm::StringSwitch<std::optional<arith::CmpIPredicate>>(
             stringifyComparisonDirection(comparisonDirection))
      .Case("EQ", arith::CmpIPredicate::eq)
      .Case("NE", arith::CmpIPredicate::ne)
      .Case("GE",
            isSigned ? arith::CmpIPredicate::sge : arith::CmpIPredicate::uge)
      .Case("GT",
            isSigned ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ugt)
      .Case("LE",
            isSigned ? arith::CmpIPredicate::sle : arith::CmpIPredicate::ule)
      .Case("LT",
            isSigned ? arith::CmpIPredicate::slt : arith::CmpIPredicate::ult)
      .Default(std::nullopt);
}

inline Value cmpComplex(Location loc, Value lhs, Value rhs,
                        stablehlo::ComparisonDirection comparisonDirection,
                        OpBuilder *b) {
  auto complexType = cast<ComplexType>(lhs.getType());
  if (isa<FloatType>(complexType.getElementType())) {
    if (comparisonDirection == stablehlo::ComparisonDirection::EQ) {
      return complex::EqualOp::create(*b, loc, lhs, rhs);
    }
    if (comparisonDirection == stablehlo::ComparisonDirection::NE) {
      return complex::NotEqualOp::create(*b, loc, lhs, rhs);
    }

    // Perform a lexicographical comparison for the (real, imaginary) pair.
    Type complexFloatTy = complexType.getElementType();

    Value lhsReal = complex::ReOp::create(*b, loc, complexFloatTy, lhs);
    Value rhsReal = complex::ReOp::create(*b, loc, complexFloatTy, rhs);

    Value lhsImag = complex::ImOp::create(*b, loc, complexFloatTy, lhs);
    Value rhsImag = complex::ImOp::create(*b, loc, complexFloatTy, rhs);

    auto predicate = getCmpPredicate<arith::CmpFPredicate>(comparisonDirection,
                                                           /*is_signed=*/true);
    assert(predicate.has_value() && "expected valid comparison direction");

    //   if (lhsReal == rhsReal && lhsImag `predicate` rhsImag ||
    //       lhsReal `predicate` rhsReal)
    Value realsAreEq = arith::CmpFOp::create(*b, loc, arith::CmpFPredicate::OEQ,
                                             lhsReal, rhsReal);
    Value imagsAreOrdered =
        arith::CmpFOp::create(*b, loc, *predicate, lhsImag, rhsImag);
    Value realsAreOrdered =
        arith::CmpFOp::create(*b, loc, *predicate, lhsReal, rhsReal);

    Value orLhs = arith::AndIOp::create(*b, loc, realsAreEq, imagsAreOrdered);
    return arith::OrIOp::create(*b, loc, orLhs, realsAreOrdered);
  }
  return nullptr;
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::CompareOp>(
    Location loc, ArrayRef<Type> /*resultTypes*/, ArrayRef<Type> argTypes,
    stablehlo::CompareOp::Adaptor adaptor, OpBuilder *b) {
  stablehlo::ComparisonDirection comparisonDirection =
      adaptor.getComparisonDirection();
  const auto &lhs = adaptor.getLhs();
  const auto &rhs = adaptor.getRhs();
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (isa<IntegerType>(elementType)) {
    bool isUnsigned = IsUnsignedIntegerType{}(elementType);
    std::optional<arith::CmpIPredicate> predicate =
        getCmpPredicate<arith::CmpIPredicate>(comparisonDirection, !isUnsigned);
    assert(predicate.has_value() && "expected valid comparison direction");
    return ScalarIOp<stablehlo::CompareOp>::create(*b, loc, predicate.value(),
                                                   lhs, rhs);
  }
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    if (adaptor.getCompareType() &&
        *adaptor.getCompareType() == stablehlo::ComparisonType::TOTALORDER) {
      // The semantics of totalorder fp compare are
      // -NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN
      auto intType = b->getIntegerType(floatType.getWidth());
      auto zero =
          arith::ConstantOp::create(*b, loc, intType, b->getZeroAttr(intType));
      auto max = arith::ConstantOp::create(
          *b, loc, intType,
          b->getIntegerAttr(intType,
                            APInt::getSignedMaxValue(floatType.getWidth())));
      // Switch from a floating point value to a integer value in such a way
      // that when using the integer value to compare, we get the same result
      // for normal values, and -NaN is treated as the smallest value, and NaN
      // is treated as the largest value.
      // If f is a float, and
      // x = bit_cast<int32_t>(f);
      // y = x < 0 ? numeric_limits<int32_t>::max() - x : x;
      // then y is ordered as an int32_t such that finite values have the
      // obvious order, -0 is ordered before 0, and -NaN and NaN appear at the
      // beginning and end of the ordering.
      auto toIntegral = [&](Value v) {
        auto x = arith::BitcastOp::create(*b, loc, intType, v);
        auto cmp =
            arith::CmpIOp::create(*b, loc, arith::CmpIPredicate::slt, x, zero);
        auto sub = arith::SubIOp::create(*b, loc, max, x);
        return arith::SelectOp::create(*b, loc, cmp, sub, x);
      };
      auto lhsInt = toIntegral(lhs);
      auto rhsInt = toIntegral(rhs);
      auto predicate =
          getCmpPredicate<arith::CmpIPredicate>(comparisonDirection,
                                                /*is_signed=*/true);
      assert(predicate.has_value() && "expected valid comparison direction");
      return arith::CmpIOp::create(*b, loc, *predicate, lhsInt, rhsInt);
    }
    std::optional<arith::CmpFPredicate> predicate =
        getCmpPredicate<arith::CmpFPredicate>(comparisonDirection,
                                              /*is_signed=*/true);
    assert(predicate.has_value() && "expected valid comparison direction");
    return ScalarFOp<stablehlo::CompareOp>::create(*b, loc, predicate.value(),
                                                   lhs, rhs);
  }
  if (auto complexType = dyn_cast<ComplexType>(elementType)) {
    return cmpComplex(loc, lhs, rhs, comparisonDirection, b);
  }
  return nullptr;
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::ReducePrecisionOp>(
    Location loc, ArrayRef<Type> /*resultTypes*/, ArrayRef<Type> argTypes,
    stablehlo::ReducePrecisionOp::Adaptor adaptor, OpBuilder *builder) {
  using llvm::APInt;
  mlir::ImplicitLocOpBuilder b(loc, *builder);

  // Integer and float types for casting and constant generation.
  auto floatType =
      cast<FloatType>(cast<TensorType>(argTypes.front()).getElementType());
  int64_t nbits = floatType.getWidth();
  auto intType = mlir::IntegerType::get(loc.getContext(), floatType.getWidth());

  Value xAsInt = arith::BitcastOp::create(b, intType, adaptor.getOperand());

  // SignificandWidth includes the implicit extra bit.
  auto srcMantissaBits = floatType.getFPMantissaWidth() - 1;
  int srcExponentBits = nbits - 1 - srcMantissaBits;

  // Clear the sign bit, it does not participate in rounding and we will restore
  // it later.
  APInt signBitMask(nbits, 1);
  signBitMask <<= nbits - 1;

  APInt expBitsMask(nbits, 1);
  expBitsMask = ((expBitsMask << srcExponentBits) - 1) << srcMantissaBits;

  auto createConstant = [&](const APInt &v) {
    return arith::ConstantIntOp::create(b, intType, v.getZExtValue())
        .getResult();
  };

  Value xAbsBits =
      arith::AndIOp::create(b, xAsInt, createConstant(~signBitMask));
  Value xIsNan = arith::CmpIOp::create(b, arith::CmpIPredicate::ugt, xAbsBits,
                                       createConstant(expBitsMask));

  int destMantissaBits = adaptor.getMantissaBits();
  if (destMantissaBits < static_cast<int>(srcMantissaBits)) {
    // Last remaining mantissa bit.
    APInt lastMantissaBitMask(nbits, 1);
    lastMantissaBitMask <<= srcMantissaBits - destMantissaBits;

    // Compute rounding bias for round-to-nearest with ties to even.  This is
    // equal to a base value of 0111... plus one bit if the last remaining
    // mantissa bit is 1.
    APInt baseRoundingBias = lastMantissaBitMask.lshr(1) - 1;

    Value mantissaDiff = arith::ConstantIntOp::create(
        b, intType, srcMantissaBits - destMantissaBits);
    Value highestMantissaMaskVal = createConstant(lastMantissaBitMask);
    Value baseRoundingBiasVal = createConstant(baseRoundingBias);
    Value xLastMantissaBit = arith::ShRUIOp::create(
        b, arith::AndIOp::create(b, xAsInt, highestMantissaMaskVal),
        mantissaDiff);
    Value xRoundingBias =
        arith::AddIOp::create(b, xLastMantissaBit, baseRoundingBiasVal);

    // Add rounding bias, and mask out truncated bits.  Note that the case
    // where adding the rounding bias overflows into the exponent bits is
    // correct; the non-masked mantissa bits will all be zero, and the
    // exponent will be incremented by one.
    APInt truncationMask = ~(lastMantissaBitMask - 1);
    Value xRounded = arith::AddIOp::create(b, xAsInt, xRoundingBias);
    xAsInt = arith::AndIOp::create(b, xRounded, createConstant(truncationMask));
  }

  int destExponentBits = adaptor.getExponentBits();
  if (destExponentBits < srcExponentBits) {
    // An exponent of 2^(n-1)-1 -- that is, 0111... with the zero in the most-
    // significant bit -- is equal to 1.0f for all exponent sizes.  Adding
    // 2^(n-1)-1 to this gives us the highest non-infinite exponent for a bit-
    // size of n, and subtracting 2^(n-1)-1 from this gives us the lowest'
    // exponent (corresponding to 0.0f).
    //
    // Thus, the f32 exponent corresponding to the highest non-infinite
    // exponent for a bit size of n is (2^7-1) + 2^(n-1)-1, and the f32
    // exponent corresponding to the lowest exponent for a bit size of n is
    // (2^7-1) - 2^(n-1)-1.
    //
    // Note that we have already checked that exponents_bits >= 1.
    APInt exponentBias(nbits, 1);
    exponentBias = (exponentBias << (srcExponentBits - 1)) - 1;

    APInt reducedExponentBias(nbits, 1);
    reducedExponentBias = (reducedExponentBias << (destExponentBits - 1)) - 1;

    APInt reducedMaxExponent = exponentBias + reducedExponentBias;
    APInt reducedMinExponent = exponentBias - reducedExponentBias;

    // Do we overflow or underflow?
    Value xExponent =
        arith::AndIOp::create(b, xAsInt, createConstant(expBitsMask));
    Value xOverflows = arith::CmpIOp::create(
        b, arith::CmpIPredicate::ugt, xExponent,
        createConstant(reducedMaxExponent << srcMantissaBits));
    Value xUnderflows = arith::CmpIOp::create(
        b, arith::CmpIPredicate::ule, xExponent,
        createConstant(reducedMinExponent << srcMantissaBits));

    // Compute appropriately-signed values of zero and infinity.
    Value xSignedZero =
        arith::AndIOp::create(b, xAsInt, createConstant(signBitMask));
    Value xSignedInf =
        arith::OrIOp::create(b, xSignedZero, createConstant(expBitsMask));

    // Force to zero or infinity if overflow or underflow.  (Note that this
    // truncates all denormal values to zero, rather than rounding them.)
    xAsInt = arith::SelectOp::create(b, xOverflows, xSignedInf, xAsInt);
    xAsInt = arith::SelectOp::create(b, xUnderflows, xSignedZero, xAsInt);
  }

  Value result = arith::BitcastOp::create(b, floatType, xAsInt);
  return arith::SelectOp::create(b, xIsNan, adaptor.getOperand(), result);
}

// FIXME(Jakub)
// template <>
// inline Value mapStableHloOpToStdScalarOp<stablehlo::CopyOp>(
//     Location /*loc*/, ArrayRef<Type> /*ResultTypes*/,
//     ArrayRef<Type> /*argTypes*/, stablehlo::CopyOp::Adaptor adaptor,
//     OpBuilder* /*b*/) {
//   return adaptor.getOperands().front();
// }

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::ComplexOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::ComplexOp::Adaptor adaptor, OpBuilder *b) {
  return MapStableHloOpToScalarOpImpl<complex::CreateOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), b);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::MaxOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::MaxOp::Adaptor adaptor, OpBuilder *b) {
  ValueRange operands = adaptor.getOperands();
  Value lhs = operands.front();
  Type complexTy = lhs.getType();

  if (!isa<ComplexType>(complexTy)) {
    return MapStableHloOpToScalarOpImpl<
        IsFloatType, arith::MaximumFOp, IsSignedIntegerType, arith::MaxSIOp,
        IsUnsignedIntegerType, arith::MaxUIOp>{}(loc, resultTypes, argTypes,
                                                 adaptor.getOperands(), b);
  }

  assert(resultTypes.size() == 1 && "MaxOp should return a single result");
  assert(operands.size() == 2 && "MaxOp should take exactly two arguments");

  Value rhs = operands.back();
  // 'max' performs a lexicographical comparison for the (real, imaginary) pair.
  Value cond = cmpComplex(loc, lhs, rhs, stablehlo::ComparisonDirection::GE, b);

  return arith::SelectOp::create(*b, loc, cond, lhs, rhs).getResult();
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::MinOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::MinOp::Adaptor adaptor, OpBuilder *b) {
  ValueRange operands = adaptor.getOperands();
  Value lhs = operands.front();
  Type complexTy = lhs.getType();

  if (!isa<ComplexType>(complexTy)) {
    return MapStableHloOpToScalarOpImpl<
        IsFloatType, arith::MinimumFOp, IsSignedIntegerType, arith::MinSIOp,
        IsUnsignedIntegerType, arith::MinUIOp>{}(loc, resultTypes, argTypes,
                                                 adaptor.getOperands(), b);
  }

  assert(resultTypes.size() == 1 && "MinOp should return a single result");
  assert(operands.size() == 2 && "MinOp should take exactly two arguments");

  Value rhs = operands.back();
  // 'min' performs a lexicographical comparison for the (real, imaginary) pair.
  Value cond = cmpComplex(loc, lhs, rhs, stablehlo::ComparisonDirection::LE, b);

  return arith::SelectOp::create(*b, loc, cond, lhs, rhs).getResult();
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::RealOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::RealOp::Adaptor adaptor, OpBuilder *b) {
  if (!isa<ComplexType>(adaptor.getOperand().getType())) {
    return adaptor.getOperand();
  }
  return MapStableHloOpToScalarOpImpl<complex::ReOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), b);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::ImagOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::ImagOp::Adaptor adaptor, OpBuilder *b) {
  if (!isa<ComplexType>(adaptor.getOperand().getType())) {
    return arith::ConstantOp::create(
        *b, loc, b->getZeroAttr(adaptor.getOperand().getType()));
  }
  return MapStableHloOpToScalarOpImpl<complex::ImOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), b);
}

// 'target_types' is the unconverted type (signed or unsigned if integer),
// 'ResultTypes' is the converted type (signless if integer).
inline Value mapConvertOpToStdScalarOp(Location loc, ArrayRef<Type> targetTypes,
                                       ArrayRef<Type> resultTypes,
                                       ArrayRef<Type> argTypes, ValueRange args,
                                       OpBuilder *b) {
  assert(targetTypes.size() == 1 && "ConvertOp should return a single result");
  assert(resultTypes.size() == 1 && "ConvertOp should return a single result");
  assert(argTypes.size() == 1 && "ConvertOp should take a single argument");
  assert(args.size() == 1 && "ConvertOp should take a single argument");

  Type sourceType = getElementTypeOrSelf(argTypes.front());
  Type targetType = getElementTypeOrSelf(targetTypes.front());
  Type convertedSourceType = getElementTypeOrSelf(args.front());

  // A boolean value is considered to be unsigned when converting to
  // floating-point. Otherwise, it will become `-1`.
  if (IsUnsignedIntegerType{}(sourceType) &&
      mlir::arith::UIToFPOp::areCastCompatible(convertedSourceType,
                                               targetType)) {
    return mlir::arith::UIToFPOp::create(*b, loc, resultTypes, args,
                                         ArrayRef<NamedAttribute>{});
  }
  if (mlir::arith::SIToFPOp::areCastCompatible(sourceType, targetType)) {
    return mlir::arith::SIToFPOp::create(*b, loc, resultTypes, args,
                                         ArrayRef<NamedAttribute>{});
  }
  if (isa<FloatType>(sourceType) && isa<FloatType>(targetType)) {
    auto src = cast<FloatType>(sourceType);
    auto res = cast<FloatType>(targetType);
    if (src.getWidth() > res.getWidth()) {
      return mlir::arith::TruncFOp::create(*b, loc, resultTypes, args,
                                           ArrayRef<NamedAttribute>{});
    }
    if (src.getWidth() < res.getWidth()) {
      return mlir::arith::ExtFOp::create(*b, loc, resultTypes, args,
                                         ArrayRef<NamedAttribute>{});
    }
    // There's no direct conversion between different 16 bit floating point
    // types, so go through 32 bit float.
    if (sourceType != targetType) {
      assert(sourceType.isBF16() || targetType.isBF16());
      Value ext = arith::ExtFOp::create(*b, loc, b->getF32Type(), args);
      return arith::TruncFOp::create(*b, loc, resultTypes, ext);
    }
    // No conversion is needed for identical float types.
    return args.front();
  }
  if (targetType.isInteger(/*width=*/1)) {
    // When casting to bool, we need to compare whether the value is equal to
    // zero.
    if (sourceType.isSignlessInteger() || sourceType.isUnsignedInteger()) {
      Value zeroIntval = arith::ConstantOp::create(
          *b, loc, b->getZeroAttr(args.front().getType()));
      return mlir::arith::CmpIOp::create(*b, loc, arith::CmpIPredicate::ne,
                                         args.front(), zeroIntval);
    }
    if (isa<FloatType>(sourceType)) {
      Value zero = arith::ConstantOp::create(
          *b, loc, b->getZeroAttr(args.front().getType()));
      return mlir::arith::CmpFOp::create(*b, loc, arith::CmpFPredicate::UNE,
                                         args.front(), zero);
    }
  }
  if (isa<IntegerType>(sourceType) && isa<IntegerType>(targetType)) {
    auto src = cast<IntegerType>(sourceType);
    auto res = cast<IntegerType>(targetType);
    if (src.getWidth() > res.getWidth()) {
      return mlir::arith::TruncIOp::create(*b, loc, resultTypes, args,
                                           ArrayRef<NamedAttribute>{});
    }
    if (src.getWidth() < res.getWidth()) {
      // Special case boolean values, so they get casted to `1` instead of `-1`.
      if (IsUnsignedIntegerType{}(src)) {
        return mlir::arith::ExtUIOp::create(*b, loc, resultTypes, args,
                                            ArrayRef<NamedAttribute>{});
      }
      return mlir::arith::ExtSIOp::create(*b, loc, resultTypes, args,
                                          ArrayRef<NamedAttribute>{});
    }
    // No conversion is needed for the same width integers
    return args.front();
  }
  if (targetType.isUnsignedInteger() &&
      mlir::arith::FPToUIOp::areCastCompatible(convertedSourceType,
                                               targetType)) {
    return mlir::arith::FPToUIOp::create(*b, loc, resultTypes, args,
                                         ArrayRef<NamedAttribute>{});
  }
  if (mlir::arith::FPToSIOp::areCastCompatible(convertedSourceType,
                                               targetType)) {
    return mlir::arith::FPToSIOp::create(*b, loc, resultTypes, args,
                                         ArrayRef<NamedAttribute>{});
  }
  if (isa<ComplexType>(targetType)) {
    Type targetElementType = cast<ComplexType>(targetType).getElementType();
    assert(!isa<ComplexType>(targetElementType) &&
           "elements of complex numbers should not be complex");
    Value targetReal;
    Value targetImag;
    if (isa<ComplexType>(sourceType)) {
      // We are converting from complex type: convert real and imaginary parts
      // separately.
      Type sourceElementType = cast<ComplexType>(sourceType).getElementType();
      assert(!isa<ComplexType>(sourceElementType) &&
             "elements of complex numbers should not be complex");
      Value sourceReal =
          mlir::complex::ReOp::create(*b, loc, sourceElementType, args.front());
      targetReal =
          mapConvertOpToStdScalarOp(loc, targetElementType, targetElementType,
                                    sourceElementType, sourceReal, b);
      Value sourceImag =
          mlir::complex::ImOp::create(*b, loc, sourceElementType, args.front());
      targetImag =
          mapConvertOpToStdScalarOp(loc, targetElementType, targetElementType,
                                    sourceElementType, sourceImag, b);
    } else {
      // We are converting from real (float, integer, etc.) type, convert the
      // real part and set the imaginary part to 0.
      targetReal = mapConvertOpToStdScalarOp(
          loc, targetElementType, targetElementType, argTypes, args, b);
      targetImag = mlir::arith::ConstantOp::create(
          *b, loc, b->getFloatAttr(targetElementType, 0.0));
    }
    return mlir::complex::CreateOp::create(*b, loc, targetType, targetReal,
                                           targetImag);
  }
  if (auto sourceComplexType = dyn_cast<ComplexType>(sourceType)) {
    auto sourceElementType = sourceComplexType.getElementType();
    // When converting from complex to a non-complex type, we take just the real
    // part of the complex number.
    Value sourceReal =
        mlir::complex::ReOp::create(*b, loc, sourceElementType, args.front());
    return mapConvertOpToStdScalarOp(loc, targetTypes, resultTypes,
                                     sourceElementType, sourceReal, b);
  }
  return nullptr;
}

/// Lower bitcast operations where the input and resulting type are the same
/// bitwidth, thus implying that the operation is fully defined by parallel
/// loops and scalar operations without any shape dimension changes.
template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::BitcastConvertOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::BitcastConvertOp::Adaptor adaptor, OpBuilder *b) {
  Type argType = getElementTypeOrSelf(argTypes.front());
  Type resultType = getElementTypeOrSelf(resultTypes.front());

  // Skip needless casts.
  if (argType == resultType) {
    return adaptor.getOperand();
  }

  if (!isa<FloatType, IntegerType>(resultType) ||
      !isa<FloatType, IntegerType>(argType)) {
    return nullptr;
  }

  if (resultType.getIntOrFloatBitWidth() != argType.getIntOrFloatBitWidth()) {
    return nullptr;
  }

  return mlir::arith::BitcastOp::create(*b, loc, resultTypes,
                                        adaptor.getOperands());
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::DotOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::DotOp::Adaptor adaptor, OpBuilder *b) {
  // Dot Op converter from lhlo to affine only accepts float and integer types.
  const auto &lhs = adaptor.getOperands()[0];
  const auto &rhs = adaptor.getOperands()[1];
  const auto &result = adaptor.getOperands()[2];
  Type elementType = lhs.getType();
  if (isa<FloatType>(elementType)) {
    Value floatMul =
        MapStableHloOpToScalarOpImpl<IsFloatType, ::mlir::arith::MulFOp>{}(
            loc, resultTypes, argTypes, {lhs, rhs}, b);
    return MapStableHloOpToScalarOpImpl<IsFloatType, ::mlir::arith::AddFOp>{}(
        loc, resultTypes, argTypes, {floatMul, result}, b);
  }
  if (isa<IntegerType>(elementType)) {
    Value intMul =
        MapStableHloOpToScalarOpImpl<IsAnyIntegerType, ::mlir::arith::MulIOp>{}(
            loc, resultTypes, argTypes, {lhs, rhs}, b);
    return MapStableHloOpToScalarOpImpl<IsAnyIntegerType,
                                        ::mlir::arith::AddIOp>{}(
        loc, resultTypes, argTypes, {intMul, result}, b);
  }
  return nullptr;
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::IsFiniteOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    stablehlo::IsFiniteOp::Adaptor adaptor, OpBuilder *b) {
  if (isa<FloatType>(adaptor.getX().getType())) {
    auto posInf = APFloat::getInf(
        cast<FloatType>(adaptor.getX().getType()).getFloatSemantics());
    auto constPosInf = arith::ConstantOp::create(
        *b, loc, b->getFloatAttr(adaptor.getX().getType(), posInf));
    Value absX = ::mlir::math::AbsFOp::create(*b, loc, adaptor.getX());
    return ::mlir::arith::CmpFOp::create(*b, loc, arith::CmpFPredicate::ONE,
                                         absX, constPosInf);
  }
  return nullptr;
}

/// Implements the conversion of HLO op to scalar op (to use within region of a
/// linalg.generic op) for compare-select style operations like min/max.
template <typename... Args>
struct CompareSelectOpToStdScalarOp {
  static Value map(Location /*loc*/,
                   stablehlo::ComparisonDirection /*comparison_direction*/,
                   ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
                   ValueRange /*args*/, OpBuilder * /*b*/) {
    return nullptr;
  }
};

/// Specialization which allows converting to a comparison operation in standard
/// dialect with a given predicate based on the element type of the operand.
template <typename SupportedType, typename StdCompareOp, typename Predicate,
          typename... Args>
struct CompareSelectOpToStdScalarOp<SupportedType, StdCompareOp, Predicate,
                                    Args...> {
  static Value map(Location loc,
                   stablehlo::ComparisonDirection comparisonDirection,
                   ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
                   ValueRange args, OpBuilder *b) {
    Type elementType = getElementTypeOrSelf(argTypes.front());
    if (isa<SupportedType>(elementType)) {
      auto predicate = getCmpPredicate<Predicate>(
          comparisonDirection, !elementType.isUnsignedInteger());
      assert(predicate.has_value() && "expected valid comparison direction");
      auto cmp =
          StdCompareOp::create(*b, loc, predicate.getValue(), args[0], args[1]);
      return ::mlir::arith::SelectOp::create(*b, loc, cmp, args[0], args[1]);
    }
    return CompareSelectOpToStdScalarOp<Args...>::map(
        loc, comparisonDirection, resultTypes, argTypes, args, b);
  }
};

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::ClampOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::ClampOp::Adaptor op, OpBuilder *b) {
  // clamp(lb, x, ub) = min(max(lb, x), ub)
  Value maxLbX = mapStableHloOpToStdScalarOp<stablehlo::MaxOp>(
      loc, resultTypes, argTypes, ValueRange{op.getMin(), op.getOperand()}, b);
  return mapStableHloOpToStdScalarOp<stablehlo::MinOp>(
      loc, resultTypes, argTypes, ValueRange{maxLbX, op.getMax()}, b);
}

template <typename U, typename S>
inline Value makeSafeIntDiv(ImplicitLocOpBuilder &lb, Type originalType,
                            Value lhs, Value rhs, Value returnedOnZero,
                            Value returnedOnSignedOverflow) {
  Type type = lhs.getType();
  auto elementType = cast<IntegerType>(getElementTypeOrSelf(type));
  Value zero = arith::ConstantOp::create(lb, lb.getZeroAttr(type));
  auto makeConstant = [&](const APInt &i) {
    return getConstantOrSplat(&lb, lb.getLoc(), type,
                              lb.getIntegerAttr(elementType, i));
  };
  Value one = makeConstant(APInt(elementType.getWidth(), 1));
  Value rhsIsZero =
      arith::CmpIOp::create(lb, arith::CmpIPredicate::eq, rhs, zero);

  // For unsigned just set the divisor to 1 when it would be 0.
  if (originalType.isUnsignedInteger()) {
    Value safeRhs = arith::SelectOp::create(lb, rhsIsZero, one, rhs);
    Value safeDiv = U::create(lb, lhs, safeRhs);
    return arith::SelectOp::create(lb, rhsIsZero, returnedOnZero, safeDiv);
  }

  // For signed also check for INT_MIN / -1.
  Value smin = makeConstant(APInt::getSignedMinValue(elementType.getWidth()));
  Value lhsIsSmin =
      arith::CmpIOp::create(lb, arith::CmpIPredicate::eq, lhs, smin);
  Value minusOne = makeConstant(APInt::getAllOnes(elementType.getWidth()));
  Value rhsIsMinusOne =
      arith::CmpIOp::create(lb, arith::CmpIPredicate::eq, rhs, minusOne);
  Value hasIntMinOverflow = arith::AndIOp::create(lb, lhsIsSmin, rhsIsMinusOne);
  Value rhsIsUnsafe = arith::OrIOp::create(lb, rhsIsZero, hasIntMinOverflow);
  Value safeRhs = arith::SelectOp::create(lb, rhsIsUnsafe, one, rhs);
  Value safeDiv = S::create(lb, lhs, safeRhs);
  Value safeSmin = arith::SelectOp::create(lb, hasIntMinOverflow,
                                           returnedOnSignedOverflow, safeDiv);
  return arith::SelectOp::create(lb, rhsIsZero, returnedOnZero, safeSmin);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::DivOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::DivOp::Adaptor adaptor, OpBuilder *b) {
  Type originalType = getElementTypeOrSelf(argTypes.front());
  if (isa<ComplexType, FloatType>(originalType)) {
    return MapStableHloOpToScalarOpImpl<IsFloatType, arith::DivFOp,
                                        IsComplexType, complex::DivOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);
  }

  // Integer division overflow behavior:
  //
  // X / 0 == -1
  // INT_SMIN /s -1 = INT_SMIN
  ImplicitLocOpBuilder lb(loc, *b);
  Type type = adaptor.getLhs().getType();
  auto elementType = cast<IntegerType>(getElementTypeOrSelf(type));
  auto makeConstant = [&](const APInt &i) {
    return getConstantOrSplat(&lb, lb.getLoc(), type,
                              lb.getIntegerAttr(elementType, i));
  };
  Value minusOne = makeConstant(APInt::getAllOnes(elementType.getWidth()));
  Value smin = makeConstant(APInt::getSignedMinValue(elementType.getWidth()));
  return makeSafeIntDiv<arith::DivUIOp, arith::DivSIOp>(
      lb, originalType, adaptor.getLhs(), adaptor.getRhs(),
      /*returnedOnZero=*/minusOne,
      /*returnedOnSignedOverflow=*/smin);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::RemOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::RemOp::Adaptor adaptor, OpBuilder *b) {
  Type originalType = getElementTypeOrSelf(argTypes.front());
  if (isa<ComplexType, FloatType>(originalType)) {
    return MapStableHloOpToScalarOpImpl<IsFloatType, arith::RemFOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);
  }

  // Integer remainder overflow behavior:
  //
  // X % 0 == X
  // INT_SMIN %s -1 = 0
  ImplicitLocOpBuilder lb(loc, *b);
  Type type = adaptor.getLhs().getType();
  Value zero = arith::ConstantOp::create(lb, lb.getZeroAttr(type));
  return makeSafeIntDiv<arith::RemUIOp, arith::RemSIOp>(
      lb, originalType, adaptor.getLhs(), adaptor.getRhs(),
      /*returnedOnZero=*/adaptor.getLhs(),
      /*returnedOnSignedOverflow=*/zero);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::NegOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::NegOp::Adaptor adaptor, OpBuilder *b) {
  Type elementType = getElementTypeOrSelf(adaptor.getOperand().getType());
  if (isa<ComplexType, FloatType>(elementType)) {
    return MapStableHloOpToScalarOpImpl<IsFloatType, ::mlir::arith::NegFOp,
                                        IsComplexType,
                                        ::mlir::complex::NegOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);
  }
  if (isa<IntegerType>(elementType)) {
    // lmhlo.neg(x, result) -> result = sub(0, x)
    Value lhs = adaptor.getOperand();
    Value zeroIntval =
        arith::ConstantOp::create(*b, loc, b->getZeroAttr(lhs.getType()));
    return ScalarIOp<stablehlo::SubtractOp>::create(*b, loc, zeroIntval, lhs);
  }
  return nullptr;
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::NotOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    stablehlo::NotOp::Adaptor adaptor, OpBuilder *b) {
  Type elementType = getElementTypeOrSelf(adaptor.getOperand().getType());
  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    // lmhlo.not(x) -> x ^ -1
    Value allOnes = getConstantOrSplat(
        b, loc, adaptor.getOperand().getType(),
        b->getIntegerAttr(integerType,
                          APInt::getAllOnes(integerType.getWidth())));
    return ::mlir::arith::XOrIOp::create(*b, loc, allOnes,
                                         adaptor.getOperand());
  }
  return nullptr;
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::LogisticOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> /*argTypes*/,
    stablehlo::LogisticOp::Adaptor adaptor, OpBuilder *b) {
  // 1.0 / (1.0 - exp(-x))
  Value negX = mapStableHloOpToStdScalarOp<stablehlo::NegOp>(
      loc, resultTypes, resultTypes, {adaptor.getOperand()}, b);
  Value expNegX = mapStableHloOpToStdScalarOp<stablehlo::ExpOp>(
      loc, resultTypes, resultTypes, {{negX}}, b);

  Value oneFloat = arith::ConstantOp::create(*b, loc, b->getF32FloatAttr(1.0));
  Value one = mapConvertOpToStdScalarOp(loc, resultTypes, resultTypes,
                                        {oneFloat.getType()}, {{oneFloat}}, b);
  Value oneAddExprNegX = mapStableHloOpToStdScalarOp<stablehlo::AddOp>(
      loc, resultTypes, resultTypes, {{expNegX, one}}, b);
  return mapStableHloOpToStdScalarOp<stablehlo::DivOp>(
      loc, resultTypes, resultTypes, {{one, oneAddExprNegX}}, b);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::PowOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::PowOp::Adaptor adaptor, OpBuilder *b) {
  auto lb = ImplicitLocOpBuilder(loc, *b);
  // Floating point can use std::powf
  auto resultType = resultTypes.front();
  if (isa<ComplexType, FloatType>(resultType)) {
    return MapStableHloOpToScalarOpImpl<IsFloatType, math::PowFOp,
                                        IsComplexType, complex::PowOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);
  }

  // Exponentiation by squaring:
  // https://en.wikipedia.org/wiki/Exponentiation_by_squaring;
  Value negOne =
      arith::ConstantOp::create(lb, lb.getIntegerAttr(resultType, -1));
  Value zero = arith::ConstantOp::create(lb, lb.getIntegerAttr(resultType, 0));
  Value one = arith::ConstantOp::create(lb, lb.getIntegerAttr(resultType, 1));
  Value two = arith::ConstantOp::create(lb, lb.getIntegerAttr(resultType, 2));
  Value step = arith::ConstantIndexOp::create(lb, 1);
  Value lowerBound = arith::ConstantIndexOp::create(lb, 0);
  // Everything else would overflow for any exponent > 1, as 2^64
  // is the larget possible exponent for a 64-bit integer, and
  // that's 1 << 6.
  Value upperBound = arith::ConstantIndexOp::create(lb, 6);
  auto originalBase = adaptor.getLhs();
  auto originalExponent = adaptor.getRhs();

  Value accum =
      scf::ForOp::create(
          lb, lowerBound, upperBound, step,
          SmallVector<Value>({one, originalBase, originalExponent}),
          [&](OpBuilder &b, Location, Value /*v*/, ValueRange iters) {
            Value accum = iters[0];
            Value base = iters[1];
            Value exponent = iters[2];

            Value condition = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::eq,
                ::mlir::arith::AndIOp::create(b, loc, exponent, one), one);
            Value multiplied =
                ::mlir::arith::MulIOp::create(b, loc, accum, base);
            accum = ::mlir::arith::SelectOp::create(b, loc, condition,
                                                    multiplied, accum);
            base = ::mlir::arith::MulIOp::create(b, loc, base, base);
            exponent = ::mlir::arith::ShRUIOp::create(b, loc, exponent, one);
            scf::YieldOp::create(b, loc,
                                 SmallVector<Value>({accum, base, exponent}));
          })
          .getResult(0);

  Value rhsIsEven = arith::CmpIOp::create(
      lb, arith::CmpIPredicate::eq,
      arith::RemSIOp::create(lb, adaptor.getRhs(), two), zero);
  Value rhsIsNegative = arith::CmpIOp::create(lb, arith::CmpIPredicate::slt,
                                              adaptor.getRhs(), zero);
  Value lhsIsOne = arith::CmpIOp::create(lb, arith::CmpIPredicate::eq,
                                         adaptor.getLhs(), one);
  Value lhsIsNegOne = arith::CmpIOp::create(lb, arith::CmpIPredicate::eq,
                                            adaptor.getLhs(), negOne);

  // The accum is correct when the rhs is non-negative. When rhs is
  // negative, we return 0 for integer, with the exception of lhs values of 1
  // and -1 which have integer results for negative exponents. Specifically, the
  // calulation is the following:
  //
  // - Return accum if the rhs is not negative.
  // - Return 1 or -1 depending on the parity of rhs when the lhs is -1.
  // - Return 1 if lhs is 1.
  // - Else return 0.
  Value ifLhsIsOne = ::mlir::arith::SelectOp::create(lb, lhsIsOne, one, zero);
  Value ifLhsIsNegOne = ::mlir::arith::SelectOp::create(
      lb, lhsIsNegOne,
      ::mlir::arith::SelectOp::create(lb, rhsIsEven, one, negOne), ifLhsIsOne);
  return ::mlir::arith::SelectOp::create(lb, rhsIsNegative, ifLhsIsNegOne,
                                         accum);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::SelectOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    stablehlo::SelectOp::Adaptor adaptor, OpBuilder *b) {
  return MapStableHloOpToScalarOpImpl<::mlir::arith::SelectOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), b);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::SignOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> /*argTypes*/,
    stablehlo::SignOp::Adaptor adaptor, OpBuilder *b) {
  Value operand = adaptor.getOperand();
  Type elementType = getElementTypeOrSelf(operand.getType());
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    Value zero =
        arith::ConstantOp::create(*b, loc, b->getZeroAttr(operand.getType()));
    Value ne0I1 = ::mlir::arith::CmpFOp::create(
        *b, loc, arith::CmpFPredicate::ONE, operand, zero);
    Value ne0Float =
        ::mlir::arith::UIToFPOp::create(*b, loc, zero.getType(), ne0I1);
    Value copySign = ::mlir::math::CopySignOp::create(*b, loc, resultTypes,
                                                      ne0Float, operand);
    auto isNan = ::mlir::arith::CmpFOp::create(
        *b, loc, arith::CmpFPredicate::UNO, operand, operand);
    return ::mlir::arith::SelectOp::create(*b, loc, isNan, operand, copySign);
  }
  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    // sign(x) = x == 0 ? 0 : ((x s>> 31) | 1)
    Value zero =
        arith::ConstantOp::create(*b, loc, b->getZeroAttr(operand.getType()));
    Value bitwidthMinusOne = getConstantOrSplat(
        b, loc, operand.getType(),
        b->getIntegerAttr(integerType, integerType.getWidth() - 1));
    Value one = getConstantOrSplat(b, loc, operand.getType(),
                                   b->getIntegerAttr(integerType, 1));
    Value cmp = ::mlir::arith::CmpIOp::create(*b, loc, arith::CmpIPredicate::eq,
                                              operand, zero);
    Value ashr =
        ::mlir::arith::ShRSIOp::create(*b, loc, operand, bitwidthMinusOne);
    Value orOp = ::mlir::arith::OrIOp::create(*b, loc, ashr, one);
    return ::mlir::arith::SelectOp::create(*b, loc, cmp, zero, orOp);
  }
  if (isa<ComplexType>(elementType)) {
    return ::mlir::complex::SignOp::create(*b, loc, elementType, operand);
  }
  return nullptr;
}

/// Construct operations to select the saturated value if the shift amount is
/// greater than the bitwidth of the type.
inline Value selectShiftedOrSaturated(ImplicitLocOpBuilder &lb, Value rhs,
                                      Value shifted, Value saturated,
                                      Type type) {
  Type etype = getElementTypeOrSelf(type);
  auto bitWidthInt = etype.getIntOrFloatBitWidth();
  Value bitWidth = getConstantOrSplat(&lb, lb.getLoc(), type,
                                      lb.getIntegerAttr(etype, bitWidthInt));
  Value cmp = mlir::arith::CmpIOp::create(lb, mlir::arith::CmpIPredicate::ugt,
                                          bitWidth, rhs);
  return mlir::arith::SelectOp::create(lb, cmp, shifted, saturated);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::ShiftLeftOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    stablehlo::ShiftLeftOp::Adaptor adaptor, OpBuilder *b) {
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value zero = arith::ConstantOp::create(lb, lb.getZeroAttr(type));
  Value shifted = mlir::arith::ShLIOp::create(lb, lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, zero, type);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::ShiftRightLogicalOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    stablehlo::ShiftRightLogicalOp::Adaptor adaptor, OpBuilder *b) {
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value zero = arith::ConstantOp::create(lb, b->getZeroAttr(type));
  Value shifted = mlir::arith::ShRUIOp::create(lb, lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, zero, type);
}

template <>
inline Value mapStableHloOpToStdScalarOp<stablehlo::ShiftRightArithmeticOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    stablehlo::ShiftRightArithmeticOp::Adaptor adaptor, OpBuilder *b) {
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();
  Type etype = getElementTypeOrSelf(type);
  auto bitWidthInt = etype.getIntOrFloatBitWidth();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value maxShift = getConstantOrSplat(
      b, loc, type, lb.getIntegerAttr(etype, bitWidthInt - 1));
  Value saturatedShifted = mlir::arith::ShRSIOp::create(lb, lhs, maxShift);
  Value shifted = mlir::arith::ShRSIOp::create(lb, lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, saturatedShifted, type);
}
} // namespace impl

struct StableHloOpToStdScalarOp {
  // Converts stablehlo 'op' to linalg and arith ops.
  template <typename StableHloOpTy>
  static Value mapOp(StableHloOpTy op, ArrayRef<Type> resultTypes,
                     ValueRange args, OpBuilder *b) {
    auto argTypes = llvm::to_vector(op->getOperandTypes());
    return mapOpWithArgTypes(op, resultTypes, argTypes, args, b);
  }

  // Converts stablehlo 'op' to linalg and arith ops. The types of 'args' may
  // already be converted, 'argTypes' are their original types.
  template <typename StableHloOpTy>
  static Value mapOpWithArgTypes(StableHloOpTy op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 OpBuilder *b) {
    static_assert(!std::is_same<StableHloOpTy, stablehlo::ConvertOp>::value);
    return mapOpOfType<StableHloOpTy>(
        op.getLoc(), resultTypes, argTypes,
        typename StableHloOpTy::Adaptor(args, op->getAttrDictionary(),
                                        op.getProperties()),
        b);
  }
  // Overload for stablehlo::ConvertOp.
  static Value mapOpWithArgTypes(stablehlo::ConvertOp op,
                                 ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 OpBuilder *b) {
    return impl::mapConvertOpToStdScalarOp(op.getLoc(), op.getType(),
                                           resultTypes, argTypes, args, b);
  }

  // Converts stablehlo 'op' to linalg and arith ops.
  template <typename StableHloOpTy>
  static Value
  mapOpOfType(Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
              typename StableHloOpTy::Adaptor adaptor, OpBuilder *b) {
    if (std::is_same<StableHloOpTy, stablehlo::ConvertOp>::value) {
      // Note: this assumes that the caller is passing result/arg types with
      // appropriate signedness.
      return impl::mapConvertOpToStdScalarOp(
          loc, resultTypes, resultTypes, argTypes, adaptor.getOperands(), b);
    }
    return impl::mapStableHloOpToStdScalarOp<StableHloOpTy>(
        loc, resultTypes, argTypes, adaptor, b);
  }
};

} // namespace stablehlo
} // namespace mlir

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_MAP_STABLEHLO_TO_SCALAR_OP_H
