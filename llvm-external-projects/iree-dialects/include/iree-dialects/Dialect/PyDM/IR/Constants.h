// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_PYDM_IR_CONSTANTS_H
#define IREE_DIALECTS_DIALECT_PYDM_IR_CONSTANTS_H

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace PYDM {

/// Built-in collection types (lists, tuples, dicts) can be implemented in terms
/// of empty, boxed, or unboxed storage. Generally at program construction
/// time, the storage will be empty or boxed, as there is seldom sufficient
/// information to determine whether object identity can be dropped at this
/// phase. The collection may be converted to unboxed storage during type
/// refinement if it is safe to do so.
///
/// The empty storage class is used for collections that are known empty,
/// which can help type refinement to ignore them.
enum class CollectionStorageClass {
  Boxed,
  Empty,
  Unboxed,
};

/// Category of the numeric type. These are arranged such that during promotion,
/// the type with the largest category value determines the category of
/// promotion.
enum class NumericCategory : int {
  Bool = 0,
  WeakInteger = 1,
  Unsigned = 2,
  Signed = 3,
  APSigned = 4,
  WeakReal = 5,
  Real = 6,
  WeakComplex = 7,
  Complex = 8,
};

/// For integer types (Unsigned, Signed, and Bool category), this is the type
/// specific sub type code for sizes that we support.
/// Only POT bit sizes up to 64bits are supported. They sort into promotion
/// order within a category.
enum class IntegerSubTypeCode : int {
  Integer8 = 0,
  Integer16 = 1,
  Integer32 = 2,
  Integer64 = 3,
};

/// As with integer types, this is the type specific code for supported
/// floating point types within the Real category. They sort into promotion
/// order with the special case that combining an FP16 and BF16 promotes to
/// FP32.
enum class RealSubTypeCode : int {
  FP16 = 0,
  BF16 = 1,
  FP32 = 2,
  FP64 = 3,
};

/// Sub type code for complex types, which consist of two floating point
/// values (either FP32 or FP64). Space is retained in the enumeration for
/// 16bit elements.
enum class ComplexSubTypeCode : int {
  UNUSED0 = 0,
  UNUSED1 = 1,
  COMPLEX64 = 2,
  COMPLEX128 = 3,
};

/// Makes a numeric category code with bit pattern:
///   1 C C C C S S
/// Where 'C' is category code and 'S' is sub type code.
/// These range from 0x40 - 0x7f
template <typename SubTypeCode>
constexpr int makeNumericTypeCode(const NumericCategory cat,
                                  const SubTypeCode subType) {
  return 0x40 | (static_cast<int>(cat) << 2) | (static_cast<int>(subType));
}

// Each built-in (to the compiler) type has a unique code, enumerated here.
// Generally, the closed part of the type system will have type codes <
// FirstCustom.
// If editing, also update the constants in rtl/modules/constants.py.
enum class BuiltinTypeCode : int {
  // Built-in types, ordered by rough "core-ness" so that lower numbers
  // are easier to spot for common cases.
  None = 0x1,
  Tuple = 0x2,
  List = 0x3,
  Str = 0x4,
  Bytes = 0x5,
  ExceptionResult = 0x6,
  Type = 0x7,

  // Start of the encoded numeric types codes. Lower 5 bits represent a bit
  // packed encoding of the numeric category (3 bits) and sub type
  // code (2 bits):
  NumericStart = 0x20,
  NumericBool = makeNumericTypeCode(NumericCategory::Bool, 0),
  WeakInteger = makeNumericTypeCode(NumericCategory::WeakInteger, 0),
  NumericUnsigned8Bit = makeNumericTypeCode(NumericCategory::Unsigned,
                                            IntegerSubTypeCode::Integer8),
  NumericUnsigned16Bit = makeNumericTypeCode(NumericCategory::Unsigned,
                                             IntegerSubTypeCode::Integer16),
  NumericUnsigned32Bit = makeNumericTypeCode(NumericCategory::Unsigned,
                                             IntegerSubTypeCode::Integer32),
  NumericUnsigned64Bit = makeNumericTypeCode(NumericCategory::Unsigned,
                                             IntegerSubTypeCode::Integer64),
  NumericSigned8Bit = makeNumericTypeCode(NumericCategory::Signed,
                                          IntegerSubTypeCode::Integer8),
  NumericSigned16Bit = makeNumericTypeCode(NumericCategory::Signed,
                                           IntegerSubTypeCode::Integer16),
  NumericSigned32Bit = makeNumericTypeCode(NumericCategory::Signed,
                                           IntegerSubTypeCode::Integer32),
  NumericSigned64Bit = makeNumericTypeCode(NumericCategory::Signed,
                                           IntegerSubTypeCode::Integer64),
  NumericAPSigned = makeNumericTypeCode(NumericCategory::APSigned, 0),
  WeakReal = makeNumericTypeCode(NumericCategory::WeakReal, 0),
  NumericRealFP16 =
      makeNumericTypeCode(NumericCategory::Real, RealSubTypeCode::FP16),
  NumericRealBF16 =
      makeNumericTypeCode(NumericCategory::Real, RealSubTypeCode::BF16),
  NumericRealFP32 =
      makeNumericTypeCode(NumericCategory::Real, RealSubTypeCode::FP32),
  NumericRealFP64 =
      makeNumericTypeCode(NumericCategory::Real, RealSubTypeCode::FP64),
  WeakComplex = makeNumericTypeCode(NumericCategory::WeakComplex, 0),
  NumericComplex64 = makeNumericTypeCode(NumericCategory::Complex,
                                         ComplexSubTypeCode::COMPLEX64),
  NumericComplex128 = makeNumericTypeCode(NumericCategory::Complex,
                                          ComplexSubTypeCode::COMPLEX128),
  NumericEnd = 0x7f,

  // Objects start at 0x100, with 0x100 being the generic "object" type
  // and then all following corresponding to user-defined types.
  Object = 0x100,
  FirstCustom = 0x101,
};

} // namespace PYDM
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_PYDM_IR_CONSTANTS_H
