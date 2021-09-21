// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_DIALECT_H
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_DIALECT_H

#include "iree-dialects/Dialect/IREEPyDM/IR/Interfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_pydm {

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

  // Weak-sized numeric types are of implementation defined size and are
  // always considered lower in the promotion order than a discrete
  // sized type of the same class.
  Bool = 0x8,
  Integer = 0x9,
  Real = 0xa,
  Complex = 0xb,

  // Discrete sized integer types.
  // TODO: Fiddle with all of these values so that promotion can be
  // done cleverly with bit twiddling of some kind.
  Integer1 = 0x10,
  Integer2 = 0x11,
  Integer4 = 0x12,
  Integer8 = 0x13,
  UInteger1 = 0x14,
  UInteger2 = 0x15,
  UInteger4 = 0x16,
  UInteger8 = 0x17,

  // Discrete sized FP types.
  Float2 = 0x18,
  Float4 = 0x19,
  Float8 = 0x1a,
  BFloat2 = 0x1b,

  // Complex.
  Complex4 = 0x1c,
  Complex8 = 0x1d,

  // Objects start at 0x100, with 0x100 being the generic "object" type
  // and then all following corresponding to user-defined types.
  Object = 0x100,
  FirstCustom = 0x101,
};

/// Base class for all unboxed primitive types.
class PrimitiveType : public mlir::Type {
 public:
  using mlir::Type::Type;
  static bool classof(Type type);
};

}  // namespace iree_pydm
}  // namespace mlir

// Include generated dialect code (this comment blocks clang-format from
// clobbering order).
#include "iree-dialects/Dialect/IREEPyDM/IR/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "iree-dialects/Dialect/IREEPyDM/IR/Types.h.inc"

namespace mlir {
namespace iree_pydm {

inline bool PrimitiveType::classof(Type type) {
  // Must corresponds with each subclass.
  return type.isa<BoolType, BytesType, IntegerType, ExceptionResultType,
                  ListType, NoneType, RealType, StrType, TupleType, TypeType>();
}

}  // namespace iree_pydm
}  // namespace mlir

#endif  // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_DIALECT_H
