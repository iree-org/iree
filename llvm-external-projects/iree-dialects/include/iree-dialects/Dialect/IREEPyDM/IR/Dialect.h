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
enum class BuiltinTypeCode : int {
  Bool = 1,
  Bytes,
  ExceptionResult,
  Integer,
  List,
  None,
  Object,
  Real,
  Str,
  Tuple,
  Type,

  FirstCustom = 100,
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
