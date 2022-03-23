// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_DIALECT_H
#define IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_DIALECT_H

#include "iree-dialects/Dialect/PyDM/IR/Constants.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace PYDM {

/// Base class for all unboxed primitive types.
class PrimitiveType : public mlir::Type {
public:
  using mlir::Type::Type;
  static bool classof(Type type);
};

} // namespace PYDM
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

// Include generated dialect code (this comment blocks clang-format from
// clobbering order).
#include "iree-dialects/Dialect/PyDM/IR/PyDMDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "iree-dialects/Dialect/PyDM/IR/PyDMTypes.h.inc"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace PYDM {

inline bool PrimitiveType::classof(Type type) {
  // Must corresponds with each subclass.
  return type.isa<BoolType, BytesType, IntegerType, ExceptionResultType,
                  ListType, NoneType, RealType, StrType, TupleType, TypeType>();
}

} // namespace PYDM
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_DIALECT_H
