// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_TOIREE_TYPECONVERTER_H
#define IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_TOIREE_TYPECONVERTER_H

#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace PYDM {

class LoweringTypeConverter : public mlir::TypeConverter {
public:
  enum class WeakFloatType {
    F32,
    F64,
  };
  LoweringTypeConverter();

  // Type mappings for builtin, weakly typed integer and floating point types.
  Type getBoolType(Builder b) const;
  Type getWeakIntegerType(Builder b) const;
  Type getWeakFloatType(Builder b) const;

  // Whether the given type is a valid lowered type.
  bool isTypeLegal(Type t) const;
  bool areTypesLegal(TypeRange types) const;

private:
  bool boolBits = 32;
  int weakIntegerBits = 32;
  WeakFloatType weakFloatType = WeakFloatType::F32;
};

} // namespace PYDM
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_TOIREE_TYPECONVERTER_H
