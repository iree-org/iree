// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_TOIREE_TYPECONVERTER_H
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_TOIREE_TYPECONVERTER_H

#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_pydm {

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

 private:
  bool boolBits = 32;
  int weakIntegerBits = 32;
  WeakFloatType weakFloatType = WeakFloatType::F32;
};

}  // namespace iree_pydm
}  // namespace mlir

#endif  // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_TOIREE_TYPECONVERTER_H
