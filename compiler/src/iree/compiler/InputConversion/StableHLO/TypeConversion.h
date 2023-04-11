// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_STABLEHLO_TYPE_CONVERSION_H
#define IREE_COMPILER_INPUTCONVERSION_STABLEHLO_TYPE_CONVERSION_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::stablehlo {

// Type converter to use as part of lowerings from dialects that carry signs
// in their types to those that are signless.
class RemoveSignTypeConverter : public TypeConverter {
 public:
  RemoveSignTypeConverter();
};

// Type converter which adds additional materializations (beyond signless)
// that are needed as part of the HloToLinalg conversion patterns.
// This is the type converter used by the test pass and is the sanctioned
// way to use the underlying patterns.
class LinalgTypeConverter : public RemoveSignTypeConverter {
 public:
  LinalgTypeConverter();
};

}  // namespace mlir::iree_compiler::stablehlo

#endif  // IREE_COMPILER_INPUTCONVERSION_STABLEHLO_TYPE_CONVERSION_H
