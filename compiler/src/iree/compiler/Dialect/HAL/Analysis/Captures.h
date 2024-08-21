// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_ANALYSIS_CAPTURES_H_
#define IREE_COMPILER_DIALECT_HAL_ANALYSIS_CAPTURES_H_

#include "mlir/IR/Value.h"

namespace mlir::iree_compiler::IREE::HAL {

// Describes the origin of an SSA value within a region.
enum class ValueOrigin {
  // Value origin is unknown or dynamic.
  Unknown,
  // Value is produced by a constant-like op in the local scope.
  LocalConstant,
  // Value is loaded from a mutable global.
  MutableGlobal,
  // Value is loaded from an immutable global.
  ImmutableGlobal,
};

// Categories a value based on the operation in the local scope producing it.
ValueOrigin categorizeValue(Value value);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_ANALYSIS_CAPTURES_H_
