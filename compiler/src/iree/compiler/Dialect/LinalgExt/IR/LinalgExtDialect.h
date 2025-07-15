// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTDIALECT_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// clang-format off: must be included after all LLVM/MLIR headers
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::LinalgExt {

//===---------------------------------------------------------------------===//
// These attributes represent user-hints for certain optimizations to kick in
//===---------------------------------------------------------------------===//

/// Attribute set/get methods for specifying that the reduction dimensions
/// of the operation are to be split to execute as parallel partial reduction
// followed by a combined step.
void setSplitReductionAttribute(Operation *op, ArrayRef<int64_t> splitSize);
std::optional<SmallVector<int64_t>> getSplitReductionSizes(Operation *op);

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTDIALECT_H_
