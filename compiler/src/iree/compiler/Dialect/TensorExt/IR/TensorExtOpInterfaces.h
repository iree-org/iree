// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTOPINTERFACES_H_
#define IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTOPINTERFACES_H_

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::TensorExt {

class SparseCastOpInterface;
// Interface verification method to verify the sparse op satisfies
// interface constraints.
LogicalResult verifySparseCastOpInterface(SparseCastOpInterface sparseOp);

} // namespace mlir::iree_compiler::IREE::TensorExt

// clang-format off: must be included after all LLVM/MLIR headers
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOpInterfaces.h.inc" // IWYU pragma: keep
// clang-format on: must be included after all LLVM/MLIR headers

#endif
