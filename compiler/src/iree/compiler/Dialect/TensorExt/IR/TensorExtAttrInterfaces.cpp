// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtAttrInterfaces.h"

#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::TensorExt {

namespace detail {

LogicalResult getSparseStridesAndOffsets(Attribute sparseAttr,
                                         ArrayRef<int64_t> shape,
                                         SmallVectorImpl<int64_t> &strides,
                                         int64_t &offset) {
  return failure();
}
} // namespace detail

} // namespace mlir::iree_compiler::IREE::TensorExt

// clang-format off: must be included after all LLVM/MLIR headers
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtAttrInterfaces.cpp.inc" // IWYU pragma: keep
// clang-format on: must be included after all LLVM/MLIR headers
