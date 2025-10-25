// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTATTRINTERFACES_H_
#define IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTATTRINTERFACES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

namespace mlir::iree_compiler::IREE::TensorExt {

namespace detail {
LogicalResult getSparseStridesAndOffsets(Attribute attr,
                                         ArrayRef<int64_t> shape,
                                         SmallVectorImpl<int64_t> &strides,
                                         int64_t &offset);
}

} // namespace mlir::iree_compiler::IREE::TensorExt

// clang-format off: must be included after all LLVM/MLIR headers
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtAttrInterfaces.h.inc" // IWYU pragma: keep
// clang-format on: must be included after all LLVM/MLIR headers

#endif
