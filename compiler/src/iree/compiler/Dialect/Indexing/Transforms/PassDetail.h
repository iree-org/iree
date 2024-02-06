// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_INDEXING_TRANSFORMS_PASS_DETAIL_H_
#define IREE_COMPILER_DIALECT_INDEXING_TRANSFORMS_PASS_DETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Indexing {

#define GEN_PASS_CLASSES
#include "iree/compiler/Dialect/Indexing/Transforms/Passes.h.inc" // IWYU pragma: keep

} // namespace mlir::iree_compiler::IREE::Indexing

#endif // IREE_COMPILER_DIALECT_INDEXING_TRANSFORMS_PASS_DETAIL_H_
