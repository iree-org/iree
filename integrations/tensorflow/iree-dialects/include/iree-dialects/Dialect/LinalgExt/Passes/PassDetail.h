// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

#define GEN_PASS_CLASSES

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h.inc" // IWYU pragma: keep

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_
