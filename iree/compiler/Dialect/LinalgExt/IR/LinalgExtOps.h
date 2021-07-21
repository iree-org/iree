// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace iree_compiler {
namespace linalg_ext {

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`.
Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim);

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`. If the shape is constant, returns the shape as an `IntegerAttr`.
OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim);

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h.inc"  // IWYU pragma: export

#endif  // IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_
