// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONVERSION_CODEGENUTILS_FUNCTIONUTILS_H_
#define IREE_COMPILER_CONVERSION_CODEGENUTILS_FUNCTIONUTILS_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

/// Returns true if the given `func` is a kernel dispatch entry point.
bool isEntryPoint(FuncOp func);

/// Returns the number of outer parallel loops of a linalgOp.
unsigned getNumOuterParallelLoops(linalg::LinalgOp op);

/// Returns the entry point op for the `funcOp`. Returns `nullptr` on failure.
IREE::HAL::ExecutableEntryPointOp getEntryPoint(FuncOp funcOp);

/// Returns the untiled type of a tiled view for both tensor and memref
/// types. Either walks the `ViewOpInterface` chain (for memrefs) or the
/// `subtensor` op chain (for tensors).
Type getUntiledType(Value tiledView);

/// Returns the untiled type of a tiled view for both tensor and memref
/// types. Either walks the `ViewOpInterface` chain (for memrefs) or the
/// `subtensor` op chain (for tensors).
ArrayRef<int64_t> getUntiledShape(Value tiledView);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_CODEGENUTILS_FUNCTIONUTILS_H_
