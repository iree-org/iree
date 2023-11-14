// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_CPUUTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_CPUUTILS_H_

#include "iree/compiler/Codegen/Utils/Utils.h"

namespace mlir {
namespace iree_compiler {

/// Find the root operation for the dispatch region. The priority is:
///   1. A Linalg operation that has reduction loops.
///   2. Any other Linalg op or LinalgExt op.
///   3. An operation that implements TilingInterface.
/// If there are multiple operations meeting the same priority, the one closer
/// to the end of the function is the root op.
FailureOr<Operation *> getRootOperation(ArrayRef<Operation *> computeOps);

/// Adjusts the tile sizes (carried by `rootOp`) to be aligned with
/// tensor.unpack inner tile sizes, if there are tensor.unpack producers. If the
/// tile sizes are not aligned, a stack buffer is needed because of
/// tensor.unpack tiling implementations.
LogicalResult adjustTileSizesForUnPackOp(func::FuncOp entryPointFn,
                                         Operation *rootOp);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_UTILS_CPUUTILS_H_
