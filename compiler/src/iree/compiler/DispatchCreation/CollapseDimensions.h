// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DISPATCHCREATION_COLLAPSEDIMENSIONS_H_
#define IREE_COMPILER_DISPATCHCREATION_COLLAPSEDIMENSIONS_H_

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::DispatchCreation {

/// Returns the runs of loops of `op` that can be collapsed into one loop. A run
/// consists of loops with the same iterator type whose dimensions appear
/// consecutively, in the same order, in every indexing map that uses any of
/// them. Runs of a single loop are omitted.
///
/// `op` must implement `LinalgFusionOpInterface` and `TilingInterface`, and all
/// of its indexing maps must be projected permutations.
SmallVector<ReassociationIndices> getCollapsibleLoops(Operation *op);

} // namespace mlir::iree_compiler::DispatchCreation

#endif // IREE_COMPILER_DISPATCHCREATION_COLLAPSEDIMENSIONS_H_
