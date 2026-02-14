// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- LoopMappingUtils.h - Loop mapping utilities ---------------------===//
//
// Utilities for computing loop iteration space mappings between operations.
// This is useful for fusion, tiling, and other transformations that need to
// understand how the loops of one operation relate to the loops of another.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_LOOPMAPPINGUTILS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_LOOPMAPPINGUTILS_H_

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Returns a bit vector of size number of loops of the operation with
/// the bits corresponding to outer parallel loops set to `true`.
llvm::SmallBitVector getOuterParallelLoops(Operation *op);

/// Computes the affine map from a root operation's outer parallel loops to a
/// candidate operation's iteration space given a MapVector containing the
/// pre-existing loop mappings.
///
/// Returns: The affine map from root's outer parallel loops to candidateOp's
///          iteration space, or failure if no valid mapping exists.
///
/// Example mapping: affine_map<(d0, d1) -> (d0, 0, d1)>
///   The root has 2 outer parallel loops (d0, d1). The candidate has 3 loops
///   where the 1st and 3rd map to d0 and d1, while the 2nd has no mapping to
///   the root's outer parallel loops (represented as 0).
FailureOr<AffineMap> getRootParallelLoopToOpMap(
    Operation *candidateOp,
    const llvm::MapVector<Operation *, AffineMap> &loopMaps);

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_LOOPMAPPINGUTILS_H_
