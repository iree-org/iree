//===-- TrackingRewriteDriver.h - Special pattern rewriter ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGREWRITEDRIVER_H
#define MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGREWRITEDRIVER_H

#include "iree-dialects/Dialect/LinalgTransform/TransformOpMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
struct RewriteListener;

/// Apply the given list of transformations to the regions of the
/// isolated-from-above operation `root` greedily until convergence. Update
/// Linalg operations in values of `trackedOperations` if they are replaced by
/// other Linalg operations during the rewriting process. Tracked operations
/// must be replaced with Linalg operations and must not be erased in the
/// patterns.
LogicalResult applyPatternsTrackAndFoldGreedily(
    Operation *root, RewriteListener &listener,
    const FrozenRewritePatternSet &patterns,
    GreedyRewriteConfig config = GreedyRewriteConfig());
}  // namespace mlir

#endif  // MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGREWRITEDRIVER_H
