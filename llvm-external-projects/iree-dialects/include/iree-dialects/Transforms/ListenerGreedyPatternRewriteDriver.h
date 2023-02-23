// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Transforms/Listener.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// The following are iree-dialects extensions to MLIR.
namespace mlir {
struct GreedyRewriteConfig;

/// Applies the specified patterns on `op` alone while also trying to fold it,
/// by selecting the highest benefits patterns in a greedy manner. Returns
/// success if no more patterns can be matched. `erased` is set to true if `op`
/// was folded away or erased as a result of becoming dead. Note: This does not
/// apply any patterns recursively to the regions of `op`. Accepts a listener
/// so the caller can be notified of rewrite events.
LogicalResult applyPatternsAndFoldGreedily(
    Operation *op, const FrozenRewritePatternSet &patterns,
    const GreedyRewriteConfig &config, RewriteListener *listener);

/// Apply the given list of transformations to the regions of the
/// isolated-from-above operation `root` greedily until convergence. Update
/// Linalg operations in values of `trackedOperations` if they are replaced by
/// other Linalg operations during the rewriting process. Tracked operations
/// must be replaced with Linalg operations and must not be erased in the
/// patterns.
static inline LogicalResult applyPatternsTrackAndFoldGreedily(
    Operation *root, RewriteListener &listener,
    const FrozenRewritePatternSet &patterns,
    GreedyRewriteConfig config = GreedyRewriteConfig()) {
  return applyPatternsAndFoldGreedily(root, patterns, config, &listener);
}

} // namespace mlir
