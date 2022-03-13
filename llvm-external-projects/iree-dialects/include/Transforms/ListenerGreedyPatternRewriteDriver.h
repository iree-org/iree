//===- ListenerGreedyPatternRewriteDriver.h - A greedy rewriter -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transforms/Listener.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

namespace mlir {
struct GreedyRewriteConfig;

/// Applies the specified patterns on `op` alone while also trying to fold it,
/// by selecting the highest benefits patterns in a greedy manner. Returns
/// success if no more patterns can be matched. `erased` is set to true if `op`
/// was folded away or erased as a result of becoming dead. Note: This does not
/// apply any patterns recursively to the regions of `op`. Accepts a listener
/// so the caller can be notified of rewrite events.
LogicalResult applyPatternsAndFoldGreedily(
    MutableArrayRef<Region> regions, const FrozenRewritePatternSet &patterns,
    const GreedyRewriteConfig &config, RewriteListener *listener);
inline LogicalResult applyPatternsAndFoldGreedily(
    Operation *op, const FrozenRewritePatternSet &patterns,
    const GreedyRewriteConfig &config, RewriteListener *listener) {
  return applyPatternsAndFoldGreedily(op->getRegions(), patterns, config,
                                      listener);
}

} // namespace mlir
