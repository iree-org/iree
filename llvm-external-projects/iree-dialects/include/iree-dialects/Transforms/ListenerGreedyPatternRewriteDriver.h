// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===---------------------------------------------------------------------===//
// BEGIN implementation modified
//===---------------------------------------------------------------------===//
// This is modified version of GreedyPatternRewriterDriver.h. A RewriteLister
// is passed to all entry points.
//===---------------------------------------------------------------------===//
// END implementation modified
//===---------------------------------------------------------------------===//

#include "iree-dialects/Transforms/Listener.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// The following are iree-dialects extensions to MLIR.
namespace mlir {
struct RewriteListener;

/// Rewrite ops in the given region, which must be isolated from above, by
/// repeatedly applying the highest benefit patterns in a greedy work-list
/// driven manner.
///
/// This variant may stop after a predefined number of iterations, see the
/// alternative below to provide a specific number of iterations before stopping
/// in absence of convergence.
///
/// Return success if the iterative process converged and no more patterns can
/// be matched in the result operation regions.
///
/// Note: This does not apply patterns to the top-level operation itself.
///       These methods also perform folding and simple dead-code elimination
///       before attempting to match any of the provided patterns.
///
/// You may configure several aspects of this with GreedyRewriteConfig.
LogicalResult applyPatternsTrackAndFoldGreedily(
    RewriteListener *listener, Region &region,
    const FrozenRewritePatternSet &patterns,
    GreedyRewriteConfig config = GreedyRewriteConfig());

/// Rewrite ops in all regions of the given op, which must be isolated from
/// above.
inline LogicalResult applyPatternsTrackAndFoldGreedily(
    RewriteListener *listener, Operation *op,
    const FrozenRewritePatternSet &patterns,
    GreedyRewriteConfig config = GreedyRewriteConfig()) {
  bool failed = false;
  for (Region &region : op->getRegions())
    failed |=
        applyPatternsTrackAndFoldGreedily(listener, region, patterns, config)
            .failed();
  return failure(failed);
}

/// Applies the specified rewrite patterns on `ops` while also trying to fold
/// these ops.
///
/// Newly created ops and other pre-existing ops that use results of rewritten
/// ops or supply operands to such ops are simplified, unless such ops are
/// excluded via `config.strictMode`. Any other ops remain unmodified (i.e.,
/// regardless of `strictMode`).
///
/// In addition to strictness, a region scope can be specified. Only ops within
/// the scope are simplified. This is similar to `applyPatternsAndFoldGreedily`,
/// where only ops within the given regions are simplified. If no scope is
/// specified, it is assumed to be the first common enclosing region of the
/// given ops.
///
/// Note that ops in `ops` could be erased as result of folding, becoming dead,
/// or via pattern rewrites. If more far reaching simplification is desired,
/// applyPatternsAndFoldGreedily should be used.
///
/// Returns success if the iterative process converged and no more patterns can
/// be matched. `changed` is set to true if the IR was modified at all.
/// `allOpsErased` is set to true if all ops in `ops` were erased.
LogicalResult
applyOpPatternsTrackAndFold(RewriteListener *listener,
                            ArrayRef<Operation *> ops,
                            const FrozenRewritePatternSet &patterns,
                            GreedyRewriteConfig config = GreedyRewriteConfig(),
                            bool *changed = nullptr, bool *allErased = nullptr);

} // namespace mlir
