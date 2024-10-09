// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SAFELOOPINVARIANTCODEMOTIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
/// Safer IREE loop invariant code motion (LICM) pass.
struct SafeLoopInvariantCodeMotionPass
    : public impl::SafeLoopInvariantCodeMotionPassBase<
          SafeLoopInvariantCodeMotionPass> {
  void runOnOperation() override;
};
} // namespace

void SafeLoopInvariantCodeMotionPass::runOnOperation() {
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  //
  // Hoisting is only performed on loops with guaranteed non-zero trip counts.
  // `scf.forall` ops with mapping attributes can never be proven to have a
  // non-zero trip count until the loop is resolved and is blanket included
  // here.
  getOperation()->walk([&](LoopLikeOpInterface loopLike) {
    if (auto forallOp = dyn_cast<scf::ForallOp>(*loopLike)) {
      if (forallOp.getMapping()) {
        return;
      }
    }

    // Skip loops without lower/upper bounds. There is no generic way to verify
    // whether a loop has at least one trip so new loop types of interest can be
    // added as needed. For example, `scf.while` needs non-trivial analysis of
    // its condition region to know that it has at least one trip.
    std::optional<SmallVector<OpFoldResult>> maybeLowerBounds =
        loopLike.getLoopLowerBounds();
    std::optional<SmallVector<OpFoldResult>> maybeUpperBounds =
        loopLike.getLoopUpperBounds();
    if (!maybeLowerBounds || !maybeUpperBounds) {
      return;
    }

    // If any lower + upper bound pair cannot be definitely verified as lb < ub
    // then the loop may have a zero trip count.
    for (auto [lb, ub] :
         llvm::zip_equal(*maybeLowerBounds, *maybeUpperBounds)) {
      if (!ValueBoundsConstraintSet::compare(lb, ValueBoundsConstraintSet::LT,
                                             ub)) {
        return;
      }
    }

    moveLoopInvariantCode(loopLike);
  });
}

} // namespace mlir::iree_compiler
