// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- RemoveSingleIterationLoop.cpp - Remove single iteration loops ------===//
//
// Removes loops that are known to be single-trip count even when the loop
// itself might be distributed.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

#define DEBUG_TYPE "iree-codegen-remove-single-iteration"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir::iree_compiler {

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Return true if we can prove that the we always run at least the first
/// iteration of the ForOp.
static bool alwaysRunsFirstIteration(scf::ForOp op) {
  // Can't perform the analysis if the loops's bounds aren't index-typed.
  if (!op.getInductionVar().getType().isIndex())
    return false;
  FailureOr<bool> isLb = ValueBoundsConstraintSet::compare(
      getAsOpFoldResult(op.getLowerBound()), ValueBoundsConstraintSet::LT,
      getAsOpFoldResult(op.getUpperBound()));
  return isLb.value_or(false);
}

/// Return true if we can prove that the we never run more than one iteration of
/// the ForOp.
static bool neverRunsSecondIteration(scf::ForOp op) {
  // Can't perform the analysis if the loops's bounds aren't index-typed.
  if (!op.getInductionVar().getType().isIndex())
    return false;
  // If the upper bound (ub) is less than or equal to the loop step, then
  // lower bound  + step must be greater than the upper bound, assuming the
  // lower bound is non-negative.
  FailureOr<bool> isUbUnderStep = ValueBoundsConstraintSet::compare(
      getAsOpFoldResult(op.getUpperBound()), ValueBoundsConstraintSet::LE,
      getAsOpFoldResult(op.getStep()));
  FailureOr<bool> isLbNonNegative = ValueBoundsConstraintSet::compare(
      getAsOpFoldResult(op.getLowerBound()), ValueBoundsConstraintSet::GE,
      getAsIndexOpFoldResult(op.getContext(), 0));
  return isUbUnderStep.value_or(false) && isLbNonNegative.value_or(false);
}

namespace {
/// Rewriting pattern that replaces single-iteration loops with their bodies.
struct SimplifyTrivialLoops : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Handle the case where we know that the loop doesn't run more than
    // once but the loop may not run at least once by replace the `loop` with an
    // `if`.
    if (!(alwaysRunsFirstIteration(op) && neverRunsSecondIteration(op))) {
      return failure();
    }

    // The first iteration is always run and the second iteration is never run
    // so the loop always have 1 iteration. Inline its body and remove the loop.
    SmallVector<Value> blockArgs;
    blockArgs.reserve(op.getInitArgs().size() + 1);
    blockArgs.push_back(op.getLowerBound());
    llvm::append_range(blockArgs, op.getInitArgs());
    replaceOpWithRegion(rewriter, op, op.getRegion(), blockArgs);
    return success();
  }
};

} // namespace

void populateRemoveSingleIterationLoopPattern(RewritePatternSet &patterns) {
  patterns.add<SimplifyTrivialLoops>(patterns.getContext());
}

} // namespace mlir::iree_compiler
