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

static void replaceForWithIf(PatternRewriter &rewriter, Operation *op,
                             scf::IfOp ifOp, Region &region,
                             ValueRange initArgs, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  rewriter.inlineBlockBefore(block, &ifOp.getThenRegion().front(),
                             ifOp.getThenRegion().front().begin(), blockArgs);
  if (initArgs.size() == 0) {
    rewriter.eraseOp(terminator);
  } else {
    rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    rewriter.create<scf::YieldOp>(ifOp.getLoc(), initArgs);
  }
  rewriter.replaceOp(op, ifOp);
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
  // Calculate loop bounds that allow for maximum iterations.
  auto lb = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::LB, op.getLowerBound());
  auto step = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::LB, op.getStep());
  auto ub = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, op.getUpperBound(),
      /*stopCondition=*/nullptr,
      /*closedUB=*/true);
  if (failed(lb) || failed(step) || failed(ub)) {
    return false;
  }
  // If the upper bound is less than or equal to lower bound  + step then the
  // loop cannot run for a second iteration assuming the step is positive.
  FailureOr<bool> isUbUnderStep = ValueBoundsConstraintSet::compare(
      getAsIndexOpFoldResult(op.getContext(), *ub),
      ValueBoundsConstraintSet::LE,
      getAsIndexOpFoldResult(op.getContext(), *step + *lb));
  FailureOr<bool> isStepNonNegative = ValueBoundsConstraintSet::compare(
      getAsOpFoldResult(op.getStep()), ValueBoundsConstraintSet::GE,
      getAsIndexOpFoldResult(op.getContext(), 0));
  return isUbUnderStep.value_or(false) && isStepNonNegative.value_or(false);
}

namespace {
/// Rewriting pattern that replaces single-iteration loops with their bodies.
struct SimplifyTrivialLoops : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (!(neverRunsSecondIteration(op))) {
      return failure();
    }

    // The second iteration is never run
    // so the loop atmost can have 1 iteration. Inline its body and remove the
    // loop.
    SmallVector<Value> blockArgs;
    blockArgs.reserve(op.getInitArgs().size() + 1);
    blockArgs.push_back(op.getLowerBound());
    llvm::append_range(blockArgs, op.getInitArgs());
    if (alwaysRunsFirstIteration(op)) {
      replaceOpWithRegion(rewriter, op, op.getRegion(), blockArgs);
    } else {
      Value count = rewriter.create<arith::CmpIOp>(
          op->getLoc(), arith::CmpIPredicate::sgt, op.getUpperBound(),
          op.getLowerBound());
      auto ifOp = rewriter.create<scf::IfOp>(
          op->getLoc(), op.getResultTypes(), count,
          /*withElseRegion=*/op.getInitArgs().size() != 0);
      replaceForWithIf(rewriter, op, ifOp, op.getRegion(), op.getInitArgs(),
                       blockArgs);
    }
    return success();
  }
};

} // namespace

void populateRemoveSingleIterationLoopPattern(RewritePatternSet &patterns) {
  patterns.add<SimplifyTrivialLoops>(patterns.getContext());
}

} // namespace mlir::iree_compiler
