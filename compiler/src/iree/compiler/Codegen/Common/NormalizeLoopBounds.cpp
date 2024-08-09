// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-codegen-normalize-loop-bounds"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_NORMALIZELOOPBOUNDSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static OpFoldResult emitNormalizedUpperBound(RewriterBase &rewriter,
                                             Location loc, OpFoldResult lb,
                                             OpFoldResult ub,
                                             OpFoldResult step) {
  AffineExpr d0, d1, d2;
  bindDims(rewriter.getContext(), d0, d1, d2);
  return affine::makeComposedFoldedAffineApply(
      rewriter, loc, (d0 - d1).ceilDiv(d2), {ub, lb, step});
}

/// Helper structure for storing the newly computed loop bounds.
namespace {
struct LoopRanges {
  SmallVector<OpFoldResult> lowerBounds;
  SmallVector<OpFoldResult> upperBounds;
  SmallVector<OpFoldResult> steps;
};
} // namespace

static FailureOr<LoopRanges>
emitNormalizedLoopBounds(RewriterBase &rewriter, Location loc, Block *body,
                         ValueRange ivs, ArrayRef<OpFoldResult> lbs,
                         ArrayRef<OpFoldResult> ubs,
                         ArrayRef<OpFoldResult> steps) {
  Attribute zero = rewriter.getIndexAttr(0);
  Attribute one = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> newLbs;
  SmallVector<OpFoldResult> newUbs;
  SmallVector<OpFoldResult> newSteps;
  for (auto &&[iv, lb, ub, step] : llvm::zip(ivs, lbs, ubs, steps)) {
    std::optional<int64_t> stepInt = getConstantIntValue(step);
    // Bail out on negative steps.
    if (!stepInt || stepInt.value() <= 0) {
      return failure();
    }

    // The lower bound and step of a normalized loop is always zero/one.
    newLbs.push_back(zero);
    newSteps.push_back(one);

    // Compute the normalized upper bound.
    OpFoldResult newUb = emitNormalizedUpperBound(rewriter, loc, lb, ub, step);
    newUbs.push_back(newUb);

    // Compute and replace the denormalized loop iterator argument in the loop
    // body with an insertion guard.
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(body);
      AffineExpr idx, stepExpr, lbExpr;
      bindDims(rewriter.getContext(), idx, stepExpr, lbExpr);
      affine::AffineApplyOp denormalizedIV = affine::makeComposedAffineApply(
          rewriter, loc, idx * stepExpr + lbExpr, {iv, step, lb});
      SmallPtrSet<Operation *, 2> preserve = {iv.getDefiningOp(),
                                              denormalizedIV};
      rewriter.replaceAllUsesExcept(iv, denormalizedIV.getResult(), preserve);
    }
  }
  return LoopRanges{newLbs, newUbs, newSteps};
}

/// Transform a `scf.for` loop with a strictly positive step
///   for %i = %lb to %ub step %s
/// into a 0-based loop with step 1
///   for %ii = 0 to ceildiv(%ub - %lb, %s) step 1
/// Insert an `affine.apply` operation to compute the denormalized index value.
static LogicalResult normalizeLoopBounds(RewriterBase &rewriter,
                                         scf::ForOp forOp) {
  OpBuilder::InsertionGuard g(rewriter);
  // Return if already normalized.
  std::optional<int64_t> lbInt = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> stepInt = getConstantIntValue(forOp.getStep());
  if (lbInt && stepInt && lbInt.value() == 0 && stepInt.value() == 1) {
    return success();
  }

  // Bail out on non-index types because the affine applies that are generated
  // require it.
  if (!isa<IndexType>(forOp.getInductionVar().getType())) {
    return failure();
  }

  Location loc = forOp.getLoc();

  rewriter.setInsertionPoint(forOp);
  FailureOr<LoopRanges> newLoopParams = emitNormalizedLoopBounds(
      rewriter, loc, forOp.getBody(), forOp.getInductionVar(),
      getAsOpFoldResult(forOp.getLowerBound()),
      getAsOpFoldResult(forOp.getUpperBound()),
      getAsOpFoldResult(forOp.getStep()));
  if (failed(newLoopParams)) {
    return failure();
  }

  assert(newLoopParams->lowerBounds.size() == 1 &&
         newLoopParams->upperBounds.size() == 1 &&
         newLoopParams->steps.size() == 1 &&
         "expected single range for scf.for");

  rewriter.modifyOpInPlace(forOp, [&]() {
    forOp.setLowerBound(getValueOrCreateConstantIndexOp(
        rewriter, loc, newLoopParams->lowerBounds.front()));
    forOp.setUpperBound(getValueOrCreateConstantIndexOp(
        rewriter, loc, newLoopParams->upperBounds.front()));
    forOp.setStep(getValueOrCreateConstantIndexOp(
        rewriter, loc, newLoopParams->steps.front()));
  });
  return success();
}

/// Transform a `scf.forall` loop with a strictly positive steps
///   forall (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1) step (%s0, %s1)
/// into a 0-based loop with step 1 (normalized)
///   forall (%i, %j) in (ceildiv(%ub0 - %lb0, %s0), ceildiv(%ub1 - %lb1, %s1))
/// Insert `affine.apply` operations to compute the denormalized index values.
static LogicalResult normalizeLoopBounds(RewriterBase &rewriter,
                                         scf::ForallOp forallOp) {
  OpBuilder::InsertionGuard g(rewriter);
  if (forallOp.isNormalized())
    return success();

  // `scf.forall` requires that all lbs/ubs/steps/ivs are index type so no need
  // to check here.

  rewriter.setInsertionPoint(forallOp);
  FailureOr<LoopRanges> newLoopParams = emitNormalizedLoopBounds(
      rewriter, forallOp.getLoc(), forallOp.getBody(),
      forallOp.getInductionVars(), forallOp.getMixedLowerBound(),
      forallOp.getMixedUpperBound(), forallOp.getMixedStep());
  if (failed(newLoopParams)) {
    return failure();
  }

  rewriter.setInsertionPointAfter(forallOp);
  auto newLoop = rewriter.create<scf::ForallOp>(
      rewriter.getUnknownLoc(), newLoopParams->lowerBounds,
      newLoopParams->upperBounds, newLoopParams->steps, forallOp.getOutputs(),
      forallOp.getMapping());
  rewriter.eraseOp(newLoop.getTerminator());
  rewriter.mergeBlocks(forallOp.getBody(), newLoop.getBody(),
                       newLoop.getBody()->getArguments());
  rewriter.replaceOp(forallOp, newLoop);

  return success();
}

namespace {
struct NormalizeLoopBoundsPass final
    : impl::NormalizeLoopBoundsPassBase<NormalizeLoopBoundsPass> {
  using impl::NormalizeLoopBoundsPassBase<
      NormalizeLoopBoundsPass>::NormalizeLoopBoundsPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    IRRewriter rewriter(op);
    if (normalizeFor) {
      op->walk([&](scf::ForOp forOp) {
        (void)normalizeLoopBounds(rewriter, forOp);
      });
    }
    if (normalizeForall) {
      op->walk([&](scf::ForallOp forallOp) {
        (void)normalizeLoopBounds(rewriter, forallOp);
      });
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
