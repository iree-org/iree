// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

namespace {

SmallVector<Value> getValuesToYield(scf::InParallelOp op) {
  return llvm::map_to_vector(op.getYieldingOps(), [](Operation &op) -> Value {
    return cast<tensor::ParallelInsertSliceOp>(&op).getDest();
  });
}

} // namespace

FailureOr<scf::ForOp> ForallOpToScfForRewriter::returningMatchAndRewrite(
    scf::ForallOp forallOp, PatternRewriter &rewriter) const {
  if (forallOp.getNumResults() > 0)
    return forallOp->emitError("only bufferized scf.forall lowers to scf.for");

  if (forallOp.getRank() > 1)
    return forallOp->emitError(
        "only single-dimension scf.forall lowers to scf.for");

  // Construct the loop bounds based on the canonical arithmetic progression.
  Location loc = forallOp.getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  // TODO: allow multi-dim.
  Value numThreads = forallOp.getUpperBound(rewriter).front();

  // Construct the op without a body builder: we need to clone the ops in the
  // body explicitly after having access to the new bbArgs.
  // As a consequence, `ensureTerminator` is not called and the `forOp` body
  // has no terminator.
  scf::InParallelOp InParallelOp = forallOp.getTerminator();
  SmallVector<Value> valuesToYield = getValuesToYield(InParallelOp);
  scf::ForOp forOp =
      rewriter.create<scf::ForOp>(loc, zero, numThreads, one, valuesToYield);

  // Move the body while replacing the threadId by the forOp iv.
  SmallVector<Value> bbArgsTranslated{forOp.getInductionVar()};
  Block *body = forOp.getBody();
  bool hasTerminator =
      !body->empty() && body->back().hasTrait<OpTrait::IsTerminator>();
  if (hasTerminator) {
    rewriter.inlineBlockBefore(&forallOp.getRegion().front(),
                               body->getTerminator(), bbArgsTranslated);
  } else {
    rewriter.mergeBlocks(&forallOp.getRegion().front(), body, bbArgsTranslated);
  }

  rewriter.setInsertionPointToStart(body);
  IRMapping bvm;
  bvm.map(valuesToYield, forOp.getRegionIterArgs());

  // Create sequential insertSlice ops.
  SmallVector<Value> toYield;
  rewriter.setInsertionPoint(InParallelOp);
  for (Operation &operation : InParallelOp.getYieldingOps()) {
    tensor::ParallelInsertSliceOp op =
        cast<tensor::ParallelInsertSliceOp>(&operation);
    toYield.push_back(rewriter.createOrFold<tensor::InsertSliceOp>(
        loc, op.getSource(), bvm.lookup(op.getDest()), op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides()));
  }

  // InParallelOp.yieldedValues come from above, not from bbArgs.
  // There is no rewriter method to make mergeBlocks update non-bbArgs.
  // Need to manually clone + bvm all uses that are now nested under forOp.
  // Warning: this replacement is currently optimistic and may change the
  // semantics as explained in the pass description in Passes.td.
  SmallVector<Operation *> opsToReplace;
  for (Value toReplace : valuesToYield) {
    for (OpOperand &u : toReplace.getUses()) {
      Operation *op = u.getOwner();
      if (!forOp->isProperAncestor(op))
        continue;
      opsToReplace.push_back(op);
    }
  }
  for (Operation *op : opsToReplace) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    Operation *cloned = rewriter.clone(*op, bvm);
    rewriter.replaceOp(op, cloned->getResults());
  }

  // Insert terminator.
  if (!hasTerminator) {
    rewriter.setInsertionPointToEnd(body);
    rewriter.create<scf::YieldOp>(loc, toYield);
  }

  // Cleanup and replace.
  rewriter.eraseOp(InParallelOp);
  rewriter.replaceOp(forallOp, forOp.getResults());

  return forOp;
}
