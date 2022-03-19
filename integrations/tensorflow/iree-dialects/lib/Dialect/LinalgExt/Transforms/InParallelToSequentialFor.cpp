//===- InParallelToSequentialFor.cpp.cpp - Rewrite InParallel as ForOp ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

namespace {

SmallVector<Value> getValuesToYield(PerformConcurrentlyOp op) {
  return llvm::to_vector(llvm::map_range(
      op.yieldingOps(), [](ParallelInsertSliceOp op) { return op.dest(); }));
}

}  // namespace

FailureOr<scf::ForOp> InParallelOpToScfForRewriter::returningMatchAndRewrite(
    InParallelOp inParallelOp, PatternRewriter &rewriter) const {
  // Construct the loop bounds based on the canonical arithmetic progression.
  Location loc = inParallelOp.getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value numThreads = inParallelOp.num_threads();

  // Construct the op without a body builder: we need to clone the ops in the
  // body explicitly after having access to the new bbArgs.
  // As a consequence, `ensureTerminator` is not called and the `forOp` body
  // has no terminator.
  PerformConcurrentlyOp performConcurrentlyOp = inParallelOp.getTerminator();
  SmallVector<Value> valuesToYield = getValuesToYield(performConcurrentlyOp);
  scf::ForOp forOp =
      rewriter.create<scf::ForOp>(loc, zero, numThreads, one, valuesToYield);

  // Move the body while replacing the threadId by the forOp iv.
  SmallVector<Value> bbArgsTranslated{forOp.getInductionVar()};
  Block *body = forOp.getBody();
  bool hasTerminator =
      !body->empty() && body->back().hasTrait<OpTrait::IsTerminator>();
  if (hasTerminator) {
    rewriter.mergeBlockBefore(&inParallelOp.region().front(),
                              body->getTerminator(), bbArgsTranslated);
  } else {
    rewriter.mergeBlocks(&inParallelOp.region().front(), body,
                         bbArgsTranslated);
  }

  rewriter.setInsertionPointToStart(body);
  BlockAndValueMapping bvm;
  bvm.map(valuesToYield, forOp.getRegionIterArgs());

  // Create sequential insertSlice ops.
  SmallVector<Value> toYield;
  rewriter.setInsertionPoint(performConcurrentlyOp);
  for (ParallelInsertSliceOp op : performConcurrentlyOp.yieldingOps()) {
    toYield.push_back(rewriter.createOrFold<tensor::InsertSliceOp>(
        loc, op.source(), bvm.lookup(op.dest()), op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides()));
  }

  // performConcurrentlyOp.yieldedValues come from above, not from bbArgs.
  // There is no rewriter method to make mergeBlocks update non-bbArgs.
  // Need to manually clone + bvm all uses that are now nested under forOp.
  // Warning: this replacement is currently optimistic and may change the
  // semantics as explained in the pass description in Passes.td.
  SmallVector<Operation *> opsToReplace;
  for (Value toReplace : valuesToYield) {
    for (OpOperand &u : toReplace.getUses()) {
      Operation *op = u.getOwner();
      if (!forOp->isProperAncestor(op)) continue;
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
  rewriter.eraseOp(performConcurrentlyOp);
  rewriter.replaceOp(inParallelOp, forOp.getResults());

  return forOp;
}
