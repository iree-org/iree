// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>

#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
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

FailureOr<Operation *>
mlir::iree_compiler::IREE::LinalgExt::ForallOpToAsyncRewriter::
    returningMatchAndRewrite(scf::ForallOp forallOp,
                             PatternRewriter &rewriter) const {
  if (forallOp.getNumResults() > 0)
    return forallOp->emitError("only bufferized scf.forall lowers to async");

  if (forallOp.getRank() > 1)
    return forallOp->emitError(
        "only single-dimension scf.forall lowers to async");

  // Only consider the top level ForallOp op and skip if it already
  // contains an ExecuteOp.
  if (forallOp->getParentOfType<scf::ForallOp>() ||
      llvm::any_of(forallOp.getBody()->getOperations(),
                   [](Operation &op) { return isa<async::ExecuteOp>(&op); }))
    return failure();

  auto *ctx = forallOp.getContext();
  Location loc = forallOp.getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  // TODO: allow multi-dim.
  Value numThreads = forallOp.getUpperBound(rewriter).front();

  // Wrap the scf.forall into an async::ExecuteOp.
  // 1. Create the async::GroupType object on which we synchronize.
  Value asyncGroup = rewriter.create<async::CreateGroupOp>(
      loc, async::GroupType::get(ctx), numThreads);

  // 2. Create a bodyless forOp.
  scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, zero, numThreads, one);
  rewriter.setInsertionPointToStart(forOp.getBody());

  // 3. Create an empty executeOp, nested within the forOp.
  auto noopExec = [&](OpBuilder &executeBuilder, Location executeLoc,
                      ValueRange executeArgs) {};
  auto executeOp =
      rewriter.create<async::ExecuteOp>(loc, /*resultTypes=*/TypeRange(),
                                        /*dependencies=*/ValueRange(),
                                        /*operands=*/ValueRange(), noopExec);

  // 3. Steal the ops nested under scf::Forall, except the terminator,
  // into the body of the async::ExecuteOp, just before the terminator.
  SmallVector<Value> bbArgsTranslated{forOp.getInductionVar()};
  rewriter.mergeBlocks(&forallOp.getRegion().front(), executeOp.getBody(),
                       bbArgsTranslated);
  // 3.b. Erase the terminator stolen from forallOp.
  rewriter.eraseOp(&executeOp.getBody()->back());
  // 3.c. Erase forallOp.
  rewriter.eraseOp(forallOp);
  // 3.d. Add ExecuteOp terminator.
  rewriter.setInsertionPointToEnd(executeOp.getBody());
  rewriter.create<async::YieldOp>(loc, ValueRange{});
  // 3.e. Add to group within the loop.
  rewriter.setInsertionPoint(forOp.getBody()->getTerminator());
  rewriter.create<async::AddToGroupOp>(loc, rewriter.getIndexType(),
                                       executeOp.getToken(), asyncGroup);

  // 4. After the iree_compiler::IREE::LinalgExt::Forall, await all async
  // tasks in `asyncGroup`.
  rewriter.setInsertionPointAfter(forOp);
  return rewriter.create<async::AwaitAllOp>(loc, asyncGroup).getOperation();
}
