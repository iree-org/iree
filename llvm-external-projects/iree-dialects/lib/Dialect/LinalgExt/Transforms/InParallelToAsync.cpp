//===- InParallelToAsync.cpp - Rewrite InParallel as Async ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdlib>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Async/IR/Async.h"
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

FailureOr<Operation *> mlir::iree_compiler::IREE::LinalgExt::
    InParallelOpToAsyncRewriter::returningMatchAndRewrite(
        iree_compiler::IREE::LinalgExt::InParallelOp inParallelOp,
        PatternRewriter &rewriter) const {
  assert(inParallelOp.getNumResults() == 0 &&
         "expected bufferized InParallelOp");

  // Only consider the top level InParallelOp op and skip if it already
  // contains an ExecuteOp.
  if (inParallelOp
          ->getParentOfType<iree_compiler::IREE::LinalgExt::InParallelOp>() ||
      llvm::any_of(inParallelOp.getBody()->getOperations(),
                   [](Operation &op) { return isa<async::ExecuteOp>(&op); }))
    return failure();

  auto *ctx = inParallelOp.getContext();
  Location loc = inParallelOp.getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value numThreads = inParallelOp.num_threads();

  // Wrap the linalg_ext.in_parallel into an async::ExecuteOp.
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

  // 3. Steal the iree_compiler::IREE::LinalgExt::InParallel ops, except the
  // terminator, into the body of the async::ExecuteOp, just before the
  // terminator.
  SmallVector<Value> bbArgsTranslated{forOp.getInductionVar()};
  rewriter.mergeBlocks(&inParallelOp.region().front(), executeOp.getBody(),
                       bbArgsTranslated);
  // 3.b. Erase the terminator stolen from inParallelOp.
  rewriter.eraseOp(&executeOp.getBody()->back());
  // 3.c. Erase inParallelOp.
  rewriter.eraseOp(inParallelOp);
  // 3.d. Add ExecuteOp terminator.
  rewriter.setInsertionPointToEnd(executeOp.getBody());
  rewriter.create<async::YieldOp>(loc, ValueRange{});
  // 3.e. Add to group within the loop.
  rewriter.setInsertionPoint(forOp.getBody()->getTerminator());
  rewriter.create<async::AddToGroupOp>(loc, rewriter.getIndexType(),
                                       executeOp.token(), asyncGroup);

  // 4. After the iree_compiler::IREE::LinalgExt::InParallel, await all async
  // tasks in `asyncGroup`.
  rewriter.setInsertionPointAfter(forOp);
  return rewriter.create<async::AwaitAllOp>(loc, asyncGroup).getOperation();
}
