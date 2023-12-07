// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- BubbleUpOrdinalOpPass.cpp -----------------------------------------===//
//
// The workgroup count computation when using slices needs the ordinal
// annotation ops to be bubbled up as much as possible. This pass implements
// patterns to bubble these operations up.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

/// Replace the following sequence
///
/// ```mlir
/// %1 = <cast> %0 : .. to index
/// %2 = flow.dispatch.workload.ordinal %1, 0
/// %3 = <some_op>(...%1)...
/// ```
///
/// with
///
/// ```mlir
/// %1 = <cast> %0 : .. to index
/// %2 = flow.dispatch.workload.ordinal %1, 0
/// %3 = <some_op>(...%2)...
/// ```
///
/// to make all the uses flow through `flow.dispatch.workload.ordinal` ops.
template <typename CastOpTy>
struct BubbleUpAcrossCastOp
    : public OpRewritePattern<IREE::Flow::DispatchWorkloadOrdinalOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Flow::DispatchWorkloadOrdinalOp ordinalOp,
                                PatternRewriter &rewriter) const override {
    auto sourceCastOp = ordinalOp.getOperand().getDefiningOp<CastOpTy>();
    if (!sourceCastOp || sourceCastOp->hasOneUse()) {
      return failure();
    }

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(sourceCastOp);
    Location loc = ordinalOp.getLoc();
    Value reverseCastOp = rewriter.create<CastOpTy>(
        loc, rewriter.getIndexType(), sourceCastOp.getIn());
    Value newOrdinalOp = rewriter.create<IREE::Flow::DispatchWorkloadOrdinalOp>(
        loc, reverseCastOp, ordinalOp.getOrdinal());
    rewriter.replaceOp(sourceCastOp, newOrdinalOp);
    rewriter.replaceOp(ordinalOp, newOrdinalOp);
    return success();
  }
};

struct BubbleUpOrdinalOpsPass
    : public BubbleUpOrdinalOpsBase<BubbleUpOrdinalOpsPass> {
  void runOnOperation() override;
};
} // namespace

void BubbleUpOrdinalOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<BubbleUpAcrossCastOp<arith::IndexCastUIOp>>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createBubbleUpOrdinalOpsPass() {
  return std::make_unique<BubbleUpOrdinalOpsPass>();
}

} // namespace mlir::iree_compiler
