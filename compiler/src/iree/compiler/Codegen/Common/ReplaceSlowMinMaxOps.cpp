// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

template <class SlowMinMaxOp, class FastMinMaxOp>
struct ReplaceSlowWithFastMinMaxOpPattern final
    : public OpRewritePattern<SlowMinMaxOp> {
  using OpRewritePattern<SlowMinMaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SlowMinMaxOp slowOp,
                                PatternRewriter &rewriter) const override {
    OperationState state(slowOp->getLoc(), FastMinMaxOp::getOperationName(),
                         slowOp->getOperands(), slowOp->getResultTypes(),
                         slowOp->getAttrs());
    Operation *fastOp = rewriter.create(state);
    rewriter.replaceOp(slowOp, fastOp->getResults());
    return success();
  }
};

template <class SlowReductionOp>
struct ReplaceSlowWithFastReductionMinMaxOpPattern final
    : public OpRewritePattern<SlowReductionOp> {
  using OpRewritePattern<SlowReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SlowReductionOp slowReductionOp,
                                PatternRewriter &rewriter) const override {
    if (slowReductionOp.getKind() == vector::CombiningKind::MINIMUMF) {
      rewriter.modifyOpInPlace(slowReductionOp, [&]() {
        slowReductionOp.setKind(vector::CombiningKind::MINNUMF);
      });
      return success();
    }
    if (slowReductionOp.getKind() == vector::CombiningKind::MAXIMUMF) {
      rewriter.modifyOpInPlace(slowReductionOp, [&]() {
        slowReductionOp.setKind(vector::CombiningKind::MAXNUMF);
      });
      return success();
    }

    return failure();
  }
};

struct ReplaceSlowMinMaxOpsPass
    : public ReplaceSlowMinMaxOpsBase<ReplaceSlowMinMaxOpsPass> {
public:
  using ReplaceSlowMinMaxOpsBase::ReplaceSlowMinMaxOpsBase;
  void runOnOperation() override;
};

} // namespace

void mlir::iree_compiler::populateReplaceSlowMinMaxOpsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      ReplaceSlowWithFastMinMaxOpPattern<arith::MinimumFOp, arith::MinNumFOp>,
      ReplaceSlowWithFastMinMaxOpPattern<arith::MaximumFOp, arith::MaxNumFOp>,
      ReplaceSlowWithFastReductionMinMaxOpPattern<vector::ReductionOp>,
      ReplaceSlowWithFastReductionMinMaxOpPattern<vector::MultiDimReductionOp>>(
      patterns.getContext());
}

void ReplaceSlowMinMaxOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateReplaceSlowMinMaxOpsPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::iree_compiler::createReplaceSlowMinMaxOpsPass() {
  return std::make_unique<ReplaceSlowMinMaxOpsPass>();
}
