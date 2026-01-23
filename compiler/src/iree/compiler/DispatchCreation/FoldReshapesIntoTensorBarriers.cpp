// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FOLDRESHAPESINTOTENSORBARRIERSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

// Move tensor.expand_shape/collapse_shape above compute_barrier.start
struct MoveReshapeAboveBarrierStart : public RewritePattern {
  MoveReshapeAboveBarrierStart(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op)) {
      return failure();
    }

    auto barrierStartOp =
        op->getOperand(0)
            .getDefiningOp<IREE::TensorExt::ComputeBarrierStartOp>();
    if (!barrierStartOp) {
      return failure();
    }

    // Update reshape's operand to use compute_barrier.start's input
    rewriter.modifyOpInPlace(
        op, [&]() { op->getOpOperand(0).set(barrierStartOp.getValue()); });

    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    Value reshapeResult = op->getResult(0);
    auto newBarrier = IREE::TensorExt::ComputeBarrierStartOp::create(
        rewriter, barrierStartOp.getLoc(), reshapeResult);

    DominanceInfo domInfo(op);
    rewriter.replaceUsesWithIf(reshapeResult, newBarrier.getResult(),
                               [&](OpOperand &use) {
                                 return domInfo.properlyDominates(
                                     newBarrier.getOperation(), use.getOwner());
                               });
    return success();
  }
};

// Move tensor.expand_shape/collapse_shape below compute_barrier.end
struct MoveReshapeBelowBarrierEnd
    : public OpRewritePattern<IREE::TensorExt::ComputeBarrierEndOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(IREE::TensorExt::ComputeBarrierEndOp barrierEndOp,
                  PatternRewriter &rewriter) const override {
    Operation *reshapeOp = barrierEndOp.getValue().getDefiningOp();
    if (!reshapeOp ||
        !isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(reshapeOp)) {
      return failure();
    }
    Value reshapeSrc = reshapeOp->getOperand(0);

    // Create a new compute_barrier.end before the reshape with the reshape's
    // source type
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(reshapeOp);
    auto newBarrier = IREE::TensorExt::ComputeBarrierEndOp::create(
        rewriter, barrierEndOp.getLoc(), reshapeSrc);

    rewriter.modifyOpInPlace(reshapeOp, [&]() {
      reshapeOp->getOpOperand(0).set(newBarrier.getResult());
    });
    rewriter.replaceOp(barrierEndOp, reshapeOp->getResult(0));
    return success();
  }
};

struct FoldReshapesIntoTensorBarriersPass final
    : public impl::FoldReshapesIntoTensorBarriersPassBase<
          FoldReshapesIntoTensorBarriersPass> {
  using Base::Base;

  void runOnOperation() override {
    auto funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<MoveReshapeAboveBarrierStart, MoveReshapeBelowBarrierEnd>(
        funcOp.getContext());

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("failed to fold reshapes into tensor barriers");
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
