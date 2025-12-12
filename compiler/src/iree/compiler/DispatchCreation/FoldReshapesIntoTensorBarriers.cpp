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

// Move tensor.expand_shape/collapse_shape above compute_barrier
// when the barrier direction is Up and the appropriate flag is set
struct MoveReshapeAboveBarrier : public RewritePattern {
  MoveReshapeAboveBarrier(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op)) {
      return failure();
    }

    auto barrierOp =
        op->getOperand(0).getDefiningOp<IREE::TensorExt::ComputeBarrierOp>();
    if (!barrierOp) {
      return failure();
    }

    // Check direction: must be Up
    auto direction = barrierOp.getDirection();
    if (direction != IREE::TensorExt::BarrierDirection::Up) {
      return failure();
    }

    // Check flags: must have the appropriate flag set
    auto flags = barrierOp.getFlags();
    if ((isa<tensor::ExpandShapeOp>(op) &&
         !bitEnumContainsAny(
             flags,
             IREE::TensorExt::TransformationFlagBitfield::AllowExpand)) ||
        (isa<tensor::CollapseShapeOp>(op) &&
         !bitEnumContainsAny(
             flags,
             IREE::TensorExt::TransformationFlagBitfield::AllowCollapse))) {
      return failure();
    }

    // Update reshape's operand to use compute_barrier's input
    rewriter.modifyOpInPlace(
        op, [&]() { op->getOpOperand(0).set(barrierOp.getValue()); });

    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    Value reshapeResult = op->getResult(0);
    auto newBarrier = IREE::TensorExt::ComputeBarrierOp::create(
        rewriter, barrierOp.getLoc(), reshapeResult, direction, flags);

    DominanceInfo domInfo(op);
    rewriter.replaceUsesWithIf(reshapeResult, newBarrier.getResult(),
                               [&](OpOperand &use) {
                                 return domInfo.properlyDominates(
                                     newBarrier.getOperation(), use.getOwner());
                               });
    return success();
  }
};

// Move tensor.expand_shape/collapse_shape below compute_barrier
// when the barrier direction is Down and the appropriate flag is set
struct MoveReshapeBelowBarrier
    : public OpRewritePattern<IREE::TensorExt::ComputeBarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::TensorExt::ComputeBarrierOp barrierOp,
                                PatternRewriter &rewriter) const override {
    Operation *reshapeOp = barrierOp.getValue().getDefiningOp();
    if (!reshapeOp ||
        !isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(reshapeOp)) {
      return failure();
    }

    // Check direction: must be Down
    auto direction = barrierOp.getDirection();
    if (direction != IREE::TensorExt::BarrierDirection::Down) {
      return failure();
    }

    // Check flags: must have the appropriate flag set
    auto flags = barrierOp.getFlags();
    if ((isa<tensor::ExpandShapeOp>(reshapeOp) &&
         !bitEnumContainsAny(
             flags,
             IREE::TensorExt::TransformationFlagBitfield::AllowExpand)) ||
        (isa<tensor::CollapseShapeOp>(reshapeOp) &&
         !bitEnumContainsAny(
             flags,
             IREE::TensorExt::TransformationFlagBitfield::AllowCollapse))) {
      return failure();
    }

    Value reshapeSrc = reshapeOp->getOperand(0);

    // Create a new compute_barrier before the reshape with the reshape's
    // source type
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(reshapeOp);
    auto newBarrier = IREE::TensorExt::ComputeBarrierOp::create(
        rewriter, barrierOp.getLoc(), reshapeSrc, direction, flags);

    rewriter.modifyOpInPlace(reshapeOp, [&]() {
      reshapeOp->getOpOperand(0).set(newBarrier.getResult());
    });
    rewriter.replaceOp(barrierOp, reshapeOp->getResult(0));
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
    patterns.add<MoveReshapeAboveBarrier, MoveReshapeBelowBarrier>(
        funcOp.getContext());

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("failed to fold reshapes into tensor barriers");
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
