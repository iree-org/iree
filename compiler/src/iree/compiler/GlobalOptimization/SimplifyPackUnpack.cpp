// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

int64_t getNumGtOneDims(ArrayRef<int64_t> shape) {
  return llvm::count_if(
      shape, [](int64_t v) { return ShapedType::isDynamic(v) || v > 1; });
}

struct Simplify1DPackToExpandShape : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  Value insertExpand(RewriterBase &rewriter, Location loc, Value operand,
                     Type newOperandType, ArrayAttr reassociation) const {
    if (operand.getType() == newOperandType)
      return operand;
    return rewriter.create<tensor::ExpandShapeOp>(loc, newOperandType, operand,
                                                  reassociation);
  }

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (packOp.getPaddingValue())
      return rewriter.notifyMatchFailure(packOp, "expects no padding value");

    RankedTensorType sourceType = packOp.getSourceType();
    RankedTensorType destType = packOp.getDestType();
    if (getNumGtOneDims(sourceType.getShape()) > 1)
      return failure();

    SmallVector<int64_t> innerTiles = packOp.getStaticTiles();
    if (getNumGtOneDims(innerTiles) > 1)
      return failure();

    auto reassociation =
        getReassociationIndicesForReshape(sourceType, destType);
    if (!reassociation)
      return failure();
    Value expanded = insertExpand(
        rewriter, packOp.getLoc(), packOp.getSource(), destType,
        getReassociationIndicesAttribute(rewriter, *reassociation));
    rewriter.replaceOp(packOp, expanded);
    return success();
  }
};

struct Simplify1DUnPackToCollapseShape
    : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  Value insertCollapse(RewriterBase &rewriter, Location loc, Value operand,
                       Type newOperandType, ArrayAttr reassociation) const {
    if (operand.getType() == newOperandType)
      return operand;
    return rewriter.create<tensor::CollapseShapeOp>(loc, newOperandType,
                                                    operand, reassociation);
  }

  LogicalResult matchAndRewrite(tensor::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType sourceType = unpackOp.getSourceType();
    RankedTensorType destType = unpackOp.getDestType();

    if (getNumGtOneDims(destType.getShape()) > 1)
      return failure();

    SmallVector<int64_t> innerTiles = unpackOp.getStaticTiles();
    if (getNumGtOneDims(innerTiles) > 1)
      return failure();

    auto reassociation =
        getReassociationIndicesForReshape(sourceType, destType);
    if (!reassociation)
      return failure();
    Value collapsed = insertCollapse(
        rewriter, unpackOp.getLoc(), unpackOp.getSource(), destType,
        getReassociationIndicesAttribute(rewriter, *reassociation));
    rewriter.replaceOp(unpackOp, collapsed);
    return success();
  }
};

struct SimplifyPackUnpackPass
    : public SimplifyPackUnpackBase<SimplifyPackUnpackPass> {

  void runOnOperation() override;
};
} // namespace

void SimplifyPackUnpackPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  tensor::populateSimplifyPackAndUnpackPatterns(patterns);
  patterns.insert<Simplify1DPackToExpandShape, Simplify1DUnPackToCollapseShape>(
      context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createSimplifyPackUnpackPass() {
  return std::make_unique<SimplifyPackUnpackPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
