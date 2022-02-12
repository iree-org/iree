// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct PushTensorDimAcrossLinalgPass
    : public PushTensorDimAcrossLinalgBase<PushTensorDimAcrossLinalgPass> {
  void runOnOperation() override;
};

class DimOpOfLinalgInit : public OpRewritePattern<tensor::DimOp> {
 public:
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const final {
    auto source = op.source();
    auto linalgInit = source.getDefiningOp<linalg::InitTensorOp>();
    if (!linalgInit) return failure();

    APInt indexAttr;
    if (!matchPattern(op.index(), m_ConstantInt(&indexAttr))) return failure();

    int index = indexAttr.getSExtValue();
    if (!linalgInit.isDynamicSize(index)) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
          op, linalgInit.getStaticSize(index));
      return success();
    }

    rewriter.replaceOp(op, linalgInit.getDynamicSize(index));
    return success();
  }
};

class DimOpOfLinalgGeneric : public OpRewritePattern<tensor::DimOp> {
 public:
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const final {
    auto source = op.source();
    auto operation = source.getDefiningOp<linalg::LinalgOp>();
    if (!operation) return failure();

    auto outputOperands = operation.getOutputOperands();
    auto outputResults = operation->getResults();
    for (auto it : llvm::zip(outputResults, outputOperands)) {
      Value result = std::get<0>(it);
      Value input = std::get<1>(it)->get();
      if (result == source) {
        rewriter.replaceOpWithNewOp<tensor::DimOp>(op, input, op.index());
        return success();
      }
    }

    return failure();
  }
};

class DimOpOfCollapseShape : public OpRewritePattern<tensor::DimOp> {
 public:
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const final {
    auto source = op.source();
    auto collapse = source.getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapse) return failure();

    APInt indexAttr;
    if (!matchPattern(op.index(), m_ConstantInt(&indexAttr))) return failure();
    int index = indexAttr.getSExtValue();
    auto src = collapse.src();
    auto map = collapse.reassociation()[index].cast<ArrayAttr>();

    Location loc = op.getLoc();

    Value accumulated;
    for (auto subIndex : map) {
      int idx = subIndex.cast<IntegerAttr>().getValue().getSExtValue();
      Value constIndex = rewriter.create<arith::ConstantIndexOp>(loc, idx);
      Value dimOp = rewriter.create<tensor::DimOp>(loc, src, constIndex);
      if (accumulated) {
        dimOp = rewriter.create<arith::MulIOp>(loc, accumulated, dimOp);
      }
      accumulated = dimOp;
    }

    rewriter.replaceOp(op, accumulated);
    return success();
  }
};

class DimOpOfExpandShape : public OpRewritePattern<tensor::DimOp> {
 public:
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const final {
    auto source = op.source();
    auto expand = source.getDefiningOp<tensor::ExpandShapeOp>();
    if (!expand) return failure();

    APInt indexAttr;
    if (!matchPattern(op.index(), m_ConstantInt(&indexAttr))) return failure();
    int index = indexAttr.getSExtValue();
    auto src = expand.src();

    int newIndex = -1;
    for (auto it : llvm::enumerate(expand.reassociation().cast<ArrayAttr>())) {
      ArrayAttr array = it.value().cast<ArrayAttr>();
      if (array.size() != 1) continue;
      if (array[0].cast<IntegerAttr>().getValue().getSExtValue() == index)
        newIndex = it.index();
    }

    if (newIndex == -1) return failure();

    Location loc = op.getLoc();
    Value idxVal = rewriter.create<arith::ConstantIndexOp>(loc, newIndex);
    rewriter.replaceOpWithNewOp<tensor::DimOp>(op, src, idxVal);
    return success();
  }
};
}  // namespace

void PushTensorDimAcrossLinalgPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.insert<DimOpOfLinalgGeneric>(&getContext());
  patterns.insert<DimOpOfLinalgInit>(&getContext());
  patterns.insert<DimOpOfCollapseShape>(&getContext());
  patterns.insert<DimOpOfExpandShape>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<FuncOp>> createPushTensorDimAcrossLinalgPass() {
  return std::make_unique<PushTensorDimAcrossLinalgPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
