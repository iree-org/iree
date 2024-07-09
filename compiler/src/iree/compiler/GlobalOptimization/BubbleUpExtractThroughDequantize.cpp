// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

// Convert extract_slice(dequant) to dequant(extract_slice)
//
// Because `extract_slice` ops and dequantize-like ops get cloned into regions
// later, its okay to bubble up through multi-use dequant ops.
struct BubbleUpExtract : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const final {
    Value source = sliceOp.getSource();
    auto linalgOp = source.getDefiningOp<linalg::LinalgOp>();
    if (!linalgOp) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "expected source to be linalg op");
    }

    if (!IREE::Flow::isBitExtendOp(linalgOp)) {
      return rewriter.notifyMatchFailure(
          sliceOp, "expected source to be dequantize-like");
    }

    if (linalgOp.getNumDpsInits() != 1) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "expected single output of linalg op");
    }

    if (!linalgOp.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "expected tensor of linalg op");
    }

    if (!sliceOp.hasUnitStride())
      return rewriter.notifyMatchFailure(sliceOp, "expected unit stride");

    if (sliceOp.getType().getRank() != sliceOp.getSourceType().getRank()) {
      return rewriter.notifyMatchFailure(sliceOp, "expected no rank reduction");
    }

    OpOperand *outOperand = linalgOp.getDpsInitOperand(0);
    AffineMap indexingMap = linalgOp.getMatchingIndexingMap(outOperand);
    if (!indexingMap.isProjectedPermutation()) {
      return rewriter.notifyMatchFailure(
          sliceOp, "expected a projected permutation for output");
    }

    SmallVector<Value> results = tensor::replaceExtractSliceWithTiledProducer(
                                     rewriter, sliceOp, linalgOp->getResult(0))
                                     ->tiledValues;
    rewriter.replaceOp(sliceOp, results);
    return success();
  }
};

struct BubbleUpExtractThroughDequantize
    : BubbleUpExtractThroughDequantizeBase<BubbleUpExtractThroughDequantize> {
  using BubbleUpExtractThroughDequantizeBase::
      BubbleUpExtractThroughDequantizeBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    {
      RewritePatternSet patterns(context);
      patterns.insert<BubbleUpExtract>(context);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> createBubbleUpExtractThroughDequantizePass() {
  return std::make_unique<BubbleUpExtractThroughDequantize>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
