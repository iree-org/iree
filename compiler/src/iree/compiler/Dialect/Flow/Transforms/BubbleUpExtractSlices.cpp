// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_BUBBLEUPEXTRACTSLICESPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

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
    if (!linalgOp || linalgOp->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          sliceOp, "expected source to implement `linalg::LinalgOp` and have a "
                   "single result");
    }

    if (!IREE::LinalgExt::isBitExtendOp(linalgOp)) {
      return rewriter.notifyMatchFailure(
          sliceOp, "expected source to be dequantize-like");
    }

    if (!sliceOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(sliceOp, "expected unit stride");
    }

    Value replacement;
    linalg::GenericOp swappedOp;
    {
      FailureOr<TilingResult> tilingResult =
          tensor::replaceExtractSliceWithTiledProducer(rewriter, sliceOp,
                                                       linalgOp->getResult(0));
      if (failed(tilingResult)) {
        return rewriter.notifyMatchFailure(
            linalgOp, "failed to swap extract_slice with op");
      }
      if (tilingResult->tiledOps.size() != 1 ||
          !isa<linalg::GenericOp>(tilingResult->tiledOps[0])) {
        return rewriter.notifyMatchFailure(
            linalgOp, "expected extract_slice to generate a `linalg.generic`");
      }
      replacement = tilingResult->tiledValues[0];
      swappedOp = cast<linalg::GenericOp>(tilingResult->tiledOps[0]);
    }

    // Check if this is a rank-reducing slice, if so we need to fold the unit
    // dimensions of the op.
    if (sliceOp.getSourceType().getRank() !=
        sliceOp.getResultType().getRank()) {

      llvm::SmallBitVector droppedDims = sliceOp.getDroppedDims();
      // Get the indexing map for the result.
      AffineMap resultMap =
          swappedOp.getIndexingMapMatchingResult(swappedOp->getResult(0));
      if (!resultMap.isProjectedPermutation()) {
        return rewriter.notifyMatchFailure(
            sliceOp,
            "expected swapped operation to have identity indexing map");
      }

      linalg::ControlDropUnitDims options;
      options.rankReductionStrategy = linalg::ControlDropUnitDims::
          RankReductionStrategy::ExtractInsertSlice;
      options.controlFn = [&](Operation *op) -> SmallVector<unsigned> {
        SmallVector<unsigned> droppedDimsVec;
        for (auto [index, expr] : llvm::enumerate(resultMap.getResults())) {
          if (!droppedDims.test(index)) {
            continue;
          }
          auto dimExpr = cast<AffineDimExpr>(expr);
          droppedDimsVec.push_back(dimExpr.getPosition());
        }
        return droppedDimsVec;
      };
      FailureOr<linalg::DropUnitDimsResult> dropUnitDims =
          linalg::dropUnitDims(rewriter, swappedOp, options);
      if (failed(dropUnitDims)) {
        return rewriter.notifyMatchFailure(
            sliceOp, "failed to drop unit dims of produced operation");
      }

      swappedOp = dropUnitDims->resultOp;
      replacement = swappedOp->getResult(0);
    }
    rewriter.replaceOp(sliceOp, replacement);
    return success();
  }
};

struct BubbleUpExtractSlicesPass
    : impl::BubbleUpExtractSlicesPassBase<BubbleUpExtractSlicesPass> {
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

} // namespace mlir::iree_compiler::IREE::Flow
