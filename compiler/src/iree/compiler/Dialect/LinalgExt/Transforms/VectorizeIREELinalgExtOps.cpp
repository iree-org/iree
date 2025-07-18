// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_VECTORIZEIREELINALGEXTOPSPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct VectorizeStaticMapScatterOpPattern final
    : OpRewritePattern<IREE::LinalgExt::MapScatterOp> {
  using OpRewritePattern<IREE::LinalgExt::MapScatterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::LinalgExt::MapScatterOp mapScatterOp,
                                PatternRewriter &rewriter) const override {
    if (mapScatterOp.isVectorized()) {
      return rewriter.notifyMatchFailure(mapScatterOp,
                                         "map_scatter is already vectorized");
    }
    ShapedType inputType = mapScatterOp.getInputType();
    if (!inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(mapScatterOp,
                                         "map_scatter has non-static shape");
    }
    Location loc = mapScatterOp.getLoc();
    rewriter.setInsertionPoint(mapScatterOp);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> zeros(inputType.getRank(), zero);
    auto inputVectorType =
        VectorType::get(inputType.getShape(), inputType.getElementType());
    Value inputVector = rewriter.create<vector::TransferReadOp>(
        loc, inputVectorType, mapScatterOp.getInput(), /*indices=*/zeros,
        /*padding=*/std::nullopt);
    auto vectorizedMapScatterOp =
        clone(rewriter, mapScatterOp, mapScatterOp.getResultTypes(),
              {inputVector, mapScatterOp.getOutput()});
    rewriter.replaceOp(mapScatterOp, vectorizedMapScatterOp);
    return success();
  }
};

struct VectorizeIREELinalgExtOpsPass final
    : impl::VectorizeIREELinalgExtOpsPassBase<VectorizeIREELinalgExtOpsPass> {
  void runOnOperation() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<VectorizeStaticMapScatterOpPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::LinalgExt
