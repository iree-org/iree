// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Utils/Indexing.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_VECTORIZEIREELINALGEXTOPSPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct VectorizeStaticMapScatterOpPattern final
    : OpRewritePattern<IREE::LinalgExt::MapScatterOp> {
  using Base::Base;
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
    const int64_t innerSize = inputType.getShape()[inputType.getRank() - 1];
    const int64_t bitWidth = inputType.getElementTypeBitWidth();
    if ((innerSize * bitWidth % 8) != 0) {
      return rewriter.notifyMatchFailure(mapScatterOp,
                                         "map_scatter on sub-byte type");
    }
    // In case of a sub-byte bitwidth, we check that there is a contiguous copy
    // on the inner dimension that is a multiple of a byte. Note that the mask
    // shouldn't depend on the inner index for this.
    if (bitWidth < 8) {
      // First check that the mask is not the forward slice of the inner index.
      Block &transformBody = mapScatterOp.getTransformationRegion().front();
      SmallVector<Value> args(transformBody.getArguments());
      Value innermostInputIdx = args[args.size() - 1];
      SetVector<Operation *> slice;
      getForwardSlice(innermostInputIdx, &slice);
      auto bodyYield =
          cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
      Operation *maskOp =
          bodyYield.getOperand(bodyYield.getNumOperands() - 1).getDefiningOp();
      if (maskOp && slice.contains(maskOp)) {
        return rewriter.notifyMatchFailure(
            mapScatterOp, "map_scatter on sub-byte type with potentially non "
                          "byte aligned transformation");
      }
      // Next check that the inner index of the yield is a unit function of
      // the inner input index.
      Value innermostOutputIdx =
          bodyYield.getOperand(bodyYield.getNumOperands() - 2);
      if (!isUnitFunctionOf(innermostOutputIdx, innermostInputIdx)) {
        return rewriter.notifyMatchFailure(
            mapScatterOp, "map_scatter on sub-byte type with potentially non "
                          "byte aligned transformation");
      }
    }
    Location loc = mapScatterOp.getLoc();
    rewriter.setInsertionPoint(mapScatterOp);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> zeros(inputType.getRank(), zero);
    auto inputVectorType =
        VectorType::get(inputType.getShape(), inputType.getElementType());
    Value inputVector = vector::TransferReadOp::create(
        rewriter, loc, inputVectorType, mapScatterOp.getInput(),
        /*indices=*/zeros,
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
