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
      Value innermostInputIdx =
          mapScatterOp.getInputIndex(mapScatterOp.getInputRank() - 1);
      SetVector<Operation *> slice;
      getForwardSlice(innermostInputIdx, &slice);
      Operation *maskOp = mapScatterOp.getMask().getDefiningOp();
      if (maskOp && slice.contains(maskOp)) {
        return rewriter.notifyMatchFailure(
            mapScatterOp, "map_scatter on sub-byte type with potentially non "
                          "byte aligned transformation");
      }
      // Next check that the inner index of the yield is a unit function of
      // the inner input index.
      Value innermostOutputIdx =
          mapScatterOp.getOutputIndex(mapScatterOp.getOutputRank() - 1);
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

struct VectorizeUnMaskOp final : OpRewritePattern<IREE::LinalgExt::UnMaskOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::LinalgExt::UnMaskOp unMaskOp,
                                PatternRewriter &rewriter) const override {
    Location loc = unMaskOp.getLoc();
    RankedTensorType srcType = unMaskOp.getSrc().getType();
    if (!srcType.hasStaticShape()) {
      return failure();
    }

    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto srcVecType =
        VectorType::get(srcType.getShape(), srcType.getElementType());
    Value readVec = vector::TransferReadOp::create(
        rewriter, unMaskOp.getLoc(),
        /*type=*/srcVecType,
        /*source=*/unMaskOp.getSrc(),
        /*indices=*/ValueRange{SmallVector<Value>(srcType.getRank(), zero)},
        /*padding=*/std::nullopt);

    auto maskType = VectorType::get(srcType.getShape(), rewriter.getI1Type());
    Value mask = vector::CreateMaskOp::create(
        rewriter, loc, maskType,
        tensor::getMixedSizes(rewriter, loc, unMaskOp.getDest()));

    auto identityMap = rewriter.getMultiDimIdentityMap(srcType.getRank());
    auto inBounds =
        rewriter.getBoolArrayAttr(SmallVector<bool>(srcType.getRank(), true));
    auto maskedWrite = vector::TransferWriteOp::create(
        rewriter, loc, readVec, unMaskOp.getDest(),
        ValueRange{SmallVector<Value>(srcType.getRank(), zero)},
        AffineMapAttr::get(identityMap), mask, inBounds);

    rewriter.replaceOp(unMaskOp, maskedWrite);

    return success();
  }
};

struct VectorizeIREELinalgExtOpsPass final
    : impl::VectorizeIREELinalgExtOpsPassBase<VectorizeIREELinalgExtOpsPass> {
  void runOnOperation() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<VectorizeStaticMapScatterOpPattern, VectorizeUnMaskOp>(
        context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::LinalgExt
