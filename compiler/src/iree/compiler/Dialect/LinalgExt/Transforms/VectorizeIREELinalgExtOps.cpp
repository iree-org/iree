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

struct VectorizeStaticMapStoreOpPattern final
    : OpRewritePattern<IREE::LinalgExt::MapStoreOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::LinalgExt::MapStoreOp mapStoreOp,
                                PatternRewriter &rewriter) const override {
    if (mapStoreOp.isVectorized()) {
      return rewriter.notifyMatchFailure(mapStoreOp,
                                         "map_store is already vectorized");
    }
    ShapedType inputType = mapStoreOp.getInputType();
    if (!inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(mapStoreOp,
                                         "map_store has non-static shape");
    }
    const int64_t innerSize = inputType.getShape()[inputType.getRank() - 1];
    const int64_t bitWidth = inputType.getElementTypeBitWidth();
    if ((innerSize * bitWidth % 8) != 0) {
      return rewriter.notifyMatchFailure(mapStoreOp,
                                         "map_store on sub-byte type");
    }
    // In case of a sub-byte bitwidth, we check that there is a contiguous copy
    // on the inner dimension that is a multiple of a byte. Note that the mask
    // shouldn't depend on the inner index for this.
    if (bitWidth < 8) {
      // First check that the mask is not the forward slice of the inner index.
      Value innermostInputIdx =
          mapStoreOp.getInputIndex(mapStoreOp.getInputRank() - 1);
      SetVector<Operation *> slice;
      getForwardSlice(innermostInputIdx, &slice);
      Operation *maskOp = mapStoreOp.getMask().getDefiningOp();
      if (maskOp && slice.contains(maskOp)) {
        return rewriter.notifyMatchFailure(
            mapStoreOp, "map_store on sub-byte type with potentially non "
                        "byte aligned transformation");
      }
      // Next check that the inner index of the yield is a unit function of
      // the inner input index.
      Value innermostOutputIdx =
          mapStoreOp.getOutputIndex(mapStoreOp.getOutputRank() - 1);
      if (!isUnitFunctionOf(innermostOutputIdx, innermostInputIdx)) {
        return rewriter.notifyMatchFailure(
            mapStoreOp, "map_store on sub-byte type with potentially non "
                        "byte aligned transformation");
      }
    }
    Location loc = mapStoreOp.getLoc();
    rewriter.setInsertionPoint(mapStoreOp);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> zeros(inputType.getRank(), zero);
    auto inputVectorType =
        VectorType::get(inputType.getShape(), inputType.getElementType());
    Value inputVector = vector::TransferReadOp::create(
        rewriter, loc, inputVectorType, mapStoreOp.getInput(),
        /*indices=*/zeros,
        /*padding=*/std::nullopt);
    auto vectorizedMapStoreOp =
        clone(rewriter, mapStoreOp, mapStoreOp.getResultTypes(),
              {inputVector, mapStoreOp.getOutput()});
    rewriter.replaceOp(mapStoreOp, vectorizedMapStoreOp);
    return success();
  }
};

struct VectorizeIREELinalgExtOpsPass final
    : impl::VectorizeIREELinalgExtOpsPassBase<VectorizeIREELinalgExtOpsPass> {
  void runOnOperation() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<VectorizeStaticMapStoreOpPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::LinalgExt
