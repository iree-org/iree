// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- SinkBitCast.cpp - Sink bitcast through reshapes ------------------===//
//
// This pass sinks iree_tensor_ext.bitcast operations through
// tensor.expand_shape and tensor.collapse_shape operations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/DispatchCreation/Passes.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Utils/ShapeUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_SINKBITCASTPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc" // IWYU pragma: keep

using IREE::TensorExt::BitCastOp;

namespace {

/// Propagate bitcast through expand_shape: expand_shape(bitcast(x)) ->
/// bitcast(expand_shape(x)).
struct PropagateBitCastThroughExpandShape final
    : public OpRewritePattern<BitCastOp> {
  using OpRewritePattern<BitCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    if (!bitcastOp->hasOneUse()) {
      return failure();
    }

    auto expandOp = dyn_cast<tensor::ExpandShapeOp>(*bitcastOp->user_begin());
    if (!expandOp) {
      return failure();
    }

    // All dims except last must match (last dim can differ due to element
    // type size change). This check also filters out rank mismatches.
    RankedTensorType bitcastSrcType = bitcastOp.getSource().getType();
    RankedTensorType bitcastDstType = bitcastOp.getResult().getType();
    if (!compareMixedShapesEqualExceptLast(
            bitcastSrcType, bitcastOp.getSourceDims(), bitcastDstType,
            bitcastOp.getResultDims())) {
      return failure();
    }

    // For rank > 1, the new expand output shape's last dim is derived from the
    // bitcast source's last dim (scaled by bitwidth ratio). We don't support
    // dynamic last dim yet since we'd need affine expressions to scale it.
    // For rank 1, the bitwidth ratio is 1:1, so it doesn't matter.
    if (bitcastSrcType.getRank() > 1 &&
        ShapedType::isDynamic(bitcastSrcType.getShape().back())) {
      return failure();
    }

    // When src_bits < dst_bits (e.g., f4 -> i32), the bitcast shrinks the last
    // dim by the bitwidth ratio (4 / 32). Assume before sinking, the last dim
    // size at bitcast source is 16, then through bitcast, the last dim size at
    // result is 16 * 4 / 32 = 2. Now if expand then splits that last dim into
    // multiple output dims (16 -> 8x2), we may need to scale multiple split
    // dims when sinking. For simplicity, we reject this case for now, by
    // checking that the last group of reassociation indices has only one
    // dimension.
    if (bitcastSrcType.getElementTypeBitWidth() <
        bitcastDstType.getElementTypeBitWidth()) {
      SmallVector<ReassociationIndices> reassoc =
          expandOp.getReassociationIndices();
      if (reassoc.empty()) {
        return failure();
      }
      const ReassociationIndices &lastGroup = reassoc.back();
      if (lastGroup.size() != 1) {
        return failure();
      }
    }

    // Compute the new expand output shape. Same as original but with the last
    // dim scaled according to the bitcast source and destination element type
    // bitwidths.
    RankedTensorType expandResultType = expandOp.getResultType();
    SmallVector<int64_t> newOutputShape(expandResultType.getShape());
    newOutputShape.back() =
        (newOutputShape.back() * bitcastDstType.getElementTypeBitWidth()) /
        bitcastSrcType.getElementTypeBitWidth();

    SmallVector<OpFoldResult> oldMixedOutputShape =
        expandOp.getMixedOutputShape();
    SmallVector<OpFoldResult> newMixedOutputShape(oldMixedOutputShape);
    newMixedOutputShape.back() = rewriter.getIndexAttr(newOutputShape.back());

    // Create the new expand_shape with the original bitcast source, but with
    // the new output shape.
    auto newExpandResultType =
        RankedTensorType::get(newOutputShape, bitcastSrcType.getElementType());
    rewriter.setInsertionPoint(expandOp);
    auto newExpandOp = tensor::ExpandShapeOp::create(
        rewriter, expandOp.getLoc(), newExpandResultType, bitcastOp.getSource(),
        expandOp.getReassociationIndices(), newMixedOutputShape);

    // Build the dynamic dims for the new bitcast.
    int64_t expandResultRank = expandResultType.getRank();
    SmallVector<Value> newDynSrcDims;
    for (int64_t i = 0; i < expandResultRank; ++i) {
      if (ShapedType::isStatic(newOutputShape[i])) {
        continue;
      }
      Value dimVal = getValueOrCreateConstantIndexOp(
          rewriter, bitcastOp.getLoc(), newMixedOutputShape[i]);
      newDynSrcDims.push_back(dimVal);
    }

    SmallVector<Value> newDynDstDims;
    for (int64_t i = 0; i < expandResultRank; ++i) {
      if (!expandResultType.isDynamicDim(i)) {
        continue;
      }
      Value dimVal = getValueOrCreateConstantIndexOp(
          rewriter, bitcastOp.getLoc(), oldMixedOutputShape[i]);
      newDynDstDims.push_back(dimVal);
    }

    rewriter.replaceOpWithNewOp<BitCastOp>(
        expandOp, expandResultType, newExpandOp, newDynSrcDims, newDynDstDims);
    rewriter.eraseOp(bitcastOp);
    return success();
  }
};

/// Propagate bitcast through collapse_shape: collapse_shape(bitcast(x)) ->
/// bitcast(collapse_shape(x)).
struct PropagateBitCastThroughCollapseShape final
    : public OpRewritePattern<BitCastOp> {
  using OpRewritePattern<BitCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    if (!bitcastOp->hasOneUse()) {
      return failure();
    }

    auto collapseOp =
        dyn_cast<tensor::CollapseShapeOp>(*bitcastOp->user_begin());
    if (!collapseOp) {
      return failure();
    }

    // All dims except last must match (last dim can differ due to element type
    // size change). This check also filters out rank mismatches.
    RankedTensorType bitcastSrcType = bitcastOp.getSource().getType();
    RankedTensorType bitcastDstType = bitcastOp.getResult().getType();
    if (!compareMixedShapesEqualExceptLast(
            bitcastSrcType, bitcastOp.getSourceDims(), bitcastDstType,
            bitcastOp.getResultDims())) {
      return failure();
    }

    // We don't support dynamic last dim yet since we'd need affine expressions
    // to scale it.
    if (ShapedType::isDynamic(bitcastSrcType.getShape().back())) {
      return failure();
    }

    // Check that all dynamic dimensions are in their own reassociation groups,
    // so we don't need to deal with the case where it is merged with other
    // dimensions.
    SmallVector<ReassociationIndices> reassoc =
        collapseOp.getReassociationIndices();
    for (const auto &group : reassoc) {
      for (int64_t dim : group) {
        if (bitcastDstType.isDynamicDim(dim) && group.size() != 1) {
          return failure();
        }
      }
    }

    // Compute the new collapsed shape. The last dim of the new collapse result
    // should be scaled by the bitcast source and destination element type bit
    // widths. Multiply first to avoid integer division truncation.
    RankedTensorType collapseResultType = collapseOp.getResultType();
    SmallVector<int64_t> newCollapseResultShape(collapseResultType.getShape());
    newCollapseResultShape.back() = (newCollapseResultShape.back() *
                                     bitcastDstType.getElementTypeBitWidth()) /
                                    bitcastSrcType.getElementTypeBitWidth();

    // Create the new collapse_shape operating on the original bitcast source,
    // but with the new result shape.
    auto newCollapseResultType = RankedTensorType::get(
        newCollapseResultShape, bitcastSrcType.getElementType());
    rewriter.setInsertionPoint(collapseOp);
    auto newCollapseOp = tensor::CollapseShapeOp::create(
        rewriter, collapseOp.getLoc(), newCollapseResultType,
        bitcastOp.getSource(), collapseOp.getReassociationIndices());

    // Use the original bitcast's source and result dynamic dims, since we
    // already check they are in their own reassociation groups (no merging with
    // other dims).
    rewriter.replaceOpWithNewOp<BitCastOp>(
        collapseOp, collapseResultType, newCollapseOp,
        bitcastOp.getSourceDims(), bitcastOp.getResultDims());
    rewriter.eraseOp(bitcastOp);
    return success();
  }
};

struct SinkBitCastPass final
    : public impl::SinkBitCastPassBase<SinkBitCastPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<PropagateBitCastThroughExpandShape,
                 PropagateBitCastThroughCollapseShape>(context);

    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
