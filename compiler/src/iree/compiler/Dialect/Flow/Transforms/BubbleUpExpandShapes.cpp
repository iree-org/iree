// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- BubbleExpandShapes.cpp --- Pass to propagate expand shapes op up -===//
//
// This pass propagates expand_shape operations up the program (and conversely)
// sinks the collapse_shape operations down the program to get the elementwise
// operations into higher dimensionality to get better fusion.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/FusionUtils.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-bubble-up-expand-shapes"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_BUBBLEUPEXPANDSHAPESPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Attention -> Expand shape fusion
//===----------------------------------------------------------------------===//

/// Pattern to fold
///
/// ```mlir
/// %0 = iree_linalg_ext.attention {
///     indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
///                      affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
///                      affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
///                      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
///     ins(%query, %key, %value) ....
/// %1 = tensor.expand_shape %0 into [[0, 1], [2], [3]] ....
/// ```
///
/// to
///
/// ```
/// %0 = iree_linalg_ext.attention {
///     indexing_maps = [
///         affine_map<(d0, d00, d1, d2, d3, d4) -> (d0, d00, d1, d2)>,
///         affine_map<(d0, d00, d1, d2, d3, d4) -> (d0, d00, d3, d2)>,
///         affine_map<(d0, d00, d1, d2, d3, d4) -> (d0, d00, d3, d4)>,
///         affine_map<(d0, d00, d1, d2, d3, d4) -> (d0, d00, d1, d4)>]}
///     ins(%expanded_query, %expanded_key, %expanded_value) ....
/// ```
///
///  Do a very specific match for now. Eventually this can be generalized to a
///  use similar analysis as to what the reshape propagation across Linalg op
///  does. TODO(#17673)
///
struct BubbleUpExpandShapeAcrossAttention
    : public OpRewritePattern<IREE::LinalgExt::AttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::AttentionOp attentionOp,
                                PatternRewriter &rewriter) const override {
    // Check for single use attention op.
    if (!attentionOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(attentionOp,
                                         "attention op has multiple uses");
    }
    auto expandShapeOp =
        dyn_cast<tensor::ExpandShapeOp>(*attentionOp->user_begin());
    if (!expandShapeOp) {
      return rewriter.notifyMatchFailure(attentionOp,
                                         "user is not an expand shape op.");
    }

    // Check attention op indexing maps.
    AffineExpr d0, d1, d2, d3, d4, d5;
    bindDims(rewriter.getContext(), d0, d1, d2, d3, d4, d5);
    auto getIndexingMap = [&](int n, ArrayRef<AffineExpr> results) {
      return AffineMap::get(n, 0, results, rewriter.getContext());
    };
    SmallVector<AffineMap> expectedMaps = {
        getIndexingMap(5, {d0, d1, d2}), getIndexingMap(5, {d0, d3, d2}),
        getIndexingMap(5, {d0, d3, d4}), getIndexingMap(5, {d0, d1, d4})};
    if (attentionOp.getIndexingMapsArray() != expectedMaps) {
      return rewriter.notifyMatchFailure(
          attentionOp, "mismatch in expected maps, and maps on attention op");
    }

    // Check reassociation indexing map.
    SmallVector<ReassociationIndices> reassociation =
        expandShapeOp.getReassociationIndices();
    SmallVector<ReassociationIndices> expectedReassocation = {{0, 1}, {2}, {3}};
    if (reassociation != expectedReassocation) {
      return rewriter.notifyMatchFailure(expandShapeOp,
                                         "unhandled reassocation");
    }

    Location loc = attentionOp.getLoc();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(expandShapeOp);

    SmallVector<OpFoldResult> expandedResultShape =
        tensor::getMixedSizes(rewriter, loc, expandShapeOp);
    OpFoldResult dim0_split0 = expandedResultShape[0];
    OpFoldResult dim0_split1 = expandedResultShape[1];
    OpFoldResult dim1 = expandedResultShape[2];
    OpFoldResult dim2 =
        tensor::getMixedSize(rewriter, loc, attentionOp.getKey(), 2);
    OpFoldResult dim3 =
        tensor::getMixedSize(rewriter, loc, attentionOp.getKey(), 1);
    OpFoldResult dim4 = expandedResultShape[3];

    SmallVector<OpFoldResult> newQuerySizes = {};
    SmallVector<Value> tmp;
    SmallVector<int64_t> newQueryShape;
    dispatchIndexOpFoldResults(newQuerySizes, tmp, newQueryShape);

    auto getReshape = [&](Value v, ArrayRef<ReassociationIndices> reassociation,
                          ArrayRef<OpFoldResult> outputShape) -> Value {
      SmallVector<int64_t> staticShape;
      SmallVector<Value> dynamicShape;
      dispatchIndexOpFoldResults(outputShape, dynamicShape, staticShape);
      Type resultType = RankedTensorType::get(
          staticShape, cast<RankedTensorType>(v.getType()).getElementType());
      return rewriter
          .create<tensor::ExpandShapeOp>(loc, resultType, v, reassociation,
                                         outputShape)
          .getResult();
    };

    Value expandedQuery = getReshape(attentionOp.getQuery(), {{0, 1}, {2}, {3}},
                                     {dim0_split0, dim0_split1, dim1, dim2});
    Value expandedKey = getReshape(attentionOp.getKey(), {{0, 1}, {2}, {3}},
                                   {dim0_split0, dim0_split1, dim3, dim2});
    Value expandedValue = getReshape(attentionOp.getValue(), {{0, 1}, {2}, {3}},
                                     {dim0_split0, dim0_split1, dim3, dim4});
    Value expandedInit = getReshape(attentionOp.getOutput(), {{0, 1}, {2}, {3}},
                                    {dim0_split0, dim0_split1, dim1, dim4});

    SmallVector<AffineMap> newIndexingMaps = {
        getIndexingMap(6, {d0, d1, d2, d3}),
        getIndexingMap(6, {d0, d1, d4, d3}),
        getIndexingMap(6, {d0, d1, d4, d5}),
        getIndexingMap(6, {d0, d1, d2, d5})};
    ArrayAttr newIndexingMapsAttr =
        rewriter.getAffineMapArrayAttr(newIndexingMaps);
    auto newAttentionOp = rewriter.create<IREE::LinalgExt::AttentionOp>(
        attentionOp.getLoc(), expandedInit.getType(), expandedQuery,
        expandedKey, expandedValue, attentionOp.getScale(), expandedInit,
        newIndexingMapsAttr);
    rewriter.replaceOp(expandShapeOp, newAttentionOp);
    return success();
  }
};

class BubbleUpExpandShapesPass
    : public impl::BubbleUpExpandShapesPassBase<BubbleUpExpandShapesPass> {
public:
  using Base::Base;

  void runOnOperation() override;
};

} // namespace

void BubbleUpExpandShapesPass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet bubbleExpandShapePatterns(context);
  linalg::ControlFusionFn bubbleUpExpansionControlFn =
      [](OpOperand *fusedOperand) {
        Operation *producer = fusedOperand->get().getDefiningOp();
        Operation *consumer = fusedOperand->getOwner();
        if (!isNonNullAndOutsideDispatch({producer, consumer})) {
          return false;
        }

        // Do not fuse by expand if consumer is dequant.
        if (isBitExtendOp(consumer)) {
          return false;
        }

        // Do not fuse producer generic op if it has more than one user
        // or any reduction iterators.
        if (auto producerGenericOp = dyn_cast<linalg::GenericOp>(producer)) {
          return producerGenericOp->hasOneUse() &&
                 llvm::all_of(producerGenericOp.getIteratorTypesArray(),
                              linalg::isParallelIterator);
        }

        // Do not fuse with any producer linalg named ops for now.
        if (isa<linalg::LinalgOp>(producer)) {
          return false;
        }

        // Do not fuse with consumer linalg named ops or reductions.
        if (auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(consumer)) {
          return isa<linalg::GenericOp>(consumerLinalgOp) &&
                 llvm::all_of(consumerLinalgOp.getIteratorTypesArray(),
                              linalg::isParallelIterator);
        }
        // Fuse in all other cases.
        return true;
      };
  linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                    bubbleUpExpansionControlFn);
  // Add patterns to do some additional cleanup (on top of canonicalizations
  // that can be done later) of reshape ops.
  tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);
  bubbleExpandShapePatterns.insert<BubbleUpExpandShapeAcrossAttention>(context);

  GreedyRewriteConfig rewriteConfig;
  rewriteConfig.maxIterations = GreedyRewriteConfig::kNoLimit;
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(bubbleExpandShapePatterns),
                                          rewriteConfig))) {
    getOperation()->emitOpError("Failed to perform elementwise operations");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::Flow
