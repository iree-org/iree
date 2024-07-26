// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_FOLDATTENTIONWITHTRANSPOSEPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

namespace {

//===----------------------------------------------------------------------===//
// Attention -> Transpose fusion
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
/// %2 = linalg.generic {
///     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
///                      affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>]}
///     ins(%1)
/// ```
///
/// to
///
/// ```
/// %0 = iree_linalg_ext.attention {
///     indexing_maps = [affine_map<(d0, d00, d1, d2, d3, d4) -> (d0, d00, d1,
///     d2)>,
///                      affine_map<(d0, d00, d1, d2, d3, d4) -> (d0, d00, d3,
///                      d2)>, affine_map<(d0, d00, d1, d2, d3, d4) -> (d0, d00,
///                      d3, d4)>, affine_map<(d0, d00, d1, d2, d3, d4) -> (d0,
///                      d1, d00, d4)>]}
///     ins(%expanded_query, %expanded_key, %expanded_value) ....
/// ```
///
///  Do a very specific match for now. Eventually this can be generalized to a
///  use similar analysis as to what the reshape propagation across Linalg op
///  does. TODO(#17673)
///
struct FoldAttentionAndTranspose
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
    // Check for single use of expand shape op.
    if (!expandShapeOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(attentionOp,
                                         "expand shape op has multiple uses");
    }
    auto transposeLikeOp =
        dyn_cast<linalg::LinalgOp>(*expandShapeOp->user_begin());
    if (!transposeLikeOp) {
      return failure();
    }
    if (!(transposeLikeOp.getNumDpsInputs() == 1 &&
          transposeLikeOp.getNumDpsInits() == 1 &&
          transposeLikeOp.getBlock()
              ->front()
              .hasTrait<OpTrait::IsTerminator>() &&
          transposeLikeOp.getNumLoops() ==
              transposeLikeOp.getNumParallelLoops())) {
      return rewriter.notifyMatchFailure(
          transposeLikeOp, "expand shape user is not a transpose");
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

    // Check the permutation maps for the transpose.
    SmallVector<AffineMap> expectedTransposeMaps = {
        getIndexingMap(4, {d0, d1, d2, d3}),
        getIndexingMap(4, {d0, d2, d1, d3})};
    if (transposeLikeOp.getIndexingMapsArray() != expectedTransposeMaps) {
      return rewriter.notifyMatchFailure(transposeLikeOp,
                                         "unhandled transpose op");
    }

    Location loc = attentionOp.getLoc();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(transposeLikeOp);

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
    Value expandedInit = transposeLikeOp.getDpsInitOperand(0)->get();

    SmallVector<AffineMap> newIndexingMaps = {
        getIndexingMap(6, {d0, d1, d2, d3}),
        getIndexingMap(6, {d0, d1, d4, d3}),
        getIndexingMap(6, {d0, d1, d4, d5}),
        getIndexingMap(6, {d0, d2, d1, d5})};
    ArrayAttr newIndexingMapsAttr =
        rewriter.getAffineMapArrayAttr(newIndexingMaps);
    auto newAttentionOp = rewriter.create<IREE::LinalgExt::AttentionOp>(
        attentionOp.getLoc(), expandedInit.getType(), expandedQuery,
        expandedKey, expandedValue, attentionOp.getScale(), expandedInit,
        newIndexingMapsAttr);
    rewriter.replaceOp(transposeLikeOp, newAttentionOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct FoldAttentionWithTransposePass
    : public impl::FoldAttentionWithTransposePassBase<
          FoldAttentionWithTransposePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<FoldAttentionAndTranspose>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::Preprocessing
