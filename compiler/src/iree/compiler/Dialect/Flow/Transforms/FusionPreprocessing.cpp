// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- FusionPreprocessing.cpp ------------------------------===//
//
// Miscellaneous patterns run before fusion.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_FUSIONPREPROCESSINGPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// GenericOpInterchangePattern
//===----------------------------------------------------------------------===//

struct GenericOpInterchangePattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> interchange;
    bool needInterchange = false;
    unsigned numParallelLoop = genericOp.getNumParallelLoops();
    if (numParallelLoop == 0)
      return failure();
    for (auto iter : llvm::enumerate(genericOp.getIteratorTypesArray())) {
      if (linalg::isParallelIterator(iter.value())) {
        interchange.push_back(iter.index());
        if (iter.index() >= numParallelLoop)
          needInterchange = true;
      }
    }
    // If all the parallel loops are outter loops skip the pattern.
    if (!needInterchange)
      return failure();
    for (auto iter : llvm::enumerate(genericOp.getIteratorTypesArray())) {
      if (linalg::isReductionIterator(iter.value())) {
        interchange.push_back(iter.index());
      }
    }
    return interchangeGenericOp(rewriter, genericOp, interchange);
  }
};

//===----------------------------------------------------------------------===//
// FoldSuccessiveTensorInsertSliceOps
//===----------------------------------------------------------------------===//

/// Pattern to fold
///
/// ```
/// %0 = linalg.fill ins(%cst : )
/// %1 = tensor.insert_slice %a into %0
/// %2 = linalg.fill ins(%cst : )
/// %3 = tensor.insert_slice %1 into %2
/// ```
///
/// to
///
/// ```
/// %2 = linalg.fill ins(%cst : )
/// %3 = tensor.insert_slice %a into %2
/// ```
struct FoldSuccessiveTensorInsertSliceOps
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto sourceInsertSlice =
        sliceOp.getSource().getDefiningOp<tensor::InsertSliceOp>();
    if (!sourceInsertSlice) {
      return failure();
    }
    auto sourceSliceFillOp =
        sourceInsertSlice.getDest().getDefiningOp<linalg::FillOp>();
    auto destSliceFillOp = sliceOp.getDest().getDefiningOp<linalg::FillOp>();
    if (!sourceSliceFillOp || !destSliceFillOp) {
      return rewriter.notifyMatchFailure(
          sliceOp, "dest of both insert_slices expected to be fill operations");
    }
    if (sourceSliceFillOp.getDpsInputOperand(0)->get() !=
        destSliceFillOp.getDpsInputOperand(0)->get()) {
      return rewriter.notifyMatchFailure(
          sliceOp, "dest of both insert_slices expected "
                   "to be fill operation with same value");
    }

    auto isAllConstantOne = [](OpFoldResult ofr) {
      return isConstantIntValue(ofr, 1);
    };
    if (!llvm::all_of(sliceOp.getMixedStrides(), isAllConstantOne) ||
        !llvm::all_of(sliceOp.getMixedStrides(), isAllConstantOne)) {
      return rewriter.notifyMatchFailure(
          sliceOp, "unhandled non-unit strides of slices");
    }

    SmallVector<OpFoldResult> sourceSliceOffsets =
        sourceInsertSlice.getMixedOffsets();
    SmallVector<OpFoldResult> destSliceOffsets = sliceOp.getMixedOffsets();
    AffineExpr d0, d1;
    bindDims(rewriter.getContext(), d0, d1);
    AffineExpr addExpr = d0 + d1;
    SmallVector<OpFoldResult> offsets = llvm::map_to_vector(
        llvm::zip_equal(sourceSliceOffsets, destSliceOffsets), [&](auto it) {
          return affine::makeComposedFoldedAffineApply(
              rewriter, sliceOp.getLoc(), addExpr,
              {std::get<0>(it), std::get<1>(it)});
        });
    SmallVector<OpFoldResult> sizes = sourceInsertSlice.getMixedSizes();
    SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        sliceOp, sourceInsertSlice.getSource(), sliceOp.getDest(), offsets,
        sizes, strides);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GatherFusionPattern
//===----------------------------------------------------------------------===//

// Specific case. The linalg generic implementation of "gather"
// cannot be fused because it there is no producer-consumer
// relationship between the two generics. This is because the indexing
// is not affine (index values come from a tensor).
struct GatherFusionPattern : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // Check if extractOp is inside a generic op
    auto consumerOp =
        dyn_cast_or_null<linalg::GenericOp>(extractOp->getParentOp());
    if (!consumerOp) {
      return rewriter.notifyMatchFailure(
          extractOp, "expected extract op to be inside a generic op");
    }

    auto producerOp = extractOp.getTensor().getDefiningOp<linalg::GenericOp>();
    if (!producerOp) {
      return rewriter.notifyMatchFailure(
          consumerOp, "expected extract operand to be a generic op");
    }

    // Check if the producerOp is fusible
    if (producerOp.getNumDpsInputs() != 1 || producerOp.getNumResults() != 1 ||
        !isElementwise(producerOp) || !isDequantizationLikeOp(producerOp)) {
      return rewriter.notifyMatchFailure(producerOp,
                                         "producer op is not fusible");
    }

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(extractOp);

    // Create a new extract op that extracts from the original tensor
    // (after the original extract). Clone the producerOp's body into the
    // consumerOp, inline the cloned block (erases the block) after the new
    // extract, and clean up.
    auto newExtractOp = rewriter.create<tensor::ExtractOp>(
        extractOp.getLoc(), producerOp.getDpsInputOperand(0)->get(),
        extractOp.getIndices());
    rewriter.cloneRegionBefore(producerOp.getRegion(), consumerOp.getRegion(),
                               consumerOp.getRegion().begin());
    Block &clonedBlock = consumerOp.getRegion().front();
    auto producerTermOp = clonedBlock.getTerminator();

    rewriter.inlineBlockBefore(
        &clonedBlock, extractOp->getNextNode(),
        {newExtractOp.getResult(), newExtractOp.getResult()});

    // Replace the the all references to the original extract result with the
    // result from the inlined producerOp.
    extractOp.getResult().replaceAllUsesWith(producerTermOp->getOperand(0));
    rewriter.eraseOp(producerTermOp);
    rewriter.eraseOp(extractOp);

    return success();
  }
};

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
///     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
///                      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
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
///  does.
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
        getIndexingMap(4, {d0, d2, d1, d3}),
        getIndexingMap(4, {d0, d1, d2, d3})};
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

struct FusionPreprocessingPass
    : public IREE::Flow::impl::FusionPreprocessingPassBase<
          FusionPreprocessingPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldAttentionAndTranspose, FoldSuccessiveTensorInsertSliceOps,
                 GenericOpInterchangePattern, GatherFusionPattern>(
        &getContext());

    // Fold away `tensor.dim` operations that can be resolved in terms of its
    // operand shapes.
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    memref::populateResolveShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
