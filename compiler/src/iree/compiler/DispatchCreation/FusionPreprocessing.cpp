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

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
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
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FUSIONPREPROCESSINGPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// ElementwiseOpInterchangePattern
//===----------------------------------------------------------------------===//

// If possible, interchange indexing maps to make input maps all identity.
struct ElementwiseOpInterchangePattern final
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isElementwise(genericOp) || genericOp.getNumResults() != 1 ||
        genericOp.getNumDpsInputs() == 0)
      return failure();

    // All input maps must be equal and non-identity. All maps, including
    // output, must be be permutations. Permutation maps are checked by
    // isElementwise but may be removed.
    AffineMap inputMap = genericOp.getIndexingMapsArray().front();
    auto *initOperand = genericOp.getDpsInitOperand(0);
    if (inputMap.isIdentity() || !inputMap.isPermutation() ||
        !genericOp.getMatchingIndexingMap(initOperand).isPermutation()) {
      return failure();
    }
    for (auto *operand : genericOp.getDpsInputOperands()) {
      if (genericOp.getMatchingIndexingMap(operand) != inputMap) {
        return failure();
      }
    }

    // Make all inputs identity.
    ArrayRef<AffineExpr> exprs = inputMap.getResults();
    auto perm = llvm::map_to_vector(exprs, [](AffineExpr e) -> unsigned {
      return cast<AffineDimExpr>(e).getPosition();
    });
    return linalg::interchangeGenericOp(rewriter, genericOp, perm);
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
struct FoldSuccessiveTensorInsertSliceOps final
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
struct GatherFusionPattern final : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;
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
        !isElementwise(producerOp) ||
        !IREE::LinalgExt::isBitExtendOp(producerOp)) {
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

struct FusionPreprocessingPass final
    : public impl::FusionPreprocessingPassBase<FusionPreprocessingPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ElementwiseOpInterchangePattern,
                 FoldSuccessiveTensorInsertSliceOps, GatherFusionPattern>(
        &getContext());

    // Fold away `tensor.dim` operations that can be resolved in terms of its
    // operand shapes.
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    memref::populateResolveShapedTypeResultDimsPatterns(patterns);
    tensor::populateFoldIntoPackAndUnpackPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
