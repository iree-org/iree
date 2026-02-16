// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Common/CombineLayoutTransformation.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "iree-codegen-common-transforms"

namespace mlir::iree_compiler {

void moveUpMemrefReshapeOps(RewriterBase &rewriter, Operation *op) {
  DominanceInfo domInfo(op);
  SmallVector<Operation *> reshapeOps;
  op->walk([&](Operation *nestedOp) {
    if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp>(nestedOp)) {
      reshapeOps.push_back(nestedOp);
    }
  });

  for (Operation *reshapeOp : reshapeOps) {
    moveOpAfterLastOperand(rewriter, domInfo, reshapeOp);
  }
}

/// Fuse consumers of forall ops into it after checking that they are tilable.

namespace {

struct FuseTilableForallConsumers final
    : OpInterfaceRewritePattern<TilingInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(TilingInterface tilableOp,
                                PatternRewriter &rewriter) const override {
    // Currently consumer fusion requires DPS, and we don't want to fuse through
    // inits anyway.
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(*tilableOp);
    if (!dpsOp) {
      return failure();
    }

    tensor::ParallelInsertSliceOp producerSlice;
    LoopLikeOpInterface sliceOwner;
    for (auto operand : dpsOp.getDpsInputs()) {
      auto forallProducer = operand.getDefiningOp<scf::ForallOp>();
      if (!forallProducer) {
        continue;
      }
      if (forallProducer->getBlock() != tilableOp->getBlock()) {
        continue;
      }
      sliceOwner = forallProducer;
      break;
    }

    if (!sliceOwner) {
      return rewriter.notifyMatchFailure(tilableOp,
                                         "no scf.forall producer to fuse into");
    }

    // The `tileAndFuseConsumerOfSlices` transform will fail if there are any
    // users of the loop that do not dominate the `tilableOp`, so we move the
    // `tilableOp` and any producers needed for dominance right after the loop.
    // TODO(Max191): Refactor `tileAndFuseConsumerOfSlices` upstream to properly
    // support forall ops with multiple results. The other results of the loop
    // can block fusion because of the dominance issue. Once this is refactored,
    // we should remove this workaround.
    llvm::SetVector<Operation *> slice;
    BackwardSliceOptions options;
    DominanceInfo domInfo;
    options.filter = [&](Operation *op) {
      return domInfo.properlyDominates(sliceOwner, op);
    };
    options.inclusive = true;
    options.omitUsesFromAbove = false;
    options.omitBlockArguments = true;
    if (succeeded(getBackwardSlice(tilableOp, &slice, options))) {
      for (Operation *op : llvm::reverse(slice)) {
        // Don't use the rewriter here because it will notify the Listener, and
        // can add the operations back on the worklist. If the fusion fails
        // after this, then the ops might get continuously added to the
        // worklist.
        Block *block = sliceOwner->getBlock();
        Block::iterator iterator = std::next(sliceOwner->getIterator());
        op->moveBefore(block, iterator);
      }
    }

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseConsumerResults =
        scf::tileAndFuseConsumer(rewriter, tilableOp, {sliceOwner});
    if (failed(fuseConsumerResults)) {
      return failure();
    }
    return success();
  }
};

} // namespace

void populateFuseTilableForallConsumersPattern(RewritePatternSet &patterns) {
  patterns.add<FuseTilableForallConsumers>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Combining Layout Transformation Ops
//===----------------------------------------------------------------------===//

namespace {

struct FoldRelayoutOpIntoMapStorePattern
    : public OpRewritePattern<IREE::LinalgExt::MapStoreOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::LinalgExt::MapStoreOp mapStoreOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = mapStoreOp.getInput().getDefiningOp();
    if (!op) {
      return failure();
    }
    // Folding tensor.pad is handled by a separate pattern.
    if (!isSupportedSingleInputRelayoutOp(op) || isa<tensor::PadOp>(op)) {
      return failure();
    }
    if (failed(foldIntoMapStore(rewriter, op, mapStoreOp))) {
      return failure();
    }
    return success();
  }
};

struct FoldPadOpIntoMapStorePattern
    : public OpRewritePattern<IREE::LinalgExt::MapStoreOp> {
  using Base::Base;
  FoldPadOpIntoMapStorePattern(MLIRContext *context,
                               PadDistributionConfigFn configFn,
                               PatternBenefit benefit = 1)
      : OpRewritePattern<IREE::LinalgExt::MapStoreOp>(context, benefit),
        padDistributionConfigFn(std::move(configFn)) {}

  LogicalResult matchAndRewrite(IREE::LinalgExt::MapStoreOp mapStoreOp,
                                PatternRewriter &rewriter) const override {
    auto padOp = mapStoreOp.getInput().getDefiningOp<tensor::PadOp>();
    if (!padOp) {
      return failure();
    }
    if (failed(foldPadIntoMapStore(rewriter, padOp, mapStoreOp,
                                   padDistributionConfigFn))) {
      return failure();
    }
    return success();
  }

private:
  PadDistributionConfigFn padDistributionConfigFn;
};

} // namespace

void populateCombineRelayoutOpPatterns(
    RewritePatternSet &patterns,
    PadDistributionConfigFn padDistributionConfigFn) {
  patterns.add<FoldRelayoutOpIntoMapStorePattern>(patterns.getContext());
  if (padDistributionConfigFn) {
    patterns.add<FoldPadOpIntoMapStorePattern>(patterns.getContext(),
                                               padDistributionConfigFn);
  }
}

/// Converts `tensor.extract_slice(tensor.expand_shape)` to
/// `tensor.expand_shape(tensor.extract_slice)`.
/// For this transformation to be possible, the slice must be fully contiguous
/// within each reassociation group of the expand_shape. If the transformation
/// is not possible, or if the slice is rank reducting, the function returns
/// failure.
///
/// Example:
/// ```
/// %reshape = tensor.expand_shape %in [[0, 1], [2, 3], [4, 5, 6]]
///     tensor<8x16x32xf32> to tensor<2x4x2x8x4x2x4xf32>
/// %slice = tensor.extract_slice %reshape ...
///     tensor<2x4x2x8x4x2x4xf32> to tensor<2x4x1x5x1x1x4xf32>
///
/// // The transformation is possible because each reassociation group has a
/// // contiguous slice. (i.e., [2x4->2x4], [2x8->1x5], [4x2x4->1x1x4])
/// // After the transformation:
///
/// %slice = tensor.extract_slice %in ...
///     tensor<8x16x32xf32> to tensor<8x5x4xf32>
/// %reshape = tensor.expand_shape %slice [[0, 1], [2, 3], [4, 5, 6]]
///     tensor<8x5x4xf32> to tensor<2x4x1x5x1x1x4xf32>
/// ```
static LogicalResult
swapExpandShapeWithSlice(RewriterBase &rewriter,
                         tensor::ExpandShapeOp expandShapeOp,
                         tensor::ExtractSliceOp sliceOp) {
  SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();

  if (sliceOp.getResultType().getRank() != sizes.size()) {
    return rewriter.notifyMatchFailure(sliceOp,
                                       "unimplemented: rank reducing slice");
  }

  // Helper variables and function for accumulating the new offset and length
  // values.
  Location loc = expandShapeOp->getLoc();
  AffineExpr d0, d1, d2;
  bindDims(rewriter.getContext(), d0, d1, d2);
  // Multiply two integers.
  auto mul = [&](OpFoldResult v1, OpFoldResult v2) {
    auto mulMap = AffineMap::get(2, 0, {d0 * d1});
    return affine::makeComposedFoldedAffineApply(rewriter, loc, mulMap,
                                                 {v1, v2});
  };

  SmallVector<OpFoldResult> outputShape =
      getMixedValues(expandShapeOp.getStaticOutputShape(),
                     expandShapeOp.getOutputShape(), rewriter);

  auto isZeroOffsetAndFullSize = [](OpFoldResult offset, OpFoldResult sliceSize,
                                    OpFoldResult size) {
    if (!isZeroInteger(offset)) {
      return false;
    }
    FailureOr<bool> maybeEqual =
        ValueBoundsConstraintSet::areEqual(sliceSize, size);
    return llvm::succeeded(maybeEqual) && maybeEqual.value();
  };

  // First verify that this is a full slice of the expanded tensor.
  for (const ReassociationIndices &indices :
       expandShapeOp.getReassociationIndices()) {
    int64_t i = 0;
    int64_t e = indices.size();
    // Find the first expanded dim after the first dim with non-unit extracted
    // size.
    for (; i < e; ++i) {
      if (!isOneInteger(sizes[indices[i]])) {
        // +1 to skip the first non-unit size dim.
        i++;
        break;
      }
    }

    // Verify that all subsequent dimensions extract the full size of the
    // source tensor.
    for (; i < e; ++i) {
      int64_t expandedDim = indices[i];
      if (!isZeroOffsetAndFullSize(offsets[expandedDim], sizes[expandedDim],
                                   outputShape[expandedDim])) {
        return rewriter.notifyMatchFailure(
            sliceOp, "Not a contiguous slice of the expanded tensor.");
      }
    }
  }

  // Compute new offsets, lengths, and strides.
  SmallVector<OpFoldResult> newOffsets, newLengths, newStrides;
  for (const ReassociationIndices &indices :
       expandShapeOp.getReassociationIndices()) {
    OpFoldResult newSize = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> basis, delinOffsets;

    int64_t i = 0;
    int64_t e = indices.size();
    // Offset = cumulative product of leading unit extracted dims.
    for (; i < e; ++i) {
      int64_t expandedDim = indices[i];
      if (!isOneInteger(sizes[expandedDim])) {
        break;
      }

      basis.push_back(outputShape[expandedDim]);
      delinOffsets.push_back(offsets[expandedDim]);
    }

    if (i != e) {
      int64_t expandedDim = indices[i];
      basis.push_back(outputShape[expandedDim]);
      delinOffsets.push_back(offsets[expandedDim]);
      newSize = sizes[expandedDim];
      i++;
    }

    for (; i < e; ++i) {
      OpFoldResult fullSize = outputShape[indices[i]];
      basis.push_back(fullSize);
      delinOffsets.push_back(rewriter.getIndexAttr(0));
      newSize = mul(newSize, fullSize);
    }
    SmallVector<Value> offsetVals =
        llvm::map_to_vector(delinOffsets, [&](OpFoldResult ofr) {
          return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
        });
    OpFoldResult newOffset =
        affine::AffineLinearizeIndexOp::create(rewriter, loc, offsetVals, basis,
                                               /*disjoint=*/true)
            .getResult();
    newOffsets.push_back(newOffset);
    newLengths.push_back(newSize);

    // Only unit stride supported.
    newStrides.push_back(rewriter.getIndexAttr(1));
  }

  // The shape of the result can be obtained from the sizes passed in.
  SmallVector<Value> dynDims;
  SmallVector<int64_t> shape;
  dispatchIndexOpFoldResults(sizes, dynDims, shape);
  RankedTensorType resultType = RankedTensorType::get(
      shape, expandShapeOp.getResultType().getElementType());

  // Create a new ExtractSliceOp and ExpandShapeOp.
  Value newSliceOp =
      tensor::ExtractSliceOp::create(rewriter, loc, expandShapeOp.getSrc(),
                                     newOffsets, newLengths, newStrides);
  auto newExpandShapeOp = tensor::ExpandShapeOp::create(
      rewriter, loc, resultType, newSliceOp,
      expandShapeOp.getReassociationIndices(), sizes);
  rewriter.replaceOp(sliceOp, newExpandShapeOp);
  return success();
}

namespace {

struct SwapExpandShapeWithSlicePattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  SwapExpandShapeWithSlicePattern(MLIRContext *context,
                                  linalg::ControlFusionFn controlFn,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::ExtractSliceOp>(context, benefit),
        controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto expandOp = sliceOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp) {
      return failure();
    }

    if (!sliceOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "unsupported: non-unit stride");
    }

    // The control function receives the source operand of the extract_slice
    // (which uses the expand_shape result).
    OpOperand &srcOperand = sliceOp.getSourceMutable();
    if (controlFn && !controlFn(&srcOperand)) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "rejected by control function");
    }

    return swapExpandShapeWithSlice(rewriter, expandOp, sliceOp);
  }

private:
  linalg::ControlFusionFn controlFn;
};

} // namespace

void populateSwapExtractWithExpandPattern(RewritePatternSet &patterns,
                                          linalg::ControlFusionFn controlFn) {
  patterns.add<SwapExpandShapeWithSlicePattern>(patterns.getContext(),
                                                std::move(controlFn));
}

namespace {

/// Pattern to fold extract_slice(broadcast) into the broadcast input when the
/// extract_slice is rank-reducing and only extracts along the broadcasted
/// dimensions (with size 1), leaving all non-broadcasted dimensions intact.
/// This is valid because the broadcast only duplicates data along the
/// broadcasted dimension, and extracting a single slice from that dimension
/// gives back the original input.
///
/// Example:
///   %broadcast = linalg.broadcast ins(%in : tensor<4x1xf16>)
///                                 outs(%out : tensor<4x1x1xf16>) dimensions =
///                                 [2]
///   %extract = tensor.extract_slice %broadcast[0, 0, 0] [4, 1, 1] [1, 1, 1]
///              : tensor<4x1x1xf16> to tensor<4x1xf16>
/// ->
///   %extract is replaced by %in (tensor<4x1xf16>)
struct FoldExtractSliceOfBroadcast final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto broadcastOp =
        extractOp.getSource().getDefiningOp<linalg::BroadcastOp>();
    if (!broadcastOp) {
      return rewriter.notifyMatchFailure(
          extractOp, "source is not a linalg.broadcast operation");
    }

    if (!extractOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(extractOp,
                                         "extract_slice has non-unit stride");
    }

    auto inputType =
        dyn_cast<RankedTensorType>(broadcastOp.getInput().getType());
    auto broadcastOutputType =
        dyn_cast<RankedTensorType>(broadcastOp.getInit().getType());
    auto extractResultType =
        dyn_cast<RankedTensorType>(extractOp.getResult().getType());
    if (!inputType || !broadcastOutputType || !extractResultType) {
      return rewriter.notifyMatchFailure(
          extractOp, "operand or result types are not RankedTensorType");
    }

    // Extract result type must match broadcast input type.
    if (inputType != extractResultType) {
      return rewriter.notifyMatchFailure(
          extractOp, "extract result type does not match broadcast input type");
    }

    // Verify that we're extracting from offset [0, 0, ..., 0] with the same
    // shape as the input (essentially undoing the broadcast).
    SmallVector<OpFoldResult> offsets = extractOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = extractOp.getMixedSizes();
    if (!llvm::all_of(offsets, isZeroInteger)) {
      return rewriter.notifyMatchFailure(
          extractOp, "extract_slice offsets are not all zeros");
    }

    // Sizes should match input dimensions (accounting for broadcast dims).
    ArrayRef<int64_t> broadcastDims = broadcastOp.getDimensions();
    int64_t broadcastRank = broadcastOutputType.getRank();

    // Verify that for broadcast dimensions, the size is 1.
    if (!llvm::all_of(broadcastDims, [&](int64_t broadcastDim) {
          return isOneInteger(sizes[broadcastDim]);
        })) {
      return rewriter.notifyMatchFailure(
          extractOp, "broadcast dimensions do not all have size 1");
    }

    // Collect the indices of dimensions in the broadcast output that were not
    // broadcasted (i.e., dimensions that existed in the original input).
    auto nonBroadcastDims = llvm::filter_to_vector(
        llvm::seq<int64_t>(0, broadcastRank),
        [&](int64_t i) { return !llvm::is_contained(broadcastDims, i); });

    // Verify that for non-broadcast dimensions, sizes match input shape.
    if (llvm::any_of(llvm::enumerate(nonBroadcastDims), [&](auto pair) {
          auto [idx, inputDim] = pair;
          int64_t inputDimSize = inputType.getDimSize(idx);
          return !isConstantIntValue(sizes[inputDim], inputDimSize);
        })) {
      return rewriter.notifyMatchFailure(
          extractOp, "non-broadcast dimension sizes do not match input shape");
    }

    rewriter.replaceOp(extractOp, broadcastOp.getInput());
    return success();
  }
};

} // namespace

void populateFoldExtractSliceOfBroadcastPattern(RewritePatternSet &patterns) {
  patterns.add<FoldExtractSliceOfBroadcast>(patterns.getContext());
}

/// Note the following pattern is adapted from the upstream pattern
/// `BubbleUpCollapseShapeThroughExtractSlice` by allowing some special cases.
///
/// Converts `tensor.extract_slice(tensor.collapse_shape)` to
///          `tensor.collapse_shape(tensor.extract_slice)`.
///
/// For this transformation to be possible - after bubbling up, the extraction
/// of the contiguous slice must be representable as a single slice obtained via
/// tensor.extract_slice within each reassociation group of the src.
///
/// In case the size and offset extracted are static then this is possible if
/// the following conditions are met within each reassociation group:
/// Let T be a tensor of shape [A0, A1, ..., An] (these are the sizes of the
/// dimensions in the reassociation group), and let S = [S0, S1, ..., Sn] be the
/// shape of a desired slice. A slice of shape S can be extracted as a
/// contiguous span of elements if and only if there exists an index k in {0, 1,
/// ..., n} such that:
///      S_i = 1 for all i < k (that is, all leading dimensions are singleton),
///      1 <= S_k <= A_k (that is, non trivial slicing occurs along exactly
///                       one dimension),
///      S_i = A_i for all i > k (that is, all trailing dimensions are preserved
///      in full).
/// In other words, the slice shape S must be of the form:
/// [ 1, 1, ..., 1, Sk, Ak + 1, Ak + 2, ...,An ]
///
/// In case the size and/or offset extracted are dynamic then this is possible
/// only if there is single dimension in the reassociation group that has a size
/// not equal to 1.
/// In other words, the tensor shape must be of the form:
/// [ 1, 1, ..., 1, A, 1, ...,1 ]
/// Note - it might be possible to enable this pattern for more cases when the
/// size/offset are dynamic via performing an analysis of the possible values
/// that could be given to the size/offset.
///
/// Example:
/// The transformation is possible because each reassociation group can be
/// represented as a contiguous slice (i.e., [8x16->2x16], [1x7->1x?],
/// [20->10]).
/// ```
/// BEFORE:
/// %collapse = tensor.collapse_shape %src [[0, 1], [2, 3], [4]] ...
///     tensor<8x16x1x7x20f32> to tensor<128x7x20xf32>
/// %slice = tensor.extract_slice %slice [0, 0, 0][32, %size, 10][1, 1, 1]
///     tensor<128x7x20xf32> to tensor<32x?x10xf32>
///
/// AFTER:
/// %slice = tensor.extract_slice %src [0, 0, 0, 0, 0][2, 16, 1, %size, 10]
//           [1, 1, 1, 1, 1] : tensor<8x16x1x7x20f32> to tensor<2x16x1x?x10xf32>
/// %collapse = tensor.collapse_shape %slice [[0, 1], [2, 3], [4]] ...
///     tensor<2x16x1x?x10xf32> to tensor<32x?x10xf32>
/// ```
///
/// Negative example:
/// The transformation is not possible because we cannot use a single slice to
/// represent the reassociation group [2x3x10->???]. If we would want the
/// collapse to be after the extraction, we would need to extract multiple
/// slices and concat them together.
/// ```
/// %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<2x3x10xf32> into
/// tensor<60xf32> %extract = tensor.extract_slice %collapse[0][15][1] :
///                                      tensor<60xf32> to tensor<15xf32>
/// ```
/// If we would want the collapse to be after the extraction, a possible
/// alternate transformation could be to extract multiple slices and concat them
/// together:
/// ```
/// %extract_1 = tensor.extract_slice %src[0, 0, 0][1, 1, 10] :
///                               tensor<2x3x10xf32> to tensor <1x1x10xf32>
/// %extract_2 = tensor.extract_slice %src[0, 1, 0][1, 1, 5] :
///                               tensor<2x3x10xf32> to tensor <1x1x5xf32>
/// %concat = tosa.concat %extract_1, %extract_2 {axis = 0 : i32} :
///                    (<1x1x10xf32>, <1x1x5xf32>) -> <1x1x15xf32>
/// %collapse = tensor.collapse_shape %concat [[0, 1, 2]] : tensor<1x1x15xf32>
///                                                       to tensor<15xf32>
/// ```
/// But this is not the intended purpose of the transformation.
static LogicalResult
swapCollapseShapeWithSlice(RewriterBase &rewriter,
                           tensor::CollapseShapeOp collapseShapeOp,
                           tensor::ExtractSliceOp sliceOp) {
  // Limit the pattern to work with extract_slice and collapse_shape ops in
  // different blocks.
  // TODO(vivian): remove this check once we have a better handle on fusion of
  // extract_slice with scf.forall loop.
  if (sliceOp->getBlock() == collapseShapeOp->getBlock()) {
    return rewriter.notifyMatchFailure(
        sliceOp, "unsupported: extract_slice and collapse_shape ops are within "
                 "the same block");
  }

  // FIXME: this is a workaround for the fact that this inf loops due to the
  // creation of `affine::AffineDelinearizeIndexOp` but still returns failure.
  SmallVector<Operation *> createdOps;
  auto scope = llvm::scope_exit([&]() {
    for (Operation *op : createdOps) {
      if (op->use_empty()) {
        rewriter.eraseOp(op);
      }
    }
  });

  // The tensor.extract_slice before applying the pattern works on the result
  // of the tensor.collapse_shape, so variables (i.e. inputs for
  // ExtractSliceOp) referring to the state before applying the pattern are
  // named with the prefix "collapsed", and ones referring to the state after
  // applying the pattern are named with the prefix "expanded".
  SmallVector<OpFoldResult> collapsedOffsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> collapsedSizes = sliceOp.getMixedSizes();

  if (static_cast<size_t>(sliceOp.getResultType().getRank()) !=
      collapsedSizes.size()) {
    return rewriter.notifyMatchFailure(sliceOp,
                                       "unimplemented: rank reducing slice");
  }

  ArrayRef<int64_t> srcShape = collapseShapeOp.getSrcType().getShape();
  SmallVector<ReassociationIndices, 4> reassociationIndices =
      collapseShapeOp.getReassociationIndices();

  // Compute new offsets, sizes, and strides for tensor.extract_slice.
  // The new tensor.extract_slice will work on a tensor that has has a rank
  // equal to the rank of the src of the collapse_shape. In each iteration of
  // the loop, the offsets and sizes will be computed per reassociation group.
  SmallVector<OpFoldResult> expandedOffsets, expandedSizes;
  SmallVector<OpFoldResult> expandedStrides(srcShape.size(),
                                            rewriter.getIndexAttr(1));

  for (auto [collapsedSize, collapsedOffset, reassocIndices] :
       llvm::zip_equal(collapsedSizes, collapsedOffsets,
                       collapseShapeOp.getReassociationIndices())) {
    // Do not support cases where both the collapsed size and offset are
    // dynamic, as this may cause failures when padding the operands.
    if (isa<Value>(collapsedSize) && isa<Value>(collapsedOffset)) {
      return rewriter.notifyMatchFailure(
          sliceOp, "collapsed size and offset cannot be both dynamic");
    }
    // CASE #1 - size or offset is dynamic.
    else if (isa<Value>(collapsedSize) || isa<Value>(collapsedOffset)) {
      // Special case especially for collapse shape of convolution filter in
      // IGEMM, while the offset is dynamic and the size is static.
      if (isa<Attribute>(collapsedSize) && isa<Value>(collapsedOffset)) {
        auto maybeStaticSize = getConstantIntValue(collapsedSize);
        if (!maybeStaticSize) {
          return rewriter.notifyMatchFailure(sliceOp,
                                             "collapsed size must be static");
        }
        auto staticSize = maybeStaticSize.value();

        // Check if offset is from a block argument or an affine.apply op of
        // form (d0 * K) or (K * d0).
        auto offsetVal = cast<Value>(collapsedOffset);
        auto collapseDefOp = offsetVal.getDefiningOp();
        if (isa<BlockArgument>(offsetVal)) {
          // The loop is already normalized.
          if (staticSize != 1) {
            return rewriter.notifyMatchFailure(
                sliceOp, "collapsed size must be 1 when the collapsed offset "
                         "is a block argument");
          }
        } else if (auto applyOp =
                       dyn_cast<affine::AffineApplyOp>(collapseDefOp)) {
          AffineMap map = applyOp.getAffineMap();
          if (map.getNumResults() != 1) {
            return rewriter.notifyMatchFailure(
                sliceOp, "affine.apply must have only one result");
          }

          // Compose all nested affine.apply chains and check if the offset is
          // multiple of collapsed size.
          SmallVector<Value> operands(applyOp.getOperands());
          affine::fullyComposeAffineMapAndOperands(&map, &operands);
          map = simplifyAffineMap(map);
          if (!map.getResult(0).isMultipleOf(staticSize)) {
            return rewriter.notifyMatchFailure(
                sliceOp,
                "offset multiplier must be multiple of collapsed size");
          }

          unsigned lastReassocSize = srcShape[reassocIndices.back()];
          if (lastReassocSize % staticSize != 0) {
            return rewriter.notifyMatchFailure(
                sliceOp,
                "the last expanded size is not divisible by collapse size");
          }
        } else {
          return rewriter.notifyMatchFailure(
              sliceOp,
              "offset is not from a block argument or affine.apply op");
        }

        // Calculate expanded offsets and sizes.
        SmallVector<OpFoldResult> expandedBasis;
        for (auto dimIdx : reassocIndices) {
          expandedBasis.push_back(rewriter.getIndexAttr(srcShape[dimIdx]));
        }
        auto delinearizeOp = affine::AffineDelinearizeIndexOp::create(
            rewriter, sliceOp.getLoc(), cast<Value>(collapsedOffset),
            expandedBasis);
        createdOps.push_back(delinearizeOp);
        ValueRange offsets = delinearizeOp.getResults();
        expandedOffsets.append(offsets.begin(), offsets.end());

        expandedSizes.append(reassocIndices.size(), rewriter.getIndexAttr(1));
        expandedSizes.back() = collapsedSize;
        continue;
      }

      // In other general case, the slice can be represented as a contiguous
      // slice only if there is a single dimension in the reassociation group
      // that has a size not equal to 1.
      int nonUnitSizeCount = 0;
      for (int64_t expandedShapeIdx : reassocIndices) {
        if (srcShape[expandedShapeIdx] != 1) {
          nonUnitSizeCount++;
          expandedSizes.push_back(collapsedSize);
          expandedOffsets.push_back(collapsedOffset);
          continue;
        }

        expandedSizes.push_back(rewriter.getIndexAttr(1));
        expandedOffsets.push_back(rewriter.getIndexAttr(0));
      }

      if (nonUnitSizeCount != 1) {
        return rewriter.notifyMatchFailure(
            sliceOp, "unsupported: slice cannot be verified to be contiguous");
      }
      continue;
    }

    // CASE #2 = size and offset are static.
    // Verify that the slice can be represented as a contiguous slice of the
    // src of the collapse_shape.
    // Checking this is done on order of most internal dimensions first,
    // so traversal is done in reverse order of the reassociation group.
    // If the expected slice shape is [1, 1, ..., 1, Sk, Ak + 1, Ak + 2,
    // ...,An] then we first find the size and offset for n...k+1 then for k
    // and then for k-1...0.

    // currentCollapsedsize and currentCollapsedOffset are initialized with
    // the original collapsed size and offset and divided by the expanded
    // shape size in each dimension as we go along the reassociation group.
    // In essence we are spreading the original collapsed size and offset over
    // the various expanded slice dimensions.
    // The variables are used both to check the validity of the slice and to
    // compute the expanded sizes and offsets.
    int64_t currentCollapsedsize = getConstantIntValue(collapsedSize).value();
    int64_t currentCollapsedOffset =
        getConstantIntValue(collapsedOffset).value();

    SmallVector<OpFoldResult> groupExpandedSizes, groupExpandedOffsets;

    ReassociationIndices reversedReassocIndices(reassocIndices.rbegin(),
                                                reassocIndices.rend());
    int64_t idx = 0;
    int64_t reassocGroupSize = reassocIndices.size();

    // First handle the trailing dimensions where the slice size should be
    // equal to the tensor shape and the offset should be 0 (n...k+1).
    for (; idx < reassocGroupSize; ++idx) {
      int64_t expandedShapeSize = srcShape[reversedReassocIndices[idx]];

      if (currentCollapsedsize < expandedShapeSize) {
        break;
      }

      // We need to make sure that the slice size can be set to the shape size
      // and the offset to 0.
      if ((currentCollapsedsize % expandedShapeSize) != 0 ||
          (currentCollapsedOffset % expandedShapeSize) != 0) {
        return rewriter.notifyMatchFailure(
            sliceOp, "unsupported: cannot be extracted as a contiguous slice "
                     "of the src of the collapse_shape");
      }

      groupExpandedSizes.push_back(rewriter.getIndexAttr(expandedShapeSize));
      groupExpandedOffsets.push_back(rewriter.getIndexAttr(0));

      currentCollapsedsize /= expandedShapeSize;
      currentCollapsedOffset /= expandedShapeSize;
    }

    // Now handle the first dim where slicing occurs on (k).
    if (idx < reassocGroupSize) {
      int64_t expandedShapeSize = srcShape[reversedReassocIndices[idx]];
      int64_t offsetInDim = currentCollapsedOffset % expandedShapeSize;
      // We need to make sure that the slice size in this dim + offset will
      // not exceed the shape size.
      if ((currentCollapsedsize + offsetInDim) >= expandedShapeSize) {
        return rewriter.notifyMatchFailure(
            sliceOp, "unsupported: slice cannot be extracted as a contiguous "
                     "slice of the src of the collapse_shape");
      }

      groupExpandedSizes.push_back(rewriter.getIndexAttr(currentCollapsedsize));
      groupExpandedOffsets.push_back(rewriter.getIndexAttr(offsetInDim));

      currentCollapsedOffset /= expandedShapeSize;
    }

    // Now handle the leading dimensions where the slice size is equal to 1
    // (k-1...0).
    // The size for these dimensions must be 1 because of how we constructed
    // the slice size of the expanded shape. We spread the original collapsed
    // size over the expanded shape sizes until we reached dimension k where
    // the remaining size was smaller than the expanded shape size, and spread
    // the remaining size on it. So, now we are left with only 1s.
    for (idx++; idx < reassocGroupSize; ++idx) {
      int64_t expandedShapeSize = srcShape[reversedReassocIndices[idx]];
      int64_t offsetInDim = currentCollapsedOffset % expandedShapeSize;
      groupExpandedSizes.push_back(rewriter.getIndexAttr(1));
      groupExpandedOffsets.push_back(rewriter.getIndexAttr(offsetInDim));
      currentCollapsedOffset /= expandedShapeSize;
    }

    expandedSizes.append(groupExpandedSizes.rbegin(),
                         groupExpandedSizes.rend());
    expandedOffsets.append(groupExpandedOffsets.rbegin(),
                           groupExpandedOffsets.rend());
  }

  Value newSliceOp = tensor::ExtractSliceOp::create(
      rewriter, collapseShapeOp->getLoc(), collapseShapeOp.getSrc(),
      expandedOffsets, expandedSizes, expandedStrides);
  rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
      sliceOp, sliceOp.getResultType(), newSliceOp,
      collapseShapeOp.getReassociationIndices());

  return success();
}

namespace {

struct SwapCollapseShapeWithSlicePattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto collapseOp =
        sliceOp.getSource().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseOp) {
      return rewriter.notifyMatchFailure(
          sliceOp,
          "tensor.extract_slice source not produced by tensor.collapse_shape");
    }

    if (!sliceOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "unsupported: non-unit stride");
    }

    return swapCollapseShapeWithSlice(rewriter, collapseOp, sliceOp);
  }
};

} // namespace

void populateSwapExtractWithCollapsePattern(RewritePatternSet &patterns) {
  patterns.add<SwapCollapseShapeWithSlicePattern>(patterns.getContext());
}

namespace {

struct RemoveOptimizationBarrier final
    : public OpRewritePattern<IREE::Util::OptimizationBarrierOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::Util::OptimizationBarrierOp barrierOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(barrierOp, barrierOp.getOperands());
    return success();
  }
};

} // namespace

void populateRemoveOptimizationBarrierPatterns(RewritePatternSet &patterns) {
  patterns.insert<RemoveOptimizationBarrier>(patterns.getContext());
}

} // namespace mlir::iree_compiler
