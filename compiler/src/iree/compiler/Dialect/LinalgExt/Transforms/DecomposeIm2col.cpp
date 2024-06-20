// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {
namespace {

// Helper method to check if a slice will be contiguous given the offset,
// slice size. This checks that `inputSize` and `offset` are both evenly
// divisible by `tileSize`.
static bool willBeContiguousSlice(OpFoldResult inputSize, OpFoldResult tileSize,
                                  OpFoldResult offset) {
  auto constInputSize = getConstantIntValue(inputSize);
  auto constTileSize = getConstantIntValue(tileSize);
  if (!constTileSize.has_value() || !constInputSize.has_value() ||
      constInputSize.value() % constTileSize.value() != 0) {
    return false;
  }
  auto constOffset = getConstantIntValue(offset);
  if (constOffset.has_value() &&
      constOffset.value() % constTileSize.value() == 0) {
    return true;
  }
  auto affineOp = cast<Value>(offset).getDefiningOp<affine::AffineApplyOp>();
  return affineOp.getMap().getResult(0).isMultipleOf(constTileSize.value());
}

// Helper method to add 2 OpFoldResult inputs with affine.apply.
static OpFoldResult addOfrs(OpBuilder &builder, Location loc, OpFoldResult a,
                            OpFoldResult b) {
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  auto addMap = AffineMap::get(2, 0, {d0 + d1});
  return affine::makeComposedFoldedAffineApply(builder, loc, addMap, {a, b});
}

/// Pattern to decompose the tiled im2col op.
struct DecomposeIm2col : public OpRewritePattern<Im2colOp> {
  using OpRewritePattern<Im2colOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Im2colOp im2colOp,
                                PatternRewriter &rewriter) const override {
    Location loc = im2colOp.getLoc();
    Value inputSlice = im2colOp.getInput();
    // Unroll all but the K loop
    SmallVector<OpFoldResult> kOffset = im2colOp.getMixedKOffset();
    SmallVector<OpFoldResult> mOffset = im2colOp.getMixedMOffset();
    // Only support single K and M output dimension for now.
    if (kOffset.size() != 1 || mOffset.size() != 1) {
      return failure();
    }

    // Step 1: Tile the im2col op to loops with contiguous slices in the
    // innermost loop.
    //
    // If the `kOffset` will index to a full contiguous slice of the K dim of
    // the input tensor, then don't tile the K loop of the im2col op and
    // maintain a larger contiguous slice.
    SmallVector<Range> iterationDomain(im2colOp.getIterationDomain(rewriter));
    OpFoldResult kTileSize = iterationDomain.back().size;
    auto constKTileSize = getConstantIntValue(kTileSize);
    if (constKTileSize) {
      kTileSize = rewriter.getIndexAttr(constKTileSize.value());
    }
    SmallVector<OpFoldResult> inputSizes =
        tensor::getMixedSizes(rewriter, loc, im2colOp.getInput());
    // Find the innermost non-batch dimension. This dimension is the fastest
    // changing dimension with the K dimension of the im2col iteration domain.
    // This means it is the innermost dimension of the extract_slice on the
    // input tensor, and the slice wants to be contiguous along this dimension.
    SetVector<int64_t> batchPosSet(im2colOp.getBatchPos().begin(),
                                   im2colOp.getBatchPos().end());
    OpFoldResult innerSliceSize;
    for (int idx = inputSizes.size() - 1; idx >= 0; --idx) {
      if (!batchPosSet.contains(idx)) {
        innerSliceSize = inputSizes[idx];
        break;
      }
    }
    bool tileK =
        !willBeContiguousSlice(innerSliceSize, kTileSize, kOffset.front());
    if (!tileK) {
      iterationDomain.pop_back();
    } else {
      kTileSize = rewriter.getIndexAttr(1);
    }

    // Build loop nest.
    SmallVector<Value> lbs, ubs, steps;
    for (auto range : iterationDomain) {
      lbs.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, range.offset));
      ubs.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, range.size));
      steps.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, range.stride));
    }
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, im2colOp.getOutput(),
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
            ValueRange iterArgs) -> scf::ValueVector { return iterArgs; });
    SmallVector<Value> ivs;
    for (scf::ForOp loop : loopNest.loops) {
      ivs.push_back(loop.getInductionVar());
    }

    // Step 2: Compute indices into the input tensor for extract_slice.
    rewriter.setInsertionPoint(loopNest.loops.front());
    SetVector<int64_t> mPosSet(im2colOp.getMPos().begin(),
                               im2colOp.getMPos().end());

    // Compute the basis for the iteration space of the convolution window
    // (i.e., the H and W dims of the convolution output).
    SmallVector<Value> mBasis;
    ArrayRef<int64_t> strides = im2colOp.getStrides();
    ArrayRef<int64_t> dilations = im2colOp.getDilations();
    SmallVector<OpFoldResult> kernelSize = im2colOp.getMixedKernelSize();
    for (auto [idx, pos] : llvm::enumerate(im2colOp.getMPos())) {
      AffineExpr x, k;
      bindDims(getContext(), x, k);
      AffineExpr mapExpr =
          (x - 1 - (k - 1) * dilations[idx]).floorDiv(strides[idx]) + 1;
      OpFoldResult size = affine::makeComposedFoldedAffineApply(
          rewriter, loc, AffineMap::get(2, 0, {mapExpr}, getContext()),
          {inputSizes[pos], kernelSize[idx]});
      mBasis.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, size));
    }

    // Delinearize the k_offset into an offset into the convolution window and
    // any reduced channels. For an NHWC conv2d, the basis for delinearization
    // would be [P, Q, C] for a PxQ kernel with C channels.
    Location nestedLoc =
        loopNest.loops.back().getBody()->getTerminator()->getLoc();
    rewriter.setInsertionPointToStart(loopNest.loops.back().getBody());

    SmallVector<OpFoldResult> kBasis;
    SmallVector<int64_t> mKernelIdx(im2colOp.getInputRank(), -1);
    for (auto [idx, mPos] : enumerate(im2colOp.getMPos())) {
      mKernelIdx[mPos] = idx;
    }
    for (auto [idx, size] : enumerate(inputSizes)) {
      if (batchPosSet.contains(idx))
        continue;
      if (mPosSet.contains(idx)) {
        kBasis.push_back(kernelSize[mKernelIdx[idx]]);
        continue;
      }
      kBasis.push_back(size);
    }
    OpFoldResult kIndex = kOffset.front();
    if (tileK) {
      kIndex = addOfrs(rewriter, nestedLoc, kOffset.front(), ivs.back());
    }
    FailureOr<SmallVector<Value>> maybeDelinKOffset = affine::delinearizeIndex(
        rewriter, nestedLoc,
        getValueOrCreateConstantIndexOp(rewriter, loc, kIndex),
        getValueOrCreateConstantIndexOp(rewriter, loc, (kBasis)));
    if (failed(maybeDelinKOffset)) {
      return failure();
    }
    SmallVector<Value> delinKOffset = maybeDelinKOffset.value();
    // Split the delinearized offsets into the window offsets (for M offsets)
    // and the K offsets for the input tensor.
    SmallVector<Value> windowOffset, inputKOffset;
    int delinKIdx = 0;
    for (int i = 0; i < im2colOp.getInputRank(); ++i) {
      if (batchPosSet.contains(i))
        continue;
      if (mPosSet.contains(i)) {
        windowOffset.push_back(delinKOffset[delinKIdx++]);
        continue;
      }
      inputKOffset.push_back(delinKOffset[delinKIdx++]);
    }

    // Compute offsets for extract. Start by delinearizing the combined offset
    // of m_offset and the offset from the tiled loop, using the mBasis. This
    // will give an index into the delinearized output space of the convolution.
    Value mArg = tileK ? ivs[ivs.size() - 2] : ivs.back();
    AffineExpr d0, d1;
    bindDims(rewriter.getContext(), d0, d1);
    auto addMap = AffineMap::get(2, 0, {d0 + d1});
    OpFoldResult linearMOffset = affine::makeComposedFoldedAffineApply(
        rewriter, nestedLoc, addMap, {mArg, mOffset[0]});
    FailureOr<SmallVector<Value>> maybeDelinMOffset = affine::delinearizeIndex(
        rewriter, nestedLoc,
        getValueOrCreateConstantIndexOp(rewriter, nestedLoc, linearMOffset),
        mBasis);
    if (failed(maybeDelinMOffset)) {
      return failure();
    }
    SmallVector<Value> delinMOffset = maybeDelinMOffset.value();

    // Compute the final offsets into the input tensor.
    SmallVector<OpFoldResult> sliceOffsets(
        im2colOp.getInputRank(),
        getAsIndexOpFoldResult(rewriter.getContext(), 0));
    SmallVector<OpFoldResult> sliceStrides(
        im2colOp.getInputRank(),
        getAsIndexOpFoldResult(rewriter.getContext(), 1));
    SmallVector<OpFoldResult> sliceSizes(inputSizes);
    // Add the offset into the convolution window, and account for strides and
    // dilations.
    for (auto [idx, mPos] : llvm::enumerate(im2colOp.getMPos())) {
      AffineExpr mOff, wOff;
      bindDims(rewriter.getContext(), mOff, wOff);
      auto map =
          AffineMap::get(2, 0, {mOff * strides[idx] + wOff * dilations[idx]});
      OpFoldResult offset = affine::makeComposedFoldedAffineApply(
          rewriter, nestedLoc, map, {delinMOffset[idx], windowOffset[idx]});
      sliceOffsets[mPos] = offset;
      sliceSizes[mPos] = getAsIndexOpFoldResult(rewriter.getContext(), 1);
    }
    // Set the K offset and size for the input tensor.
    const int64_t kPos = im2colOp.getKPos().front();
    sliceOffsets[kPos] = inputKOffset.front();
    sliceSizes[kPos] = kTileSize;

    // Set the batch offsets for the input tensor.
    int ivIdx = 0;
    for (auto bPos : im2colOp.getBatchPos()) {
      sliceOffsets[bPos] = ivs[ivIdx++];
    }

    // Step 3. Decompose the im2col op into:
    // ```
    // %extract = tensor.extract_slice %input
    // %copy = linalg.copy ins(%extract) outs(%out_slice)
    // %insert = tensor.insert_slice %copy into %loop_arg
    // ```
    //
    // Extract a slice from the input tensor.
    ShapedType outputType = im2colOp.getOutputType();
    SmallVector<int64_t> kTileSizeStatic;
    SmallVector<Value> kTileSizeDynamic;
    dispatchIndexOpFoldResult(kTileSize, kTileSizeDynamic, kTileSizeStatic);
    auto extractType =
        cast<RankedTensorType>(outputType.clone(kTileSizeStatic));
    auto extract = rewriter.create<tensor::ExtractSliceOp>(
        nestedLoc, extractType, inputSlice, sliceOffsets, sliceSizes,
        sliceStrides);

    // Insert the slice into the destination tensor.
    sliceOffsets = SmallVector<OpFoldResult>(
        im2colOp.getOutputRank(),
        getAsIndexOpFoldResult(rewriter.getContext(), 0));
    sliceSizes = SmallVector<OpFoldResult>(
        im2colOp.getOutputRank(),
        getAsIndexOpFoldResult(rewriter.getContext(), 1));
    sliceStrides = SmallVector<OpFoldResult>(
        im2colOp.getOutputRank(),
        getAsIndexOpFoldResult(rewriter.getContext(), 1));
    sliceSizes.back() = kTileSize;
    for (auto [idx, iv] : llvm::enumerate(ivs)) {
      sliceOffsets[idx] = iv;
    }
    // Insert a `linalg.copy` so there is something to vectorize in the
    // decomposition. Without this copy, the extract and insert slice ops
    // do not get vectorized, and the sequence becomes a scalar memref.copy.
    // This memref.copy could be vectorized after bufferization, but it is
    // probably better to vectorize during generic vectorization.
    Value copyDest = rewriter.create<tensor::ExtractSliceOp>(
        nestedLoc, extractType, loopNest.loops.back().getRegionIterArg(0),
        sliceOffsets, sliceSizes, sliceStrides);
    auto copiedSlice = rewriter.create<linalg::CopyOp>(
        nestedLoc, extract.getResult(), copyDest);
    auto insert = rewriter.create<tensor::InsertSliceOp>(
        nestedLoc, copiedSlice.getResult(0),
        loopNest.loops.back().getRegionIterArg(0), sliceOffsets, sliceSizes,
        sliceStrides);
    auto yieldOp =
        cast<scf::YieldOp>(loopNest.loops.back().getBody()->getTerminator());
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, insert.getResult());
    rewriter.replaceOp(im2colOp, loopNest.results[0]);
    return success();
  }
};

} // namespace

namespace {
struct DecomposeIm2colPass : public DecomposeIm2colBase<DecomposeIm2colPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void DecomposeIm2colPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<DecomposeIm2col>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDecomposeIm2colPass() {
  return std::make_unique<DecomposeIm2colPass>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
