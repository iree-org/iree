// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/Im2colUtils.h"

#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

Im2colSourceIndices computeIm2colSourceIndices(OpBuilder &b, Location loc,
                                               Im2colOp im2colOp,
                                               ArrayRef<Value> ivs,
                                               OpFoldResult innerTileSize) {
  int64_t inputRank = im2colOp.getInputRank();
  SmallVector<OpFoldResult> offsets = im2colOp.getMixedOffsets();
  SmallVector<SmallVector<OpFoldResult>> outputSizes =
      im2colOp.getMixedOutputSizes();
  int64_t batchSize = im2colOp.getBatchPos().size();
  int64_t numMOutputDims = im2colOp.getNumMOutputDims();
  llvm::SmallDenseSet<int64_t, 4> mPosSet(im2colOp.getMPos().begin(),
                                          im2colOp.getMPos().end());
  llvm::SmallDenseSet<int64_t, 4> batchPosSet(im2colOp.getBatchPos().begin(),
                                              im2colOp.getBatchPos().end());
  ArrayRef<int64_t> strides = im2colOp.getStrides();
  ArrayRef<int64_t> dilations = im2colOp.getDilations();
  ArrayRef<int64_t> inputKPerm = im2colOp.getInputKPerm();

  // Phase 1: Delinearize all canonical output dims uniformly.
  // Canonical order is [batch..., M..., K...]. For each dim d,
  // delinearize (offset[d] + iv[actualDim]) using output_sizes[d].
  SmallVector<int64_t> inverseOutputPerm =
      invertPermutationVector(im2colOp.getOutputPerm());
  int64_t numCanonicalDims = static_cast<int64_t>(outputSizes.size());
  SmallVector<Value> allCoords;
  int64_t batchCoordCount = 0;
  int64_t mCoordCount = 0;
  for (int64_t d = 0; d < numCanonicalDims; ++d) {
    int64_t actualDim = inverseOutputPerm[d];
    OpFoldResult idx = addOfrs(b, loc, offsets[d], ivs[actualDim]);
    const SmallVector<OpFoldResult> &innerSizes = outputSizes[d];
    int64_t numProduced = static_cast<int64_t>(innerSizes.size());
    if (numProduced == 1) {
      allCoords.push_back(getValueOrCreateConstantIndexOp(b, loc, idx));
    } else {
      ValueRange delinCoords =
          affine::AffineDelinearizeIndexOp::create(
              b, loc, getValueOrCreateConstantIndexOp(b, loc, idx), innerSizes,
              /*hasOuterBound=*/true)
              .getResults();
      allCoords.append(delinCoords.begin(), delinCoords.end());
    }
    if (d < batchSize) {
      batchCoordCount += numProduced;
    } else if (d < batchSize + numMOutputDims) {
      mCoordCount += numProduced;
    }
  }

  // Phase 2: Split delinearized coords into batch, M, K groups.
  auto it = allCoords.begin();
  SmallVector<Value> batchCoords(it, it + batchCoordCount);
  it += batchCoordCount;
  SmallVector<Value> mCoords(it, it + mCoordCount);
  it += mCoordCount;
  SmallVector<Value> kCoords(it, allCoords.end());

  // Phase 3: Organize coords into input dimension order.
  // K coords: apply inverse input_k_perm to map from output K order to
  // canonical input order, then split into window offsets and channel offsets.
  SmallVector<int64_t> invInputKPerm = invertPermutationVector(inputKPerm);
  // applyPermutationToVector performs a gather (result[i] = src[perm[i]]),
  // which is the correct inverse mapping from output K order to input order.
  applyPermutationToVector(kCoords, invInputKPerm);
  SmallVector<Value> windowOffset, inputKOffset;
  int64_t kIdx = 0;
  for (int64_t i = 0; i < inputRank; ++i) {
    if (batchPosSet.contains(i)) {
      continue;
    }
    if (mPosSet.contains(i)) {
      windowOffset.push_back(kCoords[kIdx++]);
      continue;
    }
    inputKOffset.push_back(kCoords[kIdx++]);
  }

  // Compute final offsets into the input tensor.
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);
  SmallVector<OpFoldResult> sliceOffsets(inputRank, zero);
  SmallVector<OpFoldResult> sliceSizes(inputRank, one);

  // Spatial dims: apply strides and dilations.
  AffineExpr mOff, wOff;
  bindDims(b.getContext(), mOff, wOff);
  for (auto [idx, mPos] : llvm::enumerate(im2colOp.getMPos())) {
    auto map =
        AffineMap::get(2, 0, {mOff * strides[idx] + wOff * dilations[idx]});
    OpFoldResult offset = affine::makeComposedFoldedAffineApply(
        b, loc, map, {mCoords[idx], windowOffset[idx]});
    sliceOffsets[mPos] = offset;
  }

  // K dims: set channel offsets directly.
  for (auto [kPos, kOff] : llvm::zip_equal(im2colOp.getKPos(), inputKOffset)) {
    sliceOffsets[kPos] = kOff;
  }

  // Batch dims: set delinearized batch coords directly.
  int64_t batchIdx = 0;
  for (int64_t bPos : im2colOp.getBatchPos()) {
    sliceOffsets[bPos] = batchCoords[batchIdx++];
  }

  // The innermost input dimension gets innerTileSize as its size.
  int64_t innerInputDim = inputRank - 1;
  sliceSizes[innerInputDim] = innerTileSize;

  return Im2colSourceIndices{sliceOffsets, sliceSizes};
}

Value computeIm2colValidSize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                             const Im2colSourceIndices &srcIndices,
                             OpFoldResult innerTileSize,
                             ArrayRef<Value> outputIVs,
                             std::optional<int64_t> vecOutputDim) {
  int64_t inputRank = im2colOp.getInputRank();
  int64_t vecInputDim = inputRank - 1;

  SmallVector<OpFoldResult> inputSizes =
      tensor::getMixedSizes(b, loc, im2colOp.getInput());

  // Get padding from the op.
  SmallVector<OpFoldResult> padLow(inputRank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> padHigh(inputRank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> inputPadLow = im2colOp.getMixedInputPadLow();
  SmallVector<OpFoldResult> inputPadHigh = im2colOp.getMixedInputPadHigh();
  if (!inputPadLow.empty()) {
    padLow = inputPadLow;
    padHigh = inputPadHigh;
  }

  // Compute adjusted offsets: subtract padLow to get unpadded-space coords.
  SmallVector<OpFoldResult> adjustedOffsets(inputRank);
  for (int64_t d = 0; d < inputRank; ++d) {
    adjustedOffsets[d] = subOfrs(b, loc, srcIndices.sliceOffsets[d], padLow[d]);
  }

  // When a dim is being vectorized, chooseDimToVectorize guarantees no low
  // padding on the vectorized input dim. Assert this invariant.
  assert((!vecOutputDim.has_value() ||
          isConstantIntValue(padLow[vecInputDim], 0)) &&
         "vectorized input dim must have zero low padding");

  // --- Helper lambdas for affine clamping patterns ---
  // All use affine ops so that IREE's IntegerDivisibilityAnalysis can track
  // divisibility through the chain (it tracks affine.apply/min/max and
  // arith.muli, but NOT arith.subi/maxsi/minsi).
  MLIRContext *ctx = b.getContext();
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  AffineExpr d1 = getAffineDimExpr(1, ctx);
  AffineMap subMap = AffineMap::get(2, 0, d0 - d1, ctx);
  AffineMap minMap = AffineMap::get(2, 0, {d0, d1}, ctx);
  AffineMap maxZeroMap =
      AffineMap::get(1, 0, {d0, getAffineConstantExpr(0, ctx)}, ctx);
  AffineMap clampHighToOneMap =
      AffineMap::get(2, 0, {d0 - d1, getAffineConstantExpr(1, ctx)}, ctx);
  AffineMap clampLowToOneMap =
      AffineMap::get(1, 0, {d0 + 1, getAffineConstantExpr(1, ctx)}, ctx);

  // max(val, 0).
  auto clampAboveZero = [&](OpFoldResult val) -> OpFoldResult {
    return affine::makeComposedFoldedAffineMax(b, loc, maxZeroMap, {val});
  };

  // 0/1 factor: 1 when coord ∈ [0, dimSize), 0 otherwise.
  auto clampToRange = [&](OpFoldResult coord, OpFoldResult dimSize) -> Value {
    // highOk = max(min(dimSize - coord, 1), 0): 1 when coord < dimSize.
    OpFoldResult highOk = clampAboveZero(affine::makeComposedFoldedAffineMin(
        b, loc, clampHighToOneMap, {dimSize, coord}));
    // lowOk = max(min(coord + 1, 1), 0): 1 when coord >= 0.
    OpFoldResult lowOk = clampAboveZero(
        affine::makeComposedFoldedAffineMin(b, loc, clampLowToOneMap, {coord}));
    Value highOkVal = getValueOrCreateConstantIndexOp(b, loc, highOk);
    Value lowOkVal = getValueOrCreateConstantIndexOp(b, loc, lowOk);
    return arith::MulIOp::create(b, loc, highOkVal, lowOkVal);
  };

  // min(max(extent - coord, 0), tileSize): how much of tileSize fits within
  // [coord, extent), clamped to [0, tileSize].
  auto remainingValid = [&](OpFoldResult extent, OpFoldResult coord,
                            OpFoldResult tileSize) -> OpFoldResult {
    OpFoldResult rawValid =
        affine::makeComposedFoldedAffineApply(b, loc, subMap, {extent, coord});
    OpFoldResult clamped = clampAboveZero(rawValid);
    return affine::makeComposedFoldedAffineMin(b, loc, minMap,
                                               {clamped, tileSize});
  };

  // 0/1 factor: 1 when pos >= low, 0 otherwise.
  auto isAboveLow = [&](OpFoldResult pos, OpFoldResult low) -> Value {
    OpFoldResult adjusted = subOfrs(b, loc, pos, low);
    OpFoldResult lowOk = clampAboveZero(affine::makeComposedFoldedAffineMin(
        b, loc, clampLowToOneMap, {adjusted}));
    return getValueOrCreateConstantIndexOp(b, loc, lowOk);
  };

  // 0/1 factor: 1 when pos < high, 0 otherwise.
  auto isBelowHigh = [&](OpFoldResult pos, OpFoldResult high) -> Value {
    OpFoldResult highOk = clampAboveZero(affine::makeComposedFoldedAffineMin(
        b, loc, clampHighToOneMap, {high, pos}));
    return getValueOrCreateConstantIndexOp(b, loc, highOk);
  };

  // --- Compute valid_size along the innermost input dimension ---
  OpFoldResult validSizeOfr = remainingValid(
      inputSizes[vecInputDim], adjustedOffsets[vecInputDim], innerTileSize);
  Value validSize = getValueOrCreateConstantIndexOp(b, loc, validSizeOfr);

  // --- Input-side bounds checking for non-vectorized dims ---
  // If an adjusted coord is outside [0, dimSize), the valid region is empty,
  // so multiply validSize by 0. This handles input padding, output-alignment
  // OOB from non-wrapping delinearization, and batch OOB.
  auto checkDimBounds = [&](int64_t dim) {
    // The vectorized input dim's range is already handled above. Skip it
    // to avoid redundant IR. In scalar mode (no vecOutputDim), all dims
    // need full bounds checking.
    if (vecOutputDim.has_value() && dim == vecInputDim) {
      return;
    }
    // Skip bounds check when this dim has no input padding — the offset
    // is guaranteed to be in [0, dimSize) by construction.
    if (isZeroInteger(padLow[dim]) && isZeroInteger(padHigh[dim])) {
      return;
    }
    Value factor = clampToRange(adjustedOffsets[dim], inputSizes[dim]);
    validSize = arith::MulIOp::create(b, loc, validSize, factor);
  };
  for (int64_t bPos : im2colOp.getBatchPos()) {
    checkDimBounds(bPos);
  }
  for (int64_t mPos : im2colOp.getMPos()) {
    checkDimBounds(mPos);
  }
  for (int64_t kPos : im2colOp.getKPos()) {
    checkDimBounds(kPos);
  }

  // --- Output-side bounds checking ---
  // For each output dim, positions in [0, pad_low) and [dim - pad_high, dim)
  // are padding positions and should produce pad_value.
  // chooseDimToVectorize guarantees output_pad_low[vecOutputDim] == 0.
  SmallVector<OpFoldResult> outPadLow = im2colOp.getMixedOutputPadLow();
  SmallVector<OpFoldResult> outPadHigh = im2colOp.getMixedOutputPadHigh();
  if (!outPadLow.empty()) {
    assert((!vecOutputDim.has_value() ||
            isConstantIntValue(outPadLow[*vecOutputDim], 0)) &&
           "vectorized output dim must have zero output low padding");
    int64_t outputRank = im2colOp.getOutputRank();
    SmallVector<OpFoldResult> outputTensorSizes =
        tensor::getMixedSizes(b, loc, im2colOp.getOutput());

    // Non-vectorized output dims: if pos is outside [padLow, dim - padHigh),
    // set validSize = 0.
    for (int64_t d = 0; d < outputRank; ++d) {
      if (vecOutputDim.has_value() && d == vecOutputDim.value()) {
        continue;
      }
      if (isConstantIntValue(outPadLow[d], 0) &&
          isConstantIntValue(outPadHigh[d], 0)) {
        continue;
      }

      OpFoldResult localPos = outputIVs[d];

      if (!isConstantIntValue(outPadLow[d], 0)) {
        Value factor = isAboveLow(localPos, outPadLow[d]);
        validSize = arith::MulIOp::create(b, loc, validSize, factor);
      }
      if (!isConstantIntValue(outPadHigh[d], 0)) {
        OpFoldResult validEnd =
            subOfrs(b, loc, outputTensorSizes[d], outPadHigh[d]);
        Value factor = isBelowHigh(localPos, validEnd);
        validSize = arith::MulIOp::create(b, loc, validSize, factor);
      }
    }

    // Vectorized output dim: clamp validSize by remaining valid output count.
    // output_pad_low[vecOutputDim] == 0 is asserted above, so only high-side
    // padding needs checking here.
    if (vecOutputDim.has_value()) {
      int64_t vd = vecOutputDim.value();
      if (!isConstantIntValue(outPadHigh[vd], 0)) {
        OpFoldResult validEnd =
            subOfrs(b, loc, outputTensorSizes[vd], outPadHigh[vd]);
        OpFoldResult vsOfr = remainingValid(validEnd, outputIVs[vd], validSize);
        validSize = getValueOrCreateConstantIndexOp(b, loc, vsOfr);
      }
    }
  }

  return validSize;
}

/// Helper to check if a slice will be contiguous given the offset and
/// slice size. Checks that `inputSize` and `offset` are both evenly
/// divisible by `tileSize`.
static bool willBeContiguousSlice(OpFoldResult inputSize, OpFoldResult tileSize,
                                  OpFoldResult offset) {
  std::optional<int64_t> constInputSize = getConstantIntValue(inputSize);
  std::optional<int64_t> constTileSize = getConstantIntValue(tileSize);
  if (!constTileSize.has_value() || !constInputSize.has_value() ||
      constInputSize.value() % constTileSize.value() != 0) {
    return false;
  }
  std::optional<int64_t> constOffset = getConstantIntValue(offset);
  if (constOffset.has_value()) {
    return constOffset.value() % constTileSize.value() == 0;
  }
  auto val = dyn_cast<Value>(offset);
  if (!val) {
    return false;
  }
  auto affineOp = val.getDefiningOp<affine::AffineApplyOp>();
  return affineOp &&
         affineOp.getMap().getResult(0).isMultipleOf(constTileSize.value());
}

std::optional<int64_t> chooseDimToVectorize(OpBuilder &b, Location loc,
                                            Im2colOp im2colOp,
                                            ArrayRef<Range> iterationDomain,
                                            ArrayRef<OpFoldResult> offsets) {
  int64_t innerInputDim = im2colOp.getInputRank() - 1;
  SmallVector<OpFoldResult> inputSizes =
      tensor::getMixedSizes(b, loc, im2colOp.getInput());
  SmallVector<SmallVector<int64_t>> vectorizationMap =
      im2colOp.getInputToOutputDimVectorizationMap();
  SmallVector<int64_t> vectorizableOutputDims = vectorizationMap[innerInputDim];
  if (vectorizableOutputDims.empty()) {
    return std::nullopt;
  }

  // Bail when the innermost input dim has non-zero low padding. Low padding
  // on the vectorized input dim would require shifting read indices which
  // complicates the valid size computation. Non-vectorized dims handle low
  // padding through the per-dim bounds checks in computeIm2colValidSize.
  SmallVector<OpFoldResult> inputPadLow = im2colOp.getMixedInputPadLow();
  if (!inputPadLow.empty() &&
      !isConstantIntValue(inputPadLow[innerInputDim], 0)) {
    return std::nullopt;
  }

  // Get output low padding for per-candidate checks below.
  SmallVector<OpFoldResult> outPadLow = im2colOp.getMixedOutputPadLow();

  SetVector<int64_t> kDimSet(llvm::from_range, im2colOp.getKOutputDims());

  // Build a map from actual output dim to canonical index for K dims.
  SmallVector<int64_t> kOutputDims = im2colOp.getKOutputDims();
  int64_t batchSize = im2colOp.getBatchPos().size();
  int64_t numMOutputDims = im2colOp.getNumMOutputDims();
  DenseMap<int64_t, int64_t> kDimToCanonicalIdx;
  for (auto [i, actualDim] : llvm::enumerate(kOutputDims)) {
    kDimToCanonicalIdx[actualDim] = batchSize + numMOutputDims + i;
  }

  // There may be multiple output dims that we can vectorize, so prioritize the
  // innermost dims first.
  llvm::sort(vectorizableOutputDims);
  // Check each dim in order from innermost to outermost, and return the first
  // one that is vectorizable.
  for (int64_t outputDimToVectorize : llvm::reverse(vectorizableOutputDims)) {
    // If a K dim is being vectorized, then it is contiguous along either the
    // input channel dimension, or the filter kernel window. If it is contiguous
    // along the kernel window, then the actual inner slice size is equal to the
    // size of the corresponding kernel window dimension. Otherwise, the inner
    // slice size is just the size of the input tensor's inner dimension.
    // Use the padded input size for the contiguity check: the im2col operates
    // in the padded coordinate space, so the effective innermost dimension
    // includes both low and high padding.
    OpFoldResult innerSliceSize = inputSizes[innerInputDim];
    SmallVector<OpFoldResult> inputPadHigh = im2colOp.getMixedInputPadHigh();
    if (!inputPadLow.empty()) {
      innerSliceSize =
          addOfrs(b, loc, innerSliceSize, inputPadLow[innerInputDim]);
      innerSliceSize =
          addOfrs(b, loc, innerSliceSize, inputPadHigh[innerInputDim]);
    }
    if (kDimSet.contains(outputDimToVectorize)) {
      for (auto [kernelSize, mPos] :
           llvm::zip_equal(im2colOp.getMixedKernelSize(), im2colOp.getMPos())) {
        if (mPos == innerInputDim) {
          innerSliceSize = kernelSize;
        }
      }
    }

    // If the input slice is contiguous along the innermost dimension, then it
    // is vectorizable. If it is not, then move on to the next innermost dim.
    SetVector<int64_t> mDimSet(llvm::from_range, im2colOp.getMOutputDims());
    OpFoldResult offset = b.getIndexAttr(0);
    if (kDimSet.contains(outputDimToVectorize)) {
      // Use the offset of this specific K dim directly (no linearization).
      offset = offsets[kDimToCanonicalIdx[outputDimToVectorize]];
    } else if (mDimSet.contains(outputDimToVectorize)) {
      // TODO(Max191): Support vectorization along the M dimension.
      continue;
    }
    // Skip dims with non-zero output low padding. Low padding on the
    // vectorized output dim would require shifting write positions and
    // complicates the valid size computation.
    if (!outPadLow.empty() &&
        !isConstantIntValue(outPadLow[outputDimToVectorize], 0)) {
      continue;
    }

    OpFoldResult outputDimSize = iterationDomain[outputDimToVectorize].size;
    if (!willBeContiguousSlice(innerSliceSize, outputDimSize, offset)) {
      continue;
    }
    return outputDimToVectorize;
  }
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
computeIm2colVectorTileSizes(OpBuilder &b, Im2colOp im2colOp) {
  Location loc = im2colOp.getLoc();
  SmallVector<Range> iterationDomain(im2colOp.getIterationDomain(b));
  SmallVector<OpFoldResult> mixedOffsets = im2colOp.getMixedOffsets();
  std::optional<int64_t> dimToVectorize =
      chooseDimToVectorize(b, loc, im2colOp, iterationDomain, mixedOffsets);

  int64_t outputRank = im2colOp.getOutputRank();
  SmallVector<int64_t> tileSizes(outputRank, 1);
  if (!dimToVectorize.has_value()) {
    return std::nullopt;
  }
  std::optional<int64_t> dimSize =
      getConstantIntValue(iterationDomain[dimToVectorize.value()].size);
  if (!dimSize.has_value()) {
    return std::nullopt;
  }
  tileSizes[dimToVectorize.value()] = dimSize.value();
  return tileSizes;
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
