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

  // For each K output dim: delinearize (offset[d] + iv[d]) using
  // output_sizes[d]. Concatenate results -> combined window+channel coords.
  SmallVector<Value> kCoords;
  SmallVector<int64_t> kOutputDims = im2colOp.getKOutputDims();
  int64_t numKOutputDims =
      im2colOp.getOutputRank() - batchSize - numMOutputDims;
  for (int64_t i = 0; i < numKOutputDims; ++i) {
    int64_t canonicalIdx = batchSize + numMOutputDims + i;
    int64_t actualDim = kOutputDims[i];
    OpFoldResult idx = addOfrs(b, loc, offsets[canonicalIdx], ivs[actualDim]);
    const SmallVector<OpFoldResult> &innerSizes = outputSizes[canonicalIdx];
    if (innerSizes.size() == 1) {
      kCoords.push_back(getValueOrCreateConstantIndexOp(b, loc, idx));
    } else {
      // Use non-wrapping delinearize (hasOuterBound=false) so that
      // positions beyond the product of output_sizes produce out-of-bounds
      // coordinates instead of wrapping. This lets the bounds computation
      // naturally handle oversized outputs (e.g. from GEMM alignment).
      SmallVector<OpFoldResult> innerBasis(innerSizes.begin() + 1,
                                           innerSizes.end());
      ValueRange delinCoords =
          affine::AffineDelinearizeIndexOp::create(
              b, loc, getValueOrCreateConstantIndexOp(b, loc, idx),
              innerBasis, /*hasOuterBound=*/false)
              .getResults();
      kCoords.append(delinCoords.begin(), delinCoords.end());
    }
  }

  // Apply input_k_perm to kCoords, then split into window offsets and
  // channel offsets. The inverse permutation maps from output K order to
  // the canonical input order (m_pos spatial + k_pos channel).
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

  // For each M output dim: delinearize (offset[d] + iv[d]) using
  // output_sizes[d]. Concatenate results -> spatial coordinates.
  SmallVector<Value> mCoords;
  SmallVector<int64_t> mOutputDims = im2colOp.getMOutputDims();
  for (int64_t i = 0; i < numMOutputDims; ++i) {
    int64_t canonicalIdx = batchSize + i;
    int64_t actualDim = mOutputDims[i];
    OpFoldResult idx = addOfrs(b, loc, offsets[canonicalIdx], ivs[actualDim]);
    const SmallVector<OpFoldResult> &innerSizes = outputSizes[canonicalIdx];
    if (innerSizes.size() == 1) {
      mCoords.push_back(getValueOrCreateConstantIndexOp(b, loc, idx));
    } else {
      // Non-wrapping delinearize: same rationale as the K dim case above.
      SmallVector<OpFoldResult> innerBasis(innerSizes.begin() + 1,
                                           innerSizes.end());
      ValueRange delinCoords =
          affine::AffineDelinearizeIndexOp::create(
              b, loc, getValueOrCreateConstantIndexOp(b, loc, idx),
              innerBasis, /*hasOuterBound=*/false)
              .getResults();
      mCoords.append(delinCoords.begin(), delinCoords.end());
    }
  }

  // Compute final offsets into the input tensor.
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);
  SmallVector<OpFoldResult> sliceOffsets(inputRank, zero);
  SmallVector<OpFoldResult> sliceSizes(inputRank, one);

  // Apply strides and dilations for spatial dimensions.
  AffineExpr mOff, wOff;
  bindDims(b.getContext(), mOff, wOff);
  for (auto [idx, mPos] : llvm::enumerate(im2colOp.getMPos())) {
    auto map =
        AffineMap::get(2, 0, {mOff * strides[idx] + wOff * dilations[idx]});
    OpFoldResult offset = affine::makeComposedFoldedAffineApply(
        b, loc, map, {mCoords[idx], windowOffset[idx]});
    sliceOffsets[mPos] = offset;
  }

  // Set K offsets.
  for (auto [kPos, kOff] :
       llvm::zip_equal(im2colOp.getKPos(), inputKOffset)) {
    sliceOffsets[kPos] = kOff;
  }

  // Set batch offsets from offset attribute + loop IVs.
  // Batch dims are first in canonical [Batch, M, K] order, so
  // canonicalIdx = ivIdx. The actual output dim comes from the inverse
  // output permutation.
  SmallVector<int64_t> inverseOutputPerm =
      invertPermutationVector(im2colOp.getOutputPerm());
  for (auto [ivIdx, bPos] : llvm::enumerate(im2colOp.getBatchPos())) {
    int64_t canonicalIdx = ivIdx;
    int64_t actualDim = inverseOutputPerm[canonicalIdx];
    sliceOffsets[bPos] = addOfrs(b, loc, offsets[canonicalIdx], ivs[actualDim]);
  }

  // The innermost input dimension gets innerTileSize as its size.
  int64_t innerInputDim = inputRank - 1;
  sliceSizes[innerInputDim] = innerTileSize;

  return Im2colSourceIndices{sliceOffsets, sliceSizes};
}

// Conservative for tiled ops with dynamic offsets: returns true whenever the
// tile could potentially exceed the valid output region, even if the tiling
// infrastructure guarantees in-bounds access. The resulting bounds-checking
// IR folds away after canonicalize + CSE.
bool hasOutputPadding(Im2colOp im2colOp) {
  SmallVector<SmallVector<OpFoldResult>> outputSizes =
      im2colOp.getMixedOutputSizes();
  SmallVector<OpFoldResult> offsets = im2colOp.getMixedOffsets();
  ArrayRef<int64_t> outputPerm = im2colOp.getOutputPerm();
  // output_perm[actual] = canonical, so invert to map canonical -> actual.
  SmallVector<int64_t> inverseOutputPerm =
      invertPermutationVector(outputPerm);
  ArrayRef<int64_t> outputShape = im2colOp.getOutputType().getShape();

  for (auto [canonical, innerSizes] : llvm::enumerate(outputSizes)) {
    // Compute the static product of output_sizes for this canonical dim.
    int64_t validSize = 1;
    for (OpFoldResult s : innerSizes) {
      std::optional<int64_t> c = getConstantIntValue(s);
      if (!c.has_value()) {
        return true; // Dynamic: conservatively assume output padding.
      }
      validSize *= c.value();
    }
    int64_t actual = inverseOutputPerm[canonical];
    int64_t tensorDim = outputShape[actual];
    if (ShapedType::isDynamic(tensorDim)) {
      return true; // Dynamic: conservatively assume output padding.
    }
    // Check if this tile's range [offset, offset+tensorDim) can exceed
    // validSize. For tiled ops, the tensor dim is the tile size and the
    // offset determines where the tile starts.
    std::optional<int64_t> constOffset = getConstantIntValue(offsets[canonical]);
    if (!constOffset.has_value()) {
      // Dynamic offset: safe only if the tile exactly spans the valid region.
      if (tensorDim != validSize) {
        return true;
      }
      continue;
    }
    if (*constOffset + tensorDim > validSize) {
      return true;
    }
  }
  return false;
}

SmallVector<OpFoldResult>
computeOutputValidSizes(OpBuilder &b, Location loc, Im2colOp im2colOp) {
  SmallVector<SmallVector<OpFoldResult>> outputSizes =
      im2colOp.getMixedOutputSizes();
  SmallVector<OpFoldResult> validSizes;
  for (const auto &innerSizes : outputSizes) {
    OpFoldResult product = innerSizes[0];
    for (size_t i = 1; i < innerSizes.size(); ++i) {
      product = mulOfrs(b, loc, product, innerSizes[i]);
    }
    validSizes.push_back(product);
  }
  return validSizes;
}

Im2colPaddingBounds
computeIm2colPaddingBounds(OpBuilder &b, Location loc, Im2colOp im2colOp,
                           const Im2colSourceIndices &srcIndices,
                           ArrayRef<OpFoldResult> inputSizes,
                           ArrayRef<OpFoldResult> padLow,
                           OpFoldResult innerTileSize,
                           ArrayRef<Value> outputIVs,
                           ArrayRef<OpFoldResult> outputOffsets,
                           std::optional<int64_t> vecOutputDim) {
  int64_t inputRank = im2colOp.getInputRank();
  int64_t vecInputDim = inputRank - 1;
  Value zeroVal = arith::ConstantIndexOp::create(b, loc, 0);

  // Compute adjusted offsets: subtract padLow to get unpadded-space coords.
  SmallVector<OpFoldResult> adjustedOffsets(inputRank);
  for (int64_t d = 0; d < inputRank; ++d) {
    adjustedOffsets[d] =
        subOfrs(b, loc, srcIndices.sliceOffsets[d], padLow[d]);
  }

  // Compute valid_size and low pad amount along the vectorized dimension.
  // lowPadAmt = max(-adjustedVecCoord, 0)
  // readStart = max(adjustedVecCoord, 0)
  // validSize = clamp(inputSize - readStart, 0, tileSize - lowPadAmt)
  OpFoldResult adjustedVecCoordOfr = adjustedOffsets[vecInputDim];
  OpFoldResult inputExtentOfr = inputSizes[vecInputDim];
  // Use affine ops for the valid-size computation so that IREE's
  // IntegerDivisibilityAnalysis can track divisibility through the chain.
  // The analysis tracks affine.apply/min/max but NOT arith.subi/maxsi/minsi.
  MLIRContext *ctx = b.getContext();
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  AffineExpr d1 = getAffineDimExpr(1, ctx);

  // vecLowPadAmt = max(-adjustedVecCoord, 0)
  AffineMap negMaxMap =
      AffineMap::get(1, 0, {-d0, getAffineConstantExpr(0, ctx)}, ctx);
  OpFoldResult vecLowPadAmtOfr = affine::makeComposedFoldedAffineMax(
      b, loc, negMaxMap, {adjustedVecCoordOfr});

  // readStart = max(adjustedVecCoord, 0)
  AffineMap posMaxMap =
      AffineMap::get(1, 0, {d0, getAffineConstantExpr(0, ctx)}, ctx);
  OpFoldResult readStartOfr = affine::makeComposedFoldedAffineMax(
      b, loc, posMaxMap, {adjustedVecCoordOfr});

  // rawValid = inputExtent - readStart
  AffineMap subMap = AffineMap::get(2, 0, d0 - d1, ctx);
  OpFoldResult rawValidOfr = affine::makeComposedFoldedAffineApply(
      b, loc, subMap, {inputExtentOfr, readStartOfr});

  // availableSpace = tileSize - vecLowPadAmt
  OpFoldResult availableSpaceOfr = affine::makeComposedFoldedAffineApply(
      b, loc, subMap, {innerTileSize, vecLowPadAmtOfr});

  // clampedLow = max(rawValid, 0)
  OpFoldResult clampedLowOfr = affine::makeComposedFoldedAffineMax(
      b, loc, posMaxMap, {rawValidOfr});

  // validSize = min(clampedLow, availableSpace)
  AffineMap minMap = AffineMap::get(2, 0, {d0, d1}, ctx);
  OpFoldResult validSizeOfr = affine::makeComposedFoldedAffineMin(
      b, loc, minMap, {clampedLowOfr, availableSpaceOfr});

  Value validSize = getValueOrCreateConstantIndexOp(b, loc, validSizeOfr);
  Value vecLowPadAmt =
      getValueOrCreateConstantIndexOp(b, loc, vecLowPadAmtOfr);

  // Incorporate out-of-bounds status of non-vectorized dims (batch, spatial,
  // and channel). If an adjusted coord is outside [0, dimSize), the valid
  // region is empty, so clamp validSize and vecLowPadAmt to 0. This handles
  // input padding (padLow/padHigh > 0), output-alignment OOB from
  // non-wrapping delinearization (coord > dimSize even with zero padding),
  // and batch OOB when tile size exceeds the batch dimension.
  // Use affine ops + arith.muli to zero out validSize and vecLowPadAmt when a
  // non-vectorized dim is out of bounds. We compute a 0-or-1 factor using
  // affine.min/max, then multiply: both affine ops and arith.muli are tracked
  // by IntegerDivisibilityAnalysis, so divisibility is preserved.
  //
  // For coord in [0, dimSize):
  //   highOk = max(min(dimSize - coord, 1), 0) = 1
  //   lowOk  = max(min(coord + 1, 1), 0)      = 1
  //   factor = highOk * lowOk = 1 → validSize unchanged.
  // For coord < 0: lowOk = 0 → factor = 0 → validSize = 0.
  // For coord >= dimSize: highOk = 0 → factor = 0 → validSize = 0.
  AffineMap clampHighToOneMap = AffineMap::get(
      2, 0, {d0 - d1, getAffineConstantExpr(1, ctx)}, ctx);
  AffineMap clampLowToOneMap = AffineMap::get(
      1, 0, {d0 + 1, getAffineConstantExpr(1, ctx)}, ctx);

  auto checkDimBounds = [&](int64_t dim) {
    if (dim == vecInputDim)
      return;
    OpFoldResult coord = adjustedOffsets[dim];
    OpFoldResult dimSize = inputSizes[dim];
    // highOk = max(min(dimSize - coord, 1), 0): 1 when coord < dimSize, else 0.
    OpFoldResult highMin = affine::makeComposedFoldedAffineMin(
        b, loc, clampHighToOneMap, {dimSize, coord});
    OpFoldResult highOk =
        affine::makeComposedFoldedAffineMax(b, loc, posMaxMap, {highMin});
    // lowOk = max(min(coord + 1, 1), 0): 1 when coord >= 0, else 0.
    OpFoldResult lowMin = affine::makeComposedFoldedAffineMin(
        b, loc, clampLowToOneMap, {coord});
    OpFoldResult lowOk =
        affine::makeComposedFoldedAffineMax(b, loc, posMaxMap, {lowMin});
    // factor = highOk * lowOk: 1 when in-bounds, 0 otherwise.
    Value highOkVal = getValueOrCreateConstantIndexOp(b, loc, highOk);
    Value lowOkVal = getValueOrCreateConstantIndexOp(b, loc, lowOk);
    Value factor = arith::MulIOp::create(b, loc, highOkVal, lowOkVal);
    // validSize *= factor; vecLowPadAmt *= factor.
    validSize = arith::MulIOp::create(b, loc, validSize, factor);
    vecLowPadAmt = arith::MulIOp::create(b, loc, vecLowPadAmt, factor);
  };
  for (int64_t bPos : im2colOp.getBatchPos())
    checkDimBounds(bPos);
  for (int64_t mPos : im2colOp.getMPos())
    checkDimBounds(mPos);
  for (int64_t kPos : im2colOp.getKPos())
    checkDimBounds(kPos);

  // Clamp read offsets to [0, inputSize-1] to keep extract_slice in-bounds.
  // All non-vectorized dims need clamping because the adjusted offset can
  // be negative (padLow > offset), beyond the input extent (non-wrapping
  // delinearize OOB), or exceed the batch dim when tile > batch size.
  // Use affine.max/affine.min for consistency and to support folding.
  SmallVector<OpFoldResult> readOffsets(inputRank);
  // clampLo = max(adj, 0): use posMaxMap already defined.
  // dimMax = dimSize - 1: use (d0, d1) -> (d0 - d1) with dimSize and 1.
  // clamped = min(clampLo, dimMax)
  OpFoldResult oneOfr = b.getIndexAttr(1);
  for (int64_t d = 0; d < inputRank; ++d) {
    OpFoldResult clampLo =
        affine::makeComposedFoldedAffineMax(b, loc, posMaxMap,
                                            {adjustedOffsets[d]});
    OpFoldResult dimMax = affine::makeComposedFoldedAffineApply(
        b, loc, subMap, {inputSizes[d], oneOfr});
    readOffsets[d] =
        affine::makeComposedFoldedAffineMin(b, loc, minMap, {clampLo, dimMax});
  }

  // Output-side bounds checking: when the output tensor is larger than the
  // valid region (from FoldOutputPadIntoIm2col), positions beyond the valid
  // output size must produce pad_value. The valid size for each output dim
  // is product(output_sizes[d]).
  //
  // When outputIVs is empty (no loop IVs for output dims), fill with zeros
  // so bounds are still checked based on offsets alone.
  int64_t outputRank = im2colOp.getOutputRank();
  OpFoldResult zero = b.getIndexAttr(0);
  SmallVector<Value> zeroOutputIVs;
  SmallVector<OpFoldResult> zeroOutputOffsets;
  if (outputIVs.empty()) {
    zeroOutputIVs.resize(outputRank, zeroVal);
    outputIVs = zeroOutputIVs;
  }
  if (outputOffsets.empty()) {
    zeroOutputOffsets.resize(outputRank, zero);
    outputOffsets = zeroOutputOffsets;
  }
  SmallVector<OpFoldResult> outputValidSizes =
      computeOutputValidSizes(b, loc, im2colOp);
  // output_perm[actual] = canonical: use it directly to map actual -> canonical.
  ArrayRef<int64_t> outputPerm = im2colOp.getOutputPerm();

  // Non-vectorized output dims: if globalPos >= validSize, set validSize=0.
  // Skip dims where we can statically prove the position is in-bounds.
  // Non-vectorized output dims: use the same 0/1 factor approach as input
  // bounds checking. Output positions are always >= 0, so only check the
  // high bound (pos < validSize). Use affine.min/max + arith.muli.
  AffineMap outClampMap = AffineMap::get(
      2, 0, {d0 - d1, getAffineConstantExpr(1, ctx)}, ctx);
  for (int64_t d = 0; d < outputRank; ++d) {
    if (vecOutputDim.has_value() && d == vecOutputDim.value())
      continue;
    int64_t canonical = outputPerm[d];
    OpFoldResult globalPos =
        addOfrs(b, loc, outputOffsets[canonical], outputIVs[d]);
    // Static optimization: if both globalPos and validSize are known
    // constants and the position is provably in-bounds, skip the check.
    std::optional<int64_t> constPos = getConstantIntValue(globalPos);
    std::optional<int64_t> constBound =
        getConstantIntValue(outputValidSizes[canonical]);
    if (constPos && constBound && *constPos < *constBound)
      continue;
    // highOk = max(min(bound - pos, 1), 0): 1 when pos < bound, else 0.
    OpFoldResult highMin = affine::makeComposedFoldedAffineMin(
        b, loc, outClampMap, {outputValidSizes[canonical], globalPos});
    OpFoldResult highOk =
        affine::makeComposedFoldedAffineMax(b, loc, posMaxMap, {highMin});
    Value factor = getValueOrCreateConstantIndexOp(b, loc, highOk);
    validSize = arith::MulIOp::create(b, loc, validSize, factor);
    vecLowPadAmt = arith::MulIOp::create(b, loc, vecLowPadAmt, factor);
  }

  // Vectorized output dim: clamp validSize by remaining output valid count.
  // Skip if we can statically prove the full vector is in-bounds.
  if (vecOutputDim.has_value()) {
    int64_t canonical = outputPerm[vecOutputDim.value()];
    OpFoldResult globalVecStart = addOfrs(
        b, loc, outputOffsets[canonical], outputIVs[vecOutputDim.value()]);
    // Static optimization: if start + tileSize <= validSize, the entire
    // vector fits and no clamping is needed.
    std::optional<int64_t> constStart = getConstantIntValue(globalVecStart);
    std::optional<int64_t> constTile = getConstantIntValue(innerTileSize);
    std::optional<int64_t> constBound =
        getConstantIntValue(outputValidSizes[canonical]);
    if (!(constStart && constTile && constBound &&
          *constStart + *constTile <= *constBound)) {
      // outputClamp = max(outputBound - vecStart, 0)
      AffineMap clampSubMap = AffineMap::get(
          2, 0, {d0 - d1, getAffineConstantExpr(0, ctx)}, ctx);
      OpFoldResult outputClamp = affine::makeComposedFoldedAffineMax(
          b, loc, clampSubMap,
          {outputValidSizes[canonical], globalVecStart});
      // validSize = min(validSize, outputClamp)
      OpFoldResult vsOfr = affine::makeComposedFoldedAffineMin(
          b, loc, minMap, {validSize, outputClamp});
      validSize = getValueOrCreateConstantIndexOp(b, loc, vsOfr);
    }
  }

  return Im2colPaddingBounds{readOffsets, validSize, vecLowPadAmt};
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
  if (!val)
    return false;
  auto affineOp = val.getDefiningOp<affine::AffineApplyOp>();
  return affineOp &&
         affineOp.getMap().getResult(0).isMultipleOf(constTileSize.value());
}

std::optional<int64_t>
chooseDimToVectorize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                     ArrayRef<Range> iterationDomain,
                     ArrayRef<OpFoldResult> inputSizes,
                     ArrayRef<OpFoldResult> offsets) {
  int64_t innerInputDim = im2colOp.getInputRank() - 1;
  SmallVector<SmallVector<int64_t>> vectorizationMap =
      im2colOp.getInputToOutputDimVectorizationMap();
  SmallVector<int64_t> vectorizableOutputDims = vectorizationMap[innerInputDim];
  if (vectorizableOutputDims.empty()) {
    return std::nullopt;
  }
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
    OpFoldResult innerSliceSize = inputSizes[innerInputDim];
    if (kDimSet.contains(outputDimToVectorize)) {
      for (auto [kernelSize, mPos] :
           llvm::zip_equal(im2colOp.getMixedKernelSize(),
                           im2colOp.getMPos())) {
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
    OpFoldResult outputDimSize = iterationDomain[outputDimToVectorize].size;
    if (!willBeContiguousSlice(innerSliceSize, outputDimSize, offset)) {
      continue;
    }
    return outputDimToVectorize;
  }
  return std::nullopt;
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
