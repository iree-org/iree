// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/DerivedConfigUtils.h"
#include <numeric>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::GPU {

static constexpr int64_t kPreferredCopyNumBits = 128;

// Helper to construct a list of tile sizes that simply uses the given vector
// size or the `targetDimSize` as the inner/outer most tile size, whichever is
// smaller. All other dims are tiled to 1.
static SmallVector<int64_t> getVectorSizeTileSizes(int64_t rank,
                                                   int64_t targetDim,
                                                   int64_t targetDimSize,
                                                   int64_t vectorSize) {
  SmallVector<int64_t> tileSizes(rank, 1);
  tileSizes[targetDim] =
      ShapedType::isDynamic(targetDimSize) || targetDimSize >= vectorSize
          ? vectorSize
          : targetDimSize;
  return tileSizes;
}

/// Derives the tiles sizes to use based on loop ranges, the number of threads,
/// and the optimal vector size. If `vectorOutermost` is true, the vector tile
/// size is set along the outermost loop dimension, instead of the default
/// innermost dimension. If `allowMultiDimCollapse` is true then this
/// will attempt to split the optimal vector size along multiple complete dims.
///
/// For example, with a vector size of 8 and loop ranges of [64 x 2 x 4] this
/// would return tile sizes of [1, 2, 4] if `allowMultiDimCollapse` is true and
/// [1, 1, 4] otherwise. If the loop ranges were instead [64 x 4 x 4] this
/// would give tile sizes of [1, 1, 4] no matter what because we won't be able
/// to collapse the vector.transfer_read that results from this choice of tile
/// size.
static SmallVector<int64_t> getVectorTileSizesFromLoopRanges(
    SmallVector<int64_t> loopRanges, int64_t numThreads, int64_t vectorSize,
    bool allowMultiDimCollapse = true, bool vectorizeOutermost = false) {
  int64_t rank = loopRanges.size();
  int64_t targetDim = vectorizeOutermost ? 0 : rank - 1;
  int64_t targetRange = loopRanges[targetDim];

  // If any loop ranges are dynamic, default to a simple vector size based
  // tile size.
  if (llvm::any_of(loopRanges, &ShapedType::isDynamic)) {
    return getVectorSizeTileSizes(rank, targetDim, targetRange, vectorSize);
  }

  // If the number of loop trips are indivisible by the number of threads then
  // also default to just the vector size (e.g., [1, ..., 1, vector_size] when
  // `targetDim` is the innermost).
  int64_t flatNumTrips = std::accumulate(loopRanges.begin(), loopRanges.end(),
                                         1, std::multiplies<int64_t>());
  if (flatNumTrips % numThreads != 0) {
    return getVectorSizeTileSizes(rank, targetDim, targetRange, vectorSize);
  }

  // Let the maximum possible vector size be the minimum between:
  //   - The requested vector size
  //   - The maximum vector size that avoids an exec mask
  int64_t maxVectorSize = std::min(vectorSize, flatNumTrips / numThreads);

  // Bail out to unit vector sizes if the target loop range is not divisible
  // by the vector size or vice-versa.
  SmallVector<int64_t> tileSizes(rank, 1);
  if (targetRange % maxVectorSize != 0 && maxVectorSize % targetRange != 0) {
    return tileSizes;
  }

  // Let the tile size for the target dim be the smaller of the vector size and
  // the target loop range. Return here, if `allowMultiDimCollapse` is false, or
  // `vectorizeOutermost` is true, because we don't expect consecutive
  // dimensions to be vectorizable contiguously for these cases.
  tileSizes[targetDim] = std::min(targetRange, maxVectorSize);
  if (targetRange >= maxVectorSize || !allowMultiDimCollapse ||
      vectorizeOutermost) {
    return tileSizes;
  }

  maxVectorSize = maxVectorSize / targetRange;
  for (int64_t i = loopRanges.size() - 2, e = 0; i >= e; --i) {
    // Only increase the tile size if the remaining vector size is divisible
    // by the loop range (and thus range <= remaining vector size).
    int64_t range = loopRanges[i];
    if (maxVectorSize % range != 0) {
      break;
    }
    tileSizes[i] = range;
    maxVectorSize = maxVectorSize / range;
  }

  return tileSizes;
}

SmallVector<int64_t> deriveLinalgOpThreadTileSizes(linalg::LinalgOp linalgOp,
                                                   int64_t numThreads) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return {};
  }
  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
  int64_t vectorSize = kPreferredCopyNumBits /
                       getElementTypeOrSelf(linalgOp->getResultTypes()[0])
                           .getIntOrFloatBitWidth();
  SmallVector<int64_t> tileSizes =
      getVectorTileSizesFromLoopRanges(loopRanges, numThreads, vectorSize);
  for (auto [tileSize, iterType] :
       llvm::zip(tileSizes, linalgOp.getIteratorTypesArray())) {
    if (iterType == utils::IteratorType::reduction) {
      tileSize = 0;
    }
  }
  return tileSizes;
}

SmallVector<int64_t>
deriveIm2colOpThreadTileSizes(IREE::LinalgExt::Im2colOp im2colOp,
                              int64_t numThreads) {
  if (!im2colOp.hasPureTensorSemantics()) {
    return {};
  }
  SmallVector<int64_t> loopRanges(im2colOp.getOutputType().getShape());
  int64_t vectorSize = kPreferredCopyNumBits /
                       getElementTypeOrSelf(im2colOp->getResultTypes()[0])
                           .getIntOrFloatBitWidth();

  // If the im2col input tensor has the batch dim at last, im2col output tensor
  // has an implicit transpose to move the batch dim in front, and tiling should
  // be along the batch dim. Currently only a single batch dim is supported for
  // tiling along the batch dim.
  unsigned innerDim = im2colOp.getInputRank() - 1;
  bool singleBatchDimInnermost = im2colOp.getBatchPos().size() == 1 &&
                                 im2colOp.getBatchPos().back() == innerDim;
  bool vectorizeOutermost = singleBatchDimInnermost ? true : false;

  // Im2col cannot coalesce past the inner/outer most dim so always default to
  // only the inner/outer most tile size being the vector size (or smaller).
  return getVectorTileSizesFromLoopRanges(loopRanges, numThreads, vectorSize,
                                          /*allowMultiDimCollapse=*/false,
                                          vectorizeOutermost);
}

SmallVector<int64_t> deriveThreadTileSizes(Operation *op) {
  std::optional<SmallVector<int64_t>> workgroupSize =
      getWorkgroupSize(op->getParentOfType<FunctionOpInterface>());
  if (!workgroupSize) {
    return {};
  }
  int64_t numThreads =
      std::accumulate(workgroupSize->begin(), workgroupSize->end(), 1,
                      std::multiplies<int64_t>());
  return TypeSwitch<Operation *, SmallVector<int64_t>>(op)
      .Case([&](linalg::LinalgOp linalgOp) -> SmallVector<int64_t> {
        return deriveLinalgOpThreadTileSizes(linalgOp, numThreads);
      })
      .Case([&](IREE::LinalgExt::Im2colOp im2colOp) -> SmallVector<int64_t> {
        return deriveIm2colOpThreadTileSizes(im2colOp, numThreads);
      })
      .Case([&](IREE::LinalgExt::ScatterOp scatterOp) -> SmallVector<int64_t> {
        int64_t loopDepth = scatterOp.getLoopIteratorTypes().size();
        SmallVector<int64_t> loopBounds =
            scatterOp.getStaticLoopRanges().value_or(
                SmallVector<int64_t>(loopDepth, ShapedType::kDynamic));
        int64_t elemBits = scatterOp.getOriginalType().getElementTypeBitWidth();
        int64_t vectorSize = kPreferredCopyNumBits / elemBits;
        return getVectorTileSizesFromLoopRanges(loopBounds, numThreads,
                                                vectorSize);
      })
      .Default([&](Operation *op) -> SmallVector<int64_t> { return {}; });
}

// TODO: make it a query.
static const int64_t kDefaultGlobalLoadBitSizePerThread = 32;

SmallVector<int64_t> globalLoadDMATileSizes(Operation *op) {
  auto funcOp = op->getParentOfType<FunctionOpInterface>();

  std::optional<SmallVector<int64_t>> workgroupSize = getWorkgroupSize(funcOp);
  if (!workgroupSize) {
    return {};
  }
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return {};
  }
  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();

  int64_t targetSubgroupSize = getGPUTargetAttr(op).getPreferredSubgroupSize();
  int64_t subgroupLoadSize =
      (kDefaultGlobalLoadBitSizePerThread * targetSubgroupSize) /
      getElementTypeOrSelf(linalgOp->getResultTypes()[0])
          .getIntOrFloatBitWidth();
  int64_t numThreads =
      std::accumulate(workgroupSize->begin(), workgroupSize->end(), 1,
                      std::multiplies<int64_t>());
  SmallVector<int64_t> tileSizes = getVectorTileSizesFromLoopRanges(
      loopRanges, numThreads, subgroupLoadSize);
  return tileSizes;
}

} // namespace mlir::iree_compiler::IREE::GPU
