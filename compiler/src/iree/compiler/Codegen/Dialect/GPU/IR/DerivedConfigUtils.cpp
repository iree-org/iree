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
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::GPU {

static constexpr int64_t kPreferredCopyNumBits = 128;

// Helper to construct a list of tile sizes that simply uses the given vector
// size or the innerDimSize as the inner most tile size, whichever is smaller.
// All other dims are tiled to 1.
static SmallVector<int64_t>
getVectorSizeTileSizes(int64_t rank, int64_t innerDimSize, int64_t vectorSize) {
  SmallVector<int64_t> tileSizes(rank, 1);
  if (ShapedType::isDynamic(innerDimSize) || innerDimSize >= vectorSize) {
    tileSizes.back() = vectorSize;
  } else {
    tileSizes.back() = innerDimSize;
  }
  return tileSizes;
}

/// Derives the tiles sizes to use based on loop ranges, the number of threads,
/// and the optimal vector size. If |allowMultiDimCollapse| is true then this
/// will attempt to split the optimal vector size along multiple complete dims.
///
/// For example, with a vector size of 8 and loop ranges of [64 x 2 x 4] this
/// would return tile sizes of [1, 2, 4] if |allowMultiDimCollapse| is true and
/// [1, 1, 4] otherwise. If the loop ranges were instead [64 x 4 x 4] this
/// would give tile sizes of [1, 1, 4] no matter what because we won't be able
/// to collapse the vector.transfer_read that results from this choice of tile
/// size.
static SmallVector<int64_t>
getVectorTileSizesFromLoopRanges(SmallVector<int64_t> loopRanges,
                                 int64_t numThreads, int64_t vectorSize,
                                 bool allowMultiDimCollapse = true) {
  // If any loop ranges are dynamic, default to a simple vector size based
  // tile size.
  if (llvm::any_of(loopRanges, &ShapedType::isDynamic)) {
    return getVectorSizeTileSizes(loopRanges.size(), loopRanges.back(),
                                  vectorSize);
  }

  // If the number of loop trips are indivisible by the number of threads then
  // also default to just the vector size (i.e. [1, ..., 1, vector_size]).
  int64_t flatNumTrips = std::accumulate(loopRanges.begin(), loopRanges.end(),
                                         1, std::multiplies<int64_t>());
  if (flatNumTrips % numThreads != 0) {
    return getVectorSizeTileSizes(loopRanges.size(), loopRanges.back(),
                                  vectorSize);
  }
  SmallVector<int64_t> tileSizes(loopRanges.size(), 1);

  // Let the maximum possible vector size be the minimum between:
  //   - The requested vector size
  //   - The maximum vector size that avoids an exec mask
  int64_t maxVectorSize = std::min(vectorSize, flatNumTrips / numThreads);

  // Bail out to unit vector sizes if the inner most loop range is not divisible
  // by the vector size or vice-versa.
  int64_t innerMostRange = loopRanges.back();
  if (innerMostRange % maxVectorSize != 0 &&
      maxVectorSize % innerMostRange != 0) {
    return tileSizes;
  }

  // Let the inner most tile size be the smaller of the target vector size
  // and the inner most loop range. If |allowMultiDimCollapse| is false, return
  // here.
  tileSizes.back() = std::min(innerMostRange, maxVectorSize);
  if (innerMostRange >= maxVectorSize || !allowMultiDimCollapse) {
    return tileSizes;
  }

  maxVectorSize = maxVectorSize / innerMostRange;
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
  // TODO: Support multi-result
  if (linalgOp->getNumResults() != 1) {
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
  // Im2col cannot coalesce past the inner most dim so always default to only
  // the inner most tile size being the vector size (or smaller).
  return getVectorTileSizesFromLoopRanges(loopRanges, numThreads, vectorSize,
                                          /*allowMultiDimCollapse=*/false);
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
      .Default([](Operation *op) -> SmallVector<int64_t> { return {}; });
}

} // namespace mlir::iree_compiler::IREE::GPU
