// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/DerivedConfigUtils.h"
#include <numeric>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
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

static SmallVector<int64_t>
getThreadTileSizesFromLoopRanges(SmallVector<int64_t> loopRanges,
                                 int64_t numThreads, int64_t vectorSize) {
  if (llvm::any_of(loopRanges,
                   [](int64_t s) { return ShapedType::isDynamic(s); })) {
    return getVectorSizeTileSizes(loopRanges.size(), loopRanges.back(),
                                  vectorSize);
  }

  int64_t flatNumTrips = std::accumulate(loopRanges.begin(), loopRanges.end(),
                                         1, std::multiplies<int64_t>());
  if (flatNumTrips % numThreads != 0) {
    return getVectorSizeTileSizes(loopRanges.size(), loopRanges.back(),
                                  vectorSize);
  }
  int64_t maxVectorSize = flatNumTrips / numThreads;

  while (maxVectorSize % vectorSize != 0) {
    vectorSize /= 2;
  }

  SmallVector<int64_t> tileSizes(loopRanges.size(), 0);
  tileSizes.back() = vectorSize;
  int64_t residualNumThreads = numThreads / (loopRanges.back() / vectorSize);
  for (int i = tileSizes.size() - 2, e = 0; i >= e; --i) {
    if (loopRanges[i] >= residualNumThreads) {
      tileSizes[i] = loopRanges[i] / residualNumThreads;
      residualNumThreads = 1;
      break;
    }
    tileSizes[i] = 1;
    residualNumThreads /= loopRanges[i];
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
  return getThreadTileSizesFromLoopRanges(loopRanges, numThreads, vectorSize);
}

SmallVector<int64_t>
deriveIm2colOpThreadTileSizes(IREE::LinalgExt::Im2colOp im2colOp,
                              int64_t numThreads) {
  if (!im2colOp.hasPureTensorSemantics()) {
    return {};
  }
  // TODO(Max191): Add `getStaticLoopRanges` to TilingInterface, and use it
  // here instead of `im2colOp.getOutputType().getShape()`. Then we can also
  // get rid of the specialization for Im2colOp vs LinalgOp and just use
  // TilingInterface ops.
  SmallVector<int64_t> loopRanges(im2colOp.getOutputType().getShape());
  int64_t vectorSize = kPreferredCopyNumBits /
                       getElementTypeOrSelf(im2colOp->getResultTypes()[0])
                           .getIntOrFloatBitWidth();
  return getThreadTileSizesFromLoopRanges(loopRanges, numThreads, vectorSize);
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
