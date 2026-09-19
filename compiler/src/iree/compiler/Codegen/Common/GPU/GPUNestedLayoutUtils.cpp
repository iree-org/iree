// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUNestedLayoutUtils.h"

#include <functional>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

namespace mlir::iree_compiler {

using namespace IREE::VectorExt;

static bool isBroadcast(AffineExpr expr) {
  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
    return constExpr.getValue() == 0;
  }
  return false;
}

SmallVector<Value> getTransferIndicesFromNestedLayout(
    OpBuilder &b, ValueRange indices, ArrayRef<int64_t> offsets,
    NestedLayoutAttr vectorLayout, AffineMap permutationMap,
    ValueRange warpIndices, ValueRange threadIndices) {

  int64_t rank = vectorLayout.getRank();
  // Permute the batch and outer vector offsets to match the order of
  // the vector dimensions using the inverse of the batch/offset order.
  ArrayRef<int64_t> batchOffsets(offsets.begin(), rank);
  ArrayRef<int64_t> outerVectorOffsets(offsets.begin() + rank, rank);

  SmallVector<Value> slicedIndices(indices);
  for (const auto &[i, dim] : llvm::enumerate(permutationMap.getResults())) {
    // Broadcasted dimension offsets can be used as-is; the read index is
    // invariant of the thread in such cases (and illegal for writes).
    if (isBroadcast(dim)) {
      continue;
    }
    unsigned pos = cast<AffineDimExpr>(dim).getPosition();
    Value offset = indices[pos];
    int64_t elementCount = vectorLayout.getElementTile()[i];
    Location loc = offset.getLoc();
    SmallVector<Value> ids = {
        warpIndices[i], arith::ConstantIndexOp::create(b, loc, batchOffsets[i]),
        arith::ConstantIndexOp::create(b, loc, outerVectorOffsets[i]),
        threadIndices[i], offset};
    // The order in which a vector dimension is "tiled" is
    // subgroups -> batches -> outer vectors -> threads -> elements
    SmallVector<int64_t> sizes = {
        vectorLayout.getSubgroupTile()[i], vectorLayout.getBatchTile()[i],
        vectorLayout.getOuterTile()[i], vectorLayout.getThreadTile()[i],
        elementCount};
    // The offset is often not an offset within `elementCount`, so, in general,
    // we can't mark this `disjoint`. However, if `offset` is known to be
    // a constant less than `elementCount`, we can do this, unlocking
    // potential optimizations.
    bool disjoint = false;
    if (std::optional<int64_t> offsetConst = getConstantIntValue(offset)) {
      disjoint = *offsetConst < elementCount;
    }
    slicedIndices[pos] =
        affine::AffineLinearizeIndexOp::create(b, loc, ids, sizes, disjoint);
  }
  return slicedIndices;
}

LogicalResult populateWarpAndThreadIndices(RewriterBase &rewriter,
                                           Value threadId, int64_t subgroupSize,
                                           NestedLayoutAttr vectorLayout,
                                           SmallVector<Value> &warpIndices,
                                           SmallVector<Value> &threadIndices) {
  // The delinearized thread IDs are returned from outer most to inner most,
  // i.e. before applying the layout described dimensions ordering.
  int64_t rank = vectorLayout.getRank();
  SmallVector<Value> threadIds =
      vectorLayout.computeThreadIds(threadId, subgroupSize, rewriter);
  if (threadIds.empty() && rank != 0) {
    return failure();
  }
  warpIndices = SmallVector<Value>(threadIds.begin(), threadIds.begin() + rank);
  threadIndices = SmallVector<Value>(threadIds.begin() + rank,
                                     threadIds.begin() + 2 * rank);
  return success();
}

SmallVector<int64_t> getElementVectorTileShape(NestedLayoutAttr vectorLayout) {
  int64_t rank = vectorLayout.getRank();
  SmallVector<int64_t> tileShape = vectorLayout.getDistributedShape();
  // We tile to a vector with BATCH, OUTER, and ELEMENT dimensions. So to access
  // the subvector only containing elements, we need indices in all BATCH and
  // OUTER (rank * 2) dimensions to have tile size 1.
  for (int i = 0, e = rank * 2; i < e; ++i) {
    tileShape[i] = 1;
  }
  return tileShape;
}

/// Distribute |total| across |shape| from the innermost dimension.
/// |shape| is updated in place to reflect the remaining shape after
/// distribution. Returns failure if any dimension doesn't divide evenly.
static FailureOr<SmallVector<int64_t>>
distributeFromInnermost(int64_t total, MutableArrayRef<int64_t> shape) {
  int64_t rank = shape.size();
  SmallVector<int64_t> result(rank, 1);
  int64_t remaining = total;
  for (int64_t i = rank - 1; i >= 0 && remaining > 1; --i) {
    int64_t take = std::min(remaining, shape[i]);
    if (shape[i] % take != 0) {
      return failure();
    }
    result[i] = take;
    shape[i] /= take;
    remaining /= take;
  }
  if (remaining != 1) {
    return failure();
  }
  return result;
}

/// Distribute |total| across |shape| from the outermost dimension.
/// |shape| is updated in place to reflect the remaining shape after
/// distribution. Returns failure if any dimension doesn't divide evenly.
static FailureOr<SmallVector<int64_t>>
distributeFromOutermost(int64_t total, MutableArrayRef<int64_t> shape) {
  int64_t rank = shape.size();
  SmallVector<int64_t> result(rank, 1);
  int64_t remaining = total;
  for (int64_t i = 0; i < rank && remaining > 1; ++i) {
    int64_t take = std::min(remaining, shape[i]);
    if (shape[i] % take != 0) {
      return failure();
    }
    result[i] = take;
    shape[i] /= take;
    remaining /= take;
  }
  if (remaining != 1) {
    return failure();
  }
  return result;
}

/// Try to compute a DMA-optimized NestedLayoutAttr for a single DMA size.
/// Returns failure if the layout is not compatible.
static FailureOr<NestedLayoutAttr>
getGlobalLoadDMALayoutForSize(MLIRContext *context, ArrayRef<int64_t> shape,
                              int64_t numThreads, int64_t subgroupSize,
                              int64_t elementBitWidth, int64_t dmaSize) {
  int64_t rank = shape.size();
  if (elementBitWidth <= 0) {
    return failure();
  }
  int64_t elementsPerDMA = dmaSize / elementBitWidth;
  if (elementsPerDMA == 0 || dmaSize % elementBitWidth != 0) {
    return failure();
  }

  int64_t totalElements = ShapedType::getNumElements(shape);
  int64_t elementsPerSubgroup = subgroupSize * elementsPerDMA;
  if (totalElements % elementsPerSubgroup != 0) {
    return failure();
  }

  SmallVector<int64_t> remainingShape(shape);

  auto elementResult = distributeFromInnermost(elementsPerDMA, remainingShape);
  if (failed(elementResult)) {
    return failure();
  }
  SmallVector<int64_t> elementTile = *elementResult;

  auto threadResult = distributeFromInnermost(subgroupSize, remainingShape);
  if (failed(threadResult)) {
    return failure();
  }
  SmallVector<int64_t> threadTile = *threadResult;

  int64_t numSubgroups = numThreads / subgroupSize;
  auto subgroupResult = distributeFromOutermost(numSubgroups, remainingShape);
  if (failed(subgroupResult)) {
    return failure();
  }
  SmallVector<int64_t> subgroupTile = *subgroupResult;

  SmallVector<int64_t> batchTile(remainingShape);
  SmallVector<int64_t> outerTile(rank, 1);
  SmallVector<int64_t> subgroupStrides = computeStrides(subgroupTile);
  SmallVector<int64_t> threadStrides = computeStrides(threadTile);

  return NestedLayoutAttr::get(context, subgroupTile, batchTile, outerTile,
                               threadTile, elementTile, subgroupStrides,
                               threadStrides);
}

FailureOr<NestedLayoutAttr>
getGlobalLoadDMALayout(MLIRContext *context, ArrayRef<int64_t> shape,
                       int64_t numThreads, int64_t subgroupSize,
                       int64_t elementBitWidth, ArrayRef<int64_t> dmaSizes,
                       std::optional<int64_t> swizzleAccessElems) {
  if (subgroupSize <= 0 || numThreads % subgroupSize != 0) {
    return failure();
  }

  SmallVector<int64_t> sorted(dmaSizes);
  llvm::sort(sorted, std::greater<>());
  if (swizzleAccessElems) {
    llvm::erase_if(sorted, [&](int64_t dmaSize) {
      if (dmaSize % elementBitWidth != 0) {
        return true;
      }
      int64_t elemsPerDMA = dmaSize / elementBitWidth;
      if (elemsPerDMA == 0) {
        return true;
      }
      return *swizzleAccessElems % elemsPerDMA != 0;
    });
  }
  for (int64_t dmaSize : sorted) {
    FailureOr<NestedLayoutAttr> layout = getGlobalLoadDMALayoutForSize(
        context, shape, numThreads, subgroupSize, elementBitWidth, dmaSize);
    if (succeeded(layout)) {
      return layout;
    }
  }
  return failure();
}

FailureOr<NestedLayoutAttr>
getDerivedThreadLayout(MLIRContext *context, ArrayRef<int64_t> workgroupSize,
                       ArrayRef<int64_t> logicalShape,
                       ArrayRef<int64_t> elementTile) {
  int64_t rank = logicalShape.size();
  if (rank == 0 || elementTile.size() != rank) {
    return failure();
  }

  SmallVector<int64_t> opShape(logicalShape);
  for (auto [size, element] : llvm::zip_equal(opShape, elementTile)) {
    if (ShapedType::isDynamic(size) || element <= 0 || size % element != 0) {
      return failure();
    }
    size /= element;
  }

  SmallVector<int64_t> threadTile(rank, 1);
  SmallVector<int64_t> threadStrides(rank, 0);
  int64_t residualThreads = ShapedType::getNumElements(workgroupSize);
  int64_t currStride = 1;
  for (auto [tile, stride, size] :
       llvm::reverse(llvm::zip(threadTile, threadStrides, opShape))) {
    int64_t threadBlock;
    if (residualThreads % size == 0) {
      threadBlock = size;
    } else if (size % residualThreads == 0) {
      threadBlock = residualThreads;
    } else {
      return failure();
    }

    tile = threadBlock;
    stride = currStride;
    size /= threadBlock;
    currStride *= threadBlock;
    residualThreads /= threadBlock;
  }

  SmallVector<int64_t> subgroupTile(rank, 1);
  SmallVector<int64_t> subgroupStrides(rank, 0);
  SmallVector<int64_t> outerTile(rank, 1);
  return NestedLayoutAttr::get(context, subgroupTile, opShape, outerTile,
                               threadTile, elementTile, subgroupStrides,
                               threadStrides);
}

} // namespace mlir::iree_compiler
