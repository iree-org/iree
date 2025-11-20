// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#define DEBUG_TYPE "iree-dispatch-creation-set-split-reduction-sizes"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_SETSPLITREDUCTIONSIZESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

static SmallVector<int64_t> getStaticReductionDimSizes(linalg::LinalgOp op) {
  SmallVector<int64_t> dimSizes;
  for (auto [loopRange, loopType] :
       llvm::zip_equal(op.getStaticLoopRanges(), op.getIteratorTypesArray())) {
    if (loopType == utils::IteratorType::reduction) {
      dimSizes.push_back(loopRange);
    }
  }
  return dimSizes;
}

static std::optional<SmallVector<int64_t>> getReductionDimSizes(Operation *Op) {
  auto fusionOp = dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(Op);
  if (!fusionOp) {
    LDBG() << "skipping op; not a LinalgFusionOpInterface op";
    return std::nullopt;
  }
  SmallVector<int64_t> loopRanges = fusionOp.getStaticLoopRanges();

  auto tilingInterfaceOp = dyn_cast<TilingInterface>(Op);
  if (!tilingInterfaceOp) {
    LDBG() << "skipping op; not a TilingInterface op";
    return std::nullopt;
  }

  SmallVector<utils::IteratorType> iters =
      tilingInterfaceOp.getLoopIteratorTypes();
  SmallVector<int64_t> reductionDimSizes;
  for (auto [range, it] : llvm::zip_equal(loopRanges, iters)) {
    if (it == utils::IteratorType::reduction) {
      reductionDimSizes.push_back(range);
    }
  }
  return reductionDimSizes;
}

static std::optional<int64_t>
findSmallestFactorWithLowerBound(int64_t x, int64_t lowerBound) {
  assert(x > 0);
  assert(lowerBound > 0);
  // We expect all numbers here to be relatively small, so just do trial
  // division (with a limit just to be safe).
  static constexpr int64_t kMaxIterations = 1 << 15;
  int64_t upperBound = std::min(x, kMaxIterations);
  for (int64_t i = lowerBound; i <= upperBound; i++) {
    if (x % i == 0) {
      return i;
    }
  }
  return std::nullopt;
};

namespace {
struct SetSplitReductionSizesPass final
    : public impl::SetSplitReductionSizesPassBase<SetSplitReductionSizesPass> {
  using Base::Base;
  void runOnOperation() override {
    // Skip pass if no target is set.
    if (splitReductionTargetSize <= 0) {
      return;
    }
    getOperation()->walk([&](PartialReductionOpInterface tilingOp) {
      // If the op already has its attribute set, don't change it.
      if (IREE::LinalgExt::getSplitReductionSizes(tilingOp).has_value()) {
        return;
      }
      // Skip ops that aren't reductions.
      unsigned numReduction = llvm::count_if(
          tilingOp.getLoopIteratorTypes(),
          [](utils::IteratorType iteratorType) {
            return iteratorType == utils::IteratorType::reduction;
          });
      if (numReduction == 0) {
        return;
      }

      // --- Case 1: Outer reduction ---
      if (auto tileSizes = getOuterReductionSizes(tilingOp)) {
        IREE::LinalgExt::setSplitReductionAttribute(tilingOp, *tileSizes);
        return;
      }

      // --- Case 2: Generic weight backward convolution ---
      if (auto tileSizes = getWeightBackwardReductionSizes(tilingOp)) {
        IREE::LinalgExt::setSplitReductionAttribute(tilingOp, *tileSizes);
        return;
      }

      // --- Case 3: Matmul-like operations ---
      if (auto tileSizes = getMatmulLikeReductionSizes(tilingOp)) {
        IREE::LinalgExt::setSplitReductionAttribute(tilingOp, *tileSizes);
        return;
      }

      // --- Case 4: arg_compare operations ---
      if (auto tileSizes = getArgCompareReductionSizes(tilingOp)) {
        IREE::LinalgExt::setSplitReductionAttribute(tilingOp, *tileSizes);
        return;
      }
    });
  }

private:
  /// Determine split reduction sizes for outer-reduction ops. This is
  /// targeting reductions such as those that appear in batch normalization,
  /// which reduce over outer dimensions of a tensor.
  std::optional<SmallVector<int64_t>>
  getOuterReductionSizes(PartialReductionOpInterface op) const {
    SmallVector<utils::IteratorType> iters = op.getLoopIteratorTypes();
    if (iters.empty() || iters.front() != utils::IteratorType::reduction) {
      LDBG() << "skipping op; not outer-reduction";
      return std::nullopt;
    }

    std::optional<SmallVector<int64_t>> maybeSizes =
        getReductionDimSizes(op.getOperation());
    if (!maybeSizes) {
      return std::nullopt;
    }
    SmallVector<int64_t> opReductionSizes = std::move(*maybeSizes);

    int64_t currentSplitReductionSize = 1;
    SmallVector<int64_t> tileSizes(opReductionSizes.size());
    // Tile dimensions until we reach or exceed the target. Tile sizes must
    // divide the dimension size evenly, and we start with inner dimensions as
    // we prefer tiling those.
    for (int64_t i = tileSizes.size() - 1; i >= 0; i--) {
      int64_t remainingSize =
          llvm::divideCeil(splitReductionTargetSize, currentSplitReductionSize);
      int64_t dimSize = opReductionSizes[i];
      if (dimSize == ShapedType::kDynamic) {
        LDBG() << "skipping op; has dynamic reduction dims";
        return std::nullopt;
      }
      int64_t tileSize =
          findSmallestFactorWithLowerBound(dimSize, remainingSize)
              .value_or(dimSize);
      tileSizes[i] = tileSize;
      currentSplitReductionSize *= tileSize;
    }
    return tileSizes;
  }

  /// Determines split reduction sizes for weight backward convolutions.
  /// These convolutions have a special CHWN layout, where the filter sizes
  /// (corresponding to output image sizes in forward convolutions) are
  /// typically large, while the output spatial dimensions are small. This makes
  /// the split reduction strategy particularly effective. Currently, splitting
  /// is only applied along the input channel dimension.
  std::optional<SmallVector<int64_t>>
  getWeightBackwardReductionSizes(PartialReductionOpInterface op) const {
    // First check if the input op is a convolution with CHWN layout.
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
    if (!linalgOp || !linalg::isaConvolutionOpInterface(linalgOp)) {
      LDBG() << "skipping op; not convolution";
      return std::nullopt;
    }

    FailureOr<mlir::linalg::ConvolutionDimensions> convDims =
        mlir::linalg::inferConvolutionDims(linalgOp);
    if (failed(convDims)) {
      LDBG() << "skipping op; failed to infer convolution dims";
      return std::nullopt;
    }

    // Require non-empty filter, output channel, and batch dimensions to derive
    // heuristics. The input channel dimension may be empty.
    if (convDims->outputChannel.empty() || convDims->batch.empty() ||
        convDims->filterLoop.empty()) {
      LDBG() << "skipping op; missing convolution dimensions";
      return std::nullopt;
    }

    OpOperand *input = linalgOp.getDpsInputOperand(0);
    OpOperand *filter = linalgOp.getDpsInputOperand(1);
    OpOperand *output = linalgOp.getDpsInitOperand(0);

    Value inputVal = input->get();
    Value filterVal = filter->get();
    Value outputVal = output->get();

    ArrayRef<int64_t> inputShape =
        cast<ShapedType>(inputVal.getType()).getShape();
    ArrayRef<int64_t> filterShape =
        cast<ShapedType>(filterVal.getType()).getShape();
    ArrayRef<int64_t> outputShape =
        cast<ShapedType>(outputVal.getType()).getShape();

    if (ShapedType::isDynamicShape(inputShape) ||
        ShapedType::isDynamicShape(filterShape) ||
        ShapedType::isDynamicShape(outputShape)) {
      LDBG() << "skipping op; has dynamic shape";
      return std::nullopt;
    }

    AffineMap filterMap = linalgOp.getMatchingIndexingMap(filter);
    AffineMap outputMap = linalgOp.getMatchingIndexingMap(output);

    auto getDimPositions = [&](ArrayRef<unsigned> dims, const AffineMap &map) {
      SmallVector<int64_t> positions;
      for (auto dim : dims) {
        positions.push_back(
            *map.getResultPosition(getAffineDimExpr(dim, map.getContext())));
      }
      llvm::sort(positions);
      return positions;
    };

    SmallVector<int64_t> batchPos = getDimPositions(convDims->batch, outputMap);
    SmallVector<int64_t> outputChannelPos =
        getDimPositions(convDims->outputChannel, outputMap);
    SmallVector<int64_t> outputImagePos =
        getDimPositions(convDims->outputImage, outputMap);
    SmallVector<int64_t> inputChannelPos =
        getDimPositions(convDims->inputChannel, filterMap);
    SmallVector<int64_t> filterPos =
        getDimPositions(convDims->filterLoop, filterMap);
    SmallVector<int64_t> depthPos = getDimPositions(convDims->depth, outputMap);

    if (outputChannelPos.empty() || batchPos.empty() || filterPos.empty()) {
      LDBG() << "skipping op; failed to get dim position from the map";
      return std::nullopt;
    }

    if (batchPos.back() != outputShape.size() - 1) {
      LDBG() << "skipping op; not batch last layout";
      return std::nullopt;
    }

    if (!inputChannelPos.empty() &&
        inputChannelPos.front() > filterPos.back()) {
      LDBG() << "skipping op; not channel first layout";
      return std::nullopt;
    }

    std::optional<SmallVector<int64_t>> maybeSizes =
        getReductionDimSizes(op.getOperation());
    if (!maybeSizes) {
      LDBG() << "skipping op; failed to get reduction sizes";
      return std::nullopt;
    }

    // Compute the product of the specified dimensions. If any dimension list is
    // empty, return 1.
    auto getSizeAt = [](ArrayRef<int64_t> shape, ArrayRef<int64_t> pos) {
      int64_t totalSize = 1;
      for (unsigned i : pos) {
        assert(!ShapedType::isDynamic(shape[i]));
        totalSize *= shape[i];
      }
      return totalSize;
    };

    int64_t outputChannelSize = getSizeAt(outputShape, outputChannelPos);
    int64_t batchSize = getSizeAt(outputShape, batchPos);
    int64_t imageSize = getSizeAt(outputShape, outputImagePos);
    int64_t depthSize = getSizeAt(outputShape, depthPos);

    // The constants below are determined based on empirical data.
    const int64_t largeParallelSize = 512;
    const int64_t largeReductionSize = 8192;
    const int64_t ratioThreshold = 64;

    // When the batch and output channel sizes are large, the workload tends
    // to distributed across many workgroups, making split reduction little to
    // no effect.
    if (outputChannelSize >= largeParallelSize &&
        batchSize >= largeParallelSize) {
      LDBG() << "skipping op; large output channel or batch size";
      return std::nullopt;
    }

    // When the reduction size is small relative to the output sizes, split
    // reduction often has no effect or even degrades performance.
    SmallVector<int64_t> tileSizes = std::move(*maybeSizes);
    int64_t reductionSize = llvm::product_of(tileSizes);
    int64_t ratio = reductionSize / std::sqrt(outputChannelSize * batchSize);
    if (ratio <= ratioThreshold && reductionSize < largeReductionSize) {
      LDBG() << "skipping op; small reduction size";
      return std::nullopt;
    }

    // Tile sizes are determined based on output (parallel dimension) sizes.
    // For larger outputs, the workload tends to be distributed across more
    // workgroups, thereby reducing the need for extensive splitting along the
    // reduction dimensions.
    int64_t outputSize = outputChannelSize * batchSize * imageSize * depthSize;
    int64_t limitParallelLoops;
    if (outputSize < 32 * 32) {
      limitParallelLoops = 2048;
    } else if (outputSize < 128 * 128) {
      limitParallelLoops = 128;
    } else if (outputSize < 256 * 256) {
      limitParallelLoops = 64;
    } else if (outputSize < 512 * 512) {
      limitParallelLoops = 16;
    } else {
      limitParallelLoops = std::min<int64_t>(16, tileSizes[0]);
    }

    // Based on the limitParallelLoops, assign tile size from the outermost
    // dimension to the innermost.
    for (int64_t i = 0; i < tileSizes.size(); i++) {
      int64_t lowerBound = llvm::divideCeil(tileSizes[i], limitParallelLoops);
      std::optional<int64_t> maybeTileSize =
          findSmallestFactorWithLowerBound(tileSizes[i], lowerBound);
      if (!maybeTileSize) {
        LDBG() << "skipping op; failed to find a split factor";
        return std::nullopt;
      }
      limitParallelLoops /= (tileSizes[i] / maybeTileSize.value());
      tileSizes[i] = maybeTileSize.value();
      // If the outer tile size is larger than 1, inner dimensions cannot be
      // split due to non-contiguous data.
      if (tileSizes[i] > 1) {
        break;
      }
    }
    return tileSizes;
  }

  /// Determines split reduction sizes for matmul-like operations where the K
  /// dimension is significantly larger than the M or N dimensions. Splitting
  /// can be applied across multiple reduction dimensions, with tile sizes
  /// varying according to the output (parallel dimension) sizes. Note that the
  /// constant thresholds are empirically derived from limited data and may not
  /// generalize to all cases.
  std::optional<SmallVector<int64_t>>
  getMatmulLikeReductionSizes(PartialReductionOpInterface op) const {
    // Matmul-like op should have at least 1 reduction, which is checked by the
    // contraction interface, and at least 2 parallel dimensions.
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
    if (!linalgOp) {
      LDBG() << "skipping op; not a linalg op";
      return std::nullopt;
    }

    FailureOr<linalg::ContractionDimensions> maybeContractionDims =
        linalg::inferContractionDims(linalgOp);
    if (failed(maybeContractionDims)) {
      LDBG() << "skipping op; failed to infer contraction dims";
      return std::nullopt;
    }

    if (linalgOp.getNumParallelLoops() < 2) {
      LDBG() << "skipping op; has less than 2 parallel dims";
      return std::nullopt;
    }

    std::optional<SmallVector<int64_t>> maybeSizes =
        getReductionDimSizes(op.getOperation());
    if (!maybeSizes) {
      LDBG() << "skipping op; failed to get reduction sizes";
      return std::nullopt;
    }

    linalg::ContractionDimensions contractionDims = *maybeContractionDims;
    auto batchDims = contractionDims.batch;
    auto mDims = contractionDims.m;
    auto nDims = contractionDims.n;
    auto kDims = contractionDims.k;

    SmallVector<int64_t> shapes = linalgOp.getStaticLoopRanges();
    if (llvm::any_of(shapes, ShapedType::isDynamic)) {
      LDBG() << "skipping op; has dynamic shape";
      return std::nullopt;
    }

    // Compute the product of the specified dimensions. If any dimension list is
    // empty, return 1.
    auto getSizeAt = [&shapes](ArrayRef<unsigned> idx) {
      int64_t totalSize = 1;
      for (unsigned i : idx) {
        assert(!ShapedType::isDynamic(shapes[i]));
        totalSize *= shapes[i];
      }
      return totalSize;
    };

    int64_t batchSize = getSizeAt(batchDims);
    int64_t mSize = getSizeAt(mDims);
    int64_t nSize = getSizeAt(nDims);
    int64_t kSize = getSizeAt(kDims);

    // The constants below are determined based on empirical data.
    const int64_t ratioThreshold = 384;
    const int64_t largeKSize = 24576;
    const int64_t largeMNSize = 1024;

    // When the M or N size is large, the workload tends to distributed across
    // many workgroups, making split reduction little to no effect.
    if (mSize > largeMNSize || nSize > largeMNSize) {
      LDBG() << "skipping op; large M or N size";
      return std::nullopt;
    }

    // When the reduction size is small relative to the M/N sizes, split
    // reduction often has no effect or even degrades performance.
    int64_t ratio = kSize / std::sqrt(mSize * nSize) / batchSize;
    if (ratio <= ratioThreshold && kSize < largeKSize) {
      LDBG() << "skipping op; small reduction size";
      return std::nullopt;
    }

    // Tile sizes are determined based on output (parallel dimension) sizes.
    // For larger outputs, the workload tends to be distributed across more
    // workgroups, thereby reducing the need for extensive splitting along the
    // reduction dimensions.
    SmallVector<int64_t> tileSizes = std::move(*maybeSizes);
    int64_t outputSize = mSize * nSize * batchSize;
    int64_t limitParallelLoops;
    if (outputSize < 16 * 16 || kSize > 1e7) {
      limitParallelLoops = 2048;
    } else if (outputSize < 64 * 64 || kSize > 1e6) {
      limitParallelLoops = 128;
    } else if (outputSize < 128 * 128) {
      limitParallelLoops = 64;
    } else if (outputSize < 384 * 384) {
      limitParallelLoops = 16;
    } else {
      limitParallelLoops = std::min<int64_t>(16, tileSizes[0]);
    }

    // Based on the limitParallelLoops, assign tile size from the outermost
    // dimension to the innermost.
    for (auto [i, tileSize] : llvm::enumerate(tileSizes)) {
      int64_t lowerBound = llvm::divideCeil(tileSize, limitParallelLoops);
      std::optional<int64_t> maybeTileSize =
          findSmallestFactorWithLowerBound(tileSize, lowerBound);
      if (!maybeTileSize) {
        LDBG() << "skipping op; failed to find a split factor";
        return std::nullopt;
      }
      limitParallelLoops /= (tileSize / maybeTileSize.value());
      tileSizes[i] = maybeTileSize.value();
      // If the outer tile size is larger than 1, inner dimensions cannot be
      // split due to non-contiguous data.
      if (tileSizes[i] > 1) {
        break;
      }
    }

    return tileSizes;
  }

  /// Determine split reduction sizes specifically for arg_compare operations.
  std::optional<SmallVector<int64_t>>
  getArgCompareReductionSizes(PartialReductionOpInterface op) const {
    auto argCompareOp =
        dyn_cast<IREE::LinalgExt::ArgCompareOp>(op.getOperation());
    if (!argCompareOp) {
      return std::nullopt;
    }

    ShapedType inputType = argCompareOp.getInputType();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t reductionDim = argCompareOp.getDimension();
    int64_t reductionSize = inputShape[reductionDim];

    if (reductionSize == ShapedType::kDynamic) {
      return std::nullopt;
    }

    if (reductionSize < splitReductionTargetSize) {
      return std::nullopt;
    }

    int64_t tileSize = findSmallestFactorWithLowerBound(
                           reductionSize, splitReductionTargetSize)
                           .value_or(reductionSize);

    LDBG() << "arg_compare split: dim=" << reductionDim
           << " size=" << reductionSize << " tile=" << tileSize;

    return SmallVector<int64_t>{tileSize};
  }
};
} // namespace
} // namespace mlir::iree_compiler::DispatchCreation
