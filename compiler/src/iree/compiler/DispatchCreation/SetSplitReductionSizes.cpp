// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
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

    if (convDims->inputChannel.empty() || convDims->outputChannel.empty() ||
        convDims->batch.empty() || convDims->filterLoop.empty()) {
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
        llvm::cast<ShapedType>(inputVal.getType()).getShape();
    ArrayRef<int64_t> filterShape =
        llvm::cast<ShapedType>(filterVal.getType()).getShape();
    ArrayRef<int64_t> outputShape =
        llvm::cast<ShapedType>(outputVal.getType()).getShape();

    if (ShapedType::isDynamicShape(inputShape) ||
        ShapedType::isDynamicShape(filterShape) ||
        ShapedType::isDynamicShape(outputShape)) {
      LDBG() << "skipping op; has dynamic shape";
      return std::nullopt;
    }

    AffineMap inputMap = linalgOp.getMatchingIndexingMap(input);
    AffineMap filterMap = linalgOp.getMatchingIndexingMap(filter);
    AffineMap outputMap = linalgOp.getMatchingIndexingMap(output);

    std::optional<int64_t> batchLastDim = outputMap.getResultPosition(
        getAffineDimExpr(convDims->batch.back(), outputMap.getContext()));
    if (!batchLastDim || batchLastDim.value() != outputShape.size() - 1) {
      LDBG() << "skipping op; not batch last layout";
      return std::nullopt;
    }

    std::optional<int64_t> inputChannelDim = filterMap.getResultPosition(
        getAffineDimExpr(convDims->inputChannel[0], filterMap.getContext()));
    std::optional<int64_t> filterDim = filterMap.getResultPosition(
        getAffineDimExpr(convDims->filterLoop[0], filterMap.getContext()));
    if (!inputChannelDim || !filterDim ||
        inputChannelDim.value() > filterDim.value()) {
      LDBG() << "skipping op; not channel first layout";
      return std::nullopt;
    }

    std::optional<int64_t> outputChannelDim = outputMap.getResultPosition(
        getAffineDimExpr(convDims->outputChannel[0], outputMap.getContext()));
    if (!outputChannelDim) {
      LDBG() << "skipping op; has no output channel dim";
      return std::nullopt;
    }

    std::optional<SmallVector<int64_t>> maybeSizes =
        getReductionDimSizes(op.getOperation());
    if (!maybeSizes) {
      LDBG() << "skipping op; failed to get reduction sizes";
      return std::nullopt;
    }

    // The constants below are determined based on empirical data.
    const int64_t largeDimSize = 512;
    const int64_t mediumDimSize = 128;
    const int64_t smallDimSize = 32;

    // When the batch and output channel sizes are large, the workload tends
    // to distributed across many workgroups, making split reduction little to
    // no effect.
    int64_t outputChannelSize = outputShape[outputChannelDim.value()];
    int64_t batchSize = outputShape[batchLastDim.value()];
    if (outputChannelSize >= largeDimSize && batchSize >= largeDimSize) {
      LDBG() << "skipping op; large output channel or batch size";
      return std::nullopt;
    }

    // When the input spatial sizes are small while the batch and output channel
    // sizes are relatively larger, split reduction often has no effect or even
    // degrades performance.
    for (auto dim : convDims->filterLoop) {
      for (auto [idx, e] : llvm::enumerate(inputMap.getResults())) {
        if (e.isFunctionOfDim(dim) && inputShape[idx] < smallDimSize &&
            outputChannelSize > mediumDimSize && batchSize > mediumDimSize) {
          LDBG() << "skipping op; small input spatial size";
          return std::nullopt;
        }
      }
    }

    // Only split along the input channel dimension.
    // TODO(vivian): split more reduction dimensions if needed.
    int64_t cDim = inputChannelDim.value();
    SmallVector<int64_t> tileSizes = std::move(*maybeSizes);
    if (tileSizes[cDim] == 1) {
      LDBG() << "skipping op; input channel size equals to 1";
      return std::nullopt;
    }
    tileSizes[cDim] = std::ceil(float(tileSizes[cDim]) / largeDimSize);
    return tileSizes;
  }
};
} // namespace
} // namespace mlir::iree_compiler::DispatchCreation
