// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/DispatchCreation/Passes.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#define DEBUG_TYPE "iree-dispatch-creation-set-split-reduction-sizes"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_SETSPLITREDUCTIONSIZESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {
static std::optional<SmallVector<int64_t>>
getStaticReductionDimSizes(TilingInterface op) {
  // We only want dimension sizes that are statically known, but
  // `TilingInterface::getIterationDomain` will create unnecessary IR if any
  // dimensions are dynamic. Special case to linalg ops for now since they have
  // a method that doesn't create IR.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
  if (!linalgOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> dimSizes;
  for (auto [loopRange, loopType] : llvm::zip_equal(
           linalgOp.getStaticLoopRanges(), op.getLoopIteratorTypes())) {
    if (loopType == utils::IteratorType::reduction) {
      dimSizes.push_back(loopRange);
    }
  }
  return dimSizes;
}

struct SetSplitReductionSizesPass final
    : public impl::SetSplitReductionSizesPassBase<SetSplitReductionSizesPass> {
  using Base::Base;
  void runOnOperation() override {
    getOperation()->walk([&](PartialReductionOpInterface tilingOp) {
      std::optional<SmallVector<int64_t>> tileSizes =
          getUserSpecifiedTileSize(tilingOp);
      if (!tileSizes) {
        return;
      }
      IREE::LinalgExt::setSplitReductionAttribute(tilingOp, tileSizes.value());
    });
  }

private:
  std::optional<SmallVector<int64_t>>
  getUserSpecifiedTileSize(PartialReductionOpInterface op) const {
    {
      // First preference given to attribute set on the op.
      std::optional<SmallVector<int64_t>> attributeTileSize =
          IREE::LinalgExt::getSplitReductionSizes(op);
      if (attributeTileSize) {
        return attributeTileSize.value();
      }
    }

    // Skip ops that aren't reductions.
    unsigned numReduction = llvm::count_if(
        op.getLoopIteratorTypes(), [](utils::IteratorType iteratorType) {
          return iteratorType == utils::IteratorType::reduction;
        });
    if (numReduction == 0) {
      return std::nullopt;
    }

    if (targetSplitReductionSize <= 0) {
      return std::nullopt;
    }
    std::optional<SmallVector<int64_t>> opReductionSizes =
        getStaticReductionDimSizes(op);
    if (!opReductionSizes.has_value()) {
      return std::nullopt;
    }
    auto findSmallestFactorWithLowerBound =
        [](int64_t x, int64_t lowerBound) -> std::optional<int64_t> {
      assert(x > 0);
      assert(lowerBound > 0);
      // If 'sqrt(x)' is greater than the lower bound, we don't need to search
      // past it.
      int64_t xSqrt = std::ceil(std::sqrt(x));
      int64_t upperBound = (xSqrt > lowerBound) ? xSqrt : x;
      // We expect all numbers here to be relatively small, so just do trial
      // division (with a limit just to be safe).
      static constexpr int64_t kMaxIterations = 1 << 15;
      upperBound = std::min(upperBound, kMaxIterations);
      for (int64_t i = lowerBound; i <= upperBound; i++) {
        if (x % i == 0) {
          return i;
        }
      }
      return std::nullopt;
    };
    int64_t currentSplitReductionSize = 1;
    SmallVector<int64_t> tileSizes(opReductionSizes->size());
    // Tile dimensions until we reach or exceed the target. Tile sizes must
    // divide the dimension size evenly, and we start with inner dimensions as
    // we prefer tiling those.
    for (int64_t i = tileSizes.size() - 1; i >= 0; i--) {
      int64_t remainingSize =
          llvm::divideCeil(targetSplitReductionSize, currentSplitReductionSize);
      int64_t dimSize = (*opReductionSizes)[i];
      if (dimSize == ShapedType::kDynamic) {
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
};
} // namespace
} // namespace mlir::iree_compiler::DispatchCreation
