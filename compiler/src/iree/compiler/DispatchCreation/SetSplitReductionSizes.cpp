// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/DispatchCreation/Passes.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

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

      std::optional<SmallVector<int64_t>> tileSizes =
          getOuterReductionSizes(tilingOp);
      if (!tileSizes) {
        return;
      }
      IREE::LinalgExt::setSplitReductionAttribute(tilingOp, tileSizes.value());
    });
  }

private:
  /// Determine split reduction sizes for outer-reduction ops. This is
  /// targeting reductions such as those that appear in batch normalization,
  /// which reduce over outer dimensions of a tensor.
  std::optional<SmallVector<int64_t>>
  getOuterReductionSizes(PartialReductionOpInterface op) const {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(*op);
    if (!linalgOp) {
      LDBG() << "skipping op; not a linalg op";
      return std::nullopt;
    }
    if (!linalg::isReductionIterator(
            linalgOp.getIteratorTypesArray().front())) {
      LDBG() << "skipping op; not outer-reduction";
      return std::nullopt;
    }

    SmallVector<int64_t> opReductionSizes =
        getStaticReductionDimSizes(linalgOp);
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
};
} // namespace
} // namespace mlir::iree_compiler::DispatchCreation
