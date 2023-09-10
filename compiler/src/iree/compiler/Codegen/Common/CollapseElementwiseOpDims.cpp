// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace {

/// Check whether the given dimensions are contiguous in the result map.
/// If none of the dimension are present in the map return true as well.
static bool areContiguousDims(int64_t dim, ArrayRef<int64_t> dims,
                              AffineMap map) {
  if (!map.isProjectedPermutation())
    return false;

  if (dims.empty())
    return true;

  // Checking for identity mapped dims for now.
  // TODO: Add constraints based on dim/dims.
  return map.getNumResults() == 0 || map.isMinorIdentityWithBroadcasting();
}

static SmallVector<ReassociationIndices>
getCollapsableDims(linalg::GenericOp genericOp) {
  if (genericOp.getNumParallelLoops() == 0) {
    return {};
  }

  // TODO(dcaballe): Implement a more sophisticated union-find?
  auto isContiguousParallelDim = [&](int64_t dim, utils::IteratorType iter,
                                     ArrayRef<int64_t> parallelDimSet,
                                     ArrayRef<AffineMap> indexingMaps) -> bool {
    if (!linalg::isParallelIterator(iter)) {
      return false;
    }

    for (AffineMap map : genericOp.getIndexingMapsArray()) {
      if (!areContiguousDims(dim, parallelDimSet, map))
        return false;
    }
    return true;
  };

  SmallVector<ReassociationIndices> collapsableDims;
  ReassociationIndices parallelDimSet;
  for (auto [dim, iter] : llvm::enumerate(genericOp.getIteratorTypesArray())) {
    if (isContiguousParallelDim(dim, iter, parallelDimSet,
                                genericOp.getIndexingMapsArray())) {
      parallelDimSet.push_back(dim);
    }

    if (!linalg::isParallelIterator(iter) ||
        dim == (genericOp.getNumParallelLoops() - 1)) {
      if (parallelDimSet.size() > 1) {
        collapsableDims.push_back(parallelDimSet);
        parallelDimSet.clear();
      }
    }
  }

  return collapsableDims;
}

struct CollapseElementwiseOpDimsPass
    : public CollapseElementwiseOpDimsBase<CollapseElementwiseOpDimsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Traverse the ops in the function to make sure we don't collapse
    // operations that would prevent fusion. For now, we only collapse
    // operations in functions with only generic ops, fill ops and tensor
    // empty ops.
    auto walkResult = funcOp.walk([](Operation *op) {
      // Linalg ops.
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        if (isa<linalg::FillOp>(op)) {
          return WalkResult::advance();
        }
        if (isa<linalg::GenericOp>(op)) {
          // TODO(dcaballe): Add extra checks on generic ops, if needed.
          return WalkResult::advance();
        }
        return WalkResult::interrupt();
      }

      // Tensor ops.
      if (isa<tensor::PackOp>(op)) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      return;
    }

    RewritePatternSet patterns(&getContext());
    linalg::populateCollapseDimensions(patterns, getCollapsableDims);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createCollapseElementwiseOpDimsPass() {
  return std::make_unique<CollapseElementwiseOpDimsPass>();
}

} // namespace iree_compiler
} // namespace mlir
