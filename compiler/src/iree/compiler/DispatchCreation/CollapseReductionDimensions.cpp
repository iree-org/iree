// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_COLLAPSEREDUCTIONDIMENSIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Check whether the given dimensions are contiguous in the result map.
/// If non of the dimension are present in the map return true as well.
static bool hasContiguousDims(AffineMap map, ArrayRef<unsigned> dims) {
  if (!map.isProjectedPermutation())
    return false;
  llvm::SmallDenseSet<unsigned> existingDims(dims.begin(), dims.end());
  for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
    if (map.getDimPosition(i) != dims[0]) {
      if (existingDims.count(map.getDimPosition(i))) {
        return false;
      }
      continue;
    }
    // Check that the following dimensions are match the order of `dims`
    for (unsigned j = 1, numDims = dims.size(); j < numDims; j++) {
      unsigned pos = i + j;
      if (pos >= map.getNumResults() || map.getDimPosition(pos) != dims[j]) {
        return false;
      }
    }
    break;
  }
  return true;
}

static SmallVector<ReassociationIndices>
collapseDimensions(linalg::LinalgOp linalgOp) {
  SmallVector<ReassociationIndices> collapseIndices;

  if (!IREE::Flow::isNonNullAndOutsideDispatch(linalgOp)) {
    return collapseIndices;
  }

  SmallVector<unsigned> reductionDims;
  linalgOp.getReductionDims(reductionDims);
  if (reductionDims.size() < 2)
    return collapseIndices;

  for (AffineMap map : linalgOp.getIndexingMapsArray()) {
    if (!hasContiguousDims(map, reductionDims))
      return collapseIndices;
  }
  ReassociationIndices indices;
  for (unsigned dim : reductionDims) {
    indices.push_back(int64_t(dim));
  }
  collapseIndices.push_back(indices);
  return collapseIndices;
}

struct CollapseReductionDimensionsPass final
    : public impl::CollapseReductionDimensionsPassBase<
          CollapseReductionDimensionsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalg::populateCollapseDimensions(patterns, collapseDimensions);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
