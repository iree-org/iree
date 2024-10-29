// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/GlobalOptimization/Passes.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_DECOMPOSELOWPARALLELISMATTENTIONPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using namespace IREE::LinalgExt;

namespace {

const int64_t kDynamicDimMaxValue = 65536; // 2^16
/// TODO: The M heuristic is a conservative choice I(Groverkss) made.
/// We ideally need more data to determine when it is good to
/// decompose.
const int64_t kMParallelismThreshold = 4;
/// The K1 heuristic is simply based on the fact that if we have a tile size
/// bigger than 256, we should probably be tiling that dimension. Non decomposed
/// attention is not tilable along K1 diomension.
const int64_t kK1ParallelismThreshold = 256;

static int64_t calculateSize(ArrayRef<int64_t> bounds, ArrayRef<int64_t> dims) {
  int64_t size = 1;
  for (int64_t dim : dims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      // TODO: We should query range information to find the maximum value
      // possible instead.
      size *= kDynamicDimMaxValue;
    } else {
      size *= bounds[dim];
    }
  }
  return size;
}

static bool hasLowParallelism(AttentionOp attnOp) {
  FailureOr<AttentionOpDetail> opInfo =
      AttentionOpDetail::get(attnOp.getIndexingMapsArray());

  FailureOr<SmallVector<int64_t>> maybeBounds = attnOp.getStaticLoopRanges();
  if (failed(maybeBounds)) {
    return false;
  }

  SmallVector<int64_t> bounds = maybeBounds.value();

  int64_t mSize = calculateSize(bounds, opInfo->getMDims());
  int64_t k1Size = calculateSize(bounds, opInfo->getK1Dims());

  // TODO: Should we take batch size into account? Sometimes doing split-k
  // gives you good parallelism.
  if (mSize <= kMParallelismThreshold) {
    return true;
  }

  if (k1Size >= kK1ParallelismThreshold) {
    return true;
  }

  return false;
}

struct DecomposeLowParallelismAttentionPass
    : public impl::DecomposeLowParallelismAttentionPassBase<
          DecomposeLowParallelismAttentionPass> {

  void runOnOperation() override {
    SmallVector<AttentionOp> candidates;

    getOperation()->walk([&candidates](AttentionOp attnOp) {
      if (hasLowParallelism(attnOp)) {
        candidates.push_back(attnOp);
      }
    });

    for (AttentionOp attnOp : candidates) {
      auto aggregateOp =
          cast<linalg::AggregatedOpInterface>(attnOp.getOperation());
      OpBuilder b(attnOp);
      if (failed(aggregateOp.decomposeOperation(b))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::GlobalOptimization
