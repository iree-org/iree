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

  // Fused Attention (aka Flash Attention) is parallelisable primarily over
  // B, M dimensions. It is possible to parallelise over N dimension, but it
  // leads to duplication of computation of the QK matmul and the softmax.
  //
  // If we do not have enough parallelism over M dimension, it is generally
  // not useful to do fused attention, instead decomposing is a better option.
  //
  // Also, one of the main reasons for using fused attention is that decomposing
  // leads to a tensor of size O(M x K2) as output, which generally grows
  // as O(seq_len^2) and consumes a lot of memory. However, if M is small, we
  // don't get much benefit on memory from doing fused attention.
  //
  // TODO: We should probably take batch size into account. Sometimes doing
  // split-k gives you good parallelism, and we can be fine.
  if (mSize <= kMParallelismThreshold) {
    return true;
  }

  // The main problem of Fused Attention is that it reduces parallelism across
  // K1, we cannot tile K1 in Fused Attention. This is generally okay, because
  // model people tell us it will be small, so we can completely unroll this
  // dimension. If these assumptions go wrong, we need to decompose, as
  // unrolling a large dimension can cause very high register pressure.
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

    IRRewriter rewriter(getOperation());
    for (AttentionOp attnOp : candidates) {
      auto aggregateOp =
          cast<linalg::AggregatedOpInterface>(attnOp.getOperation());
      rewriter.setInsertionPoint(attnOp);
      FailureOr<SmallVector<Value>> replacements =
          aggregateOp.decomposeOperation(rewriter);
      if (failed(replacements)) {
        return signalPassFailure();
      }
      rewriter.replaceAllOpUsesWith(attnOp, replacements.value());
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::GlobalOptimization
