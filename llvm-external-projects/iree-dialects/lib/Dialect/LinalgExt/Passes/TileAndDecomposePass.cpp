// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

/// Tile iree_linalg_ext.attention.
static LogicalResult tileAttention(IREE::LinalgExt::AttentionOp attnOp,
                                   SmallVector<Operation *> &ops,
                                   RewriterBase &rewriter) {
  FailureOr<TilingResult> tiledImplementation =
      attnOp.getTiledImplementation(rewriter, /*offsets=*/{}, /*sizes=*/{});
  if (failed(tiledImplementation)) {
    return rewriter.notifyMatchFailure(
        attnOp, "failed to generate tiled implementation");
  }
  attnOp.getResults()[0].replaceAllUsesWith(
      (*tiledImplementation).tiledValues[0]);
  ops = (*tiledImplementation).tiledOps;
  return success();
}

/// Decompose tiled iree_linalg_ext.attention op.
static LogicalResult decomposeTiledAttention(Operation *tiledAttnOp,
                                             SmallVector<Operation *> &ops,
                                             RewriterBase &rewriter) {
  // Cast AttentionOp to AggregatedOpInterface since this where
  // `decomposeOperation` is implemented.
  auto decomposableTiledAttnOp =
      cast<mlir::linalg::AggregatedOpInterface>(tiledAttnOp);

  // Decompose AttentionOp.
  FailureOr<SmallVector<Value>> result =
      decomposableTiledAttnOp.decomposeOperation(rewriter);
  if (failed(result)) {
    return rewriter.notifyMatchFailure(
        tiledAttnOp, "tiled linalg_ext::AttentionOp could not be decomposed");
  }
  for (Value val : *result) {
    ops.push_back(val.getDefiningOp());
  }
  // The last three Values returned are the outputs of the decomposed attention
  // op. Of which we pop back the last two (max and sum) from the `ops` vector.
  ops.pop_back();
  ops.pop_back();
  unsigned totalResult = (*result).size();
  Value newSum = (*result)[totalResult - 1];
  Value newMax = (*result)[totalResult - 2];
  Value attentionResult = (*result)[totalResult - 3];
  rewriter.replaceOp(decomposableTiledAttnOp,
                     {attentionResult, newMax, newSum});
  return success();
}

/// Utility function which tiles and then decomposes attention op via
/// FlashAttention algorithm.
/// The attention operator computes:
/// matmul(softmax(matmul(Q, transpose(K))), V)
/// where: Q is the query matrix [B x N x d]
///        K is the key matrix   [B x S x d]
///        V is the value matrix [B x S x d]
///
/// The core algorithm is as follows:
/// For each element in B,
/// 1. Load a tile from the Q matrix of size T x d -> q
/// 2. Initialize statistics: running_sum, running_max
/// 3. for i = 0 to S with step T
///    a. Load a tile from the K matrix of size T x d -> k
///    b. Load a tile from the V matrix of size T x d -> v
///    c. Compute matmul_transpose_b(q, k) -> qkT
///    d. Compute max(max(qkT) along rows, old_max) -> new_max
///    e. Compute curent estimate of softmax: exp(qKT - current_max) -> s
///    f. Compute product of fixup and old_sum -> fsum
///    g. Compute sum(sum(qkT) along rows, fsum) -> new_sum
///    h. Compute 1.0 / new_sum -> inv_new_sum
///    i. Compute softmax = softmax * inv_new_sum
///    j. Truncate softmax to fp16
///    k. Compute fsum  * inv_new_sum * accumulator -> new_accumulator
///    j. Compute matmul(s, v) and add new_accumulator
///
///
LogicalResult tileAndDecomposeAttention(IREE::LinalgExt::AttentionOp attnOp,
                                        SmallVector<Operation *> &ops,
                                        RewriterBase &rewriter, bool onlyTile) {
  if (failed(tileAttention(attnOp, ops, rewriter)))
    return failure();
  if (onlyTile)
    return success();
  auto tiledAttnOp = cast<IREE::LinalgExt::AttentionOp>(ops[ops.size() - 1]);
  ops.pop_back();
  Operation *truncateToF16 = NULL;
  Type elementType = attnOp.getQueryType().getElementType();
  if (elementType.isF16()) {
    truncateToF16 = ops[ops.size() - 1];
    ops.pop_back();
  }
  if (failed(decomposeTiledAttention(tiledAttnOp, ops, rewriter)))
    return failure();
  if (truncateToF16)
    ops.push_back(truncateToF16);
  return success();
}

/// Tile iree_linalg_ext.winograd.input_transform op.
static LogicalResult tileWinogradInputTransformOp(
    WinogradInputTransformOp inputOp, RewriterBase &rewriter,
    WinogradInputTransformOp &tiledWinogradInputTransformOp) {
  FailureOr<TilingResult> tiledImplementation =
      inputOp.getTiledImplementation(rewriter, /*offsets=*/{}, /*sizes=*/{});
  if (failed(tiledImplementation)) {
    return rewriter.notifyMatchFailure(
        inputOp, "failed to generate tiled implementation");
  }
  inputOp.getResults()[0].replaceAllUsesWith(
      (*tiledImplementation).tiledValues[0]);
  tiledWinogradInputTransformOp =
      cast<WinogradInputTransformOp>((*tiledImplementation).tiledOps[0]);
  return success();
}

/// Decompose tiled iree_linalg_ext.winograd.input_transform op.
static LogicalResult
decomposeTiledWinogradInputTransformOp(Operation *tiledWinogradInputTransformOp,
                                       RewriterBase &rewriter) {
  // Cast WinogradInputTransformOp to AggregatedOpInterface since this where
  // `decomposeOperation` is implemented.
  auto decomposableTiledWinogradInputTransformOp =
      cast<mlir::linalg::AggregatedOpInterface>(tiledWinogradInputTransformOp);
  // Decompose WinogradInputTransformOp.
  FailureOr<SmallVector<Value>> result =
      decomposableTiledWinogradInputTransformOp.decomposeOperation(rewriter);
  if (failed(result)) {
    failed(rewriter.notifyMatchFailure(
        tiledWinogradInputTransformOp,
        "tiled linalg_ext::WinogradInputTransformOp could not be decomposed"));
    return failure();
  }
  rewriter.replaceOp(decomposableTiledWinogradInputTransformOp, *result);
  return success();
}

/// The input to WinogradInputTransformOp op is either (N, H, W, C) or (N, C,
/// H, W) but the output to this op is always (T, T, N, H', W', C). Since the
/// first two dimensions are used for the inner matrix multiplication, we
/// create the loop nest over (N, H', W', C).
LogicalResult tileAndDecomposeWinogradInputTransformOp(
    WinogradInputTransformOp inputOp, RewriterBase &rewriter, bool onlyTile) {
  WinogradInputTransformOp tiledWinogradInputTransformOp;
  if (failed(tileWinogradInputTransformOp(inputOp, rewriter,
                                          tiledWinogradInputTransformOp))) {
    return failure();
  }
  if (onlyTile)
    return success();
  return decomposeTiledWinogradInputTransformOp(tiledWinogradInputTransformOp,
                                                rewriter);
}

/// Tile iree_linalg_ext.winograd.output_transform op.
static LogicalResult tileWinogradOutputTransformOp(
    WinogradOutputTransformOp outputOp, RewriterBase &rewriter,
    WinogradOutputTransformOp &tiledWinogradOutputTransformOp) {
  FailureOr<TilingResult> tiledImplementation =
      outputOp.getTiledImplementation(rewriter, /*offsets=*/{}, /*sizes=*/{});
  if (failed(tiledImplementation)) {
    return rewriter.notifyMatchFailure(
        outputOp, "failed to generate tiled implementation");
  }
  outputOp.getResults()[0].replaceAllUsesWith(
      (*tiledImplementation).tiledValues[0]);
  tiledWinogradOutputTransformOp =
      cast<WinogradOutputTransformOp>((*tiledImplementation).tiledOps[0]);
  return success();
}

/// Decompose tiled iree_linalg_ext.winograd.output_transform op.
static LogicalResult decomposeTiledWinogradOutputTransformOp(
    Operation *tiledWinogradOutputTransformOp, RewriterBase &rewriter) {
  // Cast WinogradOutputTransformOp to AggregatedOpInterface since this where
  // `decomposeOperation` is implemented.
  auto decomposableTiledWinogradOutputTransformOp =
      cast<mlir::linalg::AggregatedOpInterface>(tiledWinogradOutputTransformOp);
  // Decompose WinogradOutputTransformOp.
  FailureOr<SmallVector<Value>> result =
      decomposableTiledWinogradOutputTransformOp.decomposeOperation(rewriter);
  if (failed(result)) {
    return rewriter.notifyMatchFailure(
        tiledWinogradOutputTransformOp,
        "tiled linalg_ext::WinogradOutputTransformOp could not be decomposed");
  }
  rewriter.replaceOp(decomposableTiledWinogradOutputTransformOp, *result);
  return success();
}

/// The input to WinogradOutputTransformOp is always (T, T, N, H', W', C)
/// but the output is either (N, H, W, C) or (N, C, H, W).
LogicalResult tileAndDecomposeWinogradOutputTransformOp(
    WinogradOutputTransformOp outputOp, RewriterBase &rewriter, bool onlyTile) {
  WinogradOutputTransformOp tiledWinogradOutputTransformOp;
  if (failed(tileWinogradOutputTransformOp(outputOp, rewriter,
                                           tiledWinogradOutputTransformOp))) {
    return failure();
  }
  if (onlyTile)
    return success();
  return decomposeTiledWinogradOutputTransformOp(tiledWinogradOutputTransformOp,
                                                 rewriter);
}

namespace {

// Tile and decompose LinalgExt ops based on a given target pipeline.
LogicalResult reifyLinalgExtTransform(func::FuncOp funcOp, bool onlyTile,
                                      StringRef targetPipeline) {
  IRRewriter rewriter(funcOp.getContext());
  LogicalResult resultOfTransformations = success();
  if (targetPipeline != "SPIRV") {
    funcOp.walk([&](IREE::LinalgExt::AttentionOp attnOp) {
      SmallVector<Operation *> ops;
      if (failed(tileAndDecomposeAttention(attnOp, ops, rewriter, onlyTile)))
        resultOfTransformations = failure();
      return WalkResult::advance();
    });
  }
  if (targetPipeline != "GPU") {
    funcOp.walk([&](WinogradInputTransformOp inputOp) {
      if (failed(tileAndDecomposeWinogradInputTransformOp(inputOp, rewriter,
                                                          onlyTile)))
        resultOfTransformations = failure();
      return WalkResult::advance();
    });
    funcOp.walk([&](WinogradOutputTransformOp outputOp) {
      if (failed(tileAndDecomposeWinogradOutputTransformOp(outputOp, rewriter,
                                                           onlyTile)))
        resultOfTransformations = failure();
      return WalkResult::advance();
    });
  }
  return resultOfTransformations;
}

} // namespace

namespace {
struct TileAndDecomposePass
    : public TileAndDecomposeBase<TileAndDecomposePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  TileAndDecomposePass() = default;
  TileAndDecomposePass(bool onlyTile, std::string targetPipeline) {
    this->onlyTile = onlyTile;
    this->targetPipeline = targetPipeline;
  }
  TileAndDecomposePass(const TileAndDecomposePass &pass) {
    onlyTile = pass.onlyTile;
    targetPipeline = pass.targetPipeline;
  }
  void runOnOperation() override;
};
} // namespace

void TileAndDecomposePass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  if (failed(reifyLinalgExtTransform(getOperation(), onlyTile, targetPipeline)))
    return signalPassFailure();
}

std::unique_ptr<Pass> createTileAndDecomposePass(bool onlyTile,
                                                 std::string targetPipeline) {
  return std::make_unique<TileAndDecomposePass>(onlyTile, targetPipeline);
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
