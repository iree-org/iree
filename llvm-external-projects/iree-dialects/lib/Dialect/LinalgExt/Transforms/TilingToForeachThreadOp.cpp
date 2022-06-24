// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

static FailureOr<TilingResult> tileToForeachOp(PatternRewriter &rewriter,
                                               TilingInterface op,
                                               ValueRange numThreads) {
  Location loc = op->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<Range> loopRanges = op.getIterationDomain(rewriter);
  if (loopRanges.empty())
    return failure();

  auto destOperands = op.getDestinationOperands(rewriter);
  assert(destOperands.size() == 1 && "expected single dest operand");

  auto nonZeroNumThreads = llvm::make_filter_range(
      numThreads, [](Value v) { return !isConstantIntValue(v, 0); });

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Operation *tiledOp = nullptr;
  Operation *tileOp = rewriter.create<scf::ForeachThreadOp>(
      loc, llvm::to_vector<4>(nonZeroNumThreads),
      [&](OpBuilder &b, Location loc, ValueRange threadIds) {
        // TODO: support `getTiledImplementation` with >1 produced tiled ops.
        int64_t nLoops = loopRanges.size();
        SmallVector<OpFoldResult> tiledOffsets, tiledSizes;
        tiledOffsets.reserve(nLoops);
        tiledSizes.reserve(nLoops);
        for (unsigned loopIdx = 0, threadIdIdx = 0; loopIdx < nLoops;
             ++loopIdx) {
          assert(isConstantIntValue(loopRanges[loopIdx].stride, 1) &&
                 "only stride-1 supported atm");
          bool overflow = loopIdx >= numThreads.size();
          Value nThreads = overflow ? Value() : numThreads[loopIdx];
          bool isZero = !overflow && isConstantIntValue(nThreads, 0);
          // Degenerate case: take the whole domain.
          if (overflow || isZero) {
            tiledOffsets.push_back(loopRanges[loopIdx].offset);
            tiledSizes.push_back(loopRanges[loopIdx].size);
            continue;
          }

          // Tiled case: compute the offset and size.
          AffineExpr i, j, M, N, O;
          bindDims(rewriter.getContext(), i, j);
          bindSymbols(rewriter.getContext(), M, N, O);
          Value size = loopRanges[loopIdx].size;
          Value offset = loopRanges[loopIdx].offset;
          Value threadId = threadIds[threadIdIdx];
          // Symbolic fixed max size per thread.
          // TODO: floor + 0/1 depending on case for better load-balancing.
          Value maxSizePerThread = rewriter.createOrFold<AffineApplyOp>(
              loc, (M - N).ceilDiv(O), ValueRange{size, offset, nThreads});
          // Dynamic offset shifted by threadId * maxSizePerThread.
          Value offsetPerThread = rewriter.createOrFold<AffineApplyOp>(
              loc, i + j * M, ValueRange{offset, threadId, maxSizePerThread});
          // Dynamic upper-bound depending on the threadId.
          Value sizeMinusOffsetPerThread = rewriter.createOrFold<AffineApplyOp>(
              loc, -i + M, ValueRange{offsetPerThread, size});
          // Dynamic size that each thread processes.
          AffineBuilder AB(rewriter, loc);
          Value tileSizePerThread =
              AB.min(ValueRange{sizeMinusOffsetPerThread, maxSizePerThread});
          tiledOffsets.push_back(offsetPerThread);
          // TODO: if tileSizePerThread <= 0 early exit.
          tiledSizes.push_back(AB.max(ValueRange{zero, tileSizePerThread}));
          ++threadIdIdx;
        }

        SmallVector<Operation *> tiledOps =
            op.getTiledImplementation(b, destOperands, tiledOffsets, tiledSizes,
                                      /*tileDestOperands=*/true);
        assert(tiledOps.size() == 1 && "expected a single produced tiled op");
        tiledOp = tiledOps.front();

        auto tilingInterfaceOp = dyn_cast<TilingInterface>(tiledOp);
        assert(tilingInterfaceOp && "Tiled op is not a TilingInterface");

        auto tiledDestOperands =
            tilingInterfaceOp.getDestinationOperands(rewriter);

        // Create terminator with parallel subset insert operations.
        auto performConcurrentlyOp = b.create<scf::PerformConcurrentlyOp>(loc);
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(performConcurrentlyOp.getBody());
        for (auto it :
             llvm::zip(tiledDestOperands, tilingInterfaceOp->getResults(),
                       destOperands)) {
          createMatchingParallelSubsetInsertOp(
              b, loc,
              cast<tensor::ExtractSliceOp>(std::get<0>(it).getDefiningOp()),
              std::get<1>(it), std::get<2>(it));
        }
      });
  return TilingResult{tileOp, tiledOp};
}

FailureOr<TilingResult>
mlir::iree_compiler::IREE::LinalgExt::ForeachThreadTilingPattern::
    returningMatchAndRewrite(TilingInterface op,
                             PatternRewriter &rewriter) const {
  /// Currently only handle single result operations.
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "Not a single result");

  /// Currently only handle tiling operations on a parallel iterator type.
  auto loopIteratorTypes = op.getLoopIteratorTypes();
  // Scalar operation, nothing to do, so just return.
  if (loopIteratorTypes.empty())
    return rewriter.notifyMatchFailure(op, "Scalar op, no tiling possible");

  // Get rank and tile sizes.
  // TODO: consider moving these checks to a common place that the TransformOp
  // verifier can also use.
  SmallVector<Value> numThreads =
      options.tileSizeComputationFunction(rewriter, op);
  for (auto it : llvm::zip(numThreads, loopIteratorTypes)) {
    Optional<int64_t> maybeTileSize = getConstantIntValue(std::get<0>(it));
    if (maybeTileSize && *maybeTileSize == 0)
      continue;
    if (maybeTileSize && *maybeTileSize < 0)
      return rewriter.notifyMatchFailure(op, "Negative tile size");
    if (std::get<1>(it) != getParallelIteratorTypeName())
      return rewriter.notifyMatchFailure(op,
                                         "Trying to tile a non-parallel dim");
  }

  FailureOr<TilingResult> tilingResult =
      tileToForeachOp(rewriter, op, numThreads);
  if (failed(tilingResult))
    return failure();

  rewriter.replaceOp(op, tilingResult->tileOp->getResults());

  return tilingResult;
}
