//===- TilingToTileOp.cpp - Tiling using to TileOp TilingInterface --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

struct TilingResult {
  TileOp tileOp;
  Operation *tiledOp;
};

static TilingResult tileToTileOp(PatternRewriter &rewriter, TilingInterface op,
                                 int64_t tiledDim, Value tileSize) {
  Location loc = op->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  // TODO: Handle the case where the `loopRanges` are empty.
  SmallVector<Range> loopRanges = op.getIterationDomain(rewriter);
  assert(loopRanges.size() >= 1 &&
         "expected at least a single loop in operation");
  auto destOperands = op.getDestinationOperands(rewriter);
  Operation *tiledOp = nullptr;
  auto tileOp = rewriter.create<TileOp>(
      loc, tileSize, destOperands, tiledDim,
      [&](OpBuilder &b, Location loc, Value offset, Value size,
          ValueRange outSlices) {
        // TODO: support `getTiledImplementation` with >1 produced tiled ops.
        int64_t nLoops = loopRanges.size();
        SmallVector<OpFoldResult> tiledOffsets, tiledSizes;
        tiledOffsets.reserve(nLoops);
        tiledSizes.reserve(nLoops);
        for (unsigned i = 0; i < nLoops; ++i) {
          if (i == tiledDim) {
            tiledOffsets.push_back(offset);
            tiledSizes.push_back(size);
          } else {
            tiledOffsets.push_back(loopRanges[i].offset);
            tiledSizes.push_back(loopRanges[i].size);
          }
        }
        SmallVector<Operation *> tiledOps = op.getTiledImplementation(
            b, outSlices, tiledOffsets, tiledSizes, /*tileDestOperands=*/false);
        assert(tiledOps.size() == 1 && "expected single tiled op");
        tiledOp = tiledOps.front();
        b.create<TileYieldOp>(loc, tiledOp->getResults());
      });
  return TilingResult{tileOp, tiledOp};
}

FailureOr<Operation *> mlir::iree_compiler::IREE::LinalgExt::
    LinalgExtTilingPattern::returningMatchAndRewrite(
        TilingInterface op, PatternRewriter &rewriter) const {
  /// Currently only handle single result operations.
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "Not a single result");

  // Get rank and tile sizes.
  // TODO: consider moving these checks to a common place that the TransformOp
  // verifier can also use.
  SmallVector<Value> tileSizes =
      options.tileSizeComputationFunction(rewriter, op);
  int64_t dim = -1;
  for (auto en : llvm::enumerate(tileSizes)) {
    Optional<int64_t> maybeTileSize = getConstantIntValue(en.value());
    if (maybeTileSize && *maybeTileSize == 0) continue;
    if (maybeTileSize && *maybeTileSize < 0)
      return rewriter.notifyMatchFailure(op, "Negative tile size");
    if (dim >= 0)
      return rewriter.notifyMatchFailure(op,
                                         "Could not find a single tiling dim");
    dim = en.index();
  }
  if (dim < 0)
    return rewriter.notifyMatchFailure(op,
                                       "Could not find a single tiling dim");

  /// Currently only handle tiling operations on a parallel iterator type.
  auto loopIteratorTypes = op.getLoopIteratorTypes();
  // Scalar operation, nothing to do, so just return.
  if (loopIteratorTypes.empty())
    return rewriter.notifyMatchFailure(op, "Scalar op, no tiling possible");
  ArrayRef<StringRef> loopIteratorTypesRef(loopIteratorTypes);
  if (loopIteratorTypesRef[dim] != getParallelIteratorTypeName())
    return rewriter.notifyMatchFailure(op, "Trying to tile a non-parallel dim");

  TilingResult tilingResult = tileToTileOp(rewriter, op, dim, tileSizes[dim]);
  rewriter.replaceOp(op, tilingResult.tileOp->getResults());

  return tilingResult.tiledOp;
}
