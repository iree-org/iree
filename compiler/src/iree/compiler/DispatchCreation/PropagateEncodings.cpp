// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-propagate-encodings"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_PROPAGATEENCODINGSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Pattern to swap `tensor.collapse_shape` -> `iree_encoding.set_encoding`
struct SwapEncodingOpWithTensorCollapseShapeOp
    : public OpRewritePattern<IREE::Encoding::SetEncodingOp> {
  using Base = OpRewritePattern<IREE::Encoding::SetEncodingOp>;
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override;
};

// TODO(#20179): Support the propagation through interfaces. It is supposed to
// be done with data-flow analysis.
struct PropagateEncodingsPass
    : public DispatchCreation::impl::PropagateEncodingsPassBase<
          PropagateEncodingsPass> {
  void runOnOperation() override;
};

} // namespace

/// For a result of a `tensor.collapse_shape` given the indexing map in a
/// consumer, return the iteration space expansion needed to swap the consumer
/// and the `tensor.collapse_shape`.
static std::pair<unsigned, SmallVector<ReassociationIndices>>
getIterationSpaceReassociationIndices(
    AffineMap indexingMap, ArrayRef<ReassociationIndices> dataReassocation) {
  llvm::SmallDenseMap<unsigned, unsigned> expandedBy;
  for (auto [index, result] : llvm::enumerate(indexingMap.getResults())) {
    unsigned dim = cast<AffineDimExpr>(result).getPosition();
    expandedBy[dim] = dataReassocation[index].size();
  }
  SmallVector<ReassociationIndices> iterationReassocation(
      indexingMap.getNumDims());
  unsigned currExpandedDim = 0;
  for (auto index : llvm::seq<size_t>(0, iterationReassocation.size())) {
    if (!expandedBy.contains(index)) {
      iterationReassocation[index] = {currExpandedDim++};
      continue;
    }
    iterationReassocation[index] = llvm::to_vector<2>(llvm::seq<int64_t>(
        currExpandedDim, currExpandedDim + expandedBy[index]));
    currExpandedDim += expandedBy[index];
  }
  return {currExpandedDim, iterationReassocation};
}

LogicalResult SwapEncodingOpWithTensorCollapseShapeOp::matchAndRewrite(
    IREE::Encoding::SetEncodingOp encodingOp, PatternRewriter &rewriter) const {
  std::optional<IREE::Encoding::EncodingAttr> maybeEncoding =
      IREE::Encoding::getEncodingAttr(encodingOp.getResultType());
  if (!maybeEncoding) {
    return failure();
  }
  auto collapseOp =
      encodingOp.getSource().getDefiningOp<tensor::CollapseShapeOp>();
  if (!collapseOp) {
    return failure();
  }
  if (!(IREE::Flow::isNonNullAndOutsideDispatch(encodingOp) &&
        IREE::Flow::isNonNullAndOutsideDispatch(collapseOp))) {
    return failure();
  }

  // For now abort on cases with multiple maps.
  // TODO: Support nested maps.
  if (maybeEncoding->getMapForOperandIndex() !=
      maybeEncoding->getLastMapForOperandIndex()) {
    return rewriter.notifyMatchFailure(encodingOp,
                                       "unhandled encoding with bcast map");
  }

  SmallVector<AffineMap> currIndexingMaps = maybeEncoding->getRootMaps();
  if (llvm::any_of(currIndexingMaps,
                   [](AffineMap m) { return !m.isProjectedPermutation(); })) {
    return rewriter.notifyMatchFailure(
        encodingOp,
        "unhandled indexing maps that arent all projected permutations");
  }
  auto indexingMap = maybeEncoding->getLastMapForOperandIndex();

  // Get a mapping from original iteration space to expanded iteration space.
  MLIRContext *ctx = rewriter.getContext();
  unsigned numDims;
  SmallVector<ReassociationIndices> iterationReassocationIndices;
  std::tie(numDims, iterationReassocationIndices) =
      getIterationSpaceReassociationIndices(
          indexingMap, collapseOp.getReassociationIndices());
  auto iterationReassocationExprs =
      convertReassociationIndicesToExprs(ctx, iterationReassocationIndices);

  // Update all the indexing maps.
  SmallVector<AffineMap> newIndexingMaps;
  newIndexingMaps.reserve(currIndexingMaps.size());
  for (auto currIndexingMap : currIndexingMaps) {
    SmallVector<AffineExpr> newResultExprs;
    for (auto result : currIndexingMap.getResults()) {
      unsigned position = cast<AffineDimExpr>(result).getPosition();
      newResultExprs.append(iterationReassocationExprs[position]);
    }
    auto newMap = AffineMap::get(numDims, 0, newResultExprs, ctx);
    newIndexingMaps.emplace_back(std::move(newMap));
  }
  auto newIndexingMapAttr = rewriter.getAffineMapArrayAttr(newIndexingMaps);
  auto newEncodingAttr = IREE::Encoding::EncodingAttr::get(
      ctx, maybeEncoding->getOperandIndex(), maybeEncoding->getOpType(),
      maybeEncoding->getElementTypes(), newIndexingMapAttr,
      maybeEncoding->getRoundDimsTo(), maybeEncoding->getLayouts());

  // Create the new encoding op.
  RankedTensorType newEncodingType =
      collapseOp.getSrcType().cloneWithEncoding(newEncodingAttr);
  Value newEncodingOp = rewriter.create<IREE::Encoding::SetEncodingOp>(
      encodingOp.getLoc(), newEncodingType, collapseOp.getSrc());
  Value newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
      collapseOp.getLoc(), encodingOp.getResultType(), newEncodingOp,
      collapseOp.getReassociationIndices());
  rewriter.replaceOp(encodingOp, newCollapseOp);
  return success();
}

void PropagateEncodingsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *ctx = &getContext();
  RewritePatternSet propagationPatterns(ctx);
  propagationPatterns.insert<SwapEncodingOpWithTensorCollapseShapeOp>(ctx);
  GreedyRewriteConfig config;
  config.fold = true;
  config.cseConstants = false;
  if (failed(applyPatternsGreedily(funcOp, std::move(propagationPatterns),
                                   config))) {
    funcOp.emitOpError("failed to propagate encodings");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
