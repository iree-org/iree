// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-producers-into-dispatch-regions"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FUSEENCODINGOPSINTODISPATCHREGIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

// Return true if the op is fusable with a SetEncodingOp consumer.
// For now, just check if it is a LinalgOp.
static bool isFusableWithSetEncoding(Operation *op) {
  return isa<linalg::LinalgOp>(op);
}

/// For a result of a `tensor.collapse_shape` given the indexing map
/// in a consumer, return the iteration space expansion needed
/// to swap the consumer and the `tensor.collapse_shape`.
static std::tuple<unsigned, SmallVector<ReassociationIndices>>
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

/// Pattern to swap `tensor.collapse_shape` -> `iree_encoding.set_encoding`
struct SwapEncodingOpWithTensorCollapseShapeOp
    : public OpRewritePattern<IREE::Encoding::SetEncodingOp> {
  using Base = OpRewritePattern<IREE::Encoding::SetEncodingOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {
    auto encoding = dyn_cast_or_null<IREE::Encoding::EncodingAttr>(
        encodingOp.getResultType().getEncoding());
    if (!encoding) {
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

    // For now abort on cases with bcast map.
    if (encoding.getBcastMap()) {
      return rewriter.notifyMatchFailure(encodingOp,
                                         "unhandled encoding with bcast map");
    }

    // Get the indexing map.
    SmallVector<AffineMap> currIndexingMaps =
        llvm::map_to_vector(encoding.getUserIndexingMaps(), [](Attribute attr) {
          return cast<AffineMapAttr>(attr).getValue();
        });
    if (llvm::any_of(currIndexingMaps,
                     [](AffineMap m) { return !m.isProjectedPermutation(); })) {
      return rewriter.notifyMatchFailure(
          encodingOp,
          "unhandled indexing maps that arent all projected permutations");
    }
    // Get the operand indexing map.
    auto indexingMap = encoding.getMapForOperandIndex();

    // Get a mapping from original iteration space to expanded iteration space.
    unsigned numDims;
    SmallVector<ReassociationIndices> iterationReassocationIndices;
    auto dataReassocationIndices = collapseOp.getReassociationIndices();
    std::tie(numDims, iterationReassocationIndices) =
        getIterationSpaceReassociationIndices(
            indexingMap, collapseOp.getReassociationIndices());
    auto iterationReassocationExprs = convertReassociationIndicesToExprs(
        rewriter.getContext(), iterationReassocationIndices);

    // Update all the indexing maps.
    SmallVector<AffineMap> newIndexingMaps;
    newIndexingMaps.reserve(currIndexingMaps.size());
    for (auto currIndexingMap : currIndexingMaps) {
      SmallVector<AffineExpr> newResultExprs;
      for (auto result : currIndexingMap.getResults()) {
        unsigned position = cast<AffineDimExpr>(result).getPosition();
        newResultExprs.append(iterationReassocationExprs[position]);
      }
      AffineMap newMap =
          AffineMap::get(numDims, 0, newResultExprs, rewriter.getContext());
      newIndexingMaps.emplace_back(std::move(newMap));
    }
    auto newIndexingMapAttr = rewriter.getAffineMapArrayAttr(newIndexingMaps);
    auto newEncodingAttr = IREE::Encoding::EncodingAttr::get(
        rewriter.getContext(), encoding.getOperandIndex(), encoding.getOpType(),
        encoding.getElementTypes(), newIndexingMapAttr, /*bcast_map=*/nullptr,
        encoding.getRoundDimsTo(), encoding.getLayouts());

    // new encoding op.
    RankedTensorType collapsedSrcType = collapseOp.getSrcType();
    auto newEncodingType = RankedTensorType::get(
        collapsedSrcType.getShape(), collapsedSrcType.getElementType(),
        newEncodingAttr);
    Value newEncodingOp = rewriter.create<IREE::Encoding::SetEncodingOp>(
        encodingOp.getLoc(), newEncodingType, collapseOp.getSrc());
    Value newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
        collapseOp.getLoc(), encodingOp.getResultType(), newEncodingOp,
        dataReassocationIndices);
    rewriter.replaceOp(encodingOp, newCollapseOp);
    return success();
  }
};

struct FuseEncodingOpsIntoDispatchRegionsPass
    : public DispatchCreation::impl::FuseEncodingOpsIntoDispatchRegionsPassBase<
          FuseEncodingOpsIntoDispatchRegionsPass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Apply some propagation.
    // TODO(MaheshRavishankar): This logic needs to be folded into propagation.
    {
      RewritePatternSet propagationPatterns(context);
      propagationPatterns.insert<SwapEncodingOpWithTensorCollapseShapeOp>(
          context);
      GreedyRewriteConfig config;
      config.fold = true;
      if (failed(applyPatternsGreedily(funcOp, std::move(propagationPatterns),
                                       config))) {
        funcOp.emitOpError("failed to propagate encodings");
        return signalPassFailure();
      }
    }

    IRRewriter rewriter(context);

    SmallVector<IREE::Encoding::SetEncodingOp> encodingOps;
    funcOp->walk([&](IREE::Encoding::SetEncodingOp encodingOp) {
      if (IREE::Flow::isNonNullAndOutsideDispatch(encodingOp)) {
        encodingOps.push_back(encodingOp);
      }
    });

    for (IREE::Encoding::SetEncodingOp encodingOp : encodingOps) {
      OpOperand &operand = encodingOp.getSourceMutable();
      auto producerDispatch =
          operand.get().getDefiningOp<IREE::Flow::DispatchRegionOp>();
      // Nothing to fuse with, so wrap the `encodingOp` in its own dispatch.
      if (!producerDispatch) {
        continue;
      }

      // Find producer operation inside of the dispatch region to determine if
      // fusion is possible.
      auto result = cast<OpResult>(operand.get());
      auto dispatchReturnOp = cast<IREE::Flow::ReturnOp>(
          producerDispatch.getBody().front().getTerminator());
      auto producerInRegion = dyn_cast<OpResult>(
          dispatchReturnOp->getOperand(result.getResultNumber()));
      if (!producerInRegion) {
        continue;
      }

      // Place the op in its own dispatch region if fusion is not possible.
      if (!isFusableWithSetEncoding(producerInRegion.getOwner())) {
        continue;
      }
      // Fuse the `encodingOp` into the producer dispatch region.
      if (failed(moveFollowingOpIntoDispatchRegion(rewriter, encodingOp,
                                                   producerDispatch))) {
        return signalPassFailure();
      }
    }

    // Dynamic dims may have dominance issues after pulling encoding ops into
    // producer dispatch regions, so we need to resolve tensor.dim ops.
    RewritePatternSet patterns(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    GreedyRewriteConfig config;
    config.cseConstants = false;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
