// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

/// Returns the `producerOp` loop dimensions that map to the `opResult` shape
/// dimensions.
static FailureOr<SmallVector<int64_t>>
getIteratorIndicesTiedToOpResult(linalg::LinalgOp producerOp,
                                 OpResult opResult) {
  assert(opResult.getOwner() == producerOp && "expected producer result");
  AffineMap indexingMap = producerOp.getTiedIndexingMapForResult(opResult);
  if (!indexingMap.isProjectedPermutation())
    return failure();
  SmallVector<int64_t> tiedIndices;
  for (AffineExpr expr : indexingMap.getResults())
    tiedIndices.push_back(expr.cast<AffineDimExpr>().getPosition());
  return tiedIndices;
}

/// Returns the `producerOp` loop ranges tiled to the `sliceOp` shape or failure
/// if the computation fails.
static FailureOr<SmallVector<Range>>
getTiledIterationDomain(OpBuilder &b, linalg::LinalgOp producerOp,
                        tensor::ExtractSliceOp sliceOp) {
  // Set the iteration domain to the ranges before tiling.
  SmallVector<Range> tiledIterationDomain =
      cast<TilingInterface>(producerOp.getOperation()).getIterationDomain(b);

  // Update the tiled dimensions given the `sliceOp` offset and sizes.
  FailureOr<SmallVector<int64_t>> tiedIndices =
      getIteratorIndicesTiedToOpResult(producerOp,
                                       sliceOp.source().cast<OpResult>());
  if (failed(tiedIndices))
    return failure();
  for (auto it : llvm::zip(tiedIndices.getValue(), sliceOp.getMixedOffsets()))
    tiledIterationDomain[std::get<0>(it)].offset =
        getValueOrCreateConstantIndexOp(b, sliceOp->getLoc(), std::get<1>(it));
  for (auto it : llvm::zip(tiedIndices.getValue(), sliceOp.getMixedSizes()))
    tiledIterationDomain[std::get<0>(it)].size =
        getValueOrCreateConstantIndexOp(b, sliceOp->getLoc(), std::get<1>(it));

  return tiledIterationDomain;
}

FailureOr<SmallVector<linalg::LinalgOp>>
mlir::iree_compiler::IREE::LinalgExt::LinalgExtFusionInContainingOpPattern::
    returningMatchAndRewrite(linalg::LinalgOp producerOp,
                             PatternRewriter &rewriter) const {
  if (!producerOp.hasTensorSemantics() && producerOp->getNumResults() != 1)
    return failure();

  // Search the producer slices accessed within the containing operation.
  SmallVector<tensor::ExtractSliceOp> sliceOps;
  for (Operation *user : producerOp->getUsers()) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!sliceOp)
      continue;
    if (sliceOp->getParentOp() != containingOp)
      continue;
    sliceOps.push_back(sliceOp);
  }
  // Check for a non-empty list of fusion opportunities.
  if (sliceOps.empty())
    return failure();

  auto tilingInterfaceOp = cast<TilingInterface>(producerOp.getOperation());
  SmallVector<Value> destinationOperands =
      tilingInterfaceOp.getDestinationOperands(rewriter);

  // Try to fuse the producer in-place of the tensor::ExtractSliceOps.
  SmallVector<linalg::LinalgOp> fusedOps;
  for (tensor::ExtractSliceOp sliceOp : sliceOps) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(sliceOp);

    // Compute the tiled iteration domain.
    FailureOr<SmallVector<Range>> tiledIterationDomain =
        getTiledIterationDomain(rewriter, producerOp, sliceOp);
    if (failed(tiledIterationDomain))
      return failure();

    // Compute the tile offsets and sizes.
    SmallVector<OpFoldResult> tileOffsets, tileSizes;
    for (auto range : tiledIterationDomain.getValue()) {
      tileOffsets.push_back(range.offset);
      tileSizes.push_back(range.size);
    }

    // Tile the producer.
    FailureOr<SmallVector<Operation *>> tiledOps =
        tilingInterfaceOp.getTiledImplementation(rewriter, destinationOperands,
                                                 tileOffsets, tileSizes, true);
    if (failed(tiledOps) || tiledOps->size() != 1)
      return failure();
    fusedOps.push_back(cast<linalg::LinalgOp>(tiledOps->front()));
  }

  // Replace the tensor::ExtractSliceOps.
  for (const auto &en : enumerate(sliceOps))
    rewriter.replaceOp(en.value(), fusedOps[en.index()]->getResult(0));
  return fusedOps;
}

FailureOr<FusionResult> LinalgExtFusionPattern::returningMatchAndRewrite(
    linalg::LinalgOp consumerOp, PatternRewriter &rewriter) const {
  // Try to fuse the producers of all operands to fuse.
  SmallVector<linalg::LinalgOp> fusedOps;
  for (int64_t operandToFuse : operandsToFuse) {
    // Check the operand exists.
    if (operandToFuse >= consumerOp->getNumOperands())
      return failure();

    // Check the operand is a slice of a producer result.
    auto sliceOp = consumerOp->getOperand(operandToFuse)
                       .getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp)
      return failure();
    auto producerOp = sliceOp.source().getDefiningOp<linalg::LinalgOp>();
    if (!producerOp || producerOp->getNumResults() != 1)
      return failure();
    assert(producerOp.hasTensorSemantics() &&
           "expects producer to have tensor semantics");

    // Compute the tiled iteration domain.
    FailureOr<SmallVector<Range>> tiledIterationDomain =
        getTiledIterationDomain(rewriter, producerOp, sliceOp);
    if (failed(tiledIterationDomain))
      return failure();

    // Compute the tile offsets and sizes.
    auto tilingInterfaceOp = cast<TilingInterface>(producerOp.getOperation());
    SmallVector<Value> destinationOperands =
        tilingInterfaceOp.getDestinationOperands(rewriter);
    SmallVector<OpFoldResult> tileOffsets, tileSizes;
    for (auto range : tiledIterationDomain.getValue()) {
      tileOffsets.push_back(range.offset);
      tileSizes.push_back(range.size);
    }

    // Insert the tiled producer before the consumer op.
    FailureOr<SmallVector<Operation *>> tiledProducerOps =
        tilingInterfaceOp.getTiledImplementation(rewriter, destinationOperands,
                                                 tileOffsets, tileSizes, true);
    if (failed(tiledProducerOps) || tiledProducerOps->size() != 1)
      return failure();
    auto tiledProducerOp = cast<linalg::LinalgOp>(tiledProducerOps->front());
    fusedOps.push_back(tiledProducerOp);
  }

  // Update the consumer in-place using the tiled producer results.
  SmallVector<Value> newOperands = consumerOp->getOperands();
  for (auto it : llvm::zip(operandsToFuse, fusedOps)) {
    int64_t operandToFuse = std::get<0>(it);
    linalg::LinalgOp fusedOp = std::get<1>(it);
    newOperands[operandToFuse] = fusedOp->getResult(0);
  }
  rewriter.updateRootInPlace(consumerOp,
                             [&]() { consumerOp->setOperands(newOperands); });

  return FusionResult{consumerOp, fusedOps};
}
