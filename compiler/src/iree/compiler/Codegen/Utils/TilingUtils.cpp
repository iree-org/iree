// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/TilingUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-codegen-tiling-utils"

namespace mlir::iree_compiler {

static FailureOr<TilingResult>
bubbleUpExpandShapeSlice(OpBuilder &b, tensor::ExpandShapeOp expandShapeOp,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) {
  // Helper variables and function for accumulating the new offset and length
  // values.
  Location loc = expandShapeOp->getLoc();
  AffineExpr dim0, sym0;
  bindDims(b.getContext(), dim0);
  bindSymbols(b.getContext(), sym0);
  // Multiply two integers.
  auto mul = [&](OpFoldResult v1, OpFoldResult v2) {
    auto mulMap = AffineMap::get(1, 1, {dim0 * sym0});
    return affine::makeComposedFoldedAffineApply(b, loc, mulMap, {v1, v2});
  };

  // Compute new offsets, lengths, low padding, high padding.
  SmallVector<OpFoldResult> newOffsets, newLengths, newStrides;

  int64_t expandedDim = 0;
  for (ReassociationIndices indices : expandShapeOp.getReassociationIndices()) {
    auto offset = offsets[expandedDim];
    auto length = sizes[expandedDim];

    // Compute the offset and slice length of the collapsed dimension as
    //   newOffset =
    for (int e = indices.size() + expandedDim++; expandedDim < e;
         ++expandedDim) {
      // Bail on extracting from the middle of an expanded dim.
      if (!isConstantIntValue(offsets[expandedDim], 0))
        return failure();

      auto expandedLength = sizes[expandedDim];
      OpFoldResult srcSize =
          tensor::getMixedSize(b, loc, expandShapeOp.getResult(), expandedDim);
      std::optional<int64_t> constantLength =
          getConstantIntValue(expandedLength);
      std::optional<int64_t> constantSize = getConstantIntValue(srcSize);

      // Bail when slicing an expanded dim other than the outer most.
      // TODO: Check for the single dynamic dim per reassociation.
      if (!constantLength || !constantSize || *constantLength != *constantSize)
        return failure();

      offset = mul(offset, expandedLength);
      length = mul(length, expandedLength);
    }

    newOffsets.push_back(offset);
    newLengths.push_back(length);

    // Only unit stride supported.
    newStrides.push_back(b.getIndexAttr(1));
  }

  // The shape of the result can be obtained from the sizes passed in.
  SmallVector<Value> dynDims;
  SmallVector<int64_t> shape;
  dispatchIndexOpFoldResults(sizes, dynDims, shape);
  RankedTensorType resultType = RankedTensorType::get(
      shape, expandShapeOp.getResultType().getElementType());

  // Create a new SliceOp and ExpandShapeOp.
  Value newSliceOp = b.create<tensor::ExtractSliceOp>(
      loc, expandShapeOp.getSrc(), newOffsets, newLengths, newStrides);
  auto newExpandShapeOp = b.create<tensor::ExpandShapeOp>(
      loc, resultType, newSliceOp, expandShapeOp.getReassociationIndices());

  return TilingResult{{newExpandShapeOp}, {newExpandShapeOp.getResult()}};
}

std::optional<scf::SCFFuseProducerOfSliceResult>
fuseExpandShapeThroughSlice(RewriterBase &rewriter,
                            tensor::ExpandShapeOp expandOp,
                            tensor::ExtractSliceOp sliceOp) {
  if (!sliceOp.hasUnitStride())
    return std::nullopt;

  auto expandShapeOp =
      sliceOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
  if (!expandShapeOp)
    return std::nullopt;

  rewriter.setInsertionPoint(sliceOp);
  FailureOr<TilingResult> tilingResult = bubbleUpExpandShapeSlice(
      rewriter, expandShapeOp, sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes());
  if (failed(tilingResult))
    return std::nullopt;

  rewriter.replaceAllUsesWith(sliceOp, tilingResult->tiledValues[0]);
  rewriter.eraseOp(sliceOp);
  return scf::SCFFuseProducerOfSliceResult{expandOp->getOpResults()[0],
                                           tilingResult->tiledValues[0],
                                           tilingResult->tiledOps};
}

/// Return the untiled producer whose slice is used in a tiled consumer. The
/// method traverses the tile loop nest (`loops`) if needed, and returns the
/// `iter_args` of the outer most that is encountered. Traversing the iter_args
/// indicates that this is a destination operand of the consumer. If there was
/// no loop traversal needed, the second value of the returned tuple is empty.
static std::tuple<OpResult, std::optional<OpOperand *>>
getUntiledProducerFromSliceSource(OpOperand *source,
                                  ArrayRef<LoopLikeOpInterface> loops) {
  std::optional<OpOperand *> destinationIterArg;
  auto loopIt = loops.rbegin();
  while (auto iterArg = dyn_cast<BlockArgument>(source->get())) {
    auto loop = *loopIt;
    if (iterArg.getOwner()->getParentOp() != loop)
      break;
    source = loop.getTiedLoopInit(iterArg);
    loopIt++;
  }
  if (loopIt == loops.rend())
    destinationIterArg = source;
  return {dyn_cast<OpResult>(source->get()), destinationIterArg};
}

/// Clone of the upstream implementation of tile consumer and fuse producer
/// greedily. This includes a pattern for fusing through expand_shape ops.
FailureOr<scf::SCFTileAndFuseResult>
tileConsumerAndFuseProducersWithReshapeFusion(
    RewriterBase &rewriter, TilingInterface consumer,
    const scf::SCFTileAndFuseOptions &options) {
  // This transformation is only valid for ops that return values (i.e. not
  // valid to use with operations that have memref operands).
  if (!consumer->getNumResults()) {
    return rewriter.notifyMatchFailure(
        consumer, "invalid pattern for op with no results");
  }

  // 1. First tile the consumer.
  SetVector<Operation *> fusedProducers, tiledAndFusedOps;
  llvm::SmallDenseMap<Value, size_t> origProducerToLoopResultNum;

  FailureOr<scf::SCFTilingResult> tilingResult =
      tileUsingSCF(rewriter, consumer, options.tilingOptions);

  if (failed(tilingResult))
    return rewriter.notifyMatchFailure(consumer, "failed to tile consumer");
  for (auto *tiledOp : tilingResult->tiledOps)
    tiledAndFusedOps.insert(tiledOp);

  // If there are no loops generated, fusion is immaterial.
  auto &loops = tilingResult->loops;
  if (loops.empty()) {
    DenseMap<Value, Value> replacements;
    for (auto [origVal, replacement] :
         llvm::zip_equal(consumer->getResults(), tilingResult->replacements)) {
      replacements[origVal] = replacement;
    }
    return scf::SCFTileAndFuseResult{fusedProducers, tiledAndFusedOps, loops,
                                     replacements};
  }

  // To keep track of replacements for now just record the map from the original
  // untiled value to the result number of the for loop. Since the loop gets
  // potentially replaced during fusion, keeping the value directly wont work.
  DenseMap<Value, size_t> origValToResultNumber;
  for (auto [index, result] : llvm::enumerate(consumer->getResults())) {
    origValToResultNumber[result] = index;
  }

  // 2. Typically, the operands of the tiled operation are slices of the
  //    operands of the untiled operation. These are expressed in IR using
  //    `tensor.extract_slice` operations with source being the operands of the
  //    untiled operation. Create a worklist of these `tensor.extract_slice`
  //    operations. If the producers of the source of the `tensor.extract_slice`
  //    can be tiled such that the tiled value is generated in-place, that
  //    effectively tiles + fuses the operations.
  auto addCandidateSlices = [](Operation *fusedOp,
                               std::deque<tensor::ExtractSliceOp> &candidates) {
    for (Value operand : fusedOp->getOperands())
      if (auto sliceOp = operand.getDefiningOp<tensor::ExtractSliceOp>())
        candidates.push_back(sliceOp);
  };

  std::deque<tensor::ExtractSliceOp> candidates;
  addCandidateSlices(tiledAndFusedOps.back(), candidates);
  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    // Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();

    // Find the original producer of the slice.
    auto [fusableProducer, destinationInitArg] =
        getUntiledProducerFromSliceSource(&candidateSliceOp.getSourceMutable(),
                                          loops);
    if (!fusableProducer)
      continue;

    auto [fuseSlice, yieldReplacement] = options.fusionControlFn(
        candidateSliceOp, fusableProducer, destinationInitArg.has_value());
    if (!fuseSlice)
      continue;

    std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult = std::nullopt;
    if (auto expandShape =
            dyn_cast<tensor::ExpandShapeOp>(fusableProducer.getOwner())) {
      // TODO: Reshape fusion through destinations.
      if (destinationInitArg.has_value()) {
        continue;
      }
      fusedResult =
          fuseExpandShapeThroughSlice(rewriter, expandShape, candidateSliceOp);
    } else {
      // The operands of the fused producer might themselved be slices of
      // values produced by operations that implement the `TilingInterface`.
      // Add these operations to the worklist.
      fusedResult =
          scf::tileAndFuseProducerOfSlice(rewriter, candidateSliceOp, loops);
    }
    if (!fusedResult)
      continue;

    if (yieldReplacement) {
      // Reconstruct and yield all opResult of fusableProducerOp by default. The
      // caller can specific which one to yield by designating optional argument
      // named `yieldResultNumber` of `yieldReplacementForFusedProducer`.
      Operation *fusableProducerOp = fusableProducer.getOwner();
      if (failed(yieldReplacementForFusedProducer(
              rewriter, candidateSliceOp, fusedResult.value(), loops))) {
        return rewriter.notifyMatchFailure(
            fusableProducerOp, "failed to replacement value for this "
                               "operation from within the tiled loop");
      }
      for (auto [index, result] :
           llvm::enumerate(fusableProducerOp->getResults())) {
        origValToResultNumber[result] = loops.front()->getNumResults() -
                                        fusableProducerOp->getNumResults() +
                                        index;
      }
    }

    if (Operation *tiledAndFusedOp =
            fusedResult->tiledAndFusedProducer.getDefiningOp()) {
      fusedProducers.insert(fusedResult->origProducer.getDefiningOp());
      tiledAndFusedOps.insert(tiledAndFusedOp);
      addCandidateSlices(tiledAndFusedOp, candidates);
    }
  }

  DenseMap<Value, Value> replacements;
  for (auto [origVal, resultNumber] : origValToResultNumber) {
    replacements[origVal] = loops.front()->getResult(resultNumber);
  }

  return scf::SCFTileAndFuseResult{fusedProducers, tiledAndFusedOps, loops,
                                   replacements};
}

} // namespace mlir::iree_compiler
