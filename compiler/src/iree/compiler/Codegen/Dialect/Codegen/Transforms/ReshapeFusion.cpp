// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace {

/// Check if an InnerTiledOp can be expanded by propagating a reshape through
/// it. The main real condition is that the inner dimensions of the op are not
/// expanded. Otherwise, we artificially restrict to single result inner_tiled
/// ops for now.
static LogicalResult
canExpandInnerTiledOp(InnerTiledOp op, OpOperand *fusedOperand,
                      ArrayRef<ReassociationIndices> reassociation) {
  // Only single result inner_tiled ops are tested or used anywhere, so restrict
  // to single result for now.
  if (op->getNumResults() != 1)
    return failure();

  // Only outer dims can be expanded because inner dims depend on the `kind`
  // attribute's implementation.
  int64_t outerRank =
      op.getIndexingMapsArray()[fusedOperand->getOperandNumber()]
          .getNumResults();
  if (llvm::any_of(reassociation.drop_front(outerRank),
                   [](ArrayRef<int64_t> group) { return group.size() != 1; })) {
    return failure();
  }
  return success();
}

/// Expand an InnerTiledOp by propagating a reshape through it.
/// `fusedOperand` is the operand connected to the reshape.
/// `reassociation` describes how the collapsed dims map to expanded dims.
/// `expandedShape` is the full expanded shape (outer + inner dims).
/// `expandedValue` is the expanded value to replace the fused operand.
/// `outputReassociations` will be cleared and filled with the reassociation
/// indices for each output, to be used for collapsing the result back to its
/// original shape.
/// The outer dimensions of the InnerTiledOp are expected to not be expanded,
/// which is enforced by the canExpandInnerTiledOp precondition.
static InnerTiledOp expandInnerTiledOp(
    InnerTiledOp op, OpOperand *fusedOperand,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<OpFoldResult> expandedShape, Value expandedValue,
    SmallVectorImpl<SmallVector<ReassociationIndices>> &outputReassociations,
    PatternRewriter &rewriter) {
  assert(reassociation.size() ==
             cast<RankedTensorType>(fusedOperand->get().getType()).getRank() &&
         "expected reassociation rank to match fused operand rank");

  // Build mapping: iterDim -> list of (expandedIterDim, size).
  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
  AffineMap fusedMap = indexingMaps[fusedOperand->getOperandNumber()];
  int64_t numIterDims = fusedMap.getNumDims();
  SmallVector<SmallVector<std::pair<int64_t, OpFoldResult>>> iterDimExpansion(
      numIterDims);
  int64_t expandedDimCounter = 0;
  for (auto [resultIdx, expr] : llvm::enumerate(fusedMap.getResults())) {
    int64_t iterDim = cast<AffineDimExpr>(expr).getPosition();
    for (int64_t expandedOperandIdx : reassociation[resultIdx]) {
      iterDimExpansion[iterDim].push_back(
          {expandedDimCounter++, expandedShape[expandedOperandIdx]});
    }
  }
  // Iteration dims outside the fused map's results are independent from the
  // expansion, but update their dim position to account for earlier expanded
  // dims. Get iteration domain to query sizes of dims not in the fused operand.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);
  for (int64_t i = 0; i < numIterDims; ++i) {
    if (iterDimExpansion[i].empty())
      iterDimExpansion[i].push_back(
          {expandedDimCounter++, iterationDomain[i].size});
  }

  SmallVector<AffineMap> newIndexingMaps;
  SmallVector<Value> newOperands;
  outputReassociations.clear();
  Location loc = op.getLoc();
  for (OpOperand &operand : op->getOpOperands()) {
    AffineMap origMap = indexingMaps[operand.getOperandNumber()];
    auto operandType = cast<RankedTensorType>(operand.get().getType());
    int64_t operandOuterRank = origMap.getNumResults();
    int64_t innerRank = operandType.getRank() - operandOuterRank;
    SmallVector<AffineExpr> newMapResults;
    SmallVector<ReassociationIndices> operandReassoc;
    SmallVector<OpFoldResult> expandedOperandSizes;
    int64_t dimCounter = 0;
    for (AffineExpr expr : origMap.getResults()) {
      int64_t iterDim = cast<AffineDimExpr>(expr).getPosition();
      ReassociationIndices group;
      for (auto [expandedDim, size] : iterDimExpansion[iterDim]) {
        newMapResults.push_back(getAffineDimExpr(expandedDim, op.getContext()));
        group.push_back(dimCounter++);
        expandedOperandSizes.push_back(size);
      }
      operandReassoc.push_back(group);
    }
    // Inner dims are never expanded.
    for (int64_t i = 0; i < innerRank; ++i) {
      operandReassoc.push_back({dimCounter++});
      expandedOperandSizes.push_back(tensor::getMixedSize(
          rewriter, loc, operand.get(), operandOuterRank + i));
    }
    newIndexingMaps.push_back(
        AffineMap::get(expandedDimCounter, 0, newMapResults, op.getContext()));

    // Store output reassociations for later use.
    if (operand.getOperandNumber() >= op.getNumInputs()) {
      outputReassociations.push_back(operandReassoc);
    }

    if (&operand == fusedOperand) {
      newOperands.push_back(expandedValue);
      continue;
    }

    if (llvm::all_of(operandReassoc, [](ArrayRef<int64_t> group) {
          return group.size() == 1;
        })) {
      newOperands.push_back(operand.get());
      continue;
    }

    SmallVector<int64_t> staticShape;
    std::tie(staticShape, std::ignore) =
        decomposeMixedValues(expandedOperandSizes);
    auto expandedType =
        RankedTensorType::get(staticShape, operandType.getElementType());
    newOperands.push_back(tensor::ExpandShapeOp::create(
        rewriter, loc, expandedType, operand.get(), operandReassoc,
        expandedOperandSizes));
  }

  // Expand iterator types.
  SmallVector<utils::IteratorType> newIterTypes;
  for (auto [idx, iterType] : llvm::enumerate(op.getIteratorTypesArray())) {
    newIterTypes.append(iterDimExpansion[idx].size(), iterType);
  }

  int64_t numInputs = op.getNumInputs();
  SmallVector<Value> newInputs(newOperands.begin(),
                               newOperands.begin() + numInputs);
  SmallVector<Value> newOutputs(newOperands.begin() + numInputs,
                                newOperands.end());

  // Permutations are unchanged, since they are for inner dims, but we need to
  // convert from ArrayAttr to SmallVector<SmallVector<int64_t>>.
  std::optional<SmallVector<SmallVector<int64_t>>> newPermutations;
  if (auto permAttr = op.getPermutations()) {
    newPermutations = llvm::map_to_vector(
        permAttr->getAsRange<DenseI64ArrayAttr>(), [](DenseI64ArrayAttr perm) {
          return SmallVector<int64_t>(perm.asArrayRef());
        });
  }

  return InnerTiledOp::create(rewriter, loc, newInputs, newOutputs,
                              newIndexingMaps, newIterTypes, op.getKind(),
                              op.getSemantics(), newPermutations);
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// Pattern to propagate a tensor::CollapseShapeOp through a consumer
/// InnerTiledOp. The collapsed dimensions must not include any inner dimensions
/// of the InnerTiledOp.
///
/// Example:
///   %collapsed = tensor.collapse_shape %src [[0, 1], ...]
///   %result = inner_tiled ins(%collapsed, ...) outs(%out)
/// =>
///   %expanded_out = tensor.expand_shape %out [[0, 1], ...]
///   %result = inner_tiled ins(%src, ...) outs(%expanded_out)
///   %collapsed_result = tensor.collapse_shape %result [[0, 1], ...]
struct FoldProducerCollapseShapeWithInnerTiled
    : public OpRewritePattern<tensor::CollapseShapeOp> {
  FoldProducerCollapseShapeWithInnerTiled(MLIRContext *context,
                                          linalg::ControlFusionFn controlFn,
                                          PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::CollapseShapeOp>(context, benefit),
        controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    if (!collapseOp->hasOneUse()) {
      return failure();
    }
    OpOperand &use = *collapseOp->use_begin();
    auto innerTiledOp = dyn_cast<InnerTiledOp>(use.getOwner());
    if (!innerTiledOp || !controlFn(&use)) {
      return failure();
    }
    if (failed(canExpandInnerTiledOp(innerTiledOp, &use,
                                     collapseOp.getReassociationIndices()))) {
      return failure();
    }

    SmallVector<OpFoldResult> expandedShape = tensor::getMixedSizes(
        rewriter, collapseOp.getLoc(), collapseOp.getSrc());
    SmallVector<SmallVector<ReassociationIndices>> outputReassociations;
    InnerTiledOp expandedOp = expandInnerTiledOp(
        innerTiledOp, &use, collapseOp.getReassociationIndices(), expandedShape,
        collapseOp.getSrc(), outputReassociations, rewriter);

    SmallVector<Value> results;
    for (auto [idx, result] : llvm::enumerate(expandedOp.getResults())) {
      auto resultType =
          cast<RankedTensorType>(innerTiledOp.getResultTypes()[idx]);
      results.push_back(tensor::CollapseShapeOp::create(
          rewriter, innerTiledOp.getLoc(), resultType, result,
          outputReassociations[idx]));
    }
    rewriter.replaceOp(innerTiledOp, results);
    return success();
  }

private:
  linalg::ControlFusionFn controlFn;
};

/// Pattern to propagate a tensor::ExpandShapeOp consumer back through an
/// InnerTiledOp producer. The expanded dimensions must not include any inner
/// dimensions of the InnerTiledOp.
///
/// Example:
///   %result = inner_tiled ins(%lhs, ...) outs(%out)
///   %expanded = tensor.expand_shape %result [[0, 1], ...]
/// =>
///   %expanded_lhs = tensor.expand_shape %lhs [[0, 1], ...]
///   %expanded_out = tensor.expand_shape %out [[0, 1], ...]
///   %result = inner_tiled ins(%expanded_lhs, ...) outs(%expanded_out)
struct FoldConsumerExpandShapeWithInnerTiled
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  FoldConsumerExpandShapeWithInnerTiled(MLIRContext *context,
                                        linalg::ControlFusionFn controlFn,
                                        PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::ExpandShapeOp>(context, benefit),
        controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    auto producerResult = dyn_cast<OpResult>(expandOp.getSrc());
    if (!producerResult) {
      return failure();
    }
    auto innerTiledOp = dyn_cast<InnerTiledOp>(producerResult.getOwner());
    if (!innerTiledOp || !controlFn(&expandOp.getSrcMutable())) {
      return failure();
    }

    int64_t resultIdx = producerResult.getResultNumber();
    OpOperand *outputOperand = innerTiledOp.getDpsInitOperand(resultIdx);
    if (failed(canExpandInnerTiledOp(innerTiledOp, outputOperand,
                                     expandOp.getReassociationIndices()))) {
      return failure();
    }

    // The DPS init will be expanded in the same way as the result, so insert
    // the expand_shape on the init first in order to reuse the
    // expandInnerTiledOp transformation utility.
    SmallVector<OpFoldResult> expandedShape = expandOp.getMixedOutputShape();
    SmallVector<int64_t> staticShape;
    std::tie(staticShape, std::ignore) = decomposeMixedValues(expandedShape);
    auto sourceType = cast<RankedTensorType>(outputOperand->get().getType());
    auto expandedType =
        RankedTensorType::get(staticShape, sourceType.getElementType());
    auto expandedInit = tensor::ExpandShapeOp::create(
        rewriter, expandOp.getLoc(), expandedType, outputOperand->get(),
        expandOp.getReassociationIndices(), expandedShape);

    SmallVector<SmallVector<ReassociationIndices>> outputReassociations;
    InnerTiledOp expandedOp = expandInnerTiledOp(
        innerTiledOp, outputOperand, expandOp.getReassociationIndices(),
        expandedShape, expandedInit, outputReassociations, rewriter);
    rewriter.replaceOp(expandOp, expandedOp.getResult(resultIdx));
    return success();
  }

private:
  linalg::ControlFusionFn controlFn;
};

} // namespace

//===----------------------------------------------------------------------===//
// Populate Functions
//===----------------------------------------------------------------------===//

void populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFoldingReshapes) {
  patterns.add<FoldProducerCollapseShapeWithInnerTiled,
               FoldConsumerExpandShapeWithInnerTiled>(patterns.getContext(),
                                                      controlFoldingReshapes);
}

} // namespace mlir::iree_compiler::IREE::Codegen
