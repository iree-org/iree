// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// The content of this file is adapted from linalg's ElemenwiseOpFusion.cpp and
// modified to work with LinalgExt ops, specifically `LinalgExt::AttentionOp`.

#include <optional>
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

namespace {

/// Information needed to expand an operation to fold the reshape with
/// it.
class ExpansionInfo {
public:
  // Computes the mapping from original dimensions of the op to the dimensions
  // of the expanded op given the `indexingMap` of the fused operand/result of
  // the op, the `reassocationMaps` of the reshape op and the shape of
  // the expanded op.
  template <typename OpTy>
  LogicalResult compute(OpTy op, OpOperand *fusableOpOperand,
                        ArrayRef<AffineMap> reassociationMaps,
                        ArrayRef<int64_t> expandedShape,
                        ArrayRef<int64_t> collapsedShape,
                        PatternRewriter &rewriter);
  unsigned getOrigOpNumDims() const { return reassociation.size(); }
  unsigned getExpandedOpNumDims() const { return expandedOpNumDims; }
  ReassociationIndicesRef getExpandedDims(unsigned i) const {
    return reassociation[i];
  }
  ArrayRef<int64_t> getExpandedShapeOfDim(unsigned i) const {
    return expandedShapeMap[i];
  }
  ArrayRef<int64_t> getOriginalShape() const { return originalLoopExtent; }

private:
  /// Reassociation from the dimensions in the original operation to the
  /// dimension of the expanded operation.
  SmallVector<ReassociationIndices> reassociation;
  /// Mapping from extent of loops in the original operation, to the extent of
  /// loops in the expanded operation.
  SmallVector<SmallVector<int64_t>> expandedShapeMap;
  /// Extent of the loop in the original operation.
  SmallVector<int64_t> originalLoopExtent;
  unsigned expandedOpNumDims;
};
} // namespace

template <typename OpTy>
LogicalResult ExpansionInfo::compute(OpTy op, OpOperand *fusableOpOperand,
                                     ArrayRef<AffineMap> reassociationMaps,
                                     ArrayRef<int64_t> expandedShape,
                                     ArrayRef<int64_t> collapsedShape,
                                     PatternRewriter &rewriter) {
  if (reassociationMaps.empty())
    return failure();
  AffineMap fusedIndexMap = op.getMatchingIndexingMap(fusableOpOperand);
  SmallVector<int64_t, 4> originalLoopRange = op.getStaticLoopRanges();
  originalLoopExtent.assign(originalLoopRange.begin(), originalLoopRange.end());

  reassociation.clear();
  expandedShapeMap.clear();
  // Compute the number of dimension in the expanded op that correspond to each
  // dimension of the original op.
  SmallVector<unsigned> numExpandedDims(fusedIndexMap.getNumDims(), 1);
  expandedShapeMap.resize(fusedIndexMap.getNumDims());
  for (const auto &resultExpr : llvm::enumerate(fusedIndexMap.getResults())) {
    unsigned pos = cast<AffineDimExpr>(resultExpr.value()).getPosition();
    AffineMap foldedDims = reassociationMaps[resultExpr.index()];
    numExpandedDims[pos] = foldedDims.getNumResults();
    ArrayRef<int64_t> shape =
        expandedShape.slice(foldedDims.getDimPosition(0), numExpandedDims[pos]);
    expandedShapeMap[pos].assign(shape.begin(), shape.end());
  }
  // The remaining dimensions remain the same.
  for (unsigned i : llvm::seq<unsigned>(0, fusedIndexMap.getNumDims()))
    if (expandedShapeMap[i].empty())
      expandedShapeMap[i] = {originalLoopExtent[i]};

  // Compute reassociation map from the original op to the expanded op.
  unsigned sum = 0;
  reassociation.reserve(fusedIndexMap.getNumDims());
  for (const auto &numFoldedDim : llvm::enumerate(numExpandedDims)) {
    auto seq = llvm::seq<int64_t>(sum, sum + numFoldedDim.value());
    reassociation.emplace_back(seq.begin(), seq.end());
    sum += numFoldedDim.value();
  }
  expandedOpNumDims = sum;
  return success();
}

static AffineMap
getIndexingMapInExpandedOp(OpBuilder &builder, AffineMap indexingMap,
                           const ExpansionInfo &expansionInfo) {
  SmallVector<AffineExpr> newExprs;
  for (AffineExpr expr : indexingMap.getResults()) {
    unsigned pos = cast<AffineDimExpr>(expr).getPosition();
    auto expandedExprs = llvm::to_vector_of<AffineExpr, 6>(
        llvm::map_range(expansionInfo.getExpandedDims(pos), [&](int64_t v) {
          return builder.getAffineDimExpr(static_cast<unsigned>(v));
        }));
    newExprs.append(expandedExprs.begin(), expandedExprs.end());
  }
  return AffineMap::get(expansionInfo.getExpandedOpNumDims(),
                        indexingMap.getNumSymbols(), newExprs,
                        builder.getContext());
}

static RankedTensorType getExpandedType(RankedTensorType originalType,
                                        AffineMap indexingMap,
                                        const ExpansionInfo &expansionInfo) {
  SmallVector<int64_t> expandedShape;
  for (AffineExpr expr : indexingMap.getResults()) {
    unsigned dim = cast<AffineDimExpr>(expr).getPosition();
    auto dimExpansion = expansionInfo.getExpandedShapeOfDim(dim);
    expandedShape.append(dimExpansion.begin(), dimExpansion.end());
  }
  return RankedTensorType::get(expandedShape, originalType.getElementType());
}

static SmallVector<ReassociationIndices>
getReassociationForExpansion(AffineMap indexingMap,
                             const ExpansionInfo &expansionInfo) {
  SmallVector<ReassociationIndices> reassociation;
  unsigned numReshapeDims = 0;
  for (AffineExpr expr : indexingMap.getResults()) {
    unsigned dim = cast<AffineDimExpr>(expr).getPosition();
    auto numExpandedDims = expansionInfo.getExpandedDims(dim).size();
    SmallVector<int64_t, 2> indices = llvm::to_vector<2>(
        llvm::seq<int64_t>(numReshapeDims, numReshapeDims + numExpandedDims));
    reassociation.emplace_back(std::move(indices));
    numReshapeDims += numExpandedDims;
  }
  return reassociation;
}

template <typename OpTy>
static bool isFusableWithReshapeByDimExpansion(OpTy op,
                                               OpOperand *fusableOpOperand) {
  // Is fusable only if:
  // - All the indexing maps for operands and results are projected
  //   permutations.
  // - The fused tensor is not a scalar.
  // - All the loops for the reshaped operand are parallel loops.
  SmallVector<utils::IteratorType> iteratorTypes = op.getLoopIteratorTypes();
  AffineMap operandMap = op.getMatchingIndexingMap(fusableOpOperand);
  return op.hasPureTensorSemantics() &&
         llvm::all_of(
             op.getIndexingMapsArray(),
             [](AffineMap map) { return map.isProjectedPermutation(); }) &&
         operandMap.getNumResults() > 0;
}

static std::optional<SmallVector<Value>> fuseAttentionWithReshapeByExpansion(
    AttentionOp attentionOp, Operation *reshapeOp, OpOperand *fusableOpOperand,
    PatternRewriter &rewriter) {
  assert(isFusableWithReshapeByDimExpansion(attentionOp, fusableOpOperand) &&
         "preconditions for fuse operation failed");

  Location loc = attentionOp.getLoc();
  // Check if reshape is expanding or collapsing.
  auto expandingReshapeOp = dyn_cast<tensor::ExpandShapeOp>(*reshapeOp);
  auto collapsingReshapeOp = dyn_cast<tensor::CollapseShapeOp>(*reshapeOp);
  bool isExpanding = (expandingReshapeOp != nullptr);
  RankedTensorType expandedType = isExpanding
                                      ? expandingReshapeOp.getResultType()
                                      : collapsingReshapeOp.getSrcType();
  RankedTensorType collapsedType = isExpanding
                                       ? expandingReshapeOp.getSrcType()
                                       : collapsingReshapeOp.getResultType();

  ExpansionInfo expansionInfo;
  if (failed(expansionInfo.compute(
          attentionOp, fusableOpOperand,
          isExpanding ? expandingReshapeOp.getReassociationMaps()
                      : collapsingReshapeOp.getReassociationMaps(),
          expandedType.getShape(), collapsedType.getShape(), rewriter)))
    return std::nullopt;
  auto expandedOpIndexingMaps = llvm::to_vector_of<AffineMap, 6>(
      llvm::map_range(attentionOp.getIndexingMapsArray(), [&](AffineMap m) {
        return getIndexingMapInExpandedOp(rewriter, m, expansionInfo);
      }));

  // Set insertion point to the attention op.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(attentionOp);

  SmallVector<Value> expandedOpOperands;
  expandedOpOperands.reserve(attentionOp.getNumDpsInputs());
  for (OpOperand *opOperand : attentionOp.getDpsInputOperands()) {
    if (opOperand == fusableOpOperand) {
      expandedOpOperands.push_back(isExpanding ? expandingReshapeOp.getSrc()
                                               : collapsingReshapeOp.getSrc());
      continue;
    }
    if (auto opOperandType =
            dyn_cast<RankedTensorType>(opOperand->get().getType())) {
      AffineMap indexingMap = attentionOp.getMatchingIndexingMap(opOperand);
      RankedTensorType expandedOperandType =
          getExpandedType(opOperandType, indexingMap, expansionInfo);
      if (expandedOperandType != opOperand->get().getType()) {
        // Reshape the operand to get the right type.
        SmallVector<ReassociationIndices> reassociation =
            getReassociationForExpansion(indexingMap, expansionInfo);
        if (failed(reshapeLikeShapesAreCompatible(
                [&](const Twine &msg) {
                  return rewriter.notifyMatchFailure(attentionOp, msg);
                },
                opOperandType.getShape(), expandedOperandType.getShape(),
                reassociation,
                /*isExpandingReshape=*/true)))
          return std::nullopt;
        expandedOpOperands.push_back(rewriter.create<tensor::ExpandShapeOp>(
            loc, expandedOperandType, opOperand->get(), reassociation));
        continue;
      }
    }
    expandedOpOperands.push_back(opOperand->get());
  }

  SmallVector<Value> outputs;
  for (OpOperand &opOperand : attentionOp.getDpsInitsMutable()) {
    AffineMap indexingMap = attentionOp.getMatchingIndexingMap(&opOperand);
    auto opOperandType = cast<RankedTensorType>(opOperand.get().getType());
    RankedTensorType expandedOutputType =
        getExpandedType(opOperandType, indexingMap, expansionInfo);
    if (expandedOutputType != opOperand.get().getType()) {
      SmallVector<ReassociationIndices> reassociation =
          getReassociationForExpansion(indexingMap, expansionInfo);
      if (failed(reshapeLikeShapesAreCompatible(
              [&](const Twine &msg) {
                return rewriter.notifyMatchFailure(attentionOp, msg);
              },
              opOperandType.getShape(), expandedOutputType.getShape(),
              reassociation,
              /*isExpandingReshape=*/true)))
        return std::nullopt;
      outputs.push_back(rewriter.create<tensor::ExpandShapeOp>(
          loc, expandedOutputType, opOperand.get(), reassociation));
    } else {
      outputs.push_back(opOperand.get());
    }
  }

  Value maskOperand;
  if (expandedOpOperands.size() > 4) {
    maskOperand = expandedOpOperands[4];
  }

  // Create a new `AttentionOp` that has the computed operands/indexing maps.
  TypeRange resultTypes = ValueRange(outputs).getTypes();
  auto fusedOp = rewriter.create<AttentionOp>(
      attentionOp.getLoc(), resultTypes, expandedOpOperands[0],
      expandedOpOperands[1], expandedOpOperands[2], expandedOpOperands[3],
      outputs, rewriter.getAffineMapArrayAttr(expandedOpIndexingMaps),
      maskOperand);

  // Reshape the result values to their original shape if this is a collapsing
  // reshape folded into its consumer.
  SmallVector<Value> resultVals;
  for (OpResult opResult : attentionOp->getOpResults()) {
    int64_t resultNumber = opResult.getResultNumber();
    if (resultTypes[resultNumber] != opResult.getType()) {
      SmallVector<ReassociationIndices> reassociation =
          getReassociationForExpansion(
              attentionOp.getIndexingMapsForResults()[resultNumber],
              expansionInfo);
      resultVals.push_back(rewriter.create<tensor::CollapseShapeOp>(
          attentionOp.getLoc(), opResult.getType(),
          fusedOp->getResult(resultNumber), reassociation));
    } else {
      resultVals.push_back(fusedOp->getResult(resultNumber));
    }
  }
  // Assuming a single result.
  return resultVals;
}

namespace {

// Fold attention with its consumer expand_shape op.
struct FoldAttentionWithConsumerReshapeByExpansion
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  FoldAttentionWithConsumerReshapeByExpansion(
      MLIRContext *context, linalg::ControlFusionFn foldReshapes,
      PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::ExpandShapeOp>(context, benefit),
        controlFoldingReshapes(std::move(foldReshapes)) {}

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto producerResult = dyn_cast<OpResult>(reshapeOp.getSrc());
    if (!producerResult) {
      return rewriter.notifyMatchFailure(reshapeOp,
                                         "source not produced by an operation");
    }

    auto producer = dyn_cast<AttentionOp>(producerResult.getOwner());
    if (!producer) {
      return rewriter.notifyMatchFailure(reshapeOp,
                                         "producer is not an attention op");
    }

    if (!controlFoldingReshapes(&reshapeOp.getSrcMutable())) {
      return rewriter.notifyMatchFailure(reshapeOp,
                                         "fusion blocked by control function");
    }

    // Note: expand_shape can always be fused with attention, it is not checked
    // as a precondition. It is asserted in `fuseWithReshapeByExpansion`.
    std::optional<SmallVector<Value>> replacementValues =
        fuseAttentionWithReshapeByExpansion(
            producer, reshapeOp,
            producer.getDpsInitOperand(producerResult.getResultNumber()),
            rewriter);
    if (!replacementValues) {
      return rewriter.notifyMatchFailure(reshapeOp,
                                         "fusion by expansion failed");
    }

    Value reshapeReplacement =
        (*replacementValues)[cast<OpResult>(reshapeOp.getSrc())
                                 .getResultNumber()];
    if (auto collapseOp =
            reshapeReplacement.getDefiningOp<tensor::CollapseShapeOp>()) {
      reshapeReplacement = collapseOp.getSrc();
    }
    rewriter.replaceOp(reshapeOp, reshapeReplacement);
    rewriter.replaceOp(producer, *replacementValues);
    return success();
  }
  linalg::ControlFusionFn controlFoldingReshapes;
};

// Fold a collapse_shape op with its consumer attention op.
// class FoldWith

struct FoldAttentionWithProducerReshapeByExpansion final
    : public OpRewritePattern<AttentionOp> {
  FoldAttentionWithProducerReshapeByExpansion(
      MLIRContext *context, linalg::ControlFusionFn controlFoldingReshapes,
      PatternBenefit benefit = 1)
      : OpRewritePattern<AttentionOp>(context, benefit),
        controlFoldingReshapes(std::move(controlFoldingReshapes)) {}

  LogicalResult matchAndRewrite(AttentionOp attentionOp,
                                PatternRewriter &rewriter) const override {
    for (OpOperand *opOperand : attentionOp.getDpsInputOperands()) {
      tensor::CollapseShapeOp reshapeOp =
          opOperand->get().getDefiningOp<tensor::CollapseShapeOp>();
      if (!reshapeOp)
        continue;
      // Fold only if
      // - The tensor reshape op is folding.
      // - All constraints of fusing with reshape by expansion are met.
      if (!isFusableWithReshapeByDimExpansion(attentionOp, opOperand) ||
          (!controlFoldingReshapes(opOperand)))
        continue;

      std::optional<SmallVector<Value>> replacementValues =
          fuseAttentionWithReshapeByExpansion(attentionOp, reshapeOp, opOperand,
                                              rewriter);
      if (!replacementValues)
        return failure();
      rewriter.replaceOp(attentionOp, *replacementValues);
      return success();
    }
    return failure();
  }

  linalg::ControlFusionFn controlFoldingReshapes;
};

} // namespace

void populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFoldingReshapes) {
  patterns.add<FoldAttentionWithConsumerReshapeByExpansion>(
      patterns.getContext(), controlFoldingReshapes);
  patterns.add<FoldAttentionWithProducerReshapeByExpansion>(
      patterns.getContext(), controlFoldingReshapes);
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
