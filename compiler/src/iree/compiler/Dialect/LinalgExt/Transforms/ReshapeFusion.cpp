// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// The content of this file is adapted from linalg's ElemenwiseOpFusion.cpp and
// modified to work with LinalgExt ops, specifically `LinalgExt::AttentionOp`.

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

#include <cstdint>
#include <optional>

namespace mlir::iree_compiler::IREE::LinalgExt {

static bool
isIdentityReassoc(const SmallVector<ReassociationIndices> &indices) {
  for (auto &index : indices) {
    if (index.size() != 1)
      return false;
  }
  return true;
};

static SmallVector<ReassociationIndices>
computeReassocFromShapeMap(ArrayRef<SmallVector<int64_t>> shapeMap) {
  SmallVector<ReassociationIndices> reassoc;
  int64_t dimCount = 0;
  for (auto &shape : shapeMap) {
    reassoc.emplace_back(
        llvm::to_vector(llvm::seq<int64_t>(dimCount, dimCount + shape.size())));
    dimCount += shape.size();
  }
  return reassoc;
}

namespace {

/// Helper class that supports fusing reshapes with operands when not all of the
/// shape dims map to the iteration space.
struct ReshapeOperandInfo {
  static constexpr int64_t kNoMapping = -1;

  // Original shape of this operand.
  ArrayRef<int64_t> originalShape;

  // Similar to the results of the operand's `AffineMap` except `kNoMapping` if
  // that dim doesn't map to the iteration space. For example, the indexed
  // dimensions in a LinalgExt::ScatterOp.
  SmallVector<int64_t> operandToIterationSpace;
};

/// Information needed to expand an operation to fold the reshape with
/// it.
class ExpansionInfo {
public:
  // Computes the mapping from original dimensions of the op to the dimensions
  // of the expanded op given the `indexingMap` of the fused operand/result of
  // the op, the `reassocationMaps` of the reshape op and the shape of
  // the expanded op.
  LogicalResult compute(SmallVector<ReshapeOperandInfo> infos,
                        SmallVector<int64_t> loopRanges,
                        OpOperand *fusableOpOperand,
                        ArrayRef<ReassociationIndices> operandReassoc,
                        ArrayRef<int64_t> expandedShape);

  std::optional<Value> getOrCreateExpanded(Location loc, OpOperand *operand,
                                           RewriterBase &rewriter) {
    auto shapeMap = this->getShapeMap(operand);
    auto reassoc = computeReassocFromShapeMap(shapeMap);
    if (isIdentityReassoc(reassoc)) {
      return operand->get();
    }
    SmallVector<int64_t> flattenedArray;
    for (auto &shape : shapeMap) {
      flattenedArray.append(shape.begin(), shape.end());
    }
    auto oldType = cast<ShapedType>(operand->get().getType());
    auto newType =
        RankedTensorType::get(flattenedArray, oldType.getElementType());
    if (failed(reshapeLikeShapesAreCompatible(
            [&](const Twine &msg) {
              return rewriter.notifyMatchFailure(loc, msg);
            },
            oldType.getShape(), newType.getShape(), reassoc,
            /*isExpandingReshape=*/true))) {
      return {};
    }
    return rewriter.create<tensor::ExpandShapeOp>(loc, newType, operand->get(),
                                                  reassoc);
  };

  /// Get the shape map for the operand.
  SmallVector<SmallVector<int64_t>> getShapeMap(OpOperand *operand) const {
    auto info = reshapeInfos[operand->getOperandNumber()];
    SmallVector<SmallVector<int64_t>> shapeMap;
    for (auto [operandIdx, loopIdx] :
         llvm::enumerate(info.operandToIterationSpace)) {
      if (loopIdx == ReshapeOperandInfo::kNoMapping) {
        shapeMap.push_back(
            SmallVector<int64_t>{info.originalShape[operandIdx]});
      } else {
        shapeMap.push_back(loopShapeMap[loopIdx]);
      }
    }
    return shapeMap;
  }

  SmallVector<ReassociationIndices> getReassoc(OpOperand *operand) const {
    auto shapeMap = this->getShapeMap(operand);
    return computeReassocFromShapeMap(shapeMap);
  }

  unsigned getOrigNumLoops() const { return loopReassoc.size(); }
  unsigned getExpandedNumLoops() const { return expandedOpNumDims; }
  ReassociationIndicesRef getExpandedLoops(unsigned i) const {
    return loopReassoc[i];
  }
  ArrayRef<int64_t> getExpandedShapeOfLoop(unsigned i) const {
    return loopShapeMap[i];
  }

private:
  /// Extent of the iteration space in the original operation.
  SmallVector<int64_t> loopRanges;
  SmallVector<ReassociationIndices> loopReassoc;
  /// Mapping from extent of loops in the original operation, to the extent of
  /// loops in the expanded operation.
  SmallVector<SmallVector<int64_t>> loopShapeMap;
  unsigned expandedOpNumDims;
  /// Info about the reassociation and original shape for each operand.
  SmallVector<ReshapeOperandInfo> reshapeInfos;
};

class CollapsingInfo {
public:
  LogicalResult initialize(unsigned origNumLoops,
                           ArrayRef<ReassociationIndices> foldedIterationDims);

  /// Return mapping from collapsed loop domain to original loop domain.
  ArrayRef<ReassociationIndices> getCollapsedOpToOrigOpMapping() const {
    return collapsedOpToOrigOpIterationDim;
  }

  /// Return mapping from original loop domain to collapsed loop domain. The
  /// mapping is a pair. First value is the dimension in the collapsed loop that
  /// the original loop is mapped to. Second is the relative position in folded
  /// list of this domain. For example if the original loop domain is 3D, and
  /// the collapsed loop domain is folding all of it, i.e.
  ///
  /// ```
  /// collapsedOpToOrigOpMapping = [[0, 1, 2] [3, 4]]`
  /// ```
  ///
  /// then
  ///
  /// ```
  ///  origOpToCollapsedOpMapping[0] = {0, 0};
  ///  origOpToCollapsedOpMapping[1] = {0, 1};
  ///  origOpToCollapsedOpMapping[2] = {0, 2};
  ///  origOpToCollapsedOpMapping[3] = {1, 0};
  ///  origOpToCollapsedOpMapping[4] = {1, 1};
  /// ```
  ///
  ArrayRef<std::pair<int64_t, unsigned>> getOrigOpToCollapsedOpMapping() const {
    return origOpToCollapsedOpIterationDim;
  }

  /// Return the collapsed op iteration domain rank.
  unsigned getCollapsedOpIterationRank() const {
    return collapsedOpToOrigOpIterationDim.size();
  }

private:
  /// Map from the iteration domain index in collapsed op to the iteration
  /// domain indices in the original op.
  SmallVector<ReassociationIndices> collapsedOpToOrigOpIterationDim;

  /// Map from iteration domain index in the original op to the iteration domain
  /// index in the collapsed op.
  SmallVector<std::pair<int64_t, unsigned>> origOpToCollapsedOpIterationDim;
};

} // namespace

LogicalResult ExpansionInfo::compute(
    SmallVector<ReshapeOperandInfo> infos, SmallVector<int64_t> loopRanges,
    OpOperand *fusableOpOperand, ArrayRef<ReassociationIndices> operandReassoc,
    ArrayRef<int64_t> expandedShape) {
  if (operandReassoc.empty())
    return failure();

  int64_t operandNum = fusableOpOperand->getOperandNumber();
  ReshapeOperandInfo &fusionOperandInfo = infos[operandNum];
  this->loopShapeMap.clear();
  this->loopShapeMap.resize(loopRanges.size());
  for (auto [operandIdx, loopIdx] :
       llvm::enumerate(fusionOperandInfo.operandToIterationSpace)) {
    if (loopIdx == ReshapeOperandInfo::kNoMapping) {
      continue;
    }

    // Compute the shape map at element `loopIdx`
    ReassociationIndicesRef indices = operandReassoc[operandIdx];
    for (auto [dimIdx, shapeIdx] : llvm::enumerate(indices)) {
      this->loopShapeMap[loopIdx].push_back(expandedShape[shapeIdx]);
    }
  }

  // Fill in the remaining elements with `loopRanges`
  this->expandedOpNumDims = 0;
  for (const auto &[loopIdx, shapeMap] : llvm::enumerate(this->loopShapeMap)) {
    if (shapeMap.empty()) {
      this->loopShapeMap[loopIdx] = SmallVector<int64_t>{loopRanges[loopIdx]};
    }
    this->expandedOpNumDims += shapeMap.size();
  }

  if (llvm::all_of(this->loopShapeMap,
                   [&](auto vec) { return vec.size() == 1; })) {
    return failure();
  }
  this->loopReassoc = computeReassocFromShapeMap(this->loopShapeMap);
  this->reshapeInfos = std::move(infos);
  this->loopRanges = std::move(loopRanges);
  return success();
}

LogicalResult
CollapsingInfo::initialize(unsigned origNumLoops,
                           ArrayRef<ReassociationIndices> foldedIterationDims) {
  llvm::SmallDenseSet<int64_t, 4> processedDims;
  // Find all the dims that are folded.
  for (ReassociationIndicesRef foldedIterationDim : foldedIterationDims) {
    if (foldedIterationDim.empty())
      continue;
    // If the folded dims contain dims already folded, that's illegal
    // specification. Repetition within a list is also illegal.
    for (auto dim : foldedIterationDim) {
      if (dim >= origNumLoops)
        return failure();
      if (processedDims.count(dim))
        return failure();
      processedDims.insert(dim);
    }
    collapsedOpToOrigOpIterationDim.emplace_back(foldedIterationDim.begin(),
                                                 foldedIterationDim.end());
  }
  if (processedDims.size() > origNumLoops)
    return failure();

  // Add all the preserved dims of the original op as single
  // elements to `collapsedOpToOrigOpIterationDim`.
  for (auto dim : llvm::seq<int64_t>(0, origNumLoops)) {
    if (processedDims.count(dim))
      continue;
    collapsedOpToOrigOpIterationDim.emplace_back(ReassociationIndices{dim});
  }

  llvm::sort(collapsedOpToOrigOpIterationDim,
             [&](ReassociationIndicesRef lhs, ReassociationIndicesRef rhs) {
               return lhs[0] < rhs[0];
             });
  origOpToCollapsedOpIterationDim.resize(origNumLoops);
  for (const auto &foldedDims :
       llvm::enumerate(collapsedOpToOrigOpIterationDim)) {
    for (const auto &dim : enumerate(foldedDims.value()))
      origOpToCollapsedOpIterationDim[dim.value()] =
          std::make_pair<int64_t, unsigned>(foldedDims.index(), dim.index());
  }
  return success();
}

static SmallVector<ReshapeOperandInfo>
getAttentionReshapeInfo(LinalgExt::AttentionOp attentionOp) {
  return llvm::map_to_vector(
      attentionOp->getOpOperands(), [&](OpOperand &opOperand) {
        ReshapeOperandInfo operandInfo;
        auto operandType = dyn_cast<ShapedType>(opOperand.get().getType());
        if (!operandType) {
          assert(
              attentionOp.getMatchingIndexingMap(&opOperand).getNumResults() ==
                  0 &&
              "expected non-shaped type to have no results in indexing map");
          return operandInfo;
        }

        operandInfo.originalShape = operandType.getShape();
        for (auto result :
             attentionOp.getMatchingIndexingMap(&opOperand).getResults()) {
          operandInfo.operandToIterationSpace.push_back(
              cast<AffineDimExpr>(result).getPosition());
        }
        return operandInfo;
      });
}

static SmallVector<ReshapeOperandInfo>
getScatterReshapeInfo(LinalgExt::ScatterOp scatterOp) {
  SmallVector<ReshapeOperandInfo> infos;
  auto rankOfContiguousSlice =
      scatterOp.getOriginalType().getRank() - scatterOp.getIndexDepth();
  auto updateRank = scatterOp.getUpdateType().getRank();

  // Operand #0 Updates
  {
    ReshapeOperandInfo updateInfo;
    updateInfo.originalShape = scatterOp.getUpdateType().getShape();
    llvm::append_range(updateInfo.operandToIterationSpace,
                       llvm::seq<int64_t>(0, updateRank));
    infos.push_back(std::move(updateInfo));
  }

  // Operand#1 Indices
  {
    ReshapeOperandInfo indicesInfo;
    indicesInfo.originalShape = scatterOp.getIndicesType().getShape();
    llvm::append_range(indicesInfo.operandToIterationSpace,
                       llvm::seq<int64_t>(0, scatterOp.getBatchRank()));
    if (scatterOp.getBatchRank() != scatterOp.getIndicesType().getRank())
      indicesInfo.operandToIterationSpace.push_back(
          ReshapeOperandInfo::kNoMapping);
    infos.push_back(std::move(indicesInfo));
  }

  // Operand #2 Original
  {
    ReshapeOperandInfo originalInfo;
    originalInfo.originalShape = scatterOp.getOriginalType().getShape();
    originalInfo.operandToIterationSpace.append(scatterOp.getIndexDepth(),
                                                ReshapeOperandInfo::kNoMapping);
    llvm::append_range(
        originalInfo.operandToIterationSpace,
        llvm::seq(updateRank - rankOfContiguousSlice, updateRank));
    infos.push_back(std::move(originalInfo));
  }
  return infos;
};

static AffineMap
getIndexingMapInExpandedOp(OpBuilder &builder, AffineMap indexingMap,
                           const ExpansionInfo &expansionInfo) {
  SmallVector<AffineExpr> newExprs;
  for (AffineExpr expr : indexingMap.getResults()) {
    unsigned pos = cast<AffineDimExpr>(expr).getPosition();
    auto expandedExprs = llvm::to_vector_of<AffineExpr, 6>(
        llvm::map_range(expansionInfo.getExpandedLoops(pos), [&](int64_t v) {
          return builder.getAffineDimExpr(static_cast<unsigned>(v));
        }));
    newExprs.append(expandedExprs.begin(), expandedExprs.end());
  }
  return AffineMap::get(expansionInfo.getExpandedNumLoops(),
                        indexingMap.getNumSymbols(), newExprs,
                        builder.getContext());
}

static bool isFusableWithReshapeByDimExpansion(AttentionOp op,
                                               OpOperand *fusableOpOperand) {
  // Is fusable only if:
  // - All the indexing maps for operands and results are projected
  //   permutations.
  // - The fused tensor is not a scalar.
  // - All the loops for the reshaped operand are parallel loops.
  SmallVector<utils::IteratorType> iteratorTypes = op.getLoopIteratorTypes();
  AffineMap operandMap = op.getMatchingIndexingMap(fusableOpOperand);
  return operandMap && op.hasPureTensorSemantics() &&
         llvm::all_of(op.getIndexingMapsArray(),
                      [](AffineMap map) {
                        return map && map.isProjectedPermutation();
                      }) &&
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
  ExpansionInfo expansionInfo;
  if (failed(expansionInfo.compute(
          getAttentionReshapeInfo(attentionOp),
          attentionOp.getStaticLoopRanges().value(), fusableOpOperand,
          isExpanding ? expandingReshapeOp.getReassociationIndices()
                      : collapsingReshapeOp.getReassociationIndices(),
          expandedType.getShape())))
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
    // Reshape the operand to get the right type.
    std::optional<Value> expanded =
        expansionInfo.getOrCreateExpanded(loc, opOperand, rewriter);
    if (!expanded)
      return std::nullopt;
    expandedOpOperands.push_back(*expanded);
    continue;
  }

  Value output;
  OpOperand &outOperand = attentionOp.getOutputMutable();

  std::optional<Value> maybeOutput =
      expansionInfo.getOrCreateExpanded(loc, &outOperand, rewriter);
  if (!maybeOutput)
    return std::nullopt;
  output = *maybeOutput;

  Value maskOperand;
  if (expandedOpOperands.size() > 4) {
    maskOperand = expandedOpOperands[4];
  }

  // Create a new `AttentionOp` that has the computed operands/indexing maps.
  TypeRange resultTypes = ValueRange(output).getTypes();
  auto fusedOp = rewriter.create<AttentionOp>(
      attentionOp.getLoc(), resultTypes, expandedOpOperands[0],
      expandedOpOperands[1], expandedOpOperands[2], expandedOpOperands[3],
      output, rewriter.getAffineMapArrayAttr(expandedOpIndexingMaps),
      maskOperand);

  rewriter.inlineRegionBefore(attentionOp.getRegion(), fusedOp.getRegion(),
                              fusedOp.getRegion().begin());

  // Reshape the result values to their original shape if this is a collapsing
  // reshape folded into its consumer.
  SmallVector<Value> resultVals;
  for (OpResult opResult : attentionOp->getOpResults()) {
    int64_t resultNumber = opResult.getResultNumber();
    if (resultTypes[resultNumber] != opResult.getType()) {
      SmallVector<ReassociationIndices> reassociation =
          expansionInfo.getReassoc(attentionOp.getTiedOpOperand(opResult));
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

static std::optional<Value>
fuseScatterWithReshapeByExpansion(ScatterOp scatterOp, Operation *reshapeOp,
                                  OpOperand *fusableOpOperand,
                                  PatternRewriter &rewriter) {
  Location loc = scatterOp.getLoc();
  // Check if reshape is expanding or collapsing.
  auto expandingReshapeOp = dyn_cast<tensor::ExpandShapeOp>(*reshapeOp);
  auto collapsingReshapeOp = dyn_cast<tensor::CollapseShapeOp>(*reshapeOp);
  bool isExpanding = (expandingReshapeOp != nullptr);
  RankedTensorType expandedType = isExpanding
                                      ? expandingReshapeOp.getResultType()
                                      : collapsingReshapeOp.getSrcType();
  ExpansionInfo info;
  if (failed(info.compute(
          getScatterReshapeInfo(scatterOp),
          scatterOp.getStaticLoopRanges().value(), fusableOpOperand,
          isExpanding ? expandingReshapeOp.getReassociationIndices()
                      : collapsingReshapeOp.getReassociationIndices(),
          expandedType.getShape()))) {
    return std::nullopt;
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(scatterOp);

  OpOperand *update = scatterOp.getDpsInputOperand(0);
  OpOperand *indices = scatterOp.getDpsInputOperand(1);
  OpOperand *original = scatterOp.getDpsInitOperand(0);
  auto newUpdates = info.getOrCreateExpanded(loc, update, rewriter).value();
  auto newIndices = info.getOrCreateExpanded(loc, indices, rewriter).value();
  auto newOriginal = info.getOrCreateExpanded(loc, original, rewriter).value();

  auto newScatter = rewriter.create<ScatterOp>(
      loc, newOriginal.getType(), ValueRange{newUpdates, newIndices},
      ValueRange{newOriginal}, scatterOp.getDimensionMap(),
      scatterOp.getUniqueIndices());
  rewriter.inlineRegionBefore(scatterOp.getRegion(), newScatter.getRegion(),
                              newScatter.getRegion().begin());

  auto originalShapeMap = info.getShapeMap(original);
  SmallVector<ReassociationIndices> originalReassoc =
      computeReassocFromShapeMap(originalShapeMap);

  // Collapse back to original shape.
  if (isIdentityReassoc(originalReassoc)) {
    return {newScatter.getResult(0)};
  }
  auto newCollapse = rewriter.create<tensor::CollapseShapeOp>(
      loc, scatterOp.getOriginalType(), newScatter.getResult(0),
      originalReassoc);

  return {newCollapse};
}

//===----------------------------------------------------------------------===//
// Fuse By Expansion Patterns
//===----------------------------------------------------------------------===//

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

struct DropScatterUnitIndexDepth final : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern<ScatterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ScatterOp scatterOp,
                                PatternRewriter &rewriter) const override {
    llvm::ArrayRef<int64_t> indicesShape =
        scatterOp.getIndicesType().getShape();
    int64_t batchRank = scatterOp.getBatchRank();
    if (indicesShape.size() == batchRank || indicesShape.back() != 1) {
      return failure();
    }
    SmallVector<ReassociationIndices> reassoc;
    reassoc.reserve(indicesShape.size());
    for (auto i : llvm::seq<int64_t>(0, batchRank - 1)) {
      reassoc.emplace_back(1, i);
    }
    reassoc.push_back(ReassociationIndices{batchRank - 1, batchRank});
    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        scatterOp.getLoc(), scatterOp.getIndices(), reassoc);

    rewriter.modifyOpInPlace(scatterOp, [&]() {
      scatterOp.setOperand(ScatterOp::kIndicesOpNum, collapseOp.getResult());
    });
    return success();
  }
};

struct FoldScatterWithProducerReshapeByExpansion final
    : public OpRewritePattern<ScatterOp> {
  FoldScatterWithProducerReshapeByExpansion(
      MLIRContext *context, linalg::ControlFusionFn controlFoldingReshapes,
      PatternBenefit benefit = 1)
      : OpRewritePattern<ScatterOp>(context, benefit),
        controlFoldingReshapes(std::move(controlFoldingReshapes)) {}

  LogicalResult matchAndRewrite(ScatterOp scatterOp,
                                PatternRewriter &rewriter) const override {
    for (OpOperand &opOperand : scatterOp->getOpOperands()) {
      tensor::CollapseShapeOp reshapeOp =
          opOperand.get().getDefiningOp<tensor::CollapseShapeOp>();
      if (!reshapeOp)
        continue;
      if (!controlFoldingReshapes(&opOperand))
        continue;

      std::optional<Value> replacementValue = fuseScatterWithReshapeByExpansion(
          scatterOp, reshapeOp, &opOperand, rewriter);
      if (!replacementValue)
        return failure();
      rewriter.replaceOp(scatterOp, *replacementValue);
      return success();
    }
    return failure();
  }

  linalg::ControlFusionFn controlFoldingReshapes;
};

struct FoldScatterWithConsumerReshapeByExpansion final
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  FoldScatterWithConsumerReshapeByExpansion(
      MLIRContext *context, linalg::ControlFusionFn controlFoldingReshapes,
      PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::ExpandShapeOp>(context, benefit),
        controlFoldingReshapes(std::move(controlFoldingReshapes)) {}

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    auto producerResult = dyn_cast<OpResult>(expandOp.getSrc());
    if (!producerResult) {
      return rewriter.notifyMatchFailure(expandOp,
                                         "source not produced by an operation");
    }

    auto scatterOp = producerResult.getDefiningOp<LinalgExt::ScatterOp>();
    if (!scatterOp) {
      return failure();
    }

    if (!controlFoldingReshapes(&expandOp.getSrcMutable())) {
      return failure();
    }

    std::optional<Value> replacementValue = fuseScatterWithReshapeByExpansion(
        scatterOp, expandOp, scatterOp.getTiedOpOperand(producerResult),
        rewriter);
    if (!replacementValue)
      return failure();
    rewriter.replaceOp(scatterOp, *replacementValue);
    return success();
  }

  linalg::ControlFusionFn controlFoldingReshapes;
};

} // namespace

/// Return the `reassociation` indices to use to collapse the operand when the
/// iteration space of a generic op is collapsed.
static SmallVector<ReassociationIndices>
getOperandReassociation(AffineMap indexingMap,
                        const CollapsingInfo &collapsingInfo) {
  unsigned counter = 0;
  SmallVector<ReassociationIndices> operandReassociation;
  auto origOpToCollapsedOpMapping =
      collapsingInfo.getOrigOpToCollapsedOpMapping();
  auto collapsedOpToOrigOpMapping =
      collapsingInfo.getCollapsedOpToOrigOpMapping();
  while (counter < indexingMap.getNumResults()) {
    unsigned dim =
        cast<AffineDimExpr>(indexingMap.getResult(counter)).getPosition();
    // This is the start of a collapsed dimensions of the iteration that
    // is gauranteed to be preserved in the indexing map. The number of folded
    // dims is obtained from the collapsed op to original op mapping.
    unsigned numFoldedDims =
        collapsedOpToOrigOpMapping[origOpToCollapsedOpMapping[dim].first]
            .size();
    if (origOpToCollapsedOpMapping[dim].second == 0) {
      auto range = llvm::seq<unsigned>(counter, counter + numFoldedDims);
      operandReassociation.emplace_back(range.begin(), range.end());
    }
    counter += numFoldedDims;
  }
  return operandReassociation;
}

/// Get the new value to use for a given `OpOperand` in the collapsed operation.
static Value getCollapsedOpOperand(Location loc, AttentionOp op,
                                   OpOperand *opOperand,
                                   const CollapsingInfo &collapsingInfo,
                                   OpBuilder &builder) {
  AffineMap indexingMap = op.getMatchingIndexingMap(opOperand);
  SmallVector<ReassociationIndices> operandReassociation =
      getOperandReassociation(indexingMap, collapsingInfo);

  // If the number of entries in the reassociation for the operand is same as
  // the number of results of the indexing map, then nothing to do for this
  // operand.
  Value operand = opOperand->get();
  if (operandReassociation.size() == indexingMap.getNumResults())
    return operand;

  // Insert a reshape to collapse the dimensions.
  if (isa<MemRefType>(operand.getType())) {
    return builder
        .create<memref::CollapseShapeOp>(loc, operand, operandReassociation)
        .getResult();
  }
  return builder
      .create<tensor::CollapseShapeOp>(loc, operand, operandReassociation)
      .getResult();
}

static void collapseOperandsAndResults(AttentionOp op,
                                       const CollapsingInfo &collapsingInfo,
                                       RewriterBase &rewriter,
                                       SmallVectorImpl<Value> &inputOperands,
                                       SmallVectorImpl<Value> &outputOperands,
                                       SmallVectorImpl<Type> &resultTypes) {
  Location loc = op->getLoc();
  inputOperands =
      llvm::map_to_vector(op.getDpsInputOperands(), [&](OpOperand *opOperand) {
        return getCollapsedOpOperand(loc, op, opOperand, collapsingInfo,
                                     rewriter);
      });

  // Get the output operands and result types.
  resultTypes.reserve(op.getNumDpsInits());
  outputOperands.reserve(op.getNumDpsInits());
  for (OpOperand &output : op.getDpsInitsMutable()) {
    Value newOutput =
        getCollapsedOpOperand(loc, op, &output, collapsingInfo, rewriter);
    outputOperands.push_back(newOutput);
    // If the op has "buffer semantics", then the init operands are ranked
    // memrefs and the op has no results.
    if (!op.hasPureBufferSemantics())
      resultTypes.push_back(newOutput.getType());
  }
}

/// Compute the indexing map in the collapsed op that corresponds to the given
/// `indexingMap` of the original operation.
static AffineMap
getCollapsedOpIndexingMap(AffineMap indexingMap,
                          const CollapsingInfo &collapsingInfo) {
  MLIRContext *context = indexingMap.getContext();
  assert(indexingMap.isProjectedPermutation() &&
         "expected indexing map to be projected permutation");
  SmallVector<AffineExpr> resultExprs;
  auto origOpToCollapsedOpMapping =
      collapsingInfo.getOrigOpToCollapsedOpMapping();
  for (auto expr : indexingMap.getResults()) {
    unsigned dim = cast<AffineDimExpr>(expr).getPosition();
    // If the dim is not the first of the collapsed dim, do nothing.
    if (origOpToCollapsedOpMapping[dim].second != 0)
      continue;
    // The next n-dims are guaranteed to be collapsed. So just use the
    // iteration dimension of the collapsed op.
    resultExprs.push_back(
        getAffineDimExpr(origOpToCollapsedOpMapping[dim].first, context));
  }
  return AffineMap::get(collapsingInfo.getCollapsedOpIterationRank(), 0,
                        resultExprs, context);
}

/// Get the iterator types for the collapsed operation given the original
/// iterator types and collapsed dimensions.
static SmallVector<utils::IteratorType>
getCollapsedOpIteratorTypes(ArrayRef<utils::IteratorType> iteratorTypes,
                            const CollapsingInfo &collapsingInfo) {
  SmallVector<utils::IteratorType> collapsedIteratorTypes;
  for (ReassociationIndicesRef foldedIterDims :
       collapsingInfo.getCollapsedOpToOrigOpMapping()) {
    assert(!foldedIterDims.empty() &&
           "reassociation indices expected to have non-empty sets");
    // Just pick the iterator type of the first folded dim. Pre-condition checks
    // expected to have checked that iterator types of all folded dimensions are
    // the same.
    collapsedIteratorTypes.push_back(iteratorTypes[foldedIterDims[0]]);
  }
  return collapsedIteratorTypes;
}

/// Returns a copy of `attentionOp` with collapsed iteration dimensions.
static Operation *createCollapsedOp(AttentionOp origOp,
                                    const CollapsingInfo &collapsingInfo,
                                    RewriterBase &rewriter) {
  SmallVector<Value> inputOperands, outputOperands;
  SmallVector<Type> resultTypes;
  collapseOperandsAndResults(origOp, collapsingInfo, rewriter, inputOperands,
                             outputOperands, resultTypes);
  SmallVector<AffineMap> indexingMaps(
      llvm::map_range(origOp.getIndexingMapsArray(), [&](AffineMap map) {
        return getCollapsedOpIndexingMap(map, collapsingInfo);
      }));

  SmallVector<utils::IteratorType> iteratorTypes(getCollapsedOpIteratorTypes(
      origOp.getLoopIteratorTypes(), collapsingInfo));

  Value maskOperand;
  if (inputOperands.size() > 4) {
    maskOperand = inputOperands[4];
  }

  auto collapsedOp = rewriter.create<AttentionOp>(
      origOp.getLoc(), resultTypes, inputOperands[0], inputOperands[1],
      inputOperands[2], inputOperands[3], outputOperands[0],
      rewriter.getAffineMapArrayAttr(indexingMaps), maskOperand);
  rewriter.inlineRegionBefore(origOp.getRegion(), collapsedOp.getRegion(),
                              collapsedOp.getRegion().begin());
  return collapsedOp;
}

FailureOr<CollapseResult>
collapseOpIterationDims(AttentionOp op,
                        ArrayRef<ReassociationIndices> foldedIterationDims,
                        RewriterBase &rewriter) {
  if (op.getNumLoops() <= 1 || foldedIterationDims.empty() ||
      llvm::all_of(foldedIterationDims, [](ReassociationIndicesRef foldedDims) {
        return foldedDims.size() <= 1;
      }))
    return failure();

  FailureOr<SmallVector<int64_t>> staticLoops = op.getStaticLoopRanges();
  if (failed(staticLoops) ||
      llvm::any_of(staticLoops.value(), ShapedType::isDynamic)) {
    return failure();
  }

  CollapsingInfo collapsingInfo;
  if (failed(
          collapsingInfo.initialize(op.getNumLoops(), foldedIterationDims))) {
    return rewriter.notifyMatchFailure(
        op, "illegal to collapse specified dimensions");
  }

  Operation *collapsedOp = createCollapsedOp(op, collapsingInfo, rewriter);

  auto loc = op.getLoc();
  SmallVector<Value> results;
  for (const auto &originalResult : llvm::enumerate(op->getResults())) {
    Value collapsedOpResult = collapsedOp->getResult(originalResult.index());
    auto originalResultType =
        cast<ShapedType>(originalResult.value().getType());
    auto collapsedOpResultType = cast<ShapedType>(collapsedOpResult.getType());
    if (collapsedOpResultType.getRank() != originalResultType.getRank()) {
      AffineMap indexingMap =
          op.getIndexingMapMatchingResult(originalResult.value());
      SmallVector<ReassociationIndices> reassociation =
          getOperandReassociation(indexingMap, collapsingInfo);
      Value result;
      if (isa<MemRefType>(collapsedOpResult.getType())) {
        MemRefType expandShapeResultType = MemRefType::get(
            originalResultType.getShape(), originalResultType.getElementType());
        result = rewriter.create<memref::ExpandShapeOp>(
            loc, expandShapeResultType, collapsedOpResult, reassociation);
      } else {
        result = rewriter.create<tensor::ExpandShapeOp>(
            loc, originalResultType, collapsedOpResult, reassociation);
      }
      results.push_back(result);
    } else {
      results.push_back(collapsedOpResult);
    }
  }
  return CollapseResult{results, collapsedOp};
}

void populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFoldingReshapes) {
  patterns.add<FoldAttentionWithConsumerReshapeByExpansion>(
      patterns.getContext(), controlFoldingReshapes);
  patterns.add<FoldAttentionWithProducerReshapeByExpansion>(
      patterns.getContext(), controlFoldingReshapes);
  patterns.add<FoldScatterWithProducerReshapeByExpansion>(
      patterns.getContext(), controlFoldingReshapes);
  patterns.add<FoldScatterWithConsumerReshapeByExpansion>(
      patterns.getContext(), controlFoldingReshapes);
}

SmallVector<unsigned> defaultControlDropUnitDims(Operation *op) {
  auto fusionOp = cast<LinalgFusionOpInterface>(op);
  return llvm::to_vector(llvm::seq<unsigned>(0, fusionOp.getNumLoops()));
}

void populateFoldUnitExtentDimsPatterns(
    RewritePatternSet &patterns, const linalg::ControlDropUnitDims &options) {
  patterns.add<DropScatterUnitIndexDepth>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
