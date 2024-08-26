// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-fuse-horizontal-contractions"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FUSEHORIZONTALCONTRACTIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

struct FuseHorizontalContractionsPass final
    : public impl::FuseHorizontalContractionsPassBase<
          FuseHorizontalContractionsPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

/// Structs that captures the ops that are to be fused
struct HorizontalFusionGroup {
  // Contractions op that are to be fused.
  SmallVector<linalg::LinalgOp> contractionOps;
  // Optional truncate operations that could be following the contraction op.
  std::optional<SmallVector<linalg::GenericOp>> truncateOps;
};

/// Helper method to check operations equivalence
static bool checkOperationEquivalence(Operation *lhsOp, Operation *rhsOp) {
  // During equivalence check, it would have been easier if `checkEquivalence`
  // would just use `OpOperands *`. Since it takes `Value`s for now, just
  // check that the values are the same as operands. This is potentially
  // making the match too broad, but is an OK work-around for now.
  // TODO(MaheshRavishankar): Fix upstream `checkEquivalence` signater in
  // `OperationEquivalence::isEquivalentTo`.
  llvm::SmallDenseSet<Value, 8> operands;
  operands.insert(lhsOp->operand_begin(), lhsOp->operand_end());
  operands.insert(rhsOp->operand_begin(), rhsOp->operand_end());

  llvm::DenseMap<Value, Value> equivalentValues;
  auto checkEquivalent = [&](Value lhsValue, Value rhsValue) {
    if (operands.contains(lhsValue) && operands.contains(rhsValue)) {
      return success();
    }
    return success(equivalentValues.lookup(lhsValue) == rhsValue ||
                   equivalentValues.lookup(rhsValue) == lhsValue);
  };
  auto markEquivalent = [&](Value v1, Value v2) { equivalentValues[v1] = v2; };
  return OperationEquivalence::isEquivalentTo(
      lhsOp, rhsOp, checkEquivalent, markEquivalent,
      /*flags=*/OperationEquivalence::IgnoreLocations);
}

/// Check that an operation is a `empty -> fill -> contraction`
static bool isEmptyFillContractionDAGRootOp(
    linalg::LinalgOp linalgOp,
    std::optional<linalg::LinalgOp> seedContractionOp = std::nullopt) {
  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return false;
  }
  auto fillOp = linalgOp.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fillOp) {
    return false;
  }
  // For convenience check that the fill value is 0. This is not
  // a necessity, but easier to handle the rewrite this way.
  if (!matchPattern(fillOp.getDpsInputOperand(0)->get(), m_AnyZeroFloat()) &&
      !matchPattern(fillOp.getDpsInputOperand(0)->get(), m_Zero())) {
    return false;
  }
  if (!fillOp.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>()) {
    return false;
  }
  if (seedContractionOp) {
    return checkOperationEquivalence(linalgOp, seedContractionOp.value());
  }
  return true;
}

/// Check that a given operation is "horizontal" to the group. The operation
/// is horizontal if the `slice` of the operation does not contain any op
/// from the group.
static bool isHorizontalToGroup(Operation *op,
                                const llvm::SetVector<Operation *> &currGroup,
                                const DominanceInfo &dominanceInfo,
                                Operation *seedOp) {
  BackwardSliceOptions options;
  // Limit the slice to the seed to make sure the slice is small.
  options.filter = [&](Operation *op) {
    return !dominanceInfo.properlyDominates(op, seedOp);
  };
  llvm::SetVector<Operation *> slice;
  getBackwardSlice(op, &slice, options);
  return !llvm::any_of(currGroup, [&](Operation *groupedOp) {
    return slice.contains(groupedOp);
  });
}

/// Get user of operation that is a truncate operation.
static std::optional<linalg::GenericOp>
getTruncateOp(Operation *op,
              const llvm::SetVector<Operation *> &groupedOperations,
              const DominanceInfo &dominanceInfo,
              std::optional<linalg::GenericOp> seedTruncateOp = std::nullopt) {
  if (!op->hasOneUse()) {
    return std::nullopt;
  }
  Operation *user = *op->user_begin();
  // TODO: This test should not be really needed. We should be able to check
  // for ANY elementwise operation.
  if (!IREE::LinalgExt::isBitTruncateOp(user)) {
    return std::nullopt;
  }
  auto genericOp = dyn_cast<linalg::GenericOp>(user);
  if (!genericOp) {
    return std::nullopt;
  }
  if (seedTruncateOp) {
    if (!checkOperationEquivalence(genericOp, seedTruncateOp.value())) {
      return std::nullopt;
    }
    if (!isHorizontalToGroup(genericOp, groupedOperations, dominanceInfo,
                             seedTruncateOp.value())) {
      return std::nullopt;
    }
  }
  return genericOp;
}

/// Find all candidates that can be used for horizontal fusion. For example
/// ```
/// %0 = linalg.matmul ins(%arg0, %arg1)
/// %1 = linalg.matmul ins(%arg0, %arg2)
/// %2 = linalg.matmul ins(%arg0, %arg3)
/// ```
///
/// where all matmul share an operand can be combined into
///
/// ```
/// %4 = linalg.matmul ins(%arg0, concat(%arg1, %arg2, %arg3))
/// ```
///
/// This method recognizes such patterns. It also accounts for the quantized
/// case where individual operations might be have lower-precision operands and
/// accumulate in higher precision, followed by a `linalg.generic` that performs
/// the `truncf` on the result.
static std::optional<HorizontalFusionGroup> getHorizontalFusionGroupMembers(
    linalg::LinalgOp seedOp,
    const llvm::SmallDenseSet<linalg::LinalgOp> &groupedOperations,
    const DominanceInfo &dominanceInfo, int fusionLimit) {

  Value lhs = seedOp->getOperand(0);
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  Value rhs = seedOp->getOperand(1);
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  Value out = seedOp->getOperand(2);
  auto outType = cast<RankedTensorType>(out.getType());

  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !outType.hasStaticShape()) {
    return std::nullopt;
  }

  SetVector<Operation *> allOps;
  SmallVector<linalg::LinalgOp> contractionOps = {seedOp};
  std::optional<linalg::GenericOp> seedTruncOp =
      getTruncateOp(seedOp, allOps, dominanceInfo);
  std::optional<SmallVector<linalg::GenericOp>> truncateOps;
  if (seedTruncOp) {
    truncateOps = {seedTruncOp.value()};
  }
  allOps.insert(seedOp);
  if (seedTruncOp) {
    allOps.insert(seedTruncOp.value());
  }

  auto canBeGrouped = [&](linalg::LinalgOp linalgOp) -> bool {
    if (linalgOp->getParentOp() != seedOp->getParentOp()) {
      return false;
    }

    // Constraints of the operation itself.
    if (!isEmptyFillContractionDAGRootOp(linalgOp, seedOp)) {
      return false;
    }
    if (linalgOp->getOperand(0).getType() != lhsType ||
        linalgOp->getOperand(1).getType() != rhsType ||
        linalgOp->getOperand(2).getType() != outType) {
      return false;
    }
    if (groupedOperations.contains(linalgOp)) {
      return false;
    }

    // Structural constraints related to being able to fuse the operations.
    if (!dominanceInfo.properlyDominates(seedOp, linalgOp)) {
      return false;
    }
    if (!isHorizontalToGroup(linalgOp, allOps, dominanceInfo, seedOp)) {
      return false;
    }
    return true;
  };

  // Iterate over users of LHS to find ops that can be grouped with the seed.
  SmallVector<Operation *> lhsUsers;
  for (Operation *lhsUser : lhs.getUsers()) {
    if (lhsUser->getBlock() != seedOp->getBlock() || lhsUser == seedOp) {
      continue;
    }

    auto linalgUser = dyn_cast<linalg::LinalgOp>(lhsUser);
    if (!linalgUser || !canBeGrouped(linalgUser)) {
      continue;
    }
    lhsUsers.push_back(lhsUser);
  }

  // Sort the users so that the order is deterministic
  llvm::sort(lhsUsers, [&](Operation *lhs, Operation *rhs) {
    return dominanceInfo.properlyDominates(lhs, rhs);
  });

  // Collect all contraction op users of lhs.
  for (Operation *lhsUser : lhsUsers) {
    auto linalgUser = dyn_cast<linalg::LinalgOp>(lhsUser);
    if (!linalgUser) {
      continue;
    }

    std::optional<linalg::GenericOp> userTruncOp =
        getTruncateOp(linalgUser, allOps, dominanceInfo, seedTruncOp);
    // If there are truncate ops to fuse and current contraction op
    // does not have a compatible truncate op to fuse as well, ignore
    // the op for horizontal fusion.
    if (truncateOps && !userTruncOp) {
      continue;
    }

    contractionOps.push_back(linalgUser);
    allOps.insert(linalgUser);
    if (truncateOps) {
      truncateOps.value().push_back(userTruncOp.value());
      allOps.insert(userTruncOp.value());
    }
    if (contractionOps.size() >= fusionLimit) {
      break;
    }
  }

  if (contractionOps.size() == 1) {
    return std::nullopt;
  }

  return HorizontalFusionGroup{contractionOps, truncateOps};
}

/// Concatenate the given tensor `values`. The assumption here
/// is that all the `values` are the same type. These are concatanted
/// by adding a extra outer dimension to each value and concatenating
/// along the outer-most dim.
static Value concatenateValues(RewriterBase &rewriter, Location loc,
                               ArrayRef<Value> values) {
  assert((values.size() >= 2) && "Invalid number of operands to concatenate");
  auto valueType = cast<RankedTensorType>(values[0].getType());

  SmallVector<Value> concatOperands;
  for (auto v : values) {
    auto t = cast<RankedTensorType>(v.getType());
    SmallVector<int64_t> expandedTypeShape = {1};
    expandedTypeShape.append(t.getShape().begin(), t.getShape().end());
    auto expandedType =
        RankedTensorType::get(expandedTypeShape, t.getElementType());
    SmallVector<OpFoldResult> expandedShape = {rewriter.getIndexAttr(1)};
    auto mixedSizes = tensor::getMixedSizes(rewriter, loc, v);
    expandedShape.append(mixedSizes.begin(), mixedSizes.end());

    SmallVector<ReassociationIndices> reassoc;
    if (t.getRank() != 0) {
      reassoc.push_back({0, 1});
      for (int i = 0, e = valueType.getRank() - 1; i < e; ++i) {
        reassoc.push_back({i + 2});
      }
    }

    Value expanded = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandedType, v, reassoc, expandedShape);
    concatOperands.push_back(expanded);
  }

  Value concatedVal =
      rewriter.create<tensor::ConcatOp>(loc, /*dim=*/0, concatOperands);
  return concatedVal;
}

/// Compute the indexing map used in the concatenated operation.
/// The indexing map is either
/// 1) when shiftOnly = false, adds an extra outermost dimension to the indexing
///    map and adding that dimension as the outermost dimension in the range.
///    This is used for case where the original operands of the operations are
///    concatanated as well to get the operand for the horizontally-fused
///    operation.
/// 2) when shiftOnly = true,  adds an extra outermost dimension to the indexing
///    map without adding that dimension as the outermost dimension in the
///    range. This is used for case where the same value is used as an operand
///    for all the concatenated operations. In such cases the original operand
///    can just be broadcasted along the concatenated dimension in the
///    horizontally-fused operation.
static AffineMap getConcatenatedIndexingMap(RewriterBase &rewriter,
                                            AffineMap origIndexingMap,
                                            bool shiftOnly = false) {
  AffineMap newIndexingMap = origIndexingMap.shiftDims(1);
  if (shiftOnly) {
    return newIndexingMap;
  }
  return newIndexingMap.insertResult(rewriter.getAffineDimExpr(0), 0);
}

/// During horizontal fusion, there might be operands of the fused operations
/// whose definitions are interspersed between the fused operations. For groups
/// chosen to fuse horizontally, such operations can be moved before the
/// seed contraction operation (where the fused operation is generated).
template <typename T>
static LogicalResult
moveOperandDefs(RewriterBase &rewriter, ArrayRef<T> operations,
                Operation *insertionPoint, DominanceInfo &dominanceInfo,
                ArrayRef<linalg::LinalgOp> ignoreOperations = {}) {
  BackwardSliceOptions options;
  llvm::DenseSet<Operation *> ignoreOperationsSet;
  ignoreOperationsSet.insert(ignoreOperations.begin(), ignoreOperations.end());
  options.filter = [&](Operation *op) {
    return !dominanceInfo.properlyDominates(op, insertionPoint) &&
           !ignoreOperationsSet.contains(op);
  };
  // Set inclusive to true cause the slice is computed from the operand, and
  // we want to include the defining op (which is the point here)
  options.inclusive = true;

  llvm::SetVector<Operation *> slice;
  for (auto op : operations) {
    for (auto operand : op->getOperands()) {
      getBackwardSlice(operand, &slice, options);
    }
  }

  mlir::topologicalSort(slice);
  for (auto op : slice) {
    rewriter.moveOpBefore(op, insertionPoint);
  }
  return success();
}

/// On finding this pattern
/// ```
/// %0 = linalg.matmul ins(%arg0, %arg1)
/// %1 = linalg.matmul ins(%arg0, %arg2)
/// %2 = linalg.matmul ins(%arg0, %arg3)
/// ```
///
/// where all matmul share an operand can be combined into
/// rewrite to
///
/// ```
/// %arg1_r = tensor.expand_shape %arg1 [[0, 1], ...] : tensor<?x?xf32> to
///     tensor<1x?x?xf32>
/// %arg2_r = tensor.expand_shape %arg2 [[0, 1], ...] : tensor<?x?xf32> to
///     tensor<1x?x?xf32>
/// %arg3_r = tensor.expand_shape %arg3 [[0, 1], ...] : tensor<?x?xf32> to
///     tensor<1x?x?xf32>
/// %rhs = tensor.concat(%arg1_r, %arg2_r, %arg3_r)
/// %fused = linalg.generic {
///     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
///                      affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
///                      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>}],
///     iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
///   ins(%arg0, %rhs) ... { ... }
/// %0 = tensor.extract_slice %fused [0, 0, 0] ... : tensor<1x?x?xf32> to
///     tensor<?x?xf32>
/// %1 = tensor.extract_slice %fused [1, 0, 0] ... : tensor<1x?x?xf32> to
///     tensor<?x?xf32>
/// %2 = tensor.extract_slice %fused [2, 0, 0] ... : tensor<1x?x?xf32> to
///     tensor<?x?xf32>
/// ```
///
/// Also accounts for quantized cases where inputs are at lower precision and
/// accumulate is in higher-precision with truncate getting back to the
/// quantized sizes.
static LogicalResult fuseGroup(RewriterBase &rewriter,
                               HorizontalFusionGroup &fusionGroup,
                               DominanceInfo &dominanceInfo) {
  linalg::LinalgOp baseContractOp = fusionGroup.contractionOps.front();
  Location loc = baseContractOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(baseContractOp);

  if (failed(moveOperandDefs(
          rewriter, ArrayRef<linalg::LinalgOp>(fusionGroup.contractionOps),
          baseContractOp, dominanceInfo))) {
    return baseContractOp.emitOpError("failed to re-order operand definitions");
  }

  SmallVector<Value> rhsValues;
  SmallVector<Value> initValues;
  for (auto op : fusionGroup.contractionOps) {
    Value rhs = op.getDpsInputOperand(1)->get();
    Value init = op.getDpsInitOperand(0)->get();
    rhsValues.push_back(rhs);
    initValues.push_back(init);
  }
  Value newContractRhs = concatenateValues(rewriter, loc, rhsValues);
  Value newContractInit = concatenateValues(rewriter, loc, initValues);

  auto baseContractResultType =
      cast<RankedTensorType>(baseContractOp->getResult(0).getType());
  SmallVector<int64_t> newContractResultShape = {
      static_cast<int64_t>(rhsValues.size())};
  newContractResultShape.append(baseContractResultType.getShape().begin(),
                                baseContractResultType.getShape().end());
  auto newContractResultType = RankedTensorType::get(
      newContractResultShape, baseContractResultType.getElementType());

  Value lhs = baseContractOp->getOperand(0);

  SmallVector<utils::IteratorType> newContractIteratorTypes = {
      utils::IteratorType::parallel};
  newContractIteratorTypes.append(baseContractOp.getIteratorTypesArray());

  SmallVector<AffineMap> newContractIndexingMaps =
      baseContractOp.getIndexingMapsArray();
  newContractIndexingMaps[0] = getConcatenatedIndexingMap(
      rewriter, newContractIndexingMaps[0], /*shiftOnly=*/true);
  newContractIndexingMaps[1] =
      getConcatenatedIndexingMap(rewriter, newContractIndexingMaps[1]);
  newContractIndexingMaps[2] =
      getConcatenatedIndexingMap(rewriter, newContractIndexingMaps[2]);

  linalg::GenericOp newContractOp = rewriter.create<linalg::GenericOp>(
      loc, newContractResultType, ValueRange{lhs, newContractRhs},
      newContractInit, newContractIndexingMaps, newContractIteratorTypes);
  rewriter.cloneRegionBefore(baseContractOp->getRegion(0),
                             newContractOp.getRegion(),
                             newContractOp.getRegion().begin());

  linalg::LinalgOp concatResultOp = newContractOp;
  if (fusionGroup.truncateOps) {
    SmallVector<Value> newTruncOperands;
    SmallVector<AffineMap> newTruncIndexingMaps;
    linalg::GenericOp baseTruncOp = fusionGroup.truncateOps->front();
    SmallVector<AffineMap> baseTruncOpIndexingMaps =
        baseTruncOp.getIndexingMapsArray();

    rewriter.setInsertionPoint(baseTruncOp);
    if (failed(moveOperandDefs(
            rewriter,
            ArrayRef<linalg::GenericOp>(fusionGroup.truncateOps.value()),
            baseTruncOp, dominanceInfo, fusionGroup.contractionOps))) {
      return baseTruncOp.emitOpError(
          "failed to move operand defs for truncate operations");
    }

    for (auto [operandIndex, baseTruncOperand, baseIndexingMap] :
         llvm::enumerate(baseTruncOp->getOperands(), baseTruncOpIndexingMaps)) {
      // Collect all the operands for the trunc operation.
      SmallVector<Value> truncOperands;
      for (auto truncOp : fusionGroup.truncateOps.value()) {
        truncOperands.push_back(truncOp.getOperand(operandIndex));
      }

      // Three cases to handle here.
      // Case 1. the operand is the contraction op.
      if (llvm::all_of(llvm::zip(truncOperands, fusionGroup.contractionOps),
                       [](auto it) {
                         Value operand = std::get<0>(it);
                         return operand.getDefiningOp<linalg::LinalgOp>() ==
                                std::get<1>(it);
                       })) {
        // Use the result of the concatanted generic op
        newTruncOperands.push_back(newContractOp.getResult(0));
        newTruncIndexingMaps.push_back(
            getConcatenatedIndexingMap(rewriter, baseIndexingMap));
        continue;
      }

      // Case 2. all the operands are the same.
      if (operandIndex < baseTruncOp.getNumDpsInputs() &&
          llvm::all_equal(truncOperands)) {
        newTruncOperands.push_back(truncOperands.front());
        newTruncIndexingMaps.push_back(getConcatenatedIndexingMap(
            rewriter, baseIndexingMap, /*shiftOnly=*/true));
        continue;
      }

      // Case 3. Concatenate all the operands.
      newTruncOperands.push_back(
          concatenateValues(rewriter, loc, truncOperands));
      newTruncIndexingMaps.push_back(
          getConcatenatedIndexingMap(rewriter, baseIndexingMap));
    }

    // Insert truncate operator.
    auto baseTruncType =
        cast<RankedTensorType>(baseTruncOp.getResult(0).getType());
    SmallVector<int64_t> newTruncShape = {
        static_cast<int64_t>(rhsValues.size())};
    newTruncShape.append(baseTruncType.getShape().begin(),
                         baseTruncType.getShape().end());
    auto newTruncType =
        RankedTensorType::get(newTruncShape, baseTruncType.getElementType());
    SmallVector<utils::IteratorType> newTruncIteratorTypes = {
        utils::IteratorType::parallel};
    newTruncIteratorTypes.append(baseTruncOp.getIteratorTypesArray());

    ArrayRef newTruncOperandsRef(newTruncOperands);
    linalg::GenericOp newTruncOp = rewriter.create<linalg::GenericOp>(
        loc, newTruncType,
        newTruncOperandsRef.take_front(baseTruncOp.getNumDpsInputs()),
        newTruncOperandsRef.take_back(baseTruncOp.getNumDpsInits()),
        newTruncIndexingMaps, newTruncIteratorTypes);

    rewriter.cloneRegionBefore(baseTruncOp->getRegion(0),
                               newTruncOp.getRegion(),
                               newTruncOp.getRegion().begin());

    concatResultOp = cast<linalg::LinalgOp>(newTruncOp.getOperation());
  }

  SmallVector<SmallVector<OpFoldResult>> concatResultShape;
  if (failed(concatResultOp.reifyResultShapes(rewriter, concatResultShape))) {
    return baseContractOp.emitOpError(
        "failed to get shape of concatenated result op");
  }
  Value concatResult = concatResultOp->getResult(0);
  MutableArrayRef<OpFoldResult> extractSizes(concatResultShape[0]);
  extractSizes[0] = rewriter.getIndexAttr(1);
  auto concatResultType = cast<RankedTensorType>(concatResult.getType());

  SmallVector<OpFoldResult> extractOffsets(extractSizes.size(),
                                           rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> extractStrides(extractSizes.size(),
                                           rewriter.getIndexAttr(1));

  auto concatResultTypeShape =
      llvm::map_to_vector(concatResultType.getShape(),
                          [](size_t s) { return static_cast<int64_t>(s); });
  auto resultOutType =
      RankedTensorType::get(ArrayRef(concatResultTypeShape).drop_front(),
                            concatResultType.getElementType());

  SmallVector<Value> replacements;
  for (auto i : llvm::seq<size_t>(0, rhsValues.size())) {
    extractOffsets[0] = rewriter.getIndexAttr(static_cast<int64_t>(i));
    replacements.push_back(rewriter.create<tensor::ExtractSliceOp>(
        loc, resultOutType, concatResult, extractOffsets, extractSizes,
        extractStrides));
  }

  for (auto [index, op, replacement] :
       llvm::enumerate(fusionGroup.contractionOps, replacements)) {
    Operation *replacedOp = op;
    if (fusionGroup.truncateOps) {
      replacedOp = fusionGroup.truncateOps.value()[index];
    }
    rewriter.replaceOp(replacedOp, replacement);
  }
  return success();
}

void FuseHorizontalContractionsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  DominanceInfo dominanceInfo(getOperation());

  SmallVector<HorizontalFusionGroup> horizontalFusionGroups;
  llvm::SmallDenseSet<linalg::LinalgOp> groupedOperations;

  getOperation()->walk([&](linalg::LinalgOp linalgOp) {
    if (!isEmptyFillContractionDAGRootOp(linalgOp)) {
      return;
    }
    // Avoid already grouped operations;
    if (groupedOperations.contains(linalgOp)) {
      return;
    }

    std::optional<HorizontalFusionGroup> fusionGroup =
        getHorizontalFusionGroupMembers(linalgOp, groupedOperations,
                                        dominanceInfo, fusionLimit);

    if (!fusionGroup) {
      return;
    }

    // Update statistics.
    numFusionGroups++;
    switch (fusionGroup->contractionOps.size()) {
    case 2:
      numSize2FusionGroups++;
      break;
    case 3:
      numSize3FusionGroups++;
      break;
    default:
      break;
    }

    groupedOperations.insert(fusionGroup->contractionOps.begin(),
                             fusionGroup->contractionOps.end());
    horizontalFusionGroups.emplace_back(std::move(fusionGroup.value()));
  });

  if (horizontalFusionGroups.empty()) {
    return;
  }

  IRRewriter rewriter(context);
  for (auto &fusionGroup : horizontalFusionGroups) {
    if (failed(fuseGroup(rewriter, fusionGroup, dominanceInfo))) {
      return signalPassFailure();
    }
  }

  {
    RewritePatternSet foldReshapePatterns(context);
    tensor::populateFoldTensorEmptyPatterns(foldReshapePatterns);
    linalg::FillOp::getCanonicalizationPatterns(foldReshapePatterns, context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(foldReshapePatterns)))) {
      getOperation()->emitOpError("failed during reshape folding patterns");
      return signalPassFailure();
    }

    RewritePatternSet foldPatterns(context);
    tensor::populateFoldTensorEmptyPatterns(foldPatterns);
    linalg::FillOp::getCanonicalizationPatterns(foldPatterns, context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(foldPatterns)))) {
      getOperation()->emitOpError("failed to fold empty/fill with concats");
      return signalPassFailure();
    }
  }

  // Note: Currently these patterns are required due to early lowering of
  // tensor.concat. When we choose  to move the lowering of tensor.concat later,
  // these patterns should be dropped.
  RewritePatternSet patterns(context);
  tensor::populateDecomposeTensorConcatPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::DispatchCreation
