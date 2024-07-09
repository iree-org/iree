// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-fuse-horizontal-contractions"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_FUSEHORIZONTALCONTRACTIONSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

#define GEN_PASS_DEF_FUSEHORIZONTALCONTRACTIONSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

struct FuseHorizontalContractionsPass
    : public impl::FuseHorizontalContractionsPassBase<
          FuseHorizontalContractionsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect>();
  }
  FuseHorizontalContractionsPass() {}
  FuseHorizontalContractionsPass(const FuseHorizontalContractionsPass &pass)
      : FuseHorizontalContractionsPass() {}

  void runOnOperation() override;
};

} // namespace

/// Structs that captures the ops that are to be fused
struct HorizontalFusionGroup {
  // Contractions op that are to be fused.
  SmallVector<linalg::LinalgOp> contractionOps;
  // Optional truncate operations that could be following the contraction op.
  std::optional<SmallVector<linalg::GenericOp>> truncateOps;
  // Operation that dominates all the ops of the group.
  Operation *dominatedByAll;
};

/// Check that an operation is a `empty -> fill -> contraction`
static bool isEmptyFillContractionDAGRootOp(linalg::LinalgOp linalgOp) {
  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return false;
  }
  auto fillOp = linalgOp.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fillOp) {
    return false;
  }
  // For convenience check that the fill value is 0. This is not
  // a necessity, but easier to handle the rewrite this way.
  if (!matchPattern(fillOp.getDpsInputOperand(0)->get(), m_AnyZeroFloat())) {
    return false;
  }
  return fillOp.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>();
}

/// Get user of operation that is a truncate operation.
static std::optional<linalg::GenericOp> getTruncFUser(Operation *op) {
  if (op->getNumResults() != 1) {
    return std::nullopt;
  }
  Value result = op->getResult(0);
  if (!result.hasOneUse()) {
    return std::nullopt;
  }
  Operation *user = *result.user_begin();
  auto genericOp = dyn_cast<linalg::GenericOp>(user);
  if (!genericOp) {
    return std::nullopt;
  }
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return std::nullopt;
  }
  if (llvm::any_of(genericOp.getIndexingMapsArray(),
                   [](AffineMap map) { return !map.isIdentity(); })) {
    return std::nullopt;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return std::nullopt;
  }
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  auto yieldDefOp = yieldOp->getOperand(0).getDefiningOp<arith::TruncFOp>();
  if (!yieldDefOp) {
    return std::nullopt;
  }
  auto arg = dyn_cast<BlockArgument>(yieldDefOp->getOperand(0));
  if (!arg || arg.getParentBlock() != genericOp.getBody() ||
      arg.getArgNumber() != 0) {
    return std::nullopt;
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
    const DominanceInfo &dominanceInfo) {
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
  std::optional<linalg::GenericOp> truncOp = getTruncFUser(seedOp);
  std::optional<SmallVector<linalg::GenericOp>> truncateOps;
  if (truncOp) {
    truncateOps = {truncOp.value()};
  }
  allOps.insert(seedOp);
  if (truncOp) {
    allOps.insert(truncOp.value());
  }

  auto canBeGrouped = [&](linalg::LinalgOp linalgOp) -> bool {
    if (linalgOp->getParentOp() != seedOp->getParentOp()) {
      return false;
    }
    // The seed has to dominate the op.
    if (!dominanceInfo.properlyDominates(seedOp, linalgOp)) {
      return false;
    }
    if (!isEmptyFillContractionDAGRootOp(linalgOp)) {
      return false;
    }
    if (groupedOperations.contains(linalgOp) || allOps.contains(linalgOp)) {
      return false;
    }
    if (linalgOp->getOperand(0).getType() != lhsType ||
        linalgOp->getOperand(1).getType() != rhsType ||
        linalgOp->getOperand(2).getType() != outType) {
      return false;
    }
    // To not move around the code too much check that the new op
    // dominates all users of other ops.
    for (auto op : allOps) {
      for (auto user : op->getUsers()) {
        if (allOps.contains(user))
          continue;
        if (!dominanceInfo.properlyDominates(linalgOp, user)) {
          return false;
        }
      }
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

    std::optional<linalg::GenericOp> userTruncOp = getTruncFUser(linalgUser);
    if (truncateOps && !userTruncOp) {
      continue;
    }

    contractionOps.push_back(linalgUser);
    allOps.insert(linalgUser);
    if (truncateOps) {
      truncateOps.value().push_back(userTruncOp.value());
      allOps.insert(userTruncOp.value());
    }
  }

  if (contractionOps.size() == 1) {
    return std::nullopt;
  }

  return HorizontalFusionGroup{contractionOps, truncateOps, seedOp};
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
/// tensor<1x?x?xf32> %arg2_r = tensor.expand_shape %arg2 [[0, 1], ...] :
/// tensor<?x?xf32> to tensor<1x?x?xf32> %arg3_r = tensor.expand_shape %arg3
/// [[0, 1], ...] : tensor<?x?xf32> to tensor<1x?x?xf32> %rhs = tensor.concat
/// (%arg1_r, %arg2_r, %arg3_r) %fused = linalg.generic {
///     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
///                      affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
///                      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>}],
///     iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
///   ins(%arg0, %rhs) ... { ... }
/// %0 = tensor.extract_slice %fused [0, 0, 0] ... : tensor<1x?x?xf32> to
/// tensor<?x?xf32> %1 = tensor.extract_slice %fused [1, 0, 0] ... :
/// tensor<1x?x?xf32> to tensor<?x?xf32> %2 = tensor.extract_slice %fused [2, 0,
/// 0] ... : tensor<1x?x?xf32> to tensor<?x?xf32>
/// ```
///
/// Also accounts for quantized cases where inputs are at lower precision and
/// accumulate is in higher-precision with truncate getting back to the
/// quantized sizes.
static LogicalResult fuseGroup(RewriterBase &rewriter,
                               HorizontalFusionGroup &fusionGroup) {
  rewriter.setInsertionPoint(fusionGroup.dominatedByAll);

  linalg::LinalgOp base = fusionGroup.contractionOps.front();
  Location loc = base.getLoc();
  auto rhsType = cast<RankedTensorType>(base->getOperand(1).getType());
  auto outType = cast<RankedTensorType>(base->getResult(0).getType());
  std::optional<linalg::GenericOp> baseTruncOp;
  if (fusionGroup.truncateOps) {
    baseTruncOp = fusionGroup.truncateOps->front();
  }

  SmallVector<ReassociationIndices> reassoc;
  reassoc.push_back({0, 1});
  for (int i = 0, e = rhsType.getRank() - 1; i < e; ++i) {
    reassoc.push_back({i + 2});
  }

  SmallVector<int64_t> rhsNewShape(rhsType.getShape());
  rhsNewShape.insert(rhsNewShape.begin(), 1);
  auto concatRhsType =
      RankedTensorType::get(rhsNewShape, rhsType.getElementType());

  SmallVector<Value> rhsVals;
  for (auto op : fusionGroup.contractionOps) {
    Value thisRhs = op.getDpsInputOperand(1)->get();
    Value expanded = rewriter.create<tensor::ExpandShapeOp>(loc, concatRhsType,
                                                            thisRhs, reassoc);
    rhsVals.push_back(expanded);
  }

  Value newRhs = rewriter.create<tensor::ConcatOp>(loc, /*dim=*/0, rhsVals);

  SmallVector<int64_t> newShape(outType.getShape());
  newShape.insert(newShape.begin(), rhsVals.size());
  auto concatOutType =
      RankedTensorType::get(newShape, outType.getElementType());

  Value baseOut = base.getDpsInitOperand(0)->get();
  auto origFill = baseOut.getDefiningOp<linalg::FillOp>();
  if (!origFill) {
    // TODO: This should be avoidable if we just concatenate the fill operands
    // and add a folder for `fill -> concat`.
    return base.emitOpError("expected outs operand to be a fill op");
  }
  auto origEmpty =
      origFill.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>();
  if (!origEmpty) {
    return base.emitOpError("expected fill outs operand to be a tensor.empty");
  }

  auto newEmpty = rewriter.create<tensor::EmptyOp>(loc, concatOutType,
                                                   origEmpty.getDynamicSizes());
  auto newFill = rewriter.create<linalg::FillOp>(
      loc, concatOutType, origFill.getDpsInputOperand(0)->get(),
      newEmpty.getResult());

  Value lhs = base->getOperand(0);

  SmallVector<AffineMap> indexingMaps = base.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iteratorTypes = base.getIteratorTypesArray();
  iteratorTypes.insert(iteratorTypes.begin(), utils::IteratorType::parallel);

  indexingMaps[0] = indexingMaps[0].shiftDims(1);
  indexingMaps[1] = indexingMaps[1].shiftDims(1).insertResult(
      rewriter.getAffineDimExpr(0), 0);
  indexingMaps[2] = indexingMaps[2].shiftDims(1).insertResult(
      rewriter.getAffineDimExpr(0), 0);

  linalg::GenericOp newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, concatOutType, ValueRange{lhs, newRhs}, newFill.getResult(0),
      indexingMaps, iteratorTypes);
  rewriter.cloneRegionBefore(base->getRegion(0), newGenericOp.getRegion(),
                             newGenericOp.getRegion().begin());

  Value fusedResult = newGenericOp.getResult(0);
  if (fusionGroup.truncateOps) {
    // Insert truncate operator.
    auto truncType =
        cast<RankedTensorType>(baseTruncOp.value()->getResult(0).getType());
    auto concatTruncType =
        RankedTensorType::get(newShape, truncType.getElementType());
    size_t concatTruncRank = concatTruncType.getRank();
    Value truncOuts = rewriter.create<tensor::EmptyOp>(
        loc, concatTruncType, origEmpty.getDynamicSizes());
    SmallVector<AffineMap> truncIndexingMaps(
        2, rewriter.getMultiDimIdentityMap(concatTruncRank));
    SmallVector<utils::IteratorType> truncIteratorTypes(
        concatTruncRank, utils::IteratorType::parallel);
    auto truncateOpBody = [&](OpBuilder &b, Location loc, ValueRange args) {
      Value truncOp = b.create<arith::TruncFOp>(
          loc, concatTruncType.getElementType(), args[0]);
      rewriter.create<linalg::YieldOp>(loc, truncOp);
    };
    auto truncateOp = rewriter.create<linalg::GenericOp>(
        loc, concatTruncType, newGenericOp->getResults(), truncOuts,
        truncIndexingMaps, truncIteratorTypes, truncateOpBody);

    fusedResult = truncateOp.getResult(0);
  }

  SmallVector<Value> newOuts;

  auto fusedResultType = cast<RankedTensorType>(fusedResult.getType());
  SmallVector<int64_t> shape = llvm::to_vector(fusedResultType.getShape());

  SmallVector<OpFoldResult> sizes = newEmpty.getMixedSizes();
  sizes[0] = rewriter.getIndexAttr(1);

  SmallVector<OpFoldResult> offsets(fusedResultType.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(fusedResultType.getRank(),
                                    rewriter.getIndexAttr(1));
  auto resultOutType = RankedTensorType::get(outType.getShape(),
                                             fusedResultType.getElementType());
  for (auto i : llvm::seq<int>(0, rhsVals.size())) {
    offsets[0] = rewriter.getIndexAttr(i);
    newOuts.push_back(rewriter.create<tensor::ExtractSliceOp>(
        loc, resultOutType, fusedResult, offsets, sizes, strides));
  }

  for (auto [index, op, replacement] :
       llvm::enumerate(fusionGroup.contractionOps, newOuts)) {
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
                                        dominanceInfo);

    if (!fusionGroup) {
      return;
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
    if (failed(fuseGroup(rewriter, fusionGroup))) {
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
} // namespace mlir::iree_compiler::Flow
