// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/FusionUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

static bool operator==(const linalg::ContractionDimensions &lhs,
                       const linalg::ContractionDimensions &rhs) {
  return lhs.batch == rhs.batch && lhs.m == rhs.m && lhs.n == rhs.n &&
         lhs.k == rhs.k;
}
static bool operator!=(const linalg::ContractionDimensions &lhs,
                       const linalg::ContractionDimensions &rhs) {
  return !(lhs == rhs);
}

struct FuseHorizontalContractionsPass final
    : public impl::FuseHorizontalContractionsPassBase<
          FuseHorizontalContractionsPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

/// For indexing maps of Linalg ops passed in as `indexingMaps`, permute them
/// such that `seedLhsIndexingMap` is same as `indexingMaps[0]`. Returns the
/// permutation of the iteration space of the RHS. Returns failure if the
/// permutation of `iteratorTypes` results in a change of `iteratorTypes`. This
/// is so that permutation doesnt change the position of reduction iterator
/// type.
static std::optional<SmallVector<int64_t>>
permuteIndexingMapsToMatchSeedLhs(MLIRContext *context,
                                  AffineMap seedLhsIndexingMap,
                                  ArrayRef<utils::IteratorType> iteratorTypes,
                                  SmallVector<AffineMap> &indexingMaps) {
  if (indexingMaps.empty()) {
    return std::nullopt;
  }
  AffineMap lhsIndexingMap = indexingMaps[0];
  if (seedLhsIndexingMap == lhsIndexingMap) {
    return llvm::to_vector(llvm::seq<int64_t>(0, lhsIndexingMap.getNumDims()));
  }

  assert(lhsIndexingMap.getNumDims() == seedLhsIndexingMap.getNumDims());
  if (!lhsIndexingMap.isProjectedPermutation() ||
      !seedLhsIndexingMap.isProjectedPermutation() ||
      lhsIndexingMap.getNumResults() != seedLhsIndexingMap.getNumResults()) {
    return std::nullopt;
  }

  auto getResultDimsRange = [](ArrayRef<AffineExpr> exprs) {
    return llvm::map_range(exprs, [](AffineExpr expr) {
      return cast<AffineDimExpr>(expr).getPosition();
    });
  };
  auto seedLhsResultDimsRange =
      getResultDimsRange(seedLhsIndexingMap.getResults());
  auto lhsResultDimsRange = getResultDimsRange(lhsIndexingMap.getResults());

  // Start with an identity permutations. For now try to only swap dimensions
  // which is not a general solution.
  SmallVector<int64_t> interchangeVector =
      llvm::to_vector(llvm::seq<int64_t>(0, lhsIndexingMap.getNumDims()));
  for (auto [seedDimPos, lhsDimPos] :
       llvm::zip_equal(seedLhsResultDimsRange, lhsResultDimsRange)) {
    if (seedDimPos == lhsDimPos) {
      continue;
    }
    // If the current positions are what we started with, swap the positions.
    if (interchangeVector[lhsDimPos] == lhsDimPos &&
        interchangeVector[seedDimPos] == seedDimPos) {
      std::swap(interchangeVector[lhsDimPos], interchangeVector[seedDimPos]);
      continue;
    }
    // If this was a changed dimension, check that it is consistent.
    if (interchangeVector[lhsDimPos] != seedDimPos ||
        interchangeVector[seedDimPos] != lhsDimPos) {
      return std::nullopt;
    }
  }

  // Check that the iterator types remain the same
  SmallVector<utils::IteratorType> permutedIteratorTypes =
      llvm::to_vector(iteratorTypes);
  applyPermutationToVector(permutedIteratorTypes, interchangeVector);
  if (permutedIteratorTypes != iteratorTypes) {
    return std::nullopt;
  }

  AffineMap interchangeMap =
      AffineMap::getPermutationMap(interchangeVector, context);
  for (auto &map : indexingMaps) {
    if (!map.isEmpty()) {
      map = map.compose(interchangeMap);
    }
  }
  return interchangeVector;
}

/// Helper method to check operations equivalence
static bool checkContractionOpEquivalence(MLIRContext *context, Operation *aOp,
                                          Operation *bOp) {
  auto aLinalgOp = dyn_cast<linalg::LinalgOp>(aOp);
  auto bLinalgOp = dyn_cast<linalg::LinalgOp>(bOp);

  if (!aLinalgOp || !bLinalgOp) {
    return false;
  }
  // Contraction ops verifies that there are two operands and one result.
  assert(linalg::isaContractionOpInterface(aLinalgOp) &&
         linalg::isaContractionOpInterface(bLinalgOp) &&
         "expected lhs and rhs to be contraction ops");

  // Check that the LHS operand is the same.
  if (aLinalgOp.getDpsInputOperand(0)->get() !=
      bLinalgOp.getDpsInputOperand(0)->get()) {
    return false;
  }

  // Check that the n-dimensions are the same
  SmallVector<AffineMap> aIndexingMaps = aLinalgOp.getIndexingMapsArray();
  SmallVector<AffineMap> bIndexingMaps = bLinalgOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> aIteratorTypes =
      aLinalgOp.getIteratorTypesArray();
  SmallVector<utils::IteratorType> bIteratorTypes =
      bLinalgOp.getIteratorTypesArray();
  std::optional<SmallVector<int64_t>> bPermutationVector;
  if (aIndexingMaps[0] != bIndexingMaps[0]) {
    bPermutationVector = permuteIndexingMapsToMatchSeedLhs(
        context, aIndexingMaps[0], bIteratorTypes, bIndexingMaps);
    if (!bPermutationVector) {
      return false;
    }
  }

  FailureOr<linalg::ContractionDimensions> aContractionDims =
      linalg::inferContractionDims(aIndexingMaps);
  FailureOr<linalg::ContractionDimensions> bContactionDims =
      linalg::inferContractionDims(bIndexingMaps);
  if (failed(aContractionDims) || failed(bContactionDims)) {
    return false;
  }
  if (aContractionDims.value() != bContactionDims.value()) {
    return false;
  }

  SmallVector<int64_t> aStaticDims = aLinalgOp.getStaticLoopRanges();
  SmallVector<int64_t> bStaticDims = bLinalgOp.getStaticLoopRanges();
  if (bPermutationVector) {
    applyPermutationToVector(bStaticDims, bPermutationVector.value());
  }
  for (auto nDim : aContractionDims->n) {
    if (aStaticDims[nDim] != bStaticDims[nDim] ||
        ShapedType::isDynamic(aStaticDims[nDim])) {
      return false;
    }
  }

  auto checkSameRankAndElementType = [](Value aVal, Value bVal) {
    auto aType = dyn_cast<ShapedType>(aVal.getType());
    auto bType = dyn_cast<ShapedType>(bVal.getType());
    return aType && bType && aType.getRank() == bType.getRank() &&
           aType.getElementType() == bType.getElementType();
  };
  // Check that the RHS rank and element type are the same. We dont check the
  // type cause we allow RHS to be transposes.
  if (!checkSameRankAndElementType(aLinalgOp.getDpsInputOperand(1)->get(),
                                   bLinalgOp.getDpsInputOperand(1)->get())) {
    return false;
  }

  // Check that the output rank and element type are the same. We dont check the
  // type cause we allow output to be transposes.
  if (!checkSameRankAndElementType(aLinalgOp.getDpsInitOperand(0)->get(),
                                   bLinalgOp.getDpsInitOperand(0)->get())) {
    return false;
  }

  // Check that the iterator types are the same.
  if (aLinalgOp.getIteratorTypesArray() != bLinalgOp.getIteratorTypesArray()) {
    return false;
  }

  // Check region equivalence.
  if (!OperationEquivalence::isRegionEquivalentTo(
          &aLinalgOp->getRegion(0), &bLinalgOp->getRegion(0),
          OperationEquivalence::IgnoreLocations)) {
    return false;
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
  options.inclusive = true;
  // Limit the slice to the seed to make sure the slice is small.
  options.filter = [&](Operation *op) {
    return !dominanceInfo.properlyDominates(op, seedOp);
  };
  llvm::SetVector<Operation *> slice;
  [[maybe_unused]] LogicalResult result = getBackwardSlice(op, &slice, options);
  assert(result.succeeded());
  return !llvm::any_of(currGroup, [&](Operation *groupedOp) {
    return slice.contains(groupedOp);
  });
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
/// Note: The actual operation generated does not concat the RHS.
static std::optional<SmallVector<Operation *>> getHorizontalFusionGroupMembers(
    MLIRContext *context, linalg::LinalgOp seedOp,
    const llvm::SmallDenseSet<Operation *> &groupedOperations,
    const DominanceInfo &dominanceInfo, int fusionLimit) {

  Value lhs = seedOp->getOperand(0);

  SetVector<Operation *> allOps;
  SmallVector<Operation *> contractionOps = {seedOp};
  allOps.insert(seedOp);

  auto canBeGrouped = [&](linalg::LinalgOp linalgOp) -> bool {
    if (linalgOp->getParentOp() != seedOp->getParentOp()) {
      return false;
    }

    // Constraints of the operation itself.
    if (!linalg::isaContractionOpInterface(linalgOp) ||
        !checkContractionOpEquivalence(context, linalgOp, seedOp)) {
      return false;
    }
    if (groupedOperations.contains(linalgOp)) {
      return false;
    }

    // Structural constraints related to being able to fuse the operations.
    if (!dominanceInfo.properlyDominates(seedOp, linalgOp)) {
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
  llvm::sort(lhsUsers, [&](Operation *a, Operation *b) {
    return dominanceInfo.properlyDominates(a, b);
  });

  // Collect all contraction op users of lhs.
  for (Operation *lhsUser : lhsUsers) {
    auto linalgUser = dyn_cast<linalg::LinalgOp>(lhsUser);
    if (!linalgUser) {
      continue;
    }

    if (!isHorizontalToGroup(linalgUser, allOps, dominanceInfo, seedOp)) {
      continue;
    }

    contractionOps.push_back(linalgUser);
    allOps.insert(linalgUser);
    if (contractionOps.size() >= fusionLimit) {
      break;
    }
  }

  if (contractionOps.size() == 1) {
    return std::nullopt;
  }

  return contractionOps;
}

/// Generate the horizontally fused operation as an operation with multiple
/// results, corresponding to the results of the fused operations. It is assumed
/// that the LHS of the contraction operations fused horizontally is the same
/// and have the same indexing map for all the operations. The RHS/outputs of
/// the operations can be different, but share the same iteration space.
/// Returns the generated fused op, or `std::nullopt` when the fused op
/// could not be generated.
static std::optional<linalg::GenericOp>
fuseContractionsHorizontally(RewriterBase &rewriter, Location loc,
                             MutableArrayRef<Operation *> linalgOps) {
  if (linalgOps.empty()) {
    return std::nullopt;
  }

  SmallVector<Value> fusedIns;
  SmallVector<Value> fusedOuts;
  SmallVector<Type> fusedResultTypes;
  SmallVector<AffineMap> fusedInsIndexingMaps;
  SmallVector<AffineMap> fusedOutsIndexingMaps;

  auto seedOp = cast<linalg::LinalgOp>(linalgOps.front());
  SmallVector<utils::IteratorType> fusedIteratorTypes =
      seedOp.getIteratorTypesArray();

  OpOperand *seedOpLhs = seedOp.getDpsInputOperand(0);
  AffineMap seedOpLhsIndexingMap = seedOp.getMatchingIndexingMap(seedOpLhs);
  fusedIns.push_back(seedOpLhs->get());
  fusedInsIndexingMaps.push_back(seedOpLhsIndexingMap);

  llvm::SmallDenseSet<Operation *> droppedOps;
  for (auto op : linalgOps) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp ||
        linalgOp.getDpsInputOperand(0)->get() != seedOpLhs->get()) {
      droppedOps.insert(op);
      continue;
    }

    SmallVector<AffineMap> opIndexingMaps = linalgOp.getIndexingMapsArray();
    if (!permuteIndexingMapsToMatchSeedLhs(
            rewriter.getContext(), seedOpLhsIndexingMap, fusedIteratorTypes,
            opIndexingMaps)) {
      droppedOps.insert(op);
      continue;
    }

    // Append the RHS operands.
    SmallVector<OpOperand *> ins = linalgOp.getDpsInputOperands();
    llvm::append_range(
        fusedIns,
        llvm::map_range(ArrayRef<OpOperand *>(ins).drop_front(),
                        [](OpOperand *operand) { return operand->get(); }));

    // Append the Outs operands.
    llvm::append_range(fusedOuts, llvm::map_range(linalgOp.getDpsInitsMutable(),
                                                  [](OpOperand &operand) {
                                                    return operand.get();
                                                  }));

    // Append the result types.
    fusedResultTypes.append(linalgOp->result_type_begin(),
                            linalgOp->result_type_end());

    // Append the rhs indexing maps.
    llvm::append_range(fusedInsIndexingMaps,
                       ArrayRef<AffineMap>(opIndexingMaps)
                           .slice(1, linalgOp.getNumDpsInputs() - 1));

    // Append the outs indexing maps.
    llvm::append_range(fusedOutsIndexingMaps,
                       ArrayRef<AffineMap>(opIndexingMaps)
                           .drop_front(linalgOp.getNumDpsInputs()));
  }

  SmallVector<AffineMap> fusedIndexingMaps = std::move(fusedInsIndexingMaps);
  fusedIndexingMaps.append(fusedOutsIndexingMaps);
  auto fusedOp = rewriter.create<linalg::GenericOp>(
      loc, fusedResultTypes, fusedIns, fusedOuts, fusedIndexingMaps,
      fusedIteratorTypes, [](OpBuilder &, Location, ValueRange) {});

  Block *fusedBody = fusedOp.getBlock();
  int64_t rhsIndex = 0;
  int64_t outsIndex = fusedOp.getNumDpsInputs();
  SmallVector<Value> yieldVals;
  for (auto op : linalgOps) {
    if (droppedOps.contains(op)) {
      continue;
    }
    auto linalgOp = cast<linalg::LinalgOp>(op);
    Block *body = linalgOp.getBlock();
    SmallVector<Value> replacements = {fusedBody->getArgument(0)};
    llvm::append_range(
        replacements,
        llvm::map_range(fusedBody->getArguments().slice(
                            rhsIndex + 1, linalgOp.getNumDpsInputs() - 1),
                        [](BlockArgument arg) -> Value { return arg; }));

    llvm::append_range(
        replacements,
        llvm::map_range(fusedBody->getArguments().slice(
                            outsIndex, linalgOp.getNumDpsInits()),
                        [](BlockArgument arg) -> Value { return arg; }));

    rewriter.mergeBlocks(body, fusedBody, replacements);
    rhsIndex += linalgOp.getNumDpsInputs() - 1;
    outsIndex += linalgOp.getNumDpsInits();

    auto yieldOp = cast<linalg::YieldOp>(fusedBody->getTerminator());
    yieldVals.append(yieldOp->operand_begin(), yieldOp->operand_end());
    rewriter.eraseOp(yieldOp);
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToEnd(fusedBody);
  rewriter.create<linalg::YieldOp>(loc, yieldVals);

  unsigned resultsIndex = 0;
  for (auto linalgOp : linalgOps) {
    unsigned numResults = linalgOp->getNumResults();
    rewriter.replaceOp(linalgOp,
                       fusedOp->getResults().slice(resultsIndex, numResults));
    resultsIndex += numResults;
  }

  return fusedOp;
}

static void fuseGroup(RewriterBase &rewriter,
                      MutableArrayRef<Operation *> fusionGroup,
                      DominanceInfo &dominanceInfo) {
  if (!llvm::all_of(fusionGroup, [](Operation *op) {
        return isa_and_nonnull<linalg::LinalgOp>(op);
      })) {
    return;
  }
  auto baseContractOp = cast<linalg::LinalgOp>(fusionGroup.front());
  Location loc = baseContractOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(baseContractOp);

  if (failed(moveOperandDefs(rewriter, fusionGroup, baseContractOp,
                             dominanceInfo))) {
    return;
  }

  std::optional<linalg::GenericOp> fusedOp =
      fuseContractionsHorizontally(rewriter, loc, fusionGroup);
  (void)fusedOp;
}

void FuseHorizontalContractionsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  DominanceInfo dominanceInfo(getOperation());

  SmallVector<SmallVector<Operation *>> horizontalFusionGroups;
  llvm::SmallDenseSet<Operation *> groupedOperations;

  getOperation()->walk([&](linalg::LinalgOp linalgOp) {
    if (!linalg::isaContractionOpInterface(linalgOp)) {
      return;
    }
    // Avoid already grouped operations;
    if (groupedOperations.contains(linalgOp)) {
      return;
    }

    std::optional<SmallVector<Operation *>> fusionGroup =
        getHorizontalFusionGroupMembers(context, linalgOp, groupedOperations,
                                        dominanceInfo, fusionLimit);

    if (!fusionGroup) {
      return;
    }

    // Update statistics.
    numFusionGroups++;
    switch (fusionGroup->size()) {
    case 2:
      numSize2FusionGroups++;
      break;
    case 3:
      numSize3FusionGroups++;
      break;
    default:
      break;
    }

    groupedOperations.insert(fusionGroup->begin(), fusionGroup->end());
    horizontalFusionGroups.emplace_back(std::move(fusionGroup.value()));
  });

  if (horizontalFusionGroups.empty()) {
    return;
  }

  IRRewriter rewriter(context);
  for (auto &fusionGroup : horizontalFusionGroups) {
    fuseGroup(rewriter, fusionGroup, dominanceInfo);
  }
}
} // namespace mlir::iree_compiler::DispatchCreation
