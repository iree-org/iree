// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-collapse-dimensions"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_COLLAPSEDIMENSIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

using IREE::LinalgExt::LinalgFusionOpInterface;

namespace {
/// Pass declaration.
struct CollapseDimensionsPass final
    : public impl::CollapseDimensionsPassBase<CollapseDimensionsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===---------------------------------------------------------------------===//
// Helper functions
//===---------------------------------------------------------------------===//

/// Searches the same sequence in all the affine maps and collapses these
/// dimensions. It only applies these to "parallel" loops without mixing them
/// with "reduction" types. It is expected that the `op` has projected
/// permutations only as indexing maps. (Checked using `isEligibleForCollapse`).
static SmallVector<ReassociationIndices> getCollapsibleLoops(Operation *op) {
  auto fusionInterfaceOp = llvm::cast<LinalgFusionOpInterface>(op);
  auto tilingInterfaceOp = llvm::cast<TilingInterface>(op);

  SmallVector<ReassociationIndices> contiguousLoops;
  SmallVector<unsigned> pDims, rDims;
  findPositionsOfType(tilingInterfaceOp.getLoopIteratorTypes(),
                      utils::IteratorType::parallel, pDims);
  findPositionsOfType(tilingInterfaceOp.getLoopIteratorTypes(),
                      utils::IteratorType::reduction, rDims);
  llvm::SmallDenseSet<unsigned> pDimsSet, rDimsSet;
  pDimsSet.insert(pDims.begin(), pDims.end());
  rDimsSet.insert(rDims.begin(), rDims.end());

  auto hasAllMapsSameSequence = [&](AffineExpr preExpr, AffineExpr nextExpr) {
    // Check that all indexing maps of the `op`
    // - Either both `preExpr` and `nextExpr` contiguous, or
    // - are missing in
    // Then `preExpr` and `nextExpr` can be collapsed.
    for (AffineMap map : fusionInterfaceOp.getIndexingMapsArray()) {
      // If map has no results, no need to check.
      if (map.getNumResults() == 0) {
        continue;
      }
      for (auto [index, resultExpr] : llvm::enumerate(map.getResults())) {
        // If we find the preExpr, we should find the nextExpr.
        if (resultExpr == preExpr) {
          if (index == map.getNumResults() - 1) {
            // Reached end of list. Return false;
            return false;
          }
          if (map.getResult(index + 1) != nextExpr) {
            return false;
          }
        }
        // If we find nextExpr the previous one should be `prevExpr`.
        // This is redundant check for the most part, but is cheap enough, so
        // #YOLO
        if (resultExpr == nextExpr) {
          if (index == 0) {
            // match at beginning of the list. Return false;
            return false;
          }
          if (map.getResult(index - 1) != preExpr) {
            return false;
          }
        }
      }
    }
    return true;
  };
  auto hasSameIteratorType = [&](AffineExpr preExpr, AffineExpr nextExpr) {
    unsigned prePos = cast<AffineDimExpr>(preExpr).getPosition();
    unsigned nextPos = cast<AffineDimExpr>(nextExpr).getPosition();
    return (pDimsSet.count(prePos) && pDimsSet.count(nextPos)) ||
           (rDimsSet.count(prePos) && rDimsSet.count(nextPos));
  };

  // Find the largest sequence of dimensions that are
  // - Either preserved in all maps, or
  // - are completely absent
  // This sequence can be collapsed. To find the sequence,
  // 1) For each indexing map, take the result expressions
  // 2) Find a sequence of 2 that is found in all maps (or absent)
  // 3) Then take last element of this sequence and the next
  //    result expression, and check if this sequence of 2 is
  //    found in all maps. If so, add to sequence (to get a sequence of 3)
  //    and repeat till the last element of sequence and the next result
  //    expression is not found as a sequence in all maps.

  llvm::SmallSetVector<unsigned, 8> seenLoops;
  for (auto map : fusionInterfaceOp.getIndexingMapsArray()) {
    ReassociationIndices range;
    AffineExpr preExpr;

    auto appendAndClearRange = [&]() {
      if (range.size() > 1) {
        contiguousLoops.push_back(range);
      }
      range.clear();
    };

    for (auto nextExpr : map.getResults()) {
      unsigned position = cast<AffineDimExpr>(nextExpr).getPosition();
      if (seenLoops.contains(position)) {
        appendAndClearRange();
        continue;
      }
      if (!hasAllMapsSameSequence(preExpr, nextExpr) ||
          !hasSameIteratorType(preExpr, nextExpr)) {
        appendAndClearRange();
      }
      range.push_back(position);
      seenLoops.insert(position);
      preExpr = nextExpr;
    }
    appendAndClearRange();
  }

  return contiguousLoops;
}

/// Returns true if the given op is collapsable.
static bool isEligibleForCollapse(Operation *op) {
  if (isa<IREE::LinalgExt::AttentionOp, linalg::FillOp>(op)) {
    return true;
  }

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp) {
    return false;
  }

  auto hasEncoding = [](Type type) -> bool {
    auto rankedTensorType = dyn_cast<RankedTensorType>(type);
    return rankedTensorType && rankedTensorType.getEncoding();
  };
  if (llvm::any_of(op->getOperandTypes(), hasEncoding)) {
    return false;
  }

  // TODO(guray) Currently we can only collapse when result of all the
  // AffineMaps are dimensions. Possible to collapse cases like
  // affine_map<d0, d1+d2> with affine_map<d0, d1+d2>, however, this is not
  // supported in collapsing mechanism in MLIR. Once we have this support,
  // we can remove this if statement.
  if (llvm::any_of(genericOp.getIndexingMapsArray(), [](AffineMap map) {
        return !map.isProjectedPermutation();
      })) {
    return false;
  }

  return true;
}

// For the `operand`, get of producer loop -> consumer loop.
static FailureOr<AffineMap>
getProducerLoopToConsumerLoopsMap(OpOperand &operand) {
  auto consumer =
      dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(operand.getOwner());
  if (!consumer) {
    return failure();
  }
  auto producer =
      dyn_cast_or_null<LinalgFusionOpInterface>(operand.get().getDefiningOp());
  if (!producer) {
    return failure();
  }

  AffineMap consumerOperandMap = consumer.getMatchingIndexingMap(&operand);
  if (!consumerOperandMap.isProjectedPermutation()) {
    return failure();
  }

  AffineMap producerResultMap =
      producer.getIndexingMapMatchingResult(cast<OpResult>(operand.get()));
  if (!producerResultMap.isProjectedPermutation()) {
    return failure();
  }

  AffineMap inverseProducerResultMap =
      inverseAndBroadcastProjectedPermutation(producerResultMap);
  if (!inverseProducerResultMap) {
    return failure();
  }

  AffineMap producerLoopToConsumerLoop =
      inverseProducerResultMap.compose(consumerOperandMap);
  return producerLoopToConsumerLoop;
}

static FailureOr<AffineMap>
getConsumerLoopToProducerLoopsMap(OpOperand &operand) {
  auto consumer = dyn_cast<LinalgFusionOpInterface>(operand.getOwner());
  if (!consumer) {
    return failure();
  }
  auto producer =
      dyn_cast_or_null<LinalgFusionOpInterface>(operand.get().getDefiningOp());
  if (!producer) {
    return failure();
  }

  AffineMap consumerOperandMap = consumer.getMatchingIndexingMap(&operand);
  if (!consumerOperandMap.isProjectedPermutation()) {
    return failure();
  }

  AffineMap producerResultMap =
      producer.getIndexingMapMatchingResult(cast<OpResult>(operand.get()));
  if (!producerResultMap.isProjectedPermutation()) {
    return failure();
  }

  AffineMap inverseConsumerOperandMap =
      inverseAndBroadcastProjectedPermutation(consumerOperandMap);
  if (!inverseConsumerOperandMap) {
    return failure();
  }

  AffineMap consumerLoopToProducerLoop =
      inverseConsumerOperandMap.compose(producerResultMap);
  return consumerLoopToProducerLoop;
}

//===---------------------------------------------------------------------===//
// CollapseInfo
//===---------------------------------------------------------------------===//

namespace {
class CollapseInfo {
public:
  using CollapsableLoopsSet = llvm::SmallSetVector<int64_t, 8>;

  CollapseInfo() = default;
  CollapseInfo(Operation *op) {
    reassociation = DispatchCreation::getCollapsibleLoops(op);
    collapsableLoops = getCollapsedFromReassociation(reassociation);
  }

  // Print the current operation & reassociation indicies
  void print(raw_ostream &os) const;

  // Debug print the current operation & reassociation indicies
  void dump() const;

  // Update CollapseInfo to ensure that all dimensions collapsable in `this` are
  // also collapsable in `otherInfo`. This means:
  // 1. Any dimension not collapsable in `otherInfo` should not be
  // collapsable in `this`
  // 2. For any pair of dimensions in `this`, if they are collapsable in
  // `otherInfo`, they must be collapsable into the same dimension in
  // `otherInfo` to be collapsable into the same dimension in `this`.
  // Returns true if the operation modified the number of collapsable loops.
  bool updateFromOther(FailureOr<AffineMap> otherToThisMap,
                       const CollapseInfo &otherInfo);

  // Update `this` (which is the info for `op`) when either a producer or
  // consumer is not collapsible. This is done by considering all the dims
  // accessed by other to be uncollapsible.
  bool updateFromUncollapsible(Operation *op, OpOperand *operand);

  // Get `collapsableLoops` after applying the transformation provided by `map`.
  // Note: doesn't modify `collapsableLoops`, the tranformation is applied to a
  // copy.
  CollapsableLoopsSet getTransformedCollapsableLoops(AffineMap map) const;

  // Get `reassociation` after applying the transformation provided by `map`.
  SmallVector<ReassociationIndices>
  getTransformedReassociation(AffineMap map) const;

  // Clear internal data and returns if anything changed.
  bool clear() {
    bool isNotEmpty = reassociation.empty() || collapsableLoops.empty();
    reassociation.clear();
    collapsableLoops.clear();
    return isNotEmpty;
  }

  const CollapsableLoopsSet &getCollapsibleLoops() const {
    return collapsableLoops;
  }

  const SmallVector<ReassociationIndices> &getReassocation() const {
    return reassociation;
  }

private:
  // Get a set of all elements in `reassociation`
  static CollapsableLoopsSet
  getCollapsedFromReassociation(ArrayRef<ReassociationIndices> reassociation) {
    CollapsableLoopsSet collapsed;
    for (auto &indicies : reassociation) {
      for (int64_t index : indicies) {
        collapsed.insert(index);
      }
    }
    return collapsed;
  }

  // Update `reassociation` by removing indicies that are no longer in
  // `collapsableLoops` and spliting the reassociation indicies accordingly
  void updateReassociation();

private:
  // A vector of `ReassociationIndicies` representing contiguous dimensions that
  // can be collapsed together.
  SmallVector<ReassociationIndices> reassociation;

  // Note: `collapsableLoops` does not directly map to `reassociation`
  // because parallel and reduction iteration dimensions must be kept separate.
  CollapsableLoopsSet collapsableLoops;
};
} // namespace

// Removes any indicies in `reassociation` that are not in `collapsableLoops`,
// The reassociation indicies are split along the uncollapsable element because
// the dims aren't contiguous and cannot be collapsed. Single element
// reassociation indicies are cleaned up.
void CollapseInfo::updateReassociation() {
  SmallVector<ReassociationIndices> newReassociation;
  for (auto &indicies : reassociation) {

    // Holds dimensions that should be collapsed together
    ReassociationIndices newIndicies;
    for (int64_t index : indicies) {
      // This index is collapsable and should be kept in the reassociation
      // indicies.
      if (collapsableLoops.contains(index)) {
        newIndicies.push_back(index);
        continue;
      }

      // Because `index` isn't collapsable, the indicies in `newIndicies` are no
      // longer adjacent to the upcoming indicies. If there is >1 index to
      // collapse, add it to the new reassociation. Otherwise, discard it
      // because there is no dimension to collapse with.
      if (newIndicies.size() > 1) {
        newReassociation.push_back(newIndicies);
      }
      newIndicies.clear();
    }

    if (newIndicies.size() > 1) {
      newReassociation.push_back(newIndicies);
    }
  }
  reassociation = std::move(newReassociation);
}

// Given an AffineMap `map` get the transformed `collapsableLoops`. For example,
// if this `CollapseInfo` represents a elementwise linalg generic operating on a
// 3d tensor (so its collapsableLoops might be {0, 1, 2}), the map would be used
// to map the loops to the iteration space of its producer or consumer.
//
// Consider it's consumer accesses the result of said operation with
// affine_map<(d0, d1, d2) -> (d1, d2, d5)>
//
// Then:
// collapsableLoops = {0, 1, 2}
// map = affine_map<(d0, d1, d2) -> (d1, d2, d5)>
//
// Therefore, the collapsable loops with respect to the consumer is {1, 2, 5}.
CollapseInfo::CollapsableLoopsSet
CollapseInfo::getTransformedCollapsableLoops(AffineMap map) const {
  CollapsableLoopsSet transformedLoops;
  for (auto index : collapsableLoops) {
    assert(index < map.getNumResults() && "index has no valid mapping");
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(index));
    if (!dimExpr) {
      continue;
    }

    transformedLoops.insert(dimExpr.getPosition());
  }
  return transformedLoops;
}

SmallVector<ReassociationIndices>
CollapseInfo::getTransformedReassociation(AffineMap map) const {
  SmallVector<ReassociationIndices> transformedReassociation(
      reassociation.size());
  for (const auto &[i, indicies] : llvm::enumerate(reassociation)) {
    for (auto elem : indicies) {
      auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(elem));
      if (!dimExpr) {
        break;
      }
      transformedReassociation[i].push_back(dimExpr.getPosition());
    }
  }
  return transformedReassociation;
}

bool CollapseInfo::updateFromOther(FailureOr<AffineMap> otherToThisMap,
                                   const CollapseInfo &otherInfo) {
  if (failed(otherToThisMap)) {
    return this->clear();
  }

  CollapsableLoopsSet otherCollapsible =
      otherInfo.getTransformedCollapsableLoops(otherToThisMap.value());

  SmallVector<ReassociationIndices> otherReassoc =
      otherInfo.getTransformedReassociation(otherToThisMap.value());

  // Get a map from original index to the index it gets collapsed into
  llvm::DenseMap<long, long> otherCollapseMap;
  for (const auto &[idx, indicies] : llvm::enumerate(otherReassoc)) {
    for (const auto elem : indicies) {
      otherCollapseMap[elem] = idx;
    }
  }

  // Remove all collapsable loops in `this` that both exist and are not
  // collapsable in `other` (set intersect)
  bool didChange = collapsableLoops.remove_if([&](long elem) -> bool {
    // Exists and is collapsable
    if (otherCollapsible.contains(elem)) {
      return false;
    }

    // Does not exist in `other`.
    if (!otherToThisMap->isFunctionOfDim(elem)) {
      return false;
    }

    return true;
  });

  // Now update the reassociation indicies given the updated `collapsableLoops`
  // and `otherCollapsableMap`.
  // The idea is to reconstruct the reassociation indicies, and at each index:
  // (1) If `index` IS NOT in `collapsableLoops`, split `indicies` and don't add
  // `index` to either.
  //
  // (2) If `index` IS in `collapsableLoops` but `otherCollapseMap` maps
  // `index` to a different collapsed loop then the other indicies,  split
  // `indicies` and insert `index` into the new one.
  //
  // For example:
  // `this` reassociation = [[0, 1], [2, 3]]
  // `other` reassociation = [0, 1, 2, 3]
  // then, `other` reassociation gets updated to [[0, 1], [2, 3]] because
  // [0, 1] and [2, 3] get collapsed into different loops
  //
  // (3) Otherwise, keep the index
  constexpr long kUninitialized = -1;
  SmallVector<ReassociationIndices> newReassociation;
  for (ReassociationIndicesRef indicies : reassociation) {
    // Track the loop index that `indicies` get collapsed into.
    long collapseIntoIdx = kUninitialized;

    // Holds dimensions that should be collapsed together
    ReassociationIndices newIndicies;
    for (int64_t index : indicies) {
      if (!collapsableLoops.contains(index)) {
        // (1) Because `index` isn't collapsable, the indicies in `newIndicies`
        // are no longer adjacent to the upcoming indicies. If there is >1 index
        // to collapse, add it to the new reassociation. Otherwise, discard it
        // because there is no dimension to collapse with.
        didChange = true;
        if (newIndicies.size() > 1) {
          newReassociation.push_back(std::move(newIndicies));
        }
        newIndicies.clear();
        collapseIntoIdx = kUninitialized;
      } else if (!otherCollapseMap.contains(index)) {
        // (2) `index` does not exist in `other`.
        newIndicies.push_back(index);
      } else if (collapseIntoIdx == kUninitialized) {
        // (3) First occurance of collapsable loop, set collapseIntoIdx.
        collapseIntoIdx = otherCollapseMap.at(index);
        newIndicies.push_back(index);
      } else if (otherCollapseMap.at(index) != collapseIntoIdx) {
        // (4) `index` is collapsable but not collapsable into the other loops.
        // So, split them and look for other loops to collapse `index` into.
        didChange = true;
        if (newIndicies.size() > 1) {
          newReassociation.push_back(std::move(newIndicies));
        }
        newIndicies.clear();
        collapseIntoIdx = otherCollapseMap[index];
        newIndicies.push_back(index);
      } else {
        // (5) `index` is collapsable and can be collapsed into
        // `collapseIntoIndex`.
        newIndicies.push_back(index);
      }
    }

    if (newIndicies.size() > 1) {
      newReassociation.push_back(newIndicies);
    }
  }

  if (didChange) {
    reassociation = std::move(newReassociation);
    collapsableLoops = getCollapsedFromReassociation(reassociation);
  }
  return didChange;
}

bool CollapseInfo::updateFromUncollapsible(Operation *op, OpOperand *operand) {
  auto fusionOp = cast<LinalgFusionOpInterface>(op);
  AffineMap map = operand->getOwner() == op
                      ? fusionOp.getMatchingIndexingMap(operand)
                      : fusionOp.getIndexingMapMatchingResult(
                            cast<OpResult>(operand->get()));

  CollapseInfo::CollapsableLoopsSet uncollapsable;
  for (auto expr : map.getResults()) {
    uncollapsable.insert(cast<AffineDimExpr>(expr).getPosition());
  }
  auto initialSize = collapsableLoops.size();
  collapsableLoops.set_subtract(uncollapsable);
  updateReassociation();
  return initialSize != collapsableLoops.size();
}

void CollapseInfo::print(raw_ostream &os) const {
  os << "[CollapseDims] CollapseInfo:\n";

  os << "Reassociation: ";
  os << "[";
  for (auto &vec : reassociation) {
    os << "[";
    llvm::interleaveComma(vec, os);
    os << "]";
  }
  os << "]";
  os << "\n";

  os << "Collapsable: {";
  llvm::interleaveComma(collapsableLoops, os);
  os << "}";
}

void CollapseInfo::dump() const { print(llvm::dbgs()); }

/// Traverses all the the Ops in DispatchRegionOps and finds a Op
/// which is the sole producer of the flow.return's operand.
static FailureOr<Operation *>
findRootOp(IREE::Flow::DispatchRegionOp regionOp) {
  // Check the yielded value is from a single op.
  auto returnOp =
      cast<IREE::Flow::ReturnOp>(regionOp.getBody().front().getTerminator());
  if (!returnOp->getOperands().size()) {
    return failure();
  }

  Operation *collapsibleOp = returnOp->getOperand(0).getDefiningOp();
  if (!isEligibleForCollapse(collapsibleOp)) {
    return failure();
  }
  for (auto returnVal : returnOp->getOperands().drop_front()) {
    if (returnVal.getDefiningOp() != collapsibleOp) {
      return failure();
    }
  }

  return collapsibleOp;
}

//===---------------------------------------------------------------------===//
// Reshape Hoisting
//===---------------------------------------------------------------------===//

/// Hoist `tensor.collapse_shape` and `tensor.expand_shape` ops at the beginning
/// of the `dispatchOp` and `tensor.expand_shape` ops at the end of the
/// `dispatchOp`, out of the dispatch.
static FailureOr<IREE::Flow::DispatchRegionOp>
hoistTensorReshapesOutOfDispatchRegion(
    RewriterBase &rewriter, IREE::Flow::DispatchRegionOp dispatchOp) {
  Block &body = dispatchOp.getBody().front();
  auto returnOp = cast<IREE::Flow::ReturnOp>(body.getTerminator());

  // 1. Get the slice of operations within `dispatchOp` that produce the yielded
  // value.
  BackwardSliceOptions sliceOptions;
  sliceOptions.omitBlockArguments = true;
  sliceOptions.filter = [&](Operation *op) {
    return op->getParentOfType<IREE::Flow::DispatchRegionOp>();
  };
  SetVector<Operation *> slice;
  [[maybe_unused]] LogicalResult ret =
      getBackwardSlice(returnOp, &slice, sliceOptions);
  assert(ret.succeeded());

  // 2. Get the leaf operations that are `tensor.collapse_shape` and
  // `tensor_expand_shape` ops.
  SmallVector<Operation *> reshapeLeafs;
  for (Operation *op : slice) {
    if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op)) {
      continue;
    }
    if (llvm::all_of(op->getOperands(), [&](Value operand) {
          Operation *definingOp = operand.getDefiningOp();
          return !definingOp || slice.count(definingOp) == 0;
        })) {
      reshapeLeafs.push_back(op);
    }
  }

  // 3. Clone the leaf `tensor.collapse_shape` and `tensor_expand_shape`  ops
  // outside the dispatch.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(dispatchOp);
  for (auto reshapeOp : reshapeLeafs) {
    Operation *clonedOp = rewriter.clone(*reshapeOp);
    rewriter.replaceOp(reshapeOp, clonedOp->getResults());
  }

  // 4. From the yielded values find any that are produced by
  //    `tensor.expand_shape` operation and move them out of the dispatch. For
  //    this a new `DispatchRegionOp` is needed. For values that are yielded and
  //    produced from `tensor.expand_shape`, the type of the result changes. The
  //    dynamic dimensions of the result type also need to be updated.
  SmallVector<Type> newReturnTypes;
  SmallVector<Value> newDynamicDims;
  SmallVector<Value> newYieldVals;
  SmallVector<SmallVector<ReassociationIndices>> allReassociationIndices;
  ValueRange dynamicDimsList = dispatchOp.getResultDims();
  Location loc = dispatchOp.getLoc();
  for (auto [resultIndex, yieldedValue] :
       llvm::enumerate(returnOp->getOperands())) {
    auto expandShapeOp = yieldedValue.getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandShapeOp) {
      // 4a. Keep the same yield value if the producer is not a
      // `tensor.expand_shape` op.
      newReturnTypes.push_back(yieldedValue.getType());
      ValueRange resultDims = dispatchOp.getResultDynamicDims(resultIndex);
      newDynamicDims.append(resultDims.begin(), resultDims.end());
      newYieldVals.push_back(yieldedValue);
      continue;
    }

    // 4b. The return type is same as the type of the source of the
    // `tensor.expand_shape`.
    RankedTensorType collapsedShapeType = expandShapeOp.getSrcType();
    newReturnTypes.push_back(collapsedShapeType);
    newYieldVals.push_back(expandShapeOp.getSrc());
    SmallVector<ReassociationIndices> reassociation =
        expandShapeOp.getReassociationIndices();
    ArrayRef<int64_t> expandedShape = expandShapeOp.getResultType().getShape();

    // 4c. Dynamic dims of the result shape is obtained by taking the static
    //     shape + dynamic dims and collapsing them using the same reassociation
    //     map as the `tensor.expand_shape`.
    for (auto [index, shape] : llvm::enumerate(collapsedShapeType.getShape())) {
      int64_t staticCollapsedShape = 1;
      SmallVector<OpFoldResult> dynamicCollapsedDims;
      for (auto collapsedDim : reassociation[index]) {
        if (ShapedType::isDynamic(expandedShape[collapsedDim])) {
          dynamicCollapsedDims.push_back(dynamicDimsList.front());
          dynamicDimsList = dynamicDimsList.drop_front();
        } else {
          staticCollapsedShape *= expandedShape[collapsedDim];
        }
      }

      if (dynamicCollapsedDims.empty()) {
        // If there are no dynamic dims, there is nothing to do.
        continue;
      }
      SmallVector<AffineExpr> exprs(dynamicCollapsedDims.size());
      bindSymbolsList(rewriter.getContext(),
                      MutableArrayRef<AffineExpr>(exprs));
      AffineExpr multiplyAll = exprs.front();
      for (auto expr : ArrayRef<AffineExpr>(exprs).drop_front()) {
        multiplyAll = multiplyAll * expr;
      }
      if (staticCollapsedShape != 1) {
        multiplyAll = multiplyAll * staticCollapsedShape;
      }
      OpFoldResult collapsedShape = affine::makeComposedFoldedAffineApply(
          rewriter, loc, multiplyAll, dynamicCollapsedDims);
      newDynamicDims.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, collapsedShape));
    }
    allReassociationIndices.emplace_back(std::move(reassociation));
  }

  // 5. Create the new dispatch op.
  auto newDispatchOp = rewriter.create<IREE::Flow::DispatchRegionOp>(
      loc, newReturnTypes, newDynamicDims, dispatchOp.getWorkload());

  // 5a. Move the body over, but replace the `flow.return` to use the new yield
  // values.
  Region &newBody = newDispatchOp.getBody();
  rewriter.inlineRegionBefore(dispatchOp.getBody(), newBody, newBody.begin());
  {
    Operation *terminator = newBody.front().getTerminator();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(terminator);
    rewriter.replaceOpWithNewOp<IREE::Flow::ReturnOp>(terminator, newYieldVals);
  }

  // 5b. Move the workgroup count region over.
  Region &workgroupCountRegion = dispatchOp.getWorkgroupCount();
  if (!workgroupCountRegion.empty()) {
    Region &newWorkgroupCountRegion = newDispatchOp.getWorkgroupCount();
    rewriter.inlineRegionBefore(workgroupCountRegion, newWorkgroupCountRegion,
                                newWorkgroupCountRegion.begin());
  }

  // 6. Map the modified result values back to their original shape using
  //    `tensor.expand_shape` operations.
  ArrayRef<SmallVector<ReassociationIndices>> allReassociationIndicesRef(
      allReassociationIndices);
  for (auto [index, returnValue] :
       llvm::enumerate(newDispatchOp.getResults())) {
    Value origResult = dispatchOp->getResult(index);
    if (returnValue.getType() == origResult.getType()) {
      rewriter.replaceAllUsesWith(origResult, returnValue);
      continue;
    }

    auto shapedType = dyn_cast<ShapedType>(origResult.getType());
    assert(shapedType && "result should be shaped type");

    ValueRange dynamicDims = dispatchOp.getResultDynamicDims(index);
    SmallVector<OpFoldResult> outputShape =
        mlir::getMixedValues(shapedType.getShape(), dynamicDims, rewriter);

    auto newExpandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, origResult.getType(), returnValue,
        allReassociationIndicesRef.front(), outputShape);
    allReassociationIndicesRef = allReassociationIndicesRef.drop_front();
    rewriter.replaceAllUsesWith(origResult, newExpandShapeOp.getResult());
  }
  rewriter.eraseOp(dispatchOp);
  return newDispatchOp;
}

//===---------------------------------------------------------------------===//
// Collapse shape propagation
//===---------------------------------------------------------------------===//

// For each consumer, use it's producers to constrain which dimensions it will
// collapse. `slice` is expected to be topologically sorted (getBackwardSlice
// does this automatically).
// Returns true if the operation modified any op's `CollapseInfo`.
static bool
updateConsumersFromProducers(ArrayRef<Operation *> slice,
                             llvm::DenseMap<Operation *, CollapseInfo> &opMap) {
  bool didChange = false;

  // Slice is topologically sorted to ensure that `op`'s producers have been
  // updated before we visit it.
  for (auto op : slice) {
    auto consumerOp = cast<LinalgFusionOpInterface>(op);
    CollapseInfo &consumerInfo = opMap.find(consumerOp)->second;

    for (auto *operand :
         cast<DestinationStyleOpInterface>(op).getDpsInputOperands()) {
      auto producerOp = operand->get().getDefiningOp();
      if (!producerOp || IREE::Flow::isNonNullAndOutsideDispatch(producerOp)) {
        continue;
      }

      // If we can't find the op, the tensor is not collapsable. So, consider
      // all the dimensions of the producer to be uncollapsable.
      if (!opMap.contains(producerOp)) {
        didChange |= consumerInfo.updateFromUncollapsible(consumerOp, operand);
        continue;
      }

      const CollapseInfo &producerInfo = opMap.at(producerOp);
      FailureOr<AffineMap> consumerToProducerMap =
          getProducerLoopToConsumerLoopsMap(*operand);
      didChange |=
          consumerInfo.updateFromOther(consumerToProducerMap, producerInfo);
    }
  }
  return didChange;
}

// For each producer, use it's consumers to constrain which dimensions it will
// collapse. `slice` is expected to be topologically sorted (getBackwardSlice
// does this automatically).
// Returns true if the operation modified any op's `CollapseInfo`.
static bool
updateProducersFromConsumers(ArrayRef<Operation *> slice,
                             llvm::DenseMap<Operation *, CollapseInfo> &opMap) {
  bool didChange = false;

  // Iterate over `slice` in reverse so that we visit each `op` 's consumer
  // before visiting `op`.
  for (auto op : llvm::reverse(slice)) {
    auto producerOp = cast<LinalgFusionOpInterface>(op);
    CollapseInfo &producerInfo = opMap.find(producerOp)->second;

    for (auto &operand : producerOp->getUses()) {
      auto *consumerOp = operand.getOwner();
      if (consumerOp->hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }

      // If we can't find the op, the tensor is not collapsable. So, consider
      // all the dimensions of the consumer to be uncollapsable.
      if (!opMap.contains(consumerOp)) {
        didChange |= producerInfo.updateFromUncollapsible(producerOp, &operand);
        continue;
      }

      // Get a mapping from the consumer's iteration space to the producer's.
      const CollapseInfo &consumerInfo = opMap.at(consumerOp);

      // Only loops collapsable in both the consumer and producer may be
      // collapsed.
      FailureOr<AffineMap> consumerToProducerMap =
          getConsumerLoopToProducerLoopsMap(operand);
      didChange |=
          producerInfo.updateFromOther(consumerToProducerMap, consumerInfo);
    }
  }
  return didChange;
}

// Construct a DAG of operations with 1 root op. Find
// dimensions that can be collapsed all the way from the root to the leaves,
// ensuring that all `collapse_shape` ops can be hoisted out of the dispatch.
static bool
collapseDimensionsForDispatch(IRRewriter &rewriter,
                              IREE::Flow::DispatchRegionOp &regionOp,
                              int maxIterations) {
  // Only collapse dispatches with 1 block
  if (!llvm::hasSingleElement(regionOp.getBody())) {
    return false;
  }
  // Step 1. Find the root Op
  std::optional<Operation *> rootOp = findRootOp(regionOp);
  if (!rootOp.has_value()) {
    return false;
  }

  // Step 2. Get slice of all ops in the dispatch
  BackwardSliceOptions sliceOptions;
  sliceOptions.inclusive = true;
  sliceOptions.omitBlockArguments = true;
  sliceOptions.omitUsesFromAbove = false;
  sliceOptions.filter = [&](Operation *op) -> bool {
    auto parentOp = op->getParentOfType<IREE::Flow::DispatchRegionOp>();
    return isEligibleForCollapse(op) && parentOp == regionOp;
  };
  SetVector<Operation *> slice;
  [[maybe_unused]] LogicalResult ret =
      getBackwardSlice(rootOp.value(), &slice, sliceOptions);
  assert(ret.succeeded());

  // Step 3. Populate each op's info with a maximally collapsable reassociation
  // indicies
  llvm::DenseMap<Operation *, CollapseInfo> opMap;
  opMap.reserve(slice.size());
  for (auto *op : slice) {
    opMap[op] = CollapseInfo(op);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[CollapseDims] : After initializing opMap\n";
    for (auto &[op, info] : opMap) {
      info.dump();
      llvm::dbgs() << "\n";
      op->dump();
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\n";
  });

  bool didUpdateProducers = true;
  bool didUpdateConsumers = true;
  int iterationCount = 0;
  while (didUpdateProducers || didUpdateConsumers) {
    // Cap the max number of iterations at 10. If it hasn't converged by then,
    // don't collapse any ops in this dispatch.
    iterationCount++;
    if (iterationCount > maxIterations) {
      return false;
    }
    // Step 4. For each producer, reduce the number of collapsed dimensions
    // based on the dimensions that it's consumers can collapse.
    didUpdateProducers =
        updateProducersFromConsumers(slice.getArrayRef(), opMap);

    LLVM_DEBUG({
      llvm::dbgs() << "[CollapseDims] : After updating producers: \n";
      for (auto &[op, info] : opMap) {
        info.dump();
        llvm::dbgs() << "\n";
        op->dump();
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "\n";
    });

    // Step 5. For each consumer, update it's CollapseInfo to only collapse
    // dimensions that all of its producers can collapse. This ensures that all
    // reshapes can be propagated to leafs and be hoisted out of the dispatch.
    didUpdateConsumers =
        updateConsumersFromProducers(slice.getArrayRef(), opMap);

    LLVM_DEBUG({
      llvm::dbgs() << "[CollapseDims] : After updating consumers: \n";
      for (auto &[op, info] : opMap) {
        info.dump();
        llvm::dbgs() << "\n";
        op->dump();
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "\n";
    });
  }

  bool didCollapse = false;

  // Step 6. Collapse dimensions based on each op's CollapseInfo
  for (auto &[opToCollapse, info] : opMap) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(opToCollapse);

    using ResultsType = FailureOr<SmallVector<Value>>;
    auto maybeReplacements =
        llvm::TypeSwitch<Operation *, ResultsType>(opToCollapse)
            .Case<linalg::LinalgOp>(
                [&, &info = info](auto genericOp) -> ResultsType {
                  FailureOr<linalg::CollapseResult> maybeReplacements =
                      mlir::linalg::collapseOpIterationDims(
                          genericOp, info.getReassocation(), rewriter);
                  if (failed(maybeReplacements)) {
                    return failure();
                  }
                  return maybeReplacements->results;
                })
            .Case<IREE::LinalgExt::AttentionOp>(
                [&, &info = info](auto attentionOp) -> ResultsType {
                  FailureOr<IREE::LinalgExt::CollapseResult> maybeReplacements =
                      IREE::LinalgExt::collapseOpIterationDims(
                          attentionOp, info.getReassocation(), rewriter);
                  if (failed(maybeReplacements)) {
                    return failure();
                  }
                  return maybeReplacements->results;
                })
            .Default([&](void *) -> ResultsType {
              llvm_unreachable("no type matched");
              return failure();
            });
    if (failed(maybeReplacements)) {
      continue;
    }
    didCollapse = true;
    rewriter.replaceOp(opToCollapse, maybeReplacements.value());
  }
  return didCollapse;
}

//===---------------------------------------------------------------------===//
// Passes
//===---------------------------------------------------------------------===//

void CollapseDimensionsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = funcOp->getContext();
  IRRewriter rewriter(context);

  SmallVector<IREE::Flow::DispatchRegionOp> modifiedDispatchOps;
  auto walkRes = funcOp->walk([&](IREE::Flow::DispatchRegionOp dispatchOp) {
    FailureOr<IREE::Flow::DispatchRegionOp> newDispatchOp =
        hoistTensorReshapesOutOfDispatchRegion(
            rewriter, cast<IREE::Flow::DispatchRegionOp>(dispatchOp));
    if (failed(newDispatchOp)) {
      dispatchOp->emitOpError("failed to hoist reshapes out of dispatch");
      return WalkResult::interrupt();
    }
    if (collapseDimensionsForDispatch(rewriter, newDispatchOp.value(),
                                      maxIterations)) {
      modifiedDispatchOps.push_back(newDispatchOp.value());
    }
    return WalkResult::advance();
  });
  if (walkRes.wasInterrupted()) {
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[CollapseDims] : After collapsing ops: \n";
    funcOp.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Move all the `tensor.collapse_shape` leafs  and `tensor.expand_shape` roots
  // of the modified dispatches out of the dispatch.
  for (auto dispatchOp : modifiedDispatchOps) {
    // Hoist tensor reshape ops out of dispatch region first. Otherwise, the
    // reshape(cst) will be folded into a constant living in the dispatch. It
    // could introduce big constants inlined in the dispatch.
    FailureOr<IREE::Flow::DispatchRegionOp> newDispatchOp =
        hoistTensorReshapesOutOfDispatchRegion(
            rewriter, cast<IREE::Flow::DispatchRegionOp>(dispatchOp));
    if (failed(newDispatchOp)) {
      dispatchOp->emitOpError("failed to hoist reshapes out of dispatch");
      return signalPassFailure();
    }

    Region &body = newDispatchOp.value().getBody();
    assert(llvm::hasSingleElement(body) && "expected op with a single body");
    Block &block = body.front();
    RewritePatternSet moveReshapeOps(&getContext());
    linalg::FillOp::getCanonicalizationPatterns(moveReshapeOps, context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(moveReshapeOps);
    tensor::populateFoldTensorEmptyPatterns(moveReshapeOps);
    SmallVector<Operation *> candidateOps;
    block.walk([&](Operation *op) { candidateOps.push_back(op); });
    if (failed(
            applyOpPatternsGreedily(candidateOps, std::move(moveReshapeOps)))) {
      funcOp.emitOpError(
          "failed to propagate reshape ops introduced during collapse");
      return signalPassFailure();
    }

    // Expand affine.apply ops from dynamic dims
    newDispatchOp->walk([&](affine::AffineApplyOp op) {
      rewriter.setInsertionPoint(op);
      auto maybeExpanded = mlir::affine::expandAffineMap(
          rewriter, op.getLoc(), op.getAffineMap(),
          llvm::to_vector<4>(op.getOperands()));
      if (!maybeExpanded) {
        return;
      }
      rewriter.replaceOp(op, *maybeExpanded);
    });
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
