// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-collapse-dimensions"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_COLLAPSEDIMENSIONSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {
/// Pass declaration.
struct CollapseDimensionsPass
    : public IREE::Flow::impl::CollapseDimensionsPassBase<
          CollapseDimensionsPass> {
  void runOnOperation() override;
};
} // namespace

//===---------------------------------------------------------------------===//
// Helper functions
//===---------------------------------------------------------------------===//

/// Searches the same sequence in all the affine maps and collapses these
/// dimensions. It only applies these to "parallel" loops without mixing them
/// with "reduction" types. It is expected that the `genericOp` has projected
/// permutations only as indexing maps. (Checked using `isEligibleForCollapse`).
static SmallVector<ReassociationIndices>
getCollapsibleLoops(linalg::GenericOp genericOp) {
  SmallVector<ReassociationIndices> contiguousLoops;

  SmallVector<unsigned> pDims, rDims;
  genericOp.getParallelDims(pDims);
  genericOp.getReductionDims(rDims);
  llvm::SmallDenseSet<unsigned> pDimsSet, rDimsSet;
  pDimsSet.insert(pDims.begin(), pDims.end());
  rDimsSet.insert(rDims.begin(), rDims.end());

  auto hasAllMapsSameSequence = [&](AffineExpr preExpr, AffineExpr nextExpr) {
    // Check that all indexing maps of the `genericOp`
    // - Either both `preExpr` and `nextExpr` contiguous, or
    // - are missing in
    // Then `preExpr` and `nextExpr` can be collapsed.
    for (AffineMap map : genericOp.getIndexingMapsArray()) {
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

  ReassociationIndices range;
  AffineExpr preExpr;
  // Find the largest sequence of dimensions that are
  // - Either preserved in all maps, or
  // - are completely absent
  // This sequence can be collapsed. To find the sequence,
  // 1) Take the result expressions of one of the indexing maps
  // 2) Find a sequence of 2 that is found in all maps
  // 3) Then take last element of this sequence and the next
  //    result expression, and check if this sequence of 2 is
  //    found in all maps. If so, add to sequence (to get a sequence of 3)
  //    and repeat till the last element of sequence and the next result
  //    expression is not found as a sequence in all maps.
  for (auto nextExpr : genericOp.getIndexingMapsArray().front().getResults()) {
    unsigned position = cast<AffineDimExpr>(nextExpr).getPosition();
    if (!range.empty()) {
      if (!hasAllMapsSameSequence(preExpr, nextExpr) ||
          !hasSameIteratorType(preExpr, nextExpr)) {
        if (range.size() > 1) {
          contiguousLoops.push_back({range.begin(), range.end()});
        }
        range.clear();
      }
    }
    range.push_back(position);
    preExpr = nextExpr;
  }
  if (range.size() > 1)
    contiguousLoops.push_back(range);

  return contiguousLoops;
}

/// Returns true if the given op is collapsable.
static bool isEligibleForCollapse(linalg::GenericOp genericOp) {
  // TODO(guray) There is no mechanism to tell the collapsed indexes to
  // `tensor.expand_shape`. Once we have this support in MLIR, we can enable
  // dynamic tensor shapes.
  if (genericOp.hasDynamicShape())
    return false;

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

  // TODO(guray) Collapsing caused performance regression in a cpu
  // benchmark, so we disable it.
  if (genericOp.hasIndexSemantics())
    return false;

  // TODO(#17948) GPU codegen fails when we collapse the dimensions of softmax.
  if (llvm::any_of(genericOp.getDpsInputOperands(),
                   [&](OpOperand *operand) -> bool {
                     auto genericOperand =
                         operand->get().getDefiningOp<linalg::GenericOp>();
                     if (!genericOperand)
                       return false;

                     if (genericOperand.getNumReductionLoops() == 0)
                       return false;

                     return genericOp.getMatchingIndexingMap(operand)
                         .isProjectedPermutation();
                   })) {
    return false;
  }

  return true;
}

// For the `operand` with producers and consumers of type `genericOp`, get
// of producer loop -> consumer loop.
static FailureOr<AffineMap>
getProducerLoopToConsumerLoopsMap(OpOperand &operand) {
  linalg::GenericOp consumer = dyn_cast<linalg::GenericOp>(operand.getOwner());
  if (!consumer) {
    return failure();
  }
  linalg::GenericOp producer =
      dyn_cast_or_null<linalg::GenericOp>(operand.get().getDefiningOp());
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
  linalg::GenericOp consumer = dyn_cast<linalg::GenericOp>(operand.getOwner());
  if (!consumer) {
    return failure();
  }
  linalg::GenericOp producer =
      dyn_cast_or_null<linalg::GenericOp>(operand.get().getDefiningOp());
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
  CollapseInfo(ArrayRef<ReassociationIndices> reassociation)
      : reassociation(reassociation),
        collapsableLoops(getCollapsedFromReassociation(reassociation)) {}

  // Print the current operation & reassociation indicies
  void print(raw_ostream &os) const;

  // Debug print the current operation & reassociation indicies
  void dump() const;

  // Update `collapsableLoops` by taking the set intersection with
  // `otherCollapsable` and update the reassociation indicies accordingly.
  void updateCollapseViaIntersect(const CollapsableLoopsSet &otherCollapsable);

  // Update `collapsableLoops` by subtracting `uncollapsable` and update the
  // reassociation indicies accordingly.
  void updateCollapseViaSubtract(const CollapsableLoopsSet &uncollapsable);

  // Get `collapsableLoops` after applying the transformation provided by `map`.
  // Note: doesn't modify `collapsableLoops`, the tranformation is applied to a
  // copy.
  FailureOr<CollapsableLoopsSet>
  getTransformedCollapsableLoops(AffineMap map) const;

  // Clear internal data
  void clear() {
    reassociation.clear();
    collapsableLoops.clear();
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
FailureOr<CollapseInfo::CollapsableLoopsSet>
CollapseInfo::getTransformedCollapsableLoops(AffineMap map) const {
  if (!map) {
    return failure();
  }

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

// Update `collapsableLoops` by taking the set intersection with
// `otherCollapsable` and update the reassociation indicies accordingly.
void CollapseInfo::updateCollapseViaIntersect(
    const CollapsableLoopsSet &otherCollapsable) {

  CollapsableLoopsSet toRemove;
  for (auto elem : collapsableLoops) {
    if (!otherCollapsable.contains(elem)) {
      toRemove.insert(elem);
    }
  }
  collapsableLoops.set_subtract(toRemove);
  updateReassociation();
}

// Update `collapsableLoops` by subtracting `uncollapsable` and update the
// reassociation indicies accordingly.
void CollapseInfo::updateCollapseViaSubtract(
    const CollapsableLoopsSet &uncollapsable) {

  collapsableLoops.set_subtract(uncollapsable);
  updateReassociation();
}

void CollapseInfo::print(raw_ostream &os) const {
  os << "[CollapseDimensions] CollapseInfo:\n";

  os << "Reassociation: ";
  os << "[";
  for (auto &vec : reassociation) {
    os << "[";
    bool first = true;
    for (auto elem : vec) {
      if (!first) {
        os << ", ";
      }
      first = false;
      os << elem;
    }
    os << "]";
  }
  os << "]";
  os << "\n";

  os << "Collapsable: ";
  os << "{";
  bool first = true;
  for (auto elem : collapsableLoops) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << elem;
  }
  os << "}";
}

void CollapseInfo::dump() const { print(llvm::dbgs()); }

/// Traverses all the the Ops in DispatchRegionOps and finds a linalg.generic Op
/// which is the sole producer of the flow.return's operand.
static FailureOr<linalg::GenericOp>
findRootGenericOp(DispatchRegionOp regionOp) {
  // Check the yielded value is from a single `linalg.generic`.
  auto returnOp =
      cast<Flow::ReturnOp>(regionOp.getBody().front().getTerminator());
  if (!returnOp->getOperands().size()) {
    return failure();
  }
  auto collapsibleOp = dyn_cast_or_null<linalg::GenericOp>(
      returnOp->getOperand(0).getDefiningOp());
  if (!collapsibleOp) {
    return failure();
  }
  for (auto returnVal : returnOp->getOperands().drop_front()) {
    if (returnVal.getDefiningOp() != collapsibleOp.getOperation()) {
      return failure();
    }
  }

  return collapsibleOp;
}

//===---------------------------------------------------------------------===//
// Reshape Hoisting
//===---------------------------------------------------------------------===//

/// Hoist `tensor.collapse_shape` ops at the beginning of the `dispatchOp`
/// and `tensor.expand_shape` ops at the end of the `dispatchOp`, out of the
/// dispatch.
static FailureOr<DispatchRegionOp>
hoistTensorReshapesOutOfDispatchRegion(RewriterBase &rewriter,
                                       DispatchRegionOp dispatchOp) {
  Block &body = dispatchOp.getBody().front();
  auto returnOp = cast<Flow::ReturnOp>(body.getTerminator());

  // 1. Get the slice of operations within `dispatchOp` that produce the yielded
  // value.
  BackwardSliceOptions sliceOptions;
  sliceOptions.filter = [&](Operation *op) {
    return op->getParentOfType<DispatchRegionOp>();
  };
  SetVector<Operation *> slice;
  getBackwardSlice(returnOp, &slice, sliceOptions);

  // 2. Get the leaf operations that are tensor.collapse_shape ops.
  SmallVector<tensor::CollapseShapeOp> leafs;
  for (Operation *op : slice) {
    auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(op);
    if (!collapseShapeOp) {
      continue;
    }
    if (llvm::all_of(op->getOperands(), [&](Value operand) {
          Operation *definingOp = operand.getDefiningOp();
          return !definingOp || slice.count(definingOp) == 0;
        })) {
      leafs.push_back(collapseShapeOp);
    }
  }

  // 3. Clone the leaf `tensor.collapse_shape` ops outside the dispatch.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(dispatchOp);
  for (auto reshapeOp : leafs) {
    Operation *clonedOp = rewriter.clone(*reshapeOp.getOperation());
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
  for (Value yieldedValue : returnOp->getOperands()) {
    auto expandShapeOp = yieldedValue.getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandShapeOp) {
      // 4a. Keep the same yield value if the producer is not a
      // `tensor.expand_shape` op.
      newReturnTypes.push_back(yieldedValue.getType());
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
  auto newDispatchOp = rewriter.create<DispatchRegionOp>(
      loc, newReturnTypes, newDynamicDims, dispatchOp.getWorkload());

  // 5a. Move the body over, but replace the `flow.return` to use the new yield
  // values.
  Region &newBody = newDispatchOp.getBody();
  rewriter.inlineRegionBefore(dispatchOp.getBody(), newBody, newBody.begin());
  {
    Operation *terminator = newBody.front().getTerminator();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(terminator);
    rewriter.replaceOpWithNewOp<Flow::ReturnOp>(terminator, newYieldVals);
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
    auto newExpandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, origResult.getType(), returnValue,
        allReassociationIndicesRef.front());
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
static void updateConsumersFromProducers(
    ArrayRef<Operation *> slice,
    llvm::MapVector<linalg::GenericOp, CollapseInfo> &opMap) {

  // Slice is topologically sorted to ensure that `op`'s producers have been
  // updated before we visit it.
  for (auto op : slice) {
    auto genericOp = cast<linalg::GenericOp>(op);
    CollapseInfo &consumerInfo = opMap[genericOp];

    for (auto operand : genericOp.getDpsInputOperands()) {
      auto definingOp = operand->get().getDefiningOp();
      if (!definingOp || isNonNullAndOutsideDispatch(definingOp)) {
        continue;
      }

      // Track the dimensions that are not collapsable by this current op.
      // Initialize this with all loops in thel producer. Note: the dims are
      // relative to the consumers iteration space, not the producers. This
      // cannot be done via union of producer and consumer collapsable loops
      // because the consumer may have loops that the producer does not.
      CollapseInfo::CollapsableLoopsSet producerUncollapsable;
      for (auto expr : genericOp.getMatchingIndexingMap(operand).getResults()) {
        producerUncollapsable.insert(cast<AffineDimExpr>(expr).getPosition());
      }

      auto producerOp = dyn_cast<linalg::GenericOp>(definingOp);
      FailureOr<AffineMap> mapping =
          getProducerLoopToConsumerLoopsMap(*operand);

      // If the producer is not a generic or there is no mapping, the tensor is
      // not collapsable. So, all dimensions of the producer are uncollapsable.
      if (!producerOp || failed(mapping)) {
        consumerInfo.updateCollapseViaSubtract(producerUncollapsable);
        continue;
      }

      CollapseInfo &producerInfo = opMap[producerOp];
      FailureOr<CollapseInfo::CollapsableLoopsSet> producerCollapsable =
          producerInfo.getTransformedCollapsableLoops(mapping.value());
      if (!failed(producerCollapsable)) {
        producerUncollapsable.set_subtract(producerCollapsable.value());
      }

      consumerInfo.updateCollapseViaSubtract(producerUncollapsable);
    }
  }
}

// For each producer, use it's consumers to constrain which dimensions it will
// collapse. `slice` is expected to be topologically sorted (getBackwardSlice
// does this automatically).
static void updateProducersFromConsumers(
    ArrayRef<Operation *> slice,
    llvm::MapVector<linalg::GenericOp, CollapseInfo> &opMap) {

  // Iterate over `slice` in reverse so that we visit each `op` 's consumer
  // before visiting `op`.
  for (auto op : llvm::reverse(slice)) {
    auto genericConsumer = cast<linalg::GenericOp>(op);
    const CollapseInfo &consumerInfo = opMap[genericConsumer];
    for (auto operand : genericConsumer.getDpsInputOperands()) {
      auto definingOp = operand->get().getDefiningOp();
      if (!definingOp || isNonNullAndOutsideDispatch(definingOp)) {
        continue;
      }
      auto genericProducer = dyn_cast<linalg::GenericOp>(definingOp);
      if (!genericProducer) {
        continue;
      }

      // Get a mapping from the consumer's iteration space to the producer's.
      CollapseInfo &producerInfo = opMap[genericProducer];
      FailureOr<AffineMap> consumerToProducerMap =
          getConsumerLoopToProducerLoopsMap(*operand);
      if (failed(consumerToProducerMap)) {
        producerInfo.clear();
        continue;
      }

      // Use the map to get the consumer's collapsable loops in terms of the
      // producer.
      auto consumerCollapsable = consumerInfo.getTransformedCollapsableLoops(
          consumerToProducerMap.value());
      if (failed(consumerCollapsable)) {
        producerInfo.clear();
        continue;
      }
      // Only loops collapsable in both the consumer and producer may be
      // collapsed.
      producerInfo.updateCollapseViaIntersect(consumerCollapsable.value());
    }
  }
}

// Construct a DAG of `linalg.generic` operations with 1 root op. Find
// dimensions that can be collapsed all the way from the root to the leaves,
// ensuring that all `collapse_shape` ops can be hoisted out of the dispatch.
static bool collapseDimensionsForDispatch(IRRewriter &rewriter,
                                          DispatchRegionOp &regionOp) {
  // Only collapse dispatches with 1 block
  if (!llvm::hasSingleElement(regionOp.getBody())) {
    return false;
  }
  // Step 1. Find the root linalg.generic Op
  std::optional<linalg::GenericOp> rootGenericOp = findRootGenericOp(regionOp);
  if (!rootGenericOp.has_value())
    return false;

  // Step 2. Get slice of all linalg.generic ops in the dispatch
  BackwardSliceOptions sliceOptions;
  sliceOptions.inclusive = true;
  sliceOptions.omitBlockArguments = true;
  sliceOptions.filter = [&](Operation *op) -> bool {
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    auto parentOp = op->getParentOfType<DispatchRegionOp>();
    return genericOp && isEligibleForCollapse(genericOp) &&
           parentOp == regionOp;
  };
  SetVector<Operation *> slice;
  getBackwardSlice(rootGenericOp->getOperation(), &slice, sliceOptions);

  // Step 3. Populate each op's info with a maximally collapsable reassociation
  // indicies
  llvm::MapVector<linalg::GenericOp, CollapseInfo> opMap;
  for (auto *op : slice) {
    auto genericOp = cast<linalg::GenericOp>(op);
    opMap[genericOp] = CollapseInfo(getCollapsibleLoops(genericOp));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[CollapseDims] : After initializing opMap\n";
    for (auto &[op, info] : opMap) {
      info.dump();
      llvm::dbgs() << "\n";
      op.dump();
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\n";
  });

  // Step 4. For each producer, reduce the number of collapsed dimensions
  // based on the dimensions that it's consumers can collapse.
  updateProducersFromConsumers(slice.getArrayRef(), opMap);

  LLVM_DEBUG({
    llvm::dbgs() << "[CollapseDims] : After updating producers: \n";
    for (auto &[op, info] : opMap) {
      info.dump();
      llvm::dbgs() << "\n";
      op.dump();
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\n";
  });

  // Step 5. For each consumer, update it's CollapseInfo to only collapse
  // dimensions that all of its producers can collapse. This ensures that all
  // reshapes can be propagated to leafs and be hoisted out of the dispatch.
  updateConsumersFromProducers(slice.getArrayRef(), opMap);

  LLVM_DEBUG({
    llvm::dbgs() << "[CollapseDims] : After updating consumers: \n";
    for (auto &[op, info] : opMap) {
      info.dump();
      llvm::dbgs() << "\n";
      op.dump();
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\n";
  });
  bool didCollapse = false;

  // Step 6. Collapse dimensions based on each op's CollapseInfo
  for (auto &[genericOp, info] : opMap) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(genericOp);
    FailureOr<linalg::CollapseResult> maybeReplacements =
        mlir::linalg::collapseOpIterationDims(genericOp, info.getReassocation(),
                                              rewriter);
    if (failed(maybeReplacements))
      continue;
    didCollapse = true;
    rewriter.replaceOp(genericOp, maybeReplacements->results);
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

  SmallVector<DispatchRegionOp> modifiedDispatchOps;
  funcOp->walk([&](DispatchRegionOp dispatchOp) {
    if (collapseDimensionsForDispatch(rewriter, dispatchOp)) {
      modifiedDispatchOps.push_back(dispatchOp);
    }
  });

  LLVM_DEBUG({
    llvm::dbgs() << "[CollapseDims] : After collapsing generic ops: \n";
    funcOp.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Move all the `tensor.collapse_shape` leafs  and `tensor.expand_shape` roots
  // of the modified dispatches out of the dispatch.
  for (auto dispatchOp : modifiedDispatchOps) {
    // Hoist tensor reshape ops out of dispatch region first. Otherwise, the
    // reshape(cst) will be folded into a constant living in the dispatch. It
    // could introduce big constants inlined in the dispatch.
    FailureOr<DispatchRegionOp> newDispatchOp =
        hoistTensorReshapesOutOfDispatchRegion(
            rewriter, cast<DispatchRegionOp>(dispatchOp));
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
    block.walk([&](Operation *op) {
      if (isa<tensor::CollapseShapeOp>(op)) {
        candidateOps.push_back(op);
      }
    });
    if (failed(
            applyOpPatternsAndFold(candidateOps, std::move(moveReshapeOps)))) {
      funcOp.emitOpError(
          "failed to propagate reshape ops introduced during collapse");
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::IREE::Flow
