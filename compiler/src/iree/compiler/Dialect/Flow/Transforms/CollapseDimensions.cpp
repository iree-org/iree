// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <deque>

#define DEBUG_TYPE "iree-flow-collapse-dimensions"

namespace mlir::iree_compiler::IREE::Flow {

namespace {
/// Pass declaration.
struct CollapseDimensionsPass
    : public CollapseDimensionsBase<CollapseDimensionsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {}
  CollapseDimensionsPass() {}
  CollapseDimensionsPass(const CollapseDimensionsPass &pass)
      : CollapseDimensionsPass() {}
  void runOnOperation() override;
};
} // namespace

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
    if (!range.empty()) {
      if (!hasAllMapsSameSequence(preExpr, nextExpr) ||
          !hasSameIteratorType(preExpr, nextExpr)) {
        if (range.size() > 1) {
          contiguousLoops.push_back({range.begin(), range.end()});
        }
        range.clear();
      }
    }
    range.push_back(cast<AffineDimExpr>(nextExpr).getPosition());
    preExpr = nextExpr;
  }
  if (range.size() > 1)
    contiguousLoops.push_back(range);

  LLVM_DEBUG({
    llvm::dbgs() << "Collapsing dimensions if possible: ";
    for (auto indices : contiguousLoops) {
      llvm::dbgs() << "[";
      for (auto idx : indices)
        llvm::dbgs() << idx << ",";
      llvm::dbgs() << "]\t";
    }
    llvm::dbgs() << "\n";
  });

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

  return true;
}

/// Traverses all the the Ops in DispatchRegionOps and finds linalg.generic Op
/// without any producers.
static FailureOr<linalg::GenericOp>
findRootGenericOp(DispatchRegionOp regionOp) {
  if (!llvm::hasSingleElement(regionOp.getBody())) {
    return failure();
  }

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

  // Check that the operands of the generic op are defined outside the dispatch.
  for (OpOperand *inputOperands : collapsibleOp.getDpsInputOperands()) {
    Operation *definingOp = inputOperands->get().getDefiningOp();
    if (definingOp &&
        definingOp->getParentOfType<DispatchRegionOp>() == regionOp) {
      return failure();
    }
  }

  // Check that the output is either a `tensor.empty` or a `linalg.fill` op by
  // traversing the operations that define the `init` operands of the
  // `collapsibleOp`.
  std::deque<Operation *> worklist;
  llvm::SmallDenseSet<Operation *> visited;
  auto addDefiningOpToWorklist = [&](Value v) {
    Operation *definingOp = v.getDefiningOp();
    if (definingOp &&
        definingOp->getParentOfType<DispatchRegionOp>() == regionOp &&
        !visited.count(definingOp)) {
      worklist.push_back(definingOp);
      visited.insert(definingOp);
    }
  };
  for (Value initOperand : collapsibleOp.getDpsInits()) {
    addDefiningOpToWorklist(initOperand);
  }

  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop_front();
    if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      addDefiningOpToWorklist(fillOp.getDpsInitOperand(0)->get());
      continue;
    }
    if (isa<tensor::EmptyOp>(op)) {
      continue;
    }
    return failure();
  }
  return collapsibleOp;
}

/// Hoist `tensor.collapse_shape` ops at the beginning of the `dispatchOp`
/// and `tensor.expand_shape` ops at the end of the `dispatchOp`, out of the
/// dispatch.
static FailureOr<DispatchRegionOp>
hoistTensorReshapesOutOfDispatchRegion(RewriterBase &rewriter,
                                       DispatchRegionOp dispatchOp) {
  // Only do this for `dispatchOp` with a single operation.
  if (!llvm::hasSingleElement(dispatchOp.getBody())) {
    return failure();
  }
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

/// Traverses DispatchRegionOps to find linalg genericOps that has no
/// producers and tries to collapse its dimensions.
static bool collapseDimensions(IRRewriter &rewriter,
                               DispatchRegionOp &regionOp) {
  // Step 1. Find the root linalg.generic Op with no producer
  std::optional<linalg::GenericOp> genericOp = findRootGenericOp(regionOp);
  if (!genericOp.has_value())
    return false;

  // Step 2. Check whether it is possible to collapse
  if (!isEligibleForCollapse(genericOp.value()))
    return false;
  SmallVector<ReassociationIndices> collapseIndices;
  collapseIndices = getCollapsibleLoops(genericOp.value());
  if (collapseIndices.empty())
    return false;

  // Step 3. Collapse dimensions
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(genericOp.value());

  FailureOr<linalg::CollapseResult> maybeReplacements =
      mlir::linalg::collapseOpIterationDims(genericOp.value(), collapseIndices,
                                            rewriter);
  if (failed(maybeReplacements))
    return false;
  rewriter.replaceOp(genericOp.value(), maybeReplacements->results);
  return true;
}

void CollapseDimensionsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = funcOp->getContext();
  IRRewriter rewriter(context);

  SmallVector<DispatchRegionOp> modifiedDispatchOps;
  funcOp->walk([&](DispatchRegionOp dispatchOp) {
    if (collapseDimensions(rewriter, dispatchOp)) {
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

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCollapseDimensionsPass() {
  return std::make_unique<CollapseDimensionsPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
