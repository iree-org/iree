// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

#define DEBUG_TYPE "iree-flow-region-op-utils"

static SmallVector<Range> getLoopRangesImpl(TilingInterface tilableOp,
                                            Location loc, OpBuilder &builder) {
  SmallVector<Range> loopRanges = tilableOp.getIterationDomain(builder);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  for (auto iteratorType : llvm::enumerate(tilableOp.getLoopIteratorTypes())) {
    if (iteratorType.value() == utils::IteratorType::reduction) {
      loopRanges[iteratorType.index()].size = one;
    }
  }
  return loopRanges;
}

static SmallVector<Range> getLoopRangesImpl(tensor::InsertSliceOp insertSliceOp,
                                            Location loc, OpBuilder &builder) {
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  Value source = insertSliceOp.getSource();
  SmallVector<Range> loopRanges(insertSliceOp.getSourceType().getRank(),
                                Range{zero, one, one});
  for (auto dim : llvm::seq<unsigned>(0, loopRanges.size())) {
    loopRanges[dim].size =
        builder.create<tensor::DimOp>(loc, source, dim).getResult();
  }
  return loopRanges;
}

static SmallVector<Range> getLoopRangesImpl(tensor::ExtractSliceOp sliceOp,
                                            Location loc, OpBuilder &builder) {
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  ReifiedRankedShapedTypeDims resultDims;
  LogicalResult status = sliceOp.reifyResultShapes(builder, resultDims);
  (void)status;
  assert(succeeded(status) && "reifyResultShapes failed");
  return llvm::to_vector(llvm::map_range(resultDims[0], [&](Value v) {
    return Range{zero, v, one};
  }));
}

/// For a given operation returns the loop ranges needed to compute the op.
SmallVector<Range> Flow::getLoopRanges(Operation *op, Location loc,
                                       OpBuilder &builder) {
  return llvm::TypeSwitch<Operation *, SmallVector<Range>>(op)
      .Case([&](TilingInterface op) {
        return getLoopRangesImpl(op, loc, builder);
      })
      .Case([&](tensor::InsertSliceOp op) {
        return getLoopRangesImpl(op, loc, builder);
      })
      .Case([&](tensor::ExtractSliceOp op) {
        return getLoopRangesImpl(op, loc, builder);
      })
      .Default([](Operation *op) -> SmallVector<Range> {
        llvm_unreachable("op not supported");
      });
}

/// Return `true` if the given type is a ShapedType and has at least one
/// dynamic dimension.
static bool hasDynamicShape(Type t) {
  auto shapedType = t.dyn_cast<ShapedType>();
  if (!shapedType) return false;
  return !shapedType.hasStaticShape();
}

/// Reify the dynamic dimensions of the given value.
LogicalResult Flow::reifyDynamicResultDims(OpBuilder &b, Value value,
                                           SmallVector<Value> &dynamicDims) {
  OpBuilder::InsertionGuard guard(b);

  // Case 1: No dynamic result dims.
  if (!hasDynamicShape(value.getType())) return success();

  // There is at least one dynamic dimension, continue...
  ShapedType shapedType = value.getType().cast<ShapedType>();

  // Case 2: Value is a block argument.
  if (auto bbArg = value.dyn_cast<BlockArgument>()) {
    b.setInsertionPointToStart(bbArg.getOwner());
    for (int64_t i = 0; i < shapedType.getRank(); ++i) {
      if (shapedType.isDynamicDim(i)) {
        Value dim = b.create<tensor::DimOp>(bbArg.getLoc(), bbArg, i);
        dynamicDims.push_back(dim);
      }
    }
    return success();
  }

  // Value is an OpResult.
  Operation *op = value.getDefiningOp();
  OpResult opResult = value.cast<OpResult>();
  b.setInsertionPoint(op);

  // Case 3: Value is tied. Reify the dimensions of the tied operand.
  auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
  if (tiedOp) {
    Value tiedOperand = tiedOp.getTiedResultOperand(value);
    if (tiedOperand && tiedOperand.getType() == value.getType())
      return reifyDynamicResultDims(b, tiedOperand, dynamicDims);
  }

  // Case 4: Query ShapeAwareOpInterface.
  auto shapeAwareOp = dyn_cast<IREE::Util::ShapeAwareOpInterface>(op);
  if (shapeAwareOp) {
    ValueRange dims =
        shapeAwareOp.getResultDynamicDims(opResult.getResultNumber());
    dynamicDims.append(dims.begin(), dims.end());
    return success();
  }

  // Case 5: Query ReifyRankedShapedTypeOpInterface.
  auto reifyShapeOp = dyn_cast<ReifyRankedShapedTypeOpInterface>(op);
  if (reifyShapeOp) {
    ReifiedRankedShapedTypeDims dims;
    if (failed(reifyShapeOp.reifyResultShapes(b, dims))) return failure();
    for (int64_t i = 0; i < shapedType.getRank(); ++i)
      if (shapedType.isDynamicDim(i))
        dynamicDims.push_back(dims[opResult.getResultNumber()][i]);
    return success();
  }

  return failure();
}

// Append a result to the given DispatchRegionOp. The newly created
// DispatchRegionOp is returned.
FailureOr<Flow::DispatchRegionOp> Flow::appendDispatchRegionResult(
    RewriterBase &rewriter, Flow::DispatchRegionOp regionOp, Value result) {
  OpBuilder::InsertionGuard guard(rewriter);

  // Determine dynamic result dims.
  rewriter.setInsertionPoint(regionOp);
  SmallVector<Value> dynamicDims(regionOp.getResultDims().begin(),
                                 regionOp.getResultDims().end());
  if (failed(reifyDynamicResultDims(rewriter, result, dynamicDims)))
    return failure();

  // Determine result types of new RegionOp.
  SmallVector<Type> resultTypes(regionOp.getResultTypes().begin(),
                                regionOp.getResultTypes().end());
  resultTypes.push_back(result.getType());

  // Create new DispatchRegionOp and move over the body.
  auto newRegionOp = rewriter.create<Flow::DispatchRegionOp>(
      regionOp->getLoc(), resultTypes, dynamicDims);
  newRegionOp.getBody().takeBody(regionOp.getBody());
  rewriter.replaceOp(
      regionOp, newRegionOp.getResults().take_front(regionOp->getNumResults()));

  // Update terminator.
  Flow::ReturnOp returnOp =
      cast<Flow::ReturnOp>(newRegionOp.getBody().front().getTerminator());
  SmallVector<Value> returnedValues(returnOp.getOperands().begin(),
                                    returnOp.getOperands().end());
  returnedValues.push_back(result);
  returnOp.operandsMutable().assign(returnedValues);

  return newRegionOp;
}

Flow::DispatchRegionOp Flow::makeEmptyDispatchRegion(OpBuilder &builder,
                                                     Location loc) {
  OpBuilder::InsertionGuard guard(builder);

  // Create RegionOp.
  auto regionOp = builder.create<Flow::DispatchRegionOp>(
      loc, /*resultTypes=*/TypeRange(), /*dynamicDims=*/ValueRange());
  Block &body = regionOp.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<Flow::ReturnOp>(loc, ValueRange());

  return regionOp;
}

// Clone a `target` op that is preceding the given dispatch region op into the
// dispatch region.
LogicalResult Flow::clonePrecedingOpIntoDispatchRegion(
    RewriterBase &rewriter, Operation *target,
    Flow::DispatchRegionOp regionOp) {
  Block &body = regionOp.getBody().front();

  // Gather all uses of `target`.
  SmallVector<OpOperand *> usesInsideOfRegion;
  for (OpOperand &use : target->getUses()) {
    if (regionOp->isProperAncestor(use.getOwner()))
      usesInsideOfRegion.push_back(&use);
  }

  // Clone op into dispatch region.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&body);
  Operation *newTargetOp = rewriter.clone(*target);

  // Replace all uses in the dispatch region.
  for (OpOperand *use : usesInsideOfRegion) {
    rewriter.updateRootInPlace(use->getOwner(), [&]() {
      use->set(newTargetOp->getResult(
          use->get().cast<OpResult>().getResultNumber()));
    });
  }

  return success();
}

// Move a `target` op that is preceding the given dispatch region op into the
// dispatch region.
FailureOr<Flow::DispatchRegionOp> Flow::movePrecedingOpIntoDispatchRegion(
    RewriterBase &rewriter, Operation *target,
    Flow::DispatchRegionOp regionOp) {
#ifndef NDEBUG
  DominanceInfo domInfo;
  for (OpOperand &use : target->getUses()) {
    if (regionOp->isProperAncestor(use.getOwner())) continue;
    assert(domInfo.properlyDominates(regionOp, use.getOwner()) &&
           "found use that does not post-dominate target");
  }
#endif  // NDEBUG

  Block &body = regionOp.getBody().front();

  // Gather all uses of `target`.
  SmallVector<OpOperand *> usesOutsideOfRegion;
  for (OpOperand &use : target->getUses())
    if (!regionOp->isProperAncestor(use.getOwner()))
      usesOutsideOfRegion.push_back(&use);

  // Move op into dispatch region.
  target->moveBefore(&body.front());

  // Replace all uses outside of the dispatch region.
  if (!usesOutsideOfRegion.empty()) {
    unsigned previousNumResults = regionOp->getNumResults();

    // Note: Appending results one-by-one here so that this can be extended to
    // specific results in the future. Many ops have just one result, so this
    // should not be a large overhead.
    for (Value v : target->getResults()) {
      auto newRegionOp = appendDispatchRegionResult(rewriter, regionOp, v);
      if (failed(newRegionOp)) return failure();
      regionOp = *newRegionOp;
    }

    // Replace uses of `target` after the dispatch region.
    for (OpOperand *use : usesOutsideOfRegion) {
      rewriter.updateRootInPlace(use->getOwner(), [&]() {
        use->set(
            regionOp->getResult(previousNumResults +
                                use->get().cast<OpResult>().getResultNumber()));
      });
    }
  }

  return regionOp;
}

FailureOr<Flow::DispatchRegionOp> Flow::wrapOpInDispatchRegion(
    RewriterBase &rewriter, Operation *op) {
  // Make an empty dispatch region right before the op.
  rewriter.setInsertionPointAfter(op);
  Flow::DispatchRegionOp regionOp =
      Flow::makeEmptyDispatchRegion(rewriter, op->getLoc());

  // Move the op into the dispatch region.
  auto newRegionOp = movePrecedingOpIntoDispatchRegion(rewriter, op, regionOp);
  return newRegionOp;
}

/// Reorders the operations in `ops` such that they could be inlined into the
/// dispatch region in that order to satisfy dependencies.
SmallVector<Operation *> Flow::orderOperations(ArrayRef<Operation *> ops) {
  LLVM_DEBUG({
    llvm::dbgs() << "Ops to be inlined :\n";
    for (auto op : ops) {
      llvm::dbgs() << "\t";
      op->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  llvm::SmallMapVector<Operation *, SmallVector<Operation *>, 16>
      insertAfterMap;
  llvm::SetVector<Operation *> opSet(ops.begin(), ops.end());
  llvm::SetVector<Operation *> leafOps(ops.begin(), ops.end());
  // For each operation compute the list of operations in `ops` that use its
  // results. Also compute the operations that form the leafs of the DAG of
  // operations in `ops`.
  for (auto op : ops) {
    for (auto operand : op->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp || !opSet.count(definingOp)) continue;
      insertAfterMap[definingOp].push_back(op);
      if (leafOps.count(op)) leafOps.remove(op);
    }
  }

  // The leaves are at the head of the ordered list.
  SmallVector<Operation *> orderedOps(leafOps.begin(), leafOps.end());
  orderedOps.reserve(ops.size());
  llvm::SmallPtrSet<Operation *, 16> processed;
  processed.insert(leafOps.begin(), leafOps.end());

  // `readyOps` contains the list of operations that have been just added to the
  // `orderedOps` list. With these marked ready, they might make further
  // operations in `ops` ready as well.
  // The complexity of the algorithm is driven by these
  // - Each operations is added to `readyOps` list at most once, and is removed
  //   after being processed
  // - For every operation in `readyOps` every use of its results (within `ops`)
  //   is looked at once.
  // - For every use, the operands of the user are processed.
  // Assuming operands is O(1), i.e. constant order, the complexity is O(sum of
  // number of uses of each operation). Given that the size of `ops` is at max
  // O(10), and not O(100), this is assumed to be reasonable.
  ArrayRef<Operation *> readyOps(orderedOps);
  size_t startPos = 0;
  while (!readyOps.empty()) {
    auto op = readyOps.front();
    startPos++;
    // Check all uses of `op` within `ops`. If all of the operations that define
    // the operands of the user have been added to `orderedOps`, then the user
    // is ready to be scheduled.
    for (auto insertAfterOp : insertAfterMap[op]) {
      if (processed.count(insertAfterOp)) continue;
      if (llvm::all_of(insertAfterOp->getOperands(), [&](Value operand) {
            Operation *operandDefiningOp = operand.getDefiningOp();
            return !operandDefiningOp || !opSet.count(operandDefiningOp) ||
                   processed.count(operandDefiningOp);
          })) {
        // readyOps.push_back(insertAfterOp);
        orderedOps.push_back(insertAfterOp);
        processed.insert(insertAfterOp);
      }
    }
    readyOps = ArrayRef<Operation *>(orderedOps).drop_front(startPos);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Ops to be inlined (sorted) : \n";
    for (auto op : orderedOps) {
      llvm::dbgs() << "\t";
      op->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });
  assert(orderedOps.size() == ops.size() &&
         "ordering of inlined operations failed");
  return orderedOps;
}
