// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"

#include <iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

// NOTE: These flags are added for experimental purposes only
// for developer control. These should be treated as internal
// compiler implementation details.
static llvm::cl::opt<int> clInlineConstantByteLength(
    "iree-flow-inline-constants-max-byte-length",
    llvm::cl::desc("Maximum byte-length of tensor constant that can be inlined "
                   "into a dispatch region or 0 to disable inlining."),
    llvm::cl::init(256));

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

#define DEBUG_TYPE "iree-flow-region-op-utils"

//===----------------------------------------------------------------------===//
// Methods for getting the workload information for dispatch region creation.
//===----------------------------------------------------------------------===//

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

static SmallVector<Range> getLoopRangesFromValue(Value source, Location loc,
                                                 OpBuilder &builder) {
  SmallVector<OpFoldResult> dimValues =
      tensor::createDimValues(builder, loc, source);
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  return llvm::to_vector(llvm::map_range(dimValues, [&](OpFoldResult dimValue) {
    return Range{zero, dimValue, one};
  }));
}

static SmallVector<Range> getLoopRangesImpl(tensor::ExtractSliceOp sliceOp,
                                            Location loc, OpBuilder &builder) {
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  ReifiedRankedShapedTypeDims resultDims;
  LogicalResult status = sliceOp.reifyResultShapes(builder, resultDims);
  (void)status;
  assert(succeeded(status) && "reifyResultShapes failed");
  return llvm::to_vector(llvm::map_range(resultDims[0], [&](OpFoldResult v) {
    return Range{zero, v, one};
  }));
}

/// For a given operation returns the loop ranges needed to compute the op.
SmallVector<Range> Flow::getLoopRanges(Operation *op, Location loc,
                                       OpBuilder &builder) {
  return llvm::TypeSwitch<Operation *, SmallVector<Range>>(op)
      .Case<LinalgExt::SetEncodingOp, LinalgExt::UnsetEncodingOp,
            tensor::InsertSliceOp>([&](auto op) {
        return getLoopRangesFromValue(op.getSource(), loc, builder);
      })
      .Case<tensor::ExtractSliceOp>([&](auto sliceOp) {
        return getLoopRangesImpl(sliceOp, loc, builder);
      })
      .Case<TilingInterface>([&](TilingInterface op) {
        return getLoopRangesImpl(op, loc, builder);
      })
      .Default([](Operation *op) -> SmallVector<Range> {
        llvm_unreachable("op not supported");
      });
}

/// Compute the workload to use for the workgroup based on the root op.
static SmallVector<Value> getWorkloadForRootOp(OpBuilder &builder,
                                               Operation *rootOp) {
  // Compute workgroup count to use for the dispatch op. These are the ranges
  // of the outermost parallel loops that can be distributed.
  Location loc = rootOp->getLoc();
  SmallVector<Range> loopRanges = Flow::getLoopRanges(rootOp, loc, builder);
  AffineExpr s0, s1, s2;
  bindSymbols(builder.getContext(), s0, s1, s2);
  AffineMap workload = AffineMap::get(0, 3, (s1 - s0).ceilDiv(s2));
  return llvm::to_vector(llvm::map_range(loopRanges, [&](Range r) -> Value {
    Value offset = getValueOrCreateConstantIndexOp(builder, loc, r.offset);
    Value size = getValueOrCreateConstantIndexOp(builder, loc, r.size);
    Value stride = getValueOrCreateConstantIndexOp(builder, loc, r.stride);
    return builder.create<AffineApplyOp>(rootOp->getLoc(), workload,
                                         ValueRange{offset, size, stride});
  }));
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

  // Helper function that generates tensor.dim ops.
  auto emitTensorDimOps = [&]() {
    for (int64_t i = 0; i < shapedType.getRank(); ++i) {
      if (shapedType.isDynamicDim(i)) {
        Value dim = b.create<tensor::DimOp>(value.getLoc(), value, i);
        dynamicDims.push_back(dim);
      }
    }
  };

  // Case 2: Value is a block argument.
  if (auto bbArg = value.dyn_cast<BlockArgument>()) {
    b.setInsertionPointToStart(bbArg.getOwner());
    emitTensorDimOps();
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
        dynamicDims.push_back(dims[opResult.getResultNumber()][i].get<Value>());
    return success();
  }

  // None of the above. Insert tensor.dim ops.
  b.setInsertionPointAfter(op);
  emitTensorDimOps();
  return success();
}

// Append a result to the given DispatchRegionOp. The newly created
// DispatchRegionOp is returned.
FailureOr<Flow::DispatchRegionOp> Flow::appendDispatchRegionResult(
    RewriterBase &rewriter, Flow::DispatchRegionOp regionOp, Value result,
    const SmallVector<Value> &dynamicDims) {
#ifndef NDEBUG
  auto tensorType = result.getType().cast<RankedTensorType>();
  assert(tensorType.getNumDynamicDims() == dynamicDims.size() &&
         "incorrect number of dynamicDims provided");
#endif  // NDEBUG

  OpBuilder::InsertionGuard guard(rewriter);

  // Determine dynamic result dims.
  rewriter.setInsertionPoint(regionOp);
  SmallVector<Value> regionDynamicDims(regionOp.getResultDims().begin(),
                                       regionOp.getResultDims().end());
  regionDynamicDims.append(dynamicDims);

  // Determine result types of new RegionOp.
  SmallVector<Type> resultTypes(regionOp.getResultTypes().begin(),
                                regionOp.getResultTypes().end());
  resultTypes.push_back(result.getType());

  // Create new DispatchRegionOp and move over the body.
  auto newRegionOp = rewriter.create<Flow::DispatchRegionOp>(
      regionOp->getLoc(), resultTypes, regionDynamicDims,
      regionOp.getWorkload());
  newRegionOp.getBody().takeBody(regionOp.getBody());
  rewriter.replaceOp(
      regionOp, newRegionOp.getResults().take_front(regionOp->getNumResults()));

  // Update terminator.
  Flow::ReturnOp returnOp =
      cast<Flow::ReturnOp>(newRegionOp.getBody().front().getTerminator());
  SmallVector<Value> returnedValues(returnOp.getOperands().begin(),
                                    returnOp.getOperands().end());
  returnedValues.push_back(result);
  returnOp.getOperandsMutable().assign(returnedValues);

  return newRegionOp;
}

Flow::DispatchRegionOp Flow::makeDispatchRegionWithWorkload(
    OpBuilder &builder, Location loc, SmallVector<Value> workload) {
  OpBuilder::InsertionGuard guard(builder);

  // Create RegionOp.
  auto regionOp = builder.create<Flow::DispatchRegionOp>(
      loc, /*resultTypes=*/TypeRange(), /*dynamicDims=*/ValueRange(),
      /*workload=*/workload);
  Block &body = regionOp.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<Flow::ReturnOp>(loc, ValueRange());
  return regionOp;
}

// Clone a `target` op that is preceding the given dispatch region op into the
// dispatch region.
FailureOr<Operation *> Flow::clonePrecedingOpIntoDispatchRegion(
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

  return newTargetOp;
}

// Move a `target` op that is preceding the given dispatch region op into the
// dispatch region.
FailureOr<Flow::DispatchRegionOp> Flow::movePrecedingOpIntoDispatchRegion(
    RewriterBase &rewriter, Operation *target,
    Flow::DispatchRegionOp regionOp) {
#ifndef NDEBUG
  DominanceInfo domInfo;
  for (OpOperand &use : target->getUses()) {
    Operation *user = use.getOwner();
    if (regionOp->isProperAncestor(use.getOwner()) ||
        isa<tensor::DimOp>(user)) {
      continue;
    }
    assert(domInfo.properlyDominates(regionOp, user) &&
           "found use that does not post-dominate target");
  }
#endif  // NDEBUG

  Block &body = regionOp.getBody().front();

  // Gather all uses of `target`.
  SmallVector<OpOperand *> usesOutsideOfRegion;
  for (OpOperand &use : target->getUses())
    if (!regionOp->isProperAncestor(use.getOwner()))
      usesOutsideOfRegion.push_back(&use);

  // Compute dynamic result dims.
  SmallVector<SmallVector<Value>> dynamicDims;
  for (Value v : target->getResults()) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(target);
    SmallVector<Value> &dims = dynamicDims.emplace_back();
    if (failed(reifyDynamicResultDims(rewriter, v, dims))) return failure();
  }

  // Move op into dispatch region.
  target->moveBefore(&body.front());

  // Replace all uses outside of the dispatch region.
  if (!usesOutsideOfRegion.empty()) {
    unsigned previousNumResults = regionOp->getNumResults();

    // Note: Appending results one-by-one here so that this can be extended to
    // specific results in the future. Many ops have just one result, so this
    // should not be a large overhead.
    for (const auto &it : llvm::enumerate(target->getResults())) {
      auto newRegionOp = appendDispatchRegionResult(
          rewriter, regionOp, it.value(), dynamicDims[it.index()]);
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
    RewriterBase &rewriter, Operation *op,
    std::optional<Flow::WorkloadBuilder> workloadBuilder) {
  SmallVector<Value> workload;

  OpBuilder::InsertionGuard g(rewriter);
  if (workloadBuilder) {
    rewriter.setInsertionPoint(op);
    FailureOr<SmallVector<Value>> maybeWorkload =
        getWorkloadForRootOp(rewriter, op);
    if (failed(maybeWorkload)) {
      return op->emitOpError("failed to compute workload for op");
    }
    workload = *maybeWorkload;
  }

  rewriter.setInsertionPointAfter(op);
  Flow::DispatchRegionOp regionOp =
      Flow::makeDispatchRegionWithWorkload(rewriter, op->getLoc(), workload);

  // Move the op into the dispatch region.
  auto newRegionOp = movePrecedingOpIntoDispatchRegion(rewriter, op, regionOp);

  // Generate workload_count region
  if (succeeded(newRegionOp) && workloadBuilder.has_value()) {
    Region &workgroupCountRegion = newRegionOp->getWorkgroupCount();
    Block *body = rewriter.createBlock(&workgroupCountRegion);
    SmallVector<BlockArgument> workloadArgs;
    Location loc = newRegionOp->getLoc();
    for (Value v : workload) {
      workloadArgs.push_back(body->addArgument(v.getType(), loc));
    }
    rewriter.setInsertionPointToStart(body);
    workloadBuilder->regionBuilder(rewriter, loc, workloadArgs);
  }

  return newRegionOp;
}
//===---------------------------------------------------------------------===//
// Utilities to make a dispatch region isolated from above
//===---------------------------------------------------------------------===//

/// Operations that are cloned into dispatch regions formed with other
/// operations as roots.
bool Flow::isClonableIntoDispatchOp(Operation *op) {
  // TODO(#8637): `tensor.collapse_shape` and `tensor.expand_shape` are
  // trivially clonable too, but they cause problems
  // with bufferization. Make them clonable when fixed.
  if (isa<AffineApplyOp, arith::IndexCastOp, linalg::FillOp, tensor::EmptyOp,
          tensor::CastOp, tensor::ExtractOp, tensor::ExtractSliceOp,
          tensor::PadOp>(op)) {
    return true;
  }
  if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
    if (clInlineConstantByteLength == 0) return false;
    auto constantValueAttr = constantOp.getValue();
    auto constantType = constantOp.getType();
    if (constantValueAttr.isa<SplatElementsAttr>()) {
      return true;
    } else if (auto denseAttr =
                   constantValueAttr.dyn_cast<DenseElementsAttr>()) {
      auto shapedType = constantOp.getType().cast<ShapedType>();
      uint64_t estimatedByteLength =
          (shapedType.getNumElements() * shapedType.getElementTypeBitWidth()) /
          8;
      return denseAttr.isSplat() ||
             estimatedByteLength <= clInlineConstantByteLength;
    } else if (constantType.isIntOrIndexOrFloat()) {
      return true;
    }
  }
  if (llvm::all_of(op->getOperands(),
                   [&](Value v) { return v.getType().isIntOrFloat(); }) &&
      llvm::all_of(op->getResults(),
                   [&](Value v) { return v.getType().isIntOrFloat(); })) {
    return true;
  }
  return false;
}

/// Checks if the `Value` has a use within the dispatch that is unfusable.
static bool hasUnfusableUseInDispatch(Value v, Operation *dispatchOp) {
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    Operation *ownerWorkgroupsOp =
        user->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>();
    Operation *ownerRegionOp =
        user->getParentOfType<IREE::Flow::DispatchRegionOp>();
    Operation *owner = ownerWorkgroupsOp ? ownerWorkgroupsOp : ownerRegionOp;

    // Ignore uses outside of dispatch workgroups op.
    if (owner != dispatchOp) continue;

    // Cannot fuse producer of `dest` with `tensor.insert_slice`.
    if (auto insertSliceUser = dyn_cast<tensor::InsertSliceOp>(user)) {
      if (insertSliceUser.getDest() == v) return true;
    }
  }
  return false;
}

/// Collect all ops that should be cloned into the given dispatch region op.
static SmallVector<Operation *> getCloneableOps(
    Flow::DispatchRegionOp regionOp) {
  // Find values that are used inside of the dispatch region but defined outside
  // of the dispatch region.
  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(regionOp.getBody(), valuesDefinedAbove);
  if (valuesDefinedAbove.empty()) return {};

  // Traverse the defining ops of these values (and the ops on their reverse
  // SSA use-def chain).
  SmallVector<Operation *> result;
  llvm::SetVector<Value> visited;
  SmallVector<Value, 4> worklist;
  worklist.assign(valuesDefinedAbove.begin(), valuesDefinedAbove.end());
  while (!worklist.empty()) {
    Value outsideValue = worklist.pop_back_val();
    // Skip values that were already visited.
    if (visited.count(outsideValue)) continue;
    visited.insert(outsideValue);

    Operation *definingOp = outsideValue.getDefiningOp();
    if (!definingOp || !Flow::isClonableIntoDispatchOp(definingOp) ||
        hasUnfusableUseInDispatch(outsideValue, regionOp)) {
      valuesDefinedAbove.insert(outsideValue);
      continue;
    }
    result.push_back(definingOp);
    worklist.append(definingOp->operand_begin(), definingOp->operand_end());
    llvm::SetVector<Value> nestedValues;
    mlir::getUsedValuesDefinedAbove(definingOp->getRegions(), nestedValues);
    worklist.append(nestedValues.begin(), nestedValues.end());
  }

  return result;
}

/// Clone producers into the dispatch region.
LogicalResult Flow::cloneProducersToRegion(RewriterBase &rewriter,
                                           Flow::DispatchRegionOp regionOp) {
  SmallVector<Operation *> cloneableOps = getCloneableOps(regionOp);
  bool sortResult = mlir::computeTopologicalSorting(cloneableOps);
  (void)sortResult;
  assert(sortResult && "could not compute topological sorting");

  for (Operation *producer : llvm::reverse(cloneableOps)) {
    if (failed(
            clonePrecedingOpIntoDispatchRegion(rewriter, producer, regionOp))) {
      return failure();
    }
  }

  return success();
}
