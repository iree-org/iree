// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-flow-region-op-utils"

// NOTE: These flags are added for experimental purposes only
// for developer control. These should be treated as internal
// compiler implementation details.
static llvm::cl::opt<int> clInlineConstantByteLength(
    "iree-flow-inline-constants-max-byte-length",
    llvm::cl::desc("Maximum byte-length of tensor constant that can be inlined "
                   "into a dispatch region or 0 to disable inlining."),
    llvm::cl::init(256));

// TODO(#18457, #18447): Remove once backends support gather fusion.
static llvm::cl::opt<bool>
    clEnableGatherFusion("iree-flow-enable-gather-fusion",
                         llvm::cl::desc("Fuse gather-like ops with consumer."),
                         llvm::cl::init(false));

namespace mlir::iree_compiler::IREE::Flow {

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
      tensor::getMixedSizes(builder, loc, source);
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  return llvm::map_to_vector(dimValues, [&](OpFoldResult dimValue) {
    return Range{zero, dimValue, one};
  });
}

static SmallVector<Range>
getLoopRangesImpl(ReifyRankedShapedTypeOpInterface shapedOp, Location loc,
                  OpBuilder &builder) {
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  ReifiedRankedShapedTypeDims resultDims;
  LogicalResult status = shapedOp.reifyResultShapes(builder, resultDims);
  (void)status;
  assert(succeeded(status) && "reifyResultShapes failed");
  return llvm::map_to_vector(resultDims[0], [&](OpFoldResult v) {
    return Range{zero, v, one};
  });
}

/// For a given operation returns the loop ranges needed to compute the op.
SmallVector<Range> getLoopRanges(Operation *op, Location loc,
                                 OpBuilder &builder) {
  return llvm::TypeSwitch<Operation *, SmallVector<Range>>(op)
      .Case<Encoding::SetEncodingOp, Encoding::UnsetEncodingOp,
            tensor::InsertSliceOp>([&](auto op) {
        return getLoopRangesFromValue(op.getSource(), loc, builder);
      })
      .Case<TilingInterface>([&](TilingInterface op) {
        return getLoopRangesImpl(op, loc, builder);
      })
      .Case<ReifyRankedShapedTypeOpInterface>([&](auto shapedOp) {
        return getLoopRangesImpl(shapedOp, loc, builder);
      })
      .Default([](Operation *op) -> SmallVector<Range> {
        llvm_unreachable("op not supported");
      });
}

/// Return `true` if an operation is within a `flow.dispatch.region` or
/// `flow.dispatch.workgroups` op.
bool isNonNullAndOutsideDispatch(Operation *op) {
  if (!op)
    return false;
  Operation *parentOp = op->getParentOp();
  while (parentOp) {
    if (isa<IREE::Flow::DispatchRegionOp, IREE::Flow::DispatchWorkgroupsOp>(
            parentOp)) {
      return false;
    }
    parentOp = parentOp->getParentOp();
  }
  return true;
}
bool isNonNullAndOutsideDispatch(ArrayRef<Operation *> operations) {
  return llvm::all_of(operations, [](Operation *op) {
    return isNonNullAndOutsideDispatch(op);
  });
}

/// Compute the workload to use for the workgroup based on the root op.
static SmallVector<Value> getWorkloadForRootOp(OpBuilder &builder,
                                               Operation *rootOp) {
  // Compute workgroup count to use for the dispatch op. These are the ranges
  // of the outermost parallel loops that can be distributed.
  Location loc = rootOp->getLoc();
  SmallVector<Range> loopRanges =
      IREE::Flow::getLoopRanges(rootOp, loc, builder);
  AffineExpr s0, s1, s2;
  bindSymbols(builder.getContext(), s0, s1, s2);
  AffineMap workload = AffineMap::get(0, 3, (s1 - s0).ceilDiv(s2));
  return llvm::map_to_vector(loopRanges, [&](Range r) -> Value {
    Value offset = getValueOrCreateConstantIndexOp(builder, loc, r.offset);
    Value size = getValueOrCreateConstantIndexOp(builder, loc, r.size);
    Value stride = getValueOrCreateConstantIndexOp(builder, loc, r.stride);
    return builder.create<affine::AffineApplyOp>(
        rootOp->getLoc(), workload, ValueRange{offset, size, stride});
  });
}

/// For very specific cases where the shape is data dependent, we cannot
/// use the normal way of workgroup count calculation through use of
/// program slices within the body of the op. This is because
/// the program slice will also include other tensors.. The general solution
/// here requires making the workgroup count calculation also run on the device.
/// This isnt plumbed through yet. For these, just use
/// the workgroup count calculation based on the root of the DAG.
static bool checkShapeIsDataDependant(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::GenericOp>(op)) {
    /// Currently checked case is
    /// - linalg.generic op
    /// - all parallel iterators
    /// - output shape depends on tensor.extract;
    if (!llvm::all_of(linalgOp.getIteratorTypesArray(),
                      linalg::isParallelIterator)) {
      return false;
    }
    BackwardSliceOptions options;
    options.inclusive = true;
    options.filter = [](Operation *op) {
      // Only look for slices with a few ops to not blow up the slice
      // computation.
      if (!isa<arith::IndexCastOp, tensor::EmptyOp, tensor::ExtractOp>(op)) {
        return false;
      }
      // The slice computation method has an assert that the parent op
      // needs to have a single region with a single block. This seems
      // unnecessary for IREEs use case. For now avoid this assert by bailing if
      // any operands are block arguments.
      if (llvm::any_of(op->getOperands(), llvm::IsaPred<BlockArgument>)) {
        auto parentOp = op->getParentOp();
        if (parentOp->getNumRegions() != 1 ||
            parentOp->getRegion(0).getBlocks().size() != 1) {
          return false;
        }
      }
      return true;
    };
    llvm::SetVector<Operation *> slice;
    for (Value initOperand : linalgOp.getDpsInits()) {
      mlir::getBackwardSlice(initOperand, &slice, options);
    }
    return llvm::any_of(slice, llvm::IsaPred<tensor::ExtractOp>);
  }
  return false;
}

/// Populate the workgroup count calculation to be based on root of the DAG op.
/// These is the legacy path for workgroup count calculation that
/// arent handled by the current default.
static void createWorkgroupCountFromDagRootRegion(
    RewriterBase &rewriter, IREE::Flow::DispatchRegionOp &regionOp,
    TypeRange workloadTypes, ArrayRef<Location> workloadLocs) {
  Region &countRegion = regionOp.getWorkgroupCount();
  if (!countRegion.empty())
    return;
  Block *body = rewriter.createBlock(&countRegion, countRegion.begin(),
                                     workloadTypes, workloadLocs);
  auto args = body->getArguments();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(body);
  Location loc = regionOp.getLoc();
  auto countOp =
      rewriter.create<IREE::Flow::DispatchWorkgroupCountFromDagRootOp>(loc,
                                                                       args);
  rewriter.create<IREE::Flow::ReturnOp>(loc, countOp->getResults());
}

/// Return `true` if the given type is a ShapedType and has at least one
/// dynamic dimension.
static bool hasDynamicShape(Type t) {
  auto shapedType = llvm::dyn_cast<ShapedType>(t);
  if (!shapedType)
    return false;
  return !shapedType.hasStaticShape();
}

/// Reify the dynamic dimensions of the given value.
static LogicalResult
reifyDynamicResultDimsImpl(OpBuilder &b, Value value,
                           SmallVectorImpl<Value> &dynamicDims,
                           bool createTensorDimOps) {
  OpBuilder::InsertionGuard guard(b);

  // Case 1: No dynamic result dims.
  if (!hasDynamicShape(value.getType()))
    return success();

  // There is at least one dynamic dimension, continue...
  ShapedType shapedType = llvm::cast<ShapedType>(value.getType());

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
  if (auto bbArg = llvm::dyn_cast<BlockArgument>(value)) {
    if (!createTensorDimOps)
      return failure();

    b.setInsertionPointToStart(bbArg.getOwner());
    emitTensorDimOps();
    return success();
  }

  // Value is an OpResult.
  Operation *op = value.getDefiningOp();
  OpResult opResult = llvm::cast<OpResult>(value);

  // Case 3: Query ShapeAwareOpInterface.
  auto shapeAwareOp = dyn_cast<IREE::Util::ShapeAwareOpInterface>(op);
  if (shapeAwareOp) {
    ValueRange dims =
        shapeAwareOp.getResultDynamicDims(opResult.getResultNumber());
    dynamicDims.append(dims.begin(), dims.end());
    return success();
  }

  // Case 4: Value is tied. Reify the dimensions of the tied operand.
  auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
  if (tiedOp) {
    Value tiedOperand = tiedOp.getTiedResultOperand(value);
    if (tiedOperand && tiedOperand.getType() == value.getType())
      return reifyDynamicResultDimsImpl(b, tiedOperand, dynamicDims,
                                        /*createTensorDimOps=*/true);
  }

  // Case 5: Query ReifyRankedShapedTypeOpInterface.
  auto reifyShapeOp = dyn_cast<ReifyRankedShapedTypeOpInterface>(op);
  if (reifyShapeOp) {
    ReifiedRankedShapedTypeDims dims;
    if (failed(reifyShapeOp.reifyResultShapes(b, dims)))
      return failure();
    for (int64_t i = 0; i < shapedType.getRank(); ++i)
      if (shapedType.isDynamicDim(i))
        dynamicDims.push_back(cast<Value>(dims[opResult.getResultNumber()][i]));
    return success();
  }

  if (!createTensorDimOps)
    return failure();

  // None of the above. Insert tensor.dim ops.
  b.setInsertionPointAfter(op);
  emitTensorDimOps();
  return success();
}

/// Reify the dynamic dimensions of the given value.
/// Deprecated. Use `getOptimizedDynamicResultDims` instead.
LogicalResult reifyDynamicResultDims(OpBuilder &b, Value value,
                                     SmallVectorImpl<Value> &dynamicDims) {

  OpBuilder::InsertionGuard g(b);
  if (auto op = value.getDefiningOp()) {
    b.setInsertionPoint(op);
  }
  return reifyDynamicResultDimsImpl(b, value, dynamicDims,
                                    /*createTensorDimOps=*/true);
}

LogicalResult
getOptimizedDynamicResultDims(OpBuilder &b, Value value,
                              SmallVectorImpl<Value> &dynamicDims) {
  return reifyDynamicResultDimsImpl(b, value, dynamicDims,
                                    /*createTensorDimOps=*/false);
}

// Append a result to the given DispatchRegionOp. The newly created
// DispatchRegionOp is returned.
FailureOr<IREE::Flow::DispatchRegionOp> appendDispatchRegionResults(
    RewriterBase &rewriter, IREE::Flow::DispatchRegionOp regionOp,
    ArrayRef<Value> results, ArrayRef<SmallVector<Value>> dynamicDims) {
  if (results.empty()) {
    return regionOp;
  }
  assert(results.size() == dynamicDims.size() &&
         "expected as many dynamic dims list as the number of results");

  // Collect the current region dynamic dims, and result types.
  SmallVector<Value> regionDynamicDims(regionOp.getResultDims().begin(),
                                       regionOp.getResultDims().end());
  SmallVector<Type> resultTypes(regionOp.getResultTypes().begin(),
                                regionOp.getResultTypes().end());

  // Collect the current dispatch yielded values.
  auto returnOp =
      cast<IREE::Flow::ReturnOp>(regionOp.getBody().front().getTerminator());
  SmallVector<Value> returnedValues(returnOp.getOperands().begin(),
                                    returnOp.getOperands().end());

  for (auto [index, result] : llvm::enumerate(results)) {
#ifndef NDEBUG
    auto tensorType = cast<RankedTensorType>(result.getType());
    assert(tensorType.getNumDynamicDims() == dynamicDims[index].size() &&
           "incorrect number of dynamicDims provided");
#endif // NDEBUG
    resultTypes.push_back(result.getType());
    regionDynamicDims.append(dynamicDims[index]);
    returnedValues.push_back(result);
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(regionOp);

  // Create new DispatchRegionOp and move over the body.
  auto newRegionOp = rewriter.create<IREE::Flow::DispatchRegionOp>(
      regionOp->getLoc(), resultTypes, regionDynamicDims,
      regionOp.getWorkload());
  rewriter.inlineRegionBefore(regionOp.getBody(), newRegionOp.getBody(),
                              newRegionOp.getBody().begin());
  rewriter.replaceOp(
      regionOp, newRegionOp.getResults().take_front(regionOp->getNumResults()));

  // Update terminator.
  auto newRegionReturnOp =
      cast<IREE::Flow::ReturnOp>(newRegionOp.getBody().front().getTerminator());
  rewriter.setInsertionPoint(newRegionReturnOp);
  rewriter.replaceOpWithNewOp<IREE::Flow::ReturnOp>(newRegionReturnOp,
                                                    returnedValues);

  return newRegionOp;
}

IREE::Flow::DispatchRegionOp
makeEmptyDispatchRegion(OpBuilder &builder, Location loc, ValueRange workload) {
  OpBuilder::InsertionGuard guard(builder);

  // Create RegionOp.
  auto regionOp = builder.create<IREE::Flow::DispatchRegionOp>(
      loc, /*resultTypes=*/TypeRange(), /*dynamicDims=*/ValueRange(), workload);
  Block &body = regionOp.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<IREE::Flow::ReturnOp>(loc, ValueRange());
  return regionOp;
}

// Clone a `target` op that is preceding the given dispatch region op into the
// dispatch region.
FailureOr<Operation *>
clonePrecedingOpIntoDispatchRegion(RewriterBase &rewriter, Operation *target,
                                   IREE::Flow::DispatchRegionOp regionOp) {
  Block &body = regionOp.getBody().front();

  // Gather all uses of `target`.
  SmallVector<OpOperand *> usesInsideOfRegion;
  for (OpOperand &use : target->getUses()) {
    Operation *parentOperation = use.getOwner();
    Region *parentRegion = parentOperation->getParentRegion();

    while ((parentOperation = parentOperation->getParentOp())) {
      if (regionOp.getOperation() == parentOperation)
        break;
      parentRegion = parentOperation->getParentRegion();
    }

    if (parentOperation && &parentRegion->front() == &body) {
      usesInsideOfRegion.push_back(&use);
    }
  }

  // Clone op into dispatch region.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&body);
  Operation *newTargetOp = rewriter.clone(*target);

  // Replace all uses in the dispatch region.
  for (OpOperand *use : usesInsideOfRegion) {
    rewriter.modifyOpInPlace(use->getOwner(), [&]() {
      use->set(newTargetOp->getResult(
          llvm::cast<OpResult>(use->get()).getResultNumber()));
    });
  }

  return newTargetOp;
}

// Move a `target` op that is preceding the given dispatch region op into the
// dispatch region.
FailureOr<IREE::Flow::DispatchRegionOp>
movePrecedingOpsIntoDispatchRegion(RewriterBase &rewriter,
                                   ArrayRef<Operation *> targets,
                                   IREE::Flow::DispatchRegionOp regionOp) {
  // Values replaced by moving the `targets` into the dispatch region.
  SmallVector<Value> replacedValues;

  // List of dynamic dimensions for each new results added to the dispatch
  // region.
  SmallVector<SmallVector<Value>> dispatchOpNewResultsDynamicDims;

  // New values that are yielded from dispatch.
  SmallVector<Value> yieldedResults;

  llvm::SetVector<Operation *> targetSet;
  targetSet.insert(targets.begin(), targets.end());

  Block &body = regionOp.getBody().front();
  for (Operation *target : llvm::reverse(targets)) {
    // Clone op into dispatch region.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&body);
    Operation *clonedTarget = rewriter.clone(*target);

    // Gather all uses of `target`.
    for (auto [index, result] : llvm::enumerate(target->getResults())) {
      bool hasUsesOutsideOfRegion =
          llvm::any_of(result.getUses(), [&](OpOperand &use) {
            Operation *user = use.getOwner();
            // The use is not in
            // 1. the current dispatch
            // 2. Not in one of the targets.
            return !regionOp->isProperAncestor(user) && !targetSet.count(user);
          });
      if (hasUsesOutsideOfRegion) {
        replacedValues.push_back(result);
        yieldedResults.push_back(clonedTarget->getResult(index));
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(target);
        SmallVector<Value> &dims =
            dispatchOpNewResultsDynamicDims.emplace_back();
        if (failed(getOptimizedDynamicResultDims(rewriter, result, dims))) {
          return target->emitOpError(
              "failed to reify dynamic dims of result to be yielded from "
              "dispatch region");
        }
      }
    }

    rewriter.replaceOpUsesWithinBlock(target, clonedTarget->getResults(),
                                      &body);
  }

  FailureOr<IREE::Flow::DispatchRegionOp> newRegionOp =
      appendDispatchRegionResults(rewriter, regionOp, yieldedResults,
                                  dispatchOpNewResultsDynamicDims);

  if (failed(newRegionOp)) {
    return regionOp->emitOpError("failed to append results to op");
  }

  ValueRange replacements =
      newRegionOp->getResults().take_back(replacedValues.size());
  for (auto [index, replacedVal] : llvm::enumerate(replacedValues)) {
    rewriter.replaceAllUsesWith(replacedVal, replacements[index]);
  }
  for (auto target : llvm::reverse(targets)) {
    rewriter.eraseOp(target);
  }
  return newRegionOp.value();
}

// Move a `target` op that is following the given dispatch region op into the
// dispatch region.
FailureOr<IREE::Flow::DispatchRegionOp>
moveFollowingOpIntoDispatchRegion(RewriterBase &rewriter, Operation *target,
                                  IREE::Flow::DispatchRegionOp regionOp) {
  // Fail if any of the `target` operands do not dominate the dispatch region.
  mlir::DominanceInfo dominanceInfo(regionOp);
  for (Value operand : target->getOperands()) {
    Operation *definingOp = operand.getDefiningOp();
    if (definingOp && !dominanceInfo.dominates(definingOp, regionOp)) {
      return rewriter.notifyMatchFailure(
          target, "target operands do not dominate the dispatch region op.");
    }
  }

  // Values replaced by moving the `target` into the dispatch region.
  SmallVector<Value> replacedValues;

  // List of dynamic dimensions for each new results added to the dispatch
  // region.
  SmallVector<SmallVector<Value>> dispatchOpNewResultsDynamicDims;

  // New values that are yielded from dispatch.
  SmallVector<Value> yieldedResults;

  Block &body = regionOp.getBody().front();
  // Clone op into dispatch region.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(body.getTerminator());
  Operation *clonedTarget = rewriter.clone(*target);

  // Replace any operands returned by the `regionOp` with the results yielded
  // inside of the `regionOp`.
  for (OpOperand &operand : clonedTarget->getOpOperands()) {
    if (operand.get().getDefiningOp() != regionOp) {
      continue;
    }
    auto returnOp =
        cast<IREE::Flow::ReturnOp>(regionOp.getBody().front().getTerminator());
    auto opResult = cast<OpResult>(operand.get());
    Value yieldedValue = returnOp->getOperand(opResult.getResultNumber());
    rewriter.modifyOpInPlace(clonedTarget, [&]() {
      clonedTarget->setOperand(operand.getOperandNumber(), yieldedValue);
    });
  }

  // Gather all uses of `target`.
  for (auto [index, result] : llvm::enumerate(target->getResults())) {
    replacedValues.push_back(result);
    yieldedResults.push_back(clonedTarget->getResult(index));
    OpBuilder::InsertionGuard g1(rewriter);
    rewriter.setInsertionPoint(regionOp);
    SmallVector<Value> &dims = dispatchOpNewResultsDynamicDims.emplace_back();
    if (failed(getOptimizedDynamicResultDims(rewriter, result, dims))) {
      return target->emitOpError(
          "failed to reify dynamic dims of result to be yielded from "
          "dispatch region");
    }
  }

  FailureOr<IREE::Flow::DispatchRegionOp> newRegionOp =
      appendDispatchRegionResults(rewriter, regionOp, yieldedResults,
                                  dispatchOpNewResultsDynamicDims);

  if (failed(newRegionOp)) {
    return regionOp->emitOpError("failed to append results to op");
  }

  ValueRange replacements =
      newRegionOp->getResults().take_back(replacedValues.size());
  for (auto [index, replacedVal] : llvm::enumerate(replacedValues)) {
    rewriter.replaceAllUsesWith(replacedVal, replacements[index]);
  }
  rewriter.eraseOp(target);
  return newRegionOp.value();
}

FailureOr<IREE::Flow::DispatchRegionOp>
wrapOpInDispatchRegion(RewriterBase &rewriter, Operation *op) {
  OpBuilder::InsertionGuard g(rewriter);

  SmallVector<Value> workload;
  bool useWorkloadCountFromDagRootMode = checkShapeIsDataDependant(op);
  if (useWorkloadCountFromDagRootMode) {
    rewriter.setInsertionPoint(op);
    workload = getWorkloadForRootOp(rewriter, op);
  }

  rewriter.setInsertionPointAfter(op);
  IREE::Flow::DispatchRegionOp regionOp =
      IREE::Flow::makeEmptyDispatchRegion(rewriter, op->getLoc(), workload);

  // Move the op into the dispatch region.
  auto newRegionOp = movePrecedingOpsIntoDispatchRegion(rewriter, op, regionOp);

  if (succeeded(newRegionOp) && useWorkloadCountFromDagRootMode) {
    auto newWorkload = newRegionOp->getWorkload();
    auto workloadTypes =
        llvm::map_to_vector(newWorkload, [](Value v) { return v.getType(); });
    auto workloadLocs =
        llvm::map_to_vector(newWorkload, [](Value v) { return v.getLoc(); });
    createWorkgroupCountFromDagRootRegion(rewriter, *newRegionOp, workloadTypes,
                                          workloadLocs);
  }

  return newRegionOp;
}

FailureOr<Operation *> hoistOutOfDispatch(RewriterBase &rewriter,
                                          Operation *op) {
  assert(op && !isNonNullAndOutsideDispatch(op) &&
         "op expected to be in a dispatch");

  // Step 1: Clone the op outside of the dispatch region.

  OpBuilder::InsertionGuard g(rewriter);
  auto dispatchRegionOp = op->getParentOfType<DispatchRegionOp>();

  // If all operands of the `op` come from outside the dispatch, then the op can
  // be hoisted out before the dispatch region. Otherwise, the op can be hoisted
  // out below the dispatch if the only users of the op are the dispatch return.
  if (llvm::none_of(op->getOperands(), [&](Value operand) {
        Operation *producer = operand.getDefiningOp();
        return producer && producer->getParentOfType<DispatchRegionOp>();
      })) {
    rewriter.setInsertionPoint(dispatchRegionOp);
  } else if (llvm::all_of(op->getUsers(), [&](Operation *user) {
               return isa<IREE::Flow::ReturnOp>(user);
             })) {
    rewriter.setInsertionPointAfter(dispatchRegionOp);
  } else {
    return rewriter.notifyMatchFailure(
        op, "op has both operands and users insided of its dispatch");
  }
  Operation *hoistedOp = rewriter.clone(*op);

  // Step 2: Replace op uses inside and outside of the dispatch region with the
  //         hoisted results.

  auto getMatchingDispatchResult =
      [&](Value result) -> std::optional<OpResult> {
    for (OpOperand &use : result.getUses()) {
      if (isa<IREE::Flow::ReturnOp>(use.getOwner())) {
        return dispatchRegionOp.getResults()[use.getOperandNumber()];
      }
    }
    return std::nullopt;
  };
  bool yieldsResults = false;
  for (OpResult result : op->getResults()) {
    Value hoistedResult = hoistedOp->getResult(result.getResultNumber());
    // Replace all results yielded by the dispatch region with the hoisted
    // op results.
    std::optional<OpResult> dispResult = getMatchingDispatchResult(result);
    if (dispResult.has_value()) {
      yieldsResults = true;
      rewriter.replaceAllUsesWith(dispResult.value(), hoistedResult);
    }
    // Replace uses inside the dispatch region.
    rewriter.replaceAllUsesWith(result, hoistedResult);
  }
  // If no results were yielded from `op`, then nothing more to do.
  if (!yieldsResults) {
    return hoistedOp;
  }

  // Step 3: Collect the new set of dispatch results and dynamic dims, and
  //         create a new dispatch region to replace the old one. The new
  //         dispatch may have duplicated results,

  // Get the new dispatch region return values and dynamic dims, excluding the
  // ones coming from the `hoistedOp`.
  auto dispatchReturnOp = cast<IREE::Flow::ReturnOp>(
      dispatchRegionOp.getBody().front().getTerminator());
  SmallVector<Value, 2> newDispatchReturnOperands;
  SmallVector<Value, 4> newDispatchResultDynamicDims;
  // Keep track of which results in the original dispatch region correspond to
  // which results in the new dispatch region with `oldDispatchResultInds`.
  SmallVector<int64_t, 2> oldDispatchResultInds;
  for (OpOperand &operand : dispatchReturnOp->getOpOperands()) {
    if (operand.get().getDefiningOp() == hoistedOp) {
      continue;
    }
    oldDispatchResultInds.push_back(operand.getOperandNumber());
    newDispatchReturnOperands.push_back(operand.get());
    auto dims =
        dispatchRegionOp.getResultDynamicDims(operand.getOperandNumber());
    newDispatchResultDynamicDims.append(dims.begin(), dims.end());
  }

  // Add the operands of the `op` to the new return values of the dispatch, and
  // add their result dynamic dims to the new result dynamic dims.
  // Save the result index in the new dispatch corresponding to each hoisted op
  // operand in `resultIndsForHoistedOperands`, so uses can be replaced later.
  SmallVector<int64_t> resultIndsForHoistedOperands;
  for (OpOperand &operand : op->getOpOperands()) {
    // Only need to yield operands defined in the dispatch region.
    if (operand.get().getParentRegion() != &dispatchRegionOp.getBody()) {
      continue;
    }

    // If the operand is already yielded by the dispatch, don't yield it again,
    // and save the result index.
    bool resultAlreadyYielded = false;
    for (auto [idx, returnOperand] :
         llvm::enumerate(newDispatchReturnOperands)) {
      if (returnOperand == operand.get()) {
        resultAlreadyYielded = true;
        resultIndsForHoistedOperands.push_back(idx);
        break;
      }
    }
    if (resultAlreadyYielded) {
      break;
    }
    resultIndsForHoistedOperands.push_back(newDispatchReturnOperands.size());
    newDispatchReturnOperands.push_back(operand.get());

    // Save operand and dynamic dims to add to the dispatch region.
    SmallVector<Value> dims;
    if (failed(reifyDynamicResultDims(rewriter, operand.get(), dims))) {
      return op->emitOpError(
          "failed to reify dynamic dims of result to be yielded from "
          "dispatch region");
    }
    newDispatchResultDynamicDims.append(dims.begin(), dims.end());
  }

  // Create the new dispatch region op. `newDispatchReturnOperands` now has all
  // the original return operands, excluding the hoisted op's results, and
  // including any new results coming from the hoisted op's old operands. The
  // `newDispatchResultDynamicDims` contains the corresponding result dynamic
  // dims for `newDispatchReturnOperands`.
  SmallVector<Type> newResultTypes =
      llvm::map_to_vector(newDispatchReturnOperands,
                          [](Value operand) { return operand.getType(); });
  rewriter.setInsertionPoint(dispatchRegionOp);
  auto newDispatchRegionOp = rewriter.create<IREE::Flow::DispatchRegionOp>(
      dispatchRegionOp->getLoc(), newResultTypes, newDispatchResultDynamicDims,
      dispatchRegionOp.getWorkload());
  rewriter.inlineRegionBefore(dispatchRegionOp.getBody(),
                              newDispatchRegionOp.getBody(),
                              newDispatchRegionOp.getBody().begin());
  // Move the workgroup count region over.
  if (!dispatchRegionOp.getWorkgroupCount().empty()) {
    Region &newWorkgroupCountRegion = newDispatchRegionOp.getWorkgroupCount();
    rewriter.inlineRegionBefore(dispatchRegionOp.getWorkgroupCount(),
                                newWorkgroupCountRegion,
                                newWorkgroupCountRegion.begin());
  }
  // Need to make a new flow.return op, since the body was copied from the
  // old dispatch region.
  auto newDispatchReturnOp = cast<IREE::Flow::ReturnOp>(
      newDispatchRegionOp.getBody().front().getTerminator());
  rewriter.setInsertionPoint(newDispatchReturnOp);
  rewriter.replaceOpWithNewOp<IREE::Flow::ReturnOp>(newDispatchReturnOp,
                                                    newDispatchReturnOperands);

  // Replace operands of the `hoistedOp` with dispatch region results. They are
  // currently using values from inside the dispatch region.
  for (auto [idx, operand] : llvm::enumerate(hoistedOp->getOperands())) {
    auto newResultIdx = resultIndsForHoistedOperands[idx];
    Value newDispatchResult = newDispatchRegionOp->getResults()[newResultIdx];
    rewriter.replaceUsesWithIf(operand, newDispatchResult,
                               [&](OpOperand &opOperand) {
                                 return opOperand.getOwner() == hoistedOp;
                               });
  }

  // Step 4: Fixup all uses. Still need to replace the operands of the hoisted
  //         op, and replace the remaining uses of the old dispatch region with
  //         the new dispatch region results.

  // Replace the uses of the original dispatch region results with the final
  // dispatch region results.
  for (auto [oldIdx, newIdx] : llvm::enumerate(oldDispatchResultInds)) {
    Value newDispatchResult = newDispatchRegionOp->getResults()[newIdx];
    Value dispatchResult = dispatchRegionOp->getResults()[oldIdx];
    rewriter.replaceAllUsesWith(dispatchResult, newDispatchResult);
  }

  return hoistedOp;
}

//===---------------------------------------------------------------------===//
// Utilities to make a dispatch region isolated from above
//===---------------------------------------------------------------------===//

static bool isAttentionMaskGenerator(Operation *op) {
  for (OpOperand &use : op->getUses()) {
    if (auto attention =
            dyn_cast<IREE::LinalgExt::AttentionOp>(use.getOwner())) {
      if (attention.getMask() == use.get()) {
        return true;
      }
    }
  }
  return false;
}

/// Operations that are cloned into dispatch regions formed with other
/// operations as roots.
bool isClonableIntoDispatchOp(Operation *op,
                              ClonableIntoDispatchOptions options) {
  // TODO(#8637): `tensor.collapse_shape` and `tensor.expand_shape` are
  // trivially clonable too, but they cause problems
  // with bufferization. Make them clonable when fixed.
  if (isa<affine::AffineApplyOp, arith::IndexCastOp, linalg::FillOp,
          tensor::EmptyOp, tensor::ExtractOp, tensor::ExtractSliceOp,
          complex::CreateOp>(op)) {
    return true;
  }
  if (LinalgExt::isBitExtendOp(op)) {
    return true;
  }
  if (clEnableGatherFusion && LinalgExt::isGatherlikeOp(op)) {
    return true;
  }
  // If the operation is used for masking an AttentionOp, then we always
  // clone it. The Attention mask is usually big, and is always generated
  // from a small tensor, so it's always good to clone it.
  if (options.aggressive && isAttentionMaskGenerator(op)) {
    return true;
  }
  if (isa<arith::ConstantOp>(op) || isa<complex::ConstantOp>(op)) {
    if (clInlineConstantByteLength == 0)
      return false;
    Attribute constantValueAttr;
    if (!matchPattern(op->getResult(0), m_Constant(&constantValueAttr))) {
      return false;
    }

    auto constantType = op->getResult(0).getType();
    if (llvm::isa<SplatElementsAttr>(constantValueAttr)) {
      return true;
    } else if (auto attr = llvm::dyn_cast<ElementsAttr>(constantValueAttr)) {
      auto shapedType = llvm::cast<ShapedType>(constantType);
      uint64_t estimatedByteLength =
          (shapedType.getNumElements() *
           IREE::Util::getTypeBitWidth(shapedType.getElementType())) /
          8;
      return attr.isSplat() ||
             estimatedByteLength <= clInlineConstantByteLength;
    } else if (constantType.isIntOrIndexOrFloat() ||
               isa<ComplexType>(constantType)) {
      return true;
    }
  }
  if (op->getDialect() ==
          op->getContext()->getLoadedDialect<arith::ArithDialect>() &&
      llvm::all_of(op->getOperands(),
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

    // Do not fuse operations if they are already an operand of the
    // owner and have an index return type as that means its a shape
    // computation that needs to happen on the host.
    if (user == dispatchOp && v.getType().isIndex() &&
        isa<IREE::Flow::DispatchRegionOp>(dispatchOp)) {
      return true;
    }

    Operation *ownerWorkgroupsOp =
        user->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>();
    Operation *ownerRegionOp =
        user->getParentOfType<IREE::Flow::DispatchRegionOp>();
    Operation *owner = ownerWorkgroupsOp ? ownerWorkgroupsOp : ownerRegionOp;

    // Ignore uses outside of dispatch workgroups op.
    if (owner != dispatchOp)
      continue;

    // Cannot fuse producer of `dest` with `tensor.insert_slice`.
    if (auto insertSliceUser = dyn_cast<tensor::InsertSliceOp>(user)) {
      if (insertSliceUser.getDest() == v)
        return true;
    }
  }
  return false;
}

SmallVector<Operation *> getCloneableOps(IREE::Flow::DispatchRegionOp regionOp,
                                         ClonableIntoDispatchOptions options) {
  // Find values that are used inside of the dispatch region but defined outside
  // of the dispatch region.
  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(regionOp.getBody(), valuesDefinedAbove);
  if (valuesDefinedAbove.empty())
    return {};

  // Traverse the defining ops of these values (and the ops on their reverse
  // SSA use-def chain).
  SmallVector<Operation *> result;
  llvm::SetVector<Value> visited;
  SmallVector<Value> worklist;
  worklist.assign(valuesDefinedAbove.begin(), valuesDefinedAbove.end());
  while (!worklist.empty()) {
    Value outsideValue = worklist.pop_back_val();
    // Skip values that were already visited.
    if (visited.count(outsideValue))
      continue;
    visited.insert(outsideValue);

    Operation *definingOp = outsideValue.getDefiningOp();
    if (!definingOp ||
        !IREE::Flow::isClonableIntoDispatchOp(definingOp, options) ||
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
LogicalResult cloneProducersToRegion(RewriterBase &rewriter,
                                     IREE::Flow::DispatchRegionOp regionOp,
                                     ClonableIntoDispatchOptions options) {
  SmallVector<Operation *> cloneableOps;
  do {
    cloneableOps = getCloneableOps(regionOp, options);
    bool sortResult = mlir::computeTopologicalSorting(cloneableOps);
    (void)sortResult;
    assert(sortResult && "could not compute topological sorting");

    for (Operation *producer : llvm::reverse(cloneableOps)) {
      if (failed(clonePrecedingOpIntoDispatchRegion(rewriter, producer,
                                                    regionOp))) {
        return failure();
      }
    }
  } while (!cloneableOps.empty());

  return success();
}

} // namespace mlir::iree_compiler::IREE::Flow
