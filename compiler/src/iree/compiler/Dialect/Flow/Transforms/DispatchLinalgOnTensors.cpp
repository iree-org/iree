// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/DispatchLinalgOnTensors.h"

#include <deque>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/DispatchRegionHeuristic.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

#define DEBUG_TYPE "iree-flow-dispatch-linalg-on-tensors"

static const char kRootOpAttr[] = "__root_op__";
static const char kFusionGroupsAttr[] = "__fused_op__";

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Root and fusion group attribute handling
//===----------------------------------------------------------------------===//

/// Returns true if an op has a root operation.
static bool hasRootOpAttribute(Operation *op) {
  return static_cast<bool>(op->getAttrOfType<IntegerAttr>(kRootOpAttr));
}
/// Removes root attribute. Asserts if root attribute is not present.
static void removeRootOpAttribute(Operation *op) {
  op->removeAttr(kRootOpAttr);
}
/// Sets the root attribute for an operation. The root attribute needs a number
/// to identify the root. Asserts if root attribute is already set on an
/// operation.
static void setRootAttribute(MLIRContext *context, Operation *op,
                             int64_t rootNumber) {
  assert(!op->hasAttr(kRootOpAttr) &&
         "invalid to update root attribute on an op");
  op->setAttr(kRootOpAttr,
              IntegerAttr::get(IntegerType::get(context, 64), rootNumber));
}
/// Returns the number of the root. Asserts if the operation is not already set
/// as a root.
static int64_t getRootNumber(Operation *op) {
  return op->getAttrOfType<IntegerAttr>(kRootOpAttr).getInt();
}
/// Returns true if an op is part of a fusion group.
static bool hasFusionGroupsAttribute(Operation *op) {
  return static_cast<bool>(op->getAttrOfType<ArrayAttr>(kFusionGroupsAttr));
}
/// Returns the fusion groups for the given `op`.
static SmallVector<int64_t, 1> getFusionGroups(Operation *op) {
  SmallVector<int64_t, 1> fusionGroups = {};
  if (auto fusionGroupsAttr = op->getAttrOfType<ArrayAttr>(kFusionGroupsAttr)) {
    fusionGroups = llvm::to_vector<1>(llvm::map_range(
        fusionGroupsAttr,
        [](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
  }
  return fusionGroups;
}
/// Appends the given `op` to the `newGroups` fusion groups.
static void appendToFusionGroup(Operation *op, ArrayRef<int64_t> newGroups) {
  SmallVector<int64_t, 1> fusionGroups = getFusionGroups(op);
  fusionGroups.append(newGroups.begin(), newGroups.end());
  op->setAttr(kFusionGroupsAttr, Builder(op).getI64ArrayAttr(fusionGroups));
}
/// Returns true if the given `op` is in the `targetGroup` fusion group.
static bool isInFusionGroup(Operation *op, unsigned targetGroup) {
  if (ArrayAttr opGroupAttr = op->getAttrOfType<ArrayAttr>(kFusionGroupsAttr)) {
    return llvm::any_of(opGroupAttr, [&targetGroup](Attribute attr) {
      return attr.cast<IntegerAttr>().getInt() == targetGroup;
    });
  }
  return false;
}
/// Removes the fusion groups attribute.
static void removeFusionGroupsAttribute(Operation *op) {
  op->removeAttr(kFusionGroupsAttr);
}

//===----------------------------------------------------------------------===//
// Methods for getting the workload information for dispatch region creation.
//===----------------------------------------------------------------------===//

/// Compute the workload to use for the workgroup based on the root op.
static SmallVector<Value> getWorkloadForRootOp(OpBuilder &builder,
                                               Operation *rootOp) {
  // Compute workgroup count to use for the dispatch op. These are the ranges
  // of the outermost parallel loops that can be distributed.
  Location loc = rootOp->getLoc();
  SmallVector<Range> loopRanges = getLoopRanges(rootOp, loc, builder);
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

//===----------------------------------------------------------------------===//
// Methods that help creating the dispatch regions
//===----------------------------------------------------------------------===//

/// For an operation to be moved into the dispatch region, append `resultTypes`
/// with the type of the results dispatch region has to return. Also
/// append `resultDynamicDims` with values that represent the dynamic shapes of
/// result values returned.
static LogicalResult computeDispatchResultTypeAndDynamicDims(
    PatternRewriter &rewriter, Operation *dispatchOp,
    SmallVector<Type> &resultTypes, SmallVector<Value> &resultDynamicDims) {
  auto currResultTypes = dispatchOp->getResultTypes();
  resultTypes.append(currResultTypes.begin(), currResultTypes.end());
  auto rankedShapedTypeOp =
      dyn_cast<ReifyRankedShapedTypeOpInterface>(dispatchOp);
  if (!rankedShapedTypeOp) {
    return rewriter.notifyMatchFailure(
        dispatchOp,
        "expected op to implement the ReifyRankedShapedTypeOpInterface");
  }

  // Get the values for the result dims.
  ReifiedRankedShapedTypeDims resultDims;
  if (failed(rankedShapedTypeOp.reifyResultShapes(rewriter, resultDims))) {
    return rewriter.notifyMatchFailure(dispatchOp,
                                       "failed to reify shape of the result");
  }
  if (currResultTypes.size() != resultDims.size()) {
    return rewriter.notifyMatchFailure(
        dispatchOp, "expected as many result shapes as number of outputs");
  }
  for (auto outputType : llvm::enumerate(currResultTypes)) {
    auto shapedOutputType = outputType.value().dyn_cast<ShapedType>();
    if (!shapedOutputType) continue;
    for (auto dim : llvm::enumerate(shapedOutputType.getShape())) {
      if (ShapedType::isDynamic(dim.value())) {
        resultDynamicDims.push_back(
            resultDims[outputType.index()][dim.index()]);
      }
    }
  }
  return success();
}

/// Returns true if the operation has only uses in `tensor.dim` ops.
static bool hasComputeUsesOutsideDispatch(
    Operation *op, ArrayRef<Operation *> dispatchOps = {}) {
  return !llvm::all_of(op->getUsers(), [&](Operation *user) {
    return isa<tensor::DimOp>(user) || llvm::is_contained(dispatchOps, user);
  });
}

/// Creates a flow.dispatch.workgroup op without arguments.
/// All the necessary operands are transiently captured and rewritten late as
/// operands. This greatly simplifies transformations into the resulting op.
static FailureOr<SmallVector<Operation *>>
buildOperandLessFlowDispatchWorkgroupOp(PatternRewriter &rewriter, Location loc,
                                        ArrayRef<Value> workload,
                                        ArrayRef<Operation *> dispatchOps) {
  SmallVector<Value> resultDynamicDims;
  SmallVector<Type> resultTypes;

  // 1. Compute the result types for the dispatch and the dynamic dimensions
  //    of the result of the dispatch. If operation has only dim uses
  //    do not make the dispatch op return those values. Those uses are
  //    kept on the original op, and later patterns are expected to take care
  //    of them.
  for (auto op : dispatchOps) {
    if (!hasComputeUsesOutsideDispatch(op, dispatchOps)) continue;
    if (failed(computeDispatchResultTypeAndDynamicDims(
            rewriter, op, resultTypes, resultDynamicDims))) {
      return failure();
    }
  }

  // 2. Create a dispatch op with just the `flow.return` terminator.
  auto dispatchOp = rewriter.create<IREE::Flow::DispatchWorkgroupsOp>(
      loc, workload, resultTypes, resultDynamicDims,
      /*operands=*/ArrayRef<Value>{}, /*operandDims=*/ArrayRef<Value>{},
      /*tiedOperands=*/ArrayRef<int64_t>{});
  Region &region = dispatchOp.getWorkgroupBody();
  Block *block = &region.front();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToEnd(block);
  auto returnOp = rewriter.create<IREE::Flow::ReturnOp>(loc);
  rewriter.setInsertionPoint(returnOp);

  // 3. Clone the necessary operations into the dispatch and replace
  //    all uses of the original op with the cloned op within the dispatch.
  auto resultArgs = region.getArguments();
  unsigned resultPos = 0;
  unsigned resultDynamicDimsPos = 0;
  SmallVector<Value> dispatchOpResults = dispatchOp.getResults();
  SmallVector<Operation *> clonedOps;
  clonedOps.reserve(dispatchOps.size());
  for (auto op : dispatchOps) {
    Operation *clonedOp = rewriter.clone(*op);
    clonedOps.push_back(clonedOp);
    rewriter.replaceOpWithinBlock(op, clonedOp->getResults(), block);
    rewriter.setInsertionPoint(clonedOp);
    if (!hasComputeUsesOutsideDispatch(op, dispatchOps)) continue;

    // 3a. Replace all non-dim uses of the original operation with the
    //     corresponding result of the dispatch.
    rewriter.replaceOpWithIf(op,
                             ArrayRef<Value>(dispatchOpResults)
                                 .slice(resultPos, op->getNumResults()),
                             [&](OpOperand &operand) {
                               return !isa<tensor::DimOp>(operand.getOwner());
                             });

    // 3b. For each of the result create a `flow.dispatch.tensor.store`
    //     operation to publish the result of the cloned operation (from within
    //     the dispatch).
    for (auto clonedOpResult : clonedOp->getResults()) {
      auto resultType = clonedOpResult.getType().dyn_cast<ShapedType>();
      if (resultType) {
        OpBuilder::InsertionGuard g2(rewriter);
        rewriter.setInsertionPoint(returnOp);
        unsigned numDynamicDims = resultType.getNumDynamicDims();
        rewriter.create<IREE::Flow::DispatchTensorStoreOp>(
            loc, clonedOpResult, resultArgs[resultPos],
            ArrayRef<Value>(resultDynamicDims)
                .slice(resultDynamicDimsPos, numDynamicDims));
        resultDynamicDimsPos += numDynamicDims;
      }
      resultPos++;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "After getWorkgroupBody creation \n"
                          << *dispatchOp << "\n");

  // 4. Add a region for workgroup_count computation.
  Region &workgroupCountRegion = dispatchOp.getWorkgroupCount();
  Block *body = rewriter.createBlock(&workgroupCountRegion);
  // Assuming that there is an insertion guard in place already, change the
  // insertion point to the body.
  rewriter.setInsertionPointToStart(body);
  SmallVector<Value> workloadArgs;
  for (auto workload : llvm::enumerate(workload)) {
    workloadArgs.push_back(body->addArgument(workload.value().getType(), loc));
  }
  auto numWorkgroupsOp =
      rewriter.create<DispatchWorkgroupCountFromDagRootOp>(loc, workloadArgs);
  rewriter.create<ReturnOp>(loc, numWorkgroupsOp.getResults());

  LLVM_DEBUG(llvm::dbgs() << "After workgroup_count creation \n"
                          << *dispatchOp << "\n");

  LLVM_DEBUG(llvm::dbgs() << "Created dispatchOp shell \n"
                          << *dispatchOp << "\n");
  return clonedOps;
}

/// Returns the list of operations that are to be cloned into the dispatch
/// based on the root operation.
static SmallVector<Operation *> getOperationsToMoveIntoDispatch(
    Operation *rootOp) {
  SmallVector<Operation *> dispatchOps;
  dispatchOps.push_back(rootOp);
  if (!hasRootOpAttribute(rootOp)) return dispatchOps;

  int64_t groupNum = getRootNumber(rootOp);
  std::deque<Operation *> worklist;
  worklist.push_back(rootOp);
  llvm::SmallDenseSet<Operation *, 2> visitedOps;
  visitedOps.insert(rootOp);

  while (!worklist.empty()) {
    Operation *currRoot = worklist.front();
    worklist.pop_front();
    for (auto operand : currRoot->getOperands()) {
      auto producer = operand.getDefiningOp();
      if (!producer || visitedOps.count(producer)) continue;
      visitedOps.insert(producer);
      if (!isInFusionGroup(producer, groupNum)) continue;
      worklist.push_back(producer);
      dispatchOps.push_back(producer);
    }
  }

  bool sortResult = mlir::computeTopologicalSorting(dispatchOps);
  (void)sortResult;
  assert(sortResult && "could not compute topological sorting");
  return llvm::to_vector(llvm::reverse(dispatchOps));
}

//===---------------------------------------------------------------------===//
// Methods to legalize a dispatch region op, i.e. make it isolated from above.
//===---------------------------------------------------------------------===//

/// Checks if the `Value` has a use within the dispatch that is unfusable.
static bool hasUnfusableUseInDispatch(
    Value v, IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    // Ignore uses outside of dispatch workgroups op.
    if (user->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>() != dispatchOp)
      continue;

    // Cannot fuse producer of `dest` with `tensor.insert_slice`.
    if (auto insertSliceUser = dyn_cast<tensor::InsertSliceOp>(user)) {
      if (insertSliceUser.getDest() == v) return true;
    }
  }
  return false;
}

/// Computes the values that will eventually be used within the dispatch
/// workgroup op but defined outside the op after all clonable operations are
/// cloned into the region.
static void getUsedValuesDefinedAboveAfterCloningOps(
    OpBuilder &builder, IREE::Flow::DispatchWorkgroupsOp dispatchOp,
    llvm::SetVector<Value> &valuesDefinedAbove) {
  llvm::SmallVector<Operation *> clonedOps;
  llvm::SetVector<Value> visited;
  SmallVector<Value, 4> worklist;
  worklist.assign(valuesDefinedAbove.begin(), valuesDefinedAbove.end());
  valuesDefinedAbove.clear();
  while (!worklist.empty()) {
    Value outsideValue = worklist.pop_back_val();
    if (visited.count(outsideValue)) continue;
    visited.insert(outsideValue);

    Operation *definingOp = outsideValue.getDefiningOp();
    if (!definingOp || !(isClonableIntoDispatchOp(definingOp)) ||
        hasUnfusableUseInDispatch(outsideValue, dispatchOp)) {
      valuesDefinedAbove.insert(outsideValue);
      continue;
    }
    clonedOps.push_back(definingOp);
    worklist.append(definingOp->operand_begin(), definingOp->operand_end());
  }
  // The cloned operations form a DAG. Return the cloned operations so the
  // leaves come first, and can be cloned in-order into the dispatch region.
  bool sortResult = mlir::computeTopologicalSorting(clonedOps);
  (void)sortResult;
  assert(sortResult && "could not compute topological sorting");

  for (auto clonedOp : reverse(clonedOps)) {
    Operation *clone = builder.clone(*clonedOp);
    for (auto result : llvm::enumerate(clonedOp->getResults())) {
      result.value().replaceUsesWithIf(
          clone->getResult(result.index()), [&](OpOperand &use) {
            return use.getOwner()
                       ->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>() ==
                   dispatchOp;
          });
      valuesDefinedAbove.remove(result.value());
    }
    builder.setInsertionPoint(clone);
  }

  // Reverse the values. This is not for correctness, but more for readability
  // of the IR.
  llvm::SetVector<Value> reversedValues;
  reversedValues.insert(valuesDefinedAbove.rbegin(), valuesDefinedAbove.rend());
  std::swap(reversedValues, valuesDefinedAbove);
}

//===---------------------------------------------------------------------===//
// Methods to tie operands and results of a dispatch op.
//===---------------------------------------------------------------------===//

/// Returns the tied operand for the given `resultArg`. Returns nullptr if error
/// or not found.
static BlockArgument getTiedOperandBlockArgument(BlockArgument resultArg) {
  auto resultArgType =
      resultArg.getType().dyn_cast<IREE::Flow::DispatchTensorType>();
  if (!resultArgType ||
      resultArgType.getAccess() != IREE::Flow::TensorAccess::WriteOnly) {
    return nullptr;
  }
  // Each output block argument should just have one use.
  if (!resultArg.hasOneUse()) return nullptr;

  // And that's a flow.dispatch.output.store op.
  auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(
      (*resultArg.getUses().begin()).getOwner());
  if (!storeOp) return nullptr;

  // Check if that block argument is tied to another block argument.
  auto tieOp = storeOp.getValue().getDefiningOp<Util::TiedOpInterface>();
  if (!tieOp) return nullptr;
  auto tiedArg =
      tieOp.getTiedResult(storeOp.getValue().cast<OpResult>().getResultNumber())
          .dyn_cast_or_null<BlockArgument>();
  if (!tiedArg) return nullptr;
  assert(isa<IREE::Flow::DispatchWorkgroupsOp>(
             tiedArg.getOwner()->getParentOp()) &&
         "expected that BbArg belongs to DispatchWorkgroupsOp");

  // Check that the type of the tied argument candidate and type of the output
  // match and that the tied argument is readonly.
  auto type = tiedArg.getType().dyn_cast<IREE::Flow::DispatchTensorType>();
  if (!type || type.getAccess() != IREE::Flow::TensorAccess::ReadOnly ||
      type.getBoundElementType() != resultArgType.getBoundElementType() ||
      llvm::any_of(llvm::zip(type.getShape(), resultArgType.getShape()),
                   [](std::tuple<int64_t, int64_t> sizes) {
                     return std::get<0>(sizes) != ShapedType::kDynamicSize &&
                            std::get<1>(sizes) != ShapedType::kDynamicSize &&
                            std::get<0>(sizes) != std::get<1>(sizes);
                   })) {
    return nullptr;
  }
  return tiedArg;
}

/// Modifies `dispatchOp` to attach operand-result tie information when
/// possible.
static void tryToTieOperandsAndResults(
    IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  Block *block = dispatchOp.getBody(0);

  // Go over each result to tie operand when possible, by:
  // 1. Update the tied operand argument to take readwrite tensors.
  // 2. Erase the result argument.
  // 3. Attach the tie information to the DispatchWorkgroupsOp.
  for (auto result : llvm::enumerate(dispatchOp.getResults())) {
    if (dispatchOp.getTiedResultOperand(result.value())) continue;
    BlockArgument outputArgument =
        dispatchOp.getOutputBlockArgument(result.index());
    BlockArgument tiedOperandArgument =
        getTiedOperandBlockArgument(outputArgument);
    if (!tiedOperandArgument) continue;
    auto oldType =
        tiedOperandArgument.getType().cast<IREE::Flow::DispatchTensorType>();
    tiedOperandArgument.setType(IREE::Flow::DispatchTensorType::get(
        IREE::Flow::TensorAccess::ReadWrite, oldType.getBoundType()));
    outputArgument.replaceAllUsesWith(tiedOperandArgument);
    block->eraseArgument(outputArgument.getArgNumber());
    dispatchOp.setTiedResultOperandIndex(result.index(),
                                         tiedOperandArgument.getArgNumber());
  }
}

// After outlining in dispatch region we can rewrite the dispatch ops with
// proper captures.
static LogicalResult legalizeDispatchWorkgroupOperands(
    IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  Location loc = dispatchOp.getLoc();
  Region &region = dispatchOp.getWorkgroupBody();
  Block &block = region.front();
  OpBuilder b = OpBuilder::atBlockBegin(&block);

  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  if (valuesDefinedAbove.empty()) return success();

  getUsedValuesDefinedAboveAfterCloningOps(b, dispatchOp, valuesDefinedAbove);
  b.setInsertionPointToStart(&block);

  // Build a map from current operands to arguments.
  std::pair<unsigned, unsigned> operandsIndexAndLength =
      dispatchOp.getODSOperandIndexAndLength(1);
  std::pair<unsigned, unsigned> operandDimsIndexAndLength =
      dispatchOp.getODSOperandIndexAndLength(2);
  llvm::DenseMap<Value, BlockArgument> operandToBBArg;
  for (auto operand : llvm::enumerate(dispatchOp.getArguments())) {
    operandToBBArg[operand.value()] = block.getArgument(operand.index());
  }

  // Of the values defined above and used in the region, add values that are not
  // operands to the region already.
  unsigned numOperands = operandsIndexAndLength.second;
  unsigned numOperandDims = operandDimsIndexAndLength.second;
  for (auto value : valuesDefinedAbove) {
    BlockArgument bbArg = operandToBBArg.lookup(value);
    bool wasPresent = bbArg != nullptr;
    auto tensorType = value.getType().dyn_cast<RankedTensorType>();
    if (!bbArg) {
      // Create a new basic block argument for this value.
      Type bbArgType = value.getType();
      if (tensorType) {
        bbArgType = IREE::Flow::DispatchTensorType::get(
            TensorAccess::ReadOnly, tensorType.getShape(),
            tensorType.getElementType());
      }
      bbArg = block.insertArgument(numOperands, bbArgType, value.getLoc());
    }

    // Insert the operand if this is not already one.
    if (!wasPresent) {
      unsigned insertIdx = operandsIndexAndLength.first + numOperands;
      dispatchOp->insertOperands(insertIdx, {value});
      operandToBBArg[dispatchOp->getOperand(insertIdx)] = bbArg;
      numOperands++;
    }

    Value repl = bbArg;
    if (!wasPresent && bbArg.getType().isa<IREE::Flow::DispatchTensorType>()) {
      // This dims for this operand does not exist. Add those.
      SmallVector<Value> dynamicDimArgs;
      {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPoint(dispatchOp);

        // Fast-path for if the value comes from ops that support our dynamic
        // shape interfaces. Otherwise we have to insert tensor.dim ops.
        auto availableDims = IREE::Util::findDynamicDims(value);

        // Add operands/args for each dynamic shape dimension.
        SmallVector<Value> dynamicDimOperands;
        unsigned dynamicDimIdx = 0;
        for (auto dim : llvm::enumerate(tensorType.getShape())) {
          if (dim.value() != ShapedType::kDynamicSize) continue;
          if (availableDims.has_value()) {
            dynamicDimOperands.push_back(availableDims.value()[dynamicDimIdx]);
          } else {
            dynamicDimOperands.push_back(b.createOrFold<tensor::DimOp>(
                dispatchOp.getLoc(), value, dim.index()));
          }
          dynamicDimArgs.push_back(
              block.insertArgument(numOperands + dynamicDimIdx,
                                   b.getIndexType(), dispatchOp.getLoc()));
          ++dynamicDimIdx;
        }
        dispatchOp->insertOperands(
            operandsIndexAndLength.first + numOperands + numOperandDims,
            dynamicDimOperands);
        numOperandDims += dynamicDimOperands.size();
        dispatchOp->insertOperands(operandsIndexAndLength.first + numOperands,
                                   dynamicDimOperands);
        numOperands += dynamicDimOperands.size();
      }

      // For arguments of type flow.dispatch.tensor, create a
      // flow.dispatch.tensor.load to get the replacement values.
      repl = b.create<IREE::Flow::DispatchTensorLoadOp>(
          loc, value.getType().cast<RankedTensorType>(), bbArg, dynamicDimArgs);
    }

    value.replaceUsesWithIf(repl, [&](OpOperand &use) {
      return use.getOwner()
                 ->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>() ==
             dispatchOp;
    });
  }

  // Update the `operand_segment_sizes`.
  auto operandSegmentSizes = dispatchOp->getAttrOfType<DenseI32ArrayAttr>(
      dispatchOp.getOperandSegmentSizesAttrName());
  auto newValues = llvm::to_vector<4>(operandSegmentSizes.asArrayRef());
  newValues[1] = numOperands;
  newValues[2] = numOperandDims;
  dispatchOp->setAttr(dispatchOp.getOperandSegmentSizesAttrName(),
                      b.getDenseI32ArrayAttr(newValues));
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern that create the dispatch region.
//===----------------------------------------------------------------------===//

namespace {
template <typename OpType, template <typename> class Base>
struct CreateDispatchRegionOp : Base<OpType> {
  CreateDispatchRegionOp(MLIRContext *context,
                         const LinalgExt::LinalgTransformationFilter &filter,
                         PatternBenefit benefit = 1)
      : Base<OpType>(context, benefit), transformationFilter(filter) {}

  LogicalResult matchAndRewrite(OpType rootOp,
                                PatternRewriter &rewriter) const override {
    // TODO(ravishankarm): It is getting strange to track when to apply this
    // pattern and when not to. Need to revisit this, with dynamic shape cases
    // in mind.
    if (!hasComputeUsesOutsideDispatch(rootOp)) return failure();
    if (rootOp->template getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }

    if (failed(transformationFilter.checkAndNotify(rewriter, rootOp))) {
      return failure();
    }

    // Get the workload to use for the dispatch.
    FailureOr<SmallVector<Value>> workload =
        getWorkloadForRootOp(rewriter, rootOp.getOperation());
    if (failed(workload)) {
      return failure();
    }

    SmallVector<Operation *> dispatchOps =
        getOperationsToMoveIntoDispatch(rootOp);
    // Create a simple dispatch op with no operands, and not isolated from
    // above.
    auto clonedOps = buildOperandLessFlowDispatchWorkgroupOp(
        rewriter, rootOp.getLoc(), workload.value(), dispatchOps);
    if (failed(clonedOps)) {
      return failure();
    }

    transformationFilter.replaceLinalgTransformationFilter(rewriter, rootOp);
    transformationFilter.replaceLinalgTransformationFilter(
        rewriter, clonedOps.value()[0]);
    return success();
  }

 private:
  LinalgExt::LinalgTransformationFilter transformationFilter;
};
}  // namespace

namespace {
/// Pass declaration.
struct DispatchLinalgOnTensorsPass
    : public DispatchLinalgOnTensorsBase<DispatchLinalgOnTensorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, IREE::Flow::FlowDialect, linalg::LinalgDialect,
                scf::SCFDialect, tensor::TensorDialect>();
  }
  DispatchLinalgOnTensorsPass(bool aggressiveFusion) {
    this->aggressiveFusion = aggressiveFusion;
  }
  DispatchLinalgOnTensorsPass(const DispatchLinalgOnTensorsPass &pass)
      : DispatchLinalgOnTensorsPass(pass.aggressiveFusion) {}
  void runOnOperation() override;

 private:
  Statistic numDispatches{this, "number of dispatches",
                          "Number of Flow dispatches created"};
};
}  // namespace

/// For all ops within `funcOp` tagged as root ops, create dispatch regions.
LogicalResult createDispatchRegionsFromRootOps(mlir::Operation *funcOp,
                                               RewritePatternSet &&patterns) {
  MLIRContext *context = funcOp->getContext();

  // Create the dispatch region, first without the isolate region from above
  // property.
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return failure();
  }

  // Run canonicalization patterns and pattern to resolve tensor.dim of result
  // values into tensor.dim of its operands..
  RewritePatternSet canonicalizationPatterns(context);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(
      canonicalizationPatterns);
  if (failed(applyPatternsAndFoldGreedily(
          funcOp, std::move(canonicalizationPatterns)))) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After dispatch op formation ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // After outlining in dispatch region we can rewrite the dispatch ops with
  // proper captures to make it isolated from above.
  if (funcOp
          ->walk([&](IREE::Flow::DispatchWorkgroupsOp op) -> WalkResult {
            return legalizeDispatchWorkgroupOperands(op);
          })
          .wasInterrupted()) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After dispatch op legalization ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Now try to see if we can tie certain results to operands in order to
  // indicate sharing storage. This need to happen here because it needs to
  // access region block arguments for input/output tensors, which aren't
  // available until now.
  funcOp->walk([&](IREE::Flow::DispatchWorkgroupsOp op) {
    tryToTieOperandsAndResults(op);
  });

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After tieing operands and results ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Finally fold `tensor.insert_slice/extract_slice` operations with
  // `flow.dispatch.tensor.load/store`.
  RewritePatternSet foldExtractInsertSliceOps(context);
  populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
      foldExtractInsertSliceOps, context);
  if (failed(applyPatternsAndFoldGreedily(
          funcOp, std::move(foldExtractInsertSliceOps)))) {
    return failure();
  }

  return success();
}

/// Decide fusion groups and annotate them on the IR via attributes.
static void decideAndAnnotateFusableLinalgOps(
    FunctionOpInterface funcOp, DominanceInfo const &dominanceInfo,
    bool aggressiveFusion) {
  FusionGroupMapping fusionGroups =
      decideFusableLinalgOps(funcOp, dominanceInfo, aggressiveFusion);
  MLIRContext *ctx = funcOp.getContext();
  int64_t ctr = 0;

  for (auto &it : fusionGroups) {
    Operation *rootOp = it.first;
    setRootAttribute(ctx, rootOp, ctr);
    for (Operation *producer : it.second) appendToFusionGroup(producer, {ctr});
    ++ctr;
  }
}

void DispatchLinalgOnTensorsPass::runOnOperation() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();
  DominanceInfo const &dominanceInfo = getAnalysis<DominanceInfo>();
  decideAndAnnotateFusableLinalgOps(funcOp, dominanceInfo, aggressiveFusion);

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After annotating linalg op fusion scheme ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  {
    LinalgExt::LinalgTransformationFilter filterForComputeOps(
        [](Operation *op) { return success(hasRootOpAttribute(op)); }, {},
        StringAttr::get(context, "indispatch"));
    filterForComputeOps.setMatchByDefault();
    RewritePatternSet computeOpDispatchPatterns(context);
    computeOpDispatchPatterns.insert<
        CreateDispatchRegionOp<TilingInterface, OpInterfaceRewritePattern>,
        CreateDispatchRegionOp<tensor::InsertSliceOp, OpRewritePattern>>(
        context, filterForComputeOps);
    if (failed(createDispatchRegionsFromRootOps(
            funcOp, std::move(computeOpDispatchPatterns)))) {
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After first step of dispatch region formation ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  /// Convert remaining ops to Flow ops.
  {
    RewritePatternSet convertToFlowPatterns(context);
    populateTensorToFlowConversionPatterns(context, convertToFlowPatterns);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(
        convertToFlowPatterns);
    IREE::Flow::TensorReshapeOp::getCanonicalizationPatterns(
        convertToFlowPatterns, context);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(convertToFlowPatterns)))) {
      return signalPassFailure();
    }
  }

  /// Move yet more remaining ops into dispatch region.

  // Start with just moving the tensor.insert_slice into its dispatch.
  {
    LinalgExt::LinalgTransformationFilter filterForInsertSliceOps(
        ArrayRef<StringAttr>{}, StringAttr::get(context, "indispatch"));
    RewritePatternSet insertSliceOpDispatchPatterns(context);
    insertSliceOpDispatchPatterns.insert<
        CreateDispatchRegionOp<tensor::InsertSliceOp, OpRewritePattern>>(
        context, filterForInsertSliceOps);
    if (failed(createDispatchRegionsFromRootOps(
            funcOp, std::move(insertSliceOpDispatchPatterns)))) {
      return signalPassFailure();
    }
  }

  // Now move all remaining ops that need to be cleaned up.
  {
    LinalgExt::LinalgTransformationFilter filterForCleanupOps(
        ArrayRef<StringAttr>{}, StringAttr::get(context, "indispatch"));
    RewritePatternSet cleanUpDispatchPatterns(context);
    cleanUpDispatchPatterns.insert<
        CreateDispatchRegionOp<tensor::ExtractSliceOp, OpRewritePattern>>(
        context, filterForCleanupOps);
    if (failed(createDispatchRegionsFromRootOps(
            funcOp, std::move(cleanUpDispatchPatterns)))) {
      return signalPassFailure();
    }
  }

  // Finally walk all the ops and remove the attributes
  funcOp.walk([](Operation *op) {
    removeFusionGroupsAttribute(op);
    removeRootOpAttribute(op);
    op->removeAttr(IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker);
  });
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDispatchLinalgOnTensorsPass(bool aggressiveFusion) {
  return std::make_unique<DispatchLinalgOnTensorsPass>(aggressiveFusion);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
