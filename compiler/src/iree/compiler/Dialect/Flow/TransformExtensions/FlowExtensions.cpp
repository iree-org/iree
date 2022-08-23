// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "FlowExtensions.h"

#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/DispatchLinalgOnTensors.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::FlowExtensions::FlowExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectFlowExtension(
    DialectRegistry &registry) {
  registry.addExtensions<transform_dialect::FlowExtensions>();
}

// TODO: Upstream to ShapeType and reuse.
static SmallVector<int64_t> getIndicesOfDynamicDims(ShapedType t) {
  int64_t numDynamicDims = t.getNumDynamicDims();
  SmallVector<int64_t> res(numDynamicDims);
  for (int64_t dim = 0; dim != numDynamicDims; ++dim)
    res[dim] = t.getDynamicDimIndex(dim);
  return res;
}

//===---------------------------------------------------------------------===//
// Patterns for ForeachThreadOpToFlow rewrite.
//===---------------------------------------------------------------------===//

/// Populate the workgroup_count region of `dispatchOp`.
/// For now, this only supports constant index ops and empty workload operands.
/// Assumes the Flow::DispatchWorkgroupsOp is built with an empty region.
static LogicalResult populateWorkgroupCountComputingRegion(
    PatternRewriter &rewriter, scf::ForeachThreadOp foreachThreadOp,
    Flow::DispatchWorkgroupsOp dispatchOp) {
  Location loc = foreachThreadOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  Region &r = dispatchOp.getWorkgroupCount();
  assert(r.empty() && "expected block-less workgroup_count region");
  Block *block = rewriter.createBlock(&r);
  rewriter.setInsertionPointToStart(block);

  SmallVector<Value> results;
  // For now, this assumes that we only pull in constants.
  // TODO: Iteratively pull operations that are only consuming IndexType.
  for (Value v : foreachThreadOp.getNumThreads()) {
    auto op = dyn_cast_or_null<arith::ConstantIndexOp>(v.getDefiningOp());
    if (!op) return failure();
    results.push_back(
        cast<arith::ConstantIndexOp>(rewriter.clone(*op)).getResult());
  }
  // Resize to `3` to match IREE's assumptions.
  for (unsigned i = results.size(); i < 3; ++i) {
    results.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
  }
  rewriter.create<Flow::ReturnOp>(loc, results);

  return success();
}

/// Rewrite ParallelInsertSlice ops in `performConcurrentlyOp` as Flow
/// DispatchTensorStoreOps.
/// Ops are inserted just before the `block` terminator.
static void rewriteParallelInsertSlices(
    PatternRewriter &rewriter, scf::PerformConcurrentlyOp performConcurrentlyOp,
    Block &block, ValueRange resultTensorOperands,
    ValueRange resultTensorsDynamicDims, BlockAndValueMapping tensorToFlowBvm) {
  Location loc = performConcurrentlyOp.getLoc();
  int64_t resultIndex = 0;
  for (const Operation &yieldingOp :
       llvm::make_early_inc_range(performConcurrentlyOp.getYieldingOps())) {
    auto parallelInsertOp = cast<tensor::ParallelInsertSliceOp>(&yieldingOp);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(block.getTerminator());
    auto dynamicDims = Util::findVariadicDynamicDims(
        resultIndex, resultTensorOperands, resultTensorsDynamicDims);
    // clang-format off
    rewriter.create<Flow::DispatchTensorStoreOp>(
        loc,
        parallelInsertOp.getSource(),
        tensorToFlowBvm.lookup(parallelInsertOp.getDest()),
        dynamicDims,
        parallelInsertOp.getMixedOffsets(),
        parallelInsertOp.getMixedSizes(),
        parallelInsertOp.getMixedStrides());
    // clang-format on
    ++resultIndex;
    rewriter.eraseOp(parallelInsertOp);
  }
}

/// Rewrite ExtractSlice ops in `dispatchOp` as Flow::DispatchTensorLoadOps.
/// Takes a list of all tensor and all tensorDynamicDims operands to the
/// dispatchOp as well as a BlockAndValueMapping from tensor operands to the
/// corresponding Flow dispatch tensor bbArgs.
static void rewriteExtractSlices(PatternRewriter &rewriter,
                                 Flow::DispatchWorkgroupsOp dispatchOp,
                                 ValueRange tensorOperands,
                                 ValueRange tensorDynamicDims,
                                 BlockAndValueMapping tensorToFlowBvm) {
  dispatchOp->walk([&](tensor::ExtractSliceOp extractSliceOp) {
    Value source = extractSliceOp.getSource();
    auto it = llvm::find(tensorOperands, source);
    if (it == tensorOperands.end()) return;
    int64_t index = std::distance(tensorOperands.begin(), it);
    Value sourceFlow = tensorToFlowBvm.lookupOrNull(source);
    if (!sourceFlow) return;

    Location loc = extractSliceOp.getLoc();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(extractSliceOp);
    auto dynamicDims =
        Util::findVariadicDynamicDims(index, tensorOperands, tensorDynamicDims);
    // clang-format off
    Value load = rewriter.create<Flow::DispatchTensorLoadOp>(
        loc,
        sourceFlow,
        dynamicDims,
        extractSliceOp.getMixedOffsets(),
        extractSliceOp.getMixedSizes(),
        extractSliceOp.getMixedStrides());
    // clang-format on
    rewriter.replaceOp(extractSliceOp, load);
  });
}

static void cloneOpsIntoForeachThreadOp(RewriterBase &rewriter,
                                        scf::ForeachThreadOp foreachThreadOp) {
  // 1. Find all ops that should be cloned into the ForeachThreadOp.
  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(foreachThreadOp.getRegion(),
                                  valuesDefinedAbove);
  // Add all ops who's results are used inside the ForeachThreadOp to the
  // worklist.
  llvm::SetVector<Operation *> worklist;
  for (Value v : valuesDefinedAbove)
    if (Operation *op = v.getDefiningOp()) worklist.insert(op);
  llvm::SmallVector<Operation *> opsToClone;
  llvm::DenseSet<Operation *> visited;

  // Process all ops in the worklist.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (visited.contains(op)) continue;
    visited.insert(op);

    // Do not clone ops that are not clonable.
    if (!mlir::iree_compiler::IREE::Flow::isClonableIntoDispatchOp(op))
      continue;

    // Do not clone ParallelInsertSliceOp destinations.
    bool isDestination =
        any_of(foreachThreadOp.getTerminator().getYieldingOps(),
               [&](Operation &insertOp) {
                 return cast<tensor::ParallelInsertSliceOp>(&insertOp)
                            .getDest()
                            .getDefiningOp() == op;
               });
    if (isDestination) continue;

    opsToClone.push_back(op);

    // Add all operands to the worklist.
    for (Value operand : op->getOperands()) {
      Operation *operandOp = operand.getDefiningOp();
      if (!operandOp) continue;
      worklist.insert(operandOp);
    }
  }

  // 2. Clone ops and replace their uses inside the ForeachThreadOp.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(
      &foreachThreadOp.getRegion().getBlocks().front());
  for (Operation *op : llvm::reverse(opsToClone)) {
    Operation *cloned = rewriter.clone(*op);
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : op->getUses())
      if (foreachThreadOp->isProperAncestor(use.getOwner()))
        uses.push_back(&use);
    for (OpOperand *use : uses) {
      unsigned resultNum = use->get().cast<OpResult>().getResultNumber();
      rewriter.updateRootInPlace(
          use->getOwner(), [&]() { use->set(cloned->getOpResult(resultNum)); });
    }
  }
}

/// Rewrite a ForeachThreadOp into a Flow::DispatchWorkGroupsOp.
/// This rewrite proceeds in a few steps:
///   - Step 0: Clone certain ops into the ForeachThreadOp (as per IREE
///     heuristic), so that they are part of the dispatch region.
///   - Step 1: Compute the result types and their result dynamic dim operands.
///     This first step takes advantage of the ops contained in the
///     ForeachThreadOp terminator and that are tied to the results.
///   - Step 2: Get values defined above and separate them between non-tensors,
///     tensors and introduce appropriate tensor dims.
///   - Step 3: Create ordered vectors of operands to pass to the builder and
///     build the dispatchOp.
///   - Step 4: Populate the workgroupCount region of the dispatchOp and set
///     the workload operands to the values defined above.
///   - Step 5: Fixup dispatchOp bbArgs and terminator.
///   - Step 6: Move the body of foreachThreadOp to the dispatchOp.
///   - Step 7: Set up bvm for RAUWIf. In particular, tensor operands become
///     flow dispatch tensor bbArgs and need to be
///     flow.dispatch.tensor.load'ed.
///   - Step 8: Plug dispatch workgroup id and count values into the bvm.
///   - Step 9. Rewrite tensor::ExtractSlice and ParallelInsert ops to the
///     relevant Flow DispatchTensorLoad/Store version.
///   - Step 10: Perform RAUWIf.
///   - Step 11: Drop the terminator and replace foreachThreadOp.
// TODO: n-D ForeachThreadOp
FailureOr<Flow::DispatchWorkgroupsOp>
rewriteForeachThreadToFlowDispatchWorkgroups(
    scf::ForeachThreadOp foreachThreadOp, PatternRewriter &rewriter) {
  // Step 0: Clone ops into the ForeachThreadOp.
  cloneOpsIntoForeachThreadOp(rewriter, foreachThreadOp);

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(foreachThreadOp);

  // Entry point start just before the foreachThreadOp.
  Location loc = foreachThreadOp.getLoc();
  scf::PerformConcurrentlyOp performConcurrentlyOp =
      foreachThreadOp.getTerminator();

  // Step 1: Compute all dynamic result dims.
  // The `dest` of the ParallelInsertSliceOp are tied to the results and carry
  // over to the Flow::DispatchWorkgroupsOp.
  // Use a SetVector to ensure tensor operand uniqueness.
  llvm::SetVector<Value> resultTensorOperands, resultTensorsDynamicDims;
  for (const Operation &yieldingOp : performConcurrentlyOp.getYieldingOps()) {
    auto parallelInsertOp = cast<tensor::ParallelInsertSliceOp>(&yieldingOp);
    Value dest = parallelInsertOp.getDest();
    bool inserted = resultTensorOperands.insert(dest);
    if (!inserted) continue;
    auto dynamicDims =
        getIndicesOfDynamicDims(dest.getType().cast<ShapedType>());
    for (int64_t dim : dynamicDims)
      resultTensorsDynamicDims.insert(
          rewriter.create<tensor::DimOp>(loc, dest, dim));
  }
  assert(resultTensorOperands.size() == foreachThreadOp.getNumResults() &&
         "Expected as many resultTensorOperands as results of foreachThreadOp");

  // Step 2. Get values defined above and separate them between non-tensors,
  // tensors and introduce appropriate tensor dims.
  // Tensors that have already been recorded as resultTensorOperands are
  // omitted to avoid duplications.
  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(foreachThreadOp.getRegion(),
                                  valuesDefinedAbove);

  SmallVector<Value> nonTensorOperands, tensorOperands, tensorDynamicDims;
  for (Value v : valuesDefinedAbove) {
    auto tensorType = v.getType().dyn_cast<RankedTensorType>();
    if (!tensorType) {
      nonTensorOperands.push_back(v);
      continue;
    }
    if (resultTensorOperands.contains(v)) continue;
    tensorOperands.push_back(v);
    for (int64_t dim : getIndicesOfDynamicDims(tensorType))
      tensorDynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, v, dim));
  }

  // Step 3. Create ordered vectors of operands to pass to the builder and
  // build the dispatchOp. The dispatchOp is created with an empty
  // workgroup_count region and empty workload. They are populated next
  SmallVector<Value> nonDimOperands;
  llvm::append_range(nonDimOperands, nonTensorOperands);
  llvm::append_range(nonDimOperands, tensorOperands);
  llvm::append_range(nonDimOperands, resultTensorOperands);
  // scf::ForeachThreadOp tensors inserted into are tied to results and
  // translate to the tied operands of the dispatch.
  int64_t sizeNonTensors = nonTensorOperands.size();
  int64_t sizeNonResultTensors = tensorOperands.size();
  int64_t sizeResultTensors = resultTensorOperands.size();
  auto tiedOperandsSequence = llvm::seq<int64_t>(
      sizeNonTensors + sizeNonResultTensors,
      sizeNonTensors + sizeNonResultTensors + sizeResultTensors);
  // Separate out tensorOperands and tensorDynamicDims for RAUWIf purposes.
  SmallVector<Value> allTensorOperands = tensorOperands;
  llvm::append_range(allTensorOperands, resultTensorOperands);
  SmallVector<Value> allTensorDynamicDims = tensorDynamicDims;
  llvm::append_range(allTensorDynamicDims, resultTensorsDynamicDims);
  // clang-format off
  auto dispatchOp = rewriter.create<Flow::DispatchWorkgroupsOp>(
      loc,
      /*workload=*/ValueRange{},
      /*resultTypes=*/foreachThreadOp.getResultTypes(),
      /*resultDims=*/resultTensorsDynamicDims.getArrayRef(),
      /*operands=*/nonDimOperands,
      /*operandDims=*/allTensorDynamicDims,
      /*tiedOperands=*/llvm::to_vector<4>(tiedOperandsSequence));
  // clang-format on

  // Step 4. Outline the compute workload region and set up the workload
  // operands.
  if (failed(populateWorkgroupCountComputingRegion(rewriter, foreachThreadOp,
                                                   dispatchOp)))
    return foreachThreadOp->emitOpError(
               "failed to populate workload region for dispatchOp: ")
           << dispatchOp;

  // Step 5. Fixup dispatchOp bbArgs and terminator.
  // TODO: Ideally the builder would have created the proper bbArgs and the
  // ceremonial terminator.
  // Atm, the bbArgs for the region are missing index entries for the dynamic
  // dims of the tensor operands
  // We add them, following this convention:
  //  - The first `sizeNonTensors` bbArgs correspond to the non-tensor operands.
  //    These are already added by the builder and we leave them alone.
  //  - The next `sizeNonResultTensors + sizeResultTensors` bbArgs correspond to
  //    the tensor operands (non-result tensors followed by result tensors).
  //    These are already added by the builder and we leave them alone.
  //  - The next `tensorDynamicDims.size() + resultTensorsDynamicDims.size()`
  //    bbArgs correspond to the dynamic dimensions of the tensor operands and
  //    tensor results.
  //    These are *not yet* added by the builder and we add them explicitly.
  //    These index bbArgs are added after all tensor bbArgs and become the
  //    trailing bbArgs.
  //    Another possibility would be to interleave (tensor, tensor dynamic
  //    dims) but unless proven wrong, the trailing indices convention is
  //    quite simpler to implement: if bugs surface, these should be fixed or
  //    a real convention + verification should be adopted on the op + builder.
  Region &region = dispatchOp.getWorkgroupBody();
  Block *block = &region.front();
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(block);
    rewriter.create<Flow::ReturnOp>(loc);
  }
  // Add trailing index bbArgs and perform a basic sanity check.
  block->addArguments(
      SmallVector<Type>(allTensorDynamicDims.size(), rewriter.getIndexType()),
      SmallVector<Location>(allTensorDynamicDims.size(), loc));
  SmallVector<Value> allOperands = nonDimOperands;
  llvm::append_range(allOperands, allTensorDynamicDims);
  assert(block->getNumArguments() == allOperands.size() &&
         "Expected as many bbArgs as operands");

  // Step 6. Move the body of foreachThreadOp to the dispatchOp.
  block->getOperations().splice(
      block->begin(), foreachThreadOp.getRegion().front().getOperations());

  // Step 7. Set up bvm for RAUWIf.
  // Generally, allOperands map to their corresponding bbArg but there is a
  // twist: tensor operands map to flow.dispatch.tensor bbArgs and we need to
  // insert an explicit Flow::DispatchTensorLoadOp to get back a proper
  // tensor. Save the tensor operand -> flow tensor bbArg mapping in
  // `tensorToFlowBvm`.
  BlockAndValueMapping bvm, tensorToFlowBvm;
  auto flowBbArgs = block->getArguments().slice(
      sizeNonTensors, sizeNonResultTensors + sizeResultTensors);
  tensorToFlowBvm.map(allTensorOperands, flowBbArgs);
  assert(allOperands.size() == block->getArguments().size() &&
         "expected same number of operands and bbArgs");
  bvm.map(allOperands, block->getArguments());
  auto allTensorDimsBBArgs = block->getArguments().slice(
      nonDimOperands.size(), allTensorDynamicDims.size());
  for (auto en : llvm::enumerate(allTensorOperands)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(block);
    // Warning: findVariadicDynamicDims needs to use the RankedTensorTypes and
    // does not work out of the box with Flow::DispatchTensorType.
    auto dynamicDims = Util::findVariadicDynamicDims(
        en.index(), allTensorOperands, allTensorDimsBBArgs);
    auto loadOp = rewriter.create<Flow::DispatchTensorLoadOp>(
        loc, en.value().getType().cast<RankedTensorType>(),
        tensorToFlowBvm.lookup(en.value()), dynamicDims);
    // Replace the tensor -> flow.dispatch.tensor entry by a
    // tensor -> flow.dispatch.tensor.load entry.
    bvm.map(en.value(), loadOp.getResult());
  }

  // Step 8. Plug dispatch workgroup id and count values into the bvm.
  rewriter.setInsertionPointToStart(block);
  SmallVector<Value, 8> workgroupIds, workgroupCounts;
  for (int64_t rank :
       llvm::seq<int64_t>(0, foreachThreadOp.getThreadIndices().size())) {
    workgroupIds.push_back(
        rewriter.create<Flow::DispatchWorkgroupIDOp>(loc, rank));
    workgroupCounts.push_back(
        rewriter.create<Flow::DispatchWorkgroupCountOp>(loc, rank));
  }
  bvm.map(foreachThreadOp.getThreadIndices(), workgroupIds);
  bvm.map(foreachThreadOp.getNumThreads(), workgroupCounts);

  // Step 9. Rewrite tensor::ExtractSlice and ParallelInsert ops to the
  // relevant Flow DispatchTensorLoad/Store version.
  rewriteParallelInsertSlices(rewriter, performConcurrentlyOp, *block,
                              resultTensorOperands.getArrayRef(),
                              resultTensorsDynamicDims.getArrayRef(),
                              tensorToFlowBvm);
  rewriteExtractSlices(rewriter, dispatchOp, allTensorOperands,
                       allTensorDynamicDims, tensorToFlowBvm);

  // Step 10. Perform RAUWIf.
  for (auto mapEntry : bvm.getValueMap()) {
    assert(mapEntry.first.getType() == mapEntry.second.getType() &&
           "must have the same type");
    mapEntry.first.replaceUsesWithIf(mapEntry.second, [&](OpOperand &use) {
      return dispatchOp->isProperAncestor(use.getOwner());
    });
  }

  // Step 11. Drop the terminator and replace foreachThreadOp.
  rewriter.eraseOp(performConcurrentlyOp);
  rewriter.replaceOp(foreachThreadOp, dispatchOp.getResults());

  return dispatchOp;
}

//===---------------------------------------------------------------------===//
// IREE-specific transformations defined outside of iree_linalg_transform.
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::ForeachThreadToFlowDispatchWorkgroupsOp::applyToOne(
    scf::ForeachThreadOp target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  SimplePatternRewriter rewriter(target->getContext());
  FailureOr<Flow::DispatchWorkgroupsOp> result =
      rewriteForeachThreadToFlowDispatchWorkgroups(target, rewriter);
  if (failed(result))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  results.push_back(*result);
  return DiagnosedSilenceableFailure(success());
}

DiagnosedSilenceableFailure transform_dialect::RegionToWorkgroupsOp::applyToOne(
    Flow::DispatchRegionOp target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  IRRewriter rewriter(target->getContext());
  FailureOr<Flow::DispatchWorkgroupsOp> result =
      rewriteFlowDispatchRegionToFlowDispatchWorkgroups(target, rewriter);
  if (failed(result))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  results.push_back(*result);
  return DiagnosedSilenceableFailure(success());
}

/// Return `true` if the given type is a ShapedType and has at least one
/// dynamic dimension.
static bool hasDynamicShape(Type t) {
  auto shapedType = t.dyn_cast<ShapedType>();
  if (!shapedType) return false;
  return !shapedType.hasStaticShape();
}

/// Reify the dynamic dimensions of the given value.
static LogicalResult reifyDynamicResultDims(OpBuilder b, Value value,
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
    if (tiedOperand) {
#ifndef NDEBUG
      ShapedType tiedOperandType = tiedOperand.getType().cast<ShapedType>();
      assert(tiedOperandType == shapedType && "expected same type");
#endif  // NDEBUG
      return reifyDynamicResultDims(b, tiedOperand, dynamicDims);
    }
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
static FailureOr<Flow::DispatchRegionOp> appendDispatchRegionResult(
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

/// Return `true` if the given op post-dominates the dispatch region.
static bool isAfterRegion(Operation *op, Flow::DispatchRegionOp regionOp) {
  Operation *ancestor = regionOp->getBlock()->findAncestorOpInBlock(*op);
  assert(ancestor && "expected that op and regionOp are in the same block");
  return regionOp->isBeforeInBlock(ancestor);
}

// Clone a `target` op that is preceding the given dispatch region op into the
// dispatch region.
//
// All uses of the target inside of the dispatch region are replaced with the
// results of the cloned op.
//
// If `updateUsesOutsideOfRegion` is set, all uses of the target op after the
// dispatch region, are also updated: The target op's results are returned from
// the dispatch region an used in those places.
//
// Example when `updateUsesOutsideOfRegion` is unset:
//
// %0 = "some_op"() : () -> (tensor<?xf32>)
// %r = flow.dispatch.region -> (tensor<?xf32>{%d0}) {
//   %1 = "another_op"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
//   flow.return %1 : tensor<?xf32>
// }
// %2 = "yet_another_use"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
//
// In this example, "some_op" will be cloned into the dispatch region and the
// OpOperand of "another_op" will be replaced:
//
// %0 = "some_op"() : () -> (tensor<?xf32>)
// %r = flow.dispatch.region -> (tensor<?xf32>{%d0}) {
//   %0_clone = "some_op"() : () -> (tensor<?xf32>)
//   %1 = "another_op"(%0_clone) : (tensor<?xf32>) -> (tensor<?xf32>)
//   flow.return %1 : tensor<?xf32>
// }
// %2 = "yet_another_use"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
static FailureOr<Flow::DispatchRegionOp> clonePrecedingOpIntoDispatchRegion(
    RewriterBase &rewriter, Operation *target, Flow::DispatchRegionOp regionOp,
    bool updateUsesOutsideOfRegion) {
  assert(target->isBeforeInBlock(regionOp) &&
         "expected that target comes first");
  Block &body = regionOp.getBody().front();

  // Gather all uses of `target`.
  SmallVector<OpOperand *> usesInsideOfRegion, usesAfterRegion;
  bool hasUsesBeforeRegion = false;
  for (OpOperand &use : target->getUses()) {
    if (regionOp->isProperAncestor(use.getOwner())) {
      usesInsideOfRegion.push_back(&use);
    } else {
      // Collect only uses that post-dominate the region.
      if (isAfterRegion(use.getOwner(), regionOp)) {
        usesAfterRegion.push_back(&use);
      } else {
        hasUsesBeforeRegion = true;
      }
    }
  }

  // Clone op into dispatch region.
  Operation *newTargetOp;
  if (usesAfterRegion.empty() && !hasUsesBeforeRegion) {
    // Optimization: If there are not uses outside of the region, we can simply
    // move the target instead of cloning it.
    target->moveBefore(&body.front());
    newTargetOp = target;
  } else {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&body);
    newTargetOp = rewriter.clone(*target);

    // Replace all uses in the dispatch region.
    for (OpOperand *use : usesInsideOfRegion) {
      rewriter.updateRootInPlace(use->getOwner(), [&]() {
        use->set(newTargetOp->getResult(
            use->get().cast<OpResult>().getResultNumber()));
      });
    }
  }

  // Replace all uses outside of the dispatch region.
  if (updateUsesOutsideOfRegion && !usesAfterRegion.empty()) {
    // Fail if there are uses before the dispatch region. In that case it does
    // usually not make sense to update uses after the region; we can just keep
    // using the original op result.
    if (hasUsesBeforeRegion) return failure();

    unsigned previousNumResults = regionOp->getNumResults();

    // Note: Appending results one-by-one here so that this can be extended to
    // specific results in the future. Many ops have just one result, so this
    // should not be a large overhead.
    for (Value v : newTargetOp->getResults()) {
      auto newRegionOp = appendDispatchRegionResult(rewriter, regionOp, v);
      if (failed(newRegionOp)) return failure();
      regionOp = *newRegionOp;
    }

    // Replace uses of `target` after the dispatch region.
    for (OpOperand *use : usesAfterRegion) {
      assert(DominanceInfo().properlyDominates(regionOp, use->getOwner()) &&
             "all target uses must be inside or after regionOp");
      rewriter.updateRootInPlace(use->getOwner(), [&]() {
        use->set(
            regionOp->getResult(previousNumResults +
                                use->get().cast<OpResult>().getResultNumber()));
      });
    }
  }

  // Remove the original target if it no longer has any uses.
  if (target->use_empty()) rewriter.eraseOp(target);

  return regionOp;
}

// Move a `target` op that is preceding the given dispatch region op into the
// dispatch region. All uses of the target must be inside the region.
static FailureOr<Flow::DispatchRegionOp> movePrecedingOpIntoDispatchRegion(
    RewriterBase &rewriter, Operation *target,
    Flow::DispatchRegionOp regionOp) {
  assert(llvm::all_of(target->getUses(),
                      [&](OpOperand &use) {
                        return regionOp->isProperAncestor(use.getOwner());
                      }) &&
         "cannot move target into region");
  return clonePrecedingOpIntoDispatchRegion(
      rewriter, target, regionOp, /*updateUsesOutsideOfRegion=*/false);
}

DiagnosedSilenceableFailure
transform_dialect::ClonePrecedingOpIntoDispatchRegionOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  ArrayRef<Operation *> targetOps = state.getPayloadOps(getTarget());
  ArrayRef<Operation *> dispatchRegion =
      state.getPayloadOps(getDispatchRegion());

  if (dispatchRegion.size() != 1)
    return DiagnosedSilenceableFailure(this->emitOpError(
        "requires exactly one target/dispatch region handle"));

  auto regionOp = dyn_cast<Flow::DispatchRegionOp>(dispatchRegion.front());
  if (!regionOp)
    return DiagnosedSilenceableFailure(
        this->emitOpError("expected 'dispatch.region' operand"));

  // We are cloning ops one-by-one, so the order must be inversed (as opposed
  // to cloning all ops in one go).
  SmallVector<Operation *> targetOpsList(targetOps.begin(), targetOps.end());
  bool sortResult = computeTopologicalSorting(
      dispatchRegion.front()->getBlock(), targetOpsList);
  (void)sortResult;
  assert(sortResult && "unable to sort topologically");
  SmallVector<Operation *> orderedTargets =
      llvm::to_vector(llvm::reverse(targetOps));
  IRRewriter rewriter(regionOp->getContext());
  for (Operation *target : orderedTargets) {
    auto newRegionOp = clonePrecedingOpIntoDispatchRegion(
        rewriter, target, regionOp, getUpdateUsesOutsideOfRegion());
    if (failed(newRegionOp))
      return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
    regionOp = *newRegionOp;
  }

  transformResults.set(getTransformed().cast<OpResult>(),
                       regionOp.getOperation());
  return DiagnosedSilenceableFailure(success());
}

// Clone a `target` op that is succeeding the given dispatch region op into the
// dispatch region.
//
// All operands of the target are replaced with values defined inside of the
// dispatch region when possible.
//
// If `updateUsesOutsideOfRegion` is set, all uses of the target op after the
// dispatch region, are updated: The target op's results are returned from the
// the dispatch region an used in those places.
//
// Example when `updateUsesOutsideOfRegion` is set:
//
// %r = flow.dispatch.region -> (tensor<?xf32>{%d0}) {
//   %1 = "another_op"() : () -> (tensor<?xf32>)
//   flow.return %1 : tensor<?xf32>
// }
// %0 = "some_op"(%r) : (tensor<?xf32>) -> (tensor<?xf32>)
// %2 = "yet_another_use"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
//
// In this example, "some_op" will be cloned into the dispatch region and the
// OpOperand of "yet_another_use" will be replaced:
//
// %r:2 = flow.dispatch.region -> (tensor<?xf32>{%d0}) {
//   %1 = "another_op"() : () -> (tensor<?xf32>)
//   %0_clone = "some_op"(%1) : (tensor<?xf32>) -> (tensor<?xf32>)
//   flow.return %1, %0_clone : tensor<?xf32>, tensor<?xf32>
// }
// %2 = "yet_another_use"(%r#1) : (tensor<?xf32>) -> (tensor<?xf32>)
static FailureOr<Flow::DispatchRegionOp> cloneSucceedingOpIntoDispatchRegion(
    RewriterBase &rewriter, Operation *target, Flow::DispatchRegionOp regionOp,
    bool updateUsesOutsideOfRegion) {
  assert(regionOp->isBeforeInBlock(target) &&
         "expected that region op comes first");
  Block &body = regionOp.getBody().front();

  // Gather all uses of `target`.
  SmallVector<OpOperand *> usesOutsideOfRegion;
  for (OpOperand &use : target->getUses()) usesOutsideOfRegion.push_back(&use);

  // Clone op into dispatch region.
  auto returnOp = cast<Flow::ReturnOp>(body.getTerminator());
  Operation *newTargetOp;
  if (updateUsesOutsideOfRegion) {
    // Optimization: If all uses outside of the region are updated, the target
    // can simply be moved instead of cloned.
    target->moveBefore(returnOp);
    newTargetOp = target;
  } else {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(returnOp);
    newTargetOp = rewriter.clone(*target);
  }

  // Replace all operands that are results of the regionOp.
  for (OpOperand &operand : newTargetOp->getOpOperands()) {
    if (operand.get().getDefiningOp() == regionOp) {
      unsigned resultNumber = operand.get().cast<OpResult>().getResultNumber();
      operand.set(returnOp->getOperand(resultNumber));
    }
  }

  // Replace all uses outside of the dispatch region.
  if (updateUsesOutsideOfRegion) {
    unsigned previousNumResults = regionOp->getNumResults();

    // Note: Appending results one-by-one here so that this can be extended to
    // specific results in the future. Many ops have just one result, so this
    // should not be a large overhead.
    for (Value v : newTargetOp->getResults()) {
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

DiagnosedSilenceableFailure
transform_dialect::CloneSucceedingOpIntoDispatchRegionOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  ArrayRef<Operation *> targetOps = state.getPayloadOps(getTarget());
  ArrayRef<Operation *> dispatchRegion =
      state.getPayloadOps(getDispatchRegion());

  if (dispatchRegion.size() != 1)
    return DiagnosedSilenceableFailure(this->emitOpError(
        "requires exactly one target/dispatch region handle"));

  auto regionOp = dyn_cast<Flow::DispatchRegionOp>(dispatchRegion.front());
  if (!regionOp)
    return DiagnosedSilenceableFailure(
        this->emitOpError("expected 'dispatch.region' operand"));

  SmallVector<Operation *> orderedTargets(targetOps.begin(), targetOps.end());
  bool sortResult = computeTopologicalSorting(
      dispatchRegion.front()->getBlock(), orderedTargets);
  (void)sortResult;
  assert(sortResult && "unable to sort topologically");
  IRRewriter rewriter(regionOp->getContext());
  for (Operation *target : orderedTargets) {
    auto newRegionOp = cloneSucceedingOpIntoDispatchRegion(
        rewriter, target, regionOp, getUpdateUsesOutsideOfRegion());
    if (failed(newRegionOp))
      return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
    regionOp = *newRegionOp;
  }

  transformResults.set(getTransformed().cast<OpResult>(),
                       regionOp.getOperation());
  return DiagnosedSilenceableFailure(success());
}

static Flow::DispatchRegionOp makeEmptyDispatchRegion(RewriterBase &rewriter,
                                                      Location loc) {
  OpBuilder::InsertionGuard guard(rewriter);

  // Create RegionOp.
  auto regionOp = rewriter.create<Flow::DispatchRegionOp>(
      loc, /*resultTypes=*/TypeRange(), /*dynamicDims=*/ValueRange());
  Block &body = regionOp.getBody().emplaceBlock();
  rewriter.setInsertionPointToStart(&body);
  rewriter.create<Flow::ReturnOp>(loc, ValueRange());

  return regionOp;
}

DiagnosedSilenceableFailure
transform_dialect::WrapInDispatchRegionOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  IRRewriter rewriter(target->getContext());

  // Make an empty dispatch region right before the target.
  rewriter.setInsertionPointAfter(target);
  Flow::DispatchRegionOp regionOp =
      makeEmptyDispatchRegion(rewriter, target->getLoc());

  // Move the target into the dispatch region.
  auto newRegionOp = clonePrecedingOpIntoDispatchRegion(
      rewriter, target, regionOp, /*updateUsesOutsideOfRegion=*/true);
  if (failed(newRegionOp))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));

  results.push_back(*newRegionOp);
  return DiagnosedSilenceableFailure(success());
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensionsOps.cpp.inc"
