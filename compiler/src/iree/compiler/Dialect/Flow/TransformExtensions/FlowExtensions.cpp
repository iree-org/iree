// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "FlowExtensions.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler {

IREE::transform_dialect::FlowExtensions::FlowExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensionsOps.cpp.inc"
      >();
}

void registerTransformDialectFlowExtension(DialectRegistry &registry) {
  registry.addExtensions<IREE::transform_dialect::FlowExtensions>();
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
// Patterns for ForallOpToFlow rewrite.
//===---------------------------------------------------------------------===//

/// Populate the workgroup_count region of `dispatchOp`.
/// For now, this only supports constant index ops and empty workload operands.
/// Assumes the IREE::Flow::DispatchWorkgroupsOp is built with an empty region.
static LogicalResult populateWorkgroupCountComputingRegion(
    RewriterBase &rewriter, scf::ForallOp forallOp,
    IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  Region &r = dispatchOp.getWorkgroupCount();
  assert(r.empty() && "expected block-less workgroup_count region");
  Block *block = rewriter.createBlock(&r);
  rewriter.setInsertionPointToStart(block);

  SmallVector<Value> results;
  // For now, this assumes that we only pull in constants.
  // TODO: Iteratively pull operations that are only consuming IndexType.
  for (Value v : forallOp.getUpperBound(rewriter)) {
    auto op = dyn_cast_or_null<arith::ConstantIndexOp>(v.getDefiningOp());
    if (!op)
      return failure();
    results.push_back(
        cast<arith::ConstantIndexOp>(rewriter.clone(*op)).getResult());
  }
  // Resize to `3` to match IREE's assumptions.
  for (unsigned i = results.size(); i < 3; ++i) {
    results.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
  }
  rewriter.create<IREE::Flow::ReturnOp>(loc, results);

  return success();
}

/// Rewrite ParallelInsertSlice ops in `InParallelOp` as Flow
/// DispatchTensorStoreOps.
/// Ops are inserted just before the `block` terminator.
static void rewriteParallelInsertSlices(RewriterBase &rewriter,
                                        scf::ForallOp forallOp,
                                        scf::InParallelOp InParallelOp,
                                        Block &block,
                                        ValueRange resultTensorOperands,
                                        ValueRange resultTensorsDynamicDims,
                                        IRMapping tensorToFlowBvm) {
  Location loc = InParallelOp.getLoc();
  int64_t resultIndex = 0;
  for (const Operation &yieldingOp :
       llvm::make_early_inc_range(InParallelOp.getYieldingOps())) {
    auto parallelInsertOp = cast<tensor::ParallelInsertSliceOp>(&yieldingOp);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(block.getTerminator());
    auto dynamicDims = IREE::Util::findDynamicDimsInList(
        resultIndex, resultTensorOperands, resultTensorsDynamicDims);
    BlockArgument destBbArg =
        llvm::cast<BlockArgument>(parallelInsertOp.getDest());
    assert(destBbArg.getOwner()->getParentOp() == forallOp &&
           "expected that dest is an output bbArg");
    Value dest = forallOp.getTiedOpOperand(destBbArg)->get();
    // clang-format off
    rewriter.create<IREE::TensorExt::DispatchTensorStoreOp>(
        loc,
        parallelInsertOp.getSource(),
        tensorToFlowBvm.lookup(dest),
        dynamicDims,
        parallelInsertOp.getMixedOffsets(),
        parallelInsertOp.getMixedSizes(),
        parallelInsertOp.getMixedStrides());
    // clang-format on
    ++resultIndex;
    rewriter.eraseOp(parallelInsertOp);
  }
}

/// Rewrite ExtractSlice ops in `dispatchOp` as
/// IREE::TensorExt::DispatchTensorLoadOps. Takes a list of all tensor and all
/// tensorDynamicDims operands to the dispatchOp as well as a IRMapping from
/// tensor operands to the corresponding Flow dispatch tensor bbArgs.
static void rewriteExtractSlices(RewriterBase &rewriter, scf::ForallOp forallOp,
                                 IREE::Flow::DispatchWorkgroupsOp dispatchOp,
                                 ValueRange tensorOperands,
                                 ValueRange tensorDynamicDims,
                                 IRMapping tensorToFlowBvm) {
  dispatchOp->walk([&](tensor::ExtractSliceOp extractSliceOp) {
    Value source = extractSliceOp.getSource();
    if (auto sourceBbArg = llvm::dyn_cast<BlockArgument>(source))
      if (sourceBbArg.getOwner()->getParentOp() == forallOp.getOperation())
        source = forallOp.getTiedOpOperand(sourceBbArg)->get();

    auto it = llvm::find(tensorOperands, source);
    if (it == tensorOperands.end())
      return;
    int64_t index = std::distance(tensorOperands.begin(), it);
    Value sourceFlow = tensorToFlowBvm.lookupOrNull(source);
    if (!sourceFlow)
      return;

    Location loc = extractSliceOp.getLoc();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(extractSliceOp);
    auto dynamicDims = IREE::Util::findDynamicDimsInList(index, tensorOperands,
                                                         tensorDynamicDims);
    // clang-format off
    Value load = rewriter.create<IREE::TensorExt::DispatchTensorLoadOp>(
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

static void cloneOpsIntoForallOp(RewriterBase &rewriter,
                                 scf::ForallOp forallOp) {
  // 1. Find all ops that should be cloned into the ForallOp.
  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(forallOp.getRegion(), valuesDefinedAbove);
  // Add all ops who's results are used inside the ForallOp to the
  // worklist.
  llvm::SetVector<Operation *> worklist;
  for (Value v : valuesDefinedAbove)
    if (Operation *op = v.getDefiningOp())
      worklist.insert(op);
  llvm::SmallVector<Operation *> opsToClone;
  llvm::DenseSet<Operation *> visited;

  // Process all ops in the worklist.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (visited.contains(op))
      continue;
    visited.insert(op);

    // Do not clone ops that are not clonable.
    if (!IREE::Flow::isClonableIntoDispatchOp(op))
      continue;

    // Do not clone ParallelInsertSliceOp destinations.
    bool isDestination = any_of(
        forallOp.getTerminator().getYieldingOps(), [&](Operation &insertOp) {
          return cast<tensor::ParallelInsertSliceOp>(&insertOp)
                     .getDest()
                     .getDefiningOp() == op;
        });
    if (isDestination)
      continue;

    opsToClone.push_back(op);

    // Add all operands to the worklist.
    for (Value operand : op->getOperands()) {
      Operation *operandOp = operand.getDefiningOp();
      if (!operandOp)
        continue;
      worklist.insert(operandOp);
    }
  }

  // 2. Clone ops and replace their uses inside the ForallOp.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&forallOp.getRegion().getBlocks().front());
  for (Operation *op : llvm::reverse(opsToClone)) {
    Operation *cloned = rewriter.clone(*op);
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : op->getUses())
      if (forallOp->isProperAncestor(use.getOwner()))
        uses.push_back(&use);
    for (OpOperand *use : uses) {
      unsigned resultNum = llvm::cast<OpResult>(use->get()).getResultNumber();
      rewriter.modifyOpInPlace(
          use->getOwner(), [&]() { use->set(cloned->getOpResult(resultNum)); });
    }
  }
}

/// Rewrite a ForallOp into a IREE::Flow::DispatchWorkGroupsOp.
/// This rewrite proceeds in a few steps:
///   - Step 0: Clone certain ops into the ForallOp (as per IREE
///     heuristic), so that they are part of the dispatch region.
///   - Step 1: Compute the result types and their result dynamic dim operands.
///     This first step takes advantage of the ops contained in the
///     ForallOp terminator and that are tied to the results.
///   - Step 2: Get values defined above and separate them between non-tensors,
///     tensors and introduce appropriate tensor dims.
///   - Step 3: Create ordered vectors of operands to pass to the builder and
///     build the dispatchOp.
///   - Step 4: Populate the workgroupCount region of the dispatchOp and set
///     the workload operands to the values defined above.
///   - Step 5: Fixup dispatchOp bbArgs and terminator.
///   - Step 6: Move the body of forallOp to the dispatchOp.
///   - Step 7: Set up bvm for RAUWIf. In particular, tensor operands become
///     flow dispatch tensor bbArgs and need to be
///     iree_tensor_ext.dispatch.tensor.load'ed.
///   - Step 8: Plug dispatch workgroup id and count values into the bvm.
///   - Step 9. Rewrite tensor::ExtractSlice and ParallelInsert ops to the
///     relevant Flow DispatchTensorLoad/Store version.
///   - Step 10: Perform RAUWIf.
///   - Step 11: Drop the terminator and replace forallOp.
// TODO: n-D ForallOp
FailureOr<IREE::Flow::DispatchWorkgroupsOp>
rewriteForeachThreadToFlowDispatchWorkgroups(scf::ForallOp forallOp,
                                             RewriterBase &rewriter) {
  // Step 0: Clone ops into the ForallOp.
  cloneOpsIntoForallOp(rewriter, forallOp);

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);

  // Entry point start just before the forallOp.
  Location loc = forallOp.getLoc();
  scf::InParallelOp InParallelOp = forallOp.getTerminator();

  // Step 1: Compute all dynamic result dims.
  // The `dest` of the ParallelInsertSliceOp are tied to the results and carry
  // over to the IREE::Flow::DispatchWorkgroupsOp.
  // Use a SetVector to ensure tensor operand uniqueness.
  llvm::SetVector<Value> resultTensorOperands, resultTensorsDynamicDims;
  for (const Operation &yieldingOp : InParallelOp.getYieldingOps()) {
    auto parallelInsertOp = cast<tensor::ParallelInsertSliceOp>(&yieldingOp);
    BlockArgument destBbArg =
        llvm::cast<BlockArgument>(parallelInsertOp.getDest());
    Value dest = forallOp.getTiedOpOperand(destBbArg)->get();
    bool inserted = resultTensorOperands.insert(dest);
    if (!inserted)
      continue;
    auto dynamicDims =
        getIndicesOfDynamicDims(llvm::cast<ShapedType>(dest.getType()));
    for (int64_t dim : dynamicDims)
      resultTensorsDynamicDims.insert(
          rewriter.create<tensor::DimOp>(loc, dest, dim));
  }
  assert(resultTensorOperands.size() == forallOp.getNumResults() &&
         "Expected as many resultTensorOperands as results of forallOp");

  // Step 2. Get values defined above and separate them between non-tensors,
  // tensors and introduce appropriate tensor dims.
  // Tensors that have already been recorded as resultTensorOperands are
  // omitted to avoid duplications.
  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(forallOp.getRegion(), valuesDefinedAbove);

  SmallVector<Value> nonTensorOperands, tensorOperands, tensorDynamicDims;
  for (Value v : valuesDefinedAbove) {
    auto tensorType = llvm::dyn_cast<RankedTensorType>(v.getType());
    if (!tensorType) {
      nonTensorOperands.push_back(v);
      continue;
    }
    if (resultTensorOperands.contains(v))
      continue;
    tensorOperands.push_back(v);
    for (int64_t dim : getIndicesOfDynamicDims(tensorType))
      tensorDynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, v, dim));
  }
  // Also add shared outputs. (These are usually already added as result
  // tensor operands.)
  for (Value v : forallOp.getOutputs()) {
    auto tensorType = llvm::cast<RankedTensorType>(v.getType());
    if (resultTensorOperands.contains(v))
      continue;
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
  // scf::ForallOp tensors inserted into are tied to results and
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
  auto dispatchOp = rewriter.create<IREE::Flow::DispatchWorkgroupsOp>(
      loc,
      /*workload=*/ValueRange{},
      /*resultTypes=*/forallOp.getResultTypes(),
      /*resultDims=*/resultTensorsDynamicDims.getArrayRef(),
      /*operands=*/nonDimOperands,
      /*operandDims=*/allTensorDynamicDims,
      /*tiedOperands=*/llvm::to_vector(tiedOperandsSequence));
  // clang-format on

  // Step 4. Outline the compute workload region and set up the workload
  // operands.
  if (failed(populateWorkgroupCountComputingRegion(rewriter, forallOp,
                                                   dispatchOp)))
    return forallOp->emitOpError(
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
    rewriter.create<IREE::Flow::ReturnOp>(loc);
  }
  // Add trailing index bbArgs and perform a basic sanity check.
  block->addArguments(
      SmallVector<Type>(allTensorDynamicDims.size(), rewriter.getIndexType()),
      SmallVector<Location>(allTensorDynamicDims.size(), loc));
  SmallVector<Value> allOperands = nonDimOperands;
  llvm::append_range(allOperands, allTensorDynamicDims);
  assert(block->getNumArguments() == allOperands.size() &&
         "Expected as many bbArgs as operands");

  // Step 6. Move the body of forallOp to the dispatchOp.
  block->getOperations().splice(block->begin(),
                                forallOp.getRegion().front().getOperations());

  // Step 7. Set up bvm for RAUWIf.
  // Generally, allOperands map to their corresponding bbArg but there is a
  // twist: tensor operands map to iree_tensor_ext.dispatch.tensor bbArgs and we
  // need to insert an explicit IREE::TensorExt::DispatchTensorLoadOp to get
  // back a proper tensor. Save the tensor operand -> flow tensor bbArg mapping
  // in `tensorToFlowBvm`.
  IRMapping bvm, tensorToFlowBvm;
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
    // Warning: findDynamicDimsInList needs to use the RankedTensorTypes and
    // does not work out of the box with IREE::TensorExt::DispatchTensorType.
    auto dynamicDims = IREE::Util::findDynamicDimsInList(
        en.index(), allTensorOperands, allTensorDimsBBArgs);
    auto loadOp = rewriter.create<IREE::TensorExt::DispatchTensorLoadOp>(
        loc, llvm::cast<RankedTensorType>(en.value().getType()),
        tensorToFlowBvm.lookup(en.value()), dynamicDims);
    // Replace the tensor -> iree_tensor_ext.dispatch.tensor entry by a
    // tensor -> iree_tensor_ext.dispatch.tensor.load entry.
    bvm.map(en.value(), loadOp.getResult());
  }

  // Step 8. Plug dispatch workgroup id and count values into the bvm.
  rewriter.setInsertionPointToStart(block);
  SmallVector<Value, 8> workgroupIds, workgroupCounts;
  for (int64_t rank :
       llvm::seq<int64_t>(0, forallOp.getInductionVars().size())) {
    workgroupIds.push_back(
        rewriter.create<IREE::Flow::DispatchWorkgroupIDOp>(loc, rank));
    workgroupCounts.push_back(
        rewriter.create<IREE::Flow::DispatchWorkgroupCountOp>(loc, rank));
  }
  bvm.map(forallOp.getInductionVars(), workgroupIds);
  bvm.map(forallOp.getUpperBound(rewriter), workgroupCounts);

  // Step 9. Rewrite tensor::ExtractSlice and ParallelInsert ops to the
  // relevant Flow DispatchTensorLoad/Store version.
  rewriteParallelInsertSlices(rewriter, forallOp, InParallelOp, *block,
                              resultTensorOperands.getArrayRef(),
                              resultTensorsDynamicDims.getArrayRef(),
                              tensorToFlowBvm);
  rewriteExtractSlices(rewriter, forallOp, dispatchOp, allTensorOperands,
                       allTensorDynamicDims, tensorToFlowBvm);

  // Step 10. Perform RAUWIf.
  for (auto mapEntry : bvm.getValueMap()) {
    assert(mapEntry.first.getType() == mapEntry.second.getType() &&
           "must have the same type");
    mapEntry.first.replaceUsesWithIf(mapEntry.second, [&](OpOperand &use) {
      return dispatchOp->isProperAncestor(use.getOwner());
    });
  }

  // Step 11. Drop the terminator and replace forallOp.
  rewriter.eraseOp(InParallelOp);
  rewriter.replaceOp(forallOp, dispatchOp.getResults());

  return dispatchOp;
}

//===---------------------------------------------------------------------===//
// IREE-specific transformations defined outside of iree_linalg_transform.
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
IREE::transform_dialect::ForeachThreadToFlowDispatchWorkgroupsOp::applyToOne(
    transform::TransformRewriter &rewriter, scf::ForallOp target,
    transform::ApplyToEachResultList &results, transform::TransformState &) {
  IRRewriter patternRewriter(target->getContext());
  FailureOr<IREE::Flow::DispatchWorkgroupsOp> result =
      rewriteForeachThreadToFlowDispatchWorkgroups(target, patternRewriter);
  if (failed(result))
    return emitDefaultDefiniteFailure(target);
  results.push_back(*result);
  return DiagnosedSilenceableFailure::success();
}

void IREE::transform_dialect::ForeachThreadToFlowDispatchWorkgroupsOp::
    getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
IREE::transform_dialect::RegionToWorkgroupsOp::applyToOne(
    transform::TransformRewriter &rewriter, IREE::Flow::DispatchRegionOp target,
    transform::ApplyToEachResultList &results, transform::TransformState &) {
  FailureOr<IREE::Flow::DispatchWorkgroupsOp> result =
      rewriteFlowDispatchRegionToFlowDispatchWorkgroups(target, rewriter);
  if (failed(result))
    return emitDefaultDefiniteFailure(target);
  results.push_back(*result);
  return DiagnosedSilenceableFailure::success();
}

void IREE::transform_dialect::RegionToWorkgroupsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

} // namespace mlir::iree_compiler

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensionsOps.cpp.inc"
