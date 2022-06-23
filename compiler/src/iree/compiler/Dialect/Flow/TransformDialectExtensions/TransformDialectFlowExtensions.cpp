// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TransformDialectFlowExtensions.h"

#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::TransformDialectFlowExtensions::
    TransformDialectFlowExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Dialect/Flow/TransformDialectExtensions/TransformDialectFlowExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectFlowExtension(
    DialectRegistry &registry) {
  registry.addExtensions<transform_dialect::TransformDialectFlowExtensions>();
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
  Region &r = dispatchOp.workgroup_count();
  assert(r.empty() && "expected block-less workgroup_count region");
  Block *block = rewriter.createBlock(&r);
  rewriter.setInsertionPointToStart(block);

  // Resize to `3` to match IREE's assumptions.
  SmallVector<Value> workgroupCount{foreachThreadOp.getNumThreads()};
  workgroupCount.resize(3, rewriter.create<arith::ConstantIndexOp>(loc, 1));
  auto returnOp = rewriter.create<Flow::ReturnOp>(loc, workgroupCount);

  // Fill the region.
  // For now, this assumes that we only pull in constants.
  // TODO: Iteratively pull operations that are only consuming IndexType.
  BlockAndValueMapping bvm;
  for (auto v : foreachThreadOp.getNumThreads()) {
    auto op = dyn_cast_or_null<arith::ConstantIndexOp>(v.getDefiningOp());
    if (!op) return failure();
    rewriter.clone(*op, bvm);
  }
  rewriter.setInsertionPoint(returnOp);
  rewriter.clone(*returnOp, bvm);
  rewriter.eraseOp(returnOp);
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
       llvm::make_early_inc_range(performConcurrentlyOp.yieldingOps())) {
    auto parallelInsertOp = cast<scf::ParallelInsertSliceOp>(&yieldingOp);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(block.getTerminator());
    auto dynamicDims = Util::findVariadicDynamicDims(
        resultIndex, resultTensorOperands, resultTensorsDynamicDims);
    // clang-format off
    rewriter.create<Flow::DispatchTensorStoreOp>(
        loc,
        parallelInsertOp.getSource(),
        tensorToFlowBvm.lookup(resultTensorOperands[resultIndex++]),
        dynamicDims,
        parallelInsertOp.getMixedOffsets(),
        parallelInsertOp.getMixedSizes(),
        parallelInsertOp.getMixedStrides());
    // clang-format on
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
    Value source = extractSliceOp.source();
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

/// Rewrite a ForeachThreadOp into a Flow::DispatchWorkGroupsOp.
/// This rewrite proceeds in a few steps:
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
  for (const Operation &yieldingOp : performConcurrentlyOp.yieldingOps()) {
    auto parallelInsertOp = cast<scf::ParallelInsertSliceOp>(&yieldingOp);
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
  assert(!valuesDefinedAbove.empty() && "used values defined above is empty");

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
    return foreachThreadOp.emitError(
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
  Region &region = dispatchOp.workgroup_body();
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
        bvm.lookup(en.value()), dynamicDims);
    // Replace the tensor -> flow.dispatch.tensor entry by a
    // tensor -> flow.dispatch.tensor.load entry.
    tensorToFlowBvm.map(en.value(), bvm.lookup(en.value()));
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
transform_dialect::ForeachThreadToFlowDispatchWorkgroupsOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  if (state.getTopLevel()
          ->walk<WalkOrder::PostOrder>([&](scf::ForeachThreadOp op) {
            SimplePatternRewriter rewriter(op);
            if (failed(
                    rewriteForeachThreadToFlowDispatchWorkgroups(op, rewriter)))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return DiagnosedSilenceableFailure::definiteFailure();
  return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Flow/TransformDialectExtensions/TransformDialectFlowExtensionsOps.cpp.inc"
