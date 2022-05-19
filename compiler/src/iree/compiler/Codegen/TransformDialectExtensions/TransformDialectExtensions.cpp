// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TransformDialectExtensions.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree-dialects/Transforms/Functional.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE;

// TODO: Upstream to ShapeType and reuse.
static SmallVector<int64_t> getIndicesOfDynamicDims(ShapedType t) {
  int64_t numDynamicDims = t.getNumDynamicDims();
  SmallVector<int64_t> res(numDynamicDims);
  for (int64_t dim = 0; dim != numDynamicDims; ++dim)
    res[dim] = t.getDynamicDimIndex(dim);
  return res;
}

//===---------------------------------------------------------------------===//
// Patterns for InParallelToFlow rewrite.
//===---------------------------------------------------------------------===//

struct RewriteExtractSliceAsFlowTensorLoad
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  RewriteExtractSliceAsFlowTensorLoad(
      MLIRContext *context, BlockAndValueMapping bvm,

      DenseMap<Value, SmallVector<Value>> flowTensorToDimsMap)
      : OpRewritePattern<tensor::ExtractSliceOp>(context),
        bvm(bvm),
        flowTensorToDimsMap(flowTensorToDimsMap) {}

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    auto replacement = bvm.lookupOrNull(extractSliceOp.source())
                           .dyn_cast_or_null<BlockArgument>();
    if (!replacement) return failure();

    auto dispatchOp =
        extractSliceOp->getParentOfType<Flow::DispatchWorkgroupsOp>();
    if (!dispatchOp) return failure();

    Location loc = extractSliceOp.getLoc();
    // Get bbargs for dynamic dims of replacement.
    // SmallVector<Value> dynamicDimArgs =
    //     dispatchOp.getOperandDynamicDims(replacement.getArgNumber());
    Value loadOp = rewriter.create<Flow::DispatchTensorLoadOp>(
        loc, extractSliceOp.getType(), replacement,
        flowTensorToDimsMap.lookup(replacement),
        extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
        extractSliceOp.getMixedStrides());
    rewriter.replaceOp(extractSliceOp, loadOp);

    return success();
  }

  BlockAndValueMapping bvm;
  DenseMap<Value, SmallVector<Value>> flowTensorToDimsMap;
};

// After outlining in dispatch region we can rewrite the dispatch ops with
// proper captures.
static LogicalResult legalizeDispatchWorkgroupOperands(
    PatternRewriter &rewriter, Flow::DispatchWorkgroupsOp dispatchOp,
    const SmallVector<Value> &tiedResultOperands) {
  Location loc = dispatchOp.getLoc();
  Region &region = dispatchOp.body();
  Block &block = region.front();
  // Helper function that keeps track of bbArg insertion point.
  int64_t numResultTensors = block.getNumArguments();
  auto getArgumentIndexOfFirstResultTensor = [&]() -> int64_t {
    return block.getNumArguments() - numResultTensors;
  };

  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  if (valuesDefinedAbove.empty()) return success();

  SmallVector<Value> nonTensorOperands, tensorOperands, tensorDimOperands;
  OpBuilder::InsertionGuard insertionGuard(rewriter);
  rewriter.setInsertionPointToStart(&block);
  BlockAndValueMapping bvm;
  BlockAndValueMapping bvmTensorToFlowTensor;

  // Handle all non-tensor operands:
  //   - add a new operand and a new bbArg for each
  //   - add a bvm entry operand -> bbArg for each
  for (Value v : valuesDefinedAbove) {
    // Create a new basic block argument for this value.
    Type valueType = v.getType();
    if (valueType.isa<RankedTensorType>()) continue;
    nonTensorOperands.push_back(v);
    Value bbArg = block.insertArgument(getArgumentIndexOfFirstResultTensor(),
                                       valueType, loc);
    bvm.map(v, bbArg);
    continue;
  }

  // Handle all tensor operands:
  //   - add a new operand and a new bbArg for each
  //   - *do not* add a bvm entry operand -> bbArg yet, these will be replaced
  //     by a Flow::DispatchTensorLoadOp(bbArg) which cannot yet be constructed.
  for (Value v : valuesDefinedAbove) {
    // Create a new basic block argument for this value.
    Type valueType = v.getType();
    auto tensorType = valueType.dyn_cast<RankedTensorType>();
    if (!tensorType) continue;
    // TODO: Helper Flow::DispatchTensorType::Builder from RankedTensorType.
    Type flowTensorType = Flow::DispatchTensorType::get(
        Flow::TensorAccess::ReadOnly, tensorType.getShape(),
        tensorType.getElementType());
    Value flowTensor = block.insertArgument(
        getArgumentIndexOfFirstResultTensor(), flowTensorType, loc);
    bvmTensorToFlowTensor.map(v, flowTensor);
    tensorOperands.push_back(v);
  }

  // Revisit tensor operands to handle all tensor operand dims:
  //   - add a new tensor.dim + operand + bbArg for each dynamic dimension
  //   - add a new Flow::DispatchTensorLoadOp
  //   - add a bvm entry bbArg -> Flow::DispatchTensorLoadOp
  // Save the newly created index bbArgs for the dynamic dims of each tensor.
  DenseMap<Value, SmallVector<Value>> flowTensorToDimsMap;
  for (Value v : tensorOperands) {
    auto tensorType = v.getType().cast<RankedTensorType>();
    auto dynamicDims = getIndicesOfDynamicDims(tensorType);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(dispatchOp);
      for (int64_t dim : dynamicDims)
        tensorDimOperands.push_back(
            rewriter.create<tensor::DimOp>(loc, v, dim));
    }
    SmallVector<Value> dynamicDimArgs;
    for (int64_t i = 0, e = tensorType.getRank(); i != e; ++i) {
      dynamicDimArgs.push_back(block.insertArgument(
          getArgumentIndexOfFirstResultTensor(), rewriter.getIndexType(), loc));
    }
    Value repl = rewriter.create<Flow::DispatchTensorLoadOp>(
        loc, v.getType().cast<RankedTensorType>(),
        bvmTensorToFlowTensor.lookup(v), dynamicDimArgs);

    flowTensorToDimsMap.insert(
        std::make_pair(bvmTensorToFlowTensor.lookup(v), dynamicDimArgs));
    bvm.map(v, repl);
  }

  // Set the dispatchOp dim and non-dim operands.
  // Use the xxxMutable interface to avoid needing to update the operands
  // segment sizes attributes manually.
  SmallVector<Value> nonDimOperands;
  llvm::append_range(nonDimOperands, nonTensorOperands);
  llvm::append_range(nonDimOperands, tensorOperands);
  dispatchOp.operandsMutable().assign(nonDimOperands);
  dispatchOp.operand_dimsMutable().assign(tensorDimOperands);

  // Replace all uses of values defined above by entries in the bvm.
  // Except for those uses that we need to replace specially (i.e. the
  // tensors flowing into tensor::ExtractSliceOp).
  for (auto value : valuesDefinedAbove) {
    value.replaceUsesWithIf(bvm.lookup(value), [&](OpOperand &use) {
      if (!dispatchOp->isProperAncestor(use.getOwner())) return false;
      if (!value.getType().isa<RankedTensorType>()) return true;
      return !isa<tensor::ExtractSliceOp>(use.getOwner());
    });
  }

  RewritePatternSet patterns(dispatchOp.getContext());
  patterns.add<RewriteExtractSliceAsFlowTensorLoad>(
      dispatchOp.getContext(), bvmTensorToFlowTensor, flowTensorToDimsMap);
  (void)applyPatternsAndFoldGreedily(dispatchOp, std::move(patterns));

  // Set the tie between results and corresponding BlockArgument.
  // At this time, all tensors that are inparallel_insert_slice'd into are
  for (auto en : llvm::enumerate(tiedResultOperands)) {
    // Each tied result operand .
    BlockArgument readonlyFlowTensor =
        bvmTensorToFlowTensor.lookup(en.value()).cast<BlockArgument>();
    BlockArgument writeonlyFlowTensor =
        block.getArgument(getArgumentIndexOfFirstResultTensor() + en.index());
    auto writeOnlyFlowTensorType =
        writeonlyFlowTensor.getType().cast<Flow::DispatchTensorType>();
    writeonlyFlowTensor.setType(Flow::DispatchTensorType::get(
        Flow::TensorAccess::ReadWrite, writeOnlyFlowTensorType.getShape(),
        writeOnlyFlowTensorType.getElementType()));
    readonlyFlowTensor.replaceUsesWithIf(
        writeonlyFlowTensor, [&](OpOperand &use) {
          return dispatchOp->isProperAncestor(use.getOwner());
        });
    dispatchOp.setTiedResultOperandIndex(en.index(),
                                         readonlyFlowTensor.getArgNumber());
  }

  return success();
}

/// Pattern to rewrite a InParallelOp to the Flow dialect.
// TODO: n-D InParallelOp
struct InParallelOpToOperandlessFlowRewriter
    : public OpRewritePattern<LinalgExt::InParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<Flow::DispatchWorkgroupsOp> returningMatchAndRewrite(
      LinalgExt::InParallelOp inParallelOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(LinalgExt::InParallelOp inParallelOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(inParallelOp, rewriter);
  }
};

FailureOr<Flow::DispatchWorkgroupsOp>
InParallelOpToOperandlessFlowRewriter::returningMatchAndRewrite(
    LinalgExt::InParallelOp inParallelOp, PatternRewriter &rewriter) const {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(inParallelOp);

  // Compute all dynamic results, the dispatch region will need that.
  // Use the fact that the dest of the ParallelInsertSliceOp are tied to the
  // results.
  Location loc = inParallelOp.getLoc();
  SmallVector<Value> allResultsDynamicDims, tiedResultOperands;
  for (LinalgExt::ParallelInsertSliceOp insertSliceOp :
       inParallelOp.getTerminator().yieldingOps()) {
    Value dest = insertSliceOp.dest();
    auto dynamicDims =
        getIndicesOfDynamicDims(dest.getType().cast<ShapedType>());
    {
      for (int64_t dim : dynamicDims)
        allResultsDynamicDims.push_back(
            rewriter.create<tensor::DimOp>(loc, dest, dim));
    }
    tiedResultOperands.push_back(dest);
  }
  assert(tiedResultOperands.size() == inParallelOp.getNumResults() &&
         "Expected as many tiedResultOperands as results of inParallelOp");

  // Add a fake workload of exactly the number of threads.
  SmallVector<Value> fakeWorkload{inParallelOp.num_threads()};
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  fakeWorkload.resize(3, one);

  // Create a dispatch op with just the `flow.return` terminator.
  auto dispatchOp = rewriter.create<Flow::DispatchWorkgroupsOp>(
      loc, /*workload=*/fakeWorkload, inParallelOp.getResultTypes(),
      allResultsDynamicDims,
      /*operands=*/ArrayRef<Value>{}, /*operandDims=*/ArrayRef<Value>{},
      /*tiedOperands=*/ArrayRef<int64_t>{});
  Region &region = dispatchOp.body();
  Block *block = &region.front();
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(block);
    rewriter.create<Flow::ReturnOp>(loc);
  }

  // Move the body of inParallel to the dispatchOp.
  LinalgExt::PerformConcurrentlyOp terminator = inParallelOp.getTerminator();
  block->getOperations().splice(block->begin(),
                                inParallelOp.region().front().getOperations());

  // Plug dispatch workgroup id and count values.
  rewriter.setInsertionPointToStart(block);
  auto idXOp = rewriter.create<Flow::DispatchWorkgroupIDOp>(loc, 0);
  inParallelOp.getThreadIndex().replaceAllUsesWith(idXOp);
  auto countXOp = rewriter.create<Flow::DispatchWorkgroupCountOp>(loc, 0);
  // inParallelOp.num_threads() is given as an operand to dispatchOp, only
  // replace uses inside dispatchOp.
  inParallelOp.num_threads().replaceUsesWithIf(
      countXOp.getResult(), [&](OpOperand &use) {
        return dispatchOp->isProperAncestor(use.getOwner());
      });

  // Rewrite ops in terminator as Flow tensor store ops.
  auto resultArgs = region.getArguments();
  int64_t resultIndex = 0;
  int64_t resultDynamicDimsPos = 0;
  for (LinalgExt::ParallelInsertSliceOp parallelInsertOp :
       terminator.yieldingOps()) {
    unsigned numDynamicDims = parallelInsertOp.yieldedType()
                                  .cast<RankedTensorType>()
                                  .getNumDynamicDims();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(block->getTerminator());
    rewriter.create<Flow::DispatchTensorStoreOp>(
        loc, parallelInsertOp.source(), resultArgs[resultIndex++],
        ArrayRef<Value>(allResultsDynamicDims)
            .slice(resultDynamicDimsPos, numDynamicDims),
        parallelInsertOp.getMixedOffsets(), parallelInsertOp.getMixedSizes(),
        parallelInsertOp.getMixedStrides());
    resultDynamicDimsPos += numDynamicDims;
  }

  // Try to legalize the dispatch operands and bbArgs.
  if (failed(legalizeDispatchWorkgroupOperands(rewriter, dispatchOp,
                                               tiedResultOperands)))
    return failure();

  // Drop the inParallel terminator and replace inParallel.
  rewriter.eraseOp(terminator);
  rewriter.replaceOp(inParallelOp, dispatchOp.getResults());

  return dispatchOp;
}

//===---------------------------------------------------------------------===//
// Patterns for InParallelToHAL rewrite.
//===---------------------------------------------------------------------===//

/// Pattern to rewrite a InParallelOp to the HAL dialect.
/// This is written in a way that allows rewriting a single InParallelOp at a
/// time. Atm this is used in a global rewrite of all InParallelOps in a
/// region to properly account for ordering requirements (i.e. innermost
/// InParallelOp need to be rewritten before outermost ones). In the future,
/// finer-grained single op control may be better.
struct InParallelOpToHALRewriter
    : public OpRewritePattern<LinalgExt::InParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<SmallVector<Operation *>> returningMatchAndRewrite(
      LinalgExt::InParallelOp inParallelOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(LinalgExt::InParallelOp inParallelOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(inParallelOp, rewriter);
  }
};

static int64_t getNumEnclosingInParallelOps(Operation *op) {
  int64_t numInParallelOps = 0;
  while (auto parentOp = op->getParentOfType<LinalgExt::InParallelOp>()) {
    op = parentOp;
    ++numInParallelOps;
  }
  return numInParallelOps;
}

/// Return the unique HALExecutableEntryPointOp within parentFuncOp or creates
/// a new op whose terminator returns the triple (one, one, one).
/// This is a placeholder into which more information can be inserted to build
/// the proper workgroup counts.
/// Return nullptr if the parentFuncOp contains more than a single
/// HALExecutableEntryPointOp.
// TODO: This will not be neded once transform dialect can use real HAL ops.
static HAL::ExecutableEntryPointOp ensureEntryPointOpWithBody(
    PatternRewriter &rewriter, HAL::ExecutableVariantOp variantOp) {
  HAL::ExecutableEntryPointOp entryPointOp;
  WalkResult res = variantOp.walk([&](HAL::ExecutableEntryPointOp op) {
    if (entryPointOp) return WalkResult::interrupt();
    entryPointOp = op;
    return WalkResult::advance();
  });
  assert(entryPointOp && !res.wasInterrupted() &&
         "expected one single entry point");
  if (res.wasInterrupted()) return nullptr;

  if (entryPointOp.getWorkgroupCountBody()) return entryPointOp;

  Location loc = entryPointOp.getLoc();
  int64_t numWorkgroupDims = HAL::ExecutableEntryPointOp::getNumWorkgroupDims();
  Block &block = entryPointOp.workgroup_count_region().emplaceBlock();
  block.addArgument(rewriter.getType<HAL::DeviceType>(), loc);
  for (int64_t i = 0; i < numWorkgroupDims; ++i)
    block.addArgument(rewriter.getIndexType(), loc);

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&block);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> workgroupCounts(numWorkgroupDims, one);
  rewriter.create<HAL::ReturnOp>(loc, TypeRange{}, workgroupCounts);
  return entryPointOp;
}

FailureOr<SmallVector<Operation *>>
InParallelOpToHALRewriter::returningMatchAndRewrite(
    LinalgExt::InParallelOp inParallelOp, PatternRewriter &rewriter) const {
  if (!inParallelOp->getParentOfType<HAL::ExecutableVariantOp>())
    return inParallelOp->emitError("No enclosing HAL::ExecutableVariantOp");

  // Rewriter-based RAUW operates on Operation* atm, bail if we can't get it.
  Operation *numThreadDefiningOp = inParallelOp.num_threads().getDefiningOp();
  if (!numThreadDefiningOp) {
    return inParallelOp->emitError(
        "Cannot find a defining op for the num_threads operand");
  }

  // Rewrites must happen bottom-up to get the proper workgroup id ordering.
  LinalgExt::InParallelOp nestedInParallelOp;
  WalkResult walkResult = inParallelOp.walk([&](LinalgExt::InParallelOp op) {
    nestedInParallelOp = op;
    return (op == inParallelOp) ? WalkResult::advance()
                                : WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted()) {
    return inParallelOp->emitError(
               "Failed to rewrite top-level InParallelOp with nested "
               "InParallelOp:\n")
           << *nestedInParallelOp.getOperation();
  }

  // #of enclosing InParallelOp determine the #idx in:
  //   hal.interface.workgroup.id[#idx] : index
  //   hal.interface.workgroup.count[#idx] : index
  int64_t numEnclosingInParallelOps =
      getNumEnclosingInParallelOps(inParallelOp);
  if (numEnclosingInParallelOps >=
      HAL::ExecutableEntryPointOp::getNumWorkgroupDims()) {
    return inParallelOp->emitError(
        "Too many InParallelOps, exceeds the maximum number of workgroup "
        "dims");
  }

  // Custom hal.executable.entry_point.
  // TODO: pull in the proper operands as the bbArgs to allow dynamic sizes.
  HAL::ExecutableVariantOp variantOp =
      inParallelOp->getParentOfType<HAL::ExecutableVariantOp>();
  auto region = std::make_unique<Region>();
  HAL::ExecutableEntryPointOp entryPointOp =
      ensureEntryPointOpWithBody(rewriter, variantOp);

  // At this point, the region is known to have a body.
  HAL::ReturnOp returnOp = cast<HAL::ReturnOp>(
      entryPointOp.getWorkgroupCountBody()->getTerminator());
  // Update the numEnclosingInParallelOps^th operand with an in-body clone of
  // numThreadDefiningOp.
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(returnOp);
    // TODO: This can only work on constant ops atm. In the future, handle
    // copying full backward slices (but we'll need a better
    // HALExecutableEntryPointOp bbargs contract).
    Operation *op = rewriter.clone(*numThreadDefiningOp);
    rewriter.startRootUpdate(returnOp);
    SmallVector<Value> operands = returnOp->getOperands();
    // TODO: ensure this is already 1 or the same value otherwise we are in
    // presence of sibling InParallelOp's that are incompatible.
    operands[numEnclosingInParallelOps] = op->getResult(0);
    returnOp->setOperands(operands);
    rewriter.finalizeRootUpdate(returnOp);
  }

  Location loc = inParallelOp.getLoc();
  auto idOp = rewriter.create<HAL::InterfaceWorkgroupIDOp>(
      loc, numEnclosingInParallelOps);
  auto countOp = rewriter.create<HAL::InterfaceWorkgroupCountOp>(
      loc, numEnclosingInParallelOps);

  // Get a reference to the terminator that will subsequently be moved.
  LinalgExt::PerformConcurrentlyOp performConcurrentlyOp =
      inParallelOp.getTerminator();

  // First, update the uses of num_threads() within the inParallelOp block.
  rewriter.replaceOpWithinBlock(numThreadDefiningOp, countOp.result(),
                                &inParallelOp.region().front());

  // Steal the iree_compiler::LinalgExt::InParallel ops, right before
  // the inParallelOp. Replace the bbArg by the HAL id.
  // This includes the terminator `performConcurrentlyOp` that still needs to
  // be erased separately.
  SmallVector<Value> bbArgsTranslated{idOp.result()};
  rewriter.mergeBlockBefore(&inParallelOp.region().front(), inParallelOp,
                            bbArgsTranslated);

  // If we were operating on buffers, we are now done.
  if (inParallelOp->getNumResults() == 0) {
    rewriter.eraseOp(performConcurrentlyOp);
    rewriter.eraseOp(inParallelOp);
    return SmallVector<Operation *>();
  }

  // On tensors, we need to create sequential insertSlice ops.
  rewriter.setInsertionPoint(inParallelOp);
  SmallVector<Value> results;
  SmallVector<Operation *> resultingOps;
  for (LinalgExt::ParallelInsertSliceOp op :
       performConcurrentlyOp.yieldingOps()) {
    resultingOps.push_back(rewriter.create<tensor::InsertSliceOp>(
        loc, op.source(), op.dest(), op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides()));
    results.push_back(resultingOps.back()->getResult(0));
  }
  rewriter.replaceOp(inParallelOp, results);
  rewriter.eraseOp(performConcurrentlyOp);

  return resultingOps;
}

/// Pattern to rewrite a Flow::DispatchTensorStoreOp.
struct BridgeInParallelToFlowAbstractionGap
    : public OpRewritePattern<LinalgExt::InParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<SmallVector<Operation *>> returningMatchAndRewrite(
      LinalgExt::InParallelOp inParallelOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(LinalgExt::InParallelOp inParallelOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(inParallelOp, rewriter);
  }
};

FailureOr<SmallVector<Operation *>>
BridgeInParallelToFlowAbstractionGap::returningMatchAndRewrite(
    LinalgExt::InParallelOp inParallelOp, PatternRewriter &rewriter) const {
  if (inParallelOp->getNumResults() == 0) return SmallVector<Operation *>();

  // If any result is not a single use tensors being tied back into a HAL
  // subspan op, bail. This is not considered a failure.
  // TODO: allow all or nothing, otherwise failure.
  if (llvm::any_of(inParallelOp->getResults(), [](Value result) {
        if (!result.hasOneUse()) return true;
        auto storeOp =
            dyn_cast<Flow::DispatchTensorStoreOp>(*(result.getUsers().begin()));
        return !storeOp ||
               !storeOp.target()
                    .getDefiningOp<HAL::InterfaceBindingSubspanOp>();
      }))
    return SmallVector<Operation *>();

  for (OpResult result : inParallelOp->getResults()) {
    auto tensorStoreOp =
        cast<Flow::DispatchTensorStoreOp>(*(result.getUsers().begin()));
    auto subSpanOp =
        tensorStoreOp.target().getDefiningOp<HAL::InterfaceBindingSubspanOp>();
    LinalgExt::ParallelInsertSliceOp yieldingOp =
        inParallelOp.getTerminator().yieldingOps()[result.getResultNumber()];
    // Replace OpOperand by a clone of the HAL binding subspan op and forward
    // the uses of flow.dispatch.tensor/load.
    // TODO: check the load is of the full tensor, OTOH this whole process is
    // a hack that needs to go away ...
    SmallVector<Flow::DispatchTensorLoadOp> loadsFromSubspan;
    for (Operation *op : subSpanOp.result().getUsers()) {
      auto user = dyn_cast<Flow::DispatchTensorLoadOp>(op);
      if (user) loadsFromSubspan.push_back(user);
    }
    rewriter.setInsertionPoint(subSpanOp);
    // clang-format off
    auto newSubspanOp =
        rewriter.create<HAL::InterfaceBindingSubspanOp>(
            inParallelOp->getLoc(),
            yieldingOp.yieldedType(),
            subSpanOp.set(),
            subSpanOp.binding(),
            subSpanOp.type(),
            subSpanOp.byte_offset(),
            subSpanOp.dynamic_dims(),
            subSpanOp.alignment() ?
              rewriter.getIndexAttr(subSpanOp.alignment()->getZExtValue()) :
              IntegerAttr());
    rewriter.replaceOpWithIf(subSpanOp, newSubspanOp.result(), [](OpOperand &operand){
      return !isa<Flow::DispatchTensorLoadOp>(operand.getOwner());
    });
    // for (auto loadOp : loadsFromSubspan)
    //   rewriter.replaceOp(loadOp, newSubspanOp.result());
    // clang-format on

    yieldingOp.destMutable().assign(newSubspanOp.getResult());
    rewriter.eraseOp(tensorStoreOp);
  }

  // All results are assumed tied back into HAL. This serves purely as a DCE
  // avoidance mechanism.
  rewriter.setInsertionPoint(inParallelOp->getBlock()->getTerminator());
  rewriter.create<LinalgExt::AssumeTiedToHALOp>(inParallelOp->getLoc(),
                                                inParallelOp.getResults());

  return SmallVector<Operation *>{};
}

//===---------------------------------------------------------------------===//
// Default allocation functions for CPU backend
// TODO: register the bufferization behavior in a target-specific way.
//===---------------------------------------------------------------------===//

// Default allocation function to use with IREEs bufferization.
static Value cpuAllocationFunction(OpBuilder &builder, Location loc,
                                   ArrayRef<int64_t> staticShape,
                                   Type elementType,
                                   ArrayRef<Value> dynamicSizes) {
  MemRefType allocType = MemRefType::get(staticShape, elementType);
  return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes);
}

// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> cpuComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  return builder
      .create<memref::AllocaOp>(loc, memRefType, dynamicSizes,
                                builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult cpuComprehensiveBufferizeDeallocationFn(OpBuilder &builder,
                                                             Location loc,
                                                             Value allocation) {
  return success();
}

static LogicalResult cpuComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

//===---------------------------------------------------------------------===//
// IREE-specific transformations defined outside of iree_linalg_transform.
//===---------------------------------------------------------------------===//

// Note: with the recent TypeID changes, hiding these classes inside an
// anonymous namespace would require specific `MLIR_DECLARE_EXPLICIT_TYPE_ID`
// for each class.

// namespace {

// TODO: Move to tablegen. Until this stabilizes upstream, simple C++ is
// enough.
class IREEBufferizeOp
    : public Op<IREEBufferizeOp, transform::TransformOpInterface::Trait,
                MemoryEffectOpInterface::Trait> {
 public:
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral("transform.iree.bufferize");
  }

  Value target() { return nullptr; }

  LogicalResult apply(transform::TransformResults &results,
                      transform::TransformState &state) {
    PassManager pm(getContext());
    // Bufferize the dispatch.
    using mlir::bufferization::BufferizationOptions;
    BufferizationOptions::AllocationFn allocationFn =
        cpuComprehensiveBufferizeAllocationFn;
    BufferizationOptions::DeallocationFn deallocationFn =
        cpuComprehensiveBufferizeDeallocationFn;
    BufferizationOptions::MemCpyFn memcpyFn = cpuComprehensiveBufferizeCopyFn;
    mlir::iree_compiler::addIREEComprehensiveBufferizePasses(
        pm, allocationFn, deallocationFn, memcpyFn);
    WalkResult res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
      if (failed(pm.run(moduleOp))) {
        getOperation()->emitError()
            << "failed to bufferize ModuleOp:\n"
            << *(moduleOp.getOperation()) << "\nunder top-level:\n"
            << *state.getTopLevel();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return failure(res.wasInterrupted());
  }

  // let assemblyFormat = "attr-dict";
  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    return parser.parseOptionalAttrDict(state.attributes);
  }

  // let assemblyFormat = "attr-dict";
  void print(OpAsmPrinter &printer) {
    printer.printOptionalAttrDict((*this)->getAttrs());
  }

  // This transform may affect the entirety of the payload IR.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         transform::PayloadIRResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         transform::PayloadIRResource::get());
  }
};

// TODO: Move to tablegen. Until this stabilizes upstream, simple C++ is
// enough.
class IREESetNumWorkgroupToOneOp
    : public Op<IREESetNumWorkgroupToOneOp,
                transform::TransformOpInterface::Trait,
                MemoryEffectOpInterface::Trait> {
 public:
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral("transform.iree.set_num_workgroups_to_one");
  }

  Value target() { return nullptr; }

  LogicalResult apply(transform::TransformResults &results,
                      transform::TransformState &state) {
    auto variantOp = dyn_cast<HAL::ExecutableVariantOp>(state.getTopLevel());
    if (!variantOp) {
      return getOperation()->emitError()
             << "top-level op is not a HAL::ExecutableVariantOp: "
             << *state.getTopLevel();
    }
    return iree_compiler::setNumWorkgroupsImpl(variantOp, {});
  }

  // let assemblyFormat = "attr-dict";
  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    return parser.parseOptionalAttrDict(state.attributes);
  }

  // let assemblyFormat = "attr-dict";
  void print(OpAsmPrinter &printer) {
    printer.printOptionalAttrDict((*this)->getAttrs());
  }

  // This transform may affect the entirety of the payload IR.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         transform::PayloadIRResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         transform::PayloadIRResource::get());
  }
};

// TODO: Move to tablegen. Until this stabilizes upstream, simple C++ is
// enough.
class IREELinalgExtInParallelToFlowOp
    : public Op<IREELinalgExtInParallelToFlowOp,
                transform::TransformOpInterface::Trait,
                MemoryEffectOpInterface::Trait> {
 public:
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral("transform.iree.inparallel_to_flow");
  }

  Value target() { return nullptr; }

  LogicalResult apply(transform::TransformResults &results,
                      transform::TransformState &state) {
    if (state.getTopLevel()
            ->walk<WalkOrder::PostOrder>([&](LinalgExt::InParallelOp op) {
              if (failed(functional::applyReturningPatternAt(
                      InParallelOpToOperandlessFlowRewriter(getContext()), op)))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted())
      return failure();
    return success();
  }

  // let assemblyFormat = "attr-dict";
  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    if (parser.parseOptionalAttrDict(state.attributes)) return failure();
    return success();
  }

  // let assemblyFormat = "attr-dict";
  void print(OpAsmPrinter &printer) {
    printer.printOptionalAttrDict((*this)->getAttrs());
  }

  // This transform may affect the entirety of the payload IR.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         transform::PayloadIRResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         transform::PayloadIRResource::get());
  }
};

// TODO: Move to tablegen. Until this stabilizes upstream, simple C++ is
// enough.
class IREELinalgExtInParallelToHALOp
    : public Op<IREELinalgExtInParallelToHALOp,
                transform::TransformOpInterface::Trait,
                MemoryEffectOpInterface::Trait> {
 public:
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral("transform.iree.inparallel_to_hal");
  }

  Value target() { return nullptr; }

  LogicalResult apply(transform::TransformResults &results,
                      transform::TransformState &state) {
    // Start by bridging the InParallel/Flow/HAL abstraction gap on tensors.
    SmallVector<LinalgExt::InParallelOp> ops;
    state.getTopLevel()->walk(
        [&](LinalgExt::InParallelOp op) { ops.push_back(op); });
    for (LinalgExt::InParallelOp op : ops) {
      if (failed(functional::applyReturningPatternAt(
              BridgeInParallelToFlowAbstractionGap(getContext()), op)))
        return failure();
    }

    if (state.getTopLevel()
            ->walk<WalkOrder::PostOrder>([&](LinalgExt::InParallelOp op) {
              if (failed(functional::applyReturningPatternAt(
                      InParallelOpToHALRewriter(getContext()), op)))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted())
      return failure();

    // Apply post-distribution canonicalization passes.
    // TODO: these should be done like other transform dialect
    // canonicalizations that preserve IR handles.
    RewritePatternSet canonicalization(getContext());
    AffineApplyOp::getCanonicalizationPatterns(canonicalization, getContext());
    AffineMinOp::getCanonicalizationPatterns(canonicalization, getContext());
    iree_compiler::populateAffineMinSCFCanonicalizationPattern(
        canonicalization);
    // TODO: Careful. forwarding to Flow is more than a simple
    // canonicalization. We went through rewrite of Flow/HAL out of the
    // flow.tensor type to avoid these issues and allow various orderings of
    // parallelism, tensors, distribution and bufferization.
    Flow::populateFlowDispatchCanonicalizationPatterns(canonicalization,
                                                       getContext());
    return applyPatternsAndFoldGreedily(state.getTopLevel(),
                                        std::move(canonicalization));
  }

  // let assemblyFormat = "attr-dict";
  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    if (parser.parseOptionalAttrDict(state.attributes)) return failure();
    return success();
  }

  // let assemblyFormat = "attr-dict";
  void print(OpAsmPrinter &printer) {
    printer.printOptionalAttrDict((*this)->getAttrs());
  }

  // This transform may affect the entirety of the payload IR.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         transform::PayloadIRResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         transform::PayloadIRResource::get());
  }
};

/// Test extension of the Transform dialect. Registers additional ops and
/// declares PDL as dependent dialect since the additional ops are using PDL
/// types for operands and results.
class LinalgTransformDialectExtension
    : public mlir::transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
 public:
  LinalgTransformDialectExtension() {
    declareDependentDialect<pdl::PDLDialect>();
    // clang-format off
    registerTransformOps<IREEBufferizeOp, 
                         IREELinalgExtInParallelToFlowOp,
                         IREELinalgExtInParallelToHALOp,
                         IREESetNumWorkgroupToOneOp
                         >();
    // clang-format on
    // TODO: hook up to Tablegen.
    //     registerTransformOps<
    // #define GET_OP_LIST
    // #include "LinalgTransformDialectExtension.cpp.inc"
    //         >();
  }
};

// } // namespace anonymous

void mlir::iree_compiler::registerLinalgTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}
