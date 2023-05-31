// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

/// Compute the dynamic dims of the given value and add them to the vector.
static void appendDynamicDims(OpBuilder &b, Location loc,
                              SmallVector<Value> &argumentDims, Value tensor) {
  auto tensorType = llvm::cast<RankedTensorType>(tensor.getType());

  // Fast-path for if the value comes from ops that support our dynamic
  // shape interfaces. Otherwise we have to insert tensor.dim ops.
  auto availableDims = IREE::Util::findDynamicDims(tensor);
  if (availableDims.has_value()) {
    argumentDims.append(availableDims->begin(), availableDims->end());
    assert(tensorType.getNumDynamicDims() == availableDims->size() &&
           "not enough dynamic dims found");
    return;
  }

  for (auto dim : llvm::enumerate(tensorType.getShape())) {
    if (dim.value() != ShapedType::kDynamic) continue;
    argumentDims.push_back(
        b.createOrFold<tensor::DimOp>(loc, tensor, dim.index()));
  }
}

/// Follow the reverse SSA use-def chain of the given value (always taking the
/// tied operand) and return the first value outside of `regionOp`.
static std::optional<Value> findFirstTiedValueOutsideOfRegionOp(
    Flow::DispatchRegionOp regionOp, Value value) {
  // Check if `v` is defined outside of `regionOp`.
  auto isOutside = [&](Value v) {
    if (llvm::isa<OpResult>(v)) return !regionOp->isAncestor(v.getDefiningOp());
    assert(v.isa<BlockArgument>() && "expected bbArg");
    // DispatchRegionOp does not have block arguments.
    return true;
  };

  while (!isOutside(value)) {
    auto tiedOpInterface = value.getDefiningOp<IREE::Util::TiedOpInterface>();
    if (!tiedOpInterface)
      // Reached an op that does not implement the interface.
      return std::nullopt;
    value = tiedOpInterface.getTiedResultOperand(value);
    if (!value)
      // Nothing is tied here.
      return std::nullopt;
  }

  return value;
}

}  // namespace

/// Rewrite the DispatchRegionOp into a DispatchWorkgroupsOp. The
/// DispatchRegionOp is not isolated from above and may capture any SSA value
/// that is in scope. The generated DispatchWorkgroupsOp captures all SSA values
/// explicitly and makes them available inside the region via block arguments.
FailureOr<Flow::DispatchWorkgroupsOp>
rewriteFlowDispatchRegionToFlowDispatchWorkgroups(
    Flow::DispatchRegionOp regionOp, RewriterBase &rewriter) {
  // Only ops with a single block are supported.
  Region &region = regionOp.getBody();
  if (!region.hasOneBlock()) return failure();
  Block &body = region.front();
  auto terminator = cast<Flow::ReturnOp>(body.getTerminator());
  unsigned numResults = terminator->getNumOperands();

  // Prepare rewriter.
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = regionOp.getLoc();
  rewriter.setInsertionPoint(regionOp);

  // Compute arguments of the dispatch region.
  llvm::SetVector<Value> argumentsSet;
  mlir::getUsedValuesDefinedAbove(region, argumentsSet);
  // Unranked tensors are not supported.
  assert(!llvm::any_of(argumentsSet, [](Value v) {
    return v.getType().isa<UnrankedTensorType>();
  }) && "unranked tensors are not supported");

  // Compute dimensions of tensor args.
  SmallVector<Value> argumentDims;
  for (Value tensor : argumentsSet) {
    auto tensorType = llvm::dyn_cast<RankedTensorType>(tensor.getType());
    if (!tensorType) continue;
    appendDynamicDims(rewriter, loc, argumentDims, tensor);
  }

  // Find tied results.
  DenseSet<Value> tiedArgumentsSet;
  SmallVector<int64_t> tiedArguments(numResults,
                                     IREE::Util::TiedOpInterface::kUntiedIndex);
  for (const auto &it : llvm::enumerate(terminator->getOperands())) {
    auto tiedArgument =
        findFirstTiedValueOutsideOfRegionOp(regionOp, it.value());
    if (!tiedArgument.has_value()) continue;
    assert(argumentsSet.contains(*tiedArgument) &&
           "expected that tiedArgument is already an argument");
    // Do not tie an argument to multiple results.
    if (tiedArgumentsSet.contains(*tiedArgument)) continue;
    tiedArgumentsSet.insert(*tiedArgument);
    tiedArguments[it.index()] = std::distance(
        argumentsSet.begin(), llvm::find(argumentsSet, *tiedArgument));
  }

  // Create empty dispatch region.
  SmallVector<Value> arguments(argumentsSet.begin(), argumentsSet.end());
  arguments.append(argumentDims);
  for (unsigned i = 0; i < numResults; ++i) {
    // Tied arguments already have their dynamic result dims in `arguments`. Do
    // not add them again.
    if (tiedArguments[i] == IREE::Util::TiedOpInterface::kUntiedIndex) {
      ValueRange dims = regionOp.getResultDynamicDims(i);
      arguments.append(dims.begin(), dims.end());
    }
  }

  // Create the shell dispatch.workgroup ops.
  auto workgroupsOp = rewriter.create<IREE::Flow::DispatchWorkgroupsOp>(
      loc, regionOp.getWorkload(), regionOp.getResultTypes(),
      regionOp.getResultDims(), arguments, argumentDims, tiedArguments);
  workgroupsOp->setDialectAttrs(regionOp->getDialectAttrs());

  // Populate the workgroup count region.
  if (!regionOp.getWorkgroupCount().empty()) {
    // Move DispatchRegion's workload_count region to DispatchWorkgroupOp's
    rewriter.inlineRegionBefore(regionOp.getWorkgroupCount(),
                                workgroupsOp.getWorkgroupCount(),
                                workgroupsOp.getWorkgroupCount().begin());
    mlir::makeRegionIsolatedFromAbove(
        rewriter, workgroupsOp.getWorkgroupCount(),
        [](Operation *op) { return isa<arith::ConstantOp>(op); });
  }

  IRMapping bvm;
  bvm.map(arguments, workgroupsOp.getInputBlockArguments());

  // Create DispatchTensorLoadOp for all tensor arguments.
  assert(workgroupsOp.getWorkgroupBody().hasOneBlock() &&
         "expected one block after constructor");
  Block &newBody = workgroupsOp.getWorkgroupBody().getBlocks().front();
  assert(newBody.empty() && "expected empty block after constructor");
  rewriter.setInsertionPointToStart(&newBody);
  for (const auto &it : llvm::enumerate(arguments)) {
    auto tensorType = llvm::dyn_cast<RankedTensorType>(it.value().getType());
    if (!tensorType) continue;
    auto inputBbArg = workgroupsOp.getInputBlockArgument(it.index());
    auto dims =
        Util::findVariadicDynamicDims(it.index(), arguments, argumentDims);
    assert(dims.size() == tensorType.getNumDynamicDims() &&
           "dynamic dims not found among arguments");
    SmallVector<Value> bbArgDims = llvm::to_vector(
        llvm::map_range(dims, [&](Value v) { return bvm.lookup(v); }));
    Value loadedTensor = rewriter.create<IREE::Flow::DispatchTensorLoadOp>(
        loc, tensorType, inputBbArg, bbArgDims);
    bvm.map(it.value(), loadedTensor);
  }

  // Move regionOp body into the workgroupsOp.
  newBody.getOperations().splice(newBody.end(), body.getOperations());
  for (Value argument : arguments) {
    argument.replaceUsesWithIf(bvm.lookup(argument), [&](OpOperand &operand) {
      return workgroupsOp->isProperAncestor(operand.getOwner());
    });
  }

  // Update terminator.
  rewriter.setInsertionPoint(terminator);
  for (const auto &it : llvm::enumerate(terminator->getOperands())) {
    auto outputBbArg = workgroupsOp.getOutputBlockArgument(it.index());
    ValueRange dims;
    if (tiedArguments[it.index()] ==
        IREE::Util::TiedOpInterface::kUntiedIndex) {
      dims = regionOp.getResultDynamicDims(it.index());
    } else {
      // This assumes that the number of dynamic dims does not change when
      // following an SSA use-def chain of tied values.
      dims = Util::findVariadicDynamicDims(tiedArguments[it.index()], arguments,
                                           argumentDims);
    }
#ifndef NDEBUG
    auto tensorType = it.value().getType().cast<RankedTensorType>();
    assert(dims.size() == tensorType.getNumDynamicDims() &&
           "mismatching number of dynamic dims");
#endif  // NDEBUG
    SmallVector<Value> bbArgDims = llvm::to_vector(
        llvm::map_range(dims, [&](Value v) { return bvm.lookup(v); }));
    rewriter.create<IREE::Flow::DispatchTensorStoreOp>(loc, it.value(),
                                                       outputBbArg, bbArgDims);
  }

  // Delete the old terminator and create a new one.
  rewriter.create<IREE::Flow::ReturnOp>(loc);
  rewriter.eraseOp(terminator);

  rewriter.replaceOp(regionOp, workgroupsOp.getResults());
  return workgroupsOp;
}

namespace {
struct ConvertRegionToWorkgroupsPass
    : public ConvertRegionToWorkgroupsBase<ConvertRegionToWorkgroupsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Flow::FlowDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    SmallVector<Flow::DispatchRegionOp> ops;
    getOperation()->walk([&](Flow::DispatchRegionOp op) { ops.push_back(op); });

    IRRewriter rewriter(getOperation()->getContext());
    for (Flow::DispatchRegionOp regionOp : ops) {
      if (failed(rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp,
                                                                   rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createConvertRegionToWorkgroupsPass() {
  return std::make_unique<ConvertRegionToWorkgroupsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
