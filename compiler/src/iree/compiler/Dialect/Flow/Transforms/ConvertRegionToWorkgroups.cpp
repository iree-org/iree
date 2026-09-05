// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler::IREE::Flow {

namespace {

/// Compute the dynamic dims of the given value and add them to the vector.
static void appendDynamicDims(OpBuilder &b, Location loc,
                              SmallVector<Value> &argumentDims, Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());

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
    if (ShapedType::isStatic(dim.value())) {
      continue;
    }
    argumentDims.push_back(
        b.createOrFold<tensor::DimOp>(loc, tensor, dim.index()));
  }
}

/// Follow the reverse SSA use-def chain of the given value (always taking the
/// tied operand) and return the first value outside of `regionOp`.
static std::optional<Value>
findFirstTiedValueOutsideOfRegionOp(IREE::Flow::DispatchRegionOp regionOp,
                                    Value value) {
  // Check if `v` is defined outside of `regionOp`.
  auto isOutside = [&](Value v) {
    if (isa<OpResult>(v)) {
      return !regionOp->isAncestor(v.getDefiningOp());
    }
    assert(isa<BlockArgument>(v) && "expected bbArg");
    // DispatchRegionOp does not have block arguments.
    return true;
  };

  while (!isOutside(value)) {
    auto tiedOpInterface = value.getDefiningOp<IREE::Util::TiedOpInterface>();
    if (!tiedOpInterface) {
      // Reached an op that does not implement the interface.
      return std::nullopt;
    }
    value = tiedOpInterface.getTiedResultOperand(value);
    if (!value) {
      // Nothing is tied here.
      return std::nullopt;
    }
  }

  return value;
}

// These ops become tied Flow views after dispatch formation. Treat their
// results as indirect now so a required tie cannot mutate an earlier source.
static bool isPreConversionTensorView(Operation *op) {
  return op && isa<tensor::BitcastOp, tensor::CastOp, tensor::CollapseShapeOp,
                   tensor::ExpandShapeOp, tensor::ReshapeOp>(op);
}

// A required tie can reuse storage only through a direct, type-identical base.
static Value getRequiredDirectTiedResultBase(DispatchRegionOp regionOp,
                                             Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result || !regionOp->isProperAncestor(result.getOwner())) {
    return {};
  }
  auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(result.getOwner());
  if (!tiedOp || !tiedOp.isTiedResultRequired(result.getResultNumber())) {
    return {};
  }
  Value tiedOperand = tiedOp.getTiedResultOperand(result);
  if (!tiedOperand || isPreConversionTensorView(tiedOperand.getDefiningOp()) ||
      tiedOperand.getType() != result.getType() ||
      tiedOperand != tiedOp.getTiedResult(result.getResultNumber())) {
    return {};
  }
  return tiedOperand;
}

} // namespace

/// Rewrite the DispatchRegionOp into a DispatchWorkgroupsOp. The
/// DispatchRegionOp is not isolated from above and may capture any SSA value
/// that is in scope. The generated DispatchWorkgroupsOp captures all SSA values
/// explicitly and makes them available inside the region via block arguments.
FailureOr<IREE::Flow::DispatchWorkgroupsOp>
rewriteFlowDispatchRegionToFlowDispatchWorkgroups(
    IREE::Flow::DispatchRegionOp regionOp, RewriterBase &rewriter) {
  Region &region = regionOp.getBody();
  // Currently this does not handle empty `flow.dispatch.region` ops.
  if (region.empty()) {
    return rewriter.notifyMatchFailure(regionOp,
                                       "unhandled op with empty region");
  }
  unsigned numResults = regionOp->getNumResults();

  // Prepare rewriter.
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = regionOp.getLoc();
  rewriter.setInsertionPoint(regionOp);

  // Compute arguments of the dispatch region.
  llvm::SetVector<Value> argumentsSet;
  mlir::getUsedValuesDefinedAbove(region, argumentsSet);
  // Unranked tensors are not supported.
  assert(llvm::none_of(
             argumentsSet,
             [](Value v) { return isa<UnrankedTensorType>(v.getType()); }) &&
         "unranked tensors are not supported");

  // Compute dimensions of tensor args.
  SmallVector<Value> argumentDims;
  for (Value tensor : argumentsSet) {
    auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
    if (!tensorType) {
      continue;
    }
    appendDynamicDims(rewriter, loc, argumentDims, tensor);
  }

  // Find tied results.
  SmallVector<Value> requiredResults;
  DenseSet<Value> tiedArgumentsSet;
  SmallVector<int64_t> tiedArguments(numResults,
                                     IREE::Util::TiedOpInterface::kUntiedIndex);
  SmallVector<IREE::Flow::ReturnOp> origTerminators;
  region.walk([&](IREE::Flow::ReturnOp returnOp) {
    origTerminators.push_back(returnOp);
  });
  assert(!origTerminators.empty() && "expected at least one terminator");

  // The logic to find the tied arguments only works for single block regions.
  // For ops with multiple blocks, just ignore tied arguments for now.
  if (llvm::hasSingleElement(region)) {
    SmallVector<std::pair<unsigned, Value>> requiredTies;
    SmallVector<std::pair<unsigned, Value>> otherTies;
    for (const auto &it :
         llvm::enumerate(origTerminators.front()->getOperands())) {
      auto tiedArgument =
          findFirstTiedValueOutsideOfRegionOp(regionOp, it.value());
      if (!tiedArgument.has_value()) {
        continue;
      }
      assert(argumentsSet.contains(*tiedArgument) &&
             "expected that tiedArgument is already an argument");
      bool isRequired = getRequiredDirectTiedResultBase(regionOp, it.value()) ==
                        *tiedArgument;
      auto &candidates = isRequired ? requiredTies : otherTies;
      candidates.emplace_back(it.index(), *tiedArgument);
    }
    auto assignTies = [&](ArrayRef<std::pair<unsigned, Value>> candidates) {
      for (auto [resultIndex, tiedArgument] : candidates) {
        // Do not tie an argument to multiple results.
        if (!tiedArgumentsSet.insert(tiedArgument).second) {
          continue;
        }
        tiedArguments[resultIndex] = std::distance(
            argumentsSet.begin(), llvm::find(argumentsSet, tiedArgument));
      }
    };
    // Required bindings claim shared storage before optional result ties.
    assignTies(requiredTies);

    // An operation-required result may have no users outside the dispatch, but
    // still needs initialized writable storage. Materialize its binding here,
    // where capture access and output stores are established, so region
    // formation and canonicalization need only preserve ordinary SSA uses.
    auto returnOp = origTerminators.front();
    // A base with multiple data consumers is unavailable to every candidate.
    DenseSet<Value> checkedBases;
    for (Operation &op : region.front().without_terminator()) {
      if (!isa<IREE::Util::TiedOpInterface>(op)) {
        continue;
      }
      bool hasLiveResult = llvm::any_of(op.getResults(), [&](Value result) {
        return llvm::any_of(result.getUses(), [&](OpOperand &use) {
          return use.getOwner() != returnOp ||
                 !regionOp->getResult(use.getOperandNumber()).use_empty();
        });
      });
      if (!hasLiveResult) {
        continue;
      }
      for (Value result : op.getResults()) {
        Value tiedBase = getRequiredDirectTiedResultBase(regionOp, result);
        if (!tiedBase || !argumentsSet.contains(tiedBase) ||
            tiedArgumentsSet.contains(tiedBase) ||
            !checkedBases.insert(tiedBase).second) {
          continue;
        }
        // Limit reuse to one data consumer instead of requiring alias/liveness
        // analysis. Shape queries cannot observe a type-identical tie's writes.
        if (!llvm::all_of(tiedBase.getUses(), [&](OpOperand &use) {
              return use.getOwner() == &op ||
                     isa<tensor::DimOp>(use.getOwner());
            })) {
          continue;
        }
        tiedArgumentsSet.insert(tiedBase);
        tiedArguments.push_back(std::distance(
            argumentsSet.begin(), llvm::find(argumentsSet, tiedBase)));
        requiredResults.push_back(result);
      }
    }
    assignTies(otherTies);
  }

  // Create empty dispatch region.
  auto arguments = llvm::to_vector_of<Value>(argumentsSet);
  arguments.append(argumentDims);
  for (unsigned i = 0; i < numResults; ++i) {
    // Tied arguments already have their dynamic result dims in `arguments`. Do
    // not add them again.
    if (tiedArguments[i] == IREE::Util::TiedOpInterface::kUntiedIndex) {
      ValueRange dims = regionOp.getResultDynamicDims(i);
      arguments.append(dims.begin(), dims.end());
    }
  }

  // Required results are appended so existing dispatch result indices remain
  // unchanged. Direct type-identical ties reuse the captured storage's shape.
  SmallVector<Type> resultTypes(regionOp.getResultTypes());
  SmallVector<Value> resultDims(regionOp.getResultDims());
  for (auto [index, result] : llvm::enumerate(requiredResults)) {
    resultTypes.push_back(result.getType());
    auto dims = IREE::Util::findDynamicDimsInList(
        tiedArguments[numResults + index], arguments, argumentDims);
    resultDims.append(dims.begin(), dims.end());
  }

  // Create the shell dispatch.workgroup ops.
  auto workgroupsOp = IREE::Flow::DispatchWorkgroupsOp::create(
      rewriter, loc, regionOp.getWorkload(), resultTypes, resultDims, arguments,
      argumentDims, tiedArguments);
  workgroupsOp->setDialectAttrs(regionOp->getDialectAttrs());

  // Populate the workgroup count region.
  if (!regionOp.getWorkgroupCount().empty()) {
    // Move DispatchRegion's workload_count region to DispatchWorkgroupOp's
    rewriter.inlineRegionBefore(regionOp.getWorkgroupCount(),
                                workgroupsOp.getWorkgroupCount(),
                                workgroupsOp.getWorkgroupCount().begin());
    mlir::makeRegionIsolatedFromAbove(rewriter,
                                      workgroupsOp.getWorkgroupCount(),
                                      llvm::IsaPred<arith::ConstantOp>);
  }

  IRMapping bvm;
  bvm.map(arguments, workgroupsOp.getInputBlockArguments());

  // Create DispatchTensorLoadOp for all tensor arguments.
  Region &newBody = workgroupsOp.getWorkgroupBody();
  assert(llvm::hasSingleElement(newBody) &&
         "expected `flow.dispatch.workgroup` op to be created with a single "
         "block");

  Block *newBodyEntry = &newBody.front();
  rewriter.setInsertionPointToStart(newBodyEntry);
  SmallVector<Value> argValues;
  for (const auto &it : llvm::enumerate(arguments)) {
    auto tensorType = dyn_cast<RankedTensorType>(it.value().getType());
    if (!tensorType) {
      argValues.push_back(it.value());
      continue;
    }
    auto inputBbArg = workgroupsOp.getInputBlockArgument(it.index());
    auto dims =
        IREE::Util::findDynamicDimsInList(it.index(), arguments, argumentDims);
    assert(dims.size() == tensorType.getNumDynamicDims() &&
           "dynamic dims not found among arguments");
    SmallVector<Value> bbArgDims =
        llvm::map_to_vector(dims, [&](Value v) { return bvm.lookup(v); });
    Value loadedTensor = IREE::TensorExt::DispatchTensorLoadOp::create(
        rewriter, loc, tensorType, inputBbArg, bbArgDims);
    bvm.map(it.value(), loadedTensor);
    argValues.push_back(loadedTensor);
  }

  // Move regionOp body into the workgroupsOp.
  rewriter.inlineRegionBefore(region, newBody, newBody.end());
  // Merge the entry block of `newBody` with the original entry block from the
  // region.
  Block *origEntry = &(*(std::next(newBody.begin())));
  rewriter.mergeBlocks(origEntry, newBodyEntry);

  for (Value argument : arguments) {
    argument.replaceUsesWithIf(bvm.lookup(argument), [&](OpOperand &operand) {
      return workgroupsOp->isProperAncestor(operand.getOwner());
    });
  }

  // Update terminator.
  SmallVector<IREE::Flow::ReturnOp> terminators;
  newBody.walk(
      [&](IREE::Flow::ReturnOp returnOp) { terminators.push_back(returnOp); });
  for (auto terminator : terminators) {
    rewriter.setInsertionPoint(terminator);
    for (const auto &it : llvm::enumerate(
             llvm::concat<Value>(terminator.getOperands(), requiredResults))) {
      auto outputBbArg = workgroupsOp.getOutputBlockArgument(it.index());
      ValueRange dims;
      if (tiedArguments[it.index()] ==
          IREE::Util::TiedOpInterface::kUntiedIndex) {
        dims = regionOp.getResultDynamicDims(it.index());
      } else {
        // This assumes that the number of dynamic dims does not change when
        // following an SSA use-def chain of tied values.
        dims = IREE::Util::findDynamicDimsInList(tiedArguments[it.index()],
                                                 arguments, argumentDims);
      }
#ifndef NDEBUG
      auto tensorType = cast<RankedTensorType>(it.value().getType());
      assert(dims.size() == tensorType.getNumDynamicDims() &&
             "mismatching number of dynamic dims");
#endif // NDEBUG
      SmallVector<Value> bbArgDims =
          llvm::map_to_vector(dims, [&](Value v) { return bvm.lookup(v); });
      IREE::TensorExt::DispatchTensorStoreOp::create(rewriter, loc, it.value(),
                                                     outputBbArg, bbArgDims);
    }

    // Delete the old terminator and create a new one.
    IREE::Flow::ReturnOp::create(rewriter, loc);
    rewriter.eraseOp(terminator);
  }

  rewriter.replaceOp(regionOp,
                     workgroupsOp.getResults().take_front(numResults));
  return workgroupsOp;
}

} // namespace mlir::iree_compiler::IREE::Flow
