// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CAPTUREDYNAMICDIMSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

// TODO(benvanik): rework flow.dispatch.workgroups to hold shape dimension
// mappings for the region instead of needing this pass and the tie ops.

// Captures dynamic dimensions of !flow.dispatch.tensor operands.
// Tries to deduplicate with any that may already be captured by construction.
//
// Thanks to all dimensions being captured by the flow.dispatch.workgroups op
// we don't need to insert any shape queries on the outside. Technically in many
// cases we could avoid the need to insert the ties on the inside too but we
// leave the cleanup of redundant work to further optimization passes to keep
// this simple.
static void captureDims(IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  Region &body = dispatchOp.getWorkgroupBody();
  if (body.empty()) {
    return;
  }
  auto *entryBlock = &body.front();

  // Map of SSA values on the outside of the op to arguments on the inside.
  // This lets us avoid capturing duplicate values - they'd be cleaned up
  // eventually during canonicalization but it's messy.
  DenseMap<Value, Value> outerToInnerMap;
  unsigned argIdx = 0;
  for (auto operand : dispatchOp.getArguments()) {
    auto arg = entryBlock->getArgument(argIdx++);
    outerToInnerMap[operand] = arg;
  }
  for (auto result : dispatchOp.getResults()) {
    if (dispatchOp.getTiedResultOperand(result))
      continue; // ignored tied
    auto arg = entryBlock->getArgument(argIdx++);
    outerToInnerMap[result] = arg;
  }

  // Captures (or reuses) dynamic dimensions for the given external->internal
  // SSA value pair.
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
  auto captureTensorDims = [&](Value externalValue, Value internalValue) {
    auto tensorType =
        llvm::dyn_cast<IREE::Flow::DispatchTensorType>(internalValue.getType());
    if (!tensorType)
      return;
    if (tensorType.hasStaticShape())
      return;

    // Find the dimensions in the parent.
    auto maybeDynamicDims = IREE::Util::findDynamicDims(
        externalValue, dispatchOp->getBlock(), Block::iterator(dispatchOp));
    if (!maybeDynamicDims.has_value())
      return;
    // Convert to a vector -- we cannot use the ValueRange directly because
    // it might point into the operand list of this op, which we might mutate
    // in-place.
    auto dynamicDims = llvm::to_vector(maybeDynamicDims.value());

    // Find the insertion position. All extra arguments need to be added before
    // "writeonly" tensors corresponding to the result.
    unsigned insertionPosition = entryBlock->getNumArguments();
    for (auto argType : llvm::reverse(entryBlock->getArgumentTypes())) {
      auto flowTensorType =
          llvm::dyn_cast<IREE::Flow::DispatchTensorType>(argType);
      if (!flowTensorType ||
          flowTensorType.getAccess() != IREE::Flow::TensorAccess::WriteOnly) {
        break;
      }
      insertionPosition--;
    }

    // Capture the dynamic dimensions as args in the region.
    SmallVector<Value> capturedDims;
    for (auto dynamicDim : dynamicDims) {
      auto existing = outerToInnerMap.find(dynamicDim);
      if (existing != outerToInnerMap.end()) {
        // Already captured the dimension; reuse.
        capturedDims.push_back(existing->second);
      } else {
        // Capture the dimension.
        auto arg = entryBlock->insertArgument(
            insertionPosition++, dynamicDim.getType(), dynamicDim.getLoc());
        dispatchOp.getArgumentsMutable().append(dynamicDim);
        capturedDims.push_back(arg);
        outerToInnerMap[dynamicDim] = arg;
      }
    }

    // Insert a shape tie op into the region to associate the dims.
    auto tieOp = entryBuilder.create<IREE::Flow::DispatchTieShapeOp>(
        internalValue.getLoc(), tensorType, internalValue, capturedDims);
    internalValue.replaceAllUsesExcept(tieOp.getResult(), tieOp);
  };

  // Capture all required dimensions and add tie_shape ops.
  for (auto operand : llvm::to_vector(dispatchOp.getArguments())) {
    captureTensorDims(operand, outerToInnerMap[operand]);
  }
  for (auto result : dispatchOp.getResults()) {
    if (dispatchOp.getTiedResultOperand(result))
      continue; // ignore tied
    captureTensorDims(result, outerToInnerMap[result]);
  }
}

// Track dynamic dimensions of tensor type loop variables and results by adding
// the values corresponding to their dimensions as loop variables. Furthermore,
// the dynamic dimensions of results are also returned and any result tensors
// tied using 'flow.tensor.tie_shape'.
static void captureDims(scf::ForOp forOp) {
  struct Iterable {
    Value init;
    Value yieldOperand;
    Value bbArg;
    OpResult result;
  };

  // Collect any dynamic tensors with their corresponding block argument,
  // initializer, yield operand, and result.
  SmallVector<Iterable> dynamicTensorIterables;
  for (auto &&[init, iter, bbArg, result] :
       llvm::zip_equal(forOp.getInitArgs(), forOp.getYieldedValues(),
                       forOp.getRegionIterArgs(), forOp.getResults())) {
    auto tensorType = dyn_cast<TensorType>(init.getType());
    if (!tensorType || tensorType.hasStaticShape())
      continue;

    // Make the transform idempotent by not caring about tensors only used
    // within 'flow.tensor.tie_shape' operations.
    if (llvm::all_of(bbArg.getUsers(), llvm::IsaPred<Flow::TensorTieShapeOp>))
      continue;

    dynamicTensorIterables.push_back({init, iter, bbArg, result});
  }

  if (dynamicTensorIterables.empty())
    return;

  // Create the new dimension loop variables. Since the dynamic tensors may be
  // of different types with varying number of dynamic dimensions, 'dimBounds'
  // is used to track how many and which dimension values in 'newIterables'
  // correspond to 'dynamicTensorIterables'.
  // More specifically, the value of 'dimBounds[i]' is the first loop variable
  // modeling the dynamic dimension of 'dynamicTensorIterables[i]' and
  // value of 'dimBounds[i+1]' the index of the first loop variable that is no
  // longer a dynamic dimension of 'dynamicTensorIterables[i]'.
  SmallVector<unsigned> dimBounds;
  SmallVector<Iterable> newIterables;
  for (auto [init, iter, bbArg, result] : dynamicTensorIterables) {
    dimBounds.push_back(newIterables.size());
    std::optional<ValueRange> initDynamicDims = IREE::Util::findDynamicDims(
        init, forOp->getBlock(), Block::iterator(forOp));
    if (!initDynamicDims)
      continue;

    std::optional<ValueRange> iterDynamicDims = IREE::Util::findDynamicDims(
        iter, forOp.getBody(),
        Block::iterator(forOp.getBody()->getTerminator()));
    if (!iterDynamicDims)
      continue;

    if (iterDynamicDims->size() != initDynamicDims->size())
      continue;

    for (auto [initDim, iterDim] :
         llvm::zip_equal(*initDynamicDims, *iterDynamicDims))
      newIterables.push_back({initDim, iterDim});
  }
  dimBounds.push_back(newIterables.size());

  if (newIterables.empty())
    return;

  // A new 'scf.for' has to be created to replace the old one as new results
  // are being added.
  SmallVector<Value> newInits = llvm::to_vector(forOp.getInitArgs());
  llvm::append_range(
      newInits, llvm::map_range(newIterables, std::mem_fn(&Iterable::init)));
  OpBuilder builder(forOp);
  auto newForOp = builder.create<scf::ForOp>(
      forOp->getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInits);
  newForOp.getRegion().takeBody(forOp.getRegion());

  // Adjust the loop body taken from the old 'scf.for' to account for the new
  // loop variables. This adds new block arguments to the entry block, new
  // operands to the 'scf.yield' and links the 'OpResult' to our 'Iterable'
  // struct.
  MutableOperandRange mutableOperandRange =
      cast<scf::YieldOp>(newForOp.getBody()->getTerminator())
          .getResultsMutable();
  for (auto &&[init, iter, bbArg, result] : newIterables) {
    bbArg = newForOp.getRegion().addArgument(init.getType(), forOp.getLoc());
    result = newForOp->getResult(mutableOperandRange.size());
    mutableOperandRange.append(iter);
  }

  // Make the dimension of every dynamic tensor loop variable known by creating
  // 'flow.tensor.tie_shape' ops at the beginning of the entry block, binding
  // the dimension loop variables to the tensor. All other uses of the tensor
  // within the body are replaced.
  builder.setInsertionPointToStart(newForOp.getBody());
  for (auto [index, tensor] : llvm::enumerate(dynamicTensorIterables)) {
    auto dims =
        ArrayRef(newIterables)
            .slice(dimBounds[index], dimBounds[index + 1] - dimBounds[index]);
    if (dims.empty())
      continue;

    Value tied = builder.create<Flow::TensorTieShapeOp>(
        forOp.getLoc(), tensor.bbArg,
        llvm::map_to_vector(dims, std::mem_fn(&Iterable::bbArg)));
    tensor.bbArg.replaceAllUsesExcept(tied,
                                      /*exceptedUser=*/tied.getDefiningOp());
  }

  // Create new 'flow.tensor.tie_shape' after the new 'scf.for' loop, tying the
  // result variables of the dimensions to the resulting tensor.
  builder.setInsertionPointAfter(forOp);
  SmallVector<Value> results = llvm::to_vector_of<Value>(
      newForOp.getResults().take_front(forOp.getNumResults()));
  for (auto [index, tensor] : llvm::enumerate(dynamicTensorIterables)) {
    auto dims =
        ArrayRef(newIterables)
            .slice(dimBounds[index], dimBounds[index + 1] - dimBounds[index]);
    if (dims.empty())
      continue;

    Value &replacement = results[tensor.result.getResultNumber()];
    replacement = builder.create<Flow::TensorTieShapeOp>(
        forOp.getLoc(), replacement,
        llvm::to_vector_of<Value>(
            llvm::map_range(dims, std::mem_fn(&Iterable::result))));
  }

  forOp.replaceAllUsesWith(results);
  forOp->erase();
}

struct CaptureDynamicDimsPass
    : public IREE::Flow::impl::CaptureDynamicDimsPassBase<
          CaptureDynamicDimsPass> {
  void runOnOperation() override {
    getOperation()->walk([&](scf::ForOp forOp) { captureDims(forOp); });
    getOperation()->walk([&](IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
      captureDims(dispatchOp);
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
