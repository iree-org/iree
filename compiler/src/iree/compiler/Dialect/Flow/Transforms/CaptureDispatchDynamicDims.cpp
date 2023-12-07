// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

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

class CaptureDispatchDynamicDimsPass
    : public CaptureDispatchDynamicDimsBase<CaptureDispatchDynamicDimsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect>();
  }

  void runOnOperation() override {
    getOperation()->walk([&](IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
      captureDims(dispatchOp);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createCaptureDispatchDynamicDimsPass() {
  return std::make_unique<CaptureDispatchDynamicDimsPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
