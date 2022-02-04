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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

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
  auto *entryBlock = dispatchOp.getBody();

  // Map of SSA values on the outside of the op to arguments on the inside.
  // This lets us avoid capturing duplicate values - they'd be cleaned up
  // eventually during canonicalization but it's messy.
  DenseMap<Value, Value> outerToInnerMap;
  unsigned argIdx = 0;
  for (auto operand : dispatchOp.operands()) {
    auto arg = entryBlock->getArgument(argIdx++);
    outerToInnerMap[operand] = arg;
  }
  for (auto result : dispatchOp.results()) {
    if (dispatchOp.getTiedResultOperand(result)) continue;  // ignored tied
    auto arg = entryBlock->getArgument(argIdx++);
    outerToInnerMap[result] = arg;
  }

  // Captures (or reuses) dynamic dimensions for the given external->internal
  // SSA value pair.
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
  auto captureTensorDims = [&](Value externalValue, Value internalValue) {
    auto tensorType =
        internalValue.getType().dyn_cast<IREE::Flow::DispatchTensorType>();
    if (!tensorType) return;
    if (tensorType.hasStaticShape()) return;

    // Find the dimensions in the parent.
    auto dynamicDims = IREE::Util::findDynamicDims(
        externalValue, dispatchOp->getBlock(), Block::iterator(dispatchOp));
    if (!dynamicDims.hasValue()) return;

    // Capture the dynamic dimensions as args in the region.
    SmallVector<Value> capturedDims;
    for (auto dynamicDim : *dynamicDims) {
      auto existing = outerToInnerMap.find(dynamicDim);
      if (existing != outerToInnerMap.end()) {
        // Already captured the dimension; reuse.
        capturedDims.push_back(existing->second);
      } else {
        // Capture the dimension.
        auto arg =
            entryBlock->addArgument(dynamicDim.getType(), dynamicDim.getLoc());
        dispatchOp.operandsMutable().append(dynamicDim);
        capturedDims.push_back(arg);
        outerToInnerMap[dynamicDim] = arg;
      }
    }

    // Insert a shape tie op into the region to associate the dims.
    auto tieOp = entryBuilder.create<IREE::Flow::DispatchTieShapeOp>(
        internalValue.getLoc(), tensorType, internalValue, capturedDims);
    internalValue.replaceAllUsesExcept(tieOp.result(), tieOp);
  };

  // Capture all required dimensions and add tie_shape ops.
  for (auto operand : llvm::to_vector<4>(dispatchOp.operands())) {
    captureTensorDims(operand, outerToInnerMap[operand]);
  }
  for (auto result : dispatchOp.results()) {
    if (dispatchOp.getTiedResultOperand(result)) continue;  // ignore tied
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

}  // namespace

std::unique_ptr<Pass> createCaptureDispatchDynamicDimsPass() {
  return std::make_unique<CaptureDispatchDynamicDimsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
