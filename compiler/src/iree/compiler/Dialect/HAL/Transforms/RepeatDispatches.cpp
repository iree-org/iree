// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_REPEATDISPATCHESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-repeat-dispatches
//===----------------------------------------------------------------------===//

struct RepeatDispatchesPass
    : public IREE::HAL::impl::RepeatDispatchesPassBase<RepeatDispatchesPass> {
  using IREE::HAL::impl::RepeatDispatchesPassBase<
      RepeatDispatchesPass>::RepeatDispatchesPassBase;
  void runOnOperation() override {
    // Collect all (nested) command buffer dispatch ops.
    SmallVector<IREE::HAL::CommandBufferDispatchOp> ops;
    getOperation()->walk(
        [&ops](IREE::HAL::CommandBufferDispatchOp op) { ops.push_back(op); });
    for (auto op : ops) {
      OpBuilder builder(op);
      for (unsigned i = 1; i < repeatCount; ++i) {
        // Clone the op at its original location in the IR.
        builder.clone(*op.getOperation());
        // Add a barrier after each clone. If the original dispatch has a small
        // problem size, simply duplicating without barrier will increase the
        // number of subgroups and thus "help" filling the GPU. In the end we
        // will have an over optimistic result. Inserting barriers avoids that,
        // but it assumes that the command buffer has a linear dispatch
        // structure.
        builder.create<IREE::HAL::CommandBufferExecutionBarrierOp>(
            op.getLoc(), op.getCommandBuffer(),
            IREE::HAL::ExecutionStageBitfield::CommandRetire |
                IREE::HAL::ExecutionStageBitfield::Dispatch,
            IREE::HAL::ExecutionStageBitfield::CommandIssue |
                IREE::HAL::ExecutionStageBitfield::Dispatch,
            IREE::HAL::ExecutionBarrierFlagBitfield::None);
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
