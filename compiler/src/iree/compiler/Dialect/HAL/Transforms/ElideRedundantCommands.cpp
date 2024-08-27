// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_ELIDEREDUNDANTCOMMANDSPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

struct CommandBufferState {
  // Set after we know a full barrier has been issued; any subsequent barrier
  // until a real operation is redundant. We could track more fine-grained state
  // here such as which stages are being waited on.
  // Note that we assume no barriers by default, as the command buffer may have
  // been passed as a function/branch argument and we don't have visibility.
  // We need to use IPO to track that.
  IREE::HAL::CommandBufferExecutionBarrierOp previousFullBarrier;
};

using CommandBufferStateMap = DenseMap<Value, CommandBufferState>;

static void processOp(IREE::HAL::CommandBufferExecutionBarrierOp op,
                      CommandBufferState &state) {
  if (state.previousFullBarrier) {
    // We are following a full barrier - this is a no-op (issuing two barriers
    // doesn't make the device barrier any harder).
    op.erase();
    return;
  }

  // See if this is a full barrier. These are all we emit today so this simple
  // analysis can remain simple by pattern matching.
  if (bitEnumContainsAny(op.getSourceStageMask(),
                         IREE::HAL::ExecutionStageBitfield::CommandRetire |
                             IREE::HAL::ExecutionStageBitfield::Transfer |
                             IREE::HAL::ExecutionStageBitfield::Dispatch) &&
      bitEnumContainsAny(op.getTargetStageMask(),
                         IREE::HAL::ExecutionStageBitfield::CommandRetire |
                             IREE::HAL::ExecutionStageBitfield::Transfer |
                             IREE::HAL::ExecutionStageBitfield::Dispatch)) {
    state.previousFullBarrier = op;
  } else {
    state.previousFullBarrier = {};
  }
}

//===----------------------------------------------------------------------===//
// --iree-hal-elide-redundant-commands
//===----------------------------------------------------------------------===//

struct ElideRedundantCommandsPass
    : public IREE::HAL::impl::ElideRedundantCommandsPassBase<
          ElideRedundantCommandsPass> {
  void runOnOperation() override {
    auto parentOp = getOperation();

    // TODO(benvanik): intraprocedural at least; keep track of state at block
    // boundaries. IPO would be nice but it (today) rarely happens that we
    // pass command buffers across calls.
    for (auto &region : parentOp->getRegions()) {
      for (auto &block : region.getBlocks()) {
        // State tracking for each command buffer found.
        // Discard state on ops we don't currently analyze (because this is
        // super basic - we really need to analyze them).
        CommandBufferStateMap stateMap;
        auto invalidateState = [&](Value commandBuffer) {
          stateMap[commandBuffer] = {};
        };
        auto resetCommandBufferBarrierBit = [&](Operation *op) {
          assert(op->getNumOperands() > 0 && "must be a command buffer op");
          auto commandBuffer = op->getOperand(0);
          assert(isa<IREE::HAL::CommandBufferType>(commandBuffer.getType()) &&
                 "operand 0 must be a command buffer");
          stateMap[commandBuffer].previousFullBarrier = {};
        };
        for (auto &op : llvm::make_early_inc_range(block.getOperations())) {
          if (!op.getDialect())
            continue;
          TypeSwitch<Operation *>(&op)
              .Case([&](IREE::HAL::CommandBufferFinalizeOp op) {
                invalidateState(op.getCommandBuffer());
              })
              .Case([&](IREE::HAL::CommandBufferExecutionBarrierOp op) {
                processOp(op, stateMap[op.getCommandBuffer()]);
              })
              .Case<IREE::HAL::CommandBufferDeviceOp,
                    IREE::HAL::CommandBufferBeginDebugGroupOp,
                    IREE::HAL::CommandBufferEndDebugGroupOp,
                    IREE::HAL::CommandBufferFillBufferOp,
                    IREE::HAL::CommandBufferCopyBufferOp,
                    IREE::HAL::CommandBufferDispatchOp,
                    IREE::HAL::CommandBufferDispatchIndirectOp>(
                  [&](Operation *op) {
                    // Ok - don't impact state.
                    resetCommandBufferBarrierBit(op);
                  })
              .Default([&](Operation *op) {
                // Unknown op - discard state cache.
                // This is to avoid correctness issues with region ops (like
                // scf.if) that we don't analyze properly here. We could
                // restrict this a bit by only discarding on use of the
                // command buffer.
                stateMap.clear();
              });
        }
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
