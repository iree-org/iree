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

struct DescriptorState {
  Value buffer;
  Value offset;
  Value length;
};

struct DescriptorSetState {
  Value pipelineLayout;
  SmallVector<DescriptorState, 32> descriptors;

  DescriptorState &getDescriptor(int64_t index) {
    if (index >= descriptors.size()) {
      descriptors.resize(index + 1);
    }
    return descriptors[index];
  }

  void clear() {
    pipelineLayout = {};
    descriptors.clear();
  }
};

struct CommandBufferState {
  // Push constants can only be reused with compatible layouts.
  Value pushConstantLayout;
  SmallVector<Value, 32> pushConstants;

  SmallVector<DescriptorSetState, 4> descriptorSets;

  // Set after we know a full barrier has been issued; any subsequent barrier
  // until a real operation is redundant. We could track more fine-grained state
  // here such as which stages are being waited on.
  // Note that we assume no barriers by default, as the command buffer may have
  // been passed as a function/branch argument and we don't have visibility.
  // We need to use IPO to track that.
  IREE::HAL::CommandBufferExecutionBarrierOp previousFullBarrier;

  Value &getPushConstant(int64_t index) {
    if (index >= pushConstants.size()) {
      pushConstants.resize(index + 1);
    }
    return pushConstants[index];
  }

  DescriptorSetState *getDescriptorSet(Value set) {
    APInt setInt;
    if (!matchPattern(set, m_ConstantInt(&setInt))) {
      // Dynamic set value; not analyzable with this approach.
      return nullptr;
    }
    int64_t index = setInt.getSExtValue();
    if (index >= descriptorSets.size()) {
      descriptorSets.resize(index + 1);
    }
    return &descriptorSets[index];
  }
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

static LogicalResult processOp(IREE::HAL::CommandBufferPushConstantsOp op,
                               CommandBufferState &state) {
  // Push constant state is only shared with the same layout.
  if (state.pushConstantLayout != op.getPipelineLayout()) {
    state.pushConstantLayout = op.getPipelineLayout();
    state.pushConstants.clear();
  }

  // Today we only eat constants from the beginning or end of the range
  // (hopefully removing the entire op). Sparse constant sets aren't worth it.
  int64_t baseIndex = op.getOffset().getSExtValue();
  llvm::BitVector redundantIndices(op.getValues().size());
  for (auto value : llvm::enumerate(op.getValues())) {
    auto &stateValue = state.getPushConstant(baseIndex + value.index());
    if (value.value() == stateValue) {
      // Redundant value.
      redundantIndices.set(value.index());
    } else {
      stateValue = value.value();
    }
  }
  if (redundantIndices.none())
    return success(); // no-op

  // If all bits are set we can just kill the op.
  if (redundantIndices.all()) {
    op.erase();
    return success();
  }

  int lastRedundant = redundantIndices.find_last();
  int lastNonRedundant = redundantIndices.find_last_unset();
  if (lastRedundant != -1 && lastRedundant > lastNonRedundant) {
    // Eat the last few constants.
    int redundantCount = redundantIndices.size() - lastRedundant;
    op.getValuesMutable().erase(lastRedundant, redundantCount);
  }

  int firstRedundant = redundantIndices.find_first();
  int firstNonRedundant = redundantIndices.find_first_unset();
  if (firstRedundant != -1 && firstRedundant < firstNonRedundant) {
    // Eat the first few constants by adjusting the offset and slicing out the
    // values.
    op.setOffsetAttr(Builder(op).getIndexAttr(baseIndex + firstRedundant + 1));
    op.getValuesMutable().erase(0, firstRedundant + 1);
  }

  assert(op.getValues().size() > 0 && "should not have removed all");
  return success();
}

static LogicalResult processOp(IREE::HAL::CommandBufferPushDescriptorSetOp op,
                               CommandBufferState &state) {
  auto *setState = state.getDescriptorSet(op.getSet());
  if (!setState)
    return failure();

  bool isLayoutEqual = setState->pipelineLayout == op.getPipelineLayout();
  setState->pipelineLayout = op.getPipelineLayout();

  int64_t descriptorCount = op.getBindingBuffers().size();
  llvm::BitVector redundantIndices(descriptorCount);
  for (int64_t index = 0; index < descriptorCount; ++index) {
    auto &descriptor = setState->getDescriptor(index);
    auto buffer = op.getBindingBuffers()[index];
    auto offset = op.getBindingOffsets()[index];
    auto length = op.getBindingLengths()[index];
    if (descriptor.buffer == buffer && descriptor.offset == offset &&
        descriptor.length == length) {
      // Redundant descriptor.
      redundantIndices.set(index);
    } else {
      descriptor.buffer = buffer;
      descriptor.offset = offset;
      descriptor.length = length;
    }
  }

  // Bail early if no redundant bindings.
  if (isLayoutEqual && redundantIndices.none()) {
    return success(); // no-op
  }

  // If all bits are set we can just kill the op.
  if (isLayoutEqual && redundantIndices.all()) {
    op.erase();
    return success();
  }

  return success();
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
          assert(commandBuffer.getType().isa<IREE::HAL::CommandBufferType>() &&
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
              .Case([&](IREE::HAL::CommandBufferPushConstantsOp op) {
                resetCommandBufferBarrierBit(op);
                if (failed(processOp(op, stateMap[op.getCommandBuffer()]))) {
                  invalidateState(op.getCommandBuffer());
                }
              })
              .Case([&](IREE::HAL::CommandBufferPushDescriptorSetOp op) {
                resetCommandBufferBarrierBit(op);
                if (failed(processOp(op, stateMap[op.getCommandBuffer()]))) {
                  invalidateState(op.getCommandBuffer());
                }
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
