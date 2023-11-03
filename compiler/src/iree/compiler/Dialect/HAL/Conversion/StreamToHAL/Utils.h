// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_STREAMTOHAL_UTILS_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_STREAMTOHAL_UTILS_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

// Returns a !hal.device for the affinity specified on |op|.
Value lookupDeviceFor(Operation *op, OpBuilder &builder);

// Returns a !hal.device and queue affinity i64 for the affinity specified on
// |op|.
std::tuple<Value, Value> lookupDeviceAndQueueAffinityFor(Operation *op,
                                                         OpBuilder &builder);

// Returns the !hal.allocator for the affinity specified on |op|.
Value lookupAllocatorFor(Operation *op, OpBuilder &builder);

// Returns a !hal.allocator and queue affinity i64 for the affinity specified on
// |op|.
std::tuple<Value, Value> lookupAllocatorAndQueueAffinityFor(Operation *op,
                                                            OpBuilder &builder);

// Returns the |timepointFence| or a util.null if the wait is to be ignored.
Value getOrCreateWaitFence(Location loc, Value timepointFence,
                           PatternRewriter &rewriter);

// Returns the a new fence for |timepoint| or an existing fence if one was
// associated with an external fence. Returns util.null if no one observes the
// fence.
Value getOrCreateSignalFence(Location loc, Value device, Value timepoint,
                             PatternRewriter &rewriter);

// Scans all of the stream.cmd.* ops in the region to derive a command category.
IREE::HAL::CommandCategoryBitfield deriveCommandCategories(Region &region);

// Maps a resource type to the corresponding HAL memory types and buffer usage.
// This will fail if the resource type is not directly mappable to HAL bits.
// The bits set here are those that must be set for the buffer to be used as the
// buffer within the program with its defined resource lifetime.
LogicalResult
deriveRequiredResourceBufferBits(Location loc,
                                 IREE::Stream::ResourceType resourceType,
                                 IREE::HAL::MemoryTypeBitfield &memoryTypes,
                                 IREE::HAL::BufferUsageBitfield &bufferUsage);

// Maps a resource type to the corresponding HAL memory types and buffer usage.
// This will fail if the resource type is not directly mappable to HAL bits.
// The bits set here represent the superset of required and allowed bits and
// are useful for providing buffers back to users via the ABI that may need to
// be used for more than just what the internal program requires.
LogicalResult
deriveAllowedResourceBufferBits(Location loc,
                                IREE::Stream::ResourceType resourceType,
                                IREE::HAL::MemoryTypeBitfield &memoryTypes,
                                IREE::HAL::BufferUsageBitfield &bufferUsage);

class StreamConversionMapping {
public:
  // Maps the stream dialect |executeOp| to the hal dialect |commandBuffer|
  // value used during recording. Patterns can use this to find the SSA value
  // they need to make hal.command_buffer.* ops.
  void mapCommandBuffer(IREE::Stream::CmdExecuteOp executeOp,
                        Value commandBuffer);

  // Looks up a mapped command buffer SSA value that can be used by the given
  // stream.cmd.* op.
  Value lookupCommandBufferFor(Operation *cmdOp) const;

private:
  // Ops within stream.cmd.execute ops -> !hal.command_buffer.
  DenseMap<Operation *, Value> commandBuffers;
};

template <typename OpT>
struct StreamConversionPattern : public OpConversionPattern<OpT> {
  StreamConversionPattern(std::shared_ptr<StreamConversionMapping> mapping,
                          TypeConverter &typeConverter, MLIRContext *context,
                          PatternBenefit benefit = 1)
      : OpConversionPattern<OpT>(typeConverter, context, benefit),
        mapping(std::move(mapping)) {}

  std::shared_ptr<StreamConversionMapping> mapping;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_HAL_CONVERSION_STREAMTOHAL_UTILS_H_
