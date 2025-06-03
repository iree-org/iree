// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/StreamToHAL/Utils.h"

#include "iree/compiler/Dialect/HAL/Analysis/Captures.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

static llvm::cl::opt<bool> clExternalResourcesMappable(
    "iree-stream-external-resources-mappable",
    llvm::cl::desc("Allocates external resources as host-visible and mappable. "
                   "This can degrade performance and introduce allocation "
                   "overhead and staging buffers for readback on the host "
                   "should be managed by the calling application instead."),
    llvm::cl::init(false));

namespace mlir::iree_compiler {

Value lookupDeviceFor(Operation *op, OpBuilder &builder) {
  auto affinityAttr = IREE::Stream::AffinityAttr::lookupOrDefault(op);
  auto resolveOp = builder.create<IREE::Stream::ContextResolveOp>(
      op->getLoc(),
      TypeRange{
          builder.getType<IREE::HAL::DeviceType>(),
      },
      affinityAttr);
  return resolveOp.getResult(0);
}

std::tuple<Value, Value> lookupDeviceAndQueueAffinityFor(Operation *op,
                                                         OpBuilder &builder) {
  auto affinityAttr = IREE::Stream::AffinityAttr::lookupOrDefault(op);
  auto resolveOp = builder.create<IREE::Stream::ContextResolveOp>(
      op->getLoc(),
      TypeRange{
          builder.getType<IREE::HAL::DeviceType>(),
          builder.getI64Type(),
      },
      affinityAttr);
  return std::make_tuple(resolveOp.getResult(0), resolveOp.getResult(1));
}

std::tuple<Value, Value> lookupDeviceAndQueueAffinityFor(
    Operation *op, IREE::HAL::MemoryTypeBitfield memoryTypes,
    IREE::HAL::BufferUsageBitfield bufferUsage, OpBuilder &builder) {
  // Emit a select op to let the runtime decide which device/queue affinity to
  // use if required.
  auto affinityAttr = IREE::Stream::AffinityAttr::lookupOrDefault(op);
  if (auto optimalAttr = dyn_cast<IREE::HAL::DeviceOptimalAttr>(affinityAttr)) {
    auto selectOp = builder.create<IREE::HAL::AllocatorSelectAttrOp>(
        op->getLoc(), optimalAttr, memoryTypes, bufferUsage);
    return {selectOp.getResult(0), selectOp.getResult(1)};
  }
  // Unconditionally routed affinities go down the normal resolve path.
  return lookupDeviceAndQueueAffinityFor(op, builder);
}

Value lookupAllocatorFor(Operation *op, OpBuilder &builder) {
  auto affinityAttr = IREE::Stream::AffinityAttr::lookupOrDefault(op);
  auto resolveOp = builder.create<IREE::Stream::ContextResolveOp>(
      op->getLoc(),
      TypeRange{
          builder.getType<IREE::HAL::AllocatorType>(),
      },
      affinityAttr);
  return resolveOp.getResult(0);
}

std::tuple<Value, Value> lookupAllocatorAndQueueAffinityFor(
    Operation *op, IREE::HAL::MemoryTypeBitfield memoryTypes,
    IREE::HAL::BufferUsageBitfield bufferUsage, OpBuilder &builder) {
  auto [device, queueAffinity] =
      lookupDeviceAndQueueAffinityFor(op, memoryTypes, bufferUsage, builder);
  Value allocator =
      builder.create<IREE::HAL::DeviceAllocatorOp>(op->getLoc(), device);
  return {allocator, queueAffinity};
}

Value getOrCreateWaitFence(Location loc, Value timepointFence,
                           PatternRewriter &rewriter) {
  if (timepointFence)
    return timepointFence;
  return rewriter.create<IREE::Util::NullOp>(
      loc, rewriter.getType<IREE::HAL::FenceType>());
}

// Finds a !hal.fence bound to |timepoint| via a chain op and returns it if
// it is usable at the builder insertion point. The chain ops is only used if
// it is the only consumer of the timepoint and it is removed upon return.
static Value consumeBoundFence(Value timepoint, PatternRewriter &rewriter) {
  // Must only have one use. We can't consume a fence multiple times.
  if (!timepoint.hasOneUse())
    return nullptr; // >1 use

  // The use must be an export to a fence.
  auto chainOp = dyn_cast<IREE::Stream::TimepointChainExternalOp>(
      *timepoint.getUsers().begin());
  if (!chainOp)
    return nullptr; // non-export use
  assert(!chainOp.getExternalValues().empty());
  auto fence = chainOp.getExternalValues().front();
  if (!fence || !llvm::isa<IREE::HAL::FenceType>(fence.getType()))
    return nullptr;

  // Try really hard to figure out if the fence can be used. A larger analysis
  // pass running prior to conversion that did some code motion could help
  // ensure the fence SSA value is usable in the places it is needed - for now
  // we just do this local check that satisfies most common programs today. IPO
  // would do something like add the fence as an argument to function calls so
  // that the functions could consume it but inlining is pretty aggressive now.
  if (!IREE::Util::isValueUsableForOp(fence, rewriter.getBlock(),
                                      rewriter.getInsertionPoint())) {
    return nullptr; // unusable
  }

  // Consume the op by erasing it.
  rewriter.eraseOp(chainOp);

  return fence; // usable
}

Value getOrCreateSignalFence(Location loc, Value device, Value timepoint,
                             PatternRewriter &rewriter) {
  // Check to see if anyone is consuming the timepoint - if not then we don't
  // need create a fence.
  if (timepoint.use_empty()) {
    return rewriter.create<IREE::Util::NullOp>(
        loc, rewriter.getType<IREE::HAL::FenceType>());
  }

  // Check to see if the timepoint is associated with a fence. In common cases
  // when along ABI boundaries we can usually find an association.
  auto fence = consumeBoundFence(timepoint, rewriter);
  if (fence)
    return fence;

  // Create a new fence.
  return rewriter.create<IREE::HAL::FenceCreateOp>(
      loc, rewriter.getType<IREE::HAL::FenceType>(), device,
      IREE::HAL::FenceFlagBitfield::None);
}

IREE::HAL::CommandCategoryBitfield deriveCommandCategories(Region &region) {
  auto bits = IREE::HAL::CommandCategoryBitfield::None;
  for (auto &block : region) {
    for (auto &op : block) {
      if (isa<IREE::Stream::CmdCollectiveOp>(op) ||
          isa<IREE::Stream::CmdCallOp>(op)) {
        // Calls may do anything and collectives may be implemented as either
        // transfers or dispatches.
        bits = bits | IREE::HAL::CommandCategoryBitfield::Dispatch |
               IREE::HAL::CommandCategoryBitfield::Transfer;
      } else if (isa<IREE::Stream::CmdDispatchOp>(op)) {
        bits = bits | IREE::HAL::CommandCategoryBitfield::Dispatch;
      } else {
        bits = bits | IREE::HAL::CommandCategoryBitfield::Transfer;
      }
      for (auto &nestedRegion : op.getRegions()) {
        bits = bits | deriveCommandCategories(nestedRegion);
      }
    }
  }
  return bits;
}

LogicalResult
deriveRequiredResourceBufferBits(Location loc,
                                 IREE::Stream::ResourceType resourceType,
                                 IREE::HAL::MemoryTypeBitfield &memoryTypes,
                                 IREE::HAL::BufferUsageBitfield &bufferUsage) {
  memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
  bufferUsage = IREE::HAL::BufferUsageBitfield::None;
  switch (resourceType.getLifetime()) {
  default:
    return mlir::emitError(loc)
           << "unsupported resource lifetime: "
           << IREE::Stream::stringifyLifetime(resourceType.getLifetime());
  case IREE::Stream::Lifetime::Constant:
    // Device local; copies required to get into external resources.
    memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
    bufferUsage =
        bufferUsage | IREE::HAL::BufferUsageBitfield::SharingImmutable;
    break;
  case IREE::Stream::Lifetime::Variable:
    // Device local; copies required to get into external resources.
    memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
    break;
  case IREE::Stream::Lifetime::External:
    // We only require device-visible for external buffers (as we don't today
    // do anything else with them on the host). They may be mappable for user
    // convenience. Ideally they would have been placed in device-local memory
    // but so long as they are device visible the program will execute
    // correctly.
    memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceVisible;
    break;
  case IREE::Stream::Lifetime::Staging:
    // Host local; copies required to get into device resources.
    // We could vary this based on staging usage (upload/download) by
    // making it device-local|host-visible, but host-local means we have
    // a better chance of mapping it during uploads.
    memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::HostLocal |
                  IREE::HAL::MemoryTypeBitfield::DeviceVisible;
    bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                  IREE::HAL::BufferUsageBitfield::Mapping;
    break;
  case IREE::Stream::Lifetime::Transient:
    // Device local; copies required to get into external resources.
    memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
    break;
  }

  // TODO(benvanik): refine usage based on analysis.
  bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                IREE::HAL::BufferUsageBitfield::DispatchStorage;

  return success();
}

LogicalResult
deriveAllowedResourceBufferBits(Location loc,
                                IREE::Stream::ResourceType resourceType,
                                IREE::HAL::MemoryTypeBitfield &memoryTypes,
                                IREE::HAL::BufferUsageBitfield &bufferUsage) {
  memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
  bufferUsage = IREE::HAL::BufferUsageBitfield::None;
  if (failed(deriveRequiredResourceBufferBits(loc, resourceType, memoryTypes,
                                              bufferUsage))) {
    return failure();
  }
  switch (resourceType.getLifetime()) {
  default:
    break;
  case IREE::Stream::Lifetime::External:
    if (clExternalResourcesMappable) {
      // #yolo; these come from/go to outside the program.
      // Today we assume they are device-local|host-visible just for
      // practical purposes but that does not have to be true. We really
      // want this to be something we analyze and handle on the edges
      // (transferring devices/etc if needed).
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal |
                    IREE::HAL::MemoryTypeBitfield::HostVisible;
      // NOTE: we may not map it but users may after they get them back.
      // Another reason we should annotate this - having a buffer be
      // mappable is potentially expensive (may get a 2nd copy in memory!).
      bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Mapping;
    } else {
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
    }
    break;
  }
  return success();
}

BindingTable::BindingTable(IREE::Stream::CmdExecuteOp executeOp,
                           ValueRange bufferValues, ValueRange bufferSizes,
                           IndexSet &indexSet) {
  auto resourceValues = executeOp.getResourceOperands();
  auto capturedValues = executeOp.getBody().getArguments();

  // Categorize each resource value and add it to the table.
  for (auto [resourceValue, capturedValue, bufferValue, bufferSize] :
       llvm::zip_equal(resourceValues, capturedValues, bufferValues,
                       bufferSizes)) {
    switch (IREE::HAL::categorizeValue(resourceValue)) {
    default:
    case IREE::HAL::ValueOrigin::Unknown:
    case IREE::HAL::ValueOrigin::MutableGlobal: {
      // Indirect reference. Add to the table unless it already exists.
      Value slot = indexSet.get(indirectBuffers.size());
      if (indirectSlots.try_emplace(capturedValue, slot).second) {
        // TODO(benvanik): subset the range bound by the min/max used within the
        // consumer region. This would require emitting ops that track that
        // information (probably via util.range.min/max). For now we bind the
        // entire buffer range and let the individual commands subrange them.
        IREE::HAL::BindingValue bindingValue;
        bindingValue.buffer = bufferValue;
        bindingValue.byteOffset = indexSet.get(0);
        bindingValue.byteLength = bufferSize;
        indirectBuffers.push_back(bindingValue);
      }
      break;
    }
    case IREE::HAL::ValueOrigin::LocalConstant:
    case IREE::HAL::ValueOrigin::ImmutableGlobal: {
      // Direct reference; not in the table.
      break;
    }
    }
  }
}

std::optional<Value> BindingTable::lookupResourceSlot(Value resourceValue) {
  auto it = indirectSlots.find(resourceValue);
  if (it != indirectSlots.end()) {
    return it->second; // found a slot
  }
  return std::nullopt;
}

IREE::HAL::BindingValue CommandBufferConversionMapping::resolveBinding(
    Location loc, Value resourceValue, Value bufferValue, Value useOffset,
    Value useLength, OpBuilder &builder) {
  IREE::HAL::BindingValue bindingValue;

  // Try to resolve the resource to a slot. If not found then it's a direct
  // reference and we use the buffer provided.
  auto slot = bindingTable.lookupResourceSlot(resourceValue);
  if (slot.has_value()) {
    bindingValue.buffer = slot.value();
  } else {
    bindingValue.buffer = bufferValue;
  }

  // TODO(benvanik): adjust range by the binding table base index. Today all
  // binding table entries are the full buffers starting at zero.
  bindingValue.byteOffset = useOffset;
  bindingValue.byteLength = useLength;

  return bindingValue;
}

void StreamConversionMapping::mapCommandBuffer(
    IREE::Stream::CmdExecuteOp executeOp, Value commandBuffer,
    BindingTable bindingTable) {
  // Wrap metadata for the mapping.
  auto commandBufferMapping = std::make_shared<CommandBufferConversionMapping>(
      commandBuffer, std::move(bindingTable));

  // Map all ops nested within the command buffer so we can query later.
  opCommandBufferMap.insert(
      std::make_pair(executeOp, commandBufferMapping.get()));
  executeOp.walk([&](Operation *op) {
    opCommandBufferMap.insert(std::make_pair(op, commandBufferMapping.get()));
    return WalkResult::advance();
  });

  // Keep the mapping data alive until conversion completes.
  commandBuffers.push_back(std::move(commandBufferMapping));
}

CommandBufferConversionMapping &
StreamConversionMapping::lookupCommandBufferFor(Operation *cmdOp) const {
  auto it = opCommandBufferMap.find(cmdOp);
  assert(it != opCommandBufferMap.end() &&
         "command buffer must have been registered during conversion");
  return *it->second;
}

} // namespace mlir::iree_compiler
