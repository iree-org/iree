// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/Utils.h"
#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<bool> clExternalResourcesMappable(
    "iree-stream-external-resources-mappable",
    llvm::cl::desc("Allocates external resources as host-visible and mappable. "
                   "This can degrade performance and introduce allocation "
                   "overhead and staging buffers for readback on the host "
                   "should be managed by the calling application instead."),
    llvm::cl::init(false));

namespace mlir::iree_compiler {

LogicalResult
deriveRequiredResourceBufferBits(Location loc, IREE::Stream::Lifetime lifetime,
                                 IREE::HAL::MemoryTypeBitfield &memoryTypes,
                                 IREE::HAL::BufferUsageBitfield &bufferUsage) {
  memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
  bufferUsage = IREE::HAL::BufferUsageBitfield::None;
  switch (lifetime) {
  default:
    return mlir::emitError(loc) << "unsupported resource lifetime: "
                                << IREE::Stream::stringifyLifetime(lifetime);
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
deriveAllowedResourceBufferBits(Location loc, IREE::Stream::Lifetime lifetime,
                                IREE::HAL::MemoryTypeBitfield &memoryTypes,
                                IREE::HAL::BufferUsageBitfield &bufferUsage) {
  memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
  bufferUsage = IREE::HAL::BufferUsageBitfield::None;
  if (failed(deriveRequiredResourceBufferBits(loc, lifetime, memoryTypes,
                                              bufferUsage))) {
    return failure();
  }
  switch (lifetime) {
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

} // namespace mlir::iree_compiler
