// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/metal/metal_direct_allocator.h"

#include "absl/memory/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/metal/metal_buffer.h"

namespace iree {
namespace hal {
namespace metal {

namespace {

// Returns the proper Metal resource storage mode given the specific MemoryType.
MTLResourceOptions SelectMTLResourceStorageMode(MemoryType memory_type) {
  // There are four MTLStorageMode:
  // * Managed: The CPU and GPU may maintain separate copies of the resource, and any changes
  //   must be explicitly synchronized.
  // * Shared: The resource is stored in system memory and is accessible to both the CPU and
  //   the GPU.
  // * Private: The resource can be accessed only by the GPU.
  // * Memoryless: The resource’s contents can be accessed only by the GPU and only exist
  //   temporarily during a render pass.
  // macOS has all of the above; MTLStorageModeManaged is not available on iOS.
  //
  // The IREE HAL is modeled after Vulkan so it's quite explicit. For buffers visible to both
  // the host and the device, we would like to opt in with the explicit version
  // (MTLStorageManaged) when possible because it should be more performant: "In macOS,
  // there’s no difference in GPU performance between managed and private buffers." But for
  // iOS, MTLStorageShared should be good given we have a unified memory model there.

  if (AllBitsSet(memory_type, MemoryType::kDeviceLocal)) {
    if (AllBitsSet(memory_type, MemoryType::kHostVisible)) {
      // Device-local, host-visible.
      // TODO(antiagainst): Enable using MTLResourceStorageModeManaged on macOS once we have
      // defined invalidate/flush C APIs and wired up their usage through the stack. At the
      // moment if we use MTLResourceStorageModeManaged, due to no proper invlidate/flush
      // actions, the kernel invocations' data read/write will not be properly synchronized.
      return MTLResourceStorageModeShared;
    } else {
      // Device-local only.
      return MTLResourceStorageModePrivate;
    }
  } else {
    if (AllBitsSet(memory_type, MemoryType::kDeviceVisible)) {
      // Host-local, device-visible.
      return MTLResourceStorageModeShared;
    } else {
      // Host-local only.
      // TODO(antiagainst): we probably want to just use HostBuffer here.
      return MTLResourceStorageModeShared;
    }
  }
}

}  // namespace

// static
std::unique_ptr<MetalDirectAllocator> MetalDirectAllocator::Create(
    id<MTLDevice> device, id<MTLCommandQueue> transfer_queue) {
  IREE_TRACE_SCOPE0("MetalDirectAllocator::Create");
  return absl::WrapUnique(new MetalDirectAllocator(device, transfer_queue));
}

MetalDirectAllocator::MetalDirectAllocator(id<MTLDevice> device, id<MTLCommandQueue> transfer_queue)
    : metal_device_([device retain]), metal_transfer_queue_([transfer_queue retain]) {}

MetalDirectAllocator::~MetalDirectAllocator() {
  IREE_TRACE_SCOPE0("MetalDirectAllocator::dtor");
  [metal_transfer_queue_ release];
  [metal_device_ release];
}

bool MetalDirectAllocator::CanUseBufferLike(Allocator* source_allocator,
                                            MemoryTypeBitfield memory_type,
                                            BufferUsageBitfield buffer_usage,
                                            BufferUsageBitfield intended_usage) const {
  // TODO(benvanik): ensure there is a memory type that can satisfy the request.
  return source_allocator == this;
}

bool MetalDirectAllocator::CanAllocate(MemoryTypeBitfield memory_type,
                                       BufferUsageBitfield buffer_usage,
                                       size_t allocation_size) const {
  // TODO(benvanik): ensure there is a memory type that can satisfy the request.
  return true;
}

Status MetalDirectAllocator::MakeCompatible(MemoryTypeBitfield* memory_type,
                                            BufferUsageBitfield* buffer_usage) const {
  // TODO(benvanik): mutate to match supported memory types.
  return OkStatus();
}

StatusOr<ref_ptr<MetalBuffer>> MetalDirectAllocator::AllocateInternal(
    MemoryTypeBitfield memory_type, BufferUsageBitfield buffer_usage,
    MemoryAccessBitfield allowed_access, size_t allocation_size) {
  IREE_TRACE_SCOPE0("MetalDirectAllocator::AllocateInternal");

  MTLResourceOptions resource_options = SelectMTLResourceStorageMode(memory_type);

  // IREE is more explicit than Metal: it tracks various state by itself. There is no need
  // to incur Metal runtime overhead for hazard tracking.
  resource_options |= MTLResourceHazardTrackingModeUntracked;

  id<MTLBuffer> metal_buffer = [metal_device_ newBufferWithLength:allocation_size
                                                          options:resource_options];  // retained

  return MetalBuffer::CreateUnretained(
      this, memory_type, allowed_access, buffer_usage, allocation_size, /*byte_offset=*/0,
      /*byte_length=*/allocation_size, metal_buffer, metal_transfer_queue_);
}

StatusOr<ref_ptr<Buffer>> MetalDirectAllocator::Allocate(MemoryTypeBitfield memory_type,
                                                         BufferUsageBitfield buffer_usage,
                                                         size_t allocation_size) {
  IREE_TRACE_SCOPE0("MetalDirectAllocator::Allocate");
  return AllocateInternal(memory_type, buffer_usage, MemoryAccess::kAll, allocation_size);
}

StatusOr<ref_ptr<Buffer>> MetalDirectAllocator::AllocateConstant(BufferUsageBitfield buffer_usage,
                                                                 ref_ptr<Buffer> source_buffer) {
  IREE_TRACE_SCOPE0("MetalDirectAllocator::AllocateConstant");
  // TODO(benvanik): import memory to avoid the copy.
  IREE_ASSIGN_OR_RETURN(
      auto buffer, AllocateInternal(MemoryType::kDeviceLocal | MemoryType::kHostVisible,
                                    buffer_usage, MemoryAccess::kRead | MemoryAccess::kDiscardWrite,
                                    source_buffer->byte_length()));
  IREE_RETURN_IF_ERROR(buffer->CopyData(0, source_buffer.get(), 0, kWholeBuffer));
  return buffer;
}

StatusOr<ref_ptr<Buffer>> MetalDirectAllocator::WrapMutable(MemoryTypeBitfield memory_type,
                                                            MemoryAccessBitfield allowed_access,
                                                            BufferUsageBitfield buffer_usage,
                                                            void* data, size_t data_length) {
  IREE_TRACE_SCOPE0("MetalDirectAllocator::WrapMutable");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalDirectAllocator::WrapMutable";
}

}  // namespace metal
}  // namespace hal
}  // namespace iree
