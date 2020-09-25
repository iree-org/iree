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

#include "iree/hal/metal/metal_buffer.h"

#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/metal/metal_direct_allocator.h"

namespace iree {
namespace hal {
namespace metal {

// static
StatusOr<ref_ptr<MetalBuffer>> MetalBuffer::Create(
    MetalDirectAllocator* allocator, MemoryTypeBitfield memory_type,
    MemoryAccessBitfield allowed_access, BufferUsageBitfield usage, device_size_t allocation_size,
    device_size_t byte_offset, device_size_t byte_length, id<MTLBuffer> buffer,
    id<MTLCommandQueue> transfer_queue) {
  IREE_TRACE_SCOPE0("MetalBuffer::Create");
  return assign_ref(new MetalBuffer(allocator, memory_type, allowed_access, usage, allocation_size,
                                    byte_offset, byte_length, [buffer retain], transfer_queue));
}

// static
StatusOr<ref_ptr<MetalBuffer>> MetalBuffer::CreateUnretained(
    MetalDirectAllocator* allocator, MemoryTypeBitfield memory_type,
    MemoryAccessBitfield allowed_access, BufferUsageBitfield usage, device_size_t allocation_size,
    device_size_t byte_offset, device_size_t byte_length, id<MTLBuffer> buffer,
    id<MTLCommandQueue> transfer_queue) {
  IREE_TRACE_SCOPE0("MetalBuffer::Create");
  return assign_ref(new MetalBuffer(allocator, memory_type, allowed_access, usage, allocation_size,
                                    byte_offset, byte_length, buffer, transfer_queue));
}

MetalBuffer::MetalBuffer(MetalDirectAllocator* allocator, MemoryTypeBitfield memory_type,
                         MemoryAccessBitfield allowed_access, BufferUsageBitfield usage,
                         device_size_t allocation_size, device_size_t byte_offset,
                         device_size_t byte_length, id<MTLBuffer> buffer,
                         id<MTLCommandQueue> transfer_queue)
    : Buffer(allocator, memory_type, allowed_access, usage, allocation_size, byte_offset,
             byte_length),
      metal_transfer_queue_([transfer_queue retain]),
      metal_handle_(buffer) {}

MetalBuffer::~MetalBuffer() {
  IREE_TRACE_SCOPE0("MetalBuffer::dtor");
  [metal_handle_ release];
  [metal_transfer_queue_ release];
}

Status MetalBuffer::FillImpl(device_size_t byte_offset, device_size_t byte_length,
                             const void* pattern, device_size_t pattern_length) {
  IREE_ASSIGN_OR_RETURN(auto mapping,
                        MapMemory<uint8_t>(MemoryAccess::kDiscardWrite, byte_offset, byte_length));
  void* data_ptr = static_cast<void*>(mapping.mutable_data());
  switch (pattern_length) {
    case 1: {
      uint8_t* data = static_cast<uint8_t*>(data_ptr);
      uint8_t value_bits = *static_cast<const uint8_t*>(pattern);
      std::fill_n(data, byte_length, value_bits);
      break;
    }
    case 2: {
      uint16_t* data = static_cast<uint16_t*>(data_ptr);
      uint16_t value_bits = *static_cast<const uint16_t*>(pattern);
      std::fill_n(data, byte_length / sizeof(uint16_t), value_bits);
      break;
    }
    case 4: {
      uint32_t* data = static_cast<uint32_t*>(data_ptr);
      uint32_t value_bits = *static_cast<const uint32_t*>(pattern);
      std::fill_n(data, byte_length / sizeof(uint32_t), value_bits);
      break;
    }
    default:
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Unsupported scalar data size: " << pattern_length;
  }
  return OkStatus();
}

Status MetalBuffer::ReadDataImpl(device_size_t source_offset, void* data,
                                 device_size_t data_length) {
  IREE_ASSIGN_OR_RETURN(auto mapping,
                        MapMemory<uint8_t>(MemoryAccess::kRead, source_offset, data_length));
  std::memcpy(data, mapping.data(), mapping.byte_length());
  return OkStatus();
}

Status MetalBuffer::WriteDataImpl(device_size_t target_offset, const void* data,
                                  device_size_t data_length) {
  IREE_ASSIGN_OR_RETURN(
      auto mapping, MapMemory<uint8_t>(MemoryAccess::kDiscardWrite, target_offset, data_length));
  std::memcpy(mapping.mutable_data(), data, mapping.byte_length());
  return OkStatus();
}

Status MetalBuffer::CopyDataImpl(device_size_t target_offset, Buffer* source_buffer,
                                 device_size_t source_offset, device_size_t data_length) {
  // This is pretty terrible. Let's not do this.
  // TODO(benvanik): a way for allocators to indicate transfer compat.
  IREE_ASSIGN_OR_RETURN(auto source_mapping, source_buffer->MapMemory<uint8_t>(
                                                 MemoryAccess::kRead, source_offset, data_length));
  IREE_CHECK_EQ(data_length, source_mapping.size());
  IREE_ASSIGN_OR_RETURN(auto target_mapping, MapMemory<uint8_t>(MemoryAccess::kDiscardWrite,
                                                                target_offset, data_length));
  IREE_CHECK_EQ(data_length, target_mapping.size());
  std::memcpy(target_mapping.mutable_data(), source_mapping.data(), data_length);
  return OkStatus();
}

Status MetalBuffer::MapMemoryImpl(MappingMode mapping_mode, MemoryAccessBitfield memory_access,
                                  device_size_t local_byte_offset, device_size_t local_byte_length,
                                  void** out_data) {
  uint8_t* data_ptr = reinterpret_cast<uint8_t*>([metal_handle_ contents]);
  *out_data = data_ptr + local_byte_offset;

  // If we mapped for discard scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for
  // heap buffers we could reallocate them such that ASAN yells, but that
  // would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (AnyBitSet(memory_access & MemoryAccess::kDiscard)) {
    std::memset(data_ptr + local_byte_offset, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG

  if (requires_autosync()) {
    IREE_RETURN_IF_ERROR(InvalidateMappedMemoryImpl(local_byte_offset, local_byte_length));
  }

  return OkStatus();
}

Status MetalBuffer::UnmapMemoryImpl(device_size_t local_byte_offset,
                                    device_size_t local_byte_length, void* data) {
  if (requires_autosync()) {
    IREE_RETURN_IF_ERROR(FlushMappedMemoryImpl(local_byte_offset, local_byte_length));
  }

  return OkStatus();
}

Status MetalBuffer::InvalidateMappedMemoryImpl(device_size_t local_byte_offset,
                                               device_size_t local_byte_length) {
#ifdef IREE_PLATFORM_MACOS
  // The following is only necessary for MTLStorageManaged.
  if (metal_handle_.storageMode == MTLStorageModeManaged) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer =
          [metal_transfer_queue_ commandBufferWithUnretainedReferences];

      id<MTLBlitCommandEncoder> blit_encoder = [command_buffer blitCommandEncoder];
      [blit_encoder synchronizeResource:metal_handle_];
      [blit_encoder endEncoding];

      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }
  }
#endif

  return OkStatus();
}

Status MetalBuffer::FlushMappedMemoryImpl(device_size_t local_byte_offset,
                                          device_size_t local_byte_length) {
#ifdef IREE_PLATFORM_MACOS
  // The following is only necessary for MTLStorageManaged.
  if (metal_handle_.storageMode == MTLStorageModeManaged) {
    [metal_handle_ didModifyRange:NSMakeRange(local_byte_offset, local_byte_length)];
  }
#endif

  return OkStatus();
}

bool MetalBuffer::requires_autosync() const {
  // We only need to perform "automatic" resource synchronization if it's MTLStorageModeManaged,
  // which is only available on macOS.
#ifdef IREE_PLATFORM_MACOS
  return AllBitsSet(memory_type(), MemoryType::kHostCoherent) &&
         metal_handle_.storageMode == MTLStorageModeManaged;
#else
  return false;
#endif
}

}  // namespace metal
}  // namespace hal
}  // namespace iree
