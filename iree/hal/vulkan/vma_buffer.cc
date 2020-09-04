// Copyright 2019 Google LLC
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

#include "iree/hal/vulkan/vma_buffer.h"

#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/vma_allocator.h"

namespace iree {
namespace hal {
namespace vulkan {

VmaBuffer::VmaBuffer(VmaAllocator* allocator, MemoryTypeBitfield memory_type,
                     MemoryAccessBitfield allowed_access,
                     BufferUsageBitfield usage, device_size_t allocation_size,
                     device_size_t byte_offset, device_size_t byte_length,
                     VkBuffer buffer, VmaAllocation allocation,
                     VmaAllocationInfo allocation_info)
    : Buffer(allocator, memory_type, allowed_access, usage, allocation_size,
             byte_offset, byte_length),
      vma_(allocator->vma()),
      buffer_(buffer),
      allocation_(allocation),
      allocation_info_(allocation_info) {
  // TODO(benvanik): set debug name instead and use the
  //     VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT flag.
  vmaSetAllocationUserData(vma_, allocation_, this);
}

VmaBuffer::~VmaBuffer() {
  IREE_TRACE_SCOPE0("VmaBuffer::dtor");
  vmaDestroyBuffer(vma_, buffer_, allocation_);
}

Status VmaBuffer::FillImpl(device_size_t byte_offset, device_size_t byte_length,
                           const void* pattern, device_size_t pattern_length) {
  IREE_ASSIGN_OR_RETURN(
      auto mapping, MapMemory<uint8_t>(MemoryAccess::kDiscardWrite, byte_offset,
                                       byte_length));
  void* data_ptr = static_cast<void*>(mapping.mutable_data());
  switch (pattern_length) {
    case 1: {
      uint8_t* data = static_cast<uint8_t*>(data_ptr);
      uint8_t value_bits = *static_cast<const uint8_t*>(pattern);
      std::fill_n(data + byte_offset, byte_length, value_bits);
      break;
    }
    case 2: {
      uint16_t* data = static_cast<uint16_t*>(data_ptr);
      uint16_t value_bits = *static_cast<const uint16_t*>(pattern);
      std::fill_n(data + byte_offset / sizeof(uint16_t),
                  byte_length / sizeof(uint16_t), value_bits);
      break;
    }
    case 4: {
      uint32_t* data = static_cast<uint32_t*>(data_ptr);
      uint32_t value_bits = *static_cast<const uint32_t*>(pattern);
      std::fill_n(data + byte_offset / sizeof(uint32_t),
                  byte_length / sizeof(uint32_t), value_bits);
      break;
    }
    default:
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Unsupported scalar data size: " << pattern_length;
  }
  return OkStatus();
}

Status VmaBuffer::ReadDataImpl(device_size_t source_offset, void* data,
                               device_size_t data_length) {
  IREE_ASSIGN_OR_RETURN(
      auto mapping,
      MapMemory<uint8_t>(MemoryAccess::kRead, source_offset, data_length));
  std::memcpy(data, mapping.data(), mapping.byte_length());
  return OkStatus();
}

Status VmaBuffer::WriteDataImpl(device_size_t target_offset, const void* data,
                                device_size_t data_length) {
  IREE_ASSIGN_OR_RETURN(auto mapping,
                        MapMemory<uint8_t>(MemoryAccess::kDiscardWrite,
                                           target_offset, data_length));
  std::memcpy(mapping.mutable_data(), data, mapping.byte_length());
  return OkStatus();
}

Status VmaBuffer::CopyDataImpl(device_size_t target_offset,
                               Buffer* source_buffer,
                               device_size_t source_offset,
                               device_size_t data_length) {
  // This is pretty terrible. Let's not do this.
  // TODO(benvanik): a way for allocators to indicate transfer compat.
  IREE_ASSIGN_OR_RETURN(auto source_mapping,
                        source_buffer->MapMemory<uint8_t>(
                            MemoryAccess::kRead, source_offset, data_length));
  CHECK_EQ(data_length, source_mapping.size());
  IREE_ASSIGN_OR_RETURN(auto target_mapping,
                        MapMemory<uint8_t>(MemoryAccess::kDiscardWrite,
                                           target_offset, data_length));
  CHECK_EQ(data_length, target_mapping.size());
  std::memcpy(target_mapping.mutable_data() + target_offset,
              source_mapping.data(), data_length);
  return OkStatus();
}

Status VmaBuffer::MapMemoryImpl(MappingMode mapping_mode,
                                MemoryAccessBitfield memory_access,
                                device_size_t local_byte_offset,
                                device_size_t local_byte_length,
                                void** out_data) {
  uint8_t* data_ptr = nullptr;
  VK_RETURN_IF_ERROR(
      vmaMapMemory(vma_, allocation_, reinterpret_cast<void**>(&data_ptr)));
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

  return OkStatus();
}

Status VmaBuffer::UnmapMemoryImpl(device_size_t local_byte_offset,
                                  device_size_t local_byte_length, void* data) {
  vmaUnmapMemory(vma_, allocation_);
  return OkStatus();
}

Status VmaBuffer::InvalidateMappedMemoryImpl(device_size_t local_byte_offset,
                                             device_size_t local_byte_length) {
  vmaInvalidateAllocation(vma_, allocation_, local_byte_offset,
                          local_byte_length);
  return OkStatus();
}

Status VmaBuffer::FlushMappedMemoryImpl(device_size_t local_byte_offset,
                                        device_size_t local_byte_length) {
  vmaFlushAllocation(vma_, allocation_, local_byte_offset, local_byte_length);
  return OkStatus();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
