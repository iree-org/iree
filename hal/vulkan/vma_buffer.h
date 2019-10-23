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

#ifndef IREE_HAL_VULKAN_VMA_BUFFER_H_
#define IREE_HAL_VULKAN_VMA_BUFFER_H_

#include <vulkan/vulkan.h>

#include "hal/buffer.h"
#include "vk_mem_alloc.h"

namespace iree {
namespace hal {
namespace vulkan {

class VmaAllocator;

// A buffer implementation representing an allocation made from within a pool of
// a Vulkan Memory Allocator instance. See VmaAllocator for more information.
class VmaBuffer final : public Buffer {
 public:
  VmaBuffer(VmaAllocator* allocator, MemoryTypeBitfield memory_type,
            MemoryAccessBitfield allowed_access, BufferUsageBitfield usage,
            device_size_t allocation_size, device_size_t byte_offset,
            device_size_t byte_length, VkBuffer buffer,
            VmaAllocation allocation, VmaAllocationInfo allocation_info);
  ~VmaBuffer() override;

  VkBuffer handle() const { return buffer_; }
  VmaAllocation allocation() const { return allocation_; }
  const VmaAllocationInfo& allocation_info() const { return allocation_info_; }

  // Exposed so that VmaAllocator can reset access after initial mapping.
  using Buffer::set_allowed_access;

 private:
  Status FillImpl(device_size_t byte_offset, device_size_t byte_length,
                  const void* pattern, device_size_t pattern_length) override;
  Status ReadDataImpl(device_size_t source_offset, void* data,
                      device_size_t data_length) override;
  Status WriteDataImpl(device_size_t target_offset, const void* data,
                       device_size_t data_length) override;
  Status CopyDataImpl(device_size_t target_offset, Buffer* source_buffer,
                      device_size_t source_offset,
                      device_size_t data_length) override;
  Status MapMemoryImpl(MappingMode mapping_mode,
                       MemoryAccessBitfield memory_access,
                       device_size_t local_byte_offset,
                       device_size_t local_byte_length,
                       void** out_data) override;
  Status UnmapMemoryImpl(device_size_t local_byte_offset,
                         device_size_t local_byte_length, void* data) override;
  Status InvalidateMappedMemoryImpl(device_size_t local_byte_offset,
                                    device_size_t local_byte_length) override;
  Status FlushMappedMemoryImpl(device_size_t local_byte_offset,
                               device_size_t local_byte_length) override;

  ::VmaAllocator vma_;
  VkBuffer buffer_;
  VmaAllocation allocation_;
  VmaAllocationInfo allocation_info_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_VMA_BUFFER_H_
