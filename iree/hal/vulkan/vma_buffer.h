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

// clang-format off: Must be included before all other headers:
#include "iree/hal/vulkan/vulkan_headers.h"
// clang-format on

#include "iree/hal/cc/buffer.h"
#include "iree/hal/vulkan/internal_vk_mem_alloc.h"

namespace iree {
namespace hal {
namespace vulkan {

class VmaAllocator;

// A buffer implementation representing an allocation made from within a pool of
// a Vulkan Memory Allocator instance. See VmaAllocator for more information.
class VmaBuffer final : public Buffer {
 public:
  VmaBuffer(VmaAllocator* allocator, iree_hal_memory_type_t memory_type,
            iree_hal_memory_access_t allowed_access,
            iree_hal_buffer_usage_t usage, iree_device_size_t allocation_size,
            iree_device_size_t byte_offset, iree_device_size_t byte_length,
            VkBuffer buffer, VmaAllocation allocation,
            VmaAllocationInfo allocation_info);
  ~VmaBuffer() override;

  VkBuffer handle() const { return buffer_; }
  VmaAllocation allocation() const { return allocation_; }
  const VmaAllocationInfo& allocation_info() const { return allocation_info_; }

  // Exposed so that VmaAllocator can reset access after initial mapping.
  using Buffer::set_allowed_access;

 private:
  Status FillImpl(iree_device_size_t byte_offset,
                  iree_device_size_t byte_length, const void* pattern,
                  iree_device_size_t pattern_length) override;
  Status ReadDataImpl(iree_device_size_t source_offset, void* data,
                      iree_device_size_t data_length) override;
  Status WriteDataImpl(iree_device_size_t target_offset, const void* data,
                       iree_device_size_t data_length) override;
  Status CopyDataImpl(iree_device_size_t target_offset, Buffer* source_buffer,
                      iree_device_size_t source_offset,
                      iree_device_size_t data_length) override;
  Status MapMemoryImpl(MappingMode mapping_mode,
                       iree_hal_memory_access_t memory_access,
                       iree_device_size_t local_byte_offset,
                       iree_device_size_t local_byte_length,
                       void** out_data) override;
  Status UnmapMemoryImpl(iree_device_size_t local_byte_offset,
                         iree_device_size_t local_byte_length,
                         void* data) override;
  Status InvalidateMappedMemoryImpl(
      iree_device_size_t local_byte_offset,
      iree_device_size_t local_byte_length) override;
  Status FlushMappedMemoryImpl(iree_device_size_t local_byte_offset,
                               iree_device_size_t local_byte_length) override;

  ::VmaAllocator vma_;
  VkBuffer buffer_;
  VmaAllocation allocation_;
  VmaAllocationInfo allocation_info_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_VMA_BUFFER_H_
