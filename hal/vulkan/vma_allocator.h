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

#ifndef IREE_HAL_VULKAN_VMA_ALLOCATOR_H_
#define IREE_HAL_VULKAN_VMA_ALLOCATOR_H_

#include <vulkan/vulkan.h>

#include <memory>

#include "base/status.h"
#include "hal/allocator.h"
#include "hal/vulkan/dynamic_symbols.h"
#include "hal/vulkan/handle_util.h"
#include "hal/vulkan/internal_vk_mem_alloc.h"

namespace iree {
namespace hal {
namespace vulkan {

class VmaBuffer;

// A HAL allocator using the Vulkan Memory Allocator (VMA) to manage memory.
// VMA (//third_party/vulkan_memory_allocator) provides dlmalloc-like behavior
// with suballocations made with various policies (best fit, first fit, etc).
// This reduces the number of allocations we need from the Vulkan implementation
// (which can sometimes be limited to as little as 4096 total allowed) and
// manages higher level allocation semantics like slab allocation and
// defragmentation.
//
// VMA is internally synchronized and the functionality exposed on the HAL
// interface is thread-safe.
//
// More information:
//   https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
//   https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/
class VmaAllocator final : public Allocator {
 public:
  static StatusOr<std::unique_ptr<VmaAllocator>> Create(
      VkPhysicalDevice physical_device,
      const ref_ptr<VkDeviceHandle>& logical_device);

  ~VmaAllocator() override;

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  ::VmaAllocator vma() const { return vma_; }

  bool CanUseBufferLike(Allocator* source_allocator,
                        MemoryTypeBitfield memory_type,
                        BufferUsageBitfield buffer_usage,
                        BufferUsageBitfield intended_usage) const override;

  bool CanAllocate(MemoryTypeBitfield memory_type,
                   BufferUsageBitfield buffer_usage,
                   size_t allocation_size) const override;

  Status MakeCompatible(MemoryTypeBitfield* memory_type,
                        BufferUsageBitfield* buffer_usage) const override;

  StatusOr<ref_ptr<Buffer>> Allocate(MemoryTypeBitfield memory_type,
                                     BufferUsageBitfield buffer_usage,
                                     size_t allocation_size) override;

  StatusOr<ref_ptr<Buffer>> AllocateConstant(
      BufferUsageBitfield buffer_usage, ref_ptr<Buffer> source_buffer) override;

  StatusOr<ref_ptr<Buffer>> WrapMutable(MemoryTypeBitfield memory_type,
                                        MemoryAccessBitfield allowed_access,
                                        BufferUsageBitfield buffer_usage,
                                        void* data,
                                        size_t data_length) override;

 private:
  VmaAllocator(VkPhysicalDevice physical_device,
               const ref_ptr<VkDeviceHandle>& logical_device,
               ::VmaAllocator vma);

  StatusOr<ref_ptr<VmaBuffer>> AllocateInternal(
      MemoryTypeBitfield memory_type, BufferUsageBitfield buffer_usage,
      MemoryAccessBitfield allowed_access, size_t allocation_size,
      VmaAllocationCreateFlags flags);

  VkPhysicalDevice physical_device_;
  ref_ptr<VkDeviceHandle> logical_device_;

  // Internally synchronized. We could externally synchronize if we thought it
  // was worth it, however I'm not sure we'd be able to do much better with the
  // current Allocator API.
  ::VmaAllocator vma_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_VMA_ALLOCATOR_H_
