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

#include "iree/hal/vulkan/vma_allocator.h"

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/buffer.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/vma_buffer.h"

#if VMA_RECORDING_ENABLED
ABSL_FLAG(std::string, vma_recording_file, "",
          "File path to write a CSV containing the VMA recording.");
ABSL_FLAG(bool, vma_recording_flush_after_call, false,
          "Flush the VMA recording file after every call (useful if "
          "crashing/not exiting cleanly).");
#endif  // VMA_RECORDING_ENABLED

namespace iree {
namespace hal {
namespace vulkan {

// static
StatusOr<std::unique_ptr<VmaAllocator>> VmaAllocator::Create(
    VkPhysicalDevice physical_device,
    const ref_ptr<VkDeviceHandle>& logical_device) {
  IREE_TRACE_SCOPE0("VmaAllocator::Create");

  const auto& syms = logical_device->syms();
  VmaVulkanFunctions vulkan_fns;
  vulkan_fns.vkGetPhysicalDeviceProperties =
      syms->vkGetPhysicalDeviceProperties;
  vulkan_fns.vkGetPhysicalDeviceMemoryProperties =
      syms->vkGetPhysicalDeviceMemoryProperties;
  vulkan_fns.vkAllocateMemory = syms->vkAllocateMemory;
  vulkan_fns.vkFreeMemory = syms->vkFreeMemory;
  vulkan_fns.vkMapMemory = syms->vkMapMemory;
  vulkan_fns.vkUnmapMemory = syms->vkUnmapMemory;
  vulkan_fns.vkFlushMappedMemoryRanges = syms->vkFlushMappedMemoryRanges;
  vulkan_fns.vkInvalidateMappedMemoryRanges =
      syms->vkInvalidateMappedMemoryRanges;
  vulkan_fns.vkBindBufferMemory = syms->vkBindBufferMemory;
  vulkan_fns.vkBindImageMemory = syms->vkBindImageMemory;
  vulkan_fns.vkGetBufferMemoryRequirements =
      syms->vkGetBufferMemoryRequirements;
  vulkan_fns.vkGetImageMemoryRequirements = syms->vkGetImageMemoryRequirements;
  vulkan_fns.vkCreateBuffer = syms->vkCreateBuffer;
  vulkan_fns.vkDestroyBuffer = syms->vkDestroyBuffer;
  vulkan_fns.vkCreateImage = syms->vkCreateImage;
  vulkan_fns.vkDestroyImage = syms->vkDestroyImage;
  vulkan_fns.vkCmdCopyBuffer = syms->vkCmdCopyBuffer;

  VmaRecordSettings record_settings;
#if VMA_RECORDING_ENABLED
  record_settings.flags = absl::GetFlag(FLAGS_vma_recording_flush_after_call)
                              ? VMA_RECORD_FLUSH_AFTER_CALL_BIT
                              : 0;
  record_settings.pFilePath = absl::GetFlag(FLAGS_vma_recording_file).c_str();
#else
  record_settings.flags = 0;
  record_settings.pFilePath = nullptr;
#endif  // VMA_RECORDING_ENABLED

  VmaAllocatorCreateInfo create_info{};
  create_info.flags = 0;
  create_info.physicalDevice = physical_device;
  create_info.device = *logical_device;
  create_info.preferredLargeHeapBlockSize = 64 * 1024 * 1024;
  create_info.pAllocationCallbacks = logical_device->allocator();
  create_info.pDeviceMemoryCallbacks = nullptr;
  create_info.frameInUseCount = 0;
  create_info.pHeapSizeLimit = nullptr;
  create_info.pVulkanFunctions = &vulkan_fns;
  create_info.pRecordSettings = &record_settings;
  ::VmaAllocator vma = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(vmaCreateAllocator(&create_info, &vma));

  auto allocator =
      absl::WrapUnique(new VmaAllocator(physical_device, logical_device, vma));
  // TODO(benvanik): query memory properties/types.
  return allocator;
}

VmaAllocator::VmaAllocator(VkPhysicalDevice physical_device,
                           const ref_ptr<VkDeviceHandle>& logical_device,
                           ::VmaAllocator vma)
    : physical_device_(physical_device),
      logical_device_(add_ref(logical_device)),
      vma_(vma) {}

VmaAllocator::~VmaAllocator() {
  IREE_TRACE_SCOPE0("VmaAllocator::dtor");
  vmaDestroyAllocator(vma_);
}

bool VmaAllocator::CanUseBufferLike(Allocator* source_allocator,
                                    MemoryTypeBitfield memory_type,
                                    BufferUsageBitfield buffer_usage,
                                    BufferUsageBitfield intended_usage) const {
  // TODO(benvanik): ensure there is a memory type that can satisfy the request.
  return source_allocator == this;
}

bool VmaAllocator::CanAllocate(MemoryTypeBitfield memory_type,
                               BufferUsageBitfield buffer_usage,
                               size_t allocation_size) const {
  // TODO(benvnik): ensure there is a memory type that can satisfy the request.
  return true;
}

Status VmaAllocator::MakeCompatible(MemoryTypeBitfield* memory_type,
                                    BufferUsageBitfield* buffer_usage) const {
  // TODO(benvanik): mutate to match supported memory types.
  return OkStatus();
}

StatusOr<ref_ptr<VmaBuffer>> VmaAllocator::AllocateInternal(
    MemoryTypeBitfield memory_type, BufferUsageBitfield buffer_usage,
    MemoryAccessBitfield allowed_access, size_t allocation_size,
    VmaAllocationCreateFlags flags) {
  IREE_TRACE_SCOPE0("VmaAllocator::AllocateInternal");

  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (allocation_size == 0) allocation_size = 4;

  VkBufferCreateInfo buffer_create_info;
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = nullptr;
  buffer_create_info.flags = 0;
  buffer_create_info.size = allocation_size;
  buffer_create_info.usage = 0;
  if (AllBitsSet(buffer_usage, BufferUsage::kTransfer)) {
    buffer_create_info.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
  if (AllBitsSet(buffer_usage, BufferUsage::kDispatch)) {
    buffer_create_info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  }
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_create_info.queueFamilyIndexCount = 0;
  buffer_create_info.pQueueFamilyIndices = nullptr;

  VmaAllocationCreateInfo allocation_create_info;
  allocation_create_info.flags = flags;
  allocation_create_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
  allocation_create_info.requiredFlags = 0;
  allocation_create_info.preferredFlags = 0;
  allocation_create_info.memoryTypeBits = 0;  // Automatic selection.
  allocation_create_info.pool = VK_NULL_HANDLE;
  allocation_create_info.pUserData = nullptr;
  if (AllBitsSet(memory_type, MemoryType::kDeviceLocal)) {
    if (AllBitsSet(memory_type, MemoryType::kHostVisible)) {
      // Device-local, host-visible.
      allocation_create_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
      allocation_create_info.preferredFlags |=
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else {
      // Device-local only.
      allocation_create_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
      allocation_create_info.requiredFlags |=
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
  } else {
    if (AllBitsSet(memory_type, MemoryType::kDeviceVisible)) {
      // Host-local, device-visible.
      allocation_create_info.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
    } else {
      // Host-local only.
      allocation_create_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    }
  }
  if (AllBitsSet(memory_type, MemoryType::kHostCached)) {
    allocation_create_info.requiredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  if (AllBitsSet(memory_type, MemoryType::kHostCoherent)) {
    allocation_create_info.requiredFlags |=
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }
  if (AllBitsSet(memory_type, MemoryType::kTransient)) {
    allocation_create_info.preferredFlags |=
        VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
  }
  if (AllBitsSet(buffer_usage, BufferUsage::kMapping)) {
    allocation_create_info.requiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  }

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VmaAllocationInfo allocation_info;
  VK_RETURN_IF_ERROR(vmaCreateBuffer(vma_, &buffer_create_info,
                                     &allocation_create_info, &buffer,
                                     &allocation, &allocation_info));

  return make_ref<VmaBuffer>(this, memory_type, allowed_access, buffer_usage,
                             allocation_size, 0, allocation_size, buffer,
                             allocation, allocation_info);
}

StatusOr<ref_ptr<Buffer>> VmaAllocator::Allocate(
    MemoryTypeBitfield memory_type, BufferUsageBitfield buffer_usage,
    size_t allocation_size) {
  IREE_TRACE_SCOPE0("VmaAllocator::Allocate");
  return AllocateInternal(memory_type, buffer_usage, MemoryAccess::kAll,
                          allocation_size, /*flags=*/0);
}

StatusOr<ref_ptr<Buffer>> VmaAllocator::AllocateConstant(
    BufferUsageBitfield buffer_usage, ref_ptr<Buffer> source_buffer) {
  IREE_TRACE_SCOPE0("VmaAllocator::AllocateConstant");
  // TODO(benvanik): import memory to avoid the copy.
  IREE_ASSIGN_OR_RETURN(
      auto buffer,
      AllocateInternal(MemoryType::kDeviceLocal | MemoryType::kHostVisible,
                       buffer_usage,
                       MemoryAccess::kRead | MemoryAccess::kDiscardWrite,
                       source_buffer->byte_length(),
                       /*flags=*/0));
  IREE_RETURN_IF_ERROR(
      buffer->CopyData(0, source_buffer.get(), 0, kWholeBuffer));
  buffer->set_allowed_access(MemoryAccess::kRead);
  return buffer;
}

StatusOr<ref_ptr<Buffer>> VmaAllocator::WrapMutable(
    MemoryTypeBitfield memory_type, MemoryAccessBitfield allowed_access,
    BufferUsageBitfield buffer_usage, void* data, size_t data_length) {
  IREE_TRACE_SCOPE0("VmaAllocator::WrapMutable");
  // TODO(benvanik): import memory.
  return UnimplementedErrorBuilder(IREE_LOC)
         << "Wrapping host memory is not yet implemented";
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
