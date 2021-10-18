// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/vulkan/vma_allocator.h"

#include <cstddef>
#include <cstring>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/util/ref_ptr.h"
#include "iree/hal/vulkan/vma_buffer.h"

using namespace iree::hal::vulkan;

typedef struct iree_hal_vulkan_vma_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  VmaAllocator vma;

  IREE_STATISTICS(VkPhysicalDeviceMemoryProperties memory_props;)
  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_vulkan_vma_allocator_t;

extern const iree_hal_allocator_vtable_t iree_hal_vulkan_vma_allocator_vtable;

static iree_hal_vulkan_vma_allocator_t* iree_hal_vulkan_vma_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_vma_allocator_vtable);
  return (iree_hal_vulkan_vma_allocator_t*)base_value;
}

#if IREE_STATISTICS_ENABLE

static iree_hal_memory_type_t iree_hal_vulkan_vma_allocator_lookup_memory_type(
    iree_hal_vulkan_vma_allocator_t* allocator, uint32_t memory_type_ordinal) {
  // We could better map the types however today we only use the
  // device/host-local bits.
  VkMemoryPropertyFlags flags =
      allocator->memory_props.memoryTypes[memory_type_ordinal].propertyFlags;
  if (iree_all_bits_set(flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    return IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  } else {
    return IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  }
}

// Callback function called before vkAllocateMemory.
static void VKAPI_PTR iree_hal_vulkan_vma_allocate_callback(
    VmaAllocator VMA_NOT_NULL vma, uint32_t memoryType,
    VkDeviceMemory VMA_NOT_NULL_NON_DISPATCHABLE memory, VkDeviceSize size,
    void* VMA_NULLABLE pUserData) {
  iree_hal_vulkan_vma_allocator_t* allocator =
      (iree_hal_vulkan_vma_allocator_t*)pUserData;
  iree_hal_allocator_statistics_record_alloc(
      &allocator->statistics,
      iree_hal_vulkan_vma_allocator_lookup_memory_type(allocator, memoryType),
      (iree_device_size_t)size);
}

// Callback function called before vkFreeMemory.
static void VKAPI_PTR iree_hal_vulkan_vma_free_callback(
    VmaAllocator VMA_NOT_NULL vma, uint32_t memoryType,
    VkDeviceMemory VMA_NOT_NULL_NON_DISPATCHABLE memory, VkDeviceSize size,
    void* VMA_NULLABLE pUserData) {
  iree_hal_vulkan_vma_allocator_t* allocator =
      (iree_hal_vulkan_vma_allocator_t*)pUserData;
  iree_hal_allocator_statistics_record_free(
      &allocator->statistics,
      iree_hal_vulkan_vma_allocator_lookup_memory_type(allocator, memoryType),
      (iree_device_size_t)size);
}

#endif  // IREE_STATISTICS_ENABLE

iree_status_t iree_hal_vulkan_vma_allocator_create(
    VkInstance instance, VkPhysicalDevice physical_device,
    VkDeviceHandle* logical_device, VmaRecordSettings record_settings,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = logical_device->host_allocator();
  iree_hal_vulkan_vma_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_vulkan_vma_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;

  const auto& syms = logical_device->syms();
  VmaVulkanFunctions vulkan_fns;
  memset(&vulkan_fns, 0, sizeof(vulkan_fns));
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

  VmaDeviceMemoryCallbacks device_memory_callbacks;
  memset(&device_memory_callbacks, 0, sizeof(device_memory_callbacks));
  IREE_STATISTICS({
    device_memory_callbacks.pfnAllocate = iree_hal_vulkan_vma_allocate_callback;
    device_memory_callbacks.pfnFree = iree_hal_vulkan_vma_free_callback;
    device_memory_callbacks.pUserData = allocator;
  });

  VmaAllocatorCreateInfo create_info;
  memset(&create_info, 0, sizeof(create_info));
  create_info.flags = 0;
  create_info.physicalDevice = physical_device;
  create_info.device = *logical_device;
  create_info.instance = instance;
  create_info.preferredLargeHeapBlockSize = 64 * 1024 * 1024;
  create_info.pAllocationCallbacks = logical_device->allocator();
  create_info.pDeviceMemoryCallbacks = &device_memory_callbacks;
  create_info.frameInUseCount = 0;
  create_info.pHeapSizeLimit = NULL;
  create_info.pVulkanFunctions = &vulkan_fns;
  create_info.pRecordSettings = &record_settings;
  VmaAllocator vma = VK_NULL_HANDLE;
  iree_status_t status = VK_RESULT_TO_STATUS(
      vmaCreateAllocator(&create_info, &vma), "vmaCreateAllocator");

  if (iree_status_is_ok(status)) {
    allocator->vma = vma;

    IREE_STATISTICS({
      const VkPhysicalDeviceMemoryProperties* memory_props = NULL;
      vmaGetMemoryProperties(allocator->vma, &memory_props);
      memcpy(&allocator->memory_props, memory_props,
             sizeof(allocator->memory_props));
    });

    *out_allocator = (iree_hal_allocator_t*)allocator;
  } else {
    vmaDestroyAllocator(vma);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_vma_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_vulkan_vma_allocator_t* allocator =
      iree_hal_vulkan_vma_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  vmaDestroyAllocator(allocator->vma);
  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_vulkan_vma_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  iree_hal_vulkan_vma_allocator_t* allocator =
      (iree_hal_vulkan_vma_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static void iree_hal_vulkan_vma_allocator_query_statistics(
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_statistics_t* out_statistics) {
  IREE_STATISTICS({
    iree_hal_vulkan_vma_allocator_t* allocator =
        iree_hal_vulkan_vma_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_hal_buffer_compatibility_t
iree_hal_vulkan_vma_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_usage_t intended_usage,
    iree_device_size_t allocation_size) {
  // TODO(benvanik): check to ensure the allocator can serve the memory type.

  // Disallow usage not permitted by the buffer itself. Since we then use this
  // to determine compatibility below we'll naturally set the right compat flags
  // based on what's both allowed and intended.
  intended_usage &= allowed_usage;

  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_all_bits_set(intended_usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
    }
    if (iree_all_bits_set(intended_usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  return compatibility;
}

static iree_status_t iree_hal_vulkan_vma_allocator_make_compatible(
    iree_hal_memory_type_t* memory_type,
    iree_hal_memory_access_t* allowed_access,
    iree_hal_buffer_usage_t* allowed_usage) {
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_vma_allocator_allocate_internal(
    iree_hal_vulkan_vma_allocator_t* allocator,
    iree_hal_memory_type_t memory_type, iree_hal_buffer_usage_t allowed_usage,
    iree_hal_memory_access_t allowed_access, size_t allocation_size,
    VmaAllocationCreateFlags flags, iree_hal_buffer_t** out_buffer) {
  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (allocation_size == 0) allocation_size = 4;

  VkBufferCreateInfo buffer_create_info;
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = NULL;
  buffer_create_info.flags = 0;
  buffer_create_info.size = allocation_size;
  buffer_create_info.usage = 0;
  if (iree_all_bits_set(allowed_usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    buffer_create_info.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
  if (iree_all_bits_set(allowed_usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
    buffer_create_info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  }
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_create_info.queueFamilyIndexCount = 0;
  buffer_create_info.pQueueFamilyIndices = NULL;

  VmaAllocationCreateInfo allocation_create_info;
  allocation_create_info.flags = flags;
  allocation_create_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
  allocation_create_info.requiredFlags = 0;
  allocation_create_info.preferredFlags = 0;
  allocation_create_info.memoryTypeBits = 0;  // Automatic selection.
  allocation_create_info.pool = VK_NULL_HANDLE;
  allocation_create_info.pUserData = NULL;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
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
    if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
      // Host-local, device-visible.
      allocation_create_info.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
    } else {
      // Host-local only.
      allocation_create_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    }
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    allocation_create_info.requiredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    allocation_create_info.requiredFlags |=
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_TRANSIENT)) {
    allocation_create_info.preferredFlags |=
        VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
  }
  if (iree_all_bits_set(allowed_usage, IREE_HAL_BUFFER_USAGE_MAPPING)) {
    allocation_create_info.requiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  }

  VkBuffer handle = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VmaAllocationInfo allocation_info;
  VK_RETURN_IF_ERROR(vmaCreateBuffer(allocator->vma, &buffer_create_info,
                                     &allocation_create_info, &handle,
                                     &allocation, &allocation_info),
                     "vmaCreateBuffer");

  return iree_hal_vulkan_vma_buffer_wrap(
      (iree_hal_allocator_t*)allocator, memory_type, allowed_access,
      allowed_usage, allocation_size,
      /*byte_offset=*/0,
      /*byte_length=*/allocation_size, allocator->vma, handle, allocation,
      allocation_info, out_buffer);
}

static iree_status_t iree_hal_vulkan_vma_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_vulkan_vma_allocator_t* allocator =
      iree_hal_vulkan_vma_allocator_cast(base_allocator);

  // Coerce options into those required for use by VMA.
  iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_ALL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_vma_allocator_make_compatible(
      &memory_type, &allowed_access, &allowed_usage));

  return iree_hal_vulkan_vma_allocator_allocate_internal(
      allocator, memory_type, allowed_usage, allowed_access, allocation_size,
      /*flags=*/0, out_buffer);
}

static iree_status_t iree_hal_vulkan_vma_allocator_wrap_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer) {
  // TODO(#7242): use VK_EXT_external_memory_host to import memory.
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "wrapping of external buffers not supported");
}

const iree_hal_allocator_vtable_t iree_hal_vulkan_vma_allocator_vtable = {
    /*.destroy=*/iree_hal_vulkan_vma_allocator_destroy,
    /*.host_allocator=*/iree_hal_vulkan_vma_allocator_host_allocator,
    /*.query_statistics=*/iree_hal_vulkan_vma_allocator_query_statistics,
    /*.query_buffer_compatibility=*/
    iree_hal_vulkan_vma_allocator_query_buffer_compatibility,
    /*.allocate_buffer=*/iree_hal_vulkan_vma_allocator_allocate_buffer,
    /*.wrap_buffer=*/iree_hal_vulkan_vma_allocator_wrap_buffer,
};
