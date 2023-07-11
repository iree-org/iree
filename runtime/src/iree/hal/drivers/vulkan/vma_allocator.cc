// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/vma_allocator.h"

#include <cstddef>
#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/base_buffer.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"
#include "iree/hal/drivers/vulkan/vma_impl.h"

using namespace iree::hal::vulkan;

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_VULKAN_VMA_ALLOCATOR_ID = "Vulkan/VMA";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_vma_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_vma_buffer_t {
  iree_hal_vulkan_base_buffer_t base;
  VmaAllocator vma;
  VmaAllocation allocation;
  VmaAllocationInfo allocation_info;
} iree_hal_vulkan_vma_buffer_t;

namespace {
extern const iree_hal_buffer_vtable_t iree_hal_vulkan_vma_buffer_vtable;
}  // namespace

static iree_hal_vulkan_vma_buffer_t* iree_hal_vulkan_vma_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_vma_buffer_vtable);
  return (iree_hal_vulkan_vma_buffer_t*)base_value;
}

iree_status_t iree_hal_vulkan_vma_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    VmaAllocator vma, VkBuffer handle, VmaAllocation allocation,
    VmaAllocationInfo allocation_info, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(vma);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(allocation);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);

  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(allocator);
  iree_hal_vulkan_vma_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(
        host_allocator, allocator, &buffer->base.base, allocation_size,
        byte_offset, byte_length, memory_type, allowed_access, allowed_usage,
        &iree_hal_vulkan_vma_buffer_vtable, &buffer->base.base);
    buffer->base.device_memory = allocation_info.deviceMemory;
    buffer->base.handle = handle;
    buffer->vma = vma;
    buffer->allocation = allocation;
    buffer->allocation_info = allocation_info;

    // TODO(benvanik): set debug name instead and use the
    //     VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT flag.
    vmaSetAllocationUserData(buffer->vma, buffer->allocation, buffer);

    IREE_TRACE_ALLOC_NAMED(IREE_HAL_VULKAN_VMA_ALLOCATOR_ID,
                           (void*)buffer->base.handle, byte_length);

    *out_buffer = &buffer->base.base;
  } else {
    vmaDestroyBuffer(vma, handle, allocation);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_vma_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, (int64_t)iree_hal_buffer_allocation_size(base_buffer));

  IREE_TRACE_FREE_NAMED(IREE_HAL_VULKAN_VMA_ALLOCATOR_ID,
                        (void*)buffer->base.handle);

  vmaDestroyBuffer(buffer->vma, buffer->base.handle, buffer->allocation);
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_vma_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);

  // TODO(benvanik): add upload/download for unmapped buffers.
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_usage(iree_hal_buffer_allowed_usage(base_buffer),
                                     IREE_HAL_BUFFER_USAGE_MAPPING));

  uint8_t* data_ptr = nullptr;
  VK_RETURN_IF_ERROR(
      vmaMapMemory(buffer->vma, buffer->allocation, (void**)&data_ptr),
      "vmaMapMemory");
  mapping->contents =
      iree_make_byte_span(data_ptr + local_byte_offset, local_byte_length);

  // If we mapped for discard scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for
  // heap buffers we could reallocate them such that ASAN yells, but that
  // would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(mapping->contents.data, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_vma_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  vmaUnmapMemory(buffer->vma, buffer->allocation);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_vma_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  VK_RETURN_IF_ERROR(
      vmaInvalidateAllocation(buffer->vma, buffer->allocation,
                              local_byte_offset, local_byte_length),
      "vmaInvalidateAllocation");
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_vma_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  VK_RETURN_IF_ERROR(vmaFlushAllocation(buffer->vma, buffer->allocation,
                                        local_byte_offset, local_byte_length),
                     "vmaFlushAllocation");
  return iree_ok_status();
}

namespace {
const iree_hal_buffer_vtable_t iree_hal_vulkan_vma_buffer_vtable = {
    /*.recycle=*/iree_hal_buffer_recycle,
    /*.destroy=*/iree_hal_vulkan_vma_buffer_destroy,
    /*.map_range=*/iree_hal_vulkan_vma_buffer_map_range,
    /*.unmap_range=*/iree_hal_vulkan_vma_buffer_unmap_range,
    /*.invalidate_range=*/iree_hal_vulkan_vma_buffer_invalidate_range,
    /*.flush_range=*/iree_hal_vulkan_vma_buffer_flush_range,
};
}  // namespace

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_vma_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_vma_allocator_t {
  iree_hal_resource_t resource;
  iree_hal_device_t* device;  // unretained to avoid cycles
  iree_allocator_t host_allocator;
  VmaAllocator vma;

  // Used to quickly look up the memory type index used for a particular usage.
  iree_hal_vulkan_memory_types_t memory_types;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_vulkan_vma_allocator_t;

namespace {
extern const iree_hal_allocator_vtable_t iree_hal_vulkan_vma_allocator_vtable;
}  // namespace

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
  const VkPhysicalDeviceMemoryProperties* memory_props = NULL;
  vmaGetMemoryProperties(allocator->vma, &memory_props);
  VkMemoryPropertyFlags flags =
      memory_props->memoryTypes[memory_type_ordinal].propertyFlags;
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
    const iree_hal_vulkan_device_options_t* options, VkInstance instance,
    VkPhysicalDevice physical_device, VkDeviceHandle* logical_device,
    iree_hal_device_t* device, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(device);
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
  allocator->device = device;

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
  create_info.preferredLargeHeapBlockSize = options->large_heap_block_size;
  create_info.pAllocationCallbacks = logical_device->allocator();
  create_info.pDeviceMemoryCallbacks = &device_memory_callbacks;
  create_info.pHeapSizeLimit = NULL;
  create_info.pVulkanFunctions = &vulkan_fns;
  VmaAllocator vma = VK_NULL_HANDLE;
  iree_status_t status = VK_RESULT_TO_STATUS(
      vmaCreateAllocator(&create_info, &vma), "vmaCreateAllocator");

  if (iree_status_is_ok(status)) {
    allocator->vma = vma;

    // TODO(benvanik): when not using VMA we'll want to cache these ourselves.
    const VkPhysicalDeviceProperties* device_props = NULL;
    vmaGetPhysicalDeviceProperties(allocator->vma, &device_props);
    const VkPhysicalDeviceMemoryProperties* memory_props = NULL;
    vmaGetMemoryProperties(allocator->vma, &memory_props);
    status = iree_hal_vulkan_populate_memory_types(device_props, memory_props,
                                                   &allocator->memory_types);
  }

  if (iree_status_is_ok(status)) {
    *out_allocator = (iree_hal_allocator_t*)allocator;
  } else {
    vmaDestroyAllocator(vma);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_vma_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_vulkan_vma_allocator_t* allocator =
      iree_hal_vulkan_vma_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  vmaDestroyAllocator(allocator->vma);
  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_vulkan_vma_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_vulkan_vma_allocator_t* allocator =
      (iree_hal_vulkan_vma_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_vulkan_vma_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_vulkan_vma_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_vulkan_vma_allocator_t* allocator =
        iree_hal_vulkan_vma_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_status_t iree_hal_vulkan_vma_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  iree_hal_vulkan_vma_allocator_t* allocator =
      iree_hal_vulkan_vma_allocator_cast(base_allocator);

  // TODO(benvanik): when not using VMA we'll want to cache these ourselves.
  const VkPhysicalDeviceProperties* device_props = NULL;
  vmaGetPhysicalDeviceProperties(allocator->vma, &device_props);
  const VkPhysicalDeviceMemoryProperties* memory_props = NULL;
  vmaGetMemoryProperties(allocator->vma, &memory_props);

  const iree_hal_vulkan_memory_types_t* memory_types = &allocator->memory_types;

  return iree_hal_vulkan_query_memory_heaps(
      device_props, memory_props, memory_types, capacity, heaps, out_count);
}

static iree_hal_buffer_compatibility_t
iree_hal_vulkan_vma_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  // TODO(benvanik): check to ensure the allocator can serve the memory type.

  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  }

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_any_bit_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  // We are now optimal.
  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;

  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (*allocation_size == 0) *allocation_size = 4;

  // Align allocation sizes to 4 bytes so shaders operating on 32 bit types can
  // act safely even on buffer ranges that are not naturally aligned.
  *allocation_size = iree_host_align(*allocation_size, 4);

  return compatibility;
}

static iree_status_t iree_hal_vulkan_vma_allocator_allocate_internal(
    iree_hal_vulkan_vma_allocator_t* IREE_RESTRICT allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size, iree_const_byte_span_t initial_data,
    VmaAllocationCreateFlags flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  VkBufferCreateInfo buffer_create_info;
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = NULL;
  buffer_create_info.flags = 0;
  buffer_create_info.size = allocation_size;
  buffer_create_info.usage = 0;
  if (iree_all_bits_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    buffer_create_info.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
  if (iree_all_bits_set(params->usage,
                        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
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
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
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
    if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
      // Host-local, device-visible.
      allocation_create_info.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
    } else {
      // Host-local only.
      allocation_create_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    }
  }
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    allocation_create_info.requiredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    allocation_create_info.requiredFlags |=
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }
  if (iree_all_bits_set(params->usage, IREE_HAL_BUFFER_USAGE_MAPPING)) {
    allocation_create_info.requiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  }

  // TODO(benvanik): if on a unified memory system and initial data is present
  // we could set the mapping bit and ensure a much more efficient upload.

  VkBuffer handle = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VmaAllocationInfo allocation_info;
  VK_RETURN_IF_ERROR(vmaCreateBuffer(allocator->vma, &buffer_create_info,
                                     &allocation_create_info, &handle,
                                     &allocation, &allocation_info),
                     "vmaCreateBuffer");

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_vulkan_vma_buffer_wrap(
      (iree_hal_allocator_t*)allocator, params->type, params->access,
      params->usage, allocation_size,
      /*byte_offset=*/0,
      /*byte_length=*/allocation_size, allocator->vma, handle, allocation,
      allocation_info, &buffer);
  if (!iree_status_is_ok(status)) {
    vmaDestroyBuffer(allocator->vma, handle, allocation);
    return status;
  }

  // Copy the initial contents into the buffer. This may require staging.
  if (iree_status_is_ok(status) &&
      !iree_const_byte_span_is_empty(initial_data)) {
    status = iree_hal_device_transfer_range(
        allocator->device,
        iree_hal_make_host_transfer_buffer_span((void*)initial_data.data,
                                                initial_data.data_length),
        0, iree_hal_make_device_transfer_buffer(buffer), 0,
        initial_data.data_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_vma_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size, iree_const_byte_span_t initial_data,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_vulkan_vma_allocator_t* allocator =
      iree_hal_vulkan_vma_allocator_cast(base_allocator);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  if (!iree_all_bits_set(
          iree_hal_vulkan_vma_allocator_query_buffer_compatibility(
              base_allocator, &compat_params, &allocation_size),
          IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
  }

  return iree_hal_vulkan_vma_allocator_allocate_internal(
      allocator, &compat_params, allocation_size, initial_data,
      /*flags=*/0, out_buffer);
}

static void iree_hal_vulkan_vma_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  // VMA does the pooling for us so we don't need anything special.
  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_vulkan_vma_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // TODO(#7242): use VK_EXT_external_memory_host to import memory.
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "importing from external buffers not supported");
}

static iree_status_t iree_hal_vulkan_vma_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "exporting to external buffers not supported");
}

namespace {
const iree_hal_allocator_vtable_t iree_hal_vulkan_vma_allocator_vtable = {
    /*.destroy=*/iree_hal_vulkan_vma_allocator_destroy,
    /*.host_allocator=*/iree_hal_vulkan_vma_allocator_host_allocator,
    /*.trim=*/iree_hal_vulkan_vma_allocator_trim,
    /*.query_statistics=*/iree_hal_vulkan_vma_allocator_query_statistics,
    /*.query_memory_heaps=*/iree_hal_vulkan_vma_allocator_query_memory_heaps,
    /*.query_buffer_compatibility=*/
    iree_hal_vulkan_vma_allocator_query_buffer_compatibility,
    /*.allocate_buffer=*/iree_hal_vulkan_vma_allocator_allocate_buffer,
    /*.deallocate_buffer=*/iree_hal_vulkan_vma_allocator_deallocate_buffer,
    /*.import_buffer=*/iree_hal_vulkan_vma_allocator_import_buffer,
    /*.export_buffer=*/iree_hal_vulkan_vma_allocator_export_buffer,
};
}  // namespace
