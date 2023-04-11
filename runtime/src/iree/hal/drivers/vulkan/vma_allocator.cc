// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/vma_allocator.h"

#include <cstddef>
#include <cstring>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"
#include "iree/hal/drivers/vulkan/vma_buffer.h"

using namespace iree::hal::vulkan;

//===----------------------------------------------------------------------===//
// Memory types
//===----------------------------------------------------------------------===//

// Set of memory type indices into VkPhysicalDeviceMemoryProperties.
// The set is bucketed by usage semantics to allow us to quickly map an
// allocation request to an underlying memory type. Some buckets may point to
// the same index - for example, on a CPU or integrated GPU all of them may
// point at the same memory space.
//
// Major categories:
// - Dispatch
//   High-bandwidth device-local memory where we want to ensure that all
//   accesses don't require going over a slow bus (PCI/etc).
// - Bulk transfer
//   Low-bandwidth often host-local or host-visible memory for
//   uploading/downloading large buffers, usually backed by system memory.
//   Not expected to be usable by dispatches and just used for staging.
// - Staging transfer
//   High-bandwidth device-local memory that is also host-visible for
//   low-latency staging. These are generally small buffers used by dispatches
//   (like uniform buffers) as the amount of memory available can be very
//   limited (~256MB). Because of the device limits we only use these for
//   TRANSIENT allocations that are used by dispatches.
typedef union {
  struct {
    // Preferred memory type for device-local dispatch operations.
    // This memory _may_ be host visible, though we try to select the most
    // exclusive device local memory when available.
    int dispatch_idx;

    // Preferred memory type for bulk uploads (host->device).
    // These may be slow to access from dispatches or not possible at all, but
    // generally have the entire system memory available for storage.
    int bulk_upload_idx;
    // Preferred memory type for bulk downloads (device->host).
    int bulk_download_idx;

    // Preferred memory type for staging uploads (host->device).
    int staging_upload_idx;
    // Preferred memory type for staging downloads (device->host).
    int staging_download_idx;
  };
  int indices[5];
} iree_hal_vulkan_memory_types_t;

// Returns the total unique memory types.
static int iree_hal_vulkan_memory_types_unique_count(
    const iree_hal_vulkan_memory_types_t* memory_types) {
  uint32_t indices = 0;
  for (size_t i = 0; i < IREE_ARRAYSIZE(memory_types->indices); ++i) {
    indices |= 1u << memory_types->indices[i];
  }
  return iree_math_count_ones_u32(indices);
}

// Returns true if the memory type at |type_idx| is in a device-local heap.
static bool iree_hal_vulkan_is_heap_device_local(
    const VkPhysicalDeviceMemoryProperties* memory_props, uint32_t type_idx) {
  const uint32_t heap_idx = memory_props->memoryTypes[type_idx].heapIndex;
  return iree_all_bits_set(memory_props->memoryHeaps[heap_idx].flags,
                           VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
}

// Returns true if the memory type is not usable by us (today).
static bool iree_hal_vulkan_is_memory_type_usable(VkMemoryPropertyFlags flags) {
  return !iree_all_bits_set(flags, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) &&
         !iree_all_bits_set(flags, VK_MEMORY_PROPERTY_PROTECTED_BIT);
}

static void iree_hal_vulkan_populate_dispatch_memory_types(
    const VkPhysicalDeviceProperties* device_props,
    const VkPhysicalDeviceMemoryProperties* memory_props,
    iree_hal_vulkan_memory_types_t* out_types) {
  int least_bits_count = 0;
  int least_bits_idx = -1;
  for (uint32_t i = 0; i < memory_props->memoryTypeCount; ++i) {
    VkMemoryPropertyFlags flags = memory_props->memoryTypes[i].propertyFlags;
    if (!iree_hal_vulkan_is_heap_device_local(memory_props, i) ||
        !iree_hal_vulkan_is_memory_type_usable(flags)) {
      // Only want device-local memory that is usable for storage buffers.
      continue;
    }
    // Prefer the type that is device-local and has as few other bits set as
    // possible (host-visible/etc). On integrated systems we may not have any
    // type that is purely device-local but still want to ensure we pick
    // uncached over cached.
    int bit_count = iree_math_count_ones_u32(flags);
    if (least_bits_idx == -1) {
      least_bits_count = bit_count;
      least_bits_idx = (int)i;
    } else if (bit_count < least_bits_count) {
      least_bits_count = bit_count;
      least_bits_idx = (int)i;
    }
  }
  out_types->dispatch_idx = least_bits_idx;
}

static void iree_hal_vulkan_find_transfer_memory_types(
    const VkPhysicalDeviceProperties* device_props,
    const VkPhysicalDeviceMemoryProperties* memory_props,
    VkMemoryPropertyFlags include_flags, VkMemoryPropertyFlags exclude_flags,
    int* out_upload_idx, int* out_download_idx) {
  int cached_idx = -1;
  int uncached_idx = -1;
  int visible_idx = -1;
  for (uint32_t i = 0; i < memory_props->memoryTypeCount; ++i) {
    VkMemoryPropertyFlags flags = memory_props->memoryTypes[i].propertyFlags;
    if (!iree_all_bits_set(flags, include_flags) ||
        iree_any_bit_set(flags, exclude_flags)) {
      // Caller allows/disallows certain flags.
      continue;
    } else if (!iree_hal_vulkan_is_memory_type_usable(flags)) {
      // Only want memory that is usable for storage buffers.
      continue;
    } else if (!iree_all_bits_set(flags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
      // Must be host-visible for transfers.
      continue;
    }
    if (visible_idx == -1) visible_idx = i;
    if (iree_all_bits_set(flags, VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
      if (cached_idx == -1) cached_idx = i;
    } else {
      if (uncached_idx == -1) uncached_idx = i;
    }
  }
  // Prefer uncached for uploads to enable write-through to the device.
  *out_upload_idx = uncached_idx != -1 ? uncached_idx : visible_idx;
  // Prefer cached for downloads to enable prefetching/read caching.
  *out_download_idx = cached_idx != -1 ? cached_idx : visible_idx;
}

static void iree_hal_vulkan_populate_transfer_memory_types(
    const VkPhysicalDeviceProperties* device_props,
    const VkPhysicalDeviceMemoryProperties* memory_props,
    iree_hal_vulkan_memory_types_t* out_types) {
  int host_local_upload_idx = -1;
  int host_local_download_idx = -1;
  iree_hal_vulkan_find_transfer_memory_types(
      device_props, memory_props, /*include_flags=*/0,
      /*exclude_flags=*/VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      &host_local_upload_idx, &host_local_download_idx);
  int device_local_upload_idx = -1;
  int device_local_download_idx = -1;
  iree_hal_vulkan_find_transfer_memory_types(
      device_props, memory_props,
      /*include_flags=*/VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      /*exclude_flags=*/0, &device_local_upload_idx,
      &device_local_download_idx);

  // For bulk try first to select host-local memory.
  // In case that fails we will use device-local memory; common on integrated.
  out_types->bulk_upload_idx = host_local_upload_idx != -1
                                   ? host_local_upload_idx
                                   : device_local_upload_idx;
  out_types->bulk_download_idx = host_local_download_idx != -1
                                     ? host_local_download_idx
                                     : device_local_download_idx;

  // Always use device-local for staging if we have it. This is usually PCI-E
  // BAR/page-locked memory on discrete devices while it may just be host
  // allocations with special caching flags on integrated ones.
  out_types->staging_upload_idx = device_local_upload_idx != -1
                                      ? device_local_upload_idx
                                      : host_local_upload_idx;
  out_types->staging_download_idx = device_local_download_idx != -1
                                        ? device_local_download_idx
                                        : host_local_download_idx;
}

// Queries the underlying Vulkan implementation to decide which memory type
// should be used for particular operations.
//
// This is a train-wreck of a decision space and definitely wrong in some cases.
// The only thing we can do is try to be less wrong than RNG.
//
// Common Android:
//   - DEVICE_LOCAL (dispatch)
//   - DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (upload)
//   - DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED (download)
//   - DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (everything)
// Samsung Android: 🤡
// (https://vulkan.gpuinfo.org/displayreport.php?id=14487#memory)
//
// Swiftshader/Intel (VK_PHYSICAL_DEVICE_TYPE_CPU):
//   - DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (everything)
//
// iOS via MoltenVK (VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU):
//   - DEVICE_LOCAL (dispatch)
//   - DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED (everything)
//
// NVIDIA Tegra-like (VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU):
//   - DEVICE_LOCAL (dispatch)
//   - DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (upload)
//   - DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED (everything)
//
// NVIDIA/AMD discrete (VK_PHYSICAL_DEVICE_TYPE_GPU):
//   - DEVICE_LOCAL (dispatch)
//   - DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (staging upload)
//   - DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED (staging download)
//   - HOST_VISIBLE | HOST_COHERENT (upload)
//   - HOST_VISIBLE | HOST_CACHED (download)
static iree_status_t iree_hal_vulkan_populate_memory_types(
    const VkPhysicalDeviceProperties* device_props,
    const VkPhysicalDeviceMemoryProperties* memory_props,
    iree_hal_vulkan_memory_types_t* out_memory_types) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOT_FOUND sentinel.
  for (size_t i = 0; i < IREE_ARRAYSIZE(out_memory_types->indices); ++i) {
    out_memory_types->indices[i] = -1;
  }

  // Find the memory type that is most device-local.
  // We try to satisfy all device access requests with this type.
  iree_hal_vulkan_populate_dispatch_memory_types(device_props, memory_props,
                                                 out_memory_types);

  // Find the memory types for upload/download.
  iree_hal_vulkan_populate_transfer_memory_types(device_props, memory_props,
                                                 out_memory_types);

  // Because this is all bananas we trace out what indices we chose; this will
  // let us correlate the memory types with vulkan-info and see if we got the
  // "right" ones.
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "dispatch:");
  IREE_TRACE_ZONE_APPEND_VALUE(z0, out_memory_types->dispatch_idx);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "bulk-upload:");
  IREE_TRACE_ZONE_APPEND_VALUE(z0, out_memory_types->bulk_upload_idx);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "bulk-download:");
  IREE_TRACE_ZONE_APPEND_VALUE(z0, out_memory_types->bulk_download_idx);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "staging-upload:");
  IREE_TRACE_ZONE_APPEND_VALUE(z0, out_memory_types->staging_upload_idx);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "staging-download:");
  IREE_TRACE_ZONE_APPEND_VALUE(z0, out_memory_types->staging_download_idx);
  IREE_TRACE_ZONE_END(z0);

  // Check to make sure all memory types were found. If we didn't find any
  // special staging transfer memory we reuse bulk memory.
  if (out_memory_types->dispatch_idx == -1) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "dispatch-compatible memory type not found");
  } else if (out_memory_types->bulk_upload_idx == -1 ||
             out_memory_types->bulk_download_idx == -1 ||
             out_memory_types->staging_upload_idx == -1 ||
             out_memory_types->staging_download_idx == -1) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "transfer-compatible memory types not found");
  }
  return iree_ok_status();
}

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

// Maps a Vulkan device memory type enum to an allocator heap structure.
static void iree_hal_vulkan_map_memory_type_to_heap(
    const VkPhysicalDeviceMemoryProperties* memory_props, int type_idx,
    iree_device_size_t max_allocation_size, iree_device_size_t min_alignment,
    iree_hal_allocator_memory_heap_t* out_heap) {
  VkMemoryPropertyFlags flags =
      memory_props->memoryTypes[type_idx].propertyFlags;
  iree_hal_memory_type_t memory_type = 0;
  iree_hal_buffer_usage_t allowed_usage = 0;
  if (iree_all_bits_set(flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    memory_type |= IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    allowed_usage |= IREE_HAL_BUFFER_USAGE_TRANSFER |
                     IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMS |
                     IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                     IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ;
  }
  if (iree_all_bits_set(flags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    allowed_usage |=
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  }
  if (iree_all_bits_set(flags, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  }
  if (iree_all_bits_set(flags, VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_CACHED;
  }
  out_heap->type = memory_type;
  out_heap->allowed_usage = allowed_usage;

  // Some memory heaps have very small limits (like 256MB or less) and it may
  // be less than the maximum allocation size of the API.
  const VkMemoryHeap* memory_heap =
      &memory_props->memoryHeaps[memory_props->memoryTypes[type_idx].heapIndex];
  out_heap->max_allocation_size =
      iree_min(max_allocation_size, memory_heap->size);
  out_heap->min_alignment = min_alignment;
}

static iree_status_t iree_hal_vulkan_vma_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_vulkan_vma_allocator_t* allocator =
      iree_hal_vulkan_vma_allocator_cast(base_allocator);

  // TODO(benvanik): when not using VMA we'll want to cache these ourselves.
  const VkPhysicalDeviceProperties* device_props = NULL;
  vmaGetPhysicalDeviceProperties(allocator->vma, &device_props);
  const VkPhysicalDeviceMemoryProperties* memory_props = NULL;
  vmaGetMemoryProperties(allocator->vma, &memory_props);

  const iree_device_size_t max_allocation_size =
      device_props->limits.maxStorageBufferRange;
  const iree_device_size_t min_alignment =
      iree_max(16, device_props->limits.minStorageBufferOffsetAlignment);

  const iree_hal_vulkan_memory_types_t* memory_types = &allocator->memory_types;
  iree_host_size_t count =
      iree_hal_vulkan_memory_types_unique_count(memory_types);
  if (capacity >= count) {
    uint32_t has_idx = 0;
    iree_host_size_t i = 0;
    if (!(has_idx & (1u << memory_types->dispatch_idx))) {
      has_idx |= 1u << memory_types->dispatch_idx;
      iree_hal_vulkan_map_memory_type_to_heap(
          memory_props, memory_types->dispatch_idx, max_allocation_size,
          min_alignment, &heaps[i++]);
    }
    if (!(has_idx & (1u << memory_types->bulk_upload_idx))) {
      has_idx |= 1u << memory_types->bulk_upload_idx;
      iree_hal_vulkan_map_memory_type_to_heap(
          memory_props, memory_types->bulk_upload_idx, max_allocation_size,
          min_alignment, &heaps[i++]);
    }
    if (!(has_idx & (1u << memory_types->bulk_download_idx))) {
      has_idx |= 1u << memory_types->bulk_download_idx;
      iree_hal_vulkan_map_memory_type_to_heap(
          memory_props, memory_types->bulk_download_idx, max_allocation_size,
          min_alignment, &heaps[i++]);
    }
    if (!(has_idx & (1u << memory_types->staging_upload_idx))) {
      has_idx |= 1u << memory_types->staging_upload_idx;
      iree_hal_vulkan_map_memory_type_to_heap(
          memory_props, memory_types->staging_upload_idx, max_allocation_size,
          min_alignment, &heaps[i++]);
    }
    if (!(has_idx & (1u << memory_types->staging_download_idx))) {
      has_idx |= 1u << memory_types->staging_download_idx;
      iree_hal_vulkan_map_memory_type_to_heap(
          memory_props, memory_types->staging_download_idx, max_allocation_size,
          min_alignment, &heaps[i++]);
    }
    IREE_ASSERT(i == count);
  }

  if (out_count) *out_count = count;
  IREE_TRACE_ZONE_APPEND_VALUE(z0, count);
  IREE_TRACE_ZONE_END(z0);
  if (capacity < count) {
    // NOTE: lightweight as this is hit in normal pre-sizing usage.
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  return iree_ok_status();
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
