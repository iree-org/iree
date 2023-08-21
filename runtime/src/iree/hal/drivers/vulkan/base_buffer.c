// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/base_buffer.h"

#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// Memory types
//===----------------------------------------------------------------------===//

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

iree_status_t iree_hal_vulkan_find_memory_type(
    const VkPhysicalDeviceProperties* device_props,
    const VkPhysicalDeviceMemoryProperties* memory_props,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    uint32_t allowed_type_indices, uint32_t* out_memory_type_index) {
  *out_memory_type_index = 0;

  iree_hal_memory_type_t requested_type = params->type;
  if (device_props->deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
    // Integrated GPUs have tiny device local heaps commonly used for
    // framebuffers and other bounded resources. We don't currently try to use
    // them but could for very small transients.
    if (iree_all_bits_set(requested_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
      requested_type &= ~IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
      requested_type |= IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    }
  }

  VkMemoryPropertyFlags require_flags = 0;
  VkMemoryPropertyFlags prefer_flags = 0;
  if (iree_all_bits_set(requested_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    if (iree_all_bits_set(requested_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      // Device-local, host-visible.
      require_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      prefer_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else {
      // Device-local only.
      require_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
  } else {
    if (iree_all_bits_set(requested_type,
                          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
      // Host-local, device-visible.
      require_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    } else {
      // Host-local only.
      require_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    }
  }
  if (iree_all_bits_set(requested_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    require_flags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  if (iree_all_bits_set(requested_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    require_flags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }
  if (iree_any_bit_set(requested_type,
                       IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                           IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT)) {
    require_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  }

  int most_bits_count = 0;
  int most_bits_idx = -1;
  for (uint32_t i = 0; i < memory_props->memoryTypeCount; ++i) {
    VkMemoryPropertyFlags flags = memory_props->memoryTypes[i].propertyFlags;
    if (!iree_all_bits_set(flags, require_flags) ||
        !iree_hal_vulkan_is_memory_type_usable(flags) ||
        !iree_all_bits_set(allowed_type_indices, 1u << i)) {
      // Excluded (required bits missing or memory type is not usable).
      continue;
    }
    // When all required bits are satisfied try to find the memory type that
    // has the most preferred bits set.
    int bit_count = iree_math_count_ones_u32(flags & prefer_flags);
    if (most_bits_idx == -1) {
      most_bits_count = bit_count;
      most_bits_idx = (int)i;
    } else if (bit_count > most_bits_count) {
      most_bits_count = bit_count;
      most_bits_idx = (int)i;
    }
  }
  if (most_bits_idx == -1) {
    // No valid memory type found.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "no memory type available that satisfies the required flags");
  }

  *out_memory_type_index = (uint32_t)most_bits_idx;
  return iree_ok_status();
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
iree_status_t iree_hal_vulkan_populate_memory_types(
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
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, out_memory_types->dispatch_idx);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "bulk-upload:");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, out_memory_types->bulk_upload_idx);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "bulk-download:");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, out_memory_types->bulk_download_idx);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "staging-upload:");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, out_memory_types->staging_upload_idx);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "staging-download:");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, out_memory_types->staging_download_idx);
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

iree_status_t iree_hal_vulkan_query_memory_heaps(
    const VkPhysicalDeviceProperties* device_props,
    const VkPhysicalDeviceMemoryProperties* memory_props,
    const iree_hal_vulkan_memory_types_t* memory_types,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_device_size_t max_allocation_size =
      device_props->limits.maxStorageBufferRange;
  const iree_device_size_t min_alignment =
      iree_max(16, device_props->limits.minStorageBufferOffsetAlignment);

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
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);
  IREE_TRACE_ZONE_END(z0);
  if (capacity < count) {
    // NOTE: lightweight as this is hit in normal pre-sizing usage.
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Base buffer implementation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_vulkan_allocated_buffer_handle(
    iree_hal_buffer_t* allocated_buffer, VkDeviceMemory* out_memory,
    VkBuffer* out_handle) {
  IREE_ASSERT_ARGUMENT(allocated_buffer);
  IREE_ASSERT_ARGUMENT(out_memory);
  IREE_ASSERT_ARGUMENT(out_handle);
  iree_hal_vulkan_base_buffer_t* buffer =
      (iree_hal_vulkan_base_buffer_t*)allocated_buffer;
  *out_memory = buffer->device_memory;
  *out_handle = buffer->handle;
  return iree_ok_status();
}
