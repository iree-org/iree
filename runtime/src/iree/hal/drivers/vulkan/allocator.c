// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/allocator.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_allocator_t {
  // HAL resource header.
  iree_hal_resource_t resource;

  // Host allocator used for allocator-owned host allocations.
  iree_allocator_t host_allocator;

  // Physical-device memory properties captured during logical-device creation.
  VkPhysicalDeviceMemoryProperties2 memory_properties2;

  // Aggregate allocation statistics.
  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_vulkan_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_vulkan_allocator_vtable;

static iree_hal_vulkan_allocator_t* iree_hal_vulkan_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_allocator_vtable);
  return (iree_hal_vulkan_allocator_t*)base_value;
}

iree_status_t iree_hal_vulkan_allocator_create(
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  memset(allocator, 0, sizeof(*allocator));
  iree_hal_resource_initialize(&iree_hal_vulkan_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->memory_properties2 = physical_device->memory_properties2;

  *out_allocator = (iree_hal_allocator_t*)allocator;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_vulkan_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_vulkan_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  const iree_hal_vulkan_allocator_t* allocator =
      (const iree_hal_vulkan_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_vulkan_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  (void)base_allocator;
  return iree_ok_status();
}

static void iree_hal_vulkan_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
  IREE_STATISTICS({
    iree_hal_vulkan_allocator_t* allocator =
        iree_hal_vulkan_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_hal_memory_type_t iree_hal_vulkan_memory_type_from_vk(
    VkMemoryPropertyFlags flags) {
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  if (iree_all_bits_set(flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    memory_type |= IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  }
  if (iree_all_bits_set(flags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  }
  if (iree_all_bits_set(flags, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  }
  if (iree_all_bits_set(flags, VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_CACHED;
  }
  return memory_type;
}

static iree_hal_buffer_usage_t iree_hal_vulkan_memory_usage_from_vk(
    VkMemoryPropertyFlags flags) {
  iree_hal_buffer_usage_t allowed_usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  if (iree_all_bits_set(flags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
    allowed_usage |= IREE_HAL_BUFFER_USAGE_MAPPING;
  }
  return allowed_usage;
}

static iree_status_t iree_hal_vulkan_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  const VkPhysicalDeviceMemoryProperties* memory_properties =
      &allocator->memory_properties2.memoryProperties;
  const iree_host_size_t heap_count = memory_properties->memoryTypeCount;
  *out_count = heap_count;
  if (heaps != NULL && capacity < heap_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan memory heap query capacity %" PRIhsz
                            " is smaller than the memory type count %" PRIhsz,
                            capacity, heap_count);
  }
  if (heaps == NULL) return iree_ok_status();

  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    const VkMemoryType* memory_type = &memory_properties->memoryTypes[i];
    const VkMemoryHeap* memory_heap =
        &memory_properties->memoryHeaps[memory_type->heapIndex];
    heaps[i] = (iree_hal_allocator_memory_heap_t){
        .type = iree_hal_vulkan_memory_type_from_vk(memory_type->propertyFlags),
        .allowed_usage =
            iree_hal_vulkan_memory_usage_from_vk(memory_type->propertyFlags),
        .max_allocation_size = memory_heap->size,
        .min_alignment = 1,
    };
  }
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_vulkan_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  (void)base_allocator;
  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  if (*allocation_size == 0) {
    *allocation_size = 4;
  }
  return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
}

static iree_status_t iree_hal_vulkan_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  (void)base_allocator;
  (void)params;
  (void)allocation_size;
  *out_buffer = NULL;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "Vulkan buffer allocation requires the slab/sparse allocator");
}

static void iree_hal_vulkan_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  (void)base_allocator;
  IREE_ASSERT_ARGUMENT(base_buffer);
  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_vulkan_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  (void)base_allocator;
  (void)params;
  (void)external_buffer;
  (void)release_callback;
  *out_buffer = NULL;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "Vulkan external buffer import requires the slab/sparse allocator");
}

static iree_status_t iree_hal_vulkan_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  IREE_ASSERT_ARGUMENT(out_external_buffer);
  (void)base_allocator;
  (void)buffer;
  (void)requested_type;
  (void)requested_flags;
  memset(out_external_buffer, 0, sizeof(*out_external_buffer));
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "Vulkan external buffer export requires the slab/sparse allocator");
}

static bool iree_hal_vulkan_allocator_supports_virtual_memory(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  (void)base_allocator;
  return false;
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params,
    iree_device_size_t* IREE_RESTRICT out_minimum_page_size,
    iree_device_size_t* IREE_RESTRICT out_recommended_page_size) {
  (void)base_allocator;
  (void)params;
  *out_minimum_page_size = 0;
  *out_recommended_page_size = 0;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan virtual memory requires sparse buffer allocator support");
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
    iree_hal_buffer_t** IREE_RESTRICT out_virtual_buffer) {
  (void)base_allocator;
  (void)queue_affinity;
  (void)size;
  *out_virtual_buffer = NULL;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan virtual memory requires sparse buffer allocator support");
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_release(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer) {
  (void)base_allocator;
  (void)virtual_buffer;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan virtual memory requires sparse buffer allocator support");
}

static iree_status_t iree_hal_vulkan_allocator_physical_memory_allocate(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params, iree_device_size_t size,
    iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** IREE_RESTRICT out_physical_memory) {
  (void)base_allocator;
  (void)params;
  (void)size;
  (void)host_allocator;
  *out_physical_memory = NULL;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan virtual memory requires sparse buffer allocator support");
}

static iree_status_t iree_hal_vulkan_allocator_physical_memory_free(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory) {
  (void)base_allocator;
  (void)physical_memory;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan virtual memory requires sparse buffer allocator support");
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_map(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)physical_memory;
  (void)physical_offset;
  (void)size;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan virtual memory requires sparse buffer allocator support");
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan virtual memory requires sparse buffer allocator support");
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_protect(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_protection_t protection) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  (void)queue_affinity;
  (void)protection;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan virtual memory requires sparse buffer allocator support");
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_advise(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_advice_t advice) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  (void)queue_affinity;
  (void)advice;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan virtual memory requires sparse buffer allocator support");
}

static const iree_hal_allocator_vtable_t iree_hal_vulkan_allocator_vtable = {
    .destroy = iree_hal_vulkan_allocator_destroy,
    .host_allocator = iree_hal_vulkan_allocator_host_allocator,
    .trim = iree_hal_vulkan_allocator_trim,
    .query_statistics = iree_hal_vulkan_allocator_query_statistics,
    .query_memory_heaps = iree_hal_vulkan_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_vulkan_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_vulkan_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_vulkan_allocator_deallocate_buffer,
    .import_buffer = iree_hal_vulkan_allocator_import_buffer,
    .export_buffer = iree_hal_vulkan_allocator_export_buffer,
    .supports_virtual_memory =
        iree_hal_vulkan_allocator_supports_virtual_memory,
    .virtual_memory_query_granularity =
        iree_hal_vulkan_allocator_virtual_memory_query_granularity,
    .virtual_memory_reserve = iree_hal_vulkan_allocator_virtual_memory_reserve,
    .virtual_memory_release = iree_hal_vulkan_allocator_virtual_memory_release,
    .physical_memory_allocate =
        iree_hal_vulkan_allocator_physical_memory_allocate,
    .physical_memory_free = iree_hal_vulkan_allocator_physical_memory_free,
    .virtual_memory_map = iree_hal_vulkan_allocator_virtual_memory_map,
    .virtual_memory_unmap = iree_hal_vulkan_allocator_virtual_memory_unmap,
    .virtual_memory_protect = iree_hal_vulkan_allocator_virtual_memory_protect,
    .virtual_memory_advise = iree_hal_vulkan_allocator_virtual_memory_advise,
};
