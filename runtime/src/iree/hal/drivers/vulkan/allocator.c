// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/allocator.h"

#include <stdio.h>
#include <string.h>

#include "iree/async/notification.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/queue.h"
#include "iree/hal/drivers/vulkan/slab_provider.h"
#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/tlsf_pool.h"

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_allocator_t
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_VULKAN_ALLOCATOR_ID = "iree-hal-vulkan-unpooled";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

#define IREE_HAL_VULKAN_ALLOCATOR_DEFAULT_POOL_RANGE_LENGTH \
  (64ull * 1024ull * 1024ull)

#define IREE_HAL_VULKAN_ALLOCATOR_DEFAULT_POOL_ALIGNMENT 256ull

#define IREE_HAL_VULKAN_ALLOCATOR_DEFAULT_POOL_PRIORITY_OVERSIZED 0

#define IREE_HAL_VULKAN_ALLOCATOR_DEFAULT_POOL_PRIORITY_TLSF 1000

typedef struct iree_hal_vulkan_allocator_memory_placement_t {
  // Vulkan memory type index selected for the allocation.
  uint32_t memory_type_index;

  // Vulkan memory property flags for |memory_type_index|.
  VkMemoryPropertyFlags memory_property_flags;

  // HAL memory type exposed by allocated buffers.
  iree_hal_memory_type_t memory_type;
} iree_hal_vulkan_allocator_memory_placement_t;

typedef struct iree_hal_vulkan_allocator_pool_pair_t {
  // Slab provider acquiring whole Vulkan buffers for this memory type.
  iree_hal_slab_provider_t* slab_provider;

  // Suballocating pool used for allocations up to the default slab length.
  iree_hal_pool_t* tlsf_pool;

  // Direct per-allocation pool used for allocations larger than one slab.
  iree_hal_pool_t* oversized_pool;

  // Selection priority for this memory type among compatible pools.
  int32_t memory_priority;
} iree_hal_vulkan_allocator_pool_pair_t;

typedef struct iree_hal_vulkan_allocator_virtual_memory_mapping_t {
  // Next mapping in the allocator-owned registry.
  struct iree_hal_vulkan_allocator_virtual_memory_mapping_t* next;

  // Virtual sparse buffer containing this mapped range.
  iree_hal_buffer_t* virtual_buffer;

  // Physical allocation bound into |virtual_buffer|.
  iree_hal_physical_memory_t* physical_memory;

  // Byte offset in |virtual_buffer| where the mapping begins.
  iree_device_size_t virtual_offset;

  // Byte offset in |physical_memory| where the mapping begins.
  iree_device_size_t physical_offset;

  // Byte length of the mapped range.
  iree_device_size_t size;
} iree_hal_vulkan_allocator_virtual_memory_mapping_t;

struct iree_hal_physical_memory_t {
  // Host allocator used to free this wrapper.
  iree_allocator_t host_allocator;

  // Device-level Vulkan dispatch table copied from the allocator.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device that owns |device_memory|.
  VkDevice logical_device;

  // Allocator that owns registry synchronization for this handle.
  iree_hal_vulkan_allocator_t* owner_allocator;

  // Standalone device memory allocation.
  VkDeviceMemory device_memory;

  // Allocated physical memory byte length.
  iree_device_size_t allocation_size;

  // Total mapped byte count protected by the parent allocator registry mutex.
  iree_device_size_t mapped_size;

  // Vulkan memory type index used for |device_memory|.
  uint32_t memory_type_index;

  // HAL memory type exposed by this physical allocation.
  iree_hal_memory_type_t memory_type;
};

struct iree_hal_vulkan_allocator_t {
  // HAL resource header.
  iree_hal_resource_t resource;

  // Host allocator used for allocator-owned host allocations.
  iree_allocator_t host_allocator;

  // Parent logical device. Unowned; device owns this allocator.
  iree_hal_device_t* parent_device;

  // Device-level Vulkan dispatch table copied from the logical device.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device that owns allocations.
  VkDevice logical_device;

  // Physical-device properties captured during logical-device creation.
  VkPhysicalDeviceProperties2 properties2;

  // Vulkan 1.1 property set including maxMemoryAllocationSize.
  VkPhysicalDeviceVulkan11Properties properties11;

  // VK_EXT_external_memory_host properties, if the extension is enabled.
  VkPhysicalDeviceExternalMemoryHostPropertiesEXT
      external_memory_host_properties;

  // Physical-device memory properties captured during logical-device creation.
  VkPhysicalDeviceMemoryProperties2 memory_properties2;

  // HAL feature bits enabled on the logical device.
  iree_hal_vulkan_features_t enabled_features;

  // Device extension bits enabled on the logical device.
  iree_hal_vulkan_device_extensions_t enabled_extensions;

  // Queue affinity bits supported by this logical device.
  iree_hal_queue_affinity_t queue_affinity_mask;

  // Internal queue lane used to perform sparse memory binding. Borrowed.
  iree_hal_vulkan_queue_t* sparse_binding_queue;

  // Protects |virtual_memory_mappings| and physical-memory mapped sizes.
  iree_slim_mutex_t virtual_memory_mutex;

  // Registry of currently mapped sparse virtual memory ranges.
  iree_hal_vulkan_allocator_virtual_memory_mapping_t* virtual_memory_mappings;

  // Shared notification published when default-pool reservations are released.
  iree_async_notification_t* default_pool_notification;

  // Default queue-pool backend provider for caller-created pools.
  iree_hal_slab_provider_t* default_queue_slab_provider;

  // Per-Vulkan-memory-type provider and pool pairs.
  iree_hal_vulkan_allocator_pool_pair_t pool_pairs[VK_MAX_MEMORY_TYPES];

  // Number of initialized entries in |pool_pairs|.
  iree_host_size_t pool_pair_count;

  // Alignment used by default TLSF pools.
  iree_device_size_t default_pool_alignment;

  // Aggregate allocation statistics.
  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
};

static const iree_hal_allocator_vtable_t iree_hal_vulkan_allocator_vtable;

static iree_status_t iree_hal_vulkan_allocator_initialize_default_pools(
    iree_hal_vulkan_allocator_t* allocator, iree_async_proactor_t* proactor);

static void iree_hal_vulkan_allocator_deinitialize_default_pools(
    iree_hal_vulkan_allocator_t* allocator);

static iree_hal_vulkan_allocator_t* iree_hal_vulkan_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_allocator_vtable);
  return (iree_hal_vulkan_allocator_t*)base_value;
}

static bool iree_hal_vulkan_allocator_ranges_overlap(
    iree_device_size_t lhs_offset, iree_device_size_t lhs_size,
    iree_device_size_t rhs_offset, iree_device_size_t rhs_size) {
  const iree_device_size_t lhs_end = lhs_offset + lhs_size;
  const iree_device_size_t rhs_end = rhs_offset + rhs_size;
  return lhs_offset < rhs_end && rhs_offset < lhs_end;
}

static void iree_hal_vulkan_allocator_deinitialize_virtual_memory_registry(
    iree_hal_vulkan_allocator_t* allocator) {
  IREE_ASSERT(allocator->virtual_memory_mappings == NULL);
  iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping =
      allocator->virtual_memory_mappings;
  while (mapping) {
    iree_hal_vulkan_allocator_virtual_memory_mapping_t* next_mapping =
        mapping->next;
    iree_allocator_free(allocator->host_allocator, mapping);
    mapping = next_mapping;
  }
  allocator->virtual_memory_mappings = NULL;
  iree_slim_mutex_deinitialize(&allocator->virtual_memory_mutex);
}

iree_status_t iree_hal_vulkan_allocator_create(
    iree_hal_device_t* parent_device, const iree_hal_vulkan_device_syms_t* syms,
    VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_device_extensions_t enabled_extensions,
    iree_hal_queue_affinity_t queue_affinity_mask,
    iree_hal_vulkan_queue_t* sparse_binding_queue,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(parent_device);
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(proactor);
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
  allocator->parent_device = parent_device;
  allocator->syms = *syms;
  allocator->logical_device = logical_device;
  allocator->properties2 = physical_device->properties2;
  allocator->properties2.pNext = NULL;
  allocator->properties11 = physical_device->properties11;
  allocator->properties11.pNext = NULL;
  allocator->external_memory_host_properties =
      physical_device->external_memory_host_properties;
  allocator->external_memory_host_properties.pNext = NULL;
  allocator->memory_properties2 = physical_device->memory_properties2;
  allocator->enabled_features = enabled_features;
  allocator->enabled_extensions = enabled_extensions;
  allocator->queue_affinity_mask = queue_affinity_mask;
  allocator->sparse_binding_queue = sparse_binding_queue;
  iree_slim_mutex_initialize(&allocator->virtual_memory_mutex);

  iree_status_t status =
      iree_hal_vulkan_allocator_initialize_default_pools(allocator, proactor);
  if (iree_status_is_ok(status)) {
    *out_allocator = (iree_hal_allocator_t*)allocator;
  } else {
    iree_hal_allocator_release((iree_hal_allocator_t*)allocator);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_allocator_deinitialize_default_pools(allocator);
  iree_hal_vulkan_allocator_deinitialize_virtual_memory_registry(allocator);
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
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < allocator->pool_pair_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_pool_trim(allocator->pool_pairs[i].tlsf_pool);
  }
  return status;
}

iree_status_t iree_hal_vulkan_allocator_query_queue_pool_backend(
    iree_hal_allocator_t* base_allocator,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  IREE_ASSERT_ARGUMENT(out_backend);
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  iree_hal_queue_affinity_t normalized_affinity =
      iree_hal_queue_affinity_is_any(queue_affinity)
          ? allocator->queue_affinity_mask
          : queue_affinity;
  iree_hal_queue_affinity_and_into(normalized_affinity,
                                   allocator->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(normalized_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid Vulkan queue affinity bits specified");
  }
  out_backend->slab_provider = allocator->default_queue_slab_provider;
  out_backend->notification = allocator->default_pool_notification;
  out_backend->epoch_query = iree_hal_pool_epoch_query_null();
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

static iree_device_size_t iree_hal_vulkan_allocator_max_allocation_size(
    const iree_hal_vulkan_allocator_t* allocator,
    const VkMemoryHeap* memory_heap) {
  iree_device_size_t limit = memory_heap->size;
  if (allocator->properties11.maxMemoryAllocationSize != 0 &&
      allocator->properties11.maxMemoryAllocationSize < limit) {
    limit = allocator->properties11.maxMemoryAllocationSize;
  }
  return limit;
}

static const VkMemoryHeap* iree_hal_vulkan_allocator_memory_heap_for_type(
    const iree_hal_vulkan_allocator_t* allocator, uint32_t memory_type_index) {
  const VkPhysicalDeviceMemoryProperties* memory_properties =
      &allocator->memory_properties2.memoryProperties;
  const VkMemoryType* memory_type =
      &memory_properties->memoryTypes[memory_type_index];
  return &memory_properties->memoryHeaps[memory_type->heapIndex];
}

static iree_device_size_t
iree_hal_vulkan_allocator_max_allocation_size_for_type(
    const iree_hal_vulkan_allocator_t* allocator, uint32_t memory_type_index) {
  return iree_hal_vulkan_allocator_max_allocation_size(
      allocator, iree_hal_vulkan_allocator_memory_heap_for_type(
                     allocator, memory_type_index));
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
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT) &&
      !iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  }
  return memory_type;
}

static iree_hal_buffer_usage_t iree_hal_vulkan_allowed_usage_from_memory_type(
    iree_hal_memory_type_t memory_type) {
  iree_hal_buffer_usage_t allowed_usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH |
      IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE |
      IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT |
      IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    allowed_usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                     IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
                     IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
                     IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
                     IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;
  }
  return allowed_usage;
}

static iree_device_size_t iree_hal_vulkan_allocator_default_pool_alignment(
    const iree_hal_vulkan_allocator_t* allocator) {
  iree_device_size_t alignment =
      IREE_HAL_VULKAN_ALLOCATOR_DEFAULT_POOL_ALIGNMENT;
  const iree_device_size_t atom_size =
      (iree_device_size_t)
          allocator->properties2.properties.limits.nonCoherentAtomSize;
  if (atom_size > alignment) {
    alignment = iree_device_size_next_power_of_two(atom_size);
  }
  if (alignment < IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT) {
    alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  }
  return alignment;
}

static int32_t iree_hal_vulkan_allocator_memory_type_priority(
    iree_hal_memory_type_t memory_type) {
  int32_t priority = 0;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    priority += 100;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    priority += 10;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    priority += 5;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    priority += 2;
  }
  return priority;
}

// Heap queries feed first-match consumers such as the caching allocator. Prefer
// private device-local memory for common dispatches, then UMA/BAR memory, then
// host-local staging/readback classes.
static int32_t iree_hal_vulkan_allocator_query_memory_heap_priority(
    iree_hal_memory_type_t memory_type) {
  int32_t priority = 0;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    priority += 1000;
    if (!iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      priority += 300;
    }
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    priority += 200;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    priority += 100;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    priority += 40;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    priority += 20;
  }
  return priority;
}

static bool iree_hal_vulkan_allocator_memory_type_precedes(
    const iree_hal_vulkan_allocator_t* allocator,
    uint32_t left_memory_type_index, uint32_t right_memory_type_index) {
  const VkPhysicalDeviceMemoryProperties* memory_properties =
      &allocator->memory_properties2.memoryProperties;
  const VkMemoryType* left_memory_type =
      &memory_properties->memoryTypes[left_memory_type_index];
  const VkMemoryType* right_memory_type =
      &memory_properties->memoryTypes[right_memory_type_index];
  const iree_hal_memory_type_t left_hal_memory_type =
      iree_hal_vulkan_memory_type_from_vk(left_memory_type->propertyFlags);
  const iree_hal_memory_type_t right_hal_memory_type =
      iree_hal_vulkan_memory_type_from_vk(right_memory_type->propertyFlags);
  const int32_t left_priority =
      iree_hal_vulkan_allocator_query_memory_heap_priority(
          left_hal_memory_type);
  const int32_t right_priority =
      iree_hal_vulkan_allocator_query_memory_heap_priority(
          right_hal_memory_type);
  if (left_priority != right_priority) {
    return left_priority > right_priority;
  }
  const iree_device_size_t left_max_allocation_size =
      iree_hal_vulkan_allocator_max_allocation_size_for_type(
          allocator, left_memory_type_index);
  const iree_device_size_t right_max_allocation_size =
      iree_hal_vulkan_allocator_max_allocation_size_for_type(
          allocator, right_memory_type_index);
  if (left_max_allocation_size != right_max_allocation_size) {
    return left_max_allocation_size > right_max_allocation_size;
  }
  return left_memory_type_index < right_memory_type_index;
}

static int32_t iree_hal_vulkan_allocator_host_write_memory_priority(
    iree_hal_memory_type_t memory_type) {
  int32_t priority = 0;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    priority += 100;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    priority += 20;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    priority += 5;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    priority += 2;
  }
  return priority;
}

static int32_t iree_hal_vulkan_allocator_host_cached_memory_priority(
    iree_hal_memory_type_t memory_type) {
  int32_t priority = 0;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    priority += 100;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    priority += 20;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    priority += 5;
  }
  return priority;
}

static void iree_hal_vulkan_allocator_append_unique_memory_type(
    uint32_t memory_type_index, uint32_t* selected_indices,
    iree_host_size_t* selected_count) {
  for (iree_host_size_t i = 0; i < *selected_count; ++i) {
    if (selected_indices[i] == memory_type_index) return;
  }
  selected_indices[*selected_count] = memory_type_index;
  *selected_count = *selected_count + 1;
}

static iree_string_view_t iree_hal_vulkan_allocator_format_pool_trace_name(
    char* storage, iree_host_size_t storage_capacity, iree_string_view_t kind,
    uint32_t memory_type_index) {
  const int length =
      snprintf(storage, storage_capacity, "vulkan-%.*s-memory-type-%u",
               (int)kind.size, kind.data, memory_type_index);
  if (length < 0) return iree_string_view_empty();
  const iree_host_size_t clamped_length =
      iree_min((iree_host_size_t)length, storage_capacity - 1);
  return iree_make_string_view(storage, clamped_length);
}

static iree_status_t iree_hal_vulkan_allocator_create_pool_pair(
    iree_hal_vulkan_allocator_t* allocator, uint32_t memory_type_index,
    iree_hal_memory_type_t memory_type,
    VkMemoryPropertyFlags memory_property_flags,
    iree_hal_buffer_usage_t supported_usage,
    iree_hal_vulkan_allocator_pool_pair_t* out_pool_pair) {
  memset(out_pool_pair, 0, sizeof(*out_pool_pair));
  IREE_TRACE_ZONE_BEGIN(z0);
  out_pool_pair->memory_priority =
      iree_hal_vulkan_allocator_memory_type_priority(memory_type);

  char slab_trace_storage[64] = {0};
  iree_hal_vulkan_slab_provider_options_t slab_options = {
      .parent_device = allocator->parent_device,
      .syms = &allocator->syms,
      .logical_device = allocator->logical_device,
      .memory_type_index = memory_type_index,
      .memory_property_flags = memory_property_flags,
      .memory_type = memory_type,
      .supported_usage = supported_usage,
      .queue_affinity_mask = allocator->queue_affinity_mask,
      .min_alignment = allocator->default_pool_alignment,
      .non_coherent_atom_size =
          allocator->properties2.properties.limits.nonCoherentAtomSize,
  };
  iree_string_view_t slab_trace_name =
      iree_hal_vulkan_allocator_format_pool_trace_name(
          slab_trace_storage, IREE_ARRAYSIZE(slab_trace_storage),
          IREE_SV("slab"), memory_type_index);
  iree_status_t status = iree_hal_vulkan_slab_provider_create(
      allocator, slab_options, slab_trace_name, allocator->host_allocator,
      &out_pool_pair->slab_provider);

  iree_device_size_t range_length =
      IREE_HAL_VULKAN_ALLOCATOR_DEFAULT_POOL_RANGE_LENGTH;
  const iree_device_size_t max_allocation_size =
      iree_hal_vulkan_allocator_max_allocation_size_for_type(allocator,
                                                             memory_type_index);
  if (max_allocation_size < range_length) {
    range_length = max_allocation_size;
  }
  range_length &= ~(allocator->default_pool_alignment - 1);
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(range_length < allocator->default_pool_alignment)) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "Vulkan memory type %u max allocation size %" PRIu64
                         " is smaller than default pool alignment %" PRIu64,
                         memory_type_index, (uint64_t)max_allocation_size,
                         (uint64_t)allocator->default_pool_alignment);
  }

  char tlsf_trace_storage[64] = {0};
  if (iree_status_is_ok(status)) {
    iree_hal_tlsf_pool_options_t tlsf_options = {
        .tlsf_options =
            {
                .range_length = range_length,
                .alignment = allocator->default_pool_alignment,
                .frontier_capacity =
                    IREE_HAL_MEMORY_TLSF_DEFAULT_FRONTIER_CAPACITY,
            },
        .budget_limit = 0,
        .trace_name = iree_hal_vulkan_allocator_format_pool_trace_name(
            tlsf_trace_storage, IREE_ARRAYSIZE(tlsf_trace_storage),
            IREE_SV("tlsf"), memory_type_index),
    };
    status = iree_hal_tlsf_pool_create(
        tlsf_options, out_pool_pair->slab_provider,
        allocator->default_pool_notification, iree_hal_pool_epoch_query_null(),
        allocator->host_allocator, &out_pool_pair->tlsf_pool);
  }

  char oversized_trace_storage[64] = {0};
  if (iree_status_is_ok(status)) {
    iree_hal_passthrough_pool_options_t oversized_options = {
        .trace_name = iree_hal_vulkan_allocator_format_pool_trace_name(
            oversized_trace_storage, IREE_ARRAYSIZE(oversized_trace_storage),
            IREE_SV("oversized"), memory_type_index),
    };
    status = iree_hal_passthrough_pool_create(
        oversized_options, out_pool_pair->slab_provider,
        allocator->default_pool_notification, allocator->host_allocator,
        &out_pool_pair->oversized_pool);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_pool_release(out_pool_pair->oversized_pool);
    iree_hal_pool_release(out_pool_pair->tlsf_pool);
    iree_hal_slab_provider_release(out_pool_pair->slab_provider);
    memset(out_pool_pair, 0, sizeof(*out_pool_pair));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_allocator_deinitialize_default_pools(
    iree_hal_vulkan_allocator_t* allocator) {
  for (iree_host_size_t i = 0; i < allocator->pool_pair_count; ++i) {
    iree_hal_pool_release(allocator->pool_pairs[i].oversized_pool);
    iree_hal_pool_release(allocator->pool_pairs[i].tlsf_pool);
    iree_hal_slab_provider_release(allocator->pool_pairs[i].slab_provider);
  }
  memset(allocator->pool_pairs, 0, sizeof(allocator->pool_pairs));
  allocator->pool_pair_count = 0;
  allocator->default_queue_slab_provider = NULL;
  iree_async_notification_release(allocator->default_pool_notification);
  allocator->default_pool_notification = NULL;
}

static iree_status_t iree_hal_vulkan_allocator_initialize_default_pools(
    iree_hal_vulkan_allocator_t* allocator, iree_async_proactor_t* proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);

  allocator->default_pool_alignment =
      iree_hal_vulkan_allocator_default_pool_alignment(allocator);
  iree_status_t status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE,
      &allocator->default_pool_notification);

  const VkPhysicalDeviceMemoryProperties* memory_properties =
      &allocator->memory_properties2.memoryProperties;
  uint32_t selected_indices[3] = {0};
  iree_host_size_t selected_count = 0;
  uint32_t best_device_index = UINT32_MAX;
  int32_t best_device_priority = INT32_MIN;
  uint32_t best_host_write_index = UINT32_MAX;
  int32_t best_host_write_priority = INT32_MIN;
  uint32_t best_host_cached_index = UINT32_MAX;
  int32_t best_host_cached_priority = INT32_MIN;
  for (uint32_t i = 0; i < memory_properties->memoryTypeCount; ++i) {
    const iree_hal_memory_type_t memory_type =
        iree_hal_vulkan_memory_type_from_vk(
            memory_properties->memoryTypes[i].propertyFlags);
    if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
      const int32_t priority =
          iree_hal_vulkan_allocator_memory_type_priority(memory_type);
      if (priority > best_device_priority) {
        best_device_priority = priority;
        best_device_index = i;
      }
    }
    if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      const int32_t priority =
          iree_hal_vulkan_allocator_host_write_memory_priority(memory_type);
      if (priority > best_host_write_priority) {
        best_host_write_priority = priority;
        best_host_write_index = i;
      }
    }
    if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
      const int32_t priority =
          iree_hal_vulkan_allocator_host_cached_memory_priority(memory_type);
      if (priority > best_host_cached_priority) {
        best_host_cached_priority = priority;
        best_host_cached_index = i;
      }
    }
  }
  if (best_device_index != UINT32_MAX) {
    iree_hal_vulkan_allocator_append_unique_memory_type(
        best_device_index, selected_indices, &selected_count);
  }
  if (best_host_write_index != UINT32_MAX) {
    iree_hal_vulkan_allocator_append_unique_memory_type(
        best_host_write_index, selected_indices, &selected_count);
  }
  if (best_host_cached_index != UINT32_MAX) {
    iree_hal_vulkan_allocator_append_unique_memory_type(
        best_host_cached_index, selected_indices, &selected_count);
  }

  int32_t default_queue_provider_priority = INT32_MIN;
  for (iree_host_size_t selected_ordinal = 0;
       selected_ordinal < selected_count && iree_status_is_ok(status);
       ++selected_ordinal) {
    const uint32_t i = selected_indices[selected_ordinal];
    const VkMemoryType* vk_memory_type = &memory_properties->memoryTypes[i];
    const iree_hal_memory_type_t memory_type =
        iree_hal_vulkan_memory_type_from_vk(vk_memory_type->propertyFlags);
    const iree_hal_buffer_usage_t supported_usage =
        iree_hal_vulkan_allowed_usage_from_memory_type(memory_type);
    iree_hal_vulkan_allocator_pool_pair_t pool_pair;
    status = iree_hal_vulkan_allocator_create_pool_pair(
        allocator, i, memory_type, vk_memory_type->propertyFlags,
        supported_usage, &pool_pair);
    if (!iree_status_is_ok(status)) break;

    if (iree_status_is_ok(status)) {
      if (pool_pair.memory_priority > default_queue_provider_priority) {
        default_queue_provider_priority = pool_pair.memory_priority;
        allocator->default_queue_slab_provider = pool_pair.slab_provider;
      }
      allocator->pool_pairs[allocator->pool_pair_count++] = pool_pair;
    }
  }

  if (iree_status_is_ok(status) && allocator->pool_pair_count == 0) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan physical device reports no memory types for default pools");
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_allocator_deinitialize_default_pools(allocator);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
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

  uint32_t memory_type_indices[VK_MAX_MEMORY_TYPES];
  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    memory_type_indices[i] = (uint32_t)i;
  }
  for (iree_host_size_t i = 1; i < heap_count; ++i) {
    const uint32_t memory_type_index = memory_type_indices[i];
    iree_host_size_t insertion_index = i;
    while (insertion_index > 0 &&
           iree_hal_vulkan_allocator_memory_type_precedes(
               allocator, memory_type_index,
               memory_type_indices[insertion_index - 1])) {
      memory_type_indices[insertion_index] =
          memory_type_indices[insertion_index - 1];
      --insertion_index;
    }
    memory_type_indices[insertion_index] = memory_type_index;
  }

  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    const uint32_t memory_type_index = memory_type_indices[i];
    const VkMemoryType* memory_type =
        &memory_properties->memoryTypes[memory_type_index];
    const iree_hal_memory_type_t hal_memory_type =
        iree_hal_vulkan_memory_type_from_vk(memory_type->propertyFlags);
    heaps[i] = (iree_hal_allocator_memory_heap_t){
        .type = hal_memory_type,
        .allowed_usage =
            iree_hal_vulkan_allowed_usage_from_memory_type(hal_memory_type),
        .max_allocation_size =
            iree_hal_vulkan_allocator_max_allocation_size_for_type(
                allocator, memory_type_index),
        .min_alignment = 1,
    };
  }
  return iree_ok_status();
}

static bool iree_hal_vulkan_allocator_normalize_queue_affinity(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_hal_buffer_params_t* params) {
  iree_hal_queue_affinity_t queue_affinity =
      iree_hal_queue_affinity_is_any(params->queue_affinity)
          ? allocator->queue_affinity_mask
          : params->queue_affinity;
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   allocator->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) return false;
  params->queue_affinity = queue_affinity;
  return true;
}

static const iree_hal_buffer_usage_t iree_hal_vulkan_mapping_usage_bits =
    IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
    IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
    IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
    IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
    IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;

static bool iree_hal_vulkan_allocator_strip_optional_mapping(
    iree_hal_buffer_params_t* params) {
  if (!iree_any_bit_set(params->usage, iree_hal_vulkan_mapping_usage_bits)) {
    return true;
  }
  if (!iree_all_bits_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL)) {
    return false;
  }
  params->usage &= ~iree_hal_vulkan_mapping_usage_bits;
  return true;
}

static int iree_hal_vulkan_allocator_score_memory_type(
    const iree_hal_buffer_params_t* params,
    iree_hal_memory_type_t memory_type) {
  const bool mapping_requested =
      iree_any_bit_set(params->usage, iree_hal_vulkan_mapping_usage_bits);
  const bool dispatch_requested =
      iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_DISPATCH);
  int score = 0;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    score += dispatch_requested && !mapping_requested ? 100 : 20;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    score += mapping_requested ? 80 : 5;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    score += mapping_requested ? 20 : 0;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    score += mapping_requested ? 10 : 0;
  }
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST) &&
      iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    score += 50;
  }
  if (iree_all_bits_set(params->type,
                        IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE) &&
      iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    score += 50;
  }
  return score;
}

static bool iree_hal_vulkan_allocator_resolve_memory_placement(
    const iree_hal_vulkan_allocator_t* allocator,
    uint32_t allowed_memory_type_bits, iree_hal_buffer_params_t* params,
    iree_hal_vulkan_allocator_memory_placement_t* out_placement) {
  memset(out_placement, 0, sizeof(*out_placement));
  if (!iree_hal_vulkan_allocator_normalize_queue_affinity(allocator, params)) {
    return false;
  }
  if (!iree_device_size_is_valid_alignment(params->min_alignment)) {
    return false;
  }

  const iree_hal_memory_type_t required_type =
      params->type & ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  const VkPhysicalDeviceMemoryProperties* memory_properties =
      &allocator->memory_properties2.memoryProperties;

  bool found = false;
  int best_score = 0;
  iree_hal_buffer_params_t best_params = *params;
  iree_hal_vulkan_allocator_memory_placement_t best_placement;
  memset(&best_placement, 0, sizeof(best_placement));

  for (uint32_t i = 0; i < memory_properties->memoryTypeCount; ++i) {
    if (!iree_all_bits_set(allowed_memory_type_bits, 1u << i)) continue;
    const VkMemoryType* vk_memory_type = &memory_properties->memoryTypes[i];
    const iree_hal_memory_type_t memory_type =
        iree_hal_vulkan_memory_type_from_vk(vk_memory_type->propertyFlags);
    if (!iree_all_bits_set(memory_type, required_type)) continue;

    iree_hal_buffer_params_t candidate_params = *params;
    const iree_hal_buffer_usage_t allowed_usage =
        iree_hal_vulkan_allowed_usage_from_memory_type(memory_type);
    if (!iree_all_bits_set(allowed_usage, candidate_params.usage)) {
      if (!iree_hal_vulkan_allocator_strip_optional_mapping(
              &candidate_params)) {
        continue;
      }
      if (!iree_all_bits_set(allowed_usage, candidate_params.usage)) continue;
    }

    const int score = iree_hal_vulkan_allocator_score_memory_type(
        &candidate_params, memory_type);
    if (!found || score > best_score) {
      found = true;
      best_score = score;
      best_params = candidate_params;
      best_params.type = memory_type;
      best_placement = (iree_hal_vulkan_allocator_memory_placement_t){
          .memory_type_index = i,
          .memory_property_flags = vk_memory_type->propertyFlags,
          .memory_type = memory_type,
      };
    }
  }

  if (!found) return false;
  *params = best_params;
  *out_placement = best_placement;
  return true;
}

static iree_status_t iree_hal_vulkan_allocator_align_allocation_size(
    iree_device_size_t* allocation_size) {
  if (*allocation_size == 0) *allocation_size = 4;
  if (!iree_device_size_checked_align(*allocation_size, 4, allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan allocation size overflows 4-byte "
                            "buffer alignment");
  }
  return iree_ok_status();
}

static bool iree_hal_vulkan_allocator_allocation_size_in_range(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_device_size_t allocation_size) {
  return allocator->properties11.maxMemoryAllocationSize == 0 ||
         allocation_size <= allocator->properties11.maxMemoryAllocationSize;
}

static bool iree_hal_vulkan_allocator_supports_sparse_binding(
    const iree_hal_vulkan_allocator_t* allocator) {
  return iree_all_bits_set(allocator->enabled_features,
                           IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING) &&
         allocator->sparse_binding_queue != NULL;
}

static bool iree_hal_vulkan_allocator_supports_sparse_virtual_memory(
    const iree_hal_vulkan_allocator_t* allocator) {
  return iree_all_bits_set(
             allocator->enabled_features,
             IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED) &&
         allocator->sparse_binding_queue != NULL;
}

static bool iree_hal_vulkan_allocator_supports_host_allocation_import(
    const iree_hal_vulkan_allocator_t* allocator) {
  return iree_all_bits_set(
      allocator->enabled_extensions,
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_EXTERNAL_MEMORY_HOST);
}

static iree_status_t iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_device_size_t allocation_size, iree_device_size_t max_allocation_size,
    iree_hal_buffer_params_t* params) {
  if (!iree_hal_vulkan_allocator_supports_sparse_binding(allocator)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "Vulkan sparse binding support is required for allocation size %" PRIu64
        " above per-memory-type max allocation size %" PRIu64
        " but sparseBinding is not enabled with a sparse-capable queue",
        (uint64_t)allocation_size, (uint64_t)max_allocation_size);
  }
  if (!iree_hal_vulkan_allocator_strip_optional_mapping(params)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan sparse buffers cannot satisfy required mapping usage");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_allocator_require_virtual_memory(
    const iree_hal_vulkan_allocator_t* allocator) {
  if (iree_hal_vulkan_allocator_supports_sparse_virtual_memory(allocator)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan virtual memory requires sparseBinding, sparseResidencyBuffer, "
      "sparseResidencyAliased, and a sparse-capable queue");
}

static iree_hal_buffer_usage_t iree_hal_vulkan_allocator_virtual_buffer_usage(
    void) {
  return IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
}

static iree_hal_buffer_params_t iree_hal_vulkan_allocator_virtual_buffer_params(
    iree_hal_queue_affinity_t queue_affinity) {
  return (iree_hal_buffer_params_t){
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
              IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = iree_hal_vulkan_allocator_virtual_buffer_usage(),
      .queue_affinity = queue_affinity,
  };
}

static iree_status_t iree_hal_vulkan_allocator_prepare_virtual_memory_params(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_hal_buffer_params_t* params) {
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  iree_hal_buffer_params_canonicalize(params);
  if (!iree_hal_vulkan_allocator_strip_optional_mapping(params)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan sparse virtual memory cannot satisfy required mapping usage");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_allocator_require_virtual_buffer(
    iree_hal_buffer_t* virtual_buffer) {
  if (iree_hal_vulkan_sparse_buffer_is_virtual_reservation(virtual_buffer)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "Vulkan virtual memory operation requires a sparse virtual memory "
      "reservation");
}

static iree_hal_buffer_compatibility_t
iree_hal_vulkan_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);

  if (!iree_status_is_ok(
          iree_hal_vulkan_allocator_align_allocation_size(allocation_size))) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }
  if (!iree_hal_vulkan_allocator_allocation_size_in_range(allocator,
                                                          *allocation_size)) {
    if (!iree_status_is_ok(
            iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
                allocator, *allocation_size,
                allocator->properties11.maxMemoryAllocationSize, params))) {
      return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
    }
  }

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  if (!iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, UINT32_MAX, params, &memory_placement)) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }
  const iree_device_size_t max_allocation_size =
      iree_hal_vulkan_allocator_max_allocation_size_for_type(
          allocator, memory_placement.memory_type_index);
  if (*allocation_size > max_allocation_size) {
    if (!iree_status_is_ok(
            iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
                allocator, *allocation_size, max_allocation_size, params))) {
      return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
    }
    if (!iree_hal_vulkan_allocator_resolve_memory_placement(
            allocator, UINT32_MAX, params, &memory_placement)) {
      return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
    }
  }

  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;
  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  }
  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
  }
  if (!iree_all_bits_set(memory_placement.memory_type,
                         IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) &&
      iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE;
  }
  if (iree_hal_vulkan_allocator_supports_host_allocation_import(allocator) &&
      iree_all_bits_set(memory_placement.memory_type,
                        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
      !iree_all_bits_set(memory_placement.memory_type,
                         IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    const VkDeviceSize import_alignment =
        allocator->external_memory_host_properties
            .minImportedHostPointerAlignment;
    if (import_alignment == 0 || (import_alignment <= IREE_DEVICE_SIZE_MAX &&
                                  iree_device_size_is_valid_alignment(
                                      (iree_device_size_t)import_alignment))) {
      if (import_alignment > params->min_alignment) {
        params->min_alignment = (iree_device_size_t)import_alignment;
      }
      if (import_alignment != 0 &&
          !iree_device_size_checked_align(*allocation_size,
                                          (iree_device_size_t)import_alignment,
                                          allocation_size)) {
        return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
      }
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE;
    }
  }
  return compatibility;
}

static VkBufferUsageFlags iree_hal_vulkan_buffer_usage_from_hal(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_hal_buffer_usage_t hal_usage) {
  VkBufferUsageFlags usage = 0;
  if (iree_all_bits_set(hal_usage, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE)) {
    usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if (iree_all_bits_set(hal_usage, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET)) {
    usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
  if (iree_any_bit_set(hal_usage,
                       IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ)) {
    usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  }
  if (iree_any_bit_set(hal_usage, IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
    usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }
  if (iree_any_bit_set(hal_usage,
                       IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS)) {
    usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  }
  if (iree_all_bits_set(
          allocator->enabled_features,
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES) &&
      iree_any_bit_set(hal_usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }
  return usage;
}

static iree_status_t iree_hal_vulkan_allocator_create_buffer_handle(
    iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    VkBufferCreateFlags create_flags, const void* create_info_pnext,
    VkBuffer* out_buffer) {
  *out_buffer = VK_NULL_HANDLE;

  VkBufferCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = create_info_pnext,
      .flags = create_flags,
      .size = (VkDeviceSize)allocation_size,
      .usage = iree_hal_vulkan_buffer_usage_from_hal(allocator, params->usage),
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  return iree_vkCreateBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                             allocator->logical_device, &create_info,
                             /*pAllocator=*/NULL, out_buffer);
}

static iree_status_t iree_hal_vulkan_allocator_create_sparse_virtual_handle(
    iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    VkBuffer* out_buffer, VkMemoryRequirements* out_memory_requirements) {
  *out_buffer = VK_NULL_HANDLE;
  memset(out_memory_requirements, 0, sizeof(*out_memory_requirements));
  const VkBufferCreateFlags create_flags =
      VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
      VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
      VK_BUFFER_CREATE_SPARSE_ALIASED_BIT;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_create_buffer_handle(
      allocator, params, allocation_size, create_flags,
      /*create_info_pnext=*/NULL, out_buffer));
  iree_vkGetBufferMemoryRequirements(IREE_VULKAN_DEVICE(&allocator->syms),
                                     allocator->logical_device, *out_buffer,
                                     out_memory_requirements);
  return iree_ok_status();
}

static bool iree_hal_vulkan_allocator_uses_buffer_device_address(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_hal_buffer_usage_t hal_usage) {
  return iree_all_bits_set(
             allocator->enabled_features,
             IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES) &&
         iree_any_bit_set(hal_usage, IREE_HAL_BUFFER_USAGE_DISPATCH);
}

static VkMemoryAllocateFlags iree_hal_vulkan_allocator_memory_allocate_flags(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_hal_buffer_usage_t hal_usage) {
  return iree_hal_vulkan_allocator_uses_buffer_device_address(allocator,
                                                              hal_usage)
             ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
             : 0;
}

static iree_status_t iree_hal_vulkan_allocator_allocate_memory(
    iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_vulkan_allocator_memory_placement_t* memory_placement,
    iree_hal_buffer_usage_t hal_usage,
    const VkMemoryRequirements* memory_requirements,
    VkDeviceMemory* out_device_memory) {
  *out_device_memory = VK_NULL_HANDLE;

  VkMemoryAllocateFlagsInfo allocate_flags_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags =
          iree_hal_vulkan_allocator_memory_allocate_flags(allocator, hal_usage),
  };
  VkMemoryAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &allocate_flags_info,
      .allocationSize = memory_requirements->size,
      .memoryTypeIndex = memory_placement->memory_type_index,
  };
  return iree_vkAllocateMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                               allocator->logical_device, &allocate_info,
                               /*pAllocator=*/NULL, out_device_memory);
}

static VkDeviceAddress iree_hal_vulkan_allocator_query_device_address(
    iree_hal_vulkan_allocator_t* allocator, iree_hal_buffer_usage_t hal_usage,
    VkBuffer handle) {
  if (!iree_hal_vulkan_allocator_uses_buffer_device_address(allocator,
                                                            hal_usage)) {
    return 0;
  }
  VkBufferDeviceAddressInfo address_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = handle,
  };
  return iree_vkGetBufferDeviceAddress(IREE_VULKAN_DEVICE(&allocator->syms),
                                       allocator->logical_device,
                                       &address_info);
}

static iree_status_t iree_hal_vulkan_allocator_make_buffer_params_status(
    const iree_hal_buffer_params_t* params) {
#if IREE_STATUS_MODE
  iree_bitfield_string_temp_t temp0, temp1;
  iree_string_view_t memory_type_str =
      iree_hal_memory_type_format(params->type, &temp0);
  iree_string_view_t usage_str =
      iree_hal_buffer_usage_format(params->usage, &temp1);
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "allocator cannot allocate a Vulkan buffer with the given parameters; "
      "memory_type=%.*s, usage=%.*s",
      (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
      usage_str.data);
#else
  (void)params;
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "allocator cannot allocate a Vulkan buffer with the given parameters");
#endif  // IREE_STATUS_MODE
}

static bool iree_hal_vulkan_allocator_pool_matches(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size) {
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  const iree_hal_memory_type_t required_type =
      params.type & ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  if ((capabilities.memory_type & required_type) != required_type) {
    return false;
  }
  if ((capabilities.supported_usage & params.usage) != params.usage) {
    return false;
  }
  if (capabilities.min_allocation_size > 0 &&
      allocation_size < capabilities.min_allocation_size) {
    return false;
  }
  if (capabilities.max_allocation_size > 0 &&
      allocation_size > capabilities.max_allocation_size) {
    return false;
  }
  return true;
}

static iree_hal_pool_t* iree_hal_vulkan_allocator_select_default_pool(
    iree_hal_vulkan_allocator_t* allocator, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size) {
  iree_hal_buffer_params_canonicalize(&params);
  const iree_device_size_t requested_alignment =
      params.min_alignment ? params.min_alignment : 1;
  iree_hal_pool_t* selected_pool = NULL;
  int32_t selected_priority = INT32_MIN;
  for (iree_host_size_t i = 0; i < allocator->pool_pair_count; ++i) {
    const iree_hal_vulkan_allocator_pool_pair_t* pool_pair =
        &allocator->pool_pairs[i];
    const int32_t oversized_priority =
        IREE_HAL_VULKAN_ALLOCATOR_DEFAULT_POOL_PRIORITY_OVERSIZED +
        pool_pair->memory_priority;
    if (requested_alignment <= IREE_HAL_HEAP_BUFFER_ALIGNMENT &&
        oversized_priority > selected_priority &&
        iree_hal_vulkan_allocator_pool_matches(pool_pair->oversized_pool,
                                               params, allocation_size)) {
      selected_pool = pool_pair->oversized_pool;
      selected_priority = oversized_priority;
    }
    const int32_t tlsf_priority =
        IREE_HAL_VULKAN_ALLOCATOR_DEFAULT_POOL_PRIORITY_TLSF +
        pool_pair->memory_priority;
    if (requested_alignment <= allocator->default_pool_alignment &&
        tlsf_priority > selected_priority &&
        iree_hal_vulkan_allocator_pool_matches(pool_pair->tlsf_pool, params,
                                               allocation_size)) {
      selected_pool = pool_pair->tlsf_pool;
      selected_priority = tlsf_priority;
    }
  }
  return selected_pool;
}

iree_status_t iree_hal_vulkan_allocator_select_queue_alloca_plan(
    iree_hal_allocator_t* base_allocator, iree_hal_pool_t* requested_pool,
    iree_hal_buffer_params_t* params, iree_device_size_t* allocation_size,
    iree_hal_vulkan_queue_alloca_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(allocation_size);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (iree_hal_vulkan_queue_alloca_plan_t){
      .strategy = IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_NONE,
  };
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);

  iree_hal_buffer_params_canonicalize(params);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_align_allocation_size(allocation_size));

  if (requested_pool) {
    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(requested_pool, &capabilities);
    if (iree_any_bit_set(params->type, IREE_HAL_MEMORY_TYPE_OPTIMAL)) {
      params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
      params->type |= capabilities.memory_type;
    }

    iree_hal_vulkan_allocator_memory_placement_t memory_placement;
    if (!iree_hal_vulkan_allocator_resolve_memory_placement(
            allocator, UINT32_MAX, params, &memory_placement)) {
      return iree_hal_vulkan_allocator_make_buffer_params_status(params);
    }
    if (!iree_hal_vulkan_allocator_pool_matches(requested_pool, *params,
                                                *allocation_size)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "requested Vulkan queue allocation pool cannot satisfy allocation of "
          "%" PRIu64 " bytes",
          (uint64_t)*allocation_size);
    }
    out_plan->strategy = IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_POOL;
    out_plan->pool = requested_pool;
    return iree_ok_status();
  }

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  if (!iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, UINT32_MAX, params, &memory_placement)) {
    return iree_hal_vulkan_allocator_make_buffer_params_status(params);
  }
  iree_device_size_t max_allocation_size =
      iree_hal_vulkan_allocator_max_allocation_size_for_type(
          allocator, memory_placement.memory_type_index);
  if (*allocation_size > max_allocation_size) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
        allocator, *allocation_size, max_allocation_size, params));
    if (!iree_hal_vulkan_allocator_resolve_memory_placement(
            allocator, UINT32_MAX, params, &memory_placement)) {
      return iree_hal_vulkan_allocator_make_buffer_params_status(params);
    }
    max_allocation_size =
        iree_hal_vulkan_allocator_max_allocation_size_for_type(
            allocator, memory_placement.memory_type_index);
  }

  iree_hal_pool_t* selected_pool =
      iree_hal_vulkan_allocator_select_default_pool(allocator, *params,
                                                    *allocation_size);
  if (selected_pool) {
    out_plan->strategy = IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_POOL;
    out_plan->pool = selected_pool;
    return iree_ok_status();
  }

  if (*allocation_size > max_allocation_size) {
    out_plan->strategy = IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_SPARSE;
    out_plan->allocator = base_allocator;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "no Vulkan queue allocation pool can satisfy allocation of %" PRIu64
      " bytes",
      (uint64_t)*allocation_size);
}

iree_status_t iree_hal_vulkan_allocator_allocate_queue_sparse_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_placement_t placement,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer, iree_host_size_t* out_bind_count,
    VkSparseMemoryBind** out_binds) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  IREE_ASSERT_ARGUMENT(placement.device);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_ASSERT_ARGUMENT(out_bind_count);
  IREE_ASSERT_ARGUMENT(out_binds);
  *out_buffer = NULL;
  *out_bind_count = 0;
  *out_binds = NULL;
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);

  iree_status_t status =
      iree_hal_vulkan_allocator_align_allocation_size(&allocation_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
        allocator, allocation_size,
        allocator->properties11.maxMemoryAllocationSize, &params);
  }

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, UINT32_MAX, &params, &memory_placement)) {
    status = iree_hal_vulkan_allocator_make_buffer_params_status(&params);
  }

  VkBuffer handle = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_create_buffer_handle(
        allocator, &params, allocation_size,
        VK_BUFFER_CREATE_SPARSE_BINDING_BIT,
        /*create_info_pnext=*/NULL, &handle);
  }

  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    iree_vkGetBufferMemoryRequirements(IREE_VULKAN_DEVICE(&allocator->syms),
                                       allocator->logical_device, handle,
                                       &memory_requirements);
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, memory_requirements.memoryTypeBits, &params,
          &memory_placement)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "allocator cannot find a Vulkan memory type "
                              "compatible with sparse buffer usage");
  }

  iree_device_size_t max_allocation_size = 0;
  if (iree_status_is_ok(status)) {
    max_allocation_size =
        iree_hal_vulkan_allocator_max_allocation_size_for_type(
            allocator, memory_placement.memory_type_index);
  }

  iree_hal_buffer_t* buffer = NULL;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  VkBuffer trace_handle = VK_NULL_HANDLE;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_sparse_buffer_create_pending_bind(
        &allocator->syms, allocator->logical_device, placement,
        memory_placement.memory_type, params.access, params.usage,
        allocation_size, byte_length, handle, memory_requirements,
        memory_placement.memory_type_index, max_allocation_size,
        iree_hal_vulkan_allocator_memory_allocate_flags(allocator,
                                                        params.usage),
        host_allocator, &buffer, out_bind_count, out_binds);
    if (iree_status_is_ok(status)) {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
      trace_handle = handle;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
      handle = VK_NULL_HANDLE;
    }
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_VULKAN_ALLOCATOR_ID, (void*)trace_handle,
                           (iree_host_size_t)allocation_size);
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
    iree_allocator_free(host_allocator, *out_binds);
    *out_binds = NULL;
    *out_bind_count = 0;
    if (handle) {
      iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                           allocator->logical_device, handle,
                           /*pAllocator=*/NULL);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_allocator_allocate_direct_buffer_with_memory_type_bits(
    iree_hal_vulkan_allocator_t* allocator, uint32_t allowed_memory_type_bits,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  const iree_device_size_t byte_length = allocation_size;
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)byte_length);

  iree_status_t status =
      iree_hal_vulkan_allocator_align_allocation_size(&allocation_size);

  bool use_sparse_allocation = false;
  if (iree_status_is_ok(status)) {
    use_sparse_allocation = !iree_hal_vulkan_allocator_allocation_size_in_range(
        allocator, allocation_size);
  }

  iree_hal_buffer_params_t compat_params = *params;
  if (iree_status_is_ok(status) && use_sparse_allocation) {
    status = iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
        allocator, allocation_size,
        allocator->properties11.maxMemoryAllocationSize, &compat_params);
  }

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, allowed_memory_type_bits, &compat_params,
          &memory_placement)) {
    status = iree_hal_vulkan_allocator_make_buffer_params_status(params);
  }

  VkBuffer handle = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    const VkBufferCreateFlags create_flags =
        use_sparse_allocation ? VK_BUFFER_CREATE_SPARSE_BINDING_BIT : 0;
    status = iree_hal_vulkan_allocator_create_buffer_handle(
        allocator, &compat_params, allocation_size, create_flags,
        /*create_info_pnext=*/NULL, &handle);
  }

  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    iree_vkGetBufferMemoryRequirements(IREE_VULKAN_DEVICE(&allocator->syms),
                                       allocator->logical_device, handle,
                                       &memory_requirements);
  }
  iree_device_size_t max_allocation_size = 0;
  if (iree_status_is_ok(status)) {
    max_allocation_size =
        iree_hal_vulkan_allocator_max_allocation_size_for_type(
            allocator, memory_placement.memory_type_index);
  }
  if (iree_status_is_ok(status) && !use_sparse_allocation &&
      memory_requirements.size > max_allocation_size) {
    iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                         allocator->logical_device, handle,
                         /*pAllocator=*/NULL);
    handle = VK_NULL_HANDLE;
    use_sparse_allocation = true;
    compat_params = *params;
    status = iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
        allocator, memory_requirements.size, max_allocation_size,
        &compat_params);
    if (iree_status_is_ok(status) &&
        !iree_hal_vulkan_allocator_resolve_memory_placement(
            allocator, allowed_memory_type_bits, &compat_params,
            &memory_placement)) {
      status = iree_hal_vulkan_allocator_make_buffer_params_status(params);
    }
    if (iree_status_is_ok(status)) {
      max_allocation_size =
          iree_hal_vulkan_allocator_max_allocation_size_for_type(
              allocator, memory_placement.memory_type_index);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_allocator_create_buffer_handle(
          allocator, &compat_params, allocation_size,
          VK_BUFFER_CREATE_SPARSE_BINDING_BIT, /*create_info_pnext=*/NULL,
          &handle);
    }
    if (iree_status_is_ok(status)) {
      iree_vkGetBufferMemoryRequirements(IREE_VULKAN_DEVICE(&allocator->syms),
                                         allocator->logical_device, handle,
                                         &memory_requirements);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_resolve_memory_placement(
                 allocator,
                 memory_requirements.memoryTypeBits & allowed_memory_type_bits,
                 &compat_params, &memory_placement)
                 ? iree_ok_status()
                 : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "allocator cannot find a Vulkan memory "
                                    "type compatible with buffer usage");
  }

  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  if (iree_status_is_ok(status) && !use_sparse_allocation) {
    status = iree_hal_vulkan_allocator_allocate_memory(
        allocator, &memory_placement, compat_params.usage, &memory_requirements,
        &device_memory);
  }
  if (iree_status_is_ok(status) && !use_sparse_allocation) {
    status = iree_vkBindBufferMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                                     allocator->logical_device, handle,
                                     device_memory, /*memoryOffset=*/0);
  }

  iree_hal_buffer_t* buffer = NULL;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  VkBuffer trace_handle = VK_NULL_HANDLE;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t buffer_placement = {
        .device = allocator->parent_device,
        .queue_affinity = compat_params.queue_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    if (use_sparse_allocation) {
      status = iree_hal_vulkan_sparse_buffer_create_bound_sync(
          &allocator->syms, allocator->logical_device,
          allocator->sparse_binding_queue, buffer_placement,
          memory_placement.memory_type, compat_params.access,
          compat_params.usage, allocation_size, byte_length, handle,
          memory_requirements, memory_placement.memory_type_index,
          max_allocation_size,
          iree_hal_vulkan_allocator_memory_allocate_flags(allocator,
                                                          compat_params.usage),
          allocator->host_allocator, &buffer);
      if (iree_status_is_ok(status)) {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
        trace_handle = handle;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
        handle = VK_NULL_HANDLE;
      }
    } else {
      status = iree_hal_vulkan_buffer_create(
          &allocator->syms, allocator->logical_device, buffer_placement,
          memory_placement.memory_type, compat_params.access,
          compat_params.usage, allocation_size, byte_length,
          memory_placement.memory_property_flags,
          allocator->properties2.properties.limits.nonCoherentAtomSize,
          device_memory, handle,
          iree_hal_vulkan_allocator_query_device_address(
              allocator, compat_params.usage, handle),
          iree_hal_buffer_release_callback_null(), allocator->host_allocator,
          &buffer);
      if (iree_status_is_ok(status)) {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
        trace_handle = handle;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
        device_memory = VK_NULL_HANDLE;
        handle = VK_NULL_HANDLE;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_VULKAN_ALLOCATOR_ID, (void*)trace_handle,
                           (iree_host_size_t)allocation_size);
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
    if (device_memory) {
      iree_vkFreeMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                        allocator->logical_device, device_memory,
                        /*pAllocator=*/NULL);
    }
    if (handle) {
      iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                           allocator->logical_device, handle,
                           /*pAllocator=*/NULL);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_allocator_allocate_direct_buffer_from_type(
    iree_hal_vulkan_allocator_t* allocator, uint32_t memory_type_index,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (memory_type_index >=
      allocator->memory_properties2.memoryProperties.memoryTypeCount) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan memory type index %u exceeds physical-device memory type "
        "count %u",
        memory_type_index,
        allocator->memory_properties2.memoryProperties.memoryTypeCount);
  }
  return iree_hal_vulkan_allocator_allocate_direct_buffer_with_memory_type_bits(
      allocator, 1u << memory_type_index, params, allocation_size, out_buffer);
}

static iree_status_t iree_hal_vulkan_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  const iree_device_size_t byte_length = allocation_size;
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);

  iree_status_t status =
      iree_hal_vulkan_allocator_align_allocation_size(&allocation_size);

  iree_hal_buffer_params_t compat_params = *params;
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_allocation_size_in_range(allocator,
                                                          allocation_size)) {
    status = iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
        allocator, allocation_size,
        allocator->properties11.maxMemoryAllocationSize, &compat_params);
  }
  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, UINT32_MAX, &compat_params, &memory_placement)) {
    status = iree_hal_vulkan_allocator_make_buffer_params_status(params);
  }
  if (iree_status_is_ok(status)) {
    const iree_device_size_t max_allocation_size =
        iree_hal_vulkan_allocator_max_allocation_size_for_type(
            allocator, memory_placement.memory_type_index);
    if (allocation_size > max_allocation_size) {
      status = iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
          allocator, allocation_size, max_allocation_size, &compat_params);
      if (iree_status_is_ok(status) &&
          !iree_hal_vulkan_allocator_resolve_memory_placement(
              allocator, UINT32_MAX, &compat_params, &memory_placement)) {
        status = iree_hal_vulkan_allocator_make_buffer_params_status(params);
      }
    }
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_pool_t* pool = iree_hal_vulkan_allocator_select_default_pool(
        allocator, compat_params, allocation_size);
    if (pool) {
      status = iree_hal_pool_allocate_buffer(
          pool, compat_params, allocation_size, /*requester_frontier=*/NULL,
          iree_infinite_timeout(), &buffer);
      if (iree_status_is_ok(status) &&
          iree_hal_buffer_byte_length(buffer) != byte_length) {
        iree_hal_buffer_t* allocation_buffer = buffer;
        buffer = NULL;
        status = iree_hal_buffer_subspan(allocation_buffer, /*byte_offset=*/0,
                                         byte_length, allocator->host_allocator,
                                         &buffer);
        iree_hal_buffer_release(allocation_buffer);
      }
    } else {
      status =
          iree_hal_vulkan_allocator_allocate_direct_buffer_with_memory_type_bits(
              allocator, 1u << memory_placement.memory_type_index,
              &compat_params, allocation_size, &buffer);
    }
  }

  if (iree_status_is_ok(status)) {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, iree_hal_buffer_memory_type(buffer),
        iree_hal_buffer_byte_length(buffer)));
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, (int64_t)iree_hal_buffer_allocation_size(base_buffer));

  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
      iree_hal_buffer_byte_length(base_buffer)));

  iree_hal_buffer_destroy(base_buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_allocator_validate_host_allocation_import(
    const iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_buffer_params_t* params,
    const iree_hal_external_buffer_t* external_buffer) {
  if (external_buffer->flags != IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported Vulkan external buffer flags: 0x%x",
                            external_buffer->flags);
  }
  if (external_buffer->type != IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Vulkan external buffer type is unsupported");
  }
  if (!iree_hal_vulkan_allocator_supports_host_allocation_import(allocator)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan host allocation import requires VK_EXT_external_memory_host");
  }
  if (!external_buffer->handle.host_allocation.ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan host allocation import requires a "
                            "non-null pointer");
  }
  if (external_buffer->size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan host allocation import requires a "
                            "non-zero size");
  }
  if (!iree_device_size_is_valid_alignment(params->min_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "requested Vulkan import alignment %" PRIu64
                            " is not a power-of-two",
                            (uint64_t)params->min_alignment);
  }
  if (params->min_alignment != 0 &&
      (params->min_alignment > IREE_HOST_SIZE_MAX ||
       !iree_host_ptr_has_alignment(external_buffer->handle.host_allocation.ptr,
                                    (iree_host_size_t)params->min_alignment))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host allocation import pointer does not satisfy requested Vulkan "
        "alignment %" PRIu64,
        (uint64_t)params->min_alignment);
  }
  const VkDeviceSize import_alignment =
      allocator->external_memory_host_properties
          .minImportedHostPointerAlignment;
  if (import_alignment != 0 &&
      (import_alignment > IREE_HOST_SIZE_MAX ||
       !iree_host_ptr_has_alignment(external_buffer->handle.host_allocation.ptr,
                                    (iree_host_size_t)import_alignment))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host allocation import pointer does not satisfy Vulkan "
        "minImportedHostPointerAlignment %" PRIu64,
        (uint64_t)import_alignment);
  }
  if (import_alignment != 0 &&
      (import_alignment > IREE_DEVICE_SIZE_MAX ||
       !iree_device_size_has_alignment(external_buffer->size,
                                       (iree_device_size_t)import_alignment))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host allocation import size %" PRIu64
        " does not satisfy Vulkan minImportedHostPointerAlignment %" PRIu64,
        (uint64_t)external_buffer->size, (uint64_t)import_alignment);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_allocator_allocate_imported_host_memory(
    iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_vulkan_allocator_memory_placement_t* memory_placement,
    iree_hal_buffer_usage_t hal_usage, VkDeviceSize allocation_size,
    void* host_pointer, VkDeviceMemory* out_device_memory) {
  *out_device_memory = VK_NULL_HANDLE;

  VkImportMemoryHostPointerInfoEXT import_info = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
      .pHostPointer = host_pointer,
  };
  VkMemoryAllocateFlagsInfo allocate_flags_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .pNext = &import_info,
      .flags =
          iree_hal_vulkan_allocator_memory_allocate_flags(allocator, hal_usage),
  };
  VkMemoryAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &allocate_flags_info,
      .allocationSize = allocation_size,
      .memoryTypeIndex = memory_placement->memory_type_index,
  };
  return iree_vkAllocateMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                               allocator->logical_device, &allocate_info,
                               /*pAllocator=*/NULL, out_device_memory);
}

static iree_status_t iree_hal_vulkan_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(external_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)external_buffer->size);

  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  iree_status_t status =
      iree_hal_vulkan_allocator_validate_host_allocation_import(
          allocator, params, external_buffer);

  iree_device_size_t allocation_size = external_buffer->size;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_align_allocation_size(&allocation_size);
  }

  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  if (iree_status_is_ok(status)) {
    compatibility = iree_hal_vulkan_allocator_query_buffer_compatibility(
        base_allocator, &compat_params, &allocation_size);
    if (!iree_all_bits_set(compatibility,
                           IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)) {
#if IREE_STATUS_MODE
      iree_bitfield_string_temp_t temp0, temp1, temp2;
      iree_string_view_t memory_type_str =
          iree_hal_memory_type_format(params->type, &temp0);
      iree_string_view_t usage_str =
          iree_hal_buffer_usage_format(params->usage, &temp1);
      iree_string_view_t compatibility_str =
          iree_hal_buffer_compatibility_format(compatibility, &temp2);
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "allocator cannot import a Vulkan host allocation with the given "
          "parameters; memory_type=%.*s, usage=%.*s, compatibility=%.*s",
          (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
          usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "allocator cannot import a Vulkan host allocation with the given "
          "parameters");
#endif  // IREE_STATUS_MODE
    }
  }

  VkMemoryHostPointerPropertiesEXT host_pointer_properties = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
  };
  if (iree_status_is_ok(status)) {
    status = iree_vkGetMemoryHostPointerPropertiesEXT(
        IREE_VULKAN_DEVICE(&allocator->syms), allocator->logical_device,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        external_buffer->handle.host_allocation.ptr, &host_pointer_properties);
  }

  VkExternalMemoryBufferCreateInfo external_create_info = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
  };
  VkBuffer handle = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_create_buffer_handle(
        allocator, &compat_params, allocation_size,
        /*create_flags=*/0, &external_create_info, &handle);
  }

  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    iree_vkGetBufferMemoryRequirements(IREE_VULKAN_DEVICE(&allocator->syms),
                                       allocator->logical_device, handle,
                                       &memory_requirements);
  }
  if (iree_status_is_ok(status) &&
      memory_requirements.size > external_buffer->size) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan host allocation import requires %" PRIu64
        " bytes but the external allocation only provides %" PRIu64,
        (uint64_t)memory_requirements.size, (uint64_t)external_buffer->size);
  }

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator,
          memory_requirements.memoryTypeBits &
              host_pointer_properties.memoryTypeBits,
          &compat_params, &memory_placement)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot find a Vulkan memory type compatible with the "
        "imported host pointer");
  }

  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_allocate_imported_host_memory(
        allocator, &memory_placement, compat_params.usage,
        memory_requirements.size, external_buffer->handle.host_allocation.ptr,
        &device_memory);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vkBindBufferMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                                     allocator->logical_device, handle,
                                     device_memory, /*memoryOffset=*/0);
  }

  iree_hal_buffer_t* buffer = NULL;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  VkBuffer trace_handle = VK_NULL_HANDLE;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t placement = {
        .device = allocator->parent_device,
        .queue_affinity = compat_params.queue_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_vulkan_buffer_create(
        &allocator->syms, allocator->logical_device, placement,
        memory_placement.memory_type, compat_params.access, compat_params.usage,
        allocation_size, external_buffer->size,
        memory_placement.memory_property_flags,
        allocator->properties2.properties.limits.nonCoherentAtomSize,
        device_memory, handle,
        iree_hal_vulkan_allocator_query_device_address(
            allocator, compat_params.usage, handle),
        release_callback, allocator->host_allocator, &buffer);
    if (iree_status_is_ok(status)) {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
      trace_handle = handle;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
      device_memory = VK_NULL_HANDLE;
      handle = VK_NULL_HANDLE;
    }
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_VULKAN_ALLOCATOR_ID, (void*)trace_handle,
                           (iree_host_size_t)allocation_size);
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
    if (device_memory) {
      iree_vkFreeMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                        allocator->logical_device, device_memory,
                        /*pAllocator=*/NULL);
    }
    if (handle) {
      iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                           allocator->logical_device, handle,
                           /*pAllocator=*/NULL);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
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

static iree_status_t iree_hal_vulkan_allocator_validate_sparse_range(
    iree_device_size_t offset, iree_device_size_t size,
    iree_device_size_t container_size, iree_device_size_t page_size,
    const char* range_name) {
  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan sparse %s range must be non-zero",
                            range_name);
  }
  if (!iree_device_size_has_alignment(offset, page_size) ||
      !iree_device_size_has_alignment(size, page_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan sparse %s range offset %" PRIu64 " and size %" PRIu64
        " must be aligned to page size %" PRIu64,
        range_name, (uint64_t)offset, (uint64_t)size, (uint64_t)page_size);
  }
  iree_device_size_t range_end = 0;
  if (!iree_device_size_checked_add(offset, size, &range_end) ||
      range_end > container_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan sparse %s range [%" PRIu64 ", %" PRIu64
                            ") exceeds container size %" PRIu64,
                            range_name, (uint64_t)offset, (uint64_t)range_end,
                            (uint64_t)container_size);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_allocator_validate_sparse_page_size(
    VkMemoryRequirements memory_requirements) {
  if (memory_requirements.alignment == 0 ||
      memory_requirements.alignment > IREE_DEVICE_SIZE_MAX ||
      !iree_device_size_is_valid_alignment(
          (iree_device_size_t)memory_requirements.alignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan sparse buffer reported invalid page size %" PRIu64,
        (uint64_t)memory_requirements.alignment);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_allocator_allocate_virtual_memory_mapping(
    iree_hal_vulkan_allocator_t* allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size,
    iree_hal_vulkan_allocator_virtual_memory_mapping_t** out_mapping) {
  *out_mapping = NULL;
  iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator->host_allocator, sizeof(*mapping), (void**)&mapping));
  mapping->next = NULL;
  mapping->virtual_buffer = virtual_buffer;
  mapping->physical_memory = physical_memory;
  mapping->virtual_offset = virtual_offset;
  mapping->physical_offset = physical_offset;
  mapping->size = size;
  *out_mapping = mapping;
  return iree_ok_status();
}

static bool iree_hal_vulkan_allocator_has_virtual_memory_mappings_locked(
    iree_hal_vulkan_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer) {
  for (iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping =
           allocator->virtual_memory_mappings;
       mapping; mapping = mapping->next) {
    if (mapping->virtual_buffer == virtual_buffer) {
      return true;
    }
  }
  return false;
}

static iree_status_t
iree_hal_vulkan_allocator_validate_virtual_memory_range_unmapped_locked(
    iree_hal_vulkan_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  for (iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping =
           allocator->virtual_memory_mappings;
       mapping; mapping = mapping->next) {
    if (mapping->virtual_buffer != virtual_buffer) {
      continue;
    }
    if (iree_hal_vulkan_allocator_ranges_overlap(
            virtual_offset, size, mapping->virtual_offset, mapping->size)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan sparse virtual memory range [%" PRIu64 ", %" PRIu64
          ") overlaps existing mapping [%" PRIu64 ", %" PRIu64 ")",
          (uint64_t)virtual_offset, (uint64_t)(virtual_offset + size),
          (uint64_t)mapping->virtual_offset,
          (uint64_t)(mapping->virtual_offset + mapping->size));
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_allocator_validate_virtual_memory_range_mapped_locked(
    iree_hal_vulkan_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  const iree_device_size_t range_end = virtual_offset + size;
  iree_device_size_t cursor = virtual_offset;
  while (cursor < range_end) {
    const iree_hal_vulkan_allocator_virtual_memory_mapping_t* covering_mapping =
        NULL;
    for (iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping =
             allocator->virtual_memory_mappings;
         mapping; mapping = mapping->next) {
      const iree_device_size_t mapping_end =
          mapping->virtual_offset + mapping->size;
      if (mapping->virtual_buffer == virtual_buffer &&
          mapping->virtual_offset <= cursor && mapping_end > cursor) {
        covering_mapping = mapping;
        break;
      }
    }
    if (!covering_mapping) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Vulkan sparse virtual memory range [%" PRIu64
                              ", %" PRIu64 ") is not fully mapped",
                              (uint64_t)virtual_offset, (uint64_t)range_end);
    }
    const iree_device_size_t mapping_end =
        covering_mapping->virtual_offset + covering_mapping->size;
    cursor = mapping_end < range_end ? mapping_end : range_end;
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_allocator_insert_virtual_memory_mapping_locked(
    iree_hal_vulkan_allocator_t* allocator,
    iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping) {
  IREE_ASSERT(mapping->physical_memory->mapped_size <=
              IREE_DEVICE_SIZE_MAX - mapping->size);
  mapping->next = allocator->virtual_memory_mappings;
  allocator->virtual_memory_mappings = mapping;
  mapping->physical_memory->mapped_size += mapping->size;
}

static void iree_hal_vulkan_allocator_remove_virtual_memory_mapping_locked(
    iree_hal_vulkan_allocator_t* allocator,
    iree_hal_vulkan_allocator_virtual_memory_mapping_t** mapping_ptr,
    iree_device_size_t remove_begin, iree_device_size_t remove_end,
    iree_hal_vulkan_allocator_virtual_memory_mapping_t* split_tail_mapping,
    bool* split_tail_mapping_used) {
  iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping = *mapping_ptr;
  const iree_device_size_t mapping_begin = mapping->virtual_offset;
  const iree_device_size_t mapping_end =
      mapping->virtual_offset + mapping->size;
  const iree_device_size_t overlap_begin =
      remove_begin > mapping_begin ? remove_begin : mapping_begin;
  const iree_device_size_t overlap_end =
      remove_end < mapping_end ? remove_end : mapping_end;
  const iree_device_size_t overlap_size = overlap_end - overlap_begin;

  IREE_ASSERT(mapping->physical_memory->mapped_size >= overlap_size);
  mapping->physical_memory->mapped_size -= overlap_size;
  if (overlap_begin == mapping_begin && overlap_end == mapping_end) {
    *mapping_ptr = mapping->next;
    iree_allocator_free(allocator->host_allocator, mapping);
  } else if (overlap_begin == mapping_begin) {
    mapping->virtual_offset = overlap_end;
    mapping->physical_offset += overlap_size;
    mapping->size = mapping_end - overlap_end;
    *mapping_ptr = mapping;
  } else if (overlap_end == mapping_end) {
    mapping->size = overlap_begin - mapping_begin;
    *mapping_ptr = mapping;
  } else {
    IREE_ASSERT(split_tail_mapping != NULL);
    IREE_ASSERT(!*split_tail_mapping_used);
    split_tail_mapping->next = mapping->next;
    split_tail_mapping->virtual_buffer = mapping->virtual_buffer;
    split_tail_mapping->physical_memory = mapping->physical_memory;
    split_tail_mapping->virtual_offset = overlap_end;
    split_tail_mapping->physical_offset =
        mapping->physical_offset + (overlap_end - mapping_begin);
    split_tail_mapping->size = mapping_end - overlap_end;
    mapping->size = overlap_begin - mapping_begin;
    mapping->next = split_tail_mapping;
    *split_tail_mapping_used = true;
    *mapping_ptr = mapping;
  }
}

static void iree_hal_vulkan_allocator_unmap_virtual_memory_range_locked(
    iree_hal_vulkan_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_vulkan_allocator_virtual_memory_mapping_t* split_tail_mapping,
    bool* split_tail_mapping_used) {
  const iree_device_size_t range_end = virtual_offset + size;
  iree_hal_vulkan_allocator_virtual_memory_mapping_t** mapping_ptr =
      &allocator->virtual_memory_mappings;
  while (*mapping_ptr) {
    iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping = *mapping_ptr;
    if (mapping->virtual_buffer != virtual_buffer ||
        !iree_hal_vulkan_allocator_ranges_overlap(
            virtual_offset, size, mapping->virtual_offset, mapping->size)) {
      mapping_ptr = &mapping->next;
      continue;
    }

    iree_hal_vulkan_allocator_remove_virtual_memory_mapping_locked(
        allocator, mapping_ptr, virtual_offset, range_end, split_tail_mapping,
        split_tail_mapping_used);
    if (*mapping_ptr == mapping) {
      mapping_ptr = &mapping->next;
    }
  }
}

static bool iree_hal_vulkan_allocator_supports_virtual_memory(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  return iree_hal_vulkan_allocator_supports_sparse_virtual_memory(allocator);
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params,
    iree_device_size_t* IREE_RESTRICT out_minimum_page_size,
    iree_device_size_t* IREE_RESTRICT out_recommended_page_size) {
  *out_minimum_page_size = 0;
  *out_recommended_page_size = 0;
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);

  iree_status_t status =
      iree_hal_vulkan_allocator_prepare_virtual_memory_params(allocator,
                                                              &params);
  VkBuffer handle = VK_NULL_HANDLE;
  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_create_sparse_virtual_handle(
        allocator, &params, /*allocation_size=*/4, &handle,
        &memory_requirements);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_validate_sparse_page_size(
        memory_requirements);
  }
  if (handle) {
    iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                         allocator->logical_device, handle,
                         /*pAllocator=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_minimum_page_size = (iree_device_size_t)memory_requirements.alignment;
    *out_recommended_page_size =
        (iree_device_size_t)memory_requirements.alignment;
  }
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
    iree_hal_buffer_t** IREE_RESTRICT out_virtual_buffer) {
  *out_virtual_buffer = NULL;
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  if (size == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan virtual memory reservations must be non-zero");
  }

  iree_hal_buffer_params_t params =
      iree_hal_vulkan_allocator_virtual_buffer_params(queue_affinity);
  iree_status_t status =
      iree_hal_vulkan_allocator_prepare_virtual_memory_params(allocator,
                                                              &params);

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  memset(&memory_placement, 0, sizeof(memory_placement));
  VkBuffer handle = VK_NULL_HANDLE;
  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_create_sparse_virtual_handle(
        allocator, &params, size, &handle, &memory_requirements);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_validate_sparse_page_size(
        memory_requirements);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_validate_sparse_range(
        /*offset=*/0, size, size,
        (iree_device_size_t)memory_requirements.alignment, "reservation");
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, memory_requirements.memoryTypeBits, &params,
          &memory_placement)) {
    status = iree_hal_vulkan_allocator_make_buffer_params_status(&params);
  }

  iree_hal_buffer_t* virtual_buffer = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t placement = {
        .device = allocator->parent_device,
        .queue_affinity = params.queue_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_vulkan_sparse_buffer_create_unbound(
        &allocator->syms, allocator->logical_device, placement,
        memory_placement.memory_type, params.access, params.usage, size, size,
        handle, memory_requirements, allocator->host_allocator,
        &virtual_buffer);
    if (iree_status_is_ok(status)) {
      handle = VK_NULL_HANDLE;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_virtual_buffer = virtual_buffer;
  } else {
    iree_hal_buffer_release(virtual_buffer);
    if (handle) {
      iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                           allocator->logical_device, handle,
                           /*pAllocator=*/NULL);
    }
  }
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_release(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_buffer(virtual_buffer));

  iree_slim_mutex_lock(&allocator->virtual_memory_mutex);
  iree_status_t status = iree_ok_status();
  if (iree_hal_vulkan_allocator_has_virtual_memory_mappings_locked(
          allocator, virtual_buffer)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan sparse virtual memory reservations must be fully unmapped "
        "before release");
  } else {
    iree_hal_buffer_destroy(virtual_buffer);
  }
  iree_slim_mutex_unlock(&allocator->virtual_memory_mutex);
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_physical_memory_allocate(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params, iree_device_size_t size,
    iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** IREE_RESTRICT out_physical_memory) {
  *out_physical_memory = NULL;
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  if (size == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan physical memory allocations must be non-zero");
  }

  iree_status_t status =
      iree_hal_vulkan_allocator_prepare_virtual_memory_params(allocator,
                                                              &params);
  VkBuffer scratch_handle = VK_NULL_HANDLE;
  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_create_sparse_virtual_handle(
        allocator, &params, size, &scratch_handle, &memory_requirements);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_validate_sparse_page_size(
        memory_requirements);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_validate_sparse_range(
        /*offset=*/0, size, size,
        (iree_device_size_t)memory_requirements.alignment,
        "physical allocation");
  }

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  memset(&memory_placement, 0, sizeof(memory_placement));
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, memory_requirements.memoryTypeBits, &params,
          &memory_placement)) {
    status = iree_hal_vulkan_allocator_make_buffer_params_status(&params);
  }
  if (scratch_handle) {
    iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                         allocator->logical_device, scratch_handle,
                         /*pAllocator=*/NULL);
  }

  if (iree_status_is_ok(status)) {
    const iree_device_size_t max_allocation_size =
        iree_hal_vulkan_allocator_max_allocation_size_for_type(
            allocator, memory_placement.memory_type_index);
    if (size > max_allocation_size) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan physical memory allocation size %" PRIu64
          " exceeds per-memory-type max allocation size %" PRIu64,
          (uint64_t)size, (uint64_t)max_allocation_size);
    }
  }

  iree_hal_physical_memory_t* physical_memory = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*physical_memory),
                                   (void**)&physical_memory);
  }
  if (iree_status_is_ok(status)) {
    memset(physical_memory, 0, sizeof(*physical_memory));
    physical_memory->host_allocator = host_allocator;
    physical_memory->syms = allocator->syms;
    physical_memory->logical_device = allocator->logical_device;
    physical_memory->owner_allocator = allocator;
    physical_memory->allocation_size = size;
    physical_memory->memory_type_index = memory_placement.memory_type_index;
    physical_memory->memory_type = memory_placement.memory_type;

    VkMemoryAllocateFlagsInfo allocate_flags_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = iree_hal_vulkan_allocator_memory_allocate_flags(
            allocator, iree_hal_vulkan_allocator_virtual_buffer_usage()),
    };
    VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &allocate_flags_info,
        .allocationSize = size,
        .memoryTypeIndex = memory_placement.memory_type_index,
    };
    status = iree_vkAllocateMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                                   allocator->logical_device, &allocate_info,
                                   /*pAllocator=*/NULL,
                                   &physical_memory->device_memory);
  }

  if (iree_status_is_ok(status)) {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, physical_memory->memory_type,
        physical_memory->allocation_size));
    *out_physical_memory = physical_memory;
  } else {
    if (physical_memory && physical_memory->device_memory) {
      iree_vkFreeMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                        allocator->logical_device,
                        physical_memory->device_memory,
                        /*pAllocator=*/NULL);
    }
    iree_allocator_free(host_allocator, physical_memory);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_physical_memory_free(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  if (physical_memory->owner_allocator != allocator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan physical memory allocation belongs to a "
                            "different allocator");
  }

  iree_slim_mutex_lock(&allocator->virtual_memory_mutex);
  iree_status_t status = iree_ok_status();
  if (physical_memory->mapped_size != 0) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan physical memory allocation still has %" PRIu64 " mapped bytes",
        (uint64_t)physical_memory->mapped_size);
  } else {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
        &allocator->statistics, physical_memory->memory_type,
        physical_memory->allocation_size));
    iree_vkFreeMemory(IREE_VULKAN_DEVICE(&physical_memory->syms),
                      physical_memory->logical_device,
                      physical_memory->device_memory,
                      /*pAllocator=*/NULL);
    iree_allocator_free(physical_memory->host_allocator, physical_memory);
  }
  iree_slim_mutex_unlock(&allocator->virtual_memory_mutex);
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_map(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_buffer(virtual_buffer));
  if (physical_memory->owner_allocator != allocator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan physical memory allocation belongs to a "
                            "different allocator");
  }

  VkDeviceMemory ignored_memory = VK_NULL_HANDLE;
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_sparse_buffer_handle(
      virtual_buffer, &ignored_memory, &handle));
  VkMemoryRequirements memory_requirements = {0};
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_sparse_buffer_memory_requirements(
      virtual_buffer, &memory_requirements));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_validate_sparse_page_size(memory_requirements));
  const iree_device_size_t page_size =
      (iree_device_size_t)memory_requirements.alignment;

  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_validate_sparse_range(
      virtual_offset, size, iree_hal_buffer_allocation_size(virtual_buffer),
      page_size, "virtual map"));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_validate_sparse_range(
      physical_offset, size, physical_memory->allocation_size, page_size,
      "physical map"));
  if (!iree_all_bits_set(memory_requirements.memoryTypeBits,
                         1u << physical_memory->memory_type_index)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan physical memory type %u is not compatible with sparse virtual "
        "buffer memory type bits 0x%x",
        physical_memory->memory_type_index, memory_requirements.memoryTypeBits);
  }

  iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_allocate_virtual_memory_mapping(
          allocator, virtual_buffer, virtual_offset, physical_memory,
          physical_offset, size, &mapping));

  const VkSparseMemoryBind bind = {
      .resourceOffset = (VkDeviceSize)virtual_offset,
      .size = (VkDeviceSize)size,
      .memory = physical_memory->device_memory,
      .memoryOffset = (VkDeviceSize)physical_offset,
      .flags = 0,
  };
  iree_slim_mutex_lock(&allocator->virtual_memory_mutex);
  iree_status_t status =
      iree_hal_vulkan_allocator_validate_virtual_memory_range_unmapped_locked(
          allocator, virtual_buffer, virtual_offset, size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_sparse_buffer_bind_sync(
        allocator->sparse_binding_queue,
        iree_hal_buffer_allocation_placement(virtual_buffer), handle,
        /*bind_count=*/1, &bind);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_allocator_insert_virtual_memory_mapping_locked(allocator,
                                                                   mapping);
    mapping = NULL;
  }
  iree_slim_mutex_unlock(&allocator->virtual_memory_mutex);
  iree_allocator_free(allocator->host_allocator, mapping);
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_buffer(virtual_buffer));

  VkDeviceMemory ignored_memory = VK_NULL_HANDLE;
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_sparse_buffer_handle(
      virtual_buffer, &ignored_memory, &handle));
  VkMemoryRequirements memory_requirements = {0};
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_sparse_buffer_memory_requirements(
      virtual_buffer, &memory_requirements));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_validate_sparse_page_size(memory_requirements));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_validate_sparse_range(
      virtual_offset, size, iree_hal_buffer_allocation_size(virtual_buffer),
      (iree_device_size_t)memory_requirements.alignment, "virtual unmap"));

  iree_hal_vulkan_allocator_virtual_memory_mapping_t* split_tail_mapping = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_allocate_virtual_memory_mapping(
          allocator, virtual_buffer, virtual_offset, /*physical_memory=*/NULL,
          /*physical_offset=*/0, size, &split_tail_mapping));

  const VkSparseMemoryBind bind = {
      .resourceOffset = (VkDeviceSize)virtual_offset,
      .size = (VkDeviceSize)size,
      .memory = VK_NULL_HANDLE,
      .memoryOffset = 0,
      .flags = 0,
  };
  iree_slim_mutex_lock(&allocator->virtual_memory_mutex);
  iree_status_t status =
      iree_hal_vulkan_allocator_validate_virtual_memory_range_mapped_locked(
          allocator, virtual_buffer, virtual_offset, size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_sparse_buffer_bind_sync(
        allocator->sparse_binding_queue,
        iree_hal_buffer_allocation_placement(virtual_buffer), handle,
        /*bind_count=*/1, &bind);
  }
  bool split_tail_mapping_used = false;
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_allocator_unmap_virtual_memory_range_locked(
        allocator, virtual_buffer, virtual_offset, size, split_tail_mapping,
        &split_tail_mapping_used);
  }
  iree_slim_mutex_unlock(&allocator->virtual_memory_mutex);
  if (!split_tail_mapping_used) {
    iree_allocator_free(allocator->host_allocator, split_tail_mapping);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_protect(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_protection_t protection) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_buffer(virtual_buffer));
  (void)queue_affinity;
  VkMemoryRequirements memory_requirements = {0};
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_sparse_buffer_memory_requirements(
      virtual_buffer, &memory_requirements));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_validate_sparse_page_size(memory_requirements));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_validate_sparse_range(
      virtual_offset, size, iree_hal_buffer_allocation_size(virtual_buffer),
      (iree_device_size_t)memory_requirements.alignment, "virtual protect"));
  if (protection == IREE_HAL_MEMORY_PROTECTION_READ_WRITE) {
    iree_slim_mutex_lock(&allocator->virtual_memory_mutex);
    iree_status_t status =
        iree_hal_vulkan_allocator_validate_virtual_memory_range_mapped_locked(
            allocator, virtual_buffer, virtual_offset, size);
    iree_slim_mutex_unlock(&allocator->virtual_memory_mutex);
    return status;
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "Vulkan sparse virtual memory cannot enforce protection flags 0x%" PRIx64
      "; use unmap to revoke access",
      protection);
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_advise(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_advice_t advice) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_buffer(virtual_buffer));
  (void)queue_affinity;
  (void)advice;
  iree_device_size_t range_end = 0;
  if (!iree_device_size_checked_add(virtual_offset, size, &range_end) ||
      range_end > iree_hal_buffer_allocation_size(virtual_buffer)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan sparse virtual memory advice range [%" PRIu64 ", %" PRIu64
        ") exceeds reservation size %" PRIu64,
        (uint64_t)virtual_offset, (uint64_t)range_end,
        (uint64_t)iree_hal_buffer_allocation_size(virtual_buffer));
  }
  return iree_ok_status();
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
