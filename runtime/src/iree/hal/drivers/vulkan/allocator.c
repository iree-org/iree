// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/allocator.h"

#include <string.h>

#include "iree/hal/drivers/vulkan/buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_allocator_t
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_VULKAN_ALLOCATOR_ID = "iree-hal-vulkan-unpooled";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_vulkan_allocator_memory_placement_t {
  // Vulkan memory type index selected for the allocation.
  uint32_t memory_type_index;

  // Vulkan memory property flags for |memory_type_index|.
  VkMemoryPropertyFlags memory_property_flags;

  // HAL memory type exposed by allocated buffers.
  iree_hal_memory_type_t memory_type;
} iree_hal_vulkan_allocator_memory_placement_t;

typedef struct iree_hal_vulkan_allocator_t {
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

  // Physical-device memory properties captured during logical-device creation.
  VkPhysicalDeviceMemoryProperties2 memory_properties2;

  // HAL feature bits enabled on the logical device.
  iree_hal_vulkan_features_t enabled_features;

  // Queue affinity bits supported by this logical device.
  iree_hal_queue_affinity_t queue_affinity_mask;

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
    iree_hal_device_t* parent_device, const iree_hal_vulkan_device_syms_t* syms,
    VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_queue_affinity_t queue_affinity_mask,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(parent_device);
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(logical_device);
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
  allocator->parent_device = parent_device;
  allocator->syms = *syms;
  allocator->logical_device = logical_device;
  allocator->properties2 = physical_device->properties2;
  allocator->properties2.pNext = NULL;
  allocator->properties11 = physical_device->properties11;
  allocator->properties11.pNext = NULL;
  allocator->memory_properties2 = physical_device->memory_properties2;
  allocator->enabled_features = enabled_features;
  allocator->queue_affinity_mask = queue_affinity_mask;

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
    const iree_hal_memory_type_t hal_memory_type =
        iree_hal_vulkan_memory_type_from_vk(memory_type->propertyFlags);
    heaps[i] = (iree_hal_allocator_memory_heap_t){
        .type = hal_memory_type,
        .allowed_usage =
            iree_hal_vulkan_allowed_usage_from_memory_type(hal_memory_type),
        .max_allocation_size = iree_hal_vulkan_allocator_max_allocation_size(
            allocator, memory_heap),
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

static iree_hal_buffer_compatibility_t
iree_hal_vulkan_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  if (!iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, UINT32_MAX, params, &memory_placement)) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }
  if (!iree_status_is_ok(
          iree_hal_vulkan_allocator_align_allocation_size(allocation_size))) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }
  if (!iree_hal_vulkan_allocator_allocation_size_in_range(allocator,
                                                          *allocation_size)) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
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
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES)) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }
  return usage;
}

static iree_status_t iree_hal_vulkan_allocator_create_buffer_handle(
    iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    VkBuffer* out_buffer) {
  *out_buffer = VK_NULL_HANDLE;

  VkBufferCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = (VkDeviceSize)allocation_size,
      .usage = iree_hal_vulkan_buffer_usage_from_hal(allocator, params->usage),
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  return iree_vkCreateBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                             allocator->logical_device, &create_info,
                             /*pAllocator=*/NULL, out_buffer);
}

static iree_status_t iree_hal_vulkan_allocator_allocate_memory(
    iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_vulkan_allocator_memory_placement_t* memory_placement,
    const VkMemoryRequirements* memory_requirements,
    VkDeviceMemory* out_device_memory) {
  *out_device_memory = VK_NULL_HANDLE;

  VkMemoryAllocateFlagsInfo allocate_flags_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = iree_all_bits_set(
                   allocator->enabled_features,
                   IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES)
                   ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
                   : 0,
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
    iree_hal_vulkan_allocator_t* allocator, VkBuffer handle) {
  VkBufferDeviceAddressInfo address_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = handle,
  };
  return iree_vkGetBufferDeviceAddress(IREE_VULKAN_DEVICE(&allocator->syms),
                                       allocator->logical_device,
                                       &address_info);
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
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)byte_length);

  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  if (!iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, UINT32_MAX, &compat_params, &memory_placement)) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t memory_type_str =
        iree_hal_memory_type_format(params->type, &temp0);
    iree_string_view_t usage_str =
        iree_hal_buffer_usage_format(params->usage, &temp1);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a Vulkan buffer with the given parameters; "
        "memory_type=%.*s, usage=%.*s",
        (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
        usage_str.data);
#else
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a Vulkan buffer with the given parameters");
#endif  // IREE_STATUS_MODE
  }

  iree_status_t status =
      iree_hal_vulkan_allocator_align_allocation_size(&allocation_size);
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_allocation_size_in_range(allocator,
                                                          allocation_size)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan dense allocation size %" PRIu64
        " exceeds maxMemoryAllocationSize %" PRIu64,
        (uint64_t)allocation_size,
        (uint64_t)allocator->properties11.maxMemoryAllocationSize);
  }

  VkBuffer handle = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_create_buffer_handle(
        allocator, &compat_params, allocation_size, &handle);
  }

  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    iree_vkGetBufferMemoryRequirements(IREE_VULKAN_DEVICE(&allocator->syms),
                                       allocator->logical_device, handle,
                                       &memory_requirements);
    if (!iree_hal_vulkan_allocator_allocation_size_in_range(
            allocator, memory_requirements.size)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan memory requirements size %" PRIu64
          " exceeds maxMemoryAllocationSize %" PRIu64,
          (uint64_t)memory_requirements.size,
          (uint64_t)allocator->properties11.maxMemoryAllocationSize);
    }
  }
  if (iree_status_is_ok(status)) {
    compat_params = *params;
    status = iree_hal_vulkan_allocator_resolve_memory_placement(
                 allocator, memory_requirements.memoryTypeBits, &compat_params,
                 &memory_placement)
                 ? iree_ok_status()
                 : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "allocator cannot find a Vulkan memory "
                                    "type compatible with buffer usage");
  }

  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_allocate_memory(
        allocator, &memory_placement, &memory_requirements, &device_memory);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vkBindBufferMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                                     allocator->logical_device, handle,
                                     device_memory, /*memoryOffset=*/0);
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    const VkDeviceAddress device_address =
        iree_hal_vulkan_allocator_query_device_address(allocator, handle);
    const iree_hal_buffer_placement_t buffer_placement = {
        .device = allocator->parent_device,
        .queue_affinity = compat_params.queue_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_vulkan_buffer_create(
        &allocator->syms, allocator->logical_device, buffer_placement,
        memory_placement.memory_type, compat_params.access, compat_params.usage,
        allocation_size, byte_length, memory_placement.memory_property_flags,
        allocator->properties2.properties.limits.nonCoherentAtomSize,
        device_memory, handle, device_address,
        iree_hal_buffer_release_callback_null(), allocator->host_allocator,
        &buffer);
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_VULKAN_ALLOCATOR_ID, (void*)handle,
                           (iree_host_size_t)allocation_size);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
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
      iree_hal_buffer_allocation_size(base_buffer)));

  iree_hal_buffer_destroy(base_buffer);
  IREE_TRACE_ZONE_END(z0);
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
