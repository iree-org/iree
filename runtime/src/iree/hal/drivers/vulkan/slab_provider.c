// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/slab_provider.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/hal/memory/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_slab_provider_t
//===----------------------------------------------------------------------===//

static const char* IREE_HAL_VULKAN_SLAB_PROVIDER_TRACE_ID =
    "iree-hal-vulkan-slab-provider";

typedef struct iree_hal_vulkan_slab_provider_t {
  // Base slab provider for vtable dispatch and reference counting.
  iree_hal_slab_provider_t base;

  // Host allocator used for provider-owned host allocations.
  iree_allocator_t host_allocator;

  // Parent Vulkan allocator used to materialize whole slabs.
  iree_hal_vulkan_allocator_t* allocator;

  // Parent HAL device used for buffer placement metadata.
  iree_hal_device_t* parent_device;

  // Device-level Vulkan dispatch table copied from the logical device.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device that owns slabs from this provider.
  VkDevice logical_device;

  // Vulkan memory type index used for whole-slab allocations.
  uint32_t memory_type_index;

  // Vulkan memory property flags for |memory_type_index|.
  VkMemoryPropertyFlags memory_property_flags;

  // HAL memory type exposed by slabs from this provider.
  iree_hal_memory_type_t memory_type;

  // HAL buffer usage bits supported by slabs from this provider.
  iree_hal_buffer_usage_t supported_usage;

  // Queue affinity mask valid for buffers materialized from this provider.
  iree_hal_queue_affinity_t queue_affinity_mask;

  // Minimum alignment used by whole-slab allocation requests.
  iree_device_size_t min_alignment;

  // Physical-device nonCoherentAtomSize used for mapped-memory ranges.
  VkDeviceSize non_coherent_atom_size;

  // Named-memory trace stream for whole-slab backing allocations.
  iree_hal_memory_trace_t trace;

  // Cumulative slabs acquired from Vulkan.
  iree_atomic_int64_t total_acquired;

  // Cumulative slabs released back to Vulkan.
  iree_atomic_int64_t total_released;
} iree_hal_vulkan_slab_provider_t;

static const iree_hal_slab_provider_vtable_t
    iree_hal_vulkan_slab_provider_vtable;

static iree_hal_vulkan_slab_provider_t* iree_hal_vulkan_slab_provider_cast(
    iree_hal_slab_provider_t* base_provider) {
  return (iree_hal_vulkan_slab_provider_t*)base_provider;
}

static const iree_hal_vulkan_slab_provider_t*
iree_hal_vulkan_slab_provider_const_cast(
    const iree_hal_slab_provider_t* base_provider) {
  return (const iree_hal_vulkan_slab_provider_t*)base_provider;
}

iree_status_t iree_hal_vulkan_slab_provider_create(
    iree_hal_vulkan_allocator_t* allocator,
    iree_hal_vulkan_slab_provider_options_t options,
    iree_string_view_t trace_name, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(options.parent_device);
  IREE_ASSERT_ARGUMENT(options.syms);
  IREE_ASSERT_ARGUMENT(options.logical_device);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (options.memory_type == IREE_HAL_MEMORY_TYPE_NONE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan slab provider requires a non-empty HAL memory type");
  }
  if (options.supported_usage == IREE_HAL_BUFFER_USAGE_NONE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan slab provider requires non-empty HAL buffer usage bits");
  }
  if (!iree_device_size_is_valid_alignment(options.min_alignment)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan slab provider alignment %" PRIu64
                            " is not a power-of-two",
                            (uint64_t)options.min_alignment);
  }

  iree_hal_vulkan_slab_provider_t* provider = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*provider),
                                (void**)&provider));
  memset(provider, 0, sizeof(*provider));
  iree_hal_slab_provider_initialize(&iree_hal_vulkan_slab_provider_vtable,
                                    &provider->base);
  provider->host_allocator = host_allocator;
  provider->allocator = allocator;
  provider->parent_device = options.parent_device;
  provider->syms = *options.syms;
  provider->logical_device = options.logical_device;
  provider->memory_type_index = options.memory_type_index;
  provider->memory_property_flags = options.memory_property_flags;
  provider->memory_type = options.memory_type;
  provider->supported_usage = options.supported_usage;
  provider->queue_affinity_mask = options.queue_affinity_mask;
  provider->min_alignment = options.min_alignment;
  provider->non_coherent_atom_size =
      options.non_coherent_atom_size ? options.non_coherent_atom_size : 1;
  provider->total_acquired = IREE_ATOMIC_VAR_INIT(0);
  provider->total_released = IREE_ATOMIC_VAR_INIT(0);

  iree_status_t status = iree_hal_memory_trace_initialize(
      trace_name, IREE_HAL_VULKAN_SLAB_PROVIDER_TRACE_ID, host_allocator,
      &provider->trace);
  if (iree_status_is_ok(status)) {
    *out_provider = &provider->base;
  } else {
    iree_hal_slab_provider_release(&provider->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_cast(base_provider);
  iree_allocator_t host_allocator = provider->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_memory_trace_deinitialize(&provider->trace);
  iree_allocator_free(host_allocator, provider);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_slab_provider_query_device_address(
    iree_hal_buffer_t* buffer, VkDeviceAddress* out_device_address) {
  if (iree_hal_vulkan_buffer_isa(buffer)) {
    return iree_hal_vulkan_buffer_device_address(buffer, out_device_address);
  }
  if (iree_hal_vulkan_sparse_buffer_isa(buffer)) {
    return iree_hal_vulkan_sparse_buffer_device_address(buffer,
                                                        out_device_address);
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "Vulkan slab provider acquired a non-Vulkan buffer");
}

static iree_status_t iree_hal_vulkan_slab_provider_query_handles(
    iree_hal_buffer_t* buffer, VkDeviceMemory* out_memory,
    VkBuffer* out_handle) {
  if (iree_hal_vulkan_buffer_isa(buffer)) {
    return iree_hal_vulkan_buffer_handle(buffer, out_memory, out_handle);
  }
  if (iree_hal_vulkan_sparse_buffer_isa(buffer)) {
    return iree_hal_vulkan_sparse_buffer_handle(buffer, out_memory, out_handle);
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "Vulkan slab provider acquired a non-Vulkan buffer");
}

static iree_status_t iree_hal_vulkan_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_hal_slab_t* out_slab) {
  IREE_ASSERT_ARGUMENT(out_slab);
  iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_slab, 0, sizeof(*out_slab));

  if (IREE_UNLIKELY(min_length == 0)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "Vulkan slab allocations must be non-empty"));
  }

  iree_device_size_t allocation_size = min_length;
  const iree_device_size_t min_alignment =
      provider->min_alignment ? provider->min_alignment : 1;
  if (IREE_UNLIKELY(!iree_device_size_checked_align(
          allocation_size, min_alignment, &allocation_size))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "Vulkan slab allocation size overflow aligning %" PRIu64
                " bytes to a %" PRIu64 "-byte allocation boundary",
                (uint64_t)min_length, (uint64_t)min_alignment));
  }

  const iree_hal_buffer_params_t params = {
      .type = provider->memory_type,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = provider->supported_usage,
      .queue_affinity = provider->queue_affinity_mask,
      .min_alignment = min_alignment,
  };
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_hal_vulkan_allocator_allocate_direct_buffer_from_type(
          provider->allocator, provider->memory_type_index, &params,
          allocation_size, &buffer);

  VkDeviceAddress device_address = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_slab_provider_query_device_address(
        buffer, &device_address);
  }

  if (iree_status_is_ok(status)) {
    out_slab->base_ptr = (uint8_t*)(uintptr_t)device_address;
    out_slab->length = iree_hal_buffer_allocation_size(buffer);
    out_slab->provider_handle = (uint64_t)(uintptr_t)buffer;
    iree_hal_memory_trace_alloc(&provider->trace, out_slab->base_ptr,
                                out_slab->length);
    iree_atomic_fetch_add(&provider->total_acquired, 1,
                          iree_memory_order_relaxed);
  } else {
    iree_hal_buffer_release(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  IREE_ASSERT_ARGUMENT(slab);
  iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (slab->provider_handle) {
    iree_hal_memory_trace_free(&provider->trace, slab->base_ptr);
    iree_hal_buffer_t* buffer =
        (iree_hal_buffer_t*)(uintptr_t)slab->provider_handle;
    iree_hal_buffer_release(buffer);
    iree_atomic_fetch_add(&provider->total_released, 1,
                          iree_memory_order_relaxed);
  }
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_cast(base_provider);
  iree_hal_buffer_t* slab_buffer =
      (iree_hal_buffer_t*)(uintptr_t)slab->provider_handle;

  iree_hal_memory_type_t resolved_type = params.type;
  if (iree_any_bit_set(resolved_type, IREE_HAL_MEMORY_TYPE_OPTIMAL)) {
    resolved_type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
    resolved_type |= provider->memory_type;
  }
  if (IREE_UNLIKELY(!iree_all_bits_set(provider->memory_type, resolved_type))) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t actual_temp;
    iree_bitfield_string_temp_t requested_temp;
    iree_string_view_t actual_string =
        iree_hal_memory_type_format(provider->memory_type, &actual_temp);
    iree_string_view_t requested_string =
        iree_hal_memory_type_format(resolved_type, &requested_temp);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan slab provider memory type %.*s does not satisfy requested "
        "buffer type %.*s",
        (int)actual_string.size, actual_string.data, (int)requested_string.size,
        requested_string.data);
#else
    return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
#endif  // IREE_STATUS_MODE
  }
  if (IREE_UNLIKELY(
          !iree_all_bits_set(provider->supported_usage, params.usage))) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t actual_temp;
    iree_bitfield_string_temp_t requested_temp;
    iree_string_view_t actual_string =
        iree_hal_buffer_usage_format(provider->supported_usage, &actual_temp);
    iree_string_view_t requested_string =
        iree_hal_buffer_usage_format(params.usage, &requested_temp);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan slab provider usage %.*s does not satisfy requested buffer "
        "usage %.*s",
        (int)actual_string.size, actual_string.data, (int)requested_string.size,
        requested_string.data);
#else
    return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
#endif  // IREE_STATUS_MODE
  }

  iree_hal_queue_affinity_t queue_affinity = params.queue_affinity;
  if (iree_hal_queue_affinity_is_any(queue_affinity)) {
    queue_affinity = provider->queue_affinity_mask;
  } else {
    iree_hal_queue_affinity_and_into(queue_affinity,
                                     provider->queue_affinity_mask);
    if (IREE_UNLIKELY(iree_hal_queue_affinity_is_empty(queue_affinity))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan slab provider queue affinity 0x%016" PRIx64
          " does not cover requested buffer affinity 0x%016" PRIx64,
          provider->queue_affinity_mask, params.queue_affinity);
    }
  }

  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_slab_provider_query_handles(
      slab_buffer, &device_memory, &handle));
  VkDeviceAddress device_address = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_slab_provider_query_device_address(
      slab_buffer, &device_address));

  const iree_hal_buffer_placement_t placement = {
      .device = provider->parent_device,
      .queue_affinity = queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  return iree_hal_vulkan_buffer_create_borrowed(
      &provider->syms, provider->logical_device, placement, resolved_type,
      params.access, params.usage, slab_offset + allocation_size, slab_offset,
      allocation_size, provider->memory_property_flags,
      provider->non_coherent_atom_size, device_memory, handle, device_address,
      release_callback, provider->host_allocator, out_buffer);
}

static void iree_hal_vulkan_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, iree_hal_slab_t* slab) {
  (void)base_provider;
  (void)slab;
}

static void iree_hal_vulkan_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_trim_flags_t flags) {
  (void)base_provider;
  (void)flags;
}

static void iree_hal_vulkan_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  if (iree_hal_slab_provider_visited(visited, base_provider)) {
    return;
  }
  const iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_const_cast(base_provider);
  out_stats->total_acquired += (uint64_t)iree_atomic_load(
      &provider->total_acquired, iree_memory_order_relaxed);
  out_stats->total_released += (uint64_t)iree_atomic_load(
      &provider->total_released, iree_memory_order_relaxed);
}

static void iree_hal_vulkan_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_memory_type_t* out_memory_type,
    iree_hal_buffer_usage_t* out_supported_usage) {
  const iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_const_cast(base_provider);
  *out_memory_type = provider->memory_type;
  *out_supported_usage = provider->supported_usage;
}

static const iree_hal_slab_provider_vtable_t
    iree_hal_vulkan_slab_provider_vtable = {
        .destroy = iree_hal_vulkan_slab_provider_destroy,
        .acquire_slab = iree_hal_vulkan_slab_provider_acquire_slab,
        .release_slab = iree_hal_vulkan_slab_provider_release_slab,
        .wrap_buffer = iree_hal_vulkan_slab_provider_wrap_buffer,
        .prefault = iree_hal_vulkan_slab_provider_prefault,
        .trim = iree_hal_vulkan_slab_provider_trim,
        .query_stats = iree_hal_vulkan_slab_provider_query_stats,
        .query_properties = iree_hal_vulkan_slab_provider_query_properties,
};
