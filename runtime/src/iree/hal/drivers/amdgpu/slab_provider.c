// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/slab_provider.h"

#include <stddef.h>

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

typedef struct iree_hal_amdgpu_slab_provider_t {
  iree_hal_slab_provider_t base;
  iree_allocator_t host_allocator;

  // Borrowed from the owning logical/system device.
  iree_hal_device_t* device;
  const iree_hal_amdgpu_libhsa_t* libhsa;
  const iree_hal_amdgpu_topology_t* topology;

  // HSA pool this provider acquires slabs from.
  hsa_amd_memory_pool_t memory_pool;

  // Minimum runtime allocation granule reported by the HSA pool.
  iree_device_size_t allocation_granule;

  // Cached memory capabilities derived from the HSA pool flags.
  iree_hal_memory_type_t memory_type;
  iree_hal_buffer_usage_t supported_usage;

  // Cumulative slab acquire/release counters for query_stats().
  iree_atomic_int64_t total_acquired;
  iree_atomic_int64_t total_released;
} iree_hal_amdgpu_slab_provider_t;

static const iree_hal_slab_provider_vtable_t
    iree_hal_amdgpu_slab_provider_vtable;

static iree_hal_amdgpu_slab_provider_t* iree_hal_amdgpu_slab_provider_cast(
    iree_hal_slab_provider_t* base_provider) {
  return (iree_hal_amdgpu_slab_provider_t*)base_provider;
}

static const iree_hal_amdgpu_slab_provider_t*
iree_hal_amdgpu_slab_provider_const_cast(
    const iree_hal_slab_provider_t* base_provider) {
  return (const iree_hal_amdgpu_slab_provider_t*)base_provider;
}

static iree_status_t iree_hal_amdgpu_slab_provider_query_pool_properties(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_amd_memory_pool_t memory_pool,
    iree_device_size_t* out_allocation_granule,
    iree_hal_memory_type_t* out_memory_type,
    iree_hal_buffer_usage_t* out_supported_usage) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_allocation_granule);
  IREE_ASSERT_ARGUMENT(out_memory_type);
  IREE_ASSERT_ARGUMENT(out_supported_usage);

  size_t allocation_granule = 0;
  IREE_RETURN_IF_ERROR(
      iree_hsa_amd_memory_pool_get_info(
          IREE_LIBHSA(libhsa), memory_pool,
          HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &allocation_granule),
      "querying HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE for an AMDGPU "
      "slab provider");
  if (allocation_granule == 0 ||
      !iree_device_size_is_power_of_two(allocation_granule)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "invalid HSA runtime allocation granule for an AMDGPU memory pool: "
        "%" PRIhsz,
        (iree_host_size_t)allocation_granule);
  }

  hsa_region_segment_t segment = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
      &segment));
  if (segment != HSA_REGION_SEGMENT_GLOBAL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU slab providers require a GLOBAL HSA pool");
  }

  bool alloc_allowed = false;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed));
  if (!alloc_allowed) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU slab provider memory pool does not support runtime "
        "allocations");
  }

  hsa_region_global_flag_t global_flags = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
      &global_flags));

  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  iree_hal_buffer_usage_t supported_usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  if (iree_any_bit_set(
          global_flags,
          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
              HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED)) {
    memory_type |=
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
    supported_usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                       IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
                       IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
                       IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
                       IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;
  }

  *out_allocation_granule = (iree_device_size_t)allocation_granule;
  *out_memory_type = memory_type;
  *out_supported_usage = supported_usage;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_slab_provider_create(
    iree_hal_device_t* device, const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    hsa_amd_memory_pool_t memory_pool, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_provider = NULL;

  iree_hal_amdgpu_slab_provider_t* provider = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*provider),
                                (void**)&provider));
  memset(provider, 0, sizeof(*provider));
  iree_hal_slab_provider_initialize(&iree_hal_amdgpu_slab_provider_vtable,
                                    &provider->base);
  provider->host_allocator = host_allocator;
  provider->device = device;
  provider->libhsa = libhsa;
  provider->topology = topology;
  provider->memory_pool = memory_pool;
  provider->total_acquired = IREE_ATOMIC_VAR_INIT(0);
  provider->total_released = IREE_ATOMIC_VAR_INIT(0);

  iree_status_t status = iree_hal_amdgpu_slab_provider_query_pool_properties(
      libhsa, memory_pool, &provider->allocation_granule,
      &provider->memory_type, &provider->supported_usage);
  if (iree_status_is_ok(status)) {
    *out_provider = &provider->base;
  } else {
    iree_hal_slab_provider_release(&provider->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base_provider);
  iree_allocator_t host_allocator = provider->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(host_allocator, provider);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_hal_slab_t* out_slab) {
  IREE_ASSERT_ARGUMENT(out_slab);
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_slab, 0, sizeof(*out_slab));

  if (IREE_UNLIKELY(min_length == 0)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "AMDGPU slab allocations must be non-empty"));
  }
  iree_device_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_align(
          min_length, provider->allocation_granule, &allocation_size))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "AMDGPU slab allocation size overflow aligning %" PRIu64
                " bytes to a %" PRIu64 "-byte HSA allocation granule",
                (uint64_t)min_length, (uint64_t)provider->allocation_granule));
  }

  void* base_ptr = NULL;
  iree_status_t status = iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(provider->libhsa), provider->memory_pool,
      (size_t)allocation_size, HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &base_ptr);
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_agents_allow_access(
        IREE_LIBHSA(provider->libhsa),
        (uint32_t)provider->topology->all_agent_count,
        provider->topology->all_agents,
        /*flags=*/NULL, base_ptr);
  }

  if (iree_status_is_ok(status)) {
    out_slab->base_ptr = (uint8_t*)base_ptr;
    // Preserve the requested logical slab length. The hidden HSA allocation may
    // be larger due to runtime granule rounding, but exposing that padding here
    // would incorrectly inflate HAL buffer byte lengths in pass-through pools.
    out_slab->length = min_length;
    out_slab->provider_handle = 0;
    iree_atomic_fetch_add(&provider->total_acquired, 1,
                          iree_memory_order_relaxed);
  } else if (base_ptr) {
    IREE_IGNORE_ERROR(
        iree_hsa_amd_memory_pool_free(IREE_LIBHSA(provider->libhsa), base_ptr));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  IREE_ASSERT_ARGUMENT(slab);
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (slab->base_ptr) {
    IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(
        IREE_LIBHSA(provider->libhsa), slab->base_ptr));
    iree_atomic_fetch_add(&provider->total_released, 1,
                          iree_memory_order_relaxed);
  }
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base_provider);

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
        "AMDGPU slab provider memory type %.*s does not satisfy requested "
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
        "AMDGPU slab provider usage %.*s does not satisfy requested buffer "
        "usage %.*s",
        (int)actual_string.size, actual_string.data, (int)requested_string.size,
        requested_string.data);
#else
    return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
#endif  // IREE_STATUS_MODE
  }

  const iree_hal_buffer_placement_t placement = {
      .device = provider->device,
      .queue_affinity = params.queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  return iree_hal_amdgpu_buffer_create(
      provider->libhsa, placement, resolved_type, params.access, params.usage,
      allocation_size, allocation_size, slab->base_ptr + slab_offset,
      release_callback, provider->host_allocator, out_buffer);
}

static void iree_hal_amdgpu_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, iree_hal_slab_t* slab) {
  (void)base_provider;
  (void)slab;
}

static void iree_hal_amdgpu_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_trim_flags_t flags) {
  (void)base_provider;
  (void)flags;
}

static void iree_hal_amdgpu_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  if (iree_hal_slab_provider_visited(visited, base_provider)) {
    return;
  }
  const iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_const_cast(base_provider);
  out_stats->total_acquired += (uint64_t)iree_atomic_load(
      &provider->total_acquired, iree_memory_order_relaxed);
  out_stats->total_released += (uint64_t)iree_atomic_load(
      &provider->total_released, iree_memory_order_relaxed);
}

static void iree_hal_amdgpu_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_memory_type_t* out_memory_type,
    iree_hal_buffer_usage_t* out_supported_usage) {
  const iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_const_cast(base_provider);
  *out_memory_type = provider->memory_type;
  *out_supported_usage = provider->supported_usage;
}

static const iree_hal_slab_provider_vtable_t
    iree_hal_amdgpu_slab_provider_vtable = {
        .destroy = iree_hal_amdgpu_slab_provider_destroy,
        .acquire_slab = iree_hal_amdgpu_slab_provider_acquire_slab,
        .release_slab = iree_hal_amdgpu_slab_provider_release_slab,
        .wrap_buffer = iree_hal_amdgpu_slab_provider_wrap_buffer,
        .prefault = iree_hal_amdgpu_slab_provider_prefault,
        .trim = iree_hal_amdgpu_slab_provider_trim,
        .query_stats = iree_hal_amdgpu_slab_provider_query_stats,
        .query_properties = iree_hal_amdgpu_slab_provider_query_properties,
};
