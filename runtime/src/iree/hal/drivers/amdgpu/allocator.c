// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/allocator.h"

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_allocator_t
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_AMDGPU_ALLOCATOR_ID = "AMDGPU unpooled";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_amdgpu_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Unowned logical device. Must outlive the allocator.
  iree_hal_amdgpu_logical_device_t* logical_device;

  // Unowned libhsa handle. Must be retained by the owner.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Fine-grained memory pool on the first GPU agent.
  // Host+device visible, coherent. Used for all allocations for now.
  // A production allocator would use coarse-grained pools for device-local
  // buffers and fine-grained for host-visible buffers.
  hsa_amd_memory_pool_t device_fine_pool;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_amdgpu_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_amdgpu_allocator_vtable;

static iree_hal_amdgpu_allocator_t* iree_hal_amdgpu_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_allocator_vtable);
  return (iree_hal_amdgpu_allocator_t*)base_value;
}

iree_status_t iree_hal_amdgpu_allocator_create(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_allocator = NULL;

  iree_hal_amdgpu_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_amdgpu_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->logical_device = logical_device;
  allocator->libhsa = libhsa;

  // Find the fine-grained memory pool on the first GPU agent.
  // Fine-grained memory is host+device visible and coherent — suitable for all
  // buffer types until we add proper coarse-grained staging.
  iree_status_t status = iree_hal_amdgpu_find_fine_global_memory_pool(
      libhsa, topology->gpu_agents[0], &allocator->device_fine_pool);

  if (iree_status_is_ok(status)) {
    *out_allocator = (iree_hal_allocator_t*)allocator;
  } else {
    iree_hal_allocator_release((iree_hal_allocator_t*)allocator);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_amdgpu_allocator_t* allocator =
      iree_hal_amdgpu_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_amdgpu_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_amdgpu_allocator_t* allocator =
      (iree_hal_amdgpu_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_amdgpu_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_amdgpu_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_amdgpu_allocator_t* allocator =
        iree_hal_amdgpu_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_status_t iree_hal_amdgpu_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  // Report a single heap: fine-grained device memory (host+device visible).
  // A production implementation would report separate heaps for coarse-grained
  // device-local, fine-grained device, host-local, etc.
  const iree_host_size_t heap_count = 1;
  *out_count = heap_count;
  if (capacity < heap_count) return iree_ok_status();

  memset(heaps, 0, heap_count * sizeof(*heaps));

  // Heap 0: Fine-grained device memory (host+device visible, coherent).
  heaps[0].type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                  IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  heaps[0].allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                           IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                           IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                           IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  heaps[0].max_allocation_size = ~(iree_device_size_t)0;
  heaps[0].min_alignment = 256;

  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_amdgpu_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  // All allocations are from fine-grained device memory: device-local,
  // host-visible, and coherent.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER |
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;

  // Resolve OPTIMAL to our actual memory type.
  if (iree_any_bit_set(params->type, IREE_HAL_MEMORY_TYPE_OPTIMAL)) {
    params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
    params->type |= IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  }

  // Fine-grained memory is always host-visible. Mark as such even if not
  // requested so callers know they can map it.
  params->type |=
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_COHERENT;

  // Guard against 0-byte allocations.
  if (*allocation_size == 0) *allocation_size = 4;

  // Align to 256 bytes (HSA memory pool minimum alignment).
  *allocation_size = iree_device_size_ceil_div(*allocation_size, 256) * 256;

  return compatibility;
}

static iree_status_t iree_hal_amdgpu_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_amdgpu_allocator_t* allocator =
      iree_hal_amdgpu_allocator_cast(base_allocator);
  const iree_device_size_t byte_length = allocation_size;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)byte_length);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_amdgpu_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1, temp2;
    iree_string_view_t memory_type_str =
        iree_hal_memory_type_format(params->type, &temp0);
    iree_string_view_t usage_str =
        iree_hal_buffer_usage_format(params->usage, &temp1);
    iree_string_view_t compatibility_str =
        iree_hal_buffer_compatibility_format(compatibility, &temp2);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters; "
        "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
        (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
        usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
#endif  // IREE_STATUS_MODE
  }

  // Allocate from the fine-grained device memory pool.
  void* host_ptr = NULL;
  iree_status_t status = iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(allocator->libhsa), allocator->device_fine_pool,
      (size_t)allocation_size, HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &host_ptr);

  // Grant all agents access to the allocation. For fine-grained memory this is
  // often a no-op (ALLOWED_BY_DEFAULT), but some configurations require
  // explicit grants. The call is cheap when access is already permitted and
  // required for correctness when it isn't.
  if (iree_status_is_ok(status)) {
    const iree_hal_amdgpu_topology_t* topology =
        &allocator->logical_device->system->topology;
    status = iree_hsa_amd_agents_allow_access(
        IREE_LIBHSA(allocator->libhsa), (uint32_t)topology->all_agent_count,
        topology->all_agents,
        /*flags=*/NULL, host_ptr);
  }

  // Wrap in a HAL buffer.
  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t placement = {
        .device = (iree_hal_device_t*)allocator->logical_device,
        .queue_affinity = compat_params.queue_affinity
                              ? compat_params.queue_affinity
                              : IREE_HAL_QUEUE_AFFINITY_ANY,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_amdgpu_buffer_create(
        allocator->libhsa, placement, compat_params.type,
        IREE_HAL_MEMORY_ACCESS_ALL, compat_params.usage, allocation_size,
        byte_length, host_ptr, iree_hal_buffer_release_callback_null(),
        allocator->host_allocator, &buffer);
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_AMDGPU_ALLOCATOR_ID, host_ptr,
                           (iree_host_size_t)allocation_size);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    *out_buffer = buffer;
  } else {
    if (host_ptr) {
      IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(
          IREE_LIBHSA(allocator->libhsa), host_ptr));
    }
    iree_hal_buffer_release(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_amdgpu_allocator_t* allocator =
      iree_hal_amdgpu_allocator_cast(base_allocator);

  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
      iree_hal_buffer_allocation_size(base_buffer)));

  // The buffer's destroy method handles freeing the HSA allocation.
  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_amdgpu_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU buffer import not yet implemented");
}

static iree_status_t iree_hal_amdgpu_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "AMDGPU buffer export not yet implemented");
}

static const iree_hal_allocator_vtable_t iree_hal_amdgpu_allocator_vtable = {
    .destroy = iree_hal_amdgpu_allocator_destroy,
    .host_allocator = iree_hal_amdgpu_allocator_host_allocator,
    .trim = iree_hal_amdgpu_allocator_trim,
    .query_statistics = iree_hal_amdgpu_allocator_query_statistics,
    .query_memory_heaps = iree_hal_amdgpu_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_amdgpu_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_amdgpu_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_amdgpu_allocator_deallocate_buffer,
    .import_buffer = iree_hal_amdgpu_allocator_import_buffer,
    .export_buffer = iree_hal_amdgpu_allocator_export_buffer,
};
