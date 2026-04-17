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
static const char* IREE_HAL_AMDGPU_ALLOCATOR_ID = "iree-hal-amdgpu-unpooled";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

// HSA memory-pool allocations are reported and rounded to this alignment.
#define IREE_HAL_AMDGPU_ALLOCATOR_MIN_ALIGNMENT 256

typedef struct iree_hal_amdgpu_allocator_t {
  // HAL resource header for allocator lifetime management.
  iree_hal_resource_t resource;

  // Host allocator used for allocator-owned bookkeeping.
  iree_allocator_t host_allocator;

  // Unowned logical device. Must outlive the allocator.
  iree_hal_amdgpu_logical_device_t* logical_device;

  // Unowned libhsa handle. Must be retained by the owner.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Unowned topology used to resolve queue affinity to a physical device.
  const iree_hal_amdgpu_topology_t* topology;

  // Coarse-grained GPU-local pools used for default device-local allocations.
  hsa_amd_memory_pool_t device_coarse_pools[IREE_HAL_AMDGPU_MAX_GPU_AGENT];

  // Fine-grained GPU-local pools used only for explicit host-visible requests.
  hsa_amd_memory_pool_t device_fine_pools[IREE_HAL_AMDGPU_MAX_GPU_AGENT];

  // Fine-grained host-local pools nearest to each GPU.
  hsa_amd_memory_pool_t host_fine_pools[IREE_HAL_AMDGPU_MAX_GPU_AGENT];

  IREE_STATISTICS(
      // Aggregate allocation statistics reported through the HAL allocator API.
      iree_hal_allocator_statistics_t statistics;)
} iree_hal_amdgpu_allocator_t;

typedef struct iree_hal_amdgpu_allocator_placement_t {
  // HSA memory pool selected for the allocation.
  hsa_amd_memory_pool_t memory_pool;

  // Physical device ordinal owning |memory_pool|.
  uint32_t physical_device_ordinal;

  // Resolved HAL memory type exposed by the created buffer.
  iree_hal_memory_type_t memory_type;
} iree_hal_amdgpu_allocator_placement_t;

typedef struct iree_hal_amdgpu_imported_host_release_data_t {
  // Unowned libhsa handle used to unlock the imported host allocation.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Original host allocation pointer passed to hsa_amd_memory_lock.
  void* host_ptr;

  // Host allocator used to release this thunk after buffer destruction.
  iree_allocator_t host_allocator;

  // Optional caller callback invoked after HSA has unlocked the host memory.
  iree_hal_buffer_release_callback_t caller_release_callback;
} iree_hal_amdgpu_imported_host_release_data_t;

static const iree_hal_allocator_vtable_t iree_hal_amdgpu_allocator_vtable;

static iree_hal_amdgpu_allocator_t* iree_hal_amdgpu_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_allocator_vtable);
  return (iree_hal_amdgpu_allocator_t*)base_value;
}

static bool iree_hal_amdgpu_allocator_select_device_ordinal(
    const iree_hal_amdgpu_allocator_t* allocator,
    iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t* out_device_ordinal) {
  iree_hal_queue_affinity_t effective_affinity = queue_affinity;
  if (iree_hal_queue_affinity_is_any(effective_affinity)) {
    effective_affinity = allocator->logical_device->queue_affinity_mask;
  } else {
    iree_hal_queue_affinity_and_into(
        effective_affinity, allocator->logical_device->queue_affinity_mask);
  }
  if (iree_hal_queue_affinity_is_empty(effective_affinity)) return false;

  const int queue_ordinal =
      iree_hal_queue_affinity_find_first_set(effective_affinity);
  const iree_host_size_t device_ordinal =
      (iree_host_size_t)queue_ordinal /
      allocator->topology->gpu_agent_queue_count;
  if (device_ordinal >= allocator->topology->gpu_agent_count) return false;

  *out_device_ordinal = device_ordinal;
  return true;
}

static bool iree_hal_amdgpu_allocator_resolve_placement(
    iree_hal_amdgpu_allocator_t* allocator, iree_hal_buffer_params_t* params,
    iree_hal_amdgpu_allocator_placement_t* out_placement) {
  memset(out_placement, 0, sizeof(*out_placement));

  iree_host_size_t device_ordinal = 0;
  if (!iree_hal_amdgpu_allocator_select_device_ordinal(
          allocator, params->queue_affinity, &device_ordinal)) {
    return false;
  }

  const iree_hal_memory_type_t requested_type = params->type;
  const iree_hal_memory_type_t required_type =
      requested_type & ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  const bool requires_host_local =
      iree_all_bits_set(required_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL);
  const bool requires_host_visible =
      iree_all_bits_set(required_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE);
  const bool requires_device_local =
      iree_all_bits_set(required_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);

  hsa_amd_memory_pool_t memory_pool = {0};
  iree_hal_memory_type_t memory_type = 0;
  // Sharing hints do not affect HSA pool selection. Export is omitted because
  // it requires dedicated platform export support.
  const iree_hal_buffer_usage_t sharing_usage =
      IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE |
      IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT |
      IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE;
  iree_hal_buffer_usage_t supported_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                                            IREE_HAL_BUFFER_USAGE_DISPATCH |
                                            sharing_usage;
  if (requires_host_local) {
    if (requires_device_local) return false;
    memory_pool = allocator->host_fine_pools[device_ordinal];
    memory_type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  } else if (requires_host_visible) {
    memory_pool = allocator->device_fine_pools[device_ordinal];
    memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                  IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  } else {
    memory_pool = allocator->device_coarse_pools[device_ordinal];
    memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  }
  if (!memory_pool.handle) return false;
  if (!iree_all_bits_set(memory_type, required_type)) return false;

  if (iree_any_bit_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    supported_usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                       IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
                       IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
                       IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
                       IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;
  } else {
    const iree_hal_buffer_usage_t mapping_usage =
        IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
        IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
        IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
        IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;
    if (iree_any_bit_set(params->usage, mapping_usage)) {
      if (!iree_all_bits_set(params->usage,
                             IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL)) {
        return false;
      }
      params->usage &=
          ~(mapping_usage | IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL);
    }
  }
  if (!iree_all_bits_set(supported_usage, params->usage)) return false;

  params->type = memory_type;
  params->usage &= supported_usage;
  out_placement->memory_pool = memory_pool;
  out_placement->physical_device_ordinal = (uint32_t)device_ordinal;
  out_placement->memory_type = memory_type;
  return true;
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
  memset(allocator, 0, sizeof(*allocator));
  iree_hal_resource_initialize(&iree_hal_amdgpu_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->logical_device = logical_device;
  allocator->libhsa = libhsa;
  allocator->topology = topology;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < topology->gpu_agent_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amdgpu_find_coarse_global_memory_pool(
        libhsa, topology->gpu_agents[i], &allocator->device_coarse_pools[i]);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_find_fine_global_memory_pool(
          libhsa, topology->gpu_agents[i], &allocator->device_fine_pools[i]);
    }
    if (iree_status_is_ok(status)) {
      const iree_host_size_t host_ordinal = topology->gpu_cpu_map[i];
      allocator->host_fine_pools[i] =
          logical_device->system->host_memory_pools[host_ordinal].fine_pool;
    }
  }

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
  const iree_host_size_t heap_count = 3;
  *out_count = heap_count;
  if (capacity < heap_count) return iree_ok_status();

  memset(heaps, 0, heap_count * sizeof(*heaps));

  // Sharing hints do not affect HSA pool selection. Export is omitted because
  // it requires dedicated platform export support.
  const iree_hal_buffer_usage_t sharing_usage =
      IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE |
      IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT |
      IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE;
  const iree_hal_buffer_usage_t mappable_usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH |
      sharing_usage | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
      IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
      IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
      IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
      IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;

  // Heap 0: coarse-grained device-local memory.
  heaps[0].type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  heaps[0].allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                           IREE_HAL_BUFFER_USAGE_DISPATCH | sharing_usage;
  heaps[0].max_allocation_size = ~(iree_device_size_t)0;
  heaps[0].min_alignment = IREE_HAL_AMDGPU_ALLOCATOR_MIN_ALIGNMENT;

  // Heap 1: fine-grained device-local memory for explicit host visibility.
  heaps[1].type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                  IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  heaps[1].allowed_usage = mappable_usage;
  heaps[1].max_allocation_size = ~(iree_device_size_t)0;
  heaps[1].min_alignment = IREE_HAL_AMDGPU_ALLOCATOR_MIN_ALIGNMENT;

  // Heap 2: fine-grained host-local memory visible to the device.
  heaps[2].type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  heaps[2].allowed_usage = mappable_usage;
  heaps[2].max_allocation_size = ~(iree_device_size_t)0;
  heaps[2].min_alignment = IREE_HAL_AMDGPU_ALLOCATOR_MIN_ALIGNMENT;

  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_amdgpu_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_amdgpu_allocator_t* allocator =
      iree_hal_amdgpu_allocator_cast(base_allocator);

  iree_hal_amdgpu_allocator_placement_t placement;
  if (!iree_hal_amdgpu_allocator_resolve_placement(allocator, params,
                                                   &placement)) {
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

  if (iree_all_bits_set(params->type,
                        IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE) &&
      !iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE;
  }

  // Guard against 0-byte allocations.
  if (*allocation_size == 0) *allocation_size = 4;

  // Align to the HSA memory pool minimum.
  *allocation_size =
      iree_device_size_ceil_div(*allocation_size,
                                IREE_HAL_AMDGPU_ALLOCATOR_MIN_ALIGNMENT) *
      IREE_HAL_AMDGPU_ALLOCATOR_MIN_ALIGNMENT;

  return compatibility;
}

static void iree_hal_amdgpu_allocator_record_buffer_allocate(
    iree_hal_amdgpu_allocator_t* allocator,
    const iree_hal_amdgpu_allocator_placement_t* memory_placement,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    void* host_ptr, iree_hal_buffer_t* buffer) {
  uint64_t session_id = 0;
  const uint64_t allocation_id =
      iree_hal_amdgpu_logical_device_allocate_profile_memory_allocation_id(
          (iree_hal_device_t*)allocator->logical_device, &session_id);
  if (allocation_id == 0) {
    return;
  }

  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  event.type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE;
  event.allocation_id = allocation_id;
  event.pool_id = memory_placement->memory_pool.handle;
  event.backing_id = (uint64_t)(uintptr_t)host_ptr;
  event.physical_device_ordinal = memory_placement->physical_device_ordinal;
  event.memory_type = memory_placement->memory_type;
  event.buffer_usage = params.usage;
  event.length = allocation_size;
  event.alignment = IREE_HAL_AMDGPU_ALLOCATOR_MIN_ALIGNMENT;
  if (iree_hal_amdgpu_logical_device_record_profile_memory_event(
          (iree_hal_device_t*)allocator->logical_device, &event)) {
    iree_hal_amdgpu_buffer_set_profile_allocation(
        buffer, session_id, allocation_id, event.pool_id,
        event.physical_device_ordinal, event.alignment);
  }
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
  iree_hal_amdgpu_allocator_placement_t memory_placement;
  if (!iree_hal_amdgpu_allocator_resolve_placement(allocator, &compat_params,
                                                   &memory_placement)) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t memory_type_str =
        iree_hal_memory_type_format(params->type, &temp0);
    iree_string_view_t usage_str =
        iree_hal_buffer_usage_format(params->usage, &temp1);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters; "
        "memory_type=%.*s, usage=%.*s",
        (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
        usage_str.data);
#else
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
#endif  // IREE_STATUS_MODE
  }

  // Guard against 0-byte allocations and align to the HSA memory pool minimum.
  if (allocation_size == 0) allocation_size = 4;
  if (!iree_device_size_checked_align(allocation_size,
                                      IREE_HAL_AMDGPU_ALLOCATOR_MIN_ALIGNMENT,
                                      &allocation_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation size %" PRIdsz
                            " overflows HSA memory pool alignment",
                            allocation_size);
  }

  // Allocate from the resolved HSA memory pool.
  void* host_ptr = NULL;
  iree_status_t status = iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(allocator->libhsa), memory_placement.memory_pool,
      (size_t)allocation_size, HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &host_ptr);

  // Grant all agents access to the allocation. The call is cheap when access
  // is already permitted and required for correctness when it isn't.
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
    const iree_hal_buffer_placement_t buffer_placement = {
        .device = (iree_hal_device_t*)allocator->logical_device,
        .queue_affinity = compat_params.queue_affinity
                              ? compat_params.queue_affinity
                              : IREE_HAL_QUEUE_AFFINITY_ANY,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_amdgpu_buffer_create(
        allocator->libhsa, buffer_placement, memory_placement.memory_type,
        IREE_HAL_MEMORY_ACCESS_ALL, compat_params.usage, allocation_size,
        byte_length, host_ptr, iree_hal_buffer_release_callback_null(),
        allocator->host_allocator, &buffer);
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_AMDGPU_ALLOCATOR_ID, host_ptr,
                           (iree_host_size_t)allocation_size);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    iree_hal_amdgpu_allocator_record_buffer_allocate(
        allocator, &memory_placement, compat_params, allocation_size, host_ptr,
        buffer);
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
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, iree_hal_buffer_allocation_size(base_buffer));

  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
      iree_hal_buffer_allocation_size(base_buffer)));

  // The buffer's destroy method handles freeing the HSA allocation.
  iree_hal_buffer_destroy(base_buffer);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_allocator_release_imported_host(
    void* user_data, iree_hal_buffer_t* buffer) {
  iree_hal_amdgpu_imported_host_release_data_t* data =
      (iree_hal_amdgpu_imported_host_release_data_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_IGNORE_ERROR(
      iree_hsa_amd_memory_unlock(IREE_LIBHSA(data->libhsa), data->host_ptr));
  if (data->caller_release_callback.fn) {
    data->caller_release_callback.fn(data->caller_release_callback.user_data,
                                     buffer);
  }
  iree_allocator_free(data->host_allocator, data);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  iree_hal_amdgpu_allocator_t* allocator =
      iree_hal_amdgpu_allocator_cast(base_allocator);

  if (IREE_UNLIKELY(external_buffer->flags !=
                    IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU external buffer flags: 0x%x",
                            external_buffer->flags);
  }

  switch (external_buffer->type) {
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION:
      break;
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION:
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD:
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_WIN32:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "AMDGPU external buffer type not supported");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid AMDGPU external buffer type");
  }

  if (IREE_UNLIKELY(!external_buffer->handle.host_allocation.ptr)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host allocation import requires a non-null ptr");
  }
  if (IREE_UNLIKELY(external_buffer->size == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host allocation import requires a non-zero size");
  }
  if (IREE_UNLIKELY(
          iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unable to import host allocations as device-local memory");
  }

  iree_hal_buffer_params_t compat_params = *params;
  if (iree_any_bit_set(compat_params.type, IREE_HAL_MEMORY_TYPE_OPTIMAL)) {
    compat_params.type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
    compat_params.type |=
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  }
  if (!iree_all_bits_set(compat_params.type,
                         IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host allocation import requires device-visible memory");
  }

  iree_device_size_t allocation_size = external_buffer->size;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_amdgpu_allocator_query_buffer_compatibility(
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
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot import a buffer with the given parameters; "
        "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
        (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
        usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot import a buffer with the given parameters");
#endif  // IREE_STATUS_MODE
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, external_buffer->size);
  void* host_ptr = external_buffer->handle.host_allocation.ptr;
  void* agent_ptr = NULL;
  iree_status_t status = iree_hsa_amd_memory_lock(
      IREE_LIBHSA(allocator->libhsa), host_ptr, (size_t)external_buffer->size,
      /*agents=*/NULL, /*num_agent=*/0, &agent_ptr);

  iree_hal_amdgpu_imported_host_release_data_t* release_data = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(allocator->host_allocator, sizeof(*release_data),
                              (void**)&release_data);
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    release_data->libhsa = allocator->libhsa;
    release_data->host_ptr = host_ptr;
    release_data->host_allocator = allocator->host_allocator;
    release_data->caller_release_callback = release_callback;
    iree_hal_buffer_release_callback_t imported_release_callback = {
        .fn = iree_hal_amdgpu_allocator_release_imported_host,
        .user_data = release_data,
    };
    const iree_hal_buffer_placement_t placement = {
        .device = (iree_hal_device_t*)allocator->logical_device,
        .queue_affinity = compat_params.queue_affinity
                              ? compat_params.queue_affinity
                              : IREE_HAL_QUEUE_AFFINITY_ANY,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_amdgpu_buffer_create(
        allocator->libhsa, placement, compat_params.type, compat_params.access,
        compat_params.usage, external_buffer->size, external_buffer->size,
        agent_ptr, imported_release_callback, allocator->host_allocator,
        &buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    if (release_data) {
      iree_allocator_free(allocator->host_allocator, release_data);
    }
    if (agent_ptr) {
      IREE_IGNORE_ERROR(
          iree_hsa_amd_memory_unlock(IREE_LIBHSA(allocator->libhsa), host_ptr));
    }
    iree_hal_buffer_release(buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
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
