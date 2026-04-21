// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/allocator.h"

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_allocator_t
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_AMDGPU_ALLOCATOR_ID = "iree-hal-amdgpu-unpooled";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_amdgpu_allocator_memory_pool_t {
  // HSA memory pool used for allocations.
  hsa_amd_memory_pool_t memory_pool;

  // Allocation sizes submitted to HSA are rounded up to this granule.
  iree_device_size_t allocation_granule;

  // Base-pointer alignment guaranteed by HSA allocations from |memory_pool|.
  iree_device_size_t allocation_alignment;

  // Maximum single HSA allocation size supported by |memory_pool|.
  iree_device_size_t max_allocation_size;
} iree_hal_amdgpu_allocator_memory_pool_t;

typedef struct iree_hal_amdgpu_allocator_memory_pools_t {
  // Coarse-grained GPU-local pools used for default device-local allocations.
  iree_hal_amdgpu_allocator_memory_pool_t
      device_coarse[IREE_HAL_AMDGPU_MAX_GPU_AGENT];

  // Fine-grained GPU-local pools used only for explicit host-visible requests.
  iree_hal_amdgpu_allocator_memory_pool_t
      device_fine[IREE_HAL_AMDGPU_MAX_GPU_AGENT];

  // Fine-grained host-local pools nearest to each GPU.
  iree_hal_amdgpu_allocator_memory_pool_t
      host_fine[IREE_HAL_AMDGPU_MAX_GPU_AGENT];
} iree_hal_amdgpu_allocator_memory_pools_t;

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

  // Cached HSA memory pool properties used for placement and heap queries.
  iree_hal_amdgpu_allocator_memory_pools_t memory_pools;

  IREE_STATISTICS(
      // Aggregate allocation statistics reported through the HAL allocator API.
      iree_hal_allocator_statistics_t statistics;)
} iree_hal_amdgpu_allocator_t;

typedef struct iree_hal_amdgpu_allocator_placement_t {
  // HSA memory pool selected for the allocation.
  const iree_hal_amdgpu_allocator_memory_pool_t* memory_pool;

  // Physical device ordinal owning |memory_pool|.
  uint32_t physical_device_ordinal;

  // Resolved HAL memory type exposed by the created buffer.
  iree_hal_memory_type_t memory_type;
} iree_hal_amdgpu_allocator_placement_t;

typedef struct iree_hal_amdgpu_imported_host_release_data_t {
  // Unowned libhsa handle used to unlock the imported host allocation.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Unowned logical device used to record the paired unimport event.
  iree_hal_device_t* profile_device;

  // Original host allocation pointer passed to hsa_amd_memory_lock.
  void* host_ptr;

  // Length of the imported host allocation in bytes.
  iree_device_size_t length;

  // HAL memory type bits used to expose the imported buffer.
  iree_hal_memory_type_t memory_type;

  // HAL buffer usage bits used to expose the imported buffer.
  iree_hal_buffer_usage_t buffer_usage;

  // Profiling session id owning |profile_allocation_id|.
  uint64_t profile_session_id;

  // Session-local allocation id for the import/unimport lifecycle.
  uint64_t profile_allocation_id;

  // Session-local physical device ordinal attributed to this import.
  uint32_t profile_physical_device_ordinal;

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

static iree_status_t iree_hal_amdgpu_allocator_query_pool_properties(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_amd_memory_pool_t memory_pool,
    iree_hal_amdgpu_allocator_memory_pool_t* out_pool) {
  memset(out_pool, 0, sizeof(*out_pool));

  bool allocation_allowed = false;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allocation_allowed));
  if (!allocation_allowed) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU allocator memory pool does not support runtime allocations");
  }

  size_t allocation_granule = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &allocation_granule));
  if (allocation_granule == 0 ||
      !iree_device_size_is_power_of_two(allocation_granule)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "invalid HSA runtime allocation granule for an AMDGPU memory pool: "
        "%" PRIhsz,
        (iree_host_size_t)allocation_granule);
  }

  size_t allocation_alignment = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT, &allocation_alignment));
  if (allocation_alignment == 0 ||
      !iree_device_size_is_power_of_two(allocation_alignment)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "invalid HSA runtime allocation alignment for an AMDGPU memory pool: "
        "%" PRIhsz,
        (iree_host_size_t)allocation_alignment);
  }

  size_t max_allocation_size = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
      &max_allocation_size));
  if (max_allocation_size == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "invalid HSA max allocation size for an AMDGPU memory pool");
  }

  out_pool->memory_pool = memory_pool;
  out_pool->allocation_granule = (iree_device_size_t)allocation_granule;
  out_pool->allocation_alignment = (iree_device_size_t)allocation_alignment;
  out_pool->max_allocation_size = (iree_device_size_t)max_allocation_size;
  return iree_ok_status();
}

static bool iree_hal_amdgpu_allocator_select_device_ordinal(
    const iree_hal_amdgpu_allocator_t* allocator,
    iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t* out_device_ordinal) {
  const iree_hal_amdgpu_queue_affinity_domain_t domain = {
      .supported_affinity = allocator->logical_device->queue_affinity_mask,
      .physical_device_count = allocator->topology->gpu_agent_count,
      .queue_count_per_physical_device =
          allocator->topology->gpu_agent_queue_count,
  };
  iree_hal_amdgpu_queue_affinity_resolved_t resolved;
  if (!iree_hal_amdgpu_queue_affinity_try_resolve(domain, queue_affinity,
                                                  &resolved)) {
    return false;
  }
  *out_device_ordinal = resolved.physical_device_ordinal;
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

  const iree_hal_amdgpu_allocator_memory_pool_t* memory_pool = NULL;
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
    memory_pool = &allocator->memory_pools.host_fine[device_ordinal];
    memory_type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  } else if (requires_host_visible) {
    memory_pool = &allocator->memory_pools.device_fine[device_ordinal];
    memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                  IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  } else {
    memory_pool = &allocator->memory_pools.device_coarse[device_ordinal];
    memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  }
  if (!memory_pool->memory_pool.handle) return false;
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
    hsa_amd_memory_pool_t device_coarse_pool = {0};
    status = iree_hal_amdgpu_find_coarse_global_memory_pool(
        libhsa, topology->gpu_agents[i], &device_coarse_pool);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_allocator_query_pool_properties(
          libhsa, device_coarse_pool,
          &allocator->memory_pools.device_coarse[i]);
    }

    hsa_amd_memory_pool_t device_fine_pool = {0};
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_find_fine_global_memory_pool(
          libhsa, topology->gpu_agents[i], &device_fine_pool);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_allocator_query_pool_properties(
          libhsa, device_fine_pool, &allocator->memory_pools.device_fine[i]);
    }

    if (iree_status_is_ok(status)) {
      const iree_host_size_t host_ordinal = topology->gpu_cpu_map[i];
      status = iree_hal_amdgpu_allocator_query_pool_properties(
          libhsa,
          logical_device->system->host_memory_pools[host_ordinal].fine_pool,
          &allocator->memory_pools.host_fine[i]);
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

static iree_device_size_t iree_hal_amdgpu_allocator_min_pool_limit(
    iree_device_size_t lhs, iree_device_size_t rhs) {
  return lhs < rhs ? lhs : rhs;
}

static void iree_hal_amdgpu_allocator_query_pool_family_limits(
    const iree_hal_amdgpu_allocator_memory_pool_t* pools,
    iree_host_size_t pool_count, iree_device_size_t* out_max_allocation_size,
    iree_device_size_t* out_min_alignment) {
  iree_device_size_t max_allocation_size = pools[0].max_allocation_size;
  iree_device_size_t min_alignment = pools[0].allocation_alignment;
  for (iree_host_size_t i = 1; i < pool_count; ++i) {
    max_allocation_size = iree_hal_amdgpu_allocator_min_pool_limit(
        max_allocation_size, pools[i].max_allocation_size);
    min_alignment = iree_hal_amdgpu_allocator_min_pool_limit(
        min_alignment, pools[i].allocation_alignment);
  }
  *out_max_allocation_size = max_allocation_size;
  *out_min_alignment = min_alignment;
}

static iree_status_t iree_hal_amdgpu_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  iree_hal_amdgpu_allocator_t* allocator =
      iree_hal_amdgpu_allocator_cast(base_allocator);
  const iree_host_size_t heap_count = 3;
  *out_count = heap_count;
  if (capacity < heap_count) {
    // NOTE: lightweight as this is hit in normal pre-sizing usage.
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  memset(heaps, 0, heap_count * sizeof(*heaps));

  iree_device_size_t device_coarse_max_allocation_size = 0;
  iree_device_size_t device_coarse_min_alignment = 0;
  iree_hal_amdgpu_allocator_query_pool_family_limits(
      allocator->memory_pools.device_coarse,
      allocator->topology->gpu_agent_count, &device_coarse_max_allocation_size,
      &device_coarse_min_alignment);

  iree_device_size_t device_fine_max_allocation_size = 0;
  iree_device_size_t device_fine_min_alignment = 0;
  iree_hal_amdgpu_allocator_query_pool_family_limits(
      allocator->memory_pools.device_fine, allocator->topology->gpu_agent_count,
      &device_fine_max_allocation_size, &device_fine_min_alignment);

  iree_device_size_t host_fine_max_allocation_size = 0;
  iree_device_size_t host_fine_min_alignment = 0;
  iree_hal_amdgpu_allocator_query_pool_family_limits(
      allocator->memory_pools.host_fine, allocator->topology->gpu_agent_count,
      &host_fine_max_allocation_size, &host_fine_min_alignment);

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
  heaps[0].max_allocation_size = device_coarse_max_allocation_size;
  heaps[0].min_alignment = device_coarse_min_alignment;

  // Heap 1: fine-grained device-local memory for explicit host visibility.
  heaps[1].type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                  IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  heaps[1].allowed_usage = mappable_usage;
  heaps[1].max_allocation_size = device_fine_max_allocation_size;
  heaps[1].min_alignment = device_fine_min_alignment;

  // Heap 2: fine-grained host-local memory visible to the device.
  heaps[2].type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  heaps[2].allowed_usage = mappable_usage;
  heaps[2].max_allocation_size = host_fine_max_allocation_size;
  heaps[2].min_alignment = host_fine_min_alignment;

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
  if (!iree_device_size_is_valid_alignment(params->min_alignment)) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }

  // Guard against 0-byte allocations.
  if (*allocation_size == 0) *allocation_size = 4;

  iree_device_size_t aligned_allocation_size = 0;
  if (!iree_device_size_checked_align(*allocation_size,
                                      placement.memory_pool->allocation_granule,
                                      &aligned_allocation_size)) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }
  *allocation_size = aligned_allocation_size;

  const bool allocation_size_valid =
      aligned_allocation_size <= placement.memory_pool->max_allocation_size;
  const bool allocation_alignment_valid =
      params->min_alignment == 0 ||
      params->min_alignment <= placement.memory_pool->allocation_alignment;

  const bool allocation_compatible =
      allocation_size_valid && allocation_alignment_valid;
  const bool import_compatible =
      allocation_size_valid &&
      iree_all_bits_set(params->type,
                        IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE) &&
      !iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  if (!allocation_compatible && !import_compatible) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }

  iree_hal_buffer_compatibility_t compatibility = 0;
  if (allocation_compatible) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;
  }

  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  }
  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
  }

  if (import_compatible) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE;
  }

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
  event.pool_id = memory_placement->memory_pool->memory_pool.handle;
  event.backing_id = (uint64_t)(uintptr_t)host_ptr;
  event.physical_device_ordinal = memory_placement->physical_device_ordinal;
  event.memory_type = memory_placement->memory_type;
  event.buffer_usage = params.usage;
  event.length = allocation_size;
  event.alignment = memory_placement->memory_pool->allocation_alignment;
  if (iree_hal_amdgpu_logical_device_record_profile_memory_event(
          (iree_hal_device_t*)allocator->logical_device, &event)) {
    iree_hal_amdgpu_buffer_set_profile_allocation(
        buffer, session_id, allocation_id, event.pool_id,
        event.physical_device_ordinal, event.alignment);
  }
}

static void iree_hal_amdgpu_allocator_record_buffer_import(
    iree_hal_amdgpu_allocator_t* allocator, iree_hal_buffer_params_t params,
    const iree_hal_external_buffer_t* external_buffer,
    iree_hal_amdgpu_imported_host_release_data_t* release_data) {
  uint64_t session_id = 0;
  const uint64_t allocation_id =
      iree_hal_amdgpu_logical_device_allocate_profile_memory_allocation_id(
          (iree_hal_device_t*)allocator->logical_device, &session_id);
  if (allocation_id == 0) {
    return;
  }

  uint32_t physical_device_ordinal = UINT32_MAX;
  iree_host_size_t selected_device_ordinal = 0;
  const iree_hal_queue_affinity_t queue_affinity =
      params.queue_affinity ? params.queue_affinity
                            : IREE_HAL_QUEUE_AFFINITY_ANY;
  if (iree_hal_amdgpu_allocator_select_device_ordinal(
          allocator, queue_affinity, &selected_device_ordinal)) {
    physical_device_ordinal = (uint32_t)selected_device_ordinal;
  }

  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  event.type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT;
  event.flags = IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_EXTERNALLY_OWNED;
  event.allocation_id = allocation_id;
  event.backing_id =
      (uint64_t)(uintptr_t)external_buffer->handle.host_allocation.ptr;
  event.physical_device_ordinal = physical_device_ordinal;
  event.memory_type = params.type;
  event.buffer_usage = params.usage;
  event.length = external_buffer->size;
  event.alignment = 1;
  if (!iree_hal_amdgpu_logical_device_record_profile_memory_event(
          (iree_hal_device_t*)allocator->logical_device, &event)) {
    return;
  }

  release_data->profile_device = (iree_hal_device_t*)allocator->logical_device;
  release_data->length = external_buffer->size;
  release_data->memory_type = params.type;
  release_data->buffer_usage = params.usage;
  release_data->profile_session_id = session_id;
  release_data->profile_allocation_id = allocation_id;
  release_data->profile_physical_device_ordinal = physical_device_ordinal;
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

  if (IREE_UNLIKELY(
          !iree_device_size_is_valid_alignment(params->min_alignment))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "requested AMDGPU allocation alignment %" PRIu64
                            " is not a power-of-two",
                            (uint64_t)params->min_alignment);
  }
  if (IREE_UNLIKELY(params->min_alignment != 0 &&
                    params->min_alignment >
                        memory_placement.memory_pool->allocation_alignment)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "requested AMDGPU allocation alignment %" PRIu64
        " exceeds HSA memory pool alignment %" PRIu64,
        (uint64_t)params->min_alignment,
        (uint64_t)memory_placement.memory_pool->allocation_alignment);
  }

  // Guard against 0-byte allocations and align to the HSA allocation granule.
  if (allocation_size == 0) allocation_size = 4;
  if (!iree_device_size_checked_align(
          allocation_size, memory_placement.memory_pool->allocation_granule,
          &allocation_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation size %" PRIdsz
                            " overflows HSA memory pool allocation granule",
                            allocation_size);
  }
  if (IREE_UNLIKELY(allocation_size >
                    memory_placement.memory_pool->max_allocation_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU allocation size %" PRIu64
        " exceeds HSA memory pool max allocation size %" PRIu64,
        (uint64_t)allocation_size,
        (uint64_t)memory_placement.memory_pool->max_allocation_size);
  }

  // Allocate from the resolved HSA memory pool.
  void* host_ptr = NULL;
  iree_status_t status = iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(allocator->libhsa), memory_placement.memory_pool->memory_pool,
      (size_t)allocation_size, HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &host_ptr);

  // Grant the logical topology access to the allocation. This is intentionally
  // broad while allocator sharing semantics remain topology-wide.
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
        compat_params.access, compat_params.usage, allocation_size, byte_length,
        host_ptr, iree_hal_buffer_release_callback_null(),
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
      status = iree_status_join(
          status, iree_hsa_amd_memory_pool_free(IREE_LIBHSA(allocator->libhsa),
                                                host_ptr));
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

  iree_hal_amdgpu_hsa_cleanup_assert_success(
      iree_hsa_amd_memory_unlock_raw(data->libhsa, data->host_ptr));
  if (data->profile_allocation_id != 0) {
    iree_hal_profile_memory_event_t event =
        iree_hal_profile_memory_event_default();
    event.type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT;
    event.flags = IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_EXTERNALLY_OWNED;
    event.allocation_id = data->profile_allocation_id;
    event.backing_id = (uint64_t)(uintptr_t)data->host_ptr;
    event.physical_device_ordinal = data->profile_physical_device_ordinal;
    event.memory_type = data->memory_type;
    event.buffer_usage = data->buffer_usage;
    event.length = data->length;
    event.alignment = 1;
    iree_hal_amdgpu_logical_device_record_profile_memory_event_for_session(
        data->profile_device, data->profile_session_id, &event);
  }
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
  if (IREE_UNLIKELY(
          !iree_device_size_is_valid_alignment(params->min_alignment))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "requested AMDGPU import alignment %" PRIu64
                            " is not a power-of-two",
                            (uint64_t)params->min_alignment);
  }
  if (IREE_UNLIKELY(params->min_alignment != 0 &&
                    (params->min_alignment > IREE_HOST_SIZE_MAX ||
                     !iree_host_ptr_has_alignment(
                         external_buffer->handle.host_allocation.ptr,
                         (iree_host_size_t)params->min_alignment)))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host allocation import pointer does not satisfy requested AMDGPU "
        "alignment %" PRIu64,
        (uint64_t)params->min_alignment);
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
    if (iree_status_is_ok(status)) {
      memset(release_data, 0, sizeof(*release_data));
    }
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    release_data->libhsa = allocator->libhsa;
    release_data->host_ptr = host_ptr;
    release_data->length = external_buffer->size;
    release_data->memory_type = compat_params.type;
    release_data->buffer_usage = compat_params.usage;
    release_data->profile_physical_device_ordinal = UINT32_MAX;
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
    iree_hal_amdgpu_allocator_record_buffer_import(
        allocator, compat_params, external_buffer, release_data);
    *out_buffer = buffer;
  } else {
    if (release_data) {
      iree_allocator_free(allocator->host_allocator, release_data);
    }
    if (agent_ptr) {
      status = iree_status_join(
          status,
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
    .supports_virtual_memory = iree_hal_allocator_no_virtual_memory_supports,
    .virtual_memory_query_granularity =
        iree_hal_allocator_no_virtual_memory_query_granularity,
    .virtual_memory_reserve = iree_hal_allocator_no_virtual_memory_reserve,
    .virtual_memory_release = iree_hal_allocator_no_virtual_memory_release,
    .physical_memory_allocate = iree_hal_allocator_no_physical_memory_allocate,
    .physical_memory_free = iree_hal_allocator_no_physical_memory_free,
    .virtual_memory_map = iree_hal_allocator_no_virtual_memory_map,
    .virtual_memory_unmap = iree_hal_allocator_no_virtual_memory_unmap,
    .virtual_memory_protect = iree_hal_allocator_no_virtual_memory_protect,
    .virtual_memory_advise = iree_hal_allocator_no_virtual_memory_advise,
};
