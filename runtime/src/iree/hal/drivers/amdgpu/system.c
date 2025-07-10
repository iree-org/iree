// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/system.h"

#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/util/kfd.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

//===----------------------------------------------------------------------===//
// Memory Pools and Regions
//===----------------------------------------------------------------------===//
// Unfortunately HSA has regions (the original concept) and then memory pools
// (the new hotness) and the API uses each in different places. They roughly
// represent the same underlying memory but do so differently enough that we
// can't interchange them and have to basically treat them as independent
// concepts everywhere.

// NOTE: we could do the filtering inline in the iteration callback but that
// requires moving a lot more state into the user data struct and means we can't
// emit iree_status_t errors. Instead we iterate over all and then do things in
// local for-loops.
typedef struct iree_hal_amdgpu_hsa_memory_pool_list_t {
  iree_host_size_t count;
  hsa_amd_memory_pool_t values[32];
} iree_hal_amdgpu_hsa_memory_pool_list_t;
static hsa_status_t iree_hal_amdgpu_iterate_hsa_memory_pool(
    hsa_amd_memory_pool_t memory_pool, void* user_data) {
  iree_hal_amdgpu_hsa_memory_pool_list_t* pool_list =
      (iree_hal_amdgpu_hsa_memory_pool_list_t*)user_data;
  if (pool_list->count + 1 >= IREE_ARRAYSIZE(pool_list->values)) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  pool_list->values[pool_list->count++] = memory_pool;
  return HSA_STATUS_SUCCESS;
}

static iree_status_t iree_hal_amdgpu_system_populate_host_memory_pools(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t host_agent,
    iree_hal_amdgpu_host_memory_pools_t* host_memory_pools) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Iterate all memory pools on the CPU agent.
  iree_hal_amdgpu_hsa_memory_pool_list_t all_memory_pools = {
      .count = 0,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_agent_iterate_memory_pools(
              IREE_LIBHSA(libhsa), host_agent,
              iree_hal_amdgpu_iterate_hsa_memory_pool, &all_memory_pools));

  for (iree_host_size_t i = 0; i < all_memory_pools.count; ++i) {
    hsa_amd_memory_pool_t pool = all_memory_pools.values[i];

    // Filter to the global segment only.
    hsa_region_segment_t segment = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hsa_amd_memory_pool_get_info(IREE_LIBHSA(libhsa), pool,
                                              HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                              &segment));
    if (segment != HSA_REGION_SEGMENT_GLOBAL) continue;

    // Only care about accessible-by-all.
    bool accessible_by_all = false;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hsa_amd_memory_pool_get_info(
            IREE_LIBHSA(libhsa), pool,
            HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &accessible_by_all));
    if (!accessible_by_all) continue;

    // Must be able to allocate. This should be true for any pool we query that
    // matches the other flags. Workgroup-private pools won't have this set.
    bool alloc_allowed = false;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hsa_amd_memory_pool_get_info(
            IREE_LIBHSA(libhsa), pool,
            HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed));
    if (!alloc_allowed) continue;

    // Only want fine-grained so we can use atomics.
    hsa_region_global_flag_t global_flag = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hsa_amd_memory_pool_get_info(
                IREE_LIBHSA(libhsa), pool,
                HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag));
    if (global_flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
      if (!host_memory_pools->fine_pool.handle) {  // first only
        host_memory_pools->fine_pool = pool;
      }
    }
  }

  iree_status_t status = iree_ok_status();
  if (!host_memory_pools->fine_pool.handle) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no accessible-by-all + fine-grained shared "
                              "memory pool is available in the system");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// NOTE: we could do the filtering inline in the iteration callback but that
// requires moving a lot more state into the user data struct and means we can't
// emit iree_status_t errors. Instead we iterate over all and then do things in
// local for-loops.
typedef struct iree_hal_amdgpu_hsa_region_list_t {
  iree_host_size_t count;
  hsa_region_t values[32];
} iree_hal_amdgpu_hsa_region_list_t;
static hsa_status_t iree_hal_amdgpu_iterate_hsa_region(hsa_region_t region,
                                                       void* user_data) {
  iree_hal_amdgpu_hsa_region_list_t* pool_list =
      (iree_hal_amdgpu_hsa_region_list_t*)user_data;
  if (pool_list->count + 1 >= IREE_ARRAYSIZE(pool_list->values)) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  pool_list->values[pool_list->count++] = region;
  return HSA_STATUS_SUCCESS;
}

static iree_status_t iree_hal_amdgpu_system_populate_host_regions(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t host_agent,
    iree_hal_amdgpu_host_memory_pools_t* host_memory_pools) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Iterate all regions on the CPU agent.
  iree_hal_amdgpu_hsa_region_list_t all_regions = {
      .count = 0,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_agent_iterate_regions(IREE_LIBHSA(libhsa), host_agent,
                                         iree_hal_amdgpu_iterate_hsa_region,
                                         &all_regions));

  for (iree_host_size_t i = 0; i < all_regions.count; ++i) {
    hsa_region_t region = all_regions.values[i];

    // Filter to the global segment only.
    hsa_region_segment_t segment = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hsa_region_get_info(IREE_LIBHSA(libhsa), region,
                                     HSA_REGION_INFO_SEGMENT, &segment));
    if (segment != HSA_REGION_SEGMENT_GLOBAL) continue;

    // Must be able to allocate. This should be true for any pool we query that
    // matches the other flags. Workgroup-private pools won't have this set.
    bool alloc_allowed = false;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hsa_region_get_info(IREE_LIBHSA(libhsa), region,
                                     HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                                     &alloc_allowed));
    if (!alloc_allowed) continue;

    // Only want fine-grained so we can use atomics.
    hsa_region_global_flag_t global_flag = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hsa_region_get_info(IREE_LIBHSA(libhsa), region,
                                 HSA_REGION_INFO_GLOBAL_FLAGS, &global_flag));
    if (global_flag & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
      if (!host_memory_pools->fine_region.handle) {  // first only
        host_memory_pools->fine_region = region;
      }
    }
  }

  iree_status_t status = iree_ok_status();
  if (!host_memory_pools->fine_region.handle) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no accessible-by-all + fine-grained shared "
                              "memory region is available in the system");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_system_t
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_system_deinitialize(
    iree_hal_amdgpu_system_t* system);

static iree_status_t iree_hal_amdgpu_system_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_system_options_t options, iree_allocator_t host_allocator,
    iree_hal_amdgpu_system_t* out_system) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_system);
  IREE_TRACE_ZONE_BEGIN(z0);

  out_system->host_allocator = host_allocator;

  // Ensure all GPU agents in the topology support compatible ISAs. They should
  // all be the same today but in the future if we start allowing heterogeneous
  // (even if just lightly) we'll want to catch issues here.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_verify_device_isa_commonality(libhsa, topology));

  // Query and validate the system information.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_system_info_query(libhsa, &out_system->info));

  // Open /dev/kfd so that we can issue ioctls directly.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_kfd_open(&out_system->kfd_fd));

  // Copy the libhsa symbol table and retain HSA for the lifetime of the system.
  // The caller may destroy the provided libhsa after this call returns.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_libhsa_copy(libhsa, &out_system->libhsa));

  // Copy the topology - today it's a plain-old-data struct and we can just
  // memcpy it. This is an implementation detail, though, and in the future if
  // it allocates anything we'll need to make sure this retains the allocations
  // or does a deep copy.
  memcpy(&out_system->topology, topology, sizeof(out_system->topology));

  // Preserved as the information may be used by components of the system.
  out_system->options = options;

  // Initialize the device library, which will load the builtin executable and
  // fail if we don't have a supported arch.
  iree_status_t status = iree_hal_amdgpu_device_library_initialize(
      &out_system->libhsa, topology, host_allocator,
      &out_system->device_library);

  // Find common/shared memory pools.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t host_ordinal = 0;
         host_ordinal < topology->cpu_agent_count; ++host_ordinal) {
      iree_hal_amdgpu_host_memory_pools_t* host_memory_pools =
          &out_system->host_memory_pools[host_ordinal];
      status = iree_hal_amdgpu_system_populate_host_memory_pools(
          libhsa, topology->cpu_agents[host_ordinal], host_memory_pools);
      if (!iree_status_is_ok(status)) break;
      status = iree_hal_amdgpu_system_populate_host_regions(
          &out_system->libhsa, topology->cpu_agents[host_ordinal],
          host_memory_pools);
      if (!iree_status_is_ok(status)) break;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_system_deinitialize(out_system);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_system_deinitialize(
    iree_hal_amdgpu_system_t* system) {
  IREE_ASSERT_ARGUMENT(system);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Unload the device library - no references to it should remain.
  iree_hal_amdgpu_device_library_deinitialize(&system->device_library);

  // Close our handle to /dev/kfd prior to (potentially) unloading HSA.
  iree_hal_amdgpu_kfd_close(system->kfd_fd);

  // This may unload HSA if we were the last retainer in the process.
  iree_hal_amdgpu_libhsa_deinitialize(&system->libhsa);

  memset(system, 0, sizeof(*system));

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_system_allocate(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_system_options_t options, iree_allocator_t host_allocator,
    iree_hal_amdgpu_system_t** out_system) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_system);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_system = NULL;

  iree_hal_amdgpu_system_t* system = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(
              host_allocator,
              sizeof(*system) + topology->cpu_agent_count *
                                    sizeof(system->host_memory_pools[0]),
              (void**)&system));

  iree_status_t status = iree_hal_amdgpu_system_initialize(
      libhsa, topology, options, host_allocator, system);

  if (iree_status_is_ok(status)) {
    *out_system = system;
  } else {
    iree_allocator_free(host_allocator, system);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_system_free(iree_hal_amdgpu_system_t* system) {
  if (!system) return;
  iree_allocator_t host_allocator = system->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_system_deinitialize(system);

  iree_allocator_free(host_allocator, system);

  IREE_TRACE_ZONE_END(z0);
}
