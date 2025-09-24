// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/vmem.h"

#include "iree/hal/drivers/amdgpu/util/topology.h"

//===----------------------------------------------------------------------===//
// Virtual Memory Utilities
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_find_global_memory_pool_state_t {
  const iree_hal_amdgpu_libhsa_t* libhsa;
  hsa_amd_memory_pool_global_flag_t match_flags;
  hsa_amd_memory_pool_t best_pool;
} iree_hal_amdgpu_find_global_memory_pool_state_t;
static hsa_status_t iree_hal_amdgpu_find_global_memory_pool_iterator(
    hsa_amd_memory_pool_t memory_pool, void* user_data) {
  iree_hal_amdgpu_find_global_memory_pool_state_t* state =
      (iree_hal_amdgpu_find_global_memory_pool_state_t*)user_data;

  // Filter to the global segment only.
  hsa_region_segment_t segment = 0;
  IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(state->libhsa), memory_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
      &segment));
  if (segment != HSA_REGION_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

  // Must be able to allocate. This should be true for any pool we query that
  // matches the other flags. Workgroup-private pools won't have this set.
  bool alloc_allowed = false;
  IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(state->libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed));
  if (!alloc_allowed) return HSA_STATUS_SUCCESS;

  // Match if flags are present.
  hsa_region_global_flag_t global_flag = 0;
  IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(state->libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag));
  if (global_flag & state->match_flags) {
    state->best_pool = memory_pool;
    return HSA_STATUS_INFO_BREAK;
  }

  return HSA_STATUS_SUCCESS;
}

iree_status_t iree_hal_amdgpu_find_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_global_flag_t match_flags,
    hsa_amd_memory_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_pool, 0, sizeof(*out_pool));

  iree_hal_amdgpu_find_global_memory_pool_state_t find_state = {
      .libhsa = libhsa,
      .match_flags = match_flags,
      .best_pool = {0},
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_agent_iterate_memory_pools(
              IREE_LIBHSA(libhsa), agent,
              iree_hal_amdgpu_find_global_memory_pool_iterator, &find_state));
  if (!find_state.best_pool.handle) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_NOT_FOUND,
                             "no memory pool matching the required flags %u",
                             match_flags));
  }

  *out_pool = find_state.best_pool;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_find_coarse_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_t* out_pool) {
  return iree_hal_amdgpu_find_global_memory_pool(
      libhsa, agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, out_pool);
}

iree_status_t iree_hal_amdgpu_find_fine_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_t* out_pool) {
  return iree_hal_amdgpu_find_global_memory_pool(
      libhsa, agent,
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED,
      out_pool);
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_vmem_ringbuffer_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_vmem_ringbuffer_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t local_agent,
    hsa_amd_memory_pool_t memory_pool, iree_device_size_t min_capacity,
    iree_host_size_t access_desc_count,
    const hsa_amd_memory_access_desc_t* access_descs,
    iree_hal_amdgpu_vmem_ringbuffer_t* out_ringbuffer) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_ringbuffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, min_capacity);
  memset(out_ringbuffer, 0, sizeof(*out_ringbuffer));

  // hsa_amd_vmem_handle_create wants values aligned to this value.
  size_t alloc_rec_granule = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_memory_pool_get_info(
              IREE_LIBHSA(libhsa), memory_pool,
              HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
              &alloc_rec_granule));

  // Round up capacity and alignment to the allocation granule.
  const size_t alignment = alloc_rec_granule;
  const size_t capacity = iree_device_align(min_capacity, alloc_rec_granule);
  out_ringbuffer->capacity = capacity;

  // Reserve the virtual address space for the 3x the capacity. We'll map the
  // physical allocation into this address space.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_amd_vmem_address_reserve_align(
          IREE_LIBHSA(libhsa), &out_ringbuffer->va_base_ptr, capacity * 3,
          /*address=*/0, alignment, /*flags=*/0),
      "reserving ringbuffer capacity*3 (%" PRIdsz "*3=%" PRIdsz ")", capacity,
      capacity * 3);
  out_ringbuffer->ring_base_ptr =
      (uint8_t*)out_ringbuffer->va_base_ptr + capacity;

  // Allocate the physical memory for backing the ringbuffer.
  iree_status_t status = iree_hsa_amd_vmem_handle_create(
      IREE_LIBHSA(libhsa), memory_pool, capacity, MEMORY_TYPE_NONE,
      /*flags=*/0, &out_ringbuffer->alloc_handle);

  void* va_offsets[3] = {
      (uint8_t*)out_ringbuffer->va_base_ptr + 0 * capacity,
      (uint8_t*)out_ringbuffer->va_base_ptr + 1 * capacity,
      (uint8_t*)out_ringbuffer->va_base_ptr + 2 * capacity,
  };

  // Map the physical allocation into the virtual address space 3 times
  // (prev, base, next).
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < 3; ++i) {
    status =
        iree_hsa_amd_vmem_map(IREE_LIBHSA(libhsa), va_offsets[i], capacity, 0,
                              out_ringbuffer->alloc_handle, /*flags=*/0);
  }

  // Enable access on requested devices (no access by default).
  // Must be done per memory handle, not the entire VA.
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < 3; ++i) {
    status =
        iree_hsa_amd_vmem_set_access(IREE_LIBHSA(libhsa), va_offsets[i],
                                     capacity, access_descs, access_desc_count);
    if (!iree_status_is_ok(status)) break;
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_vmem_ringbuffer_deinitialize(libhsa, out_ringbuffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_vmem_ringbuffer_initialize_with_topology(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t local_agent,
    hsa_amd_memory_pool_t memory_pool, iree_device_size_t min_capacity,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_vmem_access_mode_t access_mode,
    iree_hal_amdgpu_vmem_ringbuffer_t* out_ringbuffer) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_ringbuffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate scratch for the access descriptors. Note that though we allocate
  // for all agents we don't pass agents with HSA_ACCESS_PERMISSION_NONE as that
  // actually causes HSA to allocate information about that agent.
  // HSA_ACCESS_PERMISSION_NONE should only be used to _remove_ access that was
  // previous granted.
  iree_host_size_t access_desc_count = 0;
  hsa_amd_memory_access_desc_t* access_descs =
      (hsa_amd_memory_access_desc_t*)iree_alloca(
          topology->all_agent_count * sizeof(hsa_amd_memory_access_desc_t));

  // Populate the access list.
  switch (access_mode) {
    case IREE_HAL_AMDGPU_ACCESS_MODE_SHARED: {
      // All devices get read/write access.
      for (iree_host_size_t i = 0; i < topology->all_agent_count; ++i) {
        access_descs[access_desc_count++] = (hsa_amd_memory_access_desc_t){
            .agent_handle = topology->all_agents[i],
            .permissions = HSA_ACCESS_PERMISSION_RW,
        };
      }
    } break;
    case IREE_HAL_AMDGPU_ACCESS_MODE_EXCLUSIVE: {
      // Only the local agent can access the ringbuffer.
      access_descs[access_desc_count++] = (hsa_amd_memory_access_desc_t){
          .agent_handle = local_agent,
          .permissions = HSA_ACCESS_PERMISSION_RW,
      };
    } break;
    case IREE_HAL_AMDGPU_ACCESS_MODE_EXCLUSIVE_CONSUMER: {
      // Local agent gets read, all agents get write.
      for (iree_host_size_t i = 0; i < topology->all_agent_count; ++i) {
        hsa_agent_t agent = topology->all_agents[i];
        hsa_access_permission_t permissions = HSA_ACCESS_PERMISSION_NONE;
        if (agent.handle == local_agent.handle) {
          permissions = HSA_ACCESS_PERMISSION_RO;
        } else {
          permissions = HSA_ACCESS_PERMISSION_WO;
        }
        access_descs[access_desc_count++] = (hsa_amd_memory_access_desc_t){
            .agent_handle = topology->all_agents[i],
            .permissions = permissions,
        };
      }
    } break;
    case IREE_HAL_AMDGPU_ACCESS_MODE_EXCLUSIVE_PRODUCER: {
      // Local agent gets write, all agents get read.
      for (iree_host_size_t i = 0; i < topology->all_agent_count; ++i) {
        hsa_agent_t agent = topology->all_agents[i];
        hsa_access_permission_t permissions = HSA_ACCESS_PERMISSION_NONE;
        if (agent.handle == local_agent.handle) {
          permissions = HSA_ACCESS_PERMISSION_WO;
        } else {
          permissions = HSA_ACCESS_PERMISSION_RO;
        }
        access_descs[access_desc_count++] = (hsa_amd_memory_access_desc_t){
            .agent_handle = topology->all_agents[i],
            .permissions = permissions,
        };
      }
    } break;
    default: {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "unhandled access mode"));
    } break;
  }

  // Route to the explicit initializer.
  iree_status_t status = iree_hal_amdgpu_vmem_ringbuffer_initialize(
      libhsa, local_agent, memory_pool, min_capacity, access_desc_count,
      access_descs, out_ringbuffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_vmem_ringbuffer_deinitialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_vmem_ringbuffer_t* ringbuffer) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(ringbuffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Unmap physical allocation and release it.
  if (ringbuffer->alloc_handle.handle) {
    void* va_offsets[3] = {
        (uint8_t*)ringbuffer->va_base_ptr + 0 * ringbuffer->capacity,
        (uint8_t*)ringbuffer->va_base_ptr + 1 * ringbuffer->capacity,
        (uint8_t*)ringbuffer->va_base_ptr + 2 * ringbuffer->capacity,
    };
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(va_offsets); ++i) {
      IREE_IGNORE_ERROR(iree_hsa_amd_vmem_unmap(
          IREE_LIBHSA(libhsa), va_offsets[i], ringbuffer->capacity));
    }
    IREE_IGNORE_ERROR(iree_hsa_amd_vmem_handle_release(
        IREE_LIBHSA(libhsa), ringbuffer->alloc_handle));
  }

  if (ringbuffer->va_base_ptr) {
    IREE_IGNORE_ERROR(iree_hsa_amd_vmem_address_free(IREE_LIBHSA(libhsa),
                                                     ringbuffer->va_base_ptr,
                                                     ringbuffer->capacity * 3));
  }

  memset(ringbuffer, 0, sizeof(*ringbuffer));

  IREE_TRACE_ZONE_END(z0);
}
