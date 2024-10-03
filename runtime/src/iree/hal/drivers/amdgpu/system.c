// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/system.h"

#include "iree/hal/drivers/amdgpu/executable.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_system_info_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_system_info_query(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_system_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_info);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_info, 0, sizeof(*out_info));

  uint16_t version_major = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_SYSTEM_INFO_VERSION_MAJOR,
                               &version_major),
      "querying HSA_SYSTEM_INFO_VERSION_MAJOR");
  if (version_major != 1) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INCOMPATIBLE,
                             "only HSA 1.x is supported "
                             "(HSA_SYSTEM_INFO_VERSION_MAJOR == 1, have %u)",
                             version_major));
  }

  hsa_endianness_t endianness = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_SYSTEM_INFO_ENDIANNESS, &endianness),
      "querying HSA_SYSTEM_INFO_ENDIANNESS");
  if (endianness != HSA_ENDIANNESS_LITTLE) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_INCOMPATIBLE,
                "only little-endian systems are supported "
                "(HSA_SYSTEM_INFO_ENDIANNESS == HSA_ENDIANNESS_LITTLE)"));
  }

  hsa_machine_model_t machine_model = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_SYSTEM_INFO_MACHINE_MODEL,
                               &machine_model),
      "querying HSA_SYSTEM_INFO_MACHINE_MODEL");
  if (machine_model != HSA_MACHINE_MODEL_LARGE) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_INCOMPATIBLE,
                "only 64-bit systems are supported "
                "(HSA_SYSTEM_INFO_MACHINE_MODEL == HSA_MACHINE_MODEL_LARGE)"));
  }

  bool svm_supported = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED,
                               &svm_supported),
      "querying HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED");
  if (!svm_supported) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INCOMPATIBLE,
                             "only systems with SVM are supported "
                             "(HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED == true)"));
  }

  bool svm_accessible_by_default = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa,
                               HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT,
                               &svm_accessible_by_default),
      "querying HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT");
  out_info->svm_accessible_by_default = svm_accessible_by_default ? 1 : 0;

  bool dmabuf_supported = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED,
                               &dmabuf_supported),
      "querying HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED");
  out_info->dmabuf_supported = dmabuf_supported ? 1 : 0;

  bool virtual_mem_api_supported = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa,
                               HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED,
                               &virtual_mem_api_supported),
      "querying HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED");
  if (!virtual_mem_api_supported) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_INCOMPATIBLE,
                "only systems with the virtual memory API are supported "
                "(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED == true)"));
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                               &out_info->timestamp_frequency),
      "querying HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY");

  uint16_t amd_loader_minor = 0;
  bool amd_loader_supported = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_major_extension_supported(
          libhsa, HSA_EXTENSION_AMD_LOADER, 1, &amd_loader_minor,
          &amd_loader_supported),
      "querying HSA_EXTENSION_AMD_LOADER version");
  if (!amd_loader_supported) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_INCOMPATIBLE,
                "only systems with the AMD loader extension are supported "
                "(HSA_EXTENSION_AMD_LOADER v1.xx)"));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Memory Pools
//===----------------------------------------------------------------------===//

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
  pool_list->values[pool_list->count++] = memory_pool;
  return HSA_STATUS_SUCCESS;
}

static iree_status_t iree_hal_amdgpu_system_find_memory_pools(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    hsa_amd_memory_pool_t* out_shared_fine_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_shared_fine_pool, 0, sizeof(*out_shared_fine_pool));

  // Iterate all memory pools on the first CPU agent.
  iree_hal_amdgpu_hsa_memory_pool_list_t cpu_memory_pools = {
      .count = 0,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_agent_iterate_memory_pools(
              libhsa, topology->cpu_agents[0],
              iree_hal_amdgpu_iterate_hsa_memory_pool, &cpu_memory_pools));

  for (iree_host_size_t i = 0; i < cpu_memory_pools.count; ++i) {
    hsa_amd_memory_pool_t pool = cpu_memory_pools.values[i];

    // Filter to the global segment only.
    hsa_region_segment_t segment = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hsa_amd_memory_pool_get_info(
                libhsa, pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment));
    if (segment != HSA_REGION_SEGMENT_GLOBAL) continue;

    // Only care about accessible-by-all.
    bool accessible_by_all = false;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hsa_amd_memory_pool_get_info(
                libhsa, pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL,
                &accessible_by_all));
    if (!accessible_by_all) continue;

    // Must be able to allocate. This should be true for any pool we query that
    // matches the other flags. Workgroup-private pools won't have this set.
    bool alloc_allowed = false;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hsa_amd_memory_pool_get_info(
                libhsa, pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                &alloc_allowed));
    if (!alloc_allowed) continue;

    // Only want fine-grained so we can use atomics.
    hsa_region_global_flag_t global_flag = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hsa_amd_memory_pool_get_info(
            libhsa, pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag));
    if (global_flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
      if (!out_shared_fine_pool->handle) {  // first only
        *out_shared_fine_pool = pool;
      }
    }
  }

  iree_status_t status = iree_ok_status();
  if (!out_shared_fine_pool->handle) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no accessible-by-all + fine-grained shared "
                              "memory pool is available in the system");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_system_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_system_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_amdgpu_system_t* out_system) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_system);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Must have at least one of each agent type in the topology.
  // This is just a guard for creating systems that don't have any GPUs so that
  // code in the implementation can assume that there's always _something_ to
  // query.
  if (topology->cpu_agent_count == 0 || topology->gpu_agent_count == 0 ||
      topology->gpu_agent_queue_count == 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "topology is invalid; must have at least one CPU agent and one GPU "
            "agent with at least one queue, have cpu_agent_count=%" PRIhsz
            ", gpu_agent_count=%" PRIhsz ", gpu_agent_queue_count=%" PRIhsz,
            topology->cpu_agent_count, topology->gpu_agent_count,
            topology->gpu_agent_queue_count));
  }

  // Query and validate the system information.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_system_info_query(libhsa, &out_system->info));

  // Ensure all GPU agents in the topology are compatible. They should all be
  // the same today.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_verify_device_isa_commonality(libhsa, topology));

  // Copy the libhsa symbol table and retain HSA for the lifetime of the system.
  // The caller may destroy the provided libhsa after this call returns.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_libhsa_copy(libhsa, &out_system->libhsa));

  // Copy the topology - today it's a plain-old-data struct and we can just
  // memcpy it. This is an implementation detail, though, and in the future if
  // it allocates anything we'll need to make sure this retains the allocations
  // or does a deep copy.
  memcpy(&out_system->topology, topology, sizeof(out_system->topology));

  // Initialize the device library, which will load the builtin executable and
  // fail if we don't have a supported arch.
  iree_status_t status = iree_hal_amdgpu_device_library_initialize(
      libhsa, topology, host_allocator, &out_system->device_library);

  // Find common/shared memory pools.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_system_find_memory_pools(
        libhsa, topology, &out_system->shared_fine_pool);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_system_deinitialize(out_system);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_system_deinitialize(iree_hal_amdgpu_system_t* system) {
  IREE_ASSERT_ARGUMENT(system);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_device_library_deinitialize(&system->device_library);

  iree_hal_amdgpu_libhsa_deinitialize(&system->libhsa);

  IREE_TRACE_ZONE_END(z0);
}
