// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/profile_aqlprofile.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Raw HSA helpers
//===----------------------------------------------------------------------===//

static hsa_status_t iree_hal_amdgpu_profile_hsa_memory_pool_allocate(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_amd_memory_pool_t memory_pool,
    size_t size, uint32_t flags, void** ptr) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  (void)libhsa;
  return hsa_amd_memory_pool_allocate(memory_pool, size, flags, ptr);
#else
  return libhsa->hsa_amd_memory_pool_allocate(memory_pool, size, flags, ptr);
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC
}

static hsa_status_t iree_hal_amdgpu_profile_hsa_memory_pool_free(
    const iree_hal_amdgpu_libhsa_t* libhsa, void* ptr) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  (void)libhsa;
  return hsa_amd_memory_pool_free(ptr);
#else
  return libhsa->hsa_amd_memory_pool_free(ptr);
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC
}

static hsa_status_t iree_hal_amdgpu_profile_hsa_agents_allow_access(
    const iree_hal_amdgpu_libhsa_t* libhsa, uint32_t num_agents,
    const hsa_agent_t* agents, const uint32_t* flags, const void* ptr) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  (void)libhsa;
  return hsa_amd_agents_allow_access(num_agents, agents, flags, ptr);
#else
  return libhsa->hsa_amd_agents_allow_access(num_agents, agents, flags, ptr);
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC
}

static hsa_status_t iree_hal_amdgpu_profile_hsa_memory_copy(
    const iree_hal_amdgpu_libhsa_t* libhsa, void* target, const void* source,
    size_t size) {
#if IREE_HAL_AMDGPU_LIBHSA_STATIC
  (void)libhsa;
  return hsa_memory_copy(target, source, size);
#else
  return libhsa->hsa_memory_copy(target, source, size);
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC
}

//===----------------------------------------------------------------------===//
// aqlprofile support
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_profile_aqlprofile_register_agent(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_libaqlprofile_t* libaqlprofile,
    hsa_agent_t device_agent,
    iree_hal_amdgpu_aqlprofile_agent_handle_t* out_agent_handle) {
  char agent_name[64] = {0};
  iree_hal_amdgpu_aqlprofile_agent_info_v1_t agent_info;
  memset(&agent_info, 0, sizeof(agent_info));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent, HSA_AGENT_INFO_NAME, agent_name));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_XCC, &agent_info.xcc_num));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES,
      &agent_info.se_num));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
      &agent_info.cu_num));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE,
      &agent_info.shader_arrays_per_se));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN, &agent_info.domain));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), device_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &agent_info.location_id));
  agent_info.agent_gfxip = agent_name;
  IREE_RETURN_IF_AQLPROFILE_ERROR(
      libaqlprofile,
      libaqlprofile->aqlprofile_register_agent_info(
          out_agent_handle, &agent_info,
          IREE_HAL_AMDGPU_AQLPROFILE_AGENT_VERSION_V1),
      "registering AMDGPU profiling agent");
  return iree_ok_status();
}

hsa_status_t iree_hal_amdgpu_profile_aqlprofile_memory_alloc(
    void** ptr, uint64_t size,
    iree_hal_amdgpu_aqlprofile_buffer_desc_flags_t flags, void* user_data) {
  iree_hal_amdgpu_profile_aqlprofile_memory_context_t* context =
      (iree_hal_amdgpu_profile_aqlprofile_memory_context_t*)user_data;
  *ptr = NULL;
  if (size == 0) return HSA_STATUS_SUCCESS;

  hsa_amd_memory_pool_t memory_pool = {0};
  const bool should_clear = flags.host_access;
  const bool should_allow_device_access =
      flags.host_access && flags.device_access;
  if (flags.host_access) {
    memory_pool = context->host_memory_pools->coarse_pool;
    if (flags.memory_hint ==
        IREE_HAL_AMDGPU_AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED) {
      memory_pool = context->host_memory_pools->kernarg_pool;
    }
  } else if (flags.device_access) {
    memory_pool = context->device_coarse_pool;
  } else {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (!memory_pool.handle) return HSA_STATUS_ERROR_INVALID_ALLOCATION;

  hsa_status_t status = iree_hal_amdgpu_profile_hsa_memory_pool_allocate(
      context->libhsa, memory_pool, (size_t)size,
      HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, ptr);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (should_clear) memset(*ptr, 0, (size_t)size);

  if (should_allow_device_access) {
    status = iree_hal_amdgpu_profile_hsa_agents_allow_access(
        context->libhsa, /*num_agents=*/1, &context->device_agent,
        /*flags=*/NULL, *ptr);
    if (status != HSA_STATUS_SUCCESS) {
      iree_hal_amdgpu_profile_hsa_memory_pool_free(context->libhsa, *ptr);
      *ptr = NULL;
    }
  }
  return status;
}

void iree_hal_amdgpu_profile_aqlprofile_memory_dealloc(void* ptr,
                                                       void* user_data) {
  if (!ptr) return;
  iree_hal_amdgpu_profile_aqlprofile_memory_context_t* context =
      (iree_hal_amdgpu_profile_aqlprofile_memory_context_t*)user_data;
  iree_hal_amdgpu_profile_hsa_memory_pool_free(context->libhsa, ptr);
}

hsa_status_t iree_hal_amdgpu_profile_aqlprofile_memory_copy(void* target,
                                                            const void* source,
                                                            size_t size,
                                                            void* user_data) {
  if (size == 0) return HSA_STATUS_SUCCESS;
  iree_hal_amdgpu_profile_aqlprofile_memory_context_t* context =
      (iree_hal_amdgpu_profile_aqlprofile_memory_context_t*)user_data;
  return iree_hal_amdgpu_profile_hsa_memory_copy(context->libhsa, target,
                                                 source, size);
}
