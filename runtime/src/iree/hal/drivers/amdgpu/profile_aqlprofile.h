// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PROFILE_AQLPROFILE_H_
#define IREE_HAL_DRIVERS_AMDGPU_PROFILE_AQLPROFILE_H_

#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/libaqlprofile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Callback context used by aqlprofile memory allocation and copy hooks.
typedef struct iree_hal_amdgpu_profile_aqlprofile_memory_context_t {
  // HSA API table used by raw callback-status functions.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // GPU agent that must be able to access allocated aqlprofile memory.
  hsa_agent_t device_agent;
  // Host memory pools nearest to |device_agent|.
  const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools;
  // Coarse-grained memory pool owned by |device_agent| for device-only trace
  // output buffers.
  hsa_amd_memory_pool_t device_coarse_pool;
} iree_hal_amdgpu_profile_aqlprofile_memory_context_t;

// Registers |device_agent| with aqlprofile and returns an opaque agent handle.
iree_status_t iree_hal_amdgpu_profile_aqlprofile_register_agent(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_libaqlprofile_t* libaqlprofile,
    hsa_agent_t device_agent,
    iree_hal_amdgpu_aqlprofile_agent_handle_t* out_agent_handle);

// Allocates memory requested by aqlprofile packet generation.
hsa_status_t iree_hal_amdgpu_profile_aqlprofile_memory_alloc(
    void** ptr, uint64_t size,
    iree_hal_amdgpu_aqlprofile_buffer_desc_flags_t flags, void* user_data);

// Releases memory previously allocated by
// iree_hal_amdgpu_profile_aqlprofile_memory_alloc.
void iree_hal_amdgpu_profile_aqlprofile_memory_dealloc(void* ptr,
                                                       void* user_data);

// Copies memory for aqlprofile packet generation.
hsa_status_t iree_hal_amdgpu_profile_aqlprofile_memory_copy(void* target,
                                                            const void* source,
                                                            size_t size,
                                                            void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PROFILE_AQLPROFILE_H_
