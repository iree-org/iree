// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// PM4 compute-dispatch launch-state derivation.
//
// This layer turns immutable AMDHSA kernel descriptor payloads into the shader
// registers required by a PM4 DISPATCH_DIRECT packet. The derivation is cold
// executable-load work; command buffers should record the resulting dwords and
// only patch dynamic userdata/kernarg addresses per invocation.

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_DISPATCH_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_DISPATCH_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/abi/kernel_descriptor.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_amdgpu_pm4_dispatch_launch_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_NONE = 0u,
  // Enables COMPUTE_DISPATCH_INITIATOR.ORDER_MODE for wave launch overlap.
  IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE = 1u << 0,
} iree_hal_amdgpu_pm4_dispatch_launch_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_pm4_dispatch_launch_flags_t;

enum {
  // Maximum COMPUTE_USER_DATA_N dwords this initial PM4 dispatch path can seed.
  IREE_HAL_AMDGPU_PM4_DISPATCH_USER_DATA_DWORD_CAPACITY = 16,
  // Static shader setup dwords before dynamic userdata or dispatch packets.
  IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT = 36,
};

typedef struct iree_hal_amdgpu_pm4_dispatch_launch_state_t {
  // Register values for COMPUTE_PGM_LO through COMPUTE_PGM_LO+5.
  uint32_t program[6];
  // Register values for COMPUTE_PGM_RSRC1 and COMPUTE_PGM_RSRC2.
  uint32_t resources[2];
  // Register value for COMPUTE_PGM_RSRC3 on gfx10+.
  uint32_t resource3;
  // Register value for COMPUTE_TMPRING_SIZE.
  uint32_t temporary_ring_size;
  // Register values for COMPUTE_RESTART_X/Y/Z.
  uint32_t restart[3];
  // Register value for COMPUTE_RESOURCE_LIMITS.
  uint32_t resource_limits;
  // Register values for COMPUTE_START_X/Y/Z, COMPUTE_NUM_THREAD_X/Y/Z, and
  // the two reserved dwords before COMPUTE_PGM_LO.
  uint32_t start_and_threads[8];
  // Number of COMPUTE_USER_DATA_N dwords seeded for this dispatch, including
  // compiler-required padding dwords that are zero-filled by the emitter.
  uint32_t user_data_dword_count;
  // Kernarg dword offset copied into preloaded user-data SGPRs.
  uint32_t kernarg_preload_dword_offset;
  // Kernarg dword count copied into preloaded user-data SGPRs.
  uint32_t kernarg_preload_dword_count;
  // User-data dword offset where the kernarg preload payload starts.
  uint32_t kernarg_preload_user_data_offset;
  // DISPATCH_DIRECT initiator bits for this kernel.
  uint32_t dispatch_initiator;
} iree_hal_amdgpu_pm4_dispatch_launch_state_t;

// Returns whether a descriptor can be represented by
// iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10.
bool iree_hal_amdgpu_pm4_dispatch_launch_state_is_supported_gfx10(
    const iree_hal_amdgpu_kernel_descriptor_t* descriptor,
    uint64_t kernel_object, const uint16_t workgroup_size[3],
    iree_hal_amdgpu_pm4_dispatch_launch_flags_t flags);

iree_status_t iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
    const iree_hal_amdgpu_kernel_descriptor_t* descriptor,
    uint64_t kernel_object, const uint16_t workgroup_size[3],
    iree_hal_amdgpu_pm4_dispatch_launch_flags_t flags,
    iree_hal_amdgpu_pm4_dispatch_launch_state_t* out_state);

iree_status_t iree_hal_amdgpu_pm4_dispatch_emit_setup(
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* state, uint32_t capacity,
    uint32_t* target_dwords, uint32_t* out_dword_count);

iree_status_t iree_hal_amdgpu_pm4_dispatch_emit_user_data(
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* state,
    uint64_t kernarg_address, const void* kernarg_preload_data,
    uint32_t capacity, uint32_t* target_dwords, uint32_t* out_dword_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_DISPATCH_H_
