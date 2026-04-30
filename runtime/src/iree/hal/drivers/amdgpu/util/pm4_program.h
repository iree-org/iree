// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_PROGRAM_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_PROGRAM_H_

#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Persistent immutable PM4 program storage.
//
// Queue-private PM4 IB slots are intentionally one AQL slot wide. Larger or
// longer-lived PM4 programs, such as profiling start/read/stop streams, use
// this object to keep immutable PM4 dwords in executable memory with explicit
// owner lifetime.
typedef struct iree_hal_amdgpu_pm4_program_t {
  // HSA API table used to free |dwords|. Not retained.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // HSA memory pool that owns |dwords|.
  hsa_amd_memory_pool_t memory_pool;
  // Device-visible immutable PM4 dwords referenced by PM4-IB AQL packets.
  IREE_AMDGPU_DEVICE_PTR uint32_t* dwords;
  // Number of valid PM4 dwords in |dwords|.
  uint32_t dword_count;
  // Allocated byte length of |dwords|.
  iree_host_size_t byte_length;
} iree_hal_amdgpu_pm4_program_t;

// Copies |source_dwords| into executable memory and grants |device_agent|
// access so the command processor can execute the program via PM4-IB AQL
// envelopes. The copied program is immutable after initialization.
iree_status_t iree_hal_amdgpu_pm4_program_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    hsa_amd_memory_pool_t memory_pool, const uint32_t* source_dwords,
    uint32_t dword_count, iree_hal_amdgpu_pm4_program_t* out_program);

// Releases the executable storage backing |program| and clears it on success.
iree_status_t iree_hal_amdgpu_pm4_program_release(
    iree_hal_amdgpu_pm4_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_PROGRAM_H_
