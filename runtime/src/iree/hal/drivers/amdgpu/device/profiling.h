// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_PROFILING_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_PROFILING_H_

#include "iree/hal/drivers/amdgpu/abi/profile.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the byte offset of the harvest source table after the kernel args.
static inline size_t
iree_hal_amdgpu_device_profile_dispatch_harvest_source_offset(void) {
  return iree_amdgpu_align(
      sizeof(iree_hal_amdgpu_profile_dispatch_harvest_args_t),
      IREE_AMDGPU_ALIGNOF(iree_hal_amdgpu_profile_dispatch_harvest_source_t));
}

// Returns the kernarg byte length required for |source_count| harvest sources.
static inline size_t
iree_hal_amdgpu_device_profile_dispatch_harvest_kernarg_length(
    uint32_t source_count) {
  return iree_hal_amdgpu_device_profile_dispatch_harvest_source_offset() +
         (size_t)source_count *
             sizeof(iree_hal_amdgpu_profile_dispatch_harvest_source_t);
}

// Returns the harvest source table embedded in |kernarg_ptr|.
static inline iree_hal_amdgpu_profile_dispatch_harvest_source_t*
iree_hal_amdgpu_device_profile_dispatch_harvest_sources(
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  uint8_t* source_ptr =
      (uint8_t*)kernarg_ptr +
      iree_hal_amdgpu_device_profile_dispatch_harvest_source_offset();
  return (iree_hal_amdgpu_profile_dispatch_harvest_source_t*)source_ptr;
}

// Populates a builtin dispatch packet and kernargs that harvest timestamps from
// profiled dispatch completion signals into queue-owned event records.
//
// |dispatch_packet| and |kernarg_ptr| must point to reserved queue storage.
// The caller owns completion-signal assignment and header commit.
iree_hal_amdgpu_profile_dispatch_harvest_source_t*
iree_hal_amdgpu_device_profile_emplace_dispatch_harvest(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        harvest_kernel_args,
    uint32_t source_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Device builtin that copies per-dispatch CP timestamps into profiling records.
IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_profile_harvest_dispatch_events(
    const iree_hal_amdgpu_profile_dispatch_harvest_source_t*
        IREE_AMDGPU_RESTRICT sources,
    uint32_t source_count);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_PROFILING_H_
