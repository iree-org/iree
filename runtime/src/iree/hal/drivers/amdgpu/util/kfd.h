// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_KFD_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_KFD_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// /dev/kfd ioctl
//===----------------------------------------------------------------------===//
// This exists to handle some driver features that aren't yet exposed through
// HSA APIs. Ideally we get them added and remove all of this. This is known
// non-portable and code using it must be tolerant (usually #ifdef'd on
// platform).

// Tries to open /dev/kfd for read/write access.
// It should be exceptionally rare that this fails: if HSA has already been
// initialized successfully the only expected failure condition would be file
// handle exhaustion.
iree_status_t iree_hal_amdgpu_kfd_open(int* out_fd);

// Closes an open /dev/kfd handle.
void iree_hal_amdgpu_kfd_close(int fd);

// Interrupt-tolerant ioctl.
int iree_hal_amdgpu_ioctl(int fd, unsigned long request, void* arg);

//===----------------------------------------------------------------------===//
// AMDKFD_IOC_GET_CLOCK_COUNTERS
//===----------------------------------------------------------------------===//
// Tracking for adding AMDKFD_IOC_GET_CLOCK_COUNTERS to the API:
// https://github.com/ROCm/ROCR-Runtime/issues/278

typedef struct iree_hal_amdgpu_clock_counters_t {
  uint64_t gpu_clock_counter;
  uint64_t cpu_clock_counter;
  uint64_t system_clock_counter;
  uint64_t system_clock_freq;
} iree_hal_amdgpu_clock_counters_t;

// Equivalent to `hsaKmtGetClockCounters` in the ROCR KMT.
// |fd| must be an open /dev/kfd file handle.
// |gpu_uid| must be the HSA_AMD_AGENT_INFO_DRIVER_UID of the node to query.
iree_status_t iree_hal_amdgpu_kfd_get_clock_counters(
    int fd, uint32_t gpu_uid, iree_hal_amdgpu_clock_counters_t* out_counters);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_KFD_H_
