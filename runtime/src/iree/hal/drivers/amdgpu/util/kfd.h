// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_KFD_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_KFD_H_

#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// /dev/kfd ioctl
//===----------------------------------------------------------------------===//
// This exists to handle some driver features that aren't yet exposed through
// HSA APIs. Ideally we get them added and remove all of this.
//
// Tracking for AMDKFD_IOC_GET_CLOCK_COUNTERS:
// https://github.com/ROCm/ROCR-Runtime/issues/278

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

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_KFD_H_
