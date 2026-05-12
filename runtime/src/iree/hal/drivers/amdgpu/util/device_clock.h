// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_DEVICE_CLOCK_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_DEVICE_CLOCK_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Device and host clock counters sampled by one platform source.
typedef struct iree_hal_amdgpu_device_clock_counters_t {
  // Device clock counter sampled for the requested GPU.
  uint64_t device_clock_counter;

  // Host CPU timestamp sampled near the device clock read.
  uint64_t host_cpu_timestamp_ns;

  // Host system clock counter sampled near the device clock read.
  uint64_t host_system_timestamp;

  // Frequency in Hz for |host_system_timestamp|.
  uint64_t host_system_frequency_hz;
} iree_hal_amdgpu_device_clock_counters_t;

// Validates that |counters| contains a usable clock-correlation sample.
iree_status_t iree_hal_amdgpu_device_clock_counters_validate(
    uint32_t driver_uid,
    const iree_hal_amdgpu_device_clock_counters_t* counters);

// Platform implementation used for device/host clock-correlation sampling.
typedef enum iree_hal_amdgpu_device_clock_source_type_e {
  IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_UNAVAILABLE = 0,
  IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_LINUX_KFD = 1,
} iree_hal_amdgpu_device_clock_source_type_t;

// Platform device-clock sampling source.
//
// Linux currently backs this with KFD's AMDKFD_IOC_GET_CLOCK_COUNTERS ioctl.
// Other platforms keep the source unavailable until their HSA runtime exposes
// equivalent device/host clock correlation.
typedef struct iree_hal_amdgpu_device_clock_source_t {
  // Active platform sampling implementation.
  iree_hal_amdgpu_device_clock_source_type_t type;

  // Opaque platform handle for the active clock source, or -1 when unavailable.
  intptr_t platform_handle;
} iree_hal_amdgpu_device_clock_source_t;

// Initializes a platform device-clock source.
iree_status_t iree_hal_amdgpu_device_clock_source_initialize(
    iree_hal_amdgpu_device_clock_source_t* out_source);

// Deinitializes |source| and releases its platform handle, if any.
void iree_hal_amdgpu_device_clock_source_deinitialize(
    iree_hal_amdgpu_device_clock_source_t* source);

// Samples clock counters for the GPU with HSA driver UID |driver_uid|.
iree_status_t iree_hal_amdgpu_device_clock_source_sample(
    const iree_hal_amdgpu_device_clock_source_t* source, uint32_t driver_uid,
    iree_hal_amdgpu_device_clock_counters_t* out_counters);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_DEVICE_CLOCK_H_
