// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_clock.h"

#include <inttypes.h>
#include <string.h>

#include "iree/hal/drivers/amdgpu/util/kfd.h"

iree_status_t iree_hal_amdgpu_device_clock_counters_validate(
    uint32_t driver_uid,
    const iree_hal_amdgpu_device_clock_counters_t* counters) {
  IREE_ASSERT_ARGUMENT(counters);
  if (IREE_UNLIKELY(counters->device_clock_counter == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "device clock source returned an invalid zero device_clock_counter for "
        "driver_uid=%" PRIu32,
        driver_uid);
  }
  if (IREE_UNLIKELY(counters->host_cpu_timestamp_ns == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device clock source returned an invalid zero "
                            "host_cpu_timestamp_ns for driver_uid=%" PRIu32,
                            driver_uid);
  }
  if (IREE_UNLIKELY(counters->host_system_timestamp == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device clock source returned an invalid zero "
                            "host_system_timestamp for driver_uid=%" PRIu32,
                            driver_uid);
  }
  if (IREE_UNLIKELY(counters->host_system_frequency_hz == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device clock source returned an invalid zero "
                            "host_system_frequency_hz for driver_uid=%" PRIu32,
                            driver_uid);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_device_clock_source_initialize(
    iree_hal_amdgpu_device_clock_source_t* out_source) {
  IREE_ASSERT_ARGUMENT(out_source);
  memset(out_source, 0, sizeof(*out_source));
  out_source->platform_handle = -1;

#if defined(IREE_PLATFORM_LINUX)
  int kfd = -1;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kfd_open(&kfd));
  out_source->platform_handle = (intptr_t)kfd;
  out_source->type = IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_LINUX_KFD;
#endif  // IREE_PLATFORM_LINUX

  return iree_ok_status();
}

void iree_hal_amdgpu_device_clock_source_deinitialize(
    iree_hal_amdgpu_device_clock_source_t* source) {
  if (!source) return;
  if (source->type == IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_LINUX_KFD) {
    iree_hal_amdgpu_kfd_close((int)source->platform_handle);
  }
  memset(source, 0, sizeof(*source));
  source->platform_handle = -1;
}

iree_status_t iree_hal_amdgpu_device_clock_source_sample(
    const iree_hal_amdgpu_device_clock_source_t* source, uint32_t driver_uid,
    iree_hal_amdgpu_device_clock_counters_t* out_counters) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(out_counters);
  memset(out_counters, 0, sizeof(*out_counters));

  switch (source->type) {
    case IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_LINUX_KFD: {
      iree_hal_amdgpu_kfd_clock_counters_t kfd_counters = {0};
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kfd_get_clock_counters(
          (int)source->platform_handle, driver_uid, &kfd_counters));
      out_counters->device_clock_counter = kfd_counters.gpu_clock_counter;
      out_counters->host_cpu_timestamp_ns = kfd_counters.cpu_clock_counter;
      out_counters->host_system_timestamp = kfd_counters.system_clock_counter;
      out_counters->host_system_frequency_hz = kfd_counters.system_clock_freq;
      return iree_hal_amdgpu_device_clock_counters_validate(driver_uid,
                                                            out_counters);
    }
    case IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_UNAVAILABLE:
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU device clock sampling is unavailable on this platform");
  }
}
