// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_clock.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

TEST(DeviceClockTest, ValidateCounters) {
  iree_hal_amdgpu_device_clock_counters_t counters = {
      /*.device_clock_counter=*/1,
      /*.host_cpu_timestamp_ns=*/2,
      /*.host_system_timestamp=*/3,
      /*.host_system_frequency_hz=*/4,
  };
  IREE_EXPECT_OK(
      iree_hal_amdgpu_device_clock_counters_validate(1234, &counters));

  counters.device_clock_counter = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amdgpu_device_clock_counters_validate(1234, &counters));
  counters.device_clock_counter = 1;

  counters.host_cpu_timestamp_ns = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amdgpu_device_clock_counters_validate(1234, &counters));
  counters.host_cpu_timestamp_ns = 2;

  counters.host_system_timestamp = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amdgpu_device_clock_counters_validate(1234, &counters));
  counters.host_system_timestamp = 3;

  counters.host_system_frequency_hz = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amdgpu_device_clock_counters_validate(1234, &counters));
}

TEST(DeviceClockTest, UnavailableSourceSampleFailsExplicitly) {
  iree_hal_amdgpu_device_clock_source_t source = {
      /*.type=*/IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_UNAVAILABLE,
      /*.platform_handle=*/-1,
  };
  iree_hal_amdgpu_device_clock_counters_t counters = {
      /*.device_clock_counter=*/1,
      /*.host_cpu_timestamp_ns=*/2,
      /*.host_system_timestamp=*/3,
      /*.host_system_frequency_hz=*/4,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amdgpu_device_clock_source_sample(&source, 1234, &counters));
  EXPECT_EQ(counters.device_clock_counter, 0);
  EXPECT_EQ(counters.host_cpu_timestamp_ns, 0);
  EXPECT_EQ(counters.host_system_timestamp, 0);
  EXPECT_EQ(counters.host_system_frequency_hz, 0);
}

}  // namespace
}  // namespace iree::hal::amdgpu
