// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

TEST(AmdgpuDriverOptionsTest, DriverParamsAreRejectedUntilDefined) {
  iree_hal_amdgpu_driver_options_t options;
  iree_hal_amdgpu_driver_options_initialize(&options);
  const iree_string_pair_t pair = iree_make_cstring_pair("unknown", "value");

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_driver_options_parse(&options, (iree_string_pair_list_t){
                                                         .count = 1,
                                                         .pairs = &pair,
                                                     }));
}

TEST(AmdgpuDriverOptionsTest, LogicalDeviceParamsAreRejectedUntilDefined) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  const iree_string_pair_t pair = iree_make_cstring_pair("unknown", "value");

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_logical_device_options_parse(
                            &options, (iree_string_pair_list_t){
                                          .count = 1,
                                          .pairs = &pair,
                                      }));
}

TEST(AmdgpuDriverOptionsTest, RejectsMissingSearchPathStorageBeforeLoadingHsa) {
  iree_hal_amdgpu_driver_options_t options;
  iree_hal_amdgpu_driver_options_initialize(&options);
  options.libhsa_search_paths = (iree_string_view_list_t){
      .count = 1,
      .values = NULL,
  };

  iree_hal_driver_t* driver = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_driver_create(IREE_SV("amdgpu"), &options,
                                    iree_allocator_system(), &driver));
  iree_hal_driver_release(driver);
}

TEST(AmdgpuDriverOptionsTest, RejectsMissingSearchPathDataBeforeLoadingHsa) {
  iree_hal_amdgpu_driver_options_t options;
  iree_hal_amdgpu_driver_options_initialize(&options);
  const iree_string_view_t search_path = iree_make_string_view(NULL, 1);
  options.libhsa_search_paths = (iree_string_view_list_t){
      .count = 1,
      .values = &search_path,
  };

  iree_hal_driver_t* driver = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_driver_create(IREE_SV("amdgpu"), &options,
                                    iree_allocator_system(), &driver));
  iree_hal_driver_release(driver);
}

static iree_status_t CreateDriverWithDefaultDeviceOptions(
    const iree_hal_amdgpu_logical_device_options_t* device_options) {
  iree_hal_amdgpu_driver_options_t options;
  iree_hal_amdgpu_driver_options_initialize(&options);
  options.default_device_options = *device_options;
  iree_hal_driver_t* driver = NULL;
  iree_status_t status = iree_hal_amdgpu_driver_create(
      IREE_SV("amdgpu"), &options, iree_allocator_system(), &driver);
  iree_hal_driver_release(driver);
  return status;
}

TEST(AmdgpuDriverOptionsTest, RejectsDeviceQueuePlacementBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.queue_placement = IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsInvalidQueuePlacementBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.queue_placement = (iree_hal_amdgpu_queue_placement_t)99;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsTraceExecutionBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.trace_execution = 1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsExclusiveExecutionBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.exclusive_execution = 1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsNegativeActiveWaitBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.wait_active_for_ns = -1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsActiveWaitBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.wait_active_for_ns = 1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

}  // namespace
}  // namespace iree::hal::amdgpu
