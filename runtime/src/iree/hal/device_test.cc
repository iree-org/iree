// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/hal/api.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal {
namespace {

TEST(DeviceProfilingTest, BeginNoneIsCommonNoop) {
  iree_hal_mock_device_options_t mock_options;
  iree_hal_mock_device_options_initialize(&mock_options);

  iree_hal_device_t* device = NULL;
  IREE_ASSERT_OK(iree_hal_mock_device_create(&mock_options,
                                             iree_allocator_system(), &device));

  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families = IREE_HAL_DEVICE_PROFILING_DATA_NONE;
  IREE_EXPECT_OK(iree_hal_device_profiling_begin(device, &profiling_options));

  iree_hal_device_release(device);
}

TEST(DeviceProfilingTest, BeginDataRequiresSink) {
  iree_hal_mock_device_options_t mock_options;
  iree_hal_mock_device_options_initialize(&mock_options);

  iree_hal_device_t* device = NULL;
  IREE_ASSERT_OK(iree_hal_mock_device_create(&mock_options,
                                             iree_allocator_system(), &device));

  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_profiling_begin(device, &profiling_options));

  iree_hal_device_release(device);
}

TEST(DeviceProfilingOptionsTest, CloneOwnsBorrowedStringsAndArrays) {
  std::string executable_export_pattern = "scale_*";
  std::string counter_set_name = "set0";
  std::string counter_name0 = "counter0";
  std::string counter_name1 = "counter1";

  iree_string_view_t counter_names[2] = {
      iree_make_string_view(counter_name0.data(), counter_name0.size()),
      iree_make_string_view(counter_name1.data(), counter_name1.size()),
  };
  iree_hal_profile_counter_set_selection_t counter_set = {0};
  counter_set.name =
      iree_make_string_view(counter_set_name.data(), counter_set_name.size());
  counter_set.counter_name_count = IREE_ARRAYSIZE(counter_names);
  counter_set.counter_names = counter_names;

  iree_hal_device_profiling_options_t source_options = {0};
  source_options.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_EXPORT_PATTERN;
  source_options.capture_filter.executable_export_pattern =
      iree_make_string_view(executable_export_pattern.data(),
                            executable_export_pattern.size());
  source_options.counter_set_count = 1;
  source_options.counter_sets = &counter_set;

  iree_hal_device_profiling_options_t cloned_options = {0};
  iree_hal_device_profiling_options_storage_t* storage = NULL;
  IREE_ASSERT_OK(iree_hal_device_profiling_options_clone(
      &source_options, iree_allocator_system(), &cloned_options, &storage));
  ASSERT_NE(nullptr, storage);

  executable_export_pattern.assign("changed_*");
  counter_set_name.assign("changed_set");
  counter_name0.assign("changed0");
  counter_name1.assign("changed1");

  EXPECT_TRUE(iree_string_view_equal(
      cloned_options.capture_filter.executable_export_pattern,
      IREE_SV("scale_*")));
  ASSERT_NE(source_options.counter_sets, cloned_options.counter_sets);
  EXPECT_TRUE(iree_string_view_equal(cloned_options.counter_sets[0].name,
                                     IREE_SV("set0")));
  ASSERT_NE(counter_names, cloned_options.counter_sets[0].counter_names);
  EXPECT_TRUE(iree_string_view_equal(
      cloned_options.counter_sets[0].counter_names[0], IREE_SV("counter0")));
  EXPECT_TRUE(iree_string_view_equal(
      cloned_options.counter_sets[0].counter_names[1], IREE_SV("counter1")));

  iree_hal_device_profiling_options_storage_free(storage,
                                                 iree_allocator_system());
}

TEST(DeviceExternalCaptureTest, BeginRequiresProvider) {
  iree_hal_mock_device_options_t mock_options;
  iree_hal_mock_device_options_initialize(&mock_options);

  iree_hal_device_t* device = NULL;
  IREE_ASSERT_OK(iree_hal_mock_device_create(&mock_options,
                                             iree_allocator_system(), &device));

  iree_hal_device_external_capture_options_t capture_options = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_external_capture_begin(device, &capture_options));

  iree_hal_device_release(device);
}

TEST(DeviceExternalCaptureTest, BeginWithoutBackendHookIsUnimplemented) {
  iree_hal_mock_device_options_t mock_options;
  iree_hal_mock_device_options_initialize(&mock_options);

  iree_hal_device_t* device = NULL;
  IREE_ASSERT_OK(iree_hal_mock_device_create(&mock_options,
                                             iree_allocator_system(), &device));

  iree_hal_device_external_capture_options_t capture_options = {};
  capture_options.provider = IREE_SV("renderdoc");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_device_external_capture_begin(device, &capture_options));

  iree_hal_device_release(device);
}

}  // namespace
}  // namespace iree::hal
