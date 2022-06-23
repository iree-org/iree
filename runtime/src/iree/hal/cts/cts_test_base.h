// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_CTS_TEST_BASE_H_
#define IREE_HAL_CTS_CTS_TEST_BASE_H_

#include <set>
#include <string>

#include "iree/base/api.h"
#include "iree/base/string_view.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

// Registers the driver that will be used with INSTANTIATE_TEST_SUITE_P.
// Leaf test binaries must implement this function.
iree_status_t register_test_driver(iree_hal_driver_registry_t* registry);

// Returns the executable format for the driver under test.
// Leaf test binaries must implement this function.
const char* get_test_executable_format();

// Returns a file's executable data for the driver under test.
// Leaf test binaries must implement this function.
iree_const_byte_span_t get_test_executable_data(iree_string_view_t file_name);

// Common setup for tests parameterized on driver names.
class CtsTestBase : public ::testing::TestWithParam<std::string> {
 protected:
  static void SetUpTestSuite() {
    IREE_CHECK_OK(register_test_driver(iree_hal_driver_registry_default()));
  }

  virtual void SetUp() {
    const std::string& driver_name = GetParam();

    // Get driver with the given name and create its default device.
    // Skip drivers that are (gracefully) unavailable, fail if creation fails.
    iree_hal_driver_t* driver = NULL;
    iree_status_t status = TryGetDriver(driver_name, &driver);
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "Skipping test as '" << driver_name
                   << "' driver is unavailable";
      return;
    }
    IREE_ASSERT_OK(status);
    driver_ = driver;

    iree_hal_device_t* device = NULL;
    status = iree_hal_driver_create_default_device(
        driver_, iree_allocator_system(), &device);
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "Skipping test as default device for '" << driver_name
                   << "' driver is unavailable";
      return;
    }
    IREE_ASSERT_OK(status);
    iree_status_free(status);
    device_ = device;

    device_allocator_ = iree_hal_device_allocator(device_);
    iree_hal_allocator_retain(device_allocator_);
  }

  virtual void TearDown() {
    if (device_allocator_) {
      iree_hal_allocator_release(device_allocator_);
      device_allocator_ = NULL;
    }
    if (device_) {
      iree_hal_device_release(device_);
      device_ = NULL;
    }
    if (driver_) {
      iree_hal_driver_release(driver_);
      driver_ = NULL;
    }
  }

  // Submits |command_buffer| to the device and waits for it to complete before
  // returning.
  iree_status_t SubmitCommandBufferAndWait(
      iree_hal_command_category_t command_categories,
      iree_hal_command_buffer_t* command_buffer) {
    iree_hal_semaphore_t* signal_semaphore = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_create(device_, 0ull, &signal_semaphore));

    iree_hal_submission_batch_t submission_batch;

    // No wait semaphores.
    submission_batch.wait_semaphores.count = 0;
    submission_batch.wait_semaphores.semaphores = NULL;
    submission_batch.wait_semaphores.payload_values = NULL;

    iree_hal_command_buffer_t* command_buffer_ptrs[] = {command_buffer};
    submission_batch.command_buffer_count = IREE_ARRAYSIZE(command_buffer_ptrs);
    submission_batch.command_buffers = command_buffer_ptrs;

    // One signal semaphore from 0 -> 1.
    iree_hal_semaphore_t* signal_semaphore_ptrs[] = {signal_semaphore};
    uint64_t payload_values[] = {1ull};
    submission_batch.signal_semaphores.count =
        IREE_ARRAYSIZE(signal_semaphore_ptrs);
    submission_batch.signal_semaphores.semaphores = signal_semaphore_ptrs;
    submission_batch.signal_semaphores.payload_values = payload_values;

    iree_status_t status =
        iree_hal_device_queue_submit(device_, command_categories,
                                     /*queue_affinity=*/0,
                                     /*batch_count=*/1, &submission_batch);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_wait(signal_semaphore, 1ull,
                                       iree_infinite_timeout());
    }

    iree_hal_semaphore_release(signal_semaphore);
    return status;
  }

  iree_hal_driver_t* driver_ = NULL;
  iree_hal_device_t* device_ = NULL;
  iree_hal_allocator_t* device_allocator_ = NULL;

 private:
  // Gets a HAL driver with the provided name, if available.
  static iree_status_t TryGetDriver(const std::string& driver_name,
                                    iree_hal_driver_t** out_driver) {
    static std::set<std::string> unavailable_driver_names;

    // If creation failed before, don't try again.
    if (unavailable_driver_names.find(driver_name) !=
        unavailable_driver_names.end()) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE, "driver unavailable");
    }

    // No existing driver, attempt to create.
    iree_hal_driver_t* driver = NULL;
    iree_status_t status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_string_view(driver_name.data(), driver_name.size()),
        iree_allocator_system(), &driver);
    if (iree_status_is_unavailable(status)) {
      unavailable_driver_names.insert(driver_name);
    }
    if (iree_status_is_ok(status)) {
      *out_driver = driver;
    }
    return status;
  }
};

struct GenerateTestName {
  template <class ParamType>
  std::string operator()(
      const ::testing::TestParamInfo<ParamType>& info) const {
    std::string name = info.param;
    std::replace(name.begin(), name.end(), '-', '_');
    return name;
  }
};

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_CTS_TEST_BASE_H_
