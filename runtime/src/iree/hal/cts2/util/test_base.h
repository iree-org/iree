// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base class for HAL conformance tests.
//
// Provides the test fixture and helper utilities shared by all HAL CTS tests.
// Test logic lives in separate .cc files (core/allocator_test.cc, etc.)
// which use CTS_REGISTER_TEST_SUITE() for self-registration.
//
// Key design:
//   - Fresh driver/device per test via SetUp/TearDown (no state leakage)
//   - Capability-gated GTEST_SKIP instead of external EXCLUDED_TESTS lists
//   - Link-time composition: test suites + backends linked together
//   - Per-test device creation avoids shared static state across backends

#ifndef IREE_HAL_CTS2_UTIL_TEST_BASE_H_
#define IREE_HAL_CTS2_UTIL_TEST_BASE_H_

#include <string>
#include <string_view>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts2/util/registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

// Base class for all HAL CTS tests. Parameterized on BackendInfo.
// Creates a fresh driver + device in SetUp(), releases in TearDown().
template <typename BaseType = ::testing::TestWithParam<BackendInfo>>
class CtsTestBase : public BaseType {
 protected:
  void SetUp() override {
    const BackendInfo& backend = this->GetParam();

    iree_hal_driver_t* driver = nullptr;
    iree_hal_device_t* device = nullptr;
    iree_status_t status = backend.factory(&driver, &device);
    if (iree_status_is_unavailable(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "Backend '" << backend.name
                   << "' unavailable on this system";
      return;
    }
    IREE_ASSERT_OK(status);
    driver_ = driver;
    device_ = device;

    // Create a device group so the device gets topology info assigned.
    IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
        device_, iree_allocator_system(), &device_group_));

    device_allocator_ = iree_hal_device_allocator(device_);
    iree_hal_allocator_retain(device_allocator_);
  }

  void TearDown() override {
    if (device_allocator_) {
      iree_hal_allocator_release(device_allocator_);
      device_allocator_ = nullptr;
    }
    if (device_) {
      iree_hal_device_release(device_);
      device_ = nullptr;
    }
    // Release the device group after the device — the device holds a raw
    // pointer to the group's embedded topology.
    if (device_group_) {
      iree_hal_device_group_release(device_group_);
      device_group_ = nullptr;
    }
    if (driver_) {
      iree_hal_driver_release(driver_);
      driver_ = nullptr;
    }
  }

  // Returns the recording mode for the current test parameterization.
  // Command buffer tests registered via CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE
  // will see either kDirect or kIndirect depending on the instantiation.
  RecordingMode recording_mode() const {
    return this->GetParam().recording_mode;
  }

  // Returns the executable format string for the current backend, or nullptr
  // if the backend does not provide pre-compiled executables.
  const char* executable_format() const {
    return this->GetParam().executable_format;
  }

  // Returns pre-compiled executable data for the given file name.
  // Returns an empty span if the backend has no executable data function or
  // the file is not found.
  iree_const_byte_span_t executable_data(iree_string_view_t file_name) const {
    ExecutableDataFn data_fn = this->GetParam().executable_data;
    if (!data_fn) return iree_const_byte_span_empty();
    return data_fn(file_name);
  }

  //===--------------------------------------------------------------------===//
  // Buffer helpers
  //===--------------------------------------------------------------------===//

  void CreateUninitializedDeviceBuffer(iree_device_size_t buffer_size,
                                       iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage =
        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                                      buffer_size, &buffer));
    *out_buffer = buffer;
  }

  void CreateZeroedDeviceBuffer(iree_device_size_t buffer_size,
                                iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                                      buffer_size, &buffer));
    IREE_ASSERT_OK(iree_hal_buffer_map_zero(buffer, 0, IREE_HAL_WHOLE_BUFFER));
    *out_buffer = buffer;
  }

  template <typename PatternType>
  void CreateFilledDeviceBuffer(iree_device_size_t buffer_size,
                                PatternType pattern,
                                iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                                      buffer_size, &buffer));
    IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, IREE_HAL_WHOLE_BUFFER,
                                            &pattern, sizeof(pattern)));
    *out_buffer = buffer;
  }

  //===--------------------------------------------------------------------===//
  // Command buffer and semaphore helpers
  //===--------------------------------------------------------------------===//

  // Submits |command_buffer| to the device and waits for completion.
  iree_status_t SubmitCommandBufferAndWait(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table =
          iree_hal_buffer_binding_table_empty()) {
    iree_hal_semaphore_list_t wait_semaphores = iree_hal_semaphore_list_empty();

    iree_hal_semaphore_t* signal_semaphore = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
    uint64_t target_payload_value = 1ull;
    iree_hal_semaphore_list_t signal_semaphores = {
        /*count=*/1,
        /*semaphores=*/&signal_semaphore,
        /*payload_values=*/&target_payload_value,
    };

    iree_status_t status = iree_hal_device_queue_execute(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
        signal_semaphores, command_buffer, binding_table,
        IREE_HAL_EXECUTE_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_wait(signal_semaphore, target_payload_value,
                                       iree_infinite_timeout(),
                                       IREE_HAL_WAIT_FLAG_DEFAULT);
    }

    iree_hal_semaphore_release(signal_semaphore);
    return status;
  }

  iree_hal_command_buffer_t* CreateEmptyCommandBuffer(
      iree_host_size_t binding_capacity = 0) {
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_EXPECT_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
        binding_capacity, &command_buffer));
    IREE_EXPECT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_EXPECT_OK(iree_hal_command_buffer_end(command_buffer));
    return command_buffer;
  }

  iree_hal_semaphore_t* CreateSemaphore() {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_EXPECT_OK(
        iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
    return semaphore;
  }

  void CheckSemaphoreValue(iree_hal_semaphore_t* semaphore,
                           uint64_t expected_value) {
    uint64_t value;
    IREE_EXPECT_OK(iree_hal_semaphore_query(semaphore, &value));
    EXPECT_EQ(expected_value, value);
  }

  // Checks that |a| contains |b|: same status code, and the string
  // representation of |b| appears within |a|'s string representation.
  // Takes ownership of both statuses.
  void CheckStatusContains(iree_status_t a, iree_status_t b) {
    EXPECT_EQ(iree_status_code(a), iree_status_code(b));
    iree_allocator_t allocator = iree_allocator_system();
    char* a_string = nullptr;
    iree_host_size_t a_string_length = 0;
    EXPECT_TRUE(
        iree_status_to_string(a, &allocator, &a_string, &a_string_length));
    char* b_string = nullptr;
    iree_host_size_t b_string_length = 0;
    EXPECT_TRUE(
        iree_status_to_string(b, &allocator, &b_string, &b_string_length));
    EXPECT_TRUE(std::string_view(a_string).find(std::string_view(b_string)) !=
                std::string_view::npos);
    iree_allocator_free(allocator, a_string);
    iree_allocator_free(allocator, b_string);
    iree_status_ignore(a);
    iree_status_ignore(b);
  }

  iree_hal_driver_t* driver_ = nullptr;
  iree_hal_device_group_t* device_group_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
  iree_hal_allocator_t* device_allocator_ = nullptr;
};

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS2_UTIL_TEST_BASE_H_
