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

// Returns the name of the driver under test.
// Leaf test binaries must implement this function.
const char* get_test_driver_name();

// Registers the driver referenced by get_test_driver_name.
// Leaf test binaries must implement this function.
iree_status_t register_test_driver(iree_hal_driver_registry_t* registry);

// Returns the executable format for the driver under test.
// Leaf test binaries must implement this function.
const char* get_test_executable_format();

// Returns a file's executable data for the driver under test.
// Leaf test binaries must implement this function.
iree_const_byte_span_t get_test_executable_data(iree_string_view_t file_name);

enum class RecordingType {
  kDirect = 0,
  kIndirect,
};

struct GenerateTestName {
  template <typename ParamType>
  std::string operator()(
      const ::testing::TestParamInfo<ParamType>& info) const {
    switch (info.param) {
      default:
        return "";
      case RecordingType::kDirect:
        return "direct";
      case RecordingType::kIndirect:
        return "indirect";
    }
  }
};

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

// Statics available in CTSTestBase without template magic.
// Note that this header is intended to be included in a single .cc so we can
// define the static member storage here.
class CTSTestResources {
 public:
  static iree_hal_driver_t* driver_;
  static iree_hal_device_t* device_;
  static iree_hal_allocator_t* device_allocator_;
};
/*static*/ iree_hal_driver_t* CTSTestResources::driver_ = NULL;
/*static*/ iree_hal_device_t* CTSTestResources::device_ = NULL;
/*static*/ iree_hal_allocator_t* CTSTestResources::device_allocator_ = NULL;

// Common setup for tests parameterized on driver names.
template <typename BaseType = ::testing::Test>
class CTSTestBase : public BaseType, public CTSTestResources {
 public:
  static void SetUpTestSuite() {
    iree_status_t status =
        register_test_driver(iree_hal_driver_registry_default());
    if (iree_status_is_already_exists(status)) {
      status = iree_status_ignore(status);
    }
    IREE_CHECK_OK(status);

    // Get driver with the given name and create its default device.
    // Skip drivers that are (gracefully) unavailable, fail if creation fails.
    iree_hal_driver_t* driver = NULL;
    status = TryGetDriver(get_test_driver_name(), &driver);
    if (iree_status_is_unavailable(status)) {
      iree_status_ignore(status);
      return;
    }
    IREE_CHECK_OK(status);
    driver_ = driver;

    iree_hal_device_t* device = NULL;
    status = iree_hal_driver_create_default_device(
        driver_, iree_allocator_system(), &device);
    if (iree_status_is_unavailable(status)) {
      iree_status_ignore(status);
      return;
    }
    IREE_CHECK_OK(status);
    device_ = device;

    device_allocator_ = iree_hal_device_allocator(device_);
    iree_hal_allocator_retain(device_allocator_);
  }

  static void TearDownTestSuite() {
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

  virtual void SetUp() {
    if (!driver_) {
      GTEST_SKIP() << "Skipping test as '" << get_test_driver_name()
                   << "' driver is unavailable";
      return;
    }
    if (!device_) {
      GTEST_SKIP() << "Skipping test as default device for '"
                   << get_test_driver_name() << "' driver is unavailable";
      return;
    }
  }

  virtual void TearDown() {}

  void CreateUninitializedDeviceBuffer(iree_device_size_t buffer_size,
                                       iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage =
        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    iree_hal_buffer_t* device_buffer = NULL;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_), params, buffer_size, out_buffer));
  }

  void CreateZeroedDeviceBuffer(iree_device_size_t buffer_size,
                                iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* device_buffer = NULL;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_), params, buffer_size,
        &device_buffer));
    IREE_ASSERT_OK(
        iree_hal_buffer_map_zero(device_buffer, 0, IREE_WHOLE_BUFFER));
    *out_buffer = device_buffer;
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
    iree_hal_buffer_t* device_buffer = NULL;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_), params, buffer_size,
        &device_buffer));
    IREE_ASSERT_OK(iree_hal_buffer_map_fill(device_buffer, 0, IREE_WHOLE_BUFFER,
                                            &pattern, sizeof(pattern)));
    *out_buffer = device_buffer;
  }

  // Submits |command_buffer| to the device and waits for it to complete before
  // returning.
  iree_status_t SubmitCommandBufferAndWait(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table =
          iree_hal_buffer_binding_table_empty()) {
    return SubmitCommandBuffersAndWait(1, &command_buffer, &binding_table);
  }

  // Submits |command_buffers| to the device and waits for them to complete
  // before returning.
  iree_status_t SubmitCommandBuffersAndWait(
      iree_host_size_t command_buffer_count,
      iree_hal_command_buffer_t** command_buffers,
      const iree_hal_buffer_binding_table_t* binding_tables = nullptr) {
    // No wait semaphores.
    iree_hal_semaphore_list_t wait_semaphores = iree_hal_semaphore_list_empty();

    // One signal semaphore from 0 -> 1.
    iree_hal_semaphore_t* signal_semaphore = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_create(device_, 0ull, &signal_semaphore));
    uint64_t target_payload_value = 1ull;
    iree_hal_semaphore_list_t signal_semaphores = {
        /*count=*/1,
        /*semaphores=*/&signal_semaphore,
        /*payload_values=*/&target_payload_value,
    };

    iree_status_t status = iree_hal_device_queue_execute(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
        signal_semaphores, command_buffer_count, command_buffers,
        binding_tables);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_wait(signal_semaphore, target_payload_value,
                                       iree_infinite_timeout());
    }

    iree_hal_semaphore_release(signal_semaphore);
    return status;
  }

  iree_hal_command_buffer_t* CreateEmptyCommandBuffer(
      iree_host_size_t binding_capacity = 0) {
    iree_hal_command_buffer_t* command_buffer = NULL;
    IREE_EXPECT_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
        binding_capacity, &command_buffer));
    IREE_EXPECT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_EXPECT_OK(iree_hal_command_buffer_end(command_buffer));
    return command_buffer;
  }

  iree_hal_semaphore_t* CreateSemaphore() {
    iree_hal_semaphore_t* semaphore = NULL;
    IREE_EXPECT_OK(iree_hal_semaphore_create(device_, 0, &semaphore));
    return semaphore;
  }

  void CheckSemaphoreValue(iree_hal_semaphore_t* semaphore,
                           uint64_t expected_value) {
    uint64_t value;
    IREE_EXPECT_OK(iree_hal_semaphore_query(semaphore, &value));
    EXPECT_EQ(expected_value, value);
  }
};

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_CTS_TEST_BASE_H_
