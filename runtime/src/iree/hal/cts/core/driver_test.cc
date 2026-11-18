// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class DriverTest : public CtsTestBase<> {
 protected:
  void CheckCreateDeviceViaPath(iree_string_view_t name,
                                iree_string_view_t path) {
    std::cout << "  Creating device '" << std::string(name.data, name.size)
              << "' with path '" << std::string(path.data, path.size) << "'\n";
    const bool is_default_path = iree_string_view_is_empty(path);

    iree_async_proactor_pool_t* proactor_pool = NULL;
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool));

    iree_hal_device_t* device = NULL;
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    iree_status_t status = iree_hal_driver_create_device_by_path(
        driver_, name, path, /*param_count=*/0, /*params=*/NULL, &create_params,
        iree_allocator_system(), &device);

    iree_async_proactor_pool_release(proactor_pool);

    // Bare driver URIs split into an empty device path, so every driver must
    // support empty-path creation even when explicit paths are driver-specific.
    if (!is_default_path && iree_status_is_not_found(status)) {
      iree_status_ignore(status);
      iree_hal_device_release(device);
      return;
    }

    IREE_ASSERT_OK(status);
    iree_string_view_t device_id = iree_hal_device_id(device);
    std::cout << "    Created device with id: '"
              << std::string(device_id.data, device_id.size) << "'\n";
    iree_hal_device_release(device);
  }
};

TEST_P(DriverTest, QueryAndCreateAvailableDevicesByID) {
  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver_, iree_allocator_system(), &device_info_count, &device_infos));

  iree_async_proactor_pool_t* proactor_pool = NULL;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool));

  std::cout << "Driver has " << device_info_count << " device(s)\n";
  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    std::cout << "  Creating device '"
              << std::string(device_infos[i].name.data,
                             device_infos[i].name.size)
              << "'\n";
    iree_hal_device_t* device = NULL;
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    IREE_ASSERT_OK(iree_hal_driver_create_device_by_id(
        driver_, device_infos[i].device_id, /*param_count=*/0,
        /*params=*/NULL, &create_params, iree_allocator_system(), &device));
    iree_string_view_t device_id = iree_hal_device_id(device);
    std::cout << "    Created device with id: '"
              << std::string(device_id.data, device_id.size) << "'\n";
    iree_hal_device_release(device);
  }

  iree_async_proactor_pool_release(proactor_pool);
  iree_allocator_free(iree_allocator_system(), device_infos);
}

TEST_P(DriverTest, QueryAndCreateAvailableDevicesByOrdinal) {
  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver_, iree_allocator_system(), &device_info_count, &device_infos));

  iree_async_proactor_pool_t* proactor_pool = NULL;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool));

  std::cout << "Driver has " << device_info_count << " device(s)\n";
  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    std::cout << "  Creating device '"
              << std::string(device_infos[i].name.data,
                             device_infos[i].name.size)
              << "'\n";
    iree_hal_device_t* device = NULL;
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    IREE_ASSERT_OK(iree_hal_driver_create_device_by_ordinal(
        driver_, i, /*param_count=*/0, /*params=*/NULL, &create_params,
        iree_allocator_system(), &device));
    iree_string_view_t device_id = iree_hal_device_id(device);
    std::cout << "    Created device with id: '"
              << std::string(device_id.data, device_id.size) << "'\n";
    iree_hal_device_release(device);
  }

  iree_async_proactor_pool_release(proactor_pool);
  iree_allocator_free(iree_allocator_system(), device_infos);
}

TEST_P(DriverTest, QueryAndCreateAvailableDevicesByPath) {
  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver_, iree_allocator_system(), &device_info_count, &device_infos));

  std::cout << "Driver has " << device_info_count << " device(s)\n";
  if (device_info_count == 0) GTEST_SKIP() << "No available devices";

  // Check creation via explicit path.
  bool tested_empty_path = false;
  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    tested_empty_path |= iree_string_view_is_empty(device_infos[i].path);
    CheckCreateDeviceViaPath(device_infos[i].name, device_infos[i].path);
  }

  // Check creation via empty path if we didn't already.
  if (!tested_empty_path) {
    CheckCreateDeviceViaPath(device_infos[0].name, iree_string_view_empty());
  }

  iree_allocator_free(iree_allocator_system(), device_infos);
}

// A device first wrapped in its own single-device group may later be placed in
// another group with a different frontier tracker. The second assignment must
// replace the first and leave the device's queues usable.
TEST_P(DriverTest, RegroupDevice) {
  DeviceCreateContext first_context;
  DeviceCreateContext second_context;
  IREE_ASSERT_OK(first_context.Initialize(iree_allocator_system()));
  IREE_ASSERT_OK(second_context.Initialize(iree_allocator_system()));

  iree_hal_device_t* device = NULL;
  IREE_ASSERT_OK(iree_hal_driver_create_default_device(
      driver_, first_context.params(), iree_allocator_system(), &device));

  iree_hal_device_group_t* first_group = NULL;
  IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
      device, first_context.frontier_tracker(), iree_allocator_system(),
      &first_group));
  iree_hal_device_group_t* second_group = NULL;
  IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
      device, second_context.frontier_tracker(), iree_allocator_system(),
      &second_group));
  iree_hal_device_group_release(first_group);

  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {1, &semaphore, &signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_barrier(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signal_list, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
  iree_hal_device_group_release(second_group);
  iree_hal_device_release(device);
}

CTS_REGISTER_TEST_SUITE(DriverTest);

}  // namespace iree::hal::cts
