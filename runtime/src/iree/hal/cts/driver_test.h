// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_DRIVER_TEST_H_
#define IREE_HAL_CTS_DRIVER_TEST_H_

#include <iostream>
#include <string>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class driver_test : public CtsTestBase {};

TEST_P(driver_test, QueryAndCreateAvailableDevicesByID) {
  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver_, iree_allocator_system(), &device_info_count, &device_infos));

  std::cout << "Driver has " << device_info_count << " device(s)\n";
  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    std::cout << "  Creating device '"
              << std::string(device_infos[i].name.data,
                             device_infos[i].name.size)
              << "'..\n";
    iree_hal_device_t* device = NULL;
    IREE_ASSERT_OK(iree_hal_driver_create_device_by_id(
        driver_, device_infos[i].device_id, /*param_count=*/0, /*params=*/NULL,
        iree_allocator_system(), &device));
    iree_string_view_t device_id = iree_hal_device_id(device);
    std::cout << "    Created device with id: '"
              << std::string(device_id.data, device_id.size) << "'\n";
    iree_hal_device_release(device);
  }

  iree_allocator_free(iree_allocator_system(), device_infos);
}

TEST_P(driver_test, QueryAndCreateAvailableDevicesByOrdinal) {
  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver_, iree_allocator_system(), &device_info_count, &device_infos));

  std::cout << "Driver has " << device_info_count << " device(s)\n";
  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    std::cout << "  Creating device '"
              << std::string(device_infos[i].name.data,
                             device_infos[i].name.size)
              << "'..\n";
    iree_hal_device_t* device = NULL;
    IREE_ASSERT_OK(iree_hal_driver_create_device_by_ordinal(
        driver_, i, /*param_count=*/0, /*params=*/NULL, iree_allocator_system(),
        &device));
    iree_string_view_t device_id = iree_hal_device_id(device);
    std::cout << "    Created device with id: '"
              << std::string(device_id.data, device_id.size) << "'\n";
    iree_hal_device_release(device);
  }

  iree_allocator_free(iree_allocator_system(), device_infos);
}

TEST_P(driver_test, QueryAndCreateAvailableDevicesByPath) {
  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver_, iree_allocator_system(), &device_info_count, &device_infos));

  std::cout << "Driver has " << device_info_count << " device(s)\n";

  if (device_info_count > 0) {
    iree_string_view_t name = device_infos[0].name;
    std::cout << "  Creating device '" << std::string(name.data, name.size)
              << "' with empty path..\n";
    iree_hal_device_t* device = NULL;
    IREE_ASSERT_OK(iree_hal_driver_create_device_by_path(
        driver_, name, iree_string_view_empty(),
        /*param_count=*/0, /*params=*/NULL, iree_allocator_system(), &device));
    iree_string_view_t device_id = iree_hal_device_id(device);
    std::cout << "    Created device with id: '"
              << std::string(device_id.data, device_id.size) << "'\n";
    iree_hal_device_release(device);
  }

  // Creation via normal index or UUID path is HAL driver specific.
  // Allow unimplemented cases for these.

  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    iree_string_view_t name = device_infos[i].name;
    std::cout << "  Creating device '" << std::string(name.data, name.size)
              << "' with index #" << i << "..\n";
    iree_hal_device_t* device = NULL;
    char index[8];
    snprintf(index, 8, "%d", i);
    iree_status_t status = iree_hal_driver_create_device_by_path(
        driver_, name, IREE_SV(index), /*param_count=*/0, /*params=*/NULL,
        iree_allocator_system(), &device);
    if (iree_status_is_not_found(status)) {
      iree_status_consume_code(status);
    } else {
      IREE_ASSERT_OK(status);
      iree_string_view_t device_id = iree_hal_device_id(device);
      std::cout << "    Created device with id: '"
                << std::string(device_id.data, device_id.size) << "'\n";
    }
    iree_hal_device_release(device);
  }

  iree_allocator_free(iree_allocator_system(), device_infos);
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_DRIVER_TEST_H_
