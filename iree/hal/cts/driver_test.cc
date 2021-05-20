// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/testing/driver_registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class DriverTest : public CtsTestBase {};

TEST_P(DriverTest, QueryAndCreateAvailableDevices) {
  iree_hal_device_info_t* device_infos;
  iree_host_size_t device_info_count;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver_, iree_allocator_system(), &device_infos, &device_info_count));

  std::cout << "Driver has " << device_info_count << " device(s)";
  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    std::cout << "  Creating device '"
              << std::string(device_infos[i].name.data,
                             device_infos[i].name.size)
              << "'";
    iree_hal_device_t* device = NULL;
    IREE_ASSERT_OK(iree_hal_driver_create_device(
        driver_, device_infos[i].device_id, iree_allocator_system(), &device));
    iree_string_view_t device_id = iree_hal_device_id(device);
    std::cout << "  Created device with id: '"
              << std::string(device_id.data, device_id.size) << "'";
    iree_hal_device_release(device);
  }

  iree_allocator_free(iree_allocator_system(), device_infos);
}

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, DriverTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
