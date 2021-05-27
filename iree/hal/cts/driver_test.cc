// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

  IREE_LOG(INFO) << "Driver has " << device_info_count << " device(s)";
  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    IREE_LOG(INFO) << "  Creating device '"
                   << std::string(device_infos[i].name.data,
                                  device_infos[i].name.size)
                   << "'";
    iree_hal_device_t* device = NULL;
    IREE_ASSERT_OK(iree_hal_driver_create_device(
        driver_, device_infos[i].device_id, iree_allocator_system(), &device));
    iree_string_view_t device_id = iree_hal_device_id(device);
    IREE_LOG(INFO) << "  Created device with id: '"
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
