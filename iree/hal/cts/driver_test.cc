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
#include "iree/hal/driver_registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class DriverTest : public CtsTestBase {};

TEST_P(DriverTest, CreateDefaultDevice) {
  IREE_LOG(INFO) << "Device details:\n" << device_->DebugString();
}

TEST_P(DriverTest, EnumerateAndCreateAvailableDevices) {
  IREE_ASSERT_OK_AND_ASSIGN(auto devices, driver_->EnumerateAvailableDevices());

  for (iree_host_size_t i = 0; i < devices.size(); ++i) {
    IREE_ASSERT_OK_AND_ASSIGN(auto device, driver_->CreateDevice(devices[i]));
    IREE_LOG(INFO) << "Device #" << i << " details:\n" << device->DebugString();
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, DriverTest,
    ::testing::ValuesIn(CtsTestBase::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
