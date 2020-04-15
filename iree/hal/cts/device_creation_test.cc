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

namespace iree {
namespace hal {
namespace cts {

class DeviceCreationTest : public CtsTestBase {};

TEST_P(DeviceCreationTest, CreateDevice) {
  LOG(INFO) << "Device details:\n" << device_->DebugString();
}

INSTANTIATE_TEST_SUITE_P(AllDrivers, DeviceCreationTest,
                         ::testing::ValuesIn(DriverRegistry::shared_registry()
                                                 ->EnumerateAvailableDrivers()),
                         GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
