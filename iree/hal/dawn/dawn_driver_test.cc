// Copyright 2019 Google LLC
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

#include "iree/hal/dawn/dawn_driver.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "iree/base/status_matchers.h"

namespace iree {
namespace hal {
namespace dawn {
namespace {

TEST(DawnDriverTest, CreateDefaultDevice) {
  DawnDriver dawn_driver;
  ASSERT_OK_AND_ASSIGN(auto default_device, dawn_driver.CreateDefaultDevice());
}

TEST(DawnDriverTest, EnumerateDevicesAndCreate) {
  DawnDriver dawn_driver;

  ASSERT_OK_AND_ASSIGN(auto available_devices,
                       dawn_driver.EnumerateAvailableDevices());
  ASSERT_GT(available_devices.size(), 0);

  ASSERT_OK_AND_ASSIGN(auto first_device,
                       dawn_driver.CreateDevice(available_devices[0]));
}

}  // namespace
}  // namespace dawn
}  // namespace hal
}  // namespace iree
