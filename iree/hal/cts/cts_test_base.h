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

#ifndef IREE_HAL_CTS_CTS_TEST_BASE_H_
#define IREE_HAL_CTS_CTS_TEST_BASE_H_

#include <map>
#include <set>

#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/hal/driver_registry.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace cts {

// Common setup for tests parameterized across all registered drivers.
class CtsTestBase : public ::testing::TestWithParam<std::string> {
 protected:
  virtual void SetUp() {
    const std::string& driver_name = GetParam();

    // Get driver with the given name and create its default device.
    // Skip drivers that are (gracefully) unavailable, fail if creation fails.
    auto driver_or = GetDriver(driver_name);
    if (IsUnavailable(driver_or.status())) {
      LOG(WARNING) << "Skipping test as driver is unavailable: "
                   << driver_or.status();
      GTEST_SKIP();
      return;
    }
    ASSERT_OK_AND_ASSIGN(driver_, std::move(driver_or));
    LOG(INFO) << "Creating default device...";
    ASSERT_OK_AND_ASSIGN(device_, driver_->CreateDefaultDevice());
    LOG(INFO) << "Created device '" << device_->info().name() << "'";
  }

  ref_ptr<Driver> driver_;
  ref_ptr<Device> device_;

 private:
  // Gets a HAL driver with the provided name, if available.
  // Drivers are reused between test cases to work around issues with driver
  // lifetimes (specifically SwiftShader for Vulkan).
  static StatusOr<ref_ptr<Driver>> GetDriver(const std::string& driver_name) {
    static std::set<std::string> unavailable_driver_names;
    static std::map<std::string, ref_ptr<Driver>> drivers;

    // If creation failed before, don't try again.
    if (unavailable_driver_names.find(driver_name) !=
        unavailable_driver_names.end()) {
      return UnavailableErrorBuilder(IREE_LOC) << "Driver unavailable";
    }

    // Reuse an existing driver if possible.
    auto found_it = drivers.find(driver_name);
    if (found_it != drivers.end()) {
      LOG(INFO) << "Reusing existing driver '" << driver_name << "'...";
      return add_ref(found_it->second);
    }

    // No existing driver, attempt to create.
    LOG(INFO) << "Creating driver '" << driver_name << "'...";
    auto driver_or = DriverRegistry::shared_registry()->Create(driver_name);
    if (IsUnavailable(driver_or.status())) {
      unavailable_driver_names.insert(driver_name);
    }
    RETURN_IF_ERROR(driver_or.status());
    drivers.insert(std::make_pair(driver_name, add_ref(driver_or.value())));
    return add_ref(driver_or.value());
  }
};

struct GenerateTestName {
  template <class ParamType>
  std::string operator()(
      const ::testing::TestParamInfo<ParamType>& info) const {
    return info.param;
  }
};

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_CTS_TEST_BASE_H_
