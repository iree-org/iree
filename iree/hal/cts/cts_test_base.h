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
  // Per-test-suite set-up. This is called before the first test in this test
  // suite. We use it to set up drivers that must be reused between test cases
  // to work around issues with driver lifetimes (specifically SwiftShader for
  // Vulkan).
  static void SetUpTestSuite() {
    auto driver_or = DriverRegistry::shared_registry()->Create("vulkan");
    if (driver_or.ok()) {
      shared_drivers_["vulkan"] = std::move(driver_or.value());
    }
  }

  // Per-test-suite tear-down. This is called after the last test in this test
  // suite. We use it to destruct driver handles before program exit. This
  // avoids us to rely on static object destruction happening after main(). It
  // can cause unexpected problems when the driver also want to perform clean up
  // at that time.
  static void TearDownTestSuite() { shared_drivers_.clear(); }

  static std::map<std::string, ref_ptr<Driver>> shared_drivers_;

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
  static StatusOr<ref_ptr<Driver>> GetDriver(const std::string& driver_name) {
    static std::set<std::string> unavailable_driver_names;

    // If creation failed before, don't try again.
    if (unavailable_driver_names.find(driver_name) !=
        unavailable_driver_names.end()) {
      return UnavailableErrorBuilder(IREE_LOC) << "Driver unavailable";
    }

    // Reuse an existing driver if possible.
    auto found_it = shared_drivers_.find(driver_name);
    if (found_it != shared_drivers_.end()) {
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
    return std::move(driver_or.value());
  }
};

std::map<std::string, ref_ptr<Driver>> CtsTestBase::shared_drivers_;

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
