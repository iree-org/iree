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
#include <mutex>
#include <set>

#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// TODO(3934): rebase this all on the C API.
#include "iree/hal/driver.h"

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
  //
  // TODO(#3933): this is a very nasty hack that indicates a serious issue. If
  // we have to do it here in our test suite it means that every user of IREE
  // will also have to do something like it. We should be reusing all drivers
  // across tests in a suite (removing the vulkan-specific behavior here) but
  // ALSO need a test that tries to create a driver twice.
  static void SetUpTestSuite() {
    iree_hal_driver_t* driver = NULL;
    iree_status_t status = iree_hal_driver_registry_try_create_by_name(
        iree_hal_driver_registry_default(), iree_make_cstring_view("vulkan"),
        iree_allocator_system(), &driver);
    if (iree_status_consume_code(status) == IREE_STATUS_OK) {
      shared_drivers_["vulkan"] =
          assign_ref(reinterpret_cast<iree::hal::Driver*>(driver));
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
      IREE_LOG(WARNING) << "Skipping test as driver is unavailable: "
                        << driver_or.status();
      GTEST_SKIP();
      return;
    }
    IREE_ASSERT_OK_AND_ASSIGN(driver_, std::move(driver_or));
    IREE_LOG(INFO) << "Creating default device...";
    IREE_ASSERT_OK_AND_ASSIGN(device_, driver_->CreateDefaultDevice());
    IREE_LOG(INFO) << "Created device '" << device_->info().name() << "'";
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
      IREE_LOG(INFO) << "Reusing existing driver '" << driver_name << "'...";
      return add_ref(found_it->second);
    }

    // No existing driver, attempt to create.
    IREE_LOG(INFO) << "Creating driver '" << driver_name << "'...";
    iree_hal_driver_t* driver = NULL;
    iree_status_t status = iree_hal_driver_registry_try_create_by_name(
        iree_hal_driver_registry_default(),
        iree_make_string_view(driver_name.data(), driver_name.size()),
        iree_allocator_system(), &driver);
    if (iree_status_is_unavailable(status)) {
      unavailable_driver_names.insert(driver_name);
    }
    IREE_RETURN_IF_ERROR(status);
    return assign_ref(reinterpret_cast<iree::hal::Driver*>(driver));
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
