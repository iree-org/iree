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

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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
    iree_hal_driver_t* driver;
    iree_status_t status = TryGetDriver(driver_name, &driver);
    if (iree_status_is_unavailable(status)) {
      IREE_LOG(WARNING) << "Skipping test as driver is unavailable";
      GTEST_SKIP();
      return;
    }
    driver_ = driver;

    iree_hal_device_t* device;
    status = iree_hal_driver_create_default_device(
        driver_, iree_allocator_system(), &device);
    IREE_ASSERT_OK(status);
    device_ = device;
    iree_string_view_t device_id = iree_hal_device_id(device_);

    device_allocator_ = iree_hal_device_allocator(device_);
    iree_hal_allocator_retain(device_allocator_);
  }

  virtual void TearDown() {
    if (device_allocator_) {
      iree_hal_allocator_release(device_allocator_);
      device_allocator_ = nullptr;
    }
    if (device_) {
      iree_hal_device_release(device_);
      device_ = nullptr;
    }
    if (driver_) {
      iree_hal_driver_release(driver_);
      driver_ = nullptr;
    }
  }

  iree_hal_driver_t* driver_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
  iree_hal_allocator_t* device_allocator_ = nullptr;

 private:
  // Gets a HAL driver with the provided name, if available.
  static iree_status_t TryGetDriver(const std::string& driver_name,
                                    iree_hal_driver_t** out_driver) {
    static std::set<std::string> unavailable_driver_names;

    // If creation failed before, don't try again.
    if (unavailable_driver_names.find(driver_name) !=
        unavailable_driver_names.end()) {
      return UnavailableErrorBuilder(IREE_LOC) << "Driver unavailable";
    }

    // No existing driver, attempt to create.
    iree_hal_driver_t* driver = NULL;
    iree_status_t status = iree_hal_driver_registry_try_create_by_name(
        iree_hal_driver_registry_default(),
        iree_make_string_view(driver_name.data(), driver_name.size()),
        iree_allocator_system(), &driver);
    if (iree_status_is_unavailable(status)) {
      unavailable_driver_names.insert(driver_name);
    }
    if (iree_status_is_ok(status)) {
      *out_driver = driver;
    }
    return status;
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
