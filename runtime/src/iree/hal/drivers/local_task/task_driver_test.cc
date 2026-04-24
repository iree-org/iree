// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Local-task driver-specific tests.
//
// These tests live outside the cross-driver CTS suite because they exercise
// behavior that is unique to the local-task driver: parsing the
// task_topology_group_count creation parameter and using it to override the
// driver's default executor topology.  hal.dispatch:concurrency is queried as
// a black-box check that the override took effect.

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::local_task {
namespace {

class TaskDriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_status_t status = iree_hal_local_task_driver_module_register(
        iree_hal_driver_registry_default());
    if (iree_status_is_already_exists(status)) {
      iree_status_ignore(status);
      status = iree_ok_status();
    }
    IREE_ASSERT_OK(status);

    IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), IREE_SV("local-task"),
        iree_allocator_system(), &driver_));

    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool_));
  }

  void TearDown() override {
    iree_async_proactor_pool_release(proactor_pool_);
    iree_hal_driver_release(driver_);
  }

  iree_hal_device_create_params_t MakeCreateParams() const {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool_;
    return create_params;
  }

  // Returns the task executor worker count for |device| as reported by the
  // device, which equals the topology group count used to construct it.
  static int64_t QueryDispatchConcurrency(iree_hal_device_t* device) {
    int64_t value = 0;
    IREE_EXPECT_OK(iree_hal_device_query_i64(device, IREE_SV("hal.dispatch"),
                                             IREE_SV("concurrency"), &value));
    return value;
  }

  iree_hal_driver_t* driver_ = nullptr;
  iree_async_proactor_pool_t* proactor_pool_ = nullptr;
};

// With no creation parameters the driver returns a device backed by the
// default queue executors and reports a non-zero worker count.
TEST_F(TaskDriverTest, DefaultDeviceHasPositiveDispatchConcurrency) {
  iree_hal_device_create_params_t create_params = MakeCreateParams();
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_create_default_device(
      driver_, &create_params, iree_allocator_system(), &device));

  EXPECT_GT(QueryDispatchConcurrency(device), 0);

  iree_hal_device_release(device);
}

// task_topology_group_count overrides the executor topology when creating a
// device by id.  Parameterized over a small set of group counts to also cover
// the single-worker degenerate case.
class TaskDriverGroupCountTest : public TaskDriverTest,
                                 public ::testing::WithParamInterface<int> {};

TEST_P(TaskDriverGroupCountTest, ByIdRespectsGroupCountOverride) {
  const int group_count = GetParam();
  const std::string value = std::to_string(group_count);
  const iree_string_pair_t params[] = {
      {{IREE_SV("task_topology_group_count")},
       {iree_make_string_view(value.data(), value.size())}},
  };

  iree_hal_device_create_params_t create_params = MakeCreateParams();
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_create_device_by_id(
      driver_, IREE_HAL_DEVICE_ID_DEFAULT, IREE_ARRAYSIZE(params), params,
      &create_params, iree_allocator_system(), &device));

  EXPECT_EQ(QueryDispatchConcurrency(device), group_count);

  iree_hal_device_release(device);
}

TEST_P(TaskDriverGroupCountTest, ByPathRespectsGroupCountOverride) {
  const int group_count = GetParam();
  const std::string value = std::to_string(group_count);
  const iree_string_pair_t params[] = {
      {{IREE_SV("task_topology_group_count")},
       {iree_make_string_view(value.data(), value.size())}},
  };

  iree_hal_device_create_params_t create_params = MakeCreateParams();
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_create_device_by_path(
      driver_, IREE_SV("local-task"), iree_string_view_empty(),
      IREE_ARRAYSIZE(params), params, &create_params, iree_allocator_system(),
      &device));

  EXPECT_EQ(QueryDispatchConcurrency(device), group_count);

  iree_hal_device_release(device);
}

// Mirrors the call site in callers that build a URI string and hand it to
// iree_hal_driver_create_device_by_uri: the URI's query string must round-trip
// through iree_uri_split_params and reach the driver's parameter parser.
TEST_P(TaskDriverGroupCountTest, ByUriRespectsGroupCountOverride) {
  const int group_count = GetParam();
  const std::string uri =
      "local-task://?task_topology_group_count=" + std::to_string(group_count);

  iree_hal_device_create_params_t create_params = MakeCreateParams();
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_create_device_by_uri(
      driver_, iree_make_string_view(uri.data(), uri.size()), &create_params,
      iree_allocator_system(), &device));

  EXPECT_EQ(QueryDispatchConcurrency(device), group_count);

  iree_hal_device_release(device);
}

INSTANTIATE_TEST_SUITE_P(GroupCounts, TaskDriverGroupCountTest,
                         ::testing::Values(1, 2, 4));

// Zero is rejected because a zero-worker executor would mean no dispatch
// concurrency at all; if a caller wants donor-thread-only execution they must
// not pass the parameter.
TEST_F(TaskDriverTest, RejectsZeroGroupCount) {
  const iree_string_pair_t params[] = {
      {{IREE_SV("task_topology_group_count")}, {IREE_SV("0")}},
  };
  iree_hal_device_create_params_t create_params = MakeCreateParams();
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_id(
      driver_, IREE_HAL_DEVICE_ID_DEFAULT, IREE_ARRAYSIZE(params), params,
      &create_params, iree_allocator_system(), &device);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
  EXPECT_EQ(device, nullptr);
}

TEST_F(TaskDriverTest, RejectsNonNumericGroupCount) {
  const iree_string_pair_t params[] = {
      {{IREE_SV("task_topology_group_count")}, {IREE_SV("not-a-number")}},
  };
  iree_hal_device_create_params_t create_params = MakeCreateParams();
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_id(
      driver_, IREE_HAL_DEVICE_ID_DEFAULT, IREE_ARRAYSIZE(params), params,
      &create_params, iree_allocator_system(), &device);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
  EXPECT_EQ(device, nullptr);
}

// Unknown keys are silently ignored so that the same parameter list can be
// passed across driver kinds.  This matches the convention used by other HAL
// drivers (see hip_driver.c).
TEST_F(TaskDriverTest, IgnoresUnknownParam) {
  const iree_string_pair_t params[] = {
      {{IREE_SV("not_a_real_local_task_param")}, {IREE_SV("anything")}},
  };
  iree_hal_device_create_params_t create_params = MakeCreateParams();
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_create_device_by_id(
      driver_, IREE_HAL_DEVICE_ID_DEFAULT, IREE_ARRAYSIZE(params), params,
      &create_params, iree_allocator_system(), &device));
  iree_hal_device_release(device);
}

}  // namespace
}  // namespace iree::hal::local_task
