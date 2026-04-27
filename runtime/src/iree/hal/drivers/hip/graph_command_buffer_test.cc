// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/api.h"
#include "iree/testing/gtest.h"

namespace iree::hal::hip {
namespace {

class HipGraphCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_hip_driver_options_t driver_options;
    iree_hal_hip_driver_options_initialize(&driver_options);

    iree_hal_hip_device_params_t device_params;
    iree_hal_hip_device_params_initialize(&device_params);
    device_params.command_buffer_mode = IREE_HAL_HIP_COMMAND_BUFFER_MODE_GRAPH;
    device_params.stream_tracing = 0;

    iree_status_t status = iree_hal_hip_driver_create(
        IREE_SV("hip"), &driver_options, &device_params,
        iree_allocator_system(), &driver_);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "HIP driver not available";
    }

    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool_);
    ASSERT_TRUE(iree_status_is_ok(status));

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool_;
    status = iree_hal_driver_create_default_device(
        driver_, &create_params, iree_allocator_system(), &device_);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "No HIP device available";
    }
  }

  void TearDown() override {
    iree_hal_device_release(device_);
    device_ = NULL;
    iree_async_proactor_pool_release(proactor_pool_);
    proactor_pool_ = NULL;
    iree_hal_driver_release(driver_);
    driver_ = NULL;
  }

  // Proactor pool required by HIP device queue operations.
  iree_async_proactor_pool_t* proactor_pool_ = NULL;
  // HIP driver configured to record command buffers into HIP graphs.
  iree_hal_driver_t* driver_ = NULL;
  // Default HIP device created from |driver_|.
  iree_hal_device_t* device_ = NULL;
};

TEST_F(HipGraphCommandBufferTest, RecordsMoreThanInitialNodeCapacity) {
  constexpr uint32_t kFillNodeCount = 129;
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);

  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      allocator, buffer_params,
      /*allocation_size=*/kFillNodeCount * sizeof(uint32_t), &buffer);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    GTEST_SKIP() << "Allocator does not support transfer buffers";
  }

  iree_hal_command_buffer_t* command_buffer = NULL;
  status = iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }

  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer,
      .offset = 0,
      .length = sizeof(uint32_t),
  };
  for (uint32_t i = 0; i < kFillNodeCount && iree_status_is_ok(status); ++i) {
    target_ref.offset = i * sizeof(uint32_t);
    status = iree_hal_command_buffer_fill_buffer(
        command_buffer, target_ref, &i, sizeof(i), IREE_HAL_FILL_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);

  EXPECT_TRUE(iree_status_is_ok(status));
}

}  // namespace
}  // namespace iree::hal::hip
