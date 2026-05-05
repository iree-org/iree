// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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

    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool_));

    iree_async_frontier_tracker_options_t frontier_options =
        iree_async_frontier_tracker_options_default();
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        frontier_options, iree_allocator_system(), &frontier_tracker_));

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool_;
    status = iree_hal_driver_create_default_device(
        driver_, &create_params, iree_allocator_system(), &device_);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "No HIP device available";
    }

    IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
        device_, frontier_tracker_, iree_allocator_system(), &device_group_));
  }

  void TearDown() override {
    iree_hal_device_release(device_);
    device_ = NULL;
    iree_hal_device_group_release(device_group_);
    device_group_ = NULL;
    iree_async_frontier_tracker_release(frontier_tracker_);
    frontier_tracker_ = NULL;
    iree_async_proactor_pool_release(proactor_pool_);
    proactor_pool_ = NULL;
    iree_hal_driver_release(driver_);
    driver_ = NULL;
  }

  // Proactor pool required by HIP device queue operations.
  iree_async_proactor_pool_t* proactor_pool_ = NULL;
  // Frontier tracker assigned to the test device through the device group.
  iree_async_frontier_tracker_t* frontier_tracker_ = NULL;
  // HIP driver configured to record command buffers into HIP graphs.
  iree_hal_driver_t* driver_ = NULL;
  // Single-device topology group that assigns queue frontier state.
  iree_hal_device_group_t* device_group_ = NULL;
  // Default HIP device created from |driver_|.
  iree_hal_device_t* device_ = NULL;
};

TEST_F(HipGraphCommandBufferTest,
       RecordsAndRepeatedlyExecutesMoreThanInitialNodeCapacity) {
  constexpr uint32_t kFillNodeCount = 129;
  constexpr uint64_t kExecutionCount = 16;
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

  iree_hal_buffer_ref_t target_ref =
      iree_hal_make_buffer_ref(buffer, 0, sizeof(uint32_t));
  for (uint32_t i = 0; i < kFillNodeCount && iree_status_is_ok(status); ++i) {
    target_ref.offset = i * sizeof(uint32_t);
    status = iree_hal_command_buffer_fill_buffer(
        command_buffer, target_ref, &i, sizeof(i), IREE_HAL_FILL_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }

  iree_hal_semaphore_t* semaphore = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore);
  }
  for (uint64_t i = 1; i <= kExecutionCount && iree_status_is_ok(status); ++i) {
    iree_hal_semaphore_list_t signal_semaphores = {
        /*.count=*/1,
        /*.semaphores=*/&semaphore,
        /*.payload_values=*/&i,
    };
    status = iree_hal_device_queue_execute(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_semaphores, command_buffer,
        iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_wait(semaphore, i, iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE);
    }
  }

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);

  IREE_EXPECT_OK(status);
}

}  // namespace
}  // namespace iree::hal::hip
