// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::metal {
namespace {

TEST(MetalDeviceTest, StandaloneDeviceQueueAllocaDoesNotCrash) {
  iree_hal_metal_device_params_t device_params;
  iree_hal_metal_device_params_initialize(&device_params);

  iree_hal_driver_t* driver = nullptr;
  IREE_ASSERT_OK(iree_hal_metal_driver_create(
      IREE_SV("metal"), &device_params, iree_allocator_system(), &driver));

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_default_device(
      driver, &create_params, iree_allocator_system(), &device);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_async_proactor_pool_release(proactor_pool);
    iree_hal_driver_release(driver);
    GTEST_SKIP() << "No Metal device available";
  }

  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), /*pool=*/nullptr, buffer_params,
      /*allocation_size=*/1024, IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);

  iree_hal_buffer_release(buffer);
  iree_hal_device_release(device);
  iree_async_proactor_pool_release(proactor_pool);
  iree_hal_driver_release(driver);
}

}  // namespace
}  // namespace iree::hal::metal
