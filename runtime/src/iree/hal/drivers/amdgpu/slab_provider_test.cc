// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

class SlabProviderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t SlabProviderTest::host_allocator_;
iree_hal_amdgpu_libhsa_t SlabProviderTest::libhsa_;
iree_hal_amdgpu_topology_t SlabProviderTest::topology_;

class TestLogicalDevice {
 public:
  ~TestLogicalDevice() {
    iree_hal_device_release(base_device);
    iree_hal_device_group_release(device_group);
  }

  iree_status_t Initialize(
      const iree_hal_amdgpu_logical_device_options_t* options,
      const iree_hal_amdgpu_libhsa_t* libhsa,
      const iree_hal_amdgpu_topology_t* topology,
      iree_allocator_t host_allocator) {
    IREE_RETURN_IF_ERROR(create_context.Initialize(host_allocator));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), options, libhsa, topology, create_context.params(),
        host_allocator, &base_device));
    return iree_hal_device_group_create_from_device(
        base_device, create_context.frontier_tracker(), host_allocator,
        &device_group);
  }

  iree_hal_amdgpu_logical_device_t* device() const {
    return (iree_hal_amdgpu_logical_device_t*)base_device;
  }

  iree_hal_device_t* hal_device() const { return base_device; }

 private:
  // Creation context supplying the proactor pool and frontier tracker.
  iree::hal::cts::DeviceCreateContext create_context;

  // Test-owned device reference released before the topology-owning group.
  iree_hal_device_t* base_device = NULL;

  // Device group that owns the topology assigned to |base_device|.
  iree_hal_device_group_t* device_group = NULL;
};

TEST_F(SlabProviderTest,
       DefaultPhysicalDevicePoolMaterializesDeviceLocalBuffer) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  auto* device = test_device.device();
  ASSERT_GE(device->physical_device_count, 1u);
  iree_hal_pool_t* default_pool = device->physical_devices[0]->default_pool;
  ASSERT_NE(default_pool, nullptr);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      default_pool, params, /*allocation_size=*/128,
      /*requester_frontier=*/NULL, iree_make_timeout_ms(0), &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_GE(iree_hal_buffer_allocation_size(buffer), 128u);
  EXPECT_GE(iree_hal_buffer_byte_length(buffer), 128u);
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  EXPECT_FALSE(iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                 IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));

  iree_hal_buffer_release(buffer);
}

TEST_F(SlabProviderTest, DefaultQueueAllocaRoutesOversizedRequests) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.default_pool.range_length = 4096;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_semaphore_t* signal_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      test_device.hal_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* alloca_signal_semaphores[] = {signal_semaphore};
  uint64_t alloca_signal_values[] = {1};
  iree_hal_semaphore_list_t alloca_signal_list = {
      IREE_ARRAYSIZE(alloca_signal_semaphores),
      alloca_signal_semaphores,
      alloca_signal_values,
  };

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      test_device.hal_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), alloca_signal_list, /*pool=*/NULL,
      params, /*allocation_size=*/8192, IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      alloca_signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_t* dealloca_wait_semaphores[] = {signal_semaphore};
  uint64_t dealloca_wait_values[] = {1};
  iree_hal_semaphore_list_t dealloca_wait_list = {
      IREE_ARRAYSIZE(dealloca_wait_semaphores),
      dealloca_wait_semaphores,
      dealloca_wait_values,
  };
  iree_hal_semaphore_t* dealloca_signal_semaphores[] = {signal_semaphore};
  uint64_t dealloca_signal_values[] = {2};
  iree_hal_semaphore_list_t dealloca_signal_list = {
      IREE_ARRAYSIZE(dealloca_signal_semaphores),
      dealloca_signal_semaphores,
      dealloca_signal_values,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      test_device.hal_device(), IREE_HAL_QUEUE_AFFINITY_ANY, dealloca_wait_list,
      dealloca_signal_list, buffer, IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(dealloca_signal_list,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(buffer);
  iree_hal_semaphore_release(signal_semaphore);
}

TEST_F(SlabProviderTest, DefaultPhysicalDevicePoolGrowsAdditionalSlabs) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.default_pool.range_length = 4096;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  auto* device = test_device.device();
  ASSERT_GE(device->physical_device_count, 1u);
  iree_hal_pool_t* default_pool = device->physical_devices[0]->default_pool;
  ASSERT_NE(default_pool, nullptr);

  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(default_pool, &capabilities);
  ASSERT_NE(capabilities.max_allocation_size, 0u);

  iree_hal_pool_reservation_t first_reservation = {0};
  iree_hal_pool_acquire_info_t first_info = {0};
  iree_hal_pool_acquire_result_t first_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  iree::Status first_status(iree_hal_pool_acquire_reservation(
      default_pool, capabilities.max_allocation_size,
      capabilities.min_allocation_size, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &first_reservation, &first_info,
      &first_result));
  const bool first_acquired =
      first_status.ok() && (first_result == IREE_HAL_POOL_ACQUIRE_OK ||
                            first_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  iree_hal_pool_reservation_t second_reservation = {0};
  iree_hal_pool_acquire_info_t second_info = {0};
  iree_hal_pool_acquire_result_t second_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  iree::Status second_status(iree_hal_pool_acquire_reservation(
      default_pool, capabilities.max_allocation_size,
      capabilities.min_allocation_size, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &second_reservation, &second_info,
      &second_result));
  const bool second_acquired =
      second_status.ok() && (second_result == IREE_HAL_POOL_ACQUIRE_OK ||
                             second_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(default_pool, &stats);

  if (second_acquired) {
    iree_hal_pool_release_reservation(default_pool, &second_reservation,
                                      /*death_frontier=*/NULL);
  }
  if (first_acquired) {
    iree_hal_pool_release_reservation(default_pool, &first_reservation,
                                      /*death_frontier=*/NULL);
  }

  EXPECT_TRUE(first_status.ok()) << first_status.ToString();
  EXPECT_EQ(first_result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_TRUE(second_status.ok()) << second_status.ToString();
  EXPECT_EQ(second_result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(stats.reservation_count, 2u);
  EXPECT_GE(stats.slab_count, 2u);
}

}  // namespace
}  // namespace iree::hal::amdgpu
