// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/signal_pool.h"

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

//===----------------------------------------------------------------------===//
// Test fixture with HSA initialization
//===----------------------------------------------------------------------===//

struct SignalPoolTest : public ::testing::Test {
  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_hal_amdgpu_topology_t topology;

  static void SetUpTestSuite() {
    IREE_TRACE_SCOPE();
    host_allocator = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
    if (topology.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    IREE_TRACE_SCOPE();
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }
};
iree_allocator_t SignalPoolTest::host_allocator;
iree_hal_amdgpu_libhsa_t SignalPoolTest::libhsa;
iree_hal_amdgpu_topology_t SignalPoolTest::topology;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_signal_pool_t
//===----------------------------------------------------------------------===//

TEST_F(SignalPoolTest, HostPoolLifetimeEmpty) {
  IREE_TRACE_SCOPE();
  iree_hal_amdgpu_host_signal_pool_t pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_signal_pool_initialize(
      &libhsa, /*initial_capacity=*/0, /*batch_size=*/4, host_allocator,
      &pool));
  iree_hal_amdgpu_host_signal_pool_deinitialize(&pool);
}

TEST_F(SignalPoolTest, HostPoolLifetimePreallocated) {
  IREE_TRACE_SCOPE();
  iree_hal_amdgpu_host_signal_pool_t pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_signal_pool_initialize(
      &libhsa, /*initial_capacity=*/16, /*batch_size=*/8, host_allocator,
      &pool));
  iree_hal_amdgpu_host_signal_pool_deinitialize(&pool);
}

TEST_F(SignalPoolTest, HostPoolAcquireRelease) {
  IREE_TRACE_SCOPE();
  iree_hal_amdgpu_host_signal_pool_t pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_signal_pool_initialize(
      &libhsa, /*initial_capacity=*/4, /*batch_size=*/4, host_allocator,
      &pool));

  // Acquire a signal and verify it has the requested initial value.
  hsa_signal_t signal = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_host_signal_pool_acquire(
      &pool, /*initial_value=*/42, &signal));
  EXPECT_NE(signal.handle, 0u);
  hsa_signal_value_t value =
      iree_hsa_signal_load_relaxed(IREE_LIBHSA(&libhsa), signal);
  EXPECT_EQ(value, 42);

  // Release and re-acquire — should get a recycled signal.
  iree_hal_amdgpu_host_signal_pool_release(&pool, signal);
  hsa_signal_t signal2 = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_host_signal_pool_acquire(
      &pool, /*initial_value=*/99, &signal2));
  EXPECT_NE(signal2.handle, 0u);
  // LIFO: should get the same signal back.
  EXPECT_EQ(signal2.handle, signal.handle);
  value = iree_hsa_signal_load_relaxed(IREE_LIBHSA(&libhsa), signal2);
  EXPECT_EQ(value, 99);

  iree_hal_amdgpu_host_signal_pool_release(&pool, signal2);
  iree_hal_amdgpu_host_signal_pool_deinitialize(&pool);
}

TEST_F(SignalPoolTest, HostPoolBatchGrowth) {
  IREE_TRACE_SCOPE();
  iree_hal_amdgpu_host_signal_pool_t pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_signal_pool_initialize(
      &libhsa, /*initial_capacity=*/0, /*batch_size=*/4, host_allocator,
      &pool));

  // Acquire more signals than one batch — forces growth.
  hsa_signal_t signals[10];
  for (int i = 0; i < 10; ++i) {
    IREE_ASSERT_OK(iree_hal_amdgpu_host_signal_pool_acquire(
        &pool, /*initial_value=*/i, &signals[i]));
    EXPECT_NE(signals[i].handle, 0u);
  }

  // All should be unique handles.
  for (int i = 0; i < 10; ++i) {
    for (int j = i + 1; j < 10; ++j) {
      EXPECT_NE(signals[i].handle, signals[j].handle)
          << "signals[" << i << "] == signals[" << j << "]";
    }
  }

  // Release all — capacity is always >= allocated_count.
  for (int i = 0; i < 10; ++i) {
    iree_hal_amdgpu_host_signal_pool_release(&pool, signals[i]);
  }

  iree_hal_amdgpu_host_signal_pool_deinitialize(&pool);
}

TEST_F(SignalPoolTest, HostPoolAllOutstandingThenRelease) {
  IREE_TRACE_SCOPE();
  iree_hal_amdgpu_host_signal_pool_t pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_signal_pool_initialize(
      &libhsa, /*initial_capacity=*/8, /*batch_size=*/8, host_allocator,
      &pool));

  // Acquire all 8 pre-created signals — pool is now empty.
  hsa_signal_t signals[8];
  for (int i = 0; i < 8; ++i) {
    IREE_ASSERT_OK(iree_hal_amdgpu_host_signal_pool_acquire(
        &pool, /*initial_value=*/0, &signals[i]));
  }

  // Release all 8 back. This exercises the case where free_count grows back
  // to allocated_count — the free list must have been sized to accommodate
  // this.
  for (int i = 0; i < 8; ++i) {
    iree_hal_amdgpu_host_signal_pool_release(&pool, signals[i]);
  }

  iree_hal_amdgpu_host_signal_pool_deinitialize(&pool);
}

}  // namespace
}  // namespace iree::hal::amdgpu
