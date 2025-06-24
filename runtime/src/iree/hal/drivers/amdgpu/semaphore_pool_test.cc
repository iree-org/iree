// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/semaphore_pool.h"

#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/device/semaphore.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

struct SemaphorePoolTest : public ::testing::Test {
  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_hal_amdgpu_topology_t topology;
  static hsa_amd_memory_pool_t cpu_memory_pool;

  static void SetUpTestSuite() {
    IREE_TRACE_SCOPE();
    host_allocator = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_ignore(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
    if (topology.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }

    hsa_agent_t cpu_agent = topology.cpu_agents[0];
    IREE_ASSERT_OK(iree_hal_amdgpu_find_fine_global_memory_pool(
        &libhsa, cpu_agent, &cpu_memory_pool));
  }

  static void TearDownTestSuite() {
    IREE_TRACE_SCOPE();
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }
};
iree_allocator_t SemaphorePoolTest::host_allocator;
iree_hal_amdgpu_libhsa_t SemaphorePoolTest::libhsa;
iree_hal_amdgpu_topology_t SemaphorePoolTest::topology;
hsa_amd_memory_pool_t SemaphorePoolTest::cpu_memory_pool;

// Tests that a pool can be initialized/deinitialized successfully.
// Note that pools do not allocate anything on initialization so this should
// never allocate.
TEST_F(SemaphorePoolTest, Lifetime) {
  IREE_TRACE_SCOPE();

  iree_hal_amdgpu_semaphore_options_t options = {0};
  iree_hal_amdgpu_semaphore_pool_t semaphore_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_pool_initialize(
      &libhsa, &topology, IREE_HAL_AMDGPU_SEMAPHORE_POOL_DEFAULT_BLOCK_CAPACITY,
      options, IREE_HAL_SEMAPHORE_FLAG_NONE, host_allocator, cpu_memory_pool,
      &semaphore_pool));

  // No-op since nothing has been allocated.
  iree_hal_amdgpu_semaphore_pool_trim(&semaphore_pool);

  iree_hal_amdgpu_semaphore_pool_deinitialize(&semaphore_pool);
}

// Tests a pool that has preallocation requests.
// We make a few requests interleaved with trims and then rely on
// deinitialization to free the remaining resources to ensure there are no
// leaks.
TEST_F(SemaphorePoolTest, LifetimePreallocate) {
  IREE_TRACE_SCOPE();

  iree_hal_amdgpu_semaphore_options_t options = {0};
  iree_hal_amdgpu_semaphore_pool_t semaphore_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_pool_initialize(
      &libhsa, &topology,
      /*block_capacity=*/32, options, IREE_HAL_SEMAPHORE_FLAG_NONE,
      host_allocator, cpu_memory_pool, &semaphore_pool));

  // No-op since nothing has been allocated yet.
  iree_hal_amdgpu_semaphore_pool_trim(&semaphore_pool);

  // No-op preallocation (can happen if we blindly pass options/flags of 0).
  IREE_ASSERT_OK(
      iree_hal_amdgpu_semaphore_pool_preallocate(&semaphore_pool, 0));

  // Preallocate one block.
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_pool_preallocate(
      &semaphore_pool, semaphore_pool.block_capacity));

  // Trim the entire block (nothing is used).
  iree_hal_amdgpu_semaphore_pool_trim(&semaphore_pool);

  // Preallocate two blocks.
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_pool_preallocate(
      &semaphore_pool, semaphore_pool.block_capacity + 1));

  // Preallocate one more block (1 buffer ceildiv capacity = 1 block).
  IREE_ASSERT_OK(
      iree_hal_amdgpu_semaphore_pool_preallocate(&semaphore_pool, 1));

  // Deinitialize with remaining preallocated blocks to test cleanup.
  iree_hal_amdgpu_semaphore_pool_deinitialize(&semaphore_pool);
}

// Tests acquiring and releasing a buffer handle from the pool.
TEST_F(SemaphorePoolTest, AcquireRelease) {
  IREE_TRACE_SCOPE();

  iree_hal_amdgpu_semaphore_options_t options = {0};
  iree_hal_amdgpu_semaphore_pool_t semaphore_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_pool_initialize(
      &libhsa, &topology,
      /*block_capacity=*/32, options, IREE_HAL_SEMAPHORE_FLAG_NONE,
      host_allocator, cpu_memory_pool, &semaphore_pool));

  // Acquire a semaphore.
  const uint64_t initial_value = 123ull;
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_pool_acquire(
      &semaphore_pool, initial_value, IREE_HAL_SEMAPHORE_FLAG_NONE,
      &semaphore));
  ASSERT_NE(semaphore, nullptr);

  // Ensure it reports the initial value that was specified.
  uint64_t reported_value = 0ull;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &reported_value));
  EXPECT_EQ(reported_value, initial_value);

  // Ensure the device-visible handle is initialized.
  iree_hal_amdgpu_device_semaphore_t* handle = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_handle(semaphore, &handle));
  ASSERT_NE(semaphore, nullptr);
  ASSERT_EQ(handle->host_semaphore, (uint64_t)semaphore);

  // Release the semaphore back to the pool - we're the last reference and it
  // should be recycled.
  iree_hal_semaphore_release(semaphore);

  iree_hal_amdgpu_semaphore_pool_deinitialize(&semaphore_pool);
}

// Explicitly tests pool growth by acquiring an entire block worth of
// semaphores+1. We then release all the semaphores that should have been in the
// first block and trim with the second block outstanding to ensure it is not
// reclaimed with the buffer outstanding.
TEST_F(SemaphorePoolTest, Growth) {
  IREE_TRACE_SCOPE();

  iree_hal_amdgpu_semaphore_options_t options = {0};
  iree_hal_amdgpu_semaphore_pool_t semaphore_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_pool_initialize(
      &libhsa, &topology, /*block_capacity=*/32, options,
      IREE_HAL_SEMAPHORE_FLAG_NONE, host_allocator, cpu_memory_pool,
      &semaphore_pool));
  // NOTE: the capacity may be larger than requested due to alignment.
  const iree_host_size_t block_capacity = semaphore_pool.block_capacity;

  std::vector<iree_hal_semaphore_t*> semaphores(block_capacity);

  // Preallocate the first block (just to put more load on that path).
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_pool_preallocate(&semaphore_pool,
                                                            block_capacity));

  // Allocate enough to consume the entire first block.
  for (iree_host_size_t i = 0; i < block_capacity; ++i) {
    IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_pool_acquire(
        &semaphore_pool, /*initial_value=*/0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
        &semaphores[i]));
    ASSERT_NE(semaphores[i], nullptr);
  }

  // Allocate +1 to trigger growth and acquire the next block.
  iree_hal_semaphore_t* growth_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_pool_acquire(
      &semaphore_pool, /*initial_value=*/0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
      &growth_semaphore));
  ASSERT_NE(growth_semaphore, nullptr);

  // Recycle all the semaphores from the first block. After this it should have
  // no outstanding semaphores allocated it from it and be a candidate for
  // trimming.
  for (iree_host_size_t i = 0; i < block_capacity; ++i) {
    iree_hal_semaphore_release(semaphores[i]);
  }

  // Trim to drop the unused first block.
  iree_hal_amdgpu_semaphore_pool_trim(&semaphore_pool);

  // Release the last semaphore and let the deinitialize cleanup the second
  // block.
  iree_hal_semaphore_release(growth_semaphore);

  iree_hal_amdgpu_semaphore_pool_deinitialize(&semaphore_pool);
}

}  // namespace
}  // namespace iree::hal::amdgpu
