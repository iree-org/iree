// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/buffer_pool.h"

#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/buffer.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

struct BufferPoolTest : public ::testing::Test {
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
iree_allocator_t BufferPoolTest::host_allocator;
iree_hal_amdgpu_libhsa_t BufferPoolTest::libhsa;
iree_hal_amdgpu_topology_t BufferPoolTest::topology;
hsa_amd_memory_pool_t BufferPoolTest::cpu_memory_pool;

// Tests that a pool can be initialized/deinitialized successfully.
// Note that pools do not allocate anything on initialization so this should
// never allocate.
TEST_F(BufferPoolTest, Lifetime) {
  IREE_TRACE_SCOPE();

  iree_hal_buffer_placement_t placement = {
      /*.device=*/NULL,  // not available in test
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.flags=*/IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
  };
  iree_hal_amdgpu_buffer_pool_t buffer_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_pool_initialize(
      &libhsa, &topology, placement,
      IREE_HAL_AMDGPU_BUFFER_POOL_DEFAULT_BLOCK_CAPACITY, host_allocator,
      cpu_memory_pool, &buffer_pool));

  // No-op since nothing has been allocated.
  iree_hal_amdgpu_buffer_pool_trim(&buffer_pool);

  iree_hal_amdgpu_buffer_pool_deinitialize(&buffer_pool);
}

// Tests a pool that has preallocation requests.
// We make a few requests interleaved with trims and then rely on
// deinitialization to free the remaining resources to ensure there are no
// leaks.
TEST_F(BufferPoolTest, LifetimePreallocate) {
  IREE_TRACE_SCOPE();

  iree_hal_buffer_placement_t placement = {
      /*.device=*/NULL,  // not available in test
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.flags=*/IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
  };
  iree_hal_amdgpu_buffer_pool_t buffer_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_pool_initialize(
      &libhsa, &topology, placement,
      /*block_capacity=*/32, host_allocator, cpu_memory_pool, &buffer_pool));

  // No-op since nothing has been allocated yet.
  iree_hal_amdgpu_buffer_pool_trim(&buffer_pool);

  // No-op preallocation (can happen if we blindly pass options/flags of 0).
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_pool_preallocate(&buffer_pool, 0));

  // Preallocate one block.
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_pool_preallocate(
      &buffer_pool, buffer_pool.block_capacity));

  // Trim the entire block (nothing is used).
  iree_hal_amdgpu_buffer_pool_trim(&buffer_pool);

  // Preallocate two blocks.
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_pool_preallocate(
      &buffer_pool, buffer_pool.block_capacity + 1));

  // Preallocate one more block (1 buffer ceildiv capacity = 1 block).
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_pool_preallocate(&buffer_pool, 1));

  // Deinitialize with remaining preallocated blocks to test cleanup.
  iree_hal_amdgpu_buffer_pool_deinitialize(&buffer_pool);
}

// Tests acquiring and releasing a buffer handle from the pool.
TEST_F(BufferPoolTest, AcquireRelease) {
  IREE_TRACE_SCOPE();

  iree_hal_buffer_placement_t placement = {
      /*.device=*/(iree_hal_device_t*)0xF00Du,  // not available in test
      /*.queue_affinity=*/1ull,
      /*.flags=*/IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
  };
  iree_hal_amdgpu_buffer_pool_t buffer_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_pool_initialize(
      &libhsa, &topology, placement,
      /*block_capacity=*/32, host_allocator, cpu_memory_pool, &buffer_pool));

  iree_hal_buffer_params_t buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT |
          IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      /*.queue_affinity=*/placement.queue_affinity,
      /*.min_alignment=*/0,
  };
  iree_device_size_t requested_size = 127u;

  // Handle is just for convenience and is stored within the buffer as well.
  iree_hal_buffer_t* buffer = NULL;
  iree_hal_amdgpu_device_allocation_handle_t* handle = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_pool_acquire(
      &buffer_pool, buffer_params, /*allocation_size=*/requested_size, &buffer,
      &handle));
  ASSERT_NE(buffer, nullptr);
  ASSERT_NE(handle, nullptr);
  EXPECT_GE(iree_hal_buffer_allocation_size(buffer), requested_size);
  EXPECT_EQ(iree_hal_buffer_allocation_placement(buffer).device,
            placement.device);
  EXPECT_EQ(iree_hal_buffer_allocation_placement(buffer).queue_affinity,
            placement.queue_affinity);
  EXPECT_EQ(iree_hal_buffer_allocation_placement(buffer).flags,
            placement.flags);
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), requested_size);

  // Handle should have no physical pointer since nothing has been allocated.
  EXPECT_EQ(handle->ptr, nullptr);

  // Ensure the buffer resolves to the handle.
  iree_hal_amdgpu_device_buffer_type_t type = 0;
  uint64_t bits = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_resolve_buffer(buffer, &type, &bits));
  EXPECT_EQ(type, IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE);
  EXPECT_EQ(bits, (uint64_t)handle);

  // Same as above but for when we're confident we're dealing with a transient
  // buffer (here we are).
  iree_hal_amdgpu_device_allocation_handle_t* queried_handle = NULL;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_resolve_transient_buffer(buffer, &queried_handle));
  EXPECT_EQ(handle, queried_handle);

  // Since the buffer is not actually allocated any attempt to map (even though
  // we requested it) should fail.
  iree_hal_buffer_mapping_t mapping = {};
  EXPECT_THAT(
      Status(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
                                       IREE_HAL_MEMORY_ACCESS_READ, 0,
                                       IREE_HAL_WHOLE_BUFFER, &mapping)),
      StatusIs(StatusCode::kFailedPrecondition));

  // Release the buffer back to the pool - we're the last reference and it
  // should be recycled.
  iree_hal_buffer_release(buffer);

  iree_hal_amdgpu_buffer_pool_deinitialize(&buffer_pool);
}

// Explicitly tests pool growth by acquiring an entire block worth of buffers+1.
// We then release all the buffers that should have been in the first block and
// trim with the second block outstanding to ensure it is not reclaimed with the
// buffer outstanding.
TEST_F(BufferPoolTest, Growth) {
  IREE_TRACE_SCOPE();

  iree_hal_buffer_placement_t placement = {
      /*.device=*/NULL,  // not available in test
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.flags=*/IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
  };
  iree_hal_amdgpu_buffer_pool_t buffer_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_pool_initialize(
      &libhsa, &topology, placement, /*block_capacity=*/32, host_allocator,
      cpu_memory_pool, &buffer_pool));
  // NOTE: the capacity may be larger than requested due to alignment.
  const iree_host_size_t block_capacity = buffer_pool.block_capacity;

  iree_hal_buffer_params_t buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE,
      /*.queue_affinity=*/placement.queue_affinity,
      /*.min_alignment=*/0,
  };
  iree_device_size_t requested_size = 128u;
  std::vector<iree_hal_buffer_t*> buffers(block_capacity);
  std::vector<iree_hal_amdgpu_device_allocation_handle_t*> handles(
      block_capacity);

  // Preallocate the first block (just to put more load on that path).
  IREE_ASSERT_OK(
      iree_hal_amdgpu_buffer_pool_preallocate(&buffer_pool, block_capacity));

  // Allocate enough to consume the entire first block.
  for (iree_host_size_t i = 0; i < block_capacity; ++i) {
    IREE_ASSERT_OK(iree_hal_amdgpu_buffer_pool_acquire(
        &buffer_pool, buffer_params, /*allocation_size=*/requested_size,
        &buffers[i], &handles[i]));
    EXPECT_EQ(handles[i]->ptr, nullptr);
  }

  // Allocate +1 to trigger growth and acquire the next block.
  iree_hal_buffer_t* growth_buffer = NULL;
  iree_hal_amdgpu_device_allocation_handle_t* growth_handle = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_pool_acquire(
      &buffer_pool, buffer_params, /*allocation_size=*/requested_size,
      &growth_buffer, &growth_handle));
  ASSERT_NE(growth_buffer, nullptr);
  ASSERT_NE(growth_handle, nullptr);
  EXPECT_EQ(growth_handle->ptr, nullptr);

  // Recycle all the buffers from the first block. After this it should have no
  // outstanding buffers allocated it from it and be a candidate for trimming.
  for (iree_host_size_t i = 0; i < block_capacity; ++i) {
    iree_hal_buffer_release(buffers[i]);
  }

  // Ensure the growth buffer is still valid (should be, as we shouldn't have
  // deallocated anything).
  EXPECT_EQ(growth_handle->ptr, nullptr);

  // Trim to drop the unused first block.
  iree_hal_amdgpu_buffer_pool_trim(&buffer_pool);

  // Check that we didn't drop the growth buffer that's in the second block.
  EXPECT_EQ(growth_handle->ptr, nullptr);

  // Release the last buffer and let the deinitialize cleanup the second block.
  iree_hal_buffer_release(growth_buffer);

  iree_hal_amdgpu_buffer_pool_deinitialize(&buffer_pool);
}

}  // namespace
}  // namespace iree::hal::amdgpu
