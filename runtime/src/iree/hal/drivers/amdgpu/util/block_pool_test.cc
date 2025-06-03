// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/block_pool.h"

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct BlockPoolTest : public ::testing::Test {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_amdgpu_libhsa_t libhsa;
  iree_hal_amdgpu_topology_t topology;

  void SetUp() override {
    IREE_TRACE_SCOPE();
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
  }

  void TearDown() override {
    IREE_TRACE_SCOPE();
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }
};

TEST_F(BlockPoolTest, LifetimeEmpty) {
  IREE_TRACE_SCOPE();

  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, gpu_agent, &memory_pool));

  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, gpu_agent, memory_pool, host_allocator, &block_pool));

  // Acquire a block. This will grow the pool as we started empty.
  iree_hal_amdgpu_block_t* block0 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_acquire(&block_pool, &block0));
  EXPECT_NE(block0->ptr, nullptr);
  EXPECT_EQ(block0->next, nullptr);

  // Acquire another block. It should have a unique address (to ensure we didn't
  // return the same block).
  iree_hal_amdgpu_block_t* block1 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_acquire(&block_pool, &block1));
  EXPECT_NE(block1->ptr, nullptr);
  EXPECT_NE(block1->ptr, block0->ptr);
  EXPECT_EQ(block1->next, nullptr);

  // Release the first block back to the pool followed by the second.
  // This ensures we support arbitrary ordering.
  iree_hal_amdgpu_block_pool_release(&block_pool, block0);
  iree_hal_amdgpu_block_pool_release(&block_pool, block1);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

TEST_F(BlockPoolTest, LifetimeInitial) {
  IREE_TRACE_SCOPE();

  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, gpu_agent, &memory_pool));

  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.initial_capacity=*/32,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, gpu_agent, memory_pool, host_allocator, &block_pool));

  // Acquire and release block. This should not grow the pool as we initialized
  // it above.
  iree_hal_amdgpu_block_t* block0 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_acquire(&block_pool, &block0));
  EXPECT_NE(block0->ptr, nullptr);
  EXPECT_EQ(block0->next, nullptr);
  iree_hal_amdgpu_block_pool_release(&block_pool, block0);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

// Allocates a few blocks to force some growth, frees some, trims, and tries to
// grow again. We have to use the reported blocks_per_allocation as the pool
// will always round up what we specify.
TEST_F(BlockPoolTest, Trimming) {
  IREE_TRACE_SCOPE();

  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, gpu_agent, &memory_pool));

  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/256 * 1024,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, gpu_agent, memory_pool, host_allocator, &block_pool));

  // Since the exact counts are device dependent we have to dynamically manage
  // our working set. We try to hit a certain number of batches.
  // Note that to ensure we get new blocks we allocate everything and then
  // selectively release resources.
  //
  // batches[0] = fully allocated
  // batches[1] = all but one allocated
  // batches[2] = only one allocated
  struct batch_t {
    std::vector<iree_hal_amdgpu_block_t*> blocks;
  };
  batch_t batches[4] = {};
  for (iree_host_size_t batch = 0; batch < IREE_ARRAYSIZE(batches); ++batch) {
    for (iree_host_size_t i = 0; i < block_pool.blocks_per_allocation; ++i) {
      iree_hal_amdgpu_block_t* block = NULL;
      IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_acquire(&block_pool, &block));
      batches[batch].blocks.push_back(block);
    }
  }
  {
    // batches[0] = fully allocated, don't release anything.
  }
  {
    // batches[1] = all but one allocated - release only the last.
    iree_hal_amdgpu_block_t* block = batches[1].blocks.back();
    batches[1].blocks.pop_back();
    iree_hal_amdgpu_block_pool_release(&block_pool, block);
  }
  {
    // batches[2] = only one allocated - release all but the first.
    for (iree_host_size_t i = 0; i < batches[2].blocks.size() - 1; ++i) {
      iree_hal_amdgpu_block_t* block = batches[2].blocks.back();
      batches[2].blocks.pop_back();
      iree_hal_amdgpu_block_pool_release(&block_pool, block);
    }
  }
  {
    // batches[3] = none allocated - release all.
    while (!batches[3].blocks.empty()) {
      iree_hal_amdgpu_block_t* block = batches[3].blocks.back();
      batches[3].blocks.pop_back();
      iree_hal_amdgpu_block_pool_release(&block_pool, block);
    }
  }

  // Trim now - we should only drop one allocation for batches[3] that has
  // no live blocks.
  iree_hal_amdgpu_block_pool_trim(&block_pool);

  // Acquire a new block - should use something we have in the pool. This is
  // something that would trigger ASAN if we freed something bad.
  iree_hal_amdgpu_block_t* block0 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_acquire(&block_pool, &block0));
  EXPECT_NE(block0->ptr, nullptr);
  EXPECT_EQ(block0->next, nullptr);
  iree_hal_amdgpu_block_pool_release(&block_pool, block0);

  // Drop all of batches[2] and try trimming again.
  for (auto* block : batches[2].blocks) {
    iree_hal_amdgpu_block_pool_release(&block_pool, block);
  }
  batches[2].blocks.clear();
  iree_hal_amdgpu_block_pool_trim(&block_pool);

  // Another test block.
  iree_hal_amdgpu_block_t* block1 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_acquire(&block_pool, &block1));
  EXPECT_NE(block1->ptr, nullptr);
  EXPECT_EQ(block1->next, nullptr);
  iree_hal_amdgpu_block_pool_release(&block_pool, block1);

  // Release all remaining blocks.
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(batches); ++i) {
    for (auto* block : batches[i].blocks) {
      iree_hal_amdgpu_block_pool_release(&block_pool, block);
    }
  }

  // Implicitly trims.
  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace iree::hal::amdgpu
