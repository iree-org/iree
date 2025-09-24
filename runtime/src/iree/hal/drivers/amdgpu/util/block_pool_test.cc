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

using iree::testing::status::StatusIs;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_block_pool_t
//===----------------------------------------------------------------------===//

struct BlockPoolTest : public ::testing::Test {
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
      iree_status_ignore(status);
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
iree_allocator_t BlockPoolTest::host_allocator;
iree_hal_amdgpu_libhsa_t BlockPoolTest::libhsa;
iree_hal_amdgpu_topology_t BlockPoolTest::topology;

TEST_F(BlockPoolTest, LifetimeEmpty) {
  IREE_TRACE_SCOPE();

  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, gpu_agent, &memory_pool));

  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
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
      /*.min_blocks_per_allocation=*/1,
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

TEST_F(BlockPoolTest, BlockSizeCheck) {
  IREE_TRACE_SCOPE();

  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, gpu_agent, &memory_pool));

  iree_hal_amdgpu_block_pool_t block_pool = {0};

  // Fail if block size is not a power-of-two.
  iree_hal_amdgpu_block_pool_options_t non_pot_options = {
      /*.block_size=*/1 * 1024 + 1,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  EXPECT_THAT(Status(iree_hal_amdgpu_block_pool_initialize(
                  &libhsa, non_pot_options, gpu_agent, memory_pool,
                  host_allocator, &block_pool)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(BlockPoolTest, AutoBlocksPerAllocation) {
  IREE_TRACE_SCOPE();

  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, gpu_agent, &memory_pool));

  // Query allocation granularity from the pool for comparison.
  size_t alloc_rec_granule = 0;
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(&libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, &alloc_rec_granule));

  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/0,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, gpu_agent, memory_pool, host_allocator, &block_pool));

  // Internals check during testing to verify the chosen blocks_per_allocation
  // exactly matches the recommended allocation granularity.
  EXPECT_EQ(block_pool.blocks_per_allocation * block_pool.block_size,
            alloc_rec_granule);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

TEST_F(BlockPoolTest, ReleaseList) {
  IREE_TRACE_SCOPE();

  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, gpu_agent, &memory_pool));

  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, gpu_agent, memory_pool, host_allocator, &block_pool));

  // Acquire blocks.
  iree_hal_amdgpu_block_t* block0 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_acquire(&block_pool, &block0));
  iree_hal_amdgpu_block_t* block1 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_acquire(&block_pool, &block1));
  iree_hal_amdgpu_block_t* block2 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_acquire(&block_pool, &block2));

  // Link together: block0 -> block1 -> block2
  block0->next = block1;
  block1->next = block2;
  block2->next = NULL;

  // Release the entire list.
  iree_hal_amdgpu_block_pool_release_list(&block_pool, block0);

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
      /*.min_blocks_per_allocation=*/1,
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

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_block_arena_t
//===----------------------------------------------------------------------===//

using BlockArenaTest = BlockPoolTest;

TEST_F(BlockArenaTest, LifetimeEmpty) {
  IREE_TRACE_SCOPE();

  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, gpu_agent, &memory_pool));
  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, gpu_agent, memory_pool, host_allocator, &block_pool));

  iree_hal_amdgpu_block_arena_t block_arena;
  iree_hal_amdgpu_block_arena_initialize(&block_pool, &block_arena);

  // No allocations made.

  iree_hal_amdgpu_block_arena_deinitialize(&block_arena);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

TEST_F(BlockArenaTest, Allocation) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));
  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, cpu_agent, memory_pool, host_allocator, &block_pool));

  iree_hal_amdgpu_block_arena_t block_arena;
  iree_hal_amdgpu_block_arena_initialize(&block_pool, &block_arena);

  // Allocate 1 byte; it should be aligned up but we don't have a way to check
  // in this test and only that its pointer is aligned. This initial allocation
  // should trigger a growth.
  uint8_t* ptr0 = NULL;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_block_arena_allocate(&block_arena, 1, (void**)&ptr0));
  EXPECT_TRUE(iree_host_size_has_alignment((iree_host_size_t)ptr0,
                                           iree_hal_amdgpu_max_align_t));
  ptr0[0] = 123;  // will crash if invalid

  // Allocate another byte. It should be aligned.
  uint8_t* ptr1 = NULL;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_block_arena_allocate(&block_arena, 1, (void**)&ptr1));
  EXPECT_TRUE(iree_host_size_has_alignment((iree_host_size_t)ptr1,
                                           iree_hal_amdgpu_max_align_t));
  ptr1[0] = 123;  // will crash if invalid

  // Fail to allocate too large of an allocation.
  uint8_t* oversize_ptr = NULL;
  EXPECT_THAT(Status(iree_hal_amdgpu_block_arena_allocate(
                  &block_arena, INT64_MAX, (void**)&oversize_ptr)),
              StatusIs(StatusCode::kInvalidArgument));

  // Allocate the largest possible allocation to force growth.
  const iree_device_size_t largest_size = block_pool.block_size;
  uint8_t* ptr2 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_arena_allocate(
      &block_arena, largest_size, (void**)&ptr2));
  EXPECT_TRUE(iree_host_size_has_alignment((iree_host_size_t)ptr2,
                                           iree_hal_amdgpu_max_align_t));
  for (iree_device_size_t i = 0; i < largest_size; ++i) {
    ptr2[i] = 123;  // will crash if invalid
  }

  iree_hal_amdgpu_block_arena_deinitialize(&block_arena);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

TEST_F(BlockArenaTest, Reset) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));
  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, cpu_agent, memory_pool, host_allocator, &block_pool));

  iree_hal_amdgpu_block_arena_t block_arena;
  iree_hal_amdgpu_block_arena_initialize(&block_pool, &block_arena);

  // Arena should be empty.
  EXPECT_EQ(block_arena.block_head, nullptr);
  EXPECT_EQ(block_arena.total_allocation_size, 0);

  // Allocate from the arena and check that it update its stats.
  // This is an implementation detail only valid in tests.
  // Note that the total size allocated will be larger than requested due to
  // alignment.
  uint8_t* ptr0 = NULL;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_block_arena_allocate(&block_arena, 1, (void**)&ptr0));
  EXPECT_NE(block_arena.block_head, nullptr);
  EXPECT_EQ(block_arena.block_head, block_arena.block_tail);  // 1 block
  EXPECT_GT(block_arena.total_allocation_size, 1);
  EXPECT_GT(block_arena.used_allocation_size, 1);
  EXPECT_GT(block_arena.block_bytes_remaining, 1);

  // Reset the arena. All blocks should be returned to the pool.
  iree_hal_amdgpu_block_arena_reset(&block_arena);

  // Ensure the internal state updated.
  EXPECT_EQ(block_arena.block_head, nullptr);
  EXPECT_EQ(block_arena.block_tail, nullptr);
  EXPECT_EQ(block_arena.total_allocation_size, 0);
  EXPECT_EQ(block_arena.used_allocation_size, 0);
  EXPECT_EQ(block_arena.block_bytes_remaining, 0);

  iree_hal_amdgpu_block_arena_deinitialize(&block_arena);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

TEST_F(BlockArenaTest, ReleaseBlocks) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));
  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, cpu_agent, memory_pool, host_allocator, &block_pool));

  iree_hal_amdgpu_block_arena_t block_arena;
  iree_hal_amdgpu_block_arena_initialize(&block_pool, &block_arena);

  // Allocate two full blocks from the arena to ensure that we test the linked
  // list behavior of the release.
  const iree_device_size_t largest_size = block_pool.block_size;
  uint8_t* ptr0 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_arena_allocate(
      &block_arena, largest_size, (void**)&ptr0));
  uint8_t* ptr1 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_arena_allocate(
      &block_arena, largest_size, (void**)&ptr1));
  uint8_t* ptr2 = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_arena_allocate(
      &block_arena, largest_size, (void**)&ptr2));

  // Release the blocks from the arena to the caller (us).
  iree_hal_amdgpu_block_t* block_head =
      iree_hal_amdgpu_block_arena_release_blocks(&block_arena);

  // Expect the arena to be empty. This is an implementation detail we are
  // poking at but it's fine for testing.
  EXPECT_EQ(block_arena.block_head, nullptr);

  // Expect 3 blocks in the list.
  iree_hal_amdgpu_block_t* block0 = block_head;
  EXPECT_NE(block0->next, nullptr);
  EXPECT_EQ(block0->prev, nullptr);
  iree_hal_amdgpu_block_t* block1 = block0->next;
  EXPECT_NE(block1->next, nullptr);
  EXPECT_EQ(block1->prev, block0);
  iree_hal_amdgpu_block_t* block2 = block1->next;
  EXPECT_EQ(block2->next, nullptr);
  EXPECT_EQ(block2->prev, block1);

  // Release all blocks back to the pool directly.
  iree_hal_amdgpu_block_pool_release_list(&block_pool, block_head);

  iree_hal_amdgpu_block_arena_deinitialize(&block_arena);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_block_allocator_t
//===----------------------------------------------------------------------===//

using BlockAllocatorTest = BlockPoolTest;

TEST_F(BlockAllocatorTest, LifetimeEmpty) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));
  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, cpu_agent, memory_pool, host_allocator, &block_pool));

  iree_hal_amdgpu_block_allocator_t allocator;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_initialize(
      &block_pool, /*min_page_size=*/64, &allocator));

  // No allocations made.

  iree_hal_amdgpu_block_allocator_deinitialize(&allocator);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

TEST_F(BlockAllocatorTest, PageSizeCheck) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));
  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, cpu_agent, memory_pool, host_allocator, &block_pool));

  iree_hal_amdgpu_block_allocator_t allocator;

  // Fail if the page size is not a power-of-two.
  EXPECT_THAT(Status(iree_hal_amdgpu_block_allocator_initialize(
                  &block_pool, /*min_page_size=*/33, &allocator)),
              StatusIs(StatusCode::kInvalidArgument));

  // Fail if page size is > block size of the backing pool.
  EXPECT_THAT(Status(iree_hal_amdgpu_block_allocator_initialize(
                  &block_pool, /*min_page_size=*/block_pool.block_size * 2,
                  &allocator)),
              StatusIs(StatusCode::kInvalidArgument));

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

TEST_F(BlockAllocatorTest, Allocate) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));
  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, cpu_agent, memory_pool, host_allocator, &block_pool));

  iree_hal_amdgpu_block_allocator_t allocator;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_initialize(
      &block_pool, /*min_page_size=*/64, &allocator));

  // Fail if the allocation page count exceeds the block page capacity.
  void* ptr = NULL;
  iree_hal_amdgpu_block_token_t token = {0};
  EXPECT_THAT(Status(iree_hal_amdgpu_block_allocator_allocate(
                  &allocator, UINT64_MAX, &ptr, &token)),
              StatusIs(StatusCode::kInvalidArgument));

  // Allow no-op frees of NULL pointers (here just testing for segfaults).
  iree_hal_amdgpu_block_allocator_free(&allocator, NULL, token);

  // Allocate from the empty pool (it should grow to satisfy the block request).
  IREE_ASSERT_OK(
      iree_hal_amdgpu_block_allocator_allocate(&allocator, 1, &ptr, &token));
  EXPECT_NE(ptr, nullptr);
  // NOTE: tokens are implementation details.
  EXPECT_EQ(token.page_count, 1);

  // Free to return the allocation to the pool.
  iree_hal_amdgpu_block_allocator_free(&allocator, ptr, token);

  // Trim the block pool - the allocator should have returned the block on the
  // free above.
  EXPECT_EQ(allocator.block_head, nullptr);
  iree_hal_amdgpu_block_pool_trim(&block_pool);
  EXPECT_EQ(block_pool.free_blocks_head, nullptr);

  iree_hal_amdgpu_block_allocator_deinitialize(&allocator);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

TEST_F(BlockAllocatorTest, AllocateFullBlocks) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));
  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, cpu_agent, memory_pool, host_allocator, &block_pool));

  iree_hal_amdgpu_block_allocator_t allocator;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_initialize(
      &block_pool, /*min_page_size=*/64, &allocator));

  // Allocate an entire block worth of pages.
  void* ptr0 = NULL;
  iree_hal_amdgpu_block_token_t token0 = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_allocate(
      &allocator, block_pool.block_size, &ptr0, &token0));
  EXPECT_NE(ptr0, nullptr);

  // Allocate another entire block worth of pages.
  void* ptr1 = NULL;
  iree_hal_amdgpu_block_token_t token1 = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_allocate(
      &allocator, block_pool.block_size, &ptr1, &token1));
  EXPECT_NE(ptr1, nullptr);

  // Free both allocations.
  iree_hal_amdgpu_block_allocator_free(&allocator, ptr0, token0);
  iree_hal_amdgpu_block_allocator_free(&allocator, ptr1, token1);

  iree_hal_amdgpu_block_allocator_deinitialize(&allocator);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

TEST_F(BlockAllocatorTest, AllocateSpillBlock) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));
  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, cpu_agent, memory_pool, host_allocator, &block_pool));

  iree_hal_amdgpu_block_allocator_t allocator;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_initialize(
      &block_pool, /*min_page_size=*/64, &allocator));

  // Allocate 1 page (remaining should be free).
  void* ptr0 = NULL;
  iree_hal_amdgpu_block_token_t token0 = {0};
  IREE_ASSERT_OK(
      iree_hal_amdgpu_block_allocator_allocate(&allocator, 1, &ptr0, &token0));
  EXPECT_NE(ptr0, nullptr);

  // Allocate an entire block worth of pages to ensure we spill.
  void* ptr1 = NULL;
  iree_hal_amdgpu_block_token_t token1 = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_allocate(
      &allocator, block_pool.block_size, &ptr1, &token1));
  EXPECT_NE(ptr1, nullptr);

  // Allocate 1 more page (should go back to the first block).
  void* ptr2 = NULL;
  iree_hal_amdgpu_block_token_t token2 = {0};
  IREE_ASSERT_OK(
      iree_hal_amdgpu_block_allocator_allocate(&allocator, 1, &ptr2, &token2));
  EXPECT_NE(ptr2, nullptr);

  // Free all allocations.
  iree_hal_amdgpu_block_allocator_free(&allocator, ptr0, token0);
  iree_hal_amdgpu_block_allocator_free(&allocator, ptr1, token1);
  iree_hal_amdgpu_block_allocator_free(&allocator, ptr2, token2);

  iree_hal_amdgpu_block_allocator_deinitialize(&allocator);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

TEST_F(BlockAllocatorTest, AllocateFragmented) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));
  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, cpu_agent, memory_pool, host_allocator, &block_pool));

  iree_hal_amdgpu_block_allocator_t allocator;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_initialize(
      &block_pool, /*min_page_size=*/64, &allocator));

  // Allocate nearly an entire block.
  void* ptr0 = NULL;
  iree_hal_amdgpu_block_token_t token0 = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_allocate(
      &allocator, (allocator.page_count - 1) * allocator.page_size, &ptr0,
      &token0));
  EXPECT_NE(ptr0, nullptr);

  // Allocate another large allocation that won't fit.
  void* ptr1 = NULL;
  iree_hal_amdgpu_block_token_t token1 = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_allocate(
      &allocator, (allocator.page_count - 1) * allocator.page_size, &ptr1,
      &token1));
  EXPECT_NE(ptr1, nullptr);

  // Free all allocations.
  iree_hal_amdgpu_block_allocator_free(&allocator, ptr0, token0);
  iree_hal_amdgpu_block_allocator_free(&allocator, ptr1, token1);

  iree_hal_amdgpu_block_allocator_deinitialize(&allocator);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

TEST_F(BlockAllocatorTest, AllocateEntireBlock) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));
  iree_hal_amdgpu_block_pool_options_t options = {
      /*.block_size=*/1 * 1024,
      /*.min_blocks_per_allocation=*/1,
      /*.initial_capacity=*/0,
  };
  iree_hal_amdgpu_block_pool_t block_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_block_pool_initialize(
      &libhsa, options, cpu_agent, memory_pool, host_allocator, &block_pool));

  iree_hal_amdgpu_block_allocator_t allocator;
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_initialize(
      &block_pool, /*min_page_size=*/64, &allocator));

  // Allocate an entire block worth of pages.
  std::vector<void*> ptrs(allocator.page_count);
  std::vector<iree_hal_amdgpu_block_token_t> tokens(allocator.page_count);
  for (size_t i = 0; i < allocator.page_count; ++i) {
    IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_allocate(
        &allocator, allocator.page_size, &ptrs[i], &tokens[i]));
    EXPECT_NE(ptrs[i], nullptr);
  }

  // Free the first allocation and reallocate it. We should deterministically
  // get the same pointer back. This tests reuse of the first page in a block.
  void* ptr0 = ptrs.front();
  iree_hal_amdgpu_block_allocator_free(&allocator, ptrs.front(),
                                       tokens.front());
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_allocate(
      &allocator, allocator.page_size, &ptrs.front(), &tokens.front()));
  EXPECT_EQ(ptrs.front(), ptr0);

  // Free the last allocation and reallocate in kind.
  void* ptrN = ptrs.back();
  iree_hal_amdgpu_block_allocator_free(&allocator, ptrs.back(), tokens.back());
  IREE_ASSERT_OK(iree_hal_amdgpu_block_allocator_allocate(
      &allocator, allocator.page_size, &ptrs.back(), &tokens.back()));
  EXPECT_EQ(ptrs.back(), ptrN);

  // Free all allocations.
  for (size_t i = 0; i < allocator.page_count; ++i) {
    iree_hal_amdgpu_block_allocator_free(&allocator, ptrs[i], tokens[i]);
  }

  iree_hal_amdgpu_block_allocator_deinitialize(&allocator);

  iree_hal_amdgpu_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace iree::hal::amdgpu
