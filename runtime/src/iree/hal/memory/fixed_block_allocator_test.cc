// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/fixed_block_allocator.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include "iree/async/frontier.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Helper to allocate a frontier on the stack with a given capacity.
#define FRONTIER_ALLOC(name, capacity)                                  \
  alignas(16) uint8_t                                                   \
      name##_storage[sizeof(iree_async_frontier_t) +                    \
                     (capacity) * sizeof(iree_async_frontier_entry_t)]; \
  iree_async_frontier_t* name =                                         \
      reinterpret_cast<iree_async_frontier_t*>(name##_storage);         \
  memset(name##_storage, 0, sizeof(name##_storage))

// Helper: builds a frontier in pre-allocated storage.
static iree_async_frontier_t* BuildFrontier(
    uint8_t* storage, iree_host_size_t storage_size,
    std::initializer_list<iree_async_frontier_entry_t> entries) {
  iree_async_frontier_t* frontier =
      reinterpret_cast<iree_async_frontier_t*>(storage);
  iree_async_frontier_initialize(frontier,
                                 static_cast<uint8_t>(entries.size()));
  uint8_t i = 0;
  for (const auto& entry : entries) {
    frontier->entries[i++] = entry;
  }
  return frontier;
}

#define MAKE_FRONTIER(name, capacity, ...)                              \
  alignas(16) uint8_t                                                   \
      name##_storage[sizeof(iree_async_frontier_t) +                    \
                     (capacity) * sizeof(iree_async_frontier_entry_t)]; \
  iree_async_frontier_t* name =                                         \
      BuildFrontier(name##_storage, sizeof(name##_storage), {__VA_ARGS__})

// Test axes: simple sequential values.
static iree_async_axis_t TestQueueAxis(uint8_t queue_index) {
  return iree_async_axis_make_queue(1, 0, 0, queue_index);
}

// Shorthand for creating frontier entries.
static iree_async_frontier_entry_t E(iree_async_axis_t axis, uint64_t epoch) {
  return {axis, epoch};
}

// Default test options: 64 blocks of 4096 bytes each.
static iree_hal_memory_fixed_block_allocator_options_t DefaultOptions() {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 4096;
  options.block_count = 64;
  options.frontier_capacity = 4;
  return options;
}

//===----------------------------------------------------------------------===//
// Lifecycle tests
//===----------------------------------------------------------------------===//

TEST(FixedBlockAllocator, AllocateFree) {
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      DefaultOptions(), iree_allocator_system(), &pool));
  iree_hal_memory_fixed_block_allocator_stats_t stats;
  iree_hal_memory_fixed_block_allocator_query_stats(pool, &stats);
  EXPECT_EQ(stats.block_count, 64);
  EXPECT_EQ(stats.allocation_count, 0u);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, InvalidOptionsZeroBlockSize) {
  iree_hal_memory_fixed_block_allocator_options_t options = DefaultOptions();
  options.block_size = 0;
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_fixed_block_allocator_allocate(
                            options, iree_allocator_system(), &pool));
}

TEST(FixedBlockAllocator, InvalidOptionsZeroBlockCount) {
  iree_hal_memory_fixed_block_allocator_options_t options = DefaultOptions();
  options.block_count = 0;
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_fixed_block_allocator_allocate(
                            options, iree_allocator_system(), &pool));
}

TEST(FixedBlockAllocator, InvalidOptionsExceedsMaxBlocks) {
  iree_hal_memory_fixed_block_allocator_options_t options = DefaultOptions();
  options.block_count = IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS + 1;
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_fixed_block_allocator_allocate(
                            options, iree_allocator_system(), &pool));
}

TEST(FixedBlockAllocator, InvalidOptionsOffsetRangeOverflow) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = UINT64_MAX;
  options.block_count = 2;
  options.frontier_capacity = 1;
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_fixed_block_allocator_allocate(
                            options, iree_allocator_system(), &pool));
}

TEST(FixedBlockAllocator, MaxBlockCountAccepted) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 64;
  options.block_count = IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS;
  options.frontier_capacity = 1;
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));
  iree_hal_memory_fixed_block_allocator_stats_t stats;
  iree_hal_memory_fixed_block_allocator_query_stats(pool, &stats);
  EXPECT_EQ(stats.block_count,
            IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, DefaultFrontierCapacity) {
  iree_hal_memory_fixed_block_allocator_options_t options = DefaultOptions();
  options.frontier_capacity = 0;  // Should use default.
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));
  EXPECT_EQ(pool->frontier_capacity,
            IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_DEFAULT_FRONTIER_CAPACITY);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

//===----------------------------------------------------------------------===//
// Single allocation tests
//===----------------------------------------------------------------------===//

TEST(FixedBlockAllocator, AcquireSingle) {
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      DefaultOptions(), iree_allocator_system(), &pool));

  iree_hal_memory_fixed_block_allocator_allocation_t allocation;
  IREE_ASSERT_OK(
      iree_hal_memory_fixed_block_allocator_acquire(pool, &allocation));

  // First allocation should be block 0 at offset 0.
  EXPECT_EQ(allocation.block_index, 0);
  EXPECT_EQ(allocation.offset, 0u);
  EXPECT_EQ(allocation.death_frontier, nullptr);
  EXPECT_EQ(allocation.block_flags,
            IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_NONE);

  iree_hal_memory_fixed_block_allocator_stats_t stats;
  iree_hal_memory_fixed_block_allocator_query_stats(pool, &stats);
  EXPECT_EQ(stats.allocation_count, 1u);

  iree_hal_memory_fixed_block_allocator_release(pool, allocation.block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, BlockOffsetsAreCorrect) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 256;
  options.block_count = 16;
  options.frontier_capacity = 2;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  // Allocate all blocks and verify offsets.
  std::vector<iree_hal_memory_fixed_block_allocator_allocation_t> allocations(
      16);
  for (int i = 0; i < 16; ++i) {
    IREE_ASSERT_OK(
        iree_hal_memory_fixed_block_allocator_acquire(pool, &allocations[i]));
  }

  // Each block's offset must equal block_index * block_size.
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(allocations[i].offset,
              static_cast<iree_device_size_t>(allocations[i].block_index) * 256)
        << "block " << i;
  }

  // All block indices must be unique.
  std::set<uint32_t> indices;
  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(indices.insert(allocations[i].block_index).second)
        << "duplicate block_index " << allocations[i].block_index;
  }

  for (int i = 0; i < 16; ++i) {
    iree_hal_memory_fixed_block_allocator_release(
        pool, allocations[i].block_index, nullptr);
  }
  iree_hal_memory_fixed_block_allocator_free(pool);
}

//===----------------------------------------------------------------------===//
// Exhaustion and reuse tests
//===----------------------------------------------------------------------===//

TEST(FixedBlockAllocator, ExhaustionReturnsResourceExhausted) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 64;
  options.block_count = 4;
  options.frontier_capacity = 2;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  // Allocate all blocks.
  iree_hal_memory_fixed_block_allocator_allocation_t allocations[4];
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(
        iree_hal_memory_fixed_block_allocator_acquire(pool, &allocations[i]));
  }

  // Next allocation must fail.
  iree_hal_memory_fixed_block_allocator_allocation_t extra;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_memory_fixed_block_allocator_acquire(pool, &extra));

  for (int i = 0; i < 4; ++i) {
    iree_hal_memory_fixed_block_allocator_release(
        pool, allocations[i].block_index, nullptr);
  }
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, TryAcquireExhaustionReturnsResult) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 64;
  options.block_count = 4;
  options.frontier_capacity = 2;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  iree_hal_memory_fixed_block_allocator_allocation_t allocations[4];
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(
        iree_hal_memory_fixed_block_allocator_acquire(pool, &allocations[i]));
  }

  iree_hal_memory_fixed_block_allocator_allocation_t extra;
  iree_hal_memory_fixed_block_allocator_acquire_result_t result =
      IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_OK;
  IREE_ASSERT_OK(
      iree_hal_memory_fixed_block_allocator_try_acquire(pool, &extra, &result));
  EXPECT_EQ(result, IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED);

  iree_hal_memory_fixed_block_allocator_stats_t stats;
  iree_hal_memory_fixed_block_allocator_query_stats(pool, &stats);
  EXPECT_EQ(stats.allocation_count, 4u);

  for (int i = 0; i < 4; ++i) {
    iree_hal_memory_fixed_block_allocator_release(
        pool, allocations[i].block_index, nullptr);
  }
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, ReleaseAndReacquire) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 128;
  options.block_count = 2;
  options.frontier_capacity = 2;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  // Acquire both blocks.
  iree_hal_memory_fixed_block_allocator_allocation_t a, b;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &a));
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &b));

  // Release one, then reacquire; should succeed.
  iree_hal_memory_fixed_block_allocator_release(pool, a.block_index, nullptr);

  iree_hal_memory_fixed_block_allocator_allocation_t reused;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &reused));
  EXPECT_EQ(reused.block_index, a.block_index);

  iree_hal_memory_fixed_block_allocator_release(pool, reused.block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_release(pool, b.block_index, nullptr);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

//===----------------------------------------------------------------------===//
// Death frontier tests
//===----------------------------------------------------------------------===//

TEST(FixedBlockAllocator, FrontierPreservedAcrossAcquireRelease) {
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      DefaultOptions(), iree_allocator_system(), &pool));

  // Acquire a block.
  iree_hal_memory_fixed_block_allocator_allocation_t first;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &first));
  EXPECT_EQ(first.death_frontier, nullptr);

  // Release with a death frontier.
  MAKE_FRONTIER(death, 2, E(TestQueueAxis(0), 42), E(TestQueueAxis(1), 100));
  iree_hal_memory_fixed_block_allocator_release(pool, first.block_index, death);

  // Reacquire the same block; death frontier should be visible.
  iree_hal_memory_fixed_block_allocator_allocation_t second;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &second));
  EXPECT_EQ(second.block_index, first.block_index);
  ASSERT_NE(second.death_frontier, nullptr);
  EXPECT_EQ(second.death_frontier->entry_count, 2);
  EXPECT_EQ(second.death_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(second.death_frontier->entries[0].epoch, 42u);
  EXPECT_EQ(second.death_frontier->entries[1].axis, TestQueueAxis(1));
  EXPECT_EQ(second.death_frontier->entries[1].epoch, 100u);
  EXPECT_EQ(second.block_flags,
            IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_NONE);

  iree_hal_memory_fixed_block_allocator_release(pool, second.block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, FrontierClearedOnReleaseWithNull) {
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      DefaultOptions(), iree_allocator_system(), &pool));

  // Acquire, release with frontier, release again with null.
  iree_hal_memory_fixed_block_allocator_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 5));
  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index, f);

  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  ASSERT_NE(alloc.death_frontier, nullptr);
  // Release with null; frontier should be cleared.
  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index,
                                                nullptr);

  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  EXPECT_EQ(alloc.death_frontier, nullptr);

  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, RestorePreservesFrontierMetadata) {
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      DefaultOptions(), iree_allocator_system(), &pool));

  iree_hal_memory_fixed_block_allocator_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));

  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 42));
  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index, death);

  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entry_count, 1);
  EXPECT_EQ(alloc.death_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(alloc.death_frontier->entries[0].epoch, 42u);

  iree_hal_memory_fixed_block_allocator_restore(pool, alloc.block_index);

  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entry_count, 1);
  EXPECT_EQ(alloc.death_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(alloc.death_frontier->entries[0].epoch, 42u);

  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, BlockDeathFrontierAccessor) {
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      DefaultOptions(), iree_allocator_system(), &pool));

  iree_hal_memory_fixed_block_allocator_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));

  // Fresh block has no frontier.
  EXPECT_EQ(iree_hal_memory_fixed_block_allocator_block_death_frontier(
                pool, alloc.block_index),
            nullptr);

  // Release with frontier, reacquire, check accessor.
  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 99));
  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index, f);
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));

  const iree_async_frontier_t* retrieved =
      iree_hal_memory_fixed_block_allocator_block_death_frontier(
          pool, alloc.block_index);
  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved->entry_count, 1);
  EXPECT_EQ(retrieved->entries[0].epoch, 99u);

  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

//===----------------------------------------------------------------------===//
// Taint tests
//===----------------------------------------------------------------------===//

TEST(FixedBlockAllocator, OversizedFrontierCausesTaint) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 256;
  options.block_count = 4;
  options.frontier_capacity = 2;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  iree_hal_memory_fixed_block_allocator_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));

  // Release with a 3-entry frontier when capacity is 2; should taint.
  MAKE_FRONTIER(big, 3, E(TestQueueAxis(0), 1), E(TestQueueAxis(1), 2),
                E(TestQueueAxis(2), 3));
  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index, big);

  // Reacquire; should see TAINTED flag and no frontier.
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  EXPECT_TRUE(alloc.block_flags &
              IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED);
  EXPECT_EQ(alloc.death_frontier, nullptr);

  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, TaintClearedOnReleaseWithFittingFrontier) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 256;
  options.block_count = 4;
  options.frontier_capacity = 2;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  // Taint the block.
  iree_hal_memory_fixed_block_allocator_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  MAKE_FRONTIER(big, 3, E(TestQueueAxis(0), 1), E(TestQueueAxis(1), 2),
                E(TestQueueAxis(2), 3));
  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index, big);
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  EXPECT_TRUE(alloc.block_flags &
              IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED);

  // Release with a fitting frontier; taint should clear.
  MAKE_FRONTIER(small, 2, E(TestQueueAxis(0), 10), E(TestQueueAxis(1), 20));
  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index, small);

  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  EXPECT_EQ(alloc.block_flags,
            IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_NONE);
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entry_count, 2);

  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, RestorePreservesTaint) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 256;
  options.block_count = 1;
  options.frontier_capacity = 1;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  iree_hal_memory_fixed_block_allocator_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  MAKE_FRONTIER(big, 2, E(TestQueueAxis(0), 1), E(TestQueueAxis(1), 2));
  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index, big);

  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  EXPECT_TRUE(alloc.block_flags &
              IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED);
  EXPECT_EQ(alloc.death_frontier, nullptr);

  iree_hal_memory_fixed_block_allocator_restore(pool, alloc.block_index);

  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  EXPECT_TRUE(alloc.block_flags &
              IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED);
  EXPECT_EQ(alloc.death_frontier, nullptr);

  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, BlockFlagsAccessor) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 64;
  options.block_count = 2;
  options.frontier_capacity = 1;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  iree_hal_memory_fixed_block_allocator_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  EXPECT_EQ(iree_hal_memory_fixed_block_allocator_block_flags(
                pool, alloc.block_index),
            IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_NONE);

  // Taint it.
  MAKE_FRONTIER(big, 2, E(TestQueueAxis(0), 1), E(TestQueueAxis(1), 2));
  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index, big);
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  EXPECT_TRUE(iree_hal_memory_fixed_block_allocator_block_flags(
                  pool, alloc.block_index) &
              IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED);

  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

//===----------------------------------------------------------------------===//
// Partial word tests (block_count not a multiple of 64)
//===----------------------------------------------------------------------===//

TEST(FixedBlockAllocator, PartialWordBlockCount) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 64;
  options.block_count = 100;  // 1 full word (64) + 36 in second word.
  options.frontier_capacity = 1;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  // Acquire all 100 blocks.
  std::vector<iree_hal_memory_fixed_block_allocator_allocation_t> allocations(
      100);
  for (int i = 0; i < 100; ++i) {
    IREE_ASSERT_OK(
        iree_hal_memory_fixed_block_allocator_acquire(pool, &allocations[i]))
        << "allocation " << i;
  }

  // 101st should fail.
  iree_hal_memory_fixed_block_allocator_allocation_t extra;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_memory_fixed_block_allocator_acquire(pool, &extra));

  // All indices must be in [0, 100) and unique.
  std::set<uint32_t> indices;
  for (int i = 0; i < 100; ++i) {
    EXPECT_LT(allocations[i].block_index, 100);
    EXPECT_TRUE(indices.insert(allocations[i].block_index).second);
  }

  for (int i = 0; i < 100; ++i) {
    iree_hal_memory_fixed_block_allocator_release(
        pool, allocations[i].block_index, nullptr);
  }
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, SingleFixedBlockAllocator) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 1024;
  options.block_count = 1;
  options.frontier_capacity = 1;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  iree_hal_memory_fixed_block_allocator_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
  EXPECT_EQ(alloc.block_index, 0);
  EXPECT_EQ(alloc.offset, 0u);

  iree_hal_memory_fixed_block_allocator_allocation_t extra;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_memory_fixed_block_allocator_acquire(pool, &extra));

  iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_free(pool);
}

//===----------------------------------------------------------------------===//
// Stats tests
//===----------------------------------------------------------------------===//

TEST(FixedBlockAllocator, StatsTracking) {
  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 64;
  options.block_count = 8;
  options.frontier_capacity = 1;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  iree_hal_memory_fixed_block_allocator_stats_t stats;
  iree_hal_memory_fixed_block_allocator_query_stats(pool, &stats);
  EXPECT_EQ(stats.block_count, 8);
  EXPECT_EQ(stats.allocation_count, 0u);

  // Acquire 3 blocks.
  iree_hal_memory_fixed_block_allocator_allocation_t a[3];
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &a[i]));
  }
  iree_hal_memory_fixed_block_allocator_query_stats(pool, &stats);
  EXPECT_EQ(stats.allocation_count, 3u);

  // Release one.
  iree_hal_memory_fixed_block_allocator_release(pool, a[1].block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_query_stats(pool, &stats);
  EXPECT_EQ(stats.allocation_count, 2u);

  // Release remaining.
  iree_hal_memory_fixed_block_allocator_release(pool, a[0].block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_release(pool, a[2].block_index,
                                                nullptr);
  iree_hal_memory_fixed_block_allocator_query_stats(pool, &stats);
  EXPECT_EQ(stats.allocation_count, 0u);

  iree_hal_memory_fixed_block_allocator_free(pool);
}

//===----------------------------------------------------------------------===//
// Multi-threaded tests
//===----------------------------------------------------------------------===//

TEST(FixedBlockAllocator, ConcurrentAcquireExhaustion) {
  static constexpr int kBlockCount = 256;
  static constexpr int kThreadCount = 8;

  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 64;
  options.block_count = kBlockCount;
  options.frontier_capacity = 1;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  // Each thread tries to acquire from the pool. Collectively they should
  // get exactly kBlockCount successes and no double allocations.
  struct ThreadResult {
    std::vector<uint32_t> block_indices;
    int failure_count = 0;
  };
  std::vector<ThreadResult> results(kThreadCount);

  std::thread threads[kThreadCount];
  for (int t = 0; t < kThreadCount; ++t) {
    threads[t] = std::thread([&, t]() {
      while (true) {
        iree_hal_memory_fixed_block_allocator_allocation_t alloc;
        iree_hal_memory_fixed_block_allocator_acquire_result_t result =
            IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED;
        IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_try_acquire(
            pool, &alloc, &result));
        if (result == IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_OK) {
          results[t].block_indices.push_back(alloc.block_index);
        } else {
          results[t].failure_count++;
          break;  // Pool exhausted.
        }
      }
    });
  }
  for (int t = 0; t < kThreadCount; ++t) {
    threads[t].join();
  }

  // Total successful acquisitions should be exactly kBlockCount.
  int total_allocations = 0;
  std::set<uint32_t> all_indices;
  for (int t = 0; t < kThreadCount; ++t) {
    total_allocations += static_cast<int>(results[t].block_indices.size());
    for (uint32_t idx : results[t].block_indices) {
      EXPECT_TRUE(all_indices.insert(idx).second)
          << "duplicate block_index " << idx;
    }
  }
  EXPECT_EQ(total_allocations, kBlockCount);

  // Release all blocks.
  for (int t = 0; t < kThreadCount; ++t) {
    for (uint32_t idx : results[t].block_indices) {
      iree_hal_memory_fixed_block_allocator_release(pool, idx, nullptr);
    }
  }
  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, ConcurrentAcquireRelease) {
  static constexpr int kBlockCount = 128;
  static constexpr int kThreadCount = 4;
  static constexpr int kIterationsPerThread = 1000;

  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 64;
  options.block_count = kBlockCount;
  options.frontier_capacity = 2;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  // Each thread repeatedly acquires and releases blocks. At completion the pool
  // should be empty (all blocks released).
  std::atomic<int> total_successes{0};

  std::thread threads[kThreadCount];
  for (int t = 0; t < kThreadCount; ++t) {
    threads[t] = std::thread([&]() {
      for (int i = 0; i < kIterationsPerThread; ++i) {
        iree_hal_memory_fixed_block_allocator_allocation_t alloc;
        iree_hal_memory_fixed_block_allocator_acquire_result_t result =
            IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED;
        IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_try_acquire(
            pool, &alloc, &result));
        if (result == IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_OK) {
          total_successes.fetch_add(1, std::memory_order_relaxed);
          // Optionally attach a frontier.
          if (i % 3 == 0) {
            MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), static_cast<uint64_t>(i)));
            iree_hal_memory_fixed_block_allocator_release(pool,
                                                          alloc.block_index, f);
          } else {
            iree_hal_memory_fixed_block_allocator_release(
                pool, alloc.block_index, nullptr);
          }
        }
      }
    });
  }
  for (int t = 0; t < kThreadCount; ++t) {
    threads[t].join();
  }

  // Pool should be empty now.
  iree_hal_memory_fixed_block_allocator_stats_t stats;
  iree_hal_memory_fixed_block_allocator_query_stats(pool, &stats);
  EXPECT_EQ(stats.allocation_count, 0u);
  EXPECT_GT(total_successes.load(), 0);

  iree_hal_memory_fixed_block_allocator_free(pool);
}

TEST(FixedBlockAllocator, ConcurrentFrontierVisibility) {
  // Verify that frontier data written by the releasing thread is visible to the
  // acquiring thread (tests acquire/release ordering on the bitmap).
  //
  // Protocol: producer releases pre-acquired blocks one at a time with a known
  // frontier. Consumer acquires, checks the frontier data, and immediately
  // releases (with no frontier). Both sides count. The producer signals done
  // via an atomic flag; the consumer runs until it has consumed all blocks.
  static constexpr int kBlockCount = 64;

  iree_hal_memory_fixed_block_allocator_options_t options = {};
  options.block_size = 64;
  options.block_count = kBlockCount;
  options.frontier_capacity = 2;

  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_allocate(
      options, iree_allocator_system(), &pool));

  // Pre-acquire all blocks.
  std::vector<uint32_t> block_indices(kBlockCount);
  for (int i = 0; i < kBlockCount; ++i) {
    iree_hal_memory_fixed_block_allocator_allocation_t alloc;
    IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_acquire(pool, &alloc));
    block_indices[i] = alloc.block_index;
  }

  std::atomic<bool> producer_done{false};
  std::atomic<int> consumed_count{0};
  std::atomic<int> frontier_verified_count{0};

  auto producer = [&]() {
    for (int i = 0; i < kBlockCount; ++i) {
      MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 1000 + i));
      iree_hal_memory_fixed_block_allocator_release(pool, block_indices[i], f);
    }
    producer_done.store(true, std::memory_order_release);
  };

  auto consumer = [&]() {
    int consumed = 0;
    while (consumed < kBlockCount) {
      iree_hal_memory_fixed_block_allocator_allocation_t alloc;
      iree_hal_memory_fixed_block_allocator_acquire_result_t result =
          IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED;
      IREE_ASSERT_OK(iree_hal_memory_fixed_block_allocator_try_acquire(
          pool, &alloc, &result));
      if (result == IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED) {
        // Yield to let the producer make progress.
        std::this_thread::yield();
        continue;
      }
      consumed++;
      if (alloc.death_frontier != nullptr) {
        EXPECT_EQ(alloc.death_frontier->entry_count, 1);
        EXPECT_GE(alloc.death_frontier->entries[0].epoch, 1000u);
        EXPECT_LT(alloc.death_frontier->entries[0].epoch, 1000u + kBlockCount);
        frontier_verified_count.fetch_add(1, std::memory_order_relaxed);
      }
      iree_hal_memory_fixed_block_allocator_release(pool, alloc.block_index,
                                                    nullptr);
    }
    consumed_count.store(consumed, std::memory_order_relaxed);
  };

  std::thread producer_thread(producer);
  std::thread consumer_thread(consumer);
  producer_thread.join();
  consumer_thread.join();

  EXPECT_EQ(consumed_count.load(), kBlockCount);
  EXPECT_GT(frontier_verified_count.load(), 0);

  iree_hal_memory_fixed_block_allocator_stats_t stats;
  iree_hal_memory_fixed_block_allocator_query_stats(pool, &stats);
  EXPECT_EQ(stats.allocation_count, 0u);

  iree_hal_memory_fixed_block_allocator_free(pool);
}

}  // namespace
