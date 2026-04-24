// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/arena.h"

#include <cstring>

#include "iree/async/frontier.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

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

static iree_async_axis_t TestQueueAxis(uint8_t queue_index) {
  return iree_async_axis_make_queue(1, 0, 0, queue_index);
}

static iree_async_frontier_entry_t E(iree_async_axis_t axis, uint64_t epoch) {
  return {axis, epoch};
}

static iree_hal_memory_arena_options_t DefaultOptions() {
  iree_hal_memory_arena_options_t options = {};
  options.capacity = 65536;
  options.frontier_capacity = 4;
  return options;
}

//===----------------------------------------------------------------------===//
// Lifecycle tests
//===----------------------------------------------------------------------===//

TEST(Arena, AllocateFree) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));
  iree_hal_memory_arena_stats_t stats;
  iree_hal_memory_arena_query_stats(arena, &stats);
  EXPECT_EQ(stats.capacity, 65536);
  EXPECT_EQ(stats.bytes_used, 0u);
  EXPECT_EQ(stats.allocation_count, 0u);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, InvalidOptionsZeroCapacity) {
  iree_hal_memory_arena_options_t options = DefaultOptions();
  options.capacity = 0;
  iree_hal_memory_arena_t* arena = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_memory_arena_allocate(options, iree_allocator_system(), &arena));
}

TEST(Arena, DefaultFrontierCapacity) {
  iree_hal_memory_arena_options_t options = {};
  options.capacity = 4096;
  options.frontier_capacity = 0;
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(
      iree_hal_memory_arena_allocate(options, iree_allocator_system(), &arena));
  EXPECT_EQ(arena->frontier_capacity,
            IREE_HAL_MEMORY_ARENA_DEFAULT_FRONTIER_CAPACITY);
  iree_hal_memory_arena_free(arena);
}

//===----------------------------------------------------------------------===//
// Acquire tests
//===----------------------------------------------------------------------===//

TEST(Arena, AcquireSingle) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 256, 1, &alloc));
  EXPECT_EQ(alloc.offset, 0u);
  EXPECT_EQ(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.flags, IREE_HAL_MEMORY_ARENA_FLAG_NONE);

  iree_hal_memory_arena_stats_t stats;
  iree_hal_memory_arena_query_stats(arena, &stats);
  EXPECT_EQ(stats.bytes_used, 256u);
  EXPECT_EQ(stats.allocation_count, 1u);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, AcquireAlignment) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  // First: 1 byte at offset 0.
  iree_hal_memory_arena_allocation_t a;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 1, 1, &a));
  EXPECT_EQ(a.offset, 0u);

  // Second: 1 byte aligned to 16. Should skip to offset 16.
  iree_hal_memory_arena_allocation_t b;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 1, 16, &b));
  EXPECT_EQ(b.offset, 16u);

  // Third: 1 byte aligned to 256. Should skip to offset 256.
  iree_hal_memory_arena_allocation_t c;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 1, 256, &c));
  EXPECT_EQ(c.offset, 256u);

  iree_hal_memory_arena_stats_t stats;
  iree_hal_memory_arena_query_stats(arena, &stats);
  // used = 0+1=1, then align to 16+1=17, then align to 256+1=257
  EXPECT_EQ(stats.bytes_used, 257u);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, AcquireExhaustion) {
  iree_hal_memory_arena_options_t options = {};
  options.capacity = 100;
  options.frontier_capacity = 1;
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(
      iree_hal_memory_arena_allocate(options, iree_allocator_system(), &arena));

  // Acquire 100 bytes; should succeed.
  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 100, 1, &alloc));
  EXPECT_EQ(alloc.offset, 0u);

  // Acquire 1 more byte; should fail.
  iree_hal_memory_arena_allocation_t extra;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_memory_arena_acquire(arena, 1, 1, &extra));

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, AcquireExhaustionFromAlignmentPadding) {
  iree_hal_memory_arena_options_t options = {};
  options.capacity = 32;
  options.frontier_capacity = 1;
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(
      iree_hal_memory_arena_allocate(options, iree_allocator_system(), &arena));

  // Acquire 1 byte (used=1).
  iree_hal_memory_arena_allocation_t a;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 1, 1, &a));

  // Acquire 1 byte at alignment 32. Aligned offset = 32, new_used = 33 > 32.
  iree_hal_memory_arena_allocation_t b;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_memory_arena_acquire(arena, 1, 32, &b));

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, AcquireInvalidZeroLength) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  iree_hal_memory_arena_allocation_t alloc;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_arena_acquire(arena, 0, 1, &alloc));
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, AcquireInvalidAlignment) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  iree_hal_memory_arena_allocation_t alloc;
  // Alignment 0.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_arena_acquire(arena, 16, 0, &alloc));
  // Non-power-of-two alignment.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_arena_acquire(arena, 16, 3, &alloc));
  iree_hal_memory_arena_free(arena);
}

//===----------------------------------------------------------------------===//
// Release and reset tests
//===----------------------------------------------------------------------===//

TEST(Arena, ReleaseResetsArena) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  // Acquire 3 regions.
  iree_hal_memory_arena_allocation_t a, b, c;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 100, 1, &a));
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 200, 1, &b));
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 300, 1, &c));

  iree_hal_memory_arena_stats_t stats;
  iree_hal_memory_arena_query_stats(arena, &stats);
  EXPECT_EQ(stats.bytes_used, 600u);
  EXPECT_EQ(stats.allocation_count, 3u);

  // Release all 3.
  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_release(arena, nullptr);

  // Arena should be reset.
  iree_hal_memory_arena_query_stats(arena, &stats);
  EXPECT_EQ(stats.bytes_used, 0u);
  EXPECT_EQ(stats.allocation_count, 0u);

  iree_hal_memory_arena_free(arena);
}

TEST(Arena, PartialReleaseNoReset) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  iree_hal_memory_arena_allocation_t a, b;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 100, 1, &a));
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 200, 1, &b));

  // Release one; arena should NOT reset.
  iree_hal_memory_arena_release(arena, nullptr);

  iree_hal_memory_arena_stats_t stats;
  iree_hal_memory_arena_query_stats(arena, &stats);
  EXPECT_EQ(stats.bytes_used, 300u);  // Still at high-water mark.
  EXPECT_EQ(stats.allocation_count, 1u);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_query_stats(arena, &stats);
  EXPECT_EQ(stats.bytes_used, 0u);  // Now reset.

  iree_hal_memory_arena_free(arena);
}

TEST(Arena, ReacquireAfterReset) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  // Batch 1: acquire, release.
  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 1024, 1, &alloc));
  EXPECT_EQ(alloc.offset, 0u);
  iree_hal_memory_arena_release(arena, nullptr);

  // Batch 2: offsets should start from 0 again.
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 512, 1, &alloc));
  EXPECT_EQ(alloc.offset, 0u);
  iree_hal_memory_arena_release(arena, nullptr);

  iree_hal_memory_arena_free(arena);
}

//===----------------------------------------------------------------------===//
// Frontier protocol tests
//===----------------------------------------------------------------------===//

TEST(Arena, EmptyPreviousFrontierOnFirstBatch) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  // First-ever acquisition: no previous batch, so death_frontier is NULL.
  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  EXPECT_EQ(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.flags, IREE_HAL_MEMORY_ARENA_FLAG_NONE);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, FrontierAccumulatedAcrossBatch) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  // Batch 1: acquire and release with a frontier.
  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 42));
  iree_hal_memory_arena_release(arena, f);

  // Batch 2: should see the accumulated frontier from batch 1.
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entry_count, 1);
  EXPECT_EQ(alloc.death_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(alloc.death_frontier->entries[0].epoch, 42u);
  EXPECT_EQ(alloc.flags, IREE_HAL_MEMORY_ARENA_FLAG_NONE);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, FrontierJoinAcrossReleases) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  // Batch 1: two acquisitions released with different axes.
  iree_hal_memory_arena_allocation_t a, b;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &a));
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &b));

  MAKE_FRONTIER(f1, 1, E(TestQueueAxis(0), 10));
  MAKE_FRONTIER(f2, 1, E(TestQueueAxis(1), 20));
  iree_hal_memory_arena_release(arena, f1);
  iree_hal_memory_arena_release(arena, f2);

  // Batch 2: should see JOIN({Q0:10}, {Q1:20}) = {Q0:10, Q1:20}.
  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entry_count, 2);
  EXPECT_EQ(alloc.death_frontier->entries[0].epoch, 10u);
  EXPECT_EQ(alloc.death_frontier->entries[1].epoch, 20u);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, FrontierEpochMaxOnSameAxis) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  // Batch 1: two releases on the same axis with different epochs.
  iree_hal_memory_arena_allocation_t a, b;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &a));
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &b));

  MAKE_FRONTIER(f1, 1, E(TestQueueAxis(0), 5));
  MAKE_FRONTIER(f2, 1, E(TestQueueAxis(0), 15));
  iree_hal_memory_arena_release(arena, f1);
  iree_hal_memory_arena_release(arena, f2);

  // Batch 2: should see max(5, 15) = 15.
  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entry_count, 1);
  EXPECT_EQ(alloc.death_frontier->entries[0].epoch, 15u);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, NullFrontierRelease) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  // Batch 1: one release with frontier, one with NULL.
  iree_hal_memory_arena_allocation_t a, b;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &a));
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &b));

  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 99));
  iree_hal_memory_arena_release(arena, f);
  iree_hal_memory_arena_release(arena, nullptr);  // Should not corrupt.

  // Batch 2: should still see the frontier from the first release.
  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entry_count, 1);
  EXPECT_EQ(alloc.death_frontier->entries[0].epoch, 99u);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, MultipleBatchCycles) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  // Batch 1: frontier {Q0:10}
  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  EXPECT_EQ(alloc.death_frontier, nullptr);  // First batch, no previous.
  MAKE_FRONTIER(f1, 1, E(TestQueueAxis(0), 10));
  iree_hal_memory_arena_release(arena, f1);

  // Batch 2: sees {Q0:10}, releases with {Q0:20}
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entries[0].epoch, 10u);
  MAKE_FRONTIER(f2, 1, E(TestQueueAxis(0), 20));
  iree_hal_memory_arena_release(arena, f2);

  // Batch 3: sees {Q0:20}
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entries[0].epoch, 20u);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, AllAcquisitionsSeePreviousFrontier) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  // Batch 1: release with frontier.
  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  MAKE_FRONTIER(f, 1, E(TestQueueAxis(0), 50));
  iree_hal_memory_arena_release(arena, f);

  // Batch 2: ALL acquisitions should see the same previous frontier.
  iree_hal_memory_arena_allocation_t a, b, c;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &a));
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &b));
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &c));

  ASSERT_NE(a.death_frontier, nullptr);
  ASSERT_NE(b.death_frontier, nullptr);
  ASSERT_NE(c.death_frontier, nullptr);
  // All three should point to the same frontier (same pointer).
  EXPECT_EQ(a.death_frontier, b.death_frontier);
  EXPECT_EQ(b.death_frontier, c.death_frontier);
  EXPECT_EQ(a.death_frontier->entries[0].epoch, 50u);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

//===----------------------------------------------------------------------===//
// Taint tests
//===----------------------------------------------------------------------===//

TEST(Arena, TaintOnFrontierOverflow) {
  iree_hal_memory_arena_options_t options = {};
  options.capacity = 4096;
  options.frontier_capacity = 1;  // Only 1 entry; merging 2 axes overflows.

  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(
      iree_hal_memory_arena_allocate(options, iree_allocator_system(), &arena));

  // Batch 1: release with 2 different axes; exceeds capacity of 1.
  iree_hal_memory_arena_allocation_t a, b;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &a));
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &b));

  MAKE_FRONTIER(f1, 1, E(TestQueueAxis(0), 10));
  MAKE_FRONTIER(f2, 1, E(TestQueueAxis(1), 20));
  iree_hal_memory_arena_release(arena, f1);
  iree_hal_memory_arena_release(arena, f2);  // Should cause taint.

  // Batch 2: should see taint, frontier should be NULL.
  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  EXPECT_TRUE(alloc.flags & IREE_HAL_MEMORY_ARENA_FLAG_TAINTED);
  EXPECT_EQ(alloc.death_frontier, nullptr);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

TEST(Arena, TaintClearedOnNextBatch) {
  iree_hal_memory_arena_options_t options = {};
  options.capacity = 4096;
  options.frontier_capacity = 1;

  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(
      iree_hal_memory_arena_allocate(options, iree_allocator_system(), &arena));

  // Batch 1: cause taint.
  iree_hal_memory_arena_allocation_t a, b;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &a));
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &b));
  MAKE_FRONTIER(f1, 1, E(TestQueueAxis(0), 10));
  MAKE_FRONTIER(f2, 1, E(TestQueueAxis(1), 20));
  iree_hal_memory_arena_release(arena, f1);
  iree_hal_memory_arena_release(arena, f2);

  // Batch 2: tainted. Release with a single-axis frontier that fits.
  iree_hal_memory_arena_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  EXPECT_TRUE(alloc.flags & IREE_HAL_MEMORY_ARENA_FLAG_TAINTED);
  MAKE_FRONTIER(f3, 1, E(TestQueueAxis(0), 30));
  iree_hal_memory_arena_release(arena, f3);

  // Batch 3: taint should be cleared, frontier should be valid.
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 64, 1, &alloc));
  EXPECT_EQ(alloc.flags, IREE_HAL_MEMORY_ARENA_FLAG_NONE);
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entry_count, 1);
  EXPECT_EQ(alloc.death_frontier->entries[0].epoch, 30u);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_free(arena);
}

//===----------------------------------------------------------------------===//
// Stats tests
//===----------------------------------------------------------------------===//

TEST(Arena, StatsTracking) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_ASSERT_OK(iree_hal_memory_arena_allocate(
      DefaultOptions(), iree_allocator_system(), &arena));

  iree_hal_memory_arena_stats_t stats;
  iree_hal_memory_arena_query_stats(arena, &stats);
  EXPECT_EQ(stats.capacity, 65536u);
  EXPECT_EQ(stats.bytes_used, 0u);
  EXPECT_EQ(stats.allocation_count, 0u);

  iree_hal_memory_arena_allocation_t a, b;
  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 100, 1, &a));
  iree_hal_memory_arena_query_stats(arena, &stats);
  EXPECT_EQ(stats.bytes_used, 100u);
  EXPECT_EQ(stats.allocation_count, 1u);

  IREE_ASSERT_OK(iree_hal_memory_arena_acquire(arena, 200, 1, &b));
  iree_hal_memory_arena_query_stats(arena, &stats);
  EXPECT_EQ(stats.bytes_used, 300u);
  EXPECT_EQ(stats.allocation_count, 2u);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_query_stats(arena, &stats);
  EXPECT_EQ(stats.bytes_used, 300u);  // Not reset yet.
  EXPECT_EQ(stats.allocation_count, 1u);

  iree_hal_memory_arena_release(arena, nullptr);
  iree_hal_memory_arena_query_stats(arena, &stats);
  EXPECT_EQ(stats.bytes_used, 0u);  // Reset.
  EXPECT_EQ(stats.allocation_count, 0u);

  iree_hal_memory_arena_free(arena);
}

}  // namespace
