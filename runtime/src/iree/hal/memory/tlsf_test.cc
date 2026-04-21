// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tlsf.h"

#include <cstring>
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

// Shorthand for creating frontier entries.
static iree_async_frontier_entry_t E(iree_async_axis_t axis, uint64_t epoch) {
  return {axis, epoch};
}

// Test axes: simple sequential values.
static iree_async_axis_t TestQueueAxis(uint8_t queue_index) {
  return iree_async_axis_make_queue(1, 0, 0, queue_index);
}

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

// Default options for a 1 MB range with 16-byte alignment and 8-entry
// frontiers. Sufficient for most tests.
static iree_hal_memory_tlsf_options_t DefaultOptions() {
  iree_hal_memory_tlsf_options_t options = {};
  options.range_length = 1024 * 1024;  // 1 MB
  options.alignment = 16;
  options.initial_block_capacity = 64;
  options.frontier_capacity = 8;
  return options;
}

static constexpr uint32_t kTestAllocatorFlagFailRealloc = 1u << 0;

struct TestAllocatorState {
  // Allocator used for commands that are not explicitly failed by test flags.
  iree_allocator_t base_allocator;

  // Bitfield of kTestAllocatorFlag* values controlling injected failures.
  uint32_t flags = 0;
};

static iree_status_t TestAllocatorCtl(void* self,
                                      iree_allocator_command_t command,
                                      const void* params, void** inout_ptr) {
  TestAllocatorState* state = reinterpret_cast<TestAllocatorState*>(self);
  if (command == IREE_ALLOCATOR_COMMAND_REALLOC &&
      (state->flags & kTestAllocatorFlagFailRealloc)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "test allocator rejected realloc");
  }
  return state->base_allocator.ctl(state->base_allocator.self, command, params,
                                   inout_ptr);
}

static iree_allocator_t TestAllocator(TestAllocatorState* state) {
  iree_allocator_t allocator = {};
  allocator.self = state;
  allocator.ctl = TestAllocatorCtl;
  return allocator;
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST(TLSFTest, InitializeAndDeinitialize) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  // After init, the entire range should be free.
  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  EXPECT_EQ(stats.bytes_allocated, 0u);
  EXPECT_EQ(stats.bytes_free, 1024u * 1024u);
  EXPECT_EQ(stats.allocation_count, 0u);
  EXPECT_EQ(stats.free_block_count, 1u);
  EXPECT_EQ(stats.tainted_coalesce_count, 0u);

  // Largest free block should be the whole range.
  EXPECT_EQ(iree_hal_memory_tlsf_largest_free_block(&tlsf), 1024u * 1024u);

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, InitializeZeroRangeFails) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.range_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));
}

TEST(TLSFTest, InitializeNonPowerOfTwoAlignmentFails) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.alignment = 17;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));
}

TEST(TLSFTest, InitializeAlignmentBelowMinFails) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.alignment = 8;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));
}

TEST(TLSFTest, InitializeRangeSmallerThanAlignmentFails) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.range_length = 8;
  options.alignment = 16;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));
}

TEST(TLSFTest, InitializeDefaultAlignment) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.alignment = 0;  // Should use default (16).
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, InitializeDefaultFrontierCapacity) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.frontier_capacity = 0;  // Should use default (8).
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, InitializeRangeRoundedDownToAlignment) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.range_length = 1000;  // Not a multiple of 16.
  options.alignment = 16;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  // 1000 rounded down to multiple of 16 = 992.
  EXPECT_EQ(stats.bytes_free, 992u);

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

//===----------------------------------------------------------------------===//
// Allocation and coalescing
//===----------------------------------------------------------------------===//

TEST(TLSFTest, AllocateSingleBlock) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));

  EXPECT_EQ(alloc.offset, 0u);
  EXPECT_EQ(alloc.length, 256u);
  EXPECT_NE(alloc.block_index, IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE);
  // Initial free block has no frontier.
  EXPECT_EQ(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED, 0u);

  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  EXPECT_EQ(stats.bytes_allocated, 256u);
  EXPECT_EQ(stats.bytes_free, 1024u * 1024u - 256u);
  EXPECT_EQ(stats.allocation_count, 1u);

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, /*death_frontier=*/NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, AllocateMultipleBlocks) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc1, alloc2, alloc3;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 512, &alloc2));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 128, &alloc3));

  // All offsets should be non-overlapping and sequentially laid out.
  EXPECT_EQ(alloc1.offset, 0u);
  EXPECT_EQ(alloc2.offset, 256u);
  EXPECT_EQ(alloc3.offset, 768u);

  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  EXPECT_EQ(stats.bytes_allocated, 256u + 512u + 128u);
  EXPECT_EQ(stats.allocation_count, 3u);

  iree_hal_memory_tlsf_free(&tlsf, alloc3.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, AllocateZeroLengthFails) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_tlsf_allocate(&tlsf, 0, &alloc));

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, AllocateExceedingRangeFails) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_memory_tlsf_allocate(&tlsf, 2 * 1024 * 1024, &alloc));

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, TryAllocateExhaustionReturnsResult) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  iree_hal_memory_tlsf_allocate_result_t result =
      IREE_HAL_MEMORY_TLSF_ALLOCATE_OK;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_try_allocate(&tlsf, 2 * 1024 * 1024,
                                                   &alloc, &result));
  EXPECT_EQ(result, IREE_HAL_MEMORY_TLSF_ALLOCATE_EXHAUSTED);

  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  EXPECT_EQ(stats.allocation_count, 0u);

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, AllocateNearSizeMaxFails) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  // A request near SIZE_MAX would wrap to a small value when aligned. The
  // overflow guard should catch this and return OUT_OF_RANGE.
  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_memory_tlsf_allocate(&tlsf, IREE_DEVICE_SIZE_MAX, &alloc));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_memory_tlsf_allocate(&tlsf, IREE_DEVICE_SIZE_MAX - 1, &alloc));
  // One byte below the overflow threshold should still be caught (alignment=16,
  // so any value > SIZE_MAX - 15 overflows when rounding up).
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_memory_tlsf_allocate(&tlsf, IREE_DEVICE_SIZE_MAX - 14, &alloc));

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, AllocationResultHasNoFreeFlag) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));

  // The block_flags in the allocation result must never include FREE. The
  // block was free before allocation, but the caller should see the
  // post-allocation state.
  EXPECT_EQ(alloc.block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE, 0u);

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, AllocateRoundsUpToAlignment) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.alignment = 64;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 1, &alloc));

  // Should be rounded up to alignment (64).
  EXPECT_EQ(alloc.offset, 0u);
  EXPECT_EQ(alloc.length, 64u);
  EXPECT_EQ(alloc.offset % 64, 0u);

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, AllocateOffsetsAreAligned) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.alignment = 256;
  options.range_length = 4096;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc1, alloc2;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));

  EXPECT_EQ(alloc1.offset % 256, 0u);
  EXPECT_EQ(alloc2.offset % 256, 0u);

  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, CoalesceRight) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  // Allocate two adjacent blocks.
  iree_hal_memory_tlsf_allocation_t alloc1, alloc2;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));

  // Free the right block first, then the left block. The left block should
  // coalesce with the right.
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, NULL);

  // After both frees, the entire range should be one free block again.
  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  EXPECT_EQ(stats.free_block_count, 1u);
  EXPECT_EQ(stats.bytes_free, 1024u * 1024u);
  EXPECT_EQ(iree_hal_memory_tlsf_largest_free_block(&tlsf), 1024u * 1024u);

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, CoalesceLeft) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc1, alloc2;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));

  // Free the left block first, then the right. The right block should
  // coalesce with the left.
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, NULL);

  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  EXPECT_EQ(stats.free_block_count, 1u);
  EXPECT_EQ(stats.bytes_free, 1024u * 1024u);

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, CoalesceBoth) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc1, alloc2, alloc3;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc3));

  // Free left and right neighbors, leaving the middle allocated.
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, alloc3.block_index, NULL);

  {
    iree_hal_memory_tlsf_stats_t stats;
    iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
    EXPECT_EQ(stats.allocation_count, 1u);
    // Two free blocks: [0,256) and [512, end).
    EXPECT_EQ(stats.free_block_count, 2u);
  }

  // Free the middle; should coalesce with both neighbors.
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, NULL);

  {
    iree_hal_memory_tlsf_stats_t stats;
    iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
    EXPECT_EQ(stats.free_block_count, 1u);
    EXPECT_EQ(stats.bytes_free, 1024u * 1024u);
  }

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, NoCoalesceWithAllocatedNeighbors) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc1, alloc2, alloc3;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc3));

  // Free only the middle block; both neighbors are allocated.
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, NULL);

  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  // Two free blocks: the middle one and the remainder at the end.
  EXPECT_EQ(stats.free_block_count, 2u);
  EXPECT_EQ(stats.allocation_count, 2u);

  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, alloc3.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, ReuseFreedBlock) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc1, alloc2;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));

  // Free the first, allocate same size; should reuse the freed block's range.
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, NULL);

  iree_hal_memory_tlsf_allocation_t alloc3;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc3));
  EXPECT_EQ(alloc3.offset, 0u);
  EXPECT_EQ(alloc3.length, 256u);

  iree_hal_memory_tlsf_free(&tlsf, alloc3.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

//===----------------------------------------------------------------------===//
// Death frontiers
//===----------------------------------------------------------------------===//

TEST(TLSFTest, FrontierPreservedAcrossAllocFree) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  // Allocate a block.
  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));

  // Free with a non-empty frontier.
  MAKE_FRONTIER(death, 2, E(TestQueueAxis(0), 10), E(TestQueueAxis(1), 20));
  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, death);

  // Allocate the same region; the death frontier should be returned.
  iree_hal_memory_tlsf_allocation_t realloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &realloc));
  ASSERT_NE(realloc.death_frontier, nullptr);
  EXPECT_EQ(realloc.death_frontier->entry_count, 2);
  EXPECT_EQ(realloc.death_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(realloc.death_frontier->entries[0].epoch, 10u);
  EXPECT_EQ(realloc.death_frontier->entries[1].axis, TestQueueAxis(1));
  EXPECT_EQ(realloc.death_frontier->entries[1].epoch, 20u);

  iree_hal_memory_tlsf_free(&tlsf, realloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, NullFrontierReturnsNullOnAlloc) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));
  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);

  // Re-allocate; frontier should be NULL (block was freed with no frontier).
  iree_hal_memory_tlsf_allocation_t realloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &realloc));
  EXPECT_EQ(realloc.death_frontier, nullptr);

  iree_hal_memory_tlsf_free(&tlsf, realloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, RestorePreservesFrontierMetadata) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));
  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 42));
  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, death);

  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entry_count, 1);
  EXPECT_EQ(alloc.death_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(alloc.death_frontier->entries[0].epoch, 42u);

  iree_hal_memory_tlsf_restore(&tlsf, alloc.block_index);

  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));
  ASSERT_NE(alloc.death_frontier, nullptr);
  EXPECT_EQ(alloc.death_frontier->entry_count, 1);
  EXPECT_EQ(alloc.death_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(alloc.death_frontier->entries[0].epoch, 42u);

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, FrontierMergeOnCoalesce) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  // Allocate three adjacent blocks.
  iree_hal_memory_tlsf_allocation_t alloc1, alloc2, alloc3;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc3));

  // Free the first two with different frontiers (same axis, different epochs).
  MAKE_FRONTIER(f1, 1, E(TestQueueAxis(0), 5));
  MAKE_FRONTIER(f2, 1, E(TestQueueAxis(0), 10));
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, f1);
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, f2);

  // Blocks should have coalesced. Re-allocate the combined range.
  iree_hal_memory_tlsf_allocation_t realloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 512, &realloc));
  EXPECT_EQ(realloc.offset, 0u);
  EXPECT_EQ(realloc.length, 512u);

  // The merged frontier should have the max epoch for the shared axis.
  ASSERT_NE(realloc.death_frontier, nullptr);
  EXPECT_EQ(realloc.death_frontier->entry_count, 1);
  EXPECT_EQ(realloc.death_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(realloc.death_frontier->entries[0].epoch, 10u);

  iree_hal_memory_tlsf_free(&tlsf, realloc.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, alloc3.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, FrontierMergeMultipleAxes) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc1, alloc2, alloc3;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc3));

  // Free with frontiers on different axes; after coalescing, the merged
  // frontier should contain both axes.
  MAKE_FRONTIER(f1, 1, E(TestQueueAxis(0), 5));
  MAKE_FRONTIER(f2, 1, E(TestQueueAxis(1), 10));
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, f1);
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, f2);

  iree_hal_memory_tlsf_allocation_t realloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 512, &realloc));
  ASSERT_NE(realloc.death_frontier, nullptr);
  EXPECT_EQ(realloc.death_frontier->entry_count, 2);

  iree_hal_memory_tlsf_free(&tlsf, realloc.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, alloc3.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, TaintOnFrontierOverflow) {
  // Create a TLSF with capacity=1 frontier entries per block. Two blocks
  // freed with different axes will overflow on coalesce.
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.frontier_capacity = 1;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc1, alloc2, alloc3;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc3));

  // Free with different axes; merge will need 2 entries but capacity is 1.
  MAKE_FRONTIER(f1, 1, E(TestQueueAxis(0), 5));
  MAKE_FRONTIER(f2, 1, E(TestQueueAxis(1), 10));
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, f1);
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, f2);

  // The coalesced block should be tainted.
  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  EXPECT_GT(stats.tainted_coalesce_count, 0u);

  // Re-allocate and check the taint flag.
  iree_hal_memory_tlsf_allocation_t realloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 512, &realloc));
  EXPECT_NE(realloc.block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED, 0u);
  // Tainted blocks have NULL frontiers (meaningless data).
  EXPECT_EQ(realloc.death_frontier, nullptr);

  iree_hal_memory_tlsf_free(&tlsf, realloc.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, alloc3.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, RestorePreservesTaint) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.range_length = 256;
  options.frontier_capacity = 1;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));
  MAKE_FRONTIER(oversized, 2, E(TestQueueAxis(0), 1), E(TestQueueAxis(1), 2));
  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, oversized);

  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));
  EXPECT_TRUE(alloc.block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED);
  EXPECT_EQ(alloc.death_frontier, nullptr);

  iree_hal_memory_tlsf_restore(&tlsf, alloc.block_index);

  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));
  EXPECT_TRUE(alloc.block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED);
  EXPECT_EQ(alloc.death_frontier, nullptr);

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, TaintPropagatesThroughRightCoalesce) {
  // When a tainted block is the right neighbor during coalescing, the
  // surviving left block must inherit the taint flag.
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.frontier_capacity = 1;
  options.range_length = 1024;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  // Allocate three adjacent blocks.
  iree_hal_memory_tlsf_allocation_t alloc1, alloc2, alloc3;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc3));

  // Create a tainted free block by freeing alloc2 and alloc3 with different
  // axes (capacity=1 causes overflow → taint on coalesce).
  MAKE_FRONTIER(f2, 1, E(TestQueueAxis(0), 5));
  MAKE_FRONTIER(f3, 1, E(TestQueueAxis(1), 10));
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, f2);
  iree_hal_memory_tlsf_free(&tlsf, alloc3.block_index, f3);

  // Now the coalesced block [256..1024) is tainted and sits to the right of
  // alloc1. Free alloc1; it should coalesce right with the tainted block
  // and the resulting block should also be tainted.
  MAKE_FRONTIER(f1, 1, E(TestQueueAxis(0), 100));
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, f1);

  // The entire range is now one free block. Allocate it and verify taint.
  iree_hal_memory_tlsf_allocation_t realloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 512, &realloc));
  EXPECT_NE(realloc.block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED, 0u);
  // Tainted frontier should be NULL (zeroed by taint propagation).
  EXPECT_EQ(realloc.death_frontier, nullptr);

  iree_hal_memory_tlsf_free(&tlsf, realloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, TaintPropagatesThroughLeftCoalesce) {
  // Symmetric case: when a tainted block is the left neighbor during
  // coalescing, the surviving block must inherit taint and zero its frontier.
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.frontier_capacity = 1;
  options.range_length = 1024;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc1, alloc2, alloc3;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc3));

  // Create a tainted free block by freeing alloc1 and alloc2 with different
  // axes (capacity=1 causes overflow → taint on coalesce).
  MAKE_FRONTIER(f1, 1, E(TestQueueAxis(0), 5));
  MAKE_FRONTIER(f2, 1, E(TestQueueAxis(1), 10));
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, f1);
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, f2);

  // Now the coalesced block [0..512) is tainted and sits to the left of
  // alloc3. Free alloc3; it should coalesce left with the tainted block
  // and the resulting block should also be tainted.
  MAKE_FRONTIER(f3, 1, E(TestQueueAxis(0), 100));
  iree_hal_memory_tlsf_free(&tlsf, alloc3.block_index, f3);

  iree_hal_memory_tlsf_allocation_t realloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 512, &realloc));
  EXPECT_NE(realloc.block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED, 0u);
  EXPECT_EQ(realloc.death_frontier, nullptr);

  iree_hal_memory_tlsf_free(&tlsf, realloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, TaintClearedOnReuse) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.frontier_capacity = 1;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  // Create a tainted block (same as above).
  iree_hal_memory_tlsf_allocation_t alloc1, alloc2;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc1));
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc2));

  MAKE_FRONTIER(f1, 1, E(TestQueueAxis(0), 5));
  MAKE_FRONTIER(f2, 1, E(TestQueueAxis(1), 10));
  iree_hal_memory_tlsf_free(&tlsf, alloc1.block_index, f1);
  iree_hal_memory_tlsf_free(&tlsf, alloc2.block_index, f2);

  // Allocate the tainted block.
  iree_hal_memory_tlsf_allocation_t tainted_alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 512, &tainted_alloc));
  EXPECT_NE(tainted_alloc.block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED,
            0u);

  // Free with a fresh frontier and re-allocate; taint should be cleared.
  MAKE_FRONTIER(fresh, 1, E(TestQueueAxis(0), 100));
  iree_hal_memory_tlsf_free(&tlsf, tainted_alloc.block_index, fresh);

  iree_hal_memory_tlsf_allocation_t clean_alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 512, &clean_alloc));
  EXPECT_EQ(clean_alloc.block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED,
            0u);
  ASSERT_NE(clean_alloc.death_frontier, nullptr);
  EXPECT_EQ(clean_alloc.death_frontier->entries[0].epoch, 100u);

  iree_hal_memory_tlsf_free(&tlsf, clean_alloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, TaintOnOversizedDeathFrontier) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.frontier_capacity = 1;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));

  // Free with a frontier that has more entries than capacity allows.
  MAKE_FRONTIER(big, 2, E(TestQueueAxis(0), 5), E(TestQueueAxis(1), 10));
  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, big);

  // Re-allocate; should be tainted because the death frontier was too large.
  iree_hal_memory_tlsf_allocation_t realloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &realloc));
  EXPECT_NE(realloc.block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED, 0u);

  iree_hal_memory_tlsf_free(&tlsf, realloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, QueryBlockFlags) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));

  // Allocated block should not have FREE flag.
  iree_hal_memory_tlsf_block_flags_t flags =
      iree_hal_memory_tlsf_block_flags(&tlsf, alloc.block_index);
  EXPECT_EQ(flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE, 0u);

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, QueryBlockDeathFrontier) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      DefaultOptions(), iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));

  // Allocated block from initial free range has no frontier.
  const iree_async_frontier_t* f =
      iree_hal_memory_tlsf_block_death_frontier(&tlsf, alloc.block_index);
  EXPECT_EQ(f, nullptr);

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, PoolGrowsOnDemand) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.initial_block_capacity = 4;  // Very small pool.
  options.range_length = 4096;
  options.alignment = 16;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  // Allocate more blocks than the initial pool capacity. Each allocation
  // may split the free block, consuming a node from the pool. With capacity=4
  // and the initial block using 1, we need more than 3 splits to trigger
  // growth.
  std::vector<iree_hal_memory_tlsf_allocation_t> allocs;
  for (int i = 0; i < 20; ++i) {
    iree_hal_memory_tlsf_allocation_t alloc;
    IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 16, &alloc));
    allocs.push_back(alloc);
  }

  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  EXPECT_EQ(stats.allocation_count, 20u);

  for (auto& alloc : allocs) {
    iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  }
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, SplitMetadataGrowthFailureDoesNotMutateAllocator) {
  TestAllocatorState allocator_state = {};
  allocator_state.base_allocator = iree_allocator_system();
  allocator_state.flags = kTestAllocatorFlagFailRealloc;

  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.initial_block_capacity = 1;
  options.range_length = 4096;
  options.alignment = 16;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_initialize(
      options, TestAllocator(&allocator_state), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  iree_hal_memory_tlsf_allocate_result_t result =
      IREE_HAL_MEMORY_TLSF_ALLOCATE_EXHAUSTED;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_memory_tlsf_try_allocate(&tlsf, 16, &alloc, &result));
  EXPECT_EQ(result, IREE_HAL_MEMORY_TLSF_ALLOCATE_EXHAUSTED);

  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  EXPECT_EQ(stats.bytes_allocated, 0u);
  EXPECT_EQ(stats.bytes_free, 4096u);
  EXPECT_EQ(stats.allocation_count, 0u);
  EXPECT_EQ(stats.free_block_count, 1u);
  EXPECT_EQ(iree_hal_memory_tlsf_largest_free_block(&tlsf), 4096u);

  allocator_state.flags = 0;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_try_allocate(&tlsf, 16, &alloc, &result));
  ASSERT_EQ(result, IREE_HAL_MEMORY_TLSF_ALLOCATE_OK);
  EXPECT_EQ(alloc.length, 16u);

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, ExhaustionAfterFragmentation) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.range_length = 256;
  options.alignment = 16;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  // Fill the entire range with 16-byte blocks.
  std::vector<iree_hal_memory_tlsf_allocation_t> allocs;
  for (int i = 0; i < 16; ++i) {
    iree_hal_memory_tlsf_allocation_t alloc;
    IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 16, &alloc));
    allocs.push_back(alloc);
  }

  // Range is fully allocated; next allocation should fail.
  iree_hal_memory_tlsf_allocation_t overflow;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_memory_tlsf_allocate(&tlsf, 16, &overflow));

  // Free every other block; creates 8 x 16-byte holes.
  for (int i = 0; i < 16; i += 2) {
    iree_hal_memory_tlsf_free(&tlsf, allocs[i].block_index, NULL);
  }

  // 128 bytes free but fragmented. 32-byte allocation should fail
  // (no contiguous 32-byte block exists).
  iree_hal_memory_tlsf_allocation_t frag;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_memory_tlsf_allocate(&tlsf, 32, &frag));

  // But 16-byte allocation should succeed.
  iree_hal_memory_tlsf_allocation_t small;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 16, &small));
  allocs.push_back(small);

  // Clean up: free remaining.
  for (int i = 1; i < 16; i += 2) {
    iree_hal_memory_tlsf_free(&tlsf, allocs[i].block_index, NULL);
  }
  iree_hal_memory_tlsf_free(&tlsf, small.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, AllocateEntireRange) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.range_length = 4096;
  options.alignment = 16;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  // Allocate the entire range in one shot.
  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 4096, &alloc));
  EXPECT_EQ(alloc.offset, 0u);
  EXPECT_EQ(alloc.length, 4096u);

  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  EXPECT_EQ(stats.bytes_allocated, 4096u);
  EXPECT_EQ(stats.bytes_free, 0u);
  EXPECT_EQ(stats.free_block_count, 0u);
  EXPECT_EQ(iree_hal_memory_tlsf_largest_free_block(&tlsf), 0u);

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, LargestFreeBlockTracking) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.range_length = 1024;
  options.alignment = 16;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  EXPECT_EQ(iree_hal_memory_tlsf_largest_free_block(&tlsf), 1024u);

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 256, &alloc));
  EXPECT_EQ(iree_hal_memory_tlsf_largest_free_block(&tlsf), 768u);

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  EXPECT_EQ(iree_hal_memory_tlsf_largest_free_block(&tlsf), 1024u);

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, AllocFreeCycleStress) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.range_length = 16384;
  options.alignment = 16;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  // Perform many alloc/free cycles at various sizes to exercise the bitmap
  // and coalescing logic under realistic workload patterns.
  std::vector<iree_hal_memory_tlsf_allocation_t> live;
  for (int round = 0; round < 100; ++round) {
    // Allocate a batch.
    for (int i = 0; i < 10; ++i) {
      iree_hal_memory_tlsf_allocation_t alloc;
      iree_device_size_t size = 16 * (1 + (round * 7 + i * 3) % 20);
      iree_hal_memory_tlsf_allocate_result_t result =
          IREE_HAL_MEMORY_TLSF_ALLOCATE_EXHAUSTED;
      IREE_ASSERT_OK(
          iree_hal_memory_tlsf_try_allocate(&tlsf, size, &alloc, &result));
      if (result == IREE_HAL_MEMORY_TLSF_ALLOCATE_OK) {
        live.push_back(alloc);
      }
    }
    // Free a subset (every other block).
    for (size_t i = 0; i < live.size(); i += 2) {
      if (i < live.size()) {
        iree_hal_memory_tlsf_free(&tlsf, live[i].block_index, NULL);
        live[i].block_index = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
      }
    }
    // Compact the live list.
    std::vector<iree_hal_memory_tlsf_allocation_t> remaining;
    for (auto& a : live) {
      if (a.block_index != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
        remaining.push_back(a);
      }
    }
    live = std::move(remaining);
  }

  // Free all remaining.
  for (auto& a : live) {
    iree_hal_memory_tlsf_free(&tlsf, a.block_index, NULL);
  }

  // After everything is freed, the range should be fully coalesced.
  iree_hal_memory_tlsf_stats_t stats;
  iree_hal_memory_tlsf_query_stats(&tlsf, &stats);
  EXPECT_EQ(stats.allocation_count, 0u);
  EXPECT_EQ(stats.bytes_allocated, 0u);
  EXPECT_EQ(stats.bytes_free, 16384u);
  // Should coalesce back to a single free block.
  EXPECT_EQ(stats.free_block_count, 1u);

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, LargeAlignment) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.range_length = 1024 * 1024;
  options.alignment = 4096;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 100, &alloc));
  EXPECT_EQ(alloc.offset % 4096, 0u);
  EXPECT_EQ(alloc.length, 4096u);

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

TEST(TLSFTest, MinimumRange) {
  iree_hal_memory_tlsf_t tlsf;
  auto options = DefaultOptions();
  options.range_length = 16;
  options.alignment = 16;
  IREE_ASSERT_OK(
      iree_hal_memory_tlsf_initialize(options, iree_allocator_system(), &tlsf));

  // Only one allocation possible.
  iree_hal_memory_tlsf_allocation_t alloc;
  IREE_ASSERT_OK(iree_hal_memory_tlsf_allocate(&tlsf, 16, &alloc));
  EXPECT_EQ(alloc.offset, 0u);
  EXPECT_EQ(alloc.length, 16u);

  // Second allocation fails.
  iree_hal_memory_tlsf_allocation_t overflow;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_memory_tlsf_allocate(&tlsf, 16, &overflow));

  iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}

}  // namespace
