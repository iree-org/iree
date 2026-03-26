// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"

#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Block size used by all tests. 4KB is a reasonable default that fits several
// allocations per block and exercises the block-chaining logic.
static constexpr iree_host_size_t kBlockSize = 4096;

//===----------------------------------------------------------------------===//
// iree_arena_block_pool_t
//===----------------------------------------------------------------------===//

TEST(ArenaBlockPool, Lifetime) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  EXPECT_EQ(pool.total_block_size, kBlockSize);
  EXPECT_EQ(pool.usable_block_size, kBlockSize - sizeof(iree_arena_block_t));
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(ArenaBlockPool, Preallocate) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  IREE_ASSERT_OK(iree_arena_block_pool_preallocate(&pool, 4));
  // Preallocated blocks should be available for acquire.
  for (int i = 0; i < 4; ++i) {
    iree_arena_block_t* block = NULL;
    void* ptr = NULL;
    IREE_ASSERT_OK(iree_arena_block_pool_acquire(&pool, &block, &ptr));
    ASSERT_NE(block, nullptr);
    ASSERT_NE(ptr, nullptr);
    iree_arena_block_pool_release(&pool, block, block);
  }
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(ArenaBlockPool, AcquireRelease) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_block_t* block = NULL;
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_block_pool_acquire(&pool, &block, &ptr));
  ASSERT_NE(block, nullptr);
  ASSERT_NE(ptr, nullptr);
  // The returned pointer should be usable for the full usable_block_size.
  memset(ptr, 0xAB, pool.usable_block_size);
  iree_arena_block_pool_release(&pool, block, block);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(ArenaBlockPool, Trim) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  IREE_ASSERT_OK(iree_arena_block_pool_preallocate(&pool, 8));
  // Trim releases all free blocks back to the system allocator.
  iree_arena_block_pool_trim(&pool);
  // The pool is still usable after trimming — new acquires allocate fresh.
  iree_arena_block_t* block = NULL;
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_block_pool_acquire(&pool, &block, &ptr));
  ASSERT_NE(block, nullptr);
  iree_arena_block_pool_release(&pool, block, block);
  iree_arena_block_pool_deinitialize(&pool);
}

//===----------------------------------------------------------------------===//
// iree_arena_allocator_t
//===----------------------------------------------------------------------===//

TEST(Arena, Lifetime) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, BasicAllocation) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, 128, &ptr));
  ASSERT_NE(ptr, nullptr);
  // Returned memory should be writable.
  memset(ptr, 0xCD, 128);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, MultipleAllocations) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Allocate several chunks that fit within a single block.
  void* pointers[16] = {};
  for (int i = 0; i < 16; ++i) {
    IREE_ASSERT_OK(iree_arena_allocate(&arena, 64, &pointers[i]));
    ASSERT_NE(pointers[i], nullptr);
  }
  // All pointers should be distinct.
  for (int i = 0; i < 16; ++i) {
    for (int j = i + 1; j < 16; ++j) {
      EXPECT_NE(pointers[i], pointers[j])
          << "allocations " << i << " and " << j << " overlap";
    }
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, NaturalAlignment) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Every allocation from the arena should be naturally aligned to
  // iree_max_align_t regardless of the requested size.
  for (iree_host_size_t size = 1; size <= 37; ++size) {
    void* ptr = NULL;
    IREE_ASSERT_OK(iree_arena_allocate(&arena, size, &ptr));
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignof(max_align_t), 0u)
        << "allocation of " << size << " bytes not naturally aligned";
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, OversizedAllocation) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Allocate something larger than the block size.
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, kBlockSize * 4, &ptr));
  ASSERT_NE(ptr, nullptr);
  memset(ptr, 0xEF, kBlockSize * 4);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, Reset) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Allocate some memory and then reset. The arena should be reusable.
  for (int round = 0; round < 3; ++round) {
    void* ptr = NULL;
    IREE_ASSERT_OK(iree_arena_allocate(&arena, 256, &ptr));
    ASSERT_NE(ptr, nullptr);
    // Also allocate an oversized chunk to exercise that cleanup path.
    IREE_ASSERT_OK(iree_arena_allocate(&arena, kBlockSize * 2, &ptr));
    ASSERT_NE(ptr, nullptr);
    iree_arena_reset(&arena);
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, BlockChaining) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Allocate enough to force multiple block acquisitions. Each allocation
  // consumes more than half the usable block size, so two cannot fit in one
  // block.
  iree_host_size_t half_plus_one = pool.usable_block_size / 2 + 1;
  void* ptr1 = NULL;
  void* ptr2 = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, half_plus_one, &ptr1));
  IREE_ASSERT_OK(iree_arena_allocate(&arena, half_plus_one, &ptr2));
  ASSERT_NE(ptr1, nullptr);
  ASSERT_NE(ptr2, nullptr);
  EXPECT_NE(ptr1, ptr2);
  // Two allocations each exceeding half the block forced a second block.
  EXPECT_GE(arena.total_allocation_size, kBlockSize * 2);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

//===----------------------------------------------------------------------===//
// iree_arena_allocate_aligned
//===----------------------------------------------------------------------===//

TEST(Arena, AlignedAtOrBelowNatural) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Requesting alignment at or below iree_max_align_t should work the same as
  // a normal allocation — no extra padding.
  void* ptr = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate_aligned(&arena, 64, alignof(max_align_t), &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignof(max_align_t), 0u);

  // Sub-natural alignment.
  IREE_ASSERT_OK(iree_arena_allocate_aligned(&arena, 64, 4, &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 4, 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, AlignedAboveNatural) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Request 64-byte alignment (cache-line), which exceeds iree_max_align_t
  // on most platforms (typically 8 or 16).
  static constexpr iree_host_size_t kAlignment = 64;
  for (int i = 0; i < 8; ++i) {
    void* ptr = NULL;
    IREE_ASSERT_OK(iree_arena_allocate_aligned(&arena, 100, kAlignment, &ptr));
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % kAlignment, 0u)
        << "allocation " << i << " not aligned to " << kAlignment;
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, AlignedLargePowerOfTwo) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // 256-byte alignment with a small allocation — exercises the over-allocation
  // and pointer-forward logic with significant padding waste.
  static constexpr iree_host_size_t kAlignment = 256;
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate_aligned(&arena, 32, kAlignment, &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % kAlignment, 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, AlignedZeroLength) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Zero-length aligned allocation should succeed (returns a valid pointer
  // that happens to be aligned but has no usable bytes).
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate_aligned(&arena, 0, 64, &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

//===----------------------------------------------------------------------===//
// iree_arena_allocator (iree_allocator_t interface)
//===----------------------------------------------------------------------===//

TEST(Arena, AllocatorInterface) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  iree_allocator_t allocator = iree_arena_allocator(&arena);

  // malloc.
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_allocator_malloc(allocator, 64, &ptr));
  ASSERT_NE(ptr, nullptr);

  // calloc should zero-fill.
  void* zeroed = NULL;
  IREE_ASSERT_OK(iree_allocator_malloc(allocator, 0, &zeroed));
  // Free is a no-op (arenas don't free individual allocations).
  iree_allocator_free(allocator, ptr);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

}  // namespace
