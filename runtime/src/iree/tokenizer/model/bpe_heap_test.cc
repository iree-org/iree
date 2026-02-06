// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/model/bpe_heap.h"

#include "iree/testing/gtest.h"

namespace {

TEST(BpeHeapTest, EmptyHeap) {
  iree_tokenizer_bpe_heap_entry_t storage[16];
  iree_tokenizer_bpe_heap_t heap;
  iree_tokenizer_bpe_heap_initialize(&heap, storage, 16);

  EXPECT_TRUE(iree_tokenizer_bpe_heap_is_empty(&heap));
  EXPECT_EQ(iree_tokenizer_bpe_heap_size(&heap), 0u);
}

TEST(BpeHeapTest, PushPop) {
  iree_tokenizer_bpe_heap_entry_t storage[16];
  iree_tokenizer_bpe_heap_t heap;
  iree_tokenizer_bpe_heap_initialize(&heap, storage, 16);

  iree_tokenizer_bpe_heap_push(&heap, {10, 0});
  EXPECT_FALSE(iree_tokenizer_bpe_heap_is_empty(&heap));
  EXPECT_EQ(iree_tokenizer_bpe_heap_size(&heap), 1u);

  auto entry = iree_tokenizer_bpe_heap_pop(&heap);
  EXPECT_EQ(entry.rank, 10u);
  EXPECT_EQ(entry.left_start_byte, 0u);
  EXPECT_TRUE(iree_tokenizer_bpe_heap_is_empty(&heap));
}

TEST(BpeHeapTest, MinHeapOrdering) {
  iree_tokenizer_bpe_heap_entry_t storage[16];
  iree_tokenizer_bpe_heap_t heap;
  iree_tokenizer_bpe_heap_initialize(&heap, storage, 16);

  // Push in non-sorted order.
  iree_tokenizer_bpe_heap_push(&heap, {5, 0});
  iree_tokenizer_bpe_heap_push(&heap, {1, 1});
  iree_tokenizer_bpe_heap_push(&heap, {3, 2});
  iree_tokenizer_bpe_heap_push(&heap, {7, 3});
  iree_tokenizer_bpe_heap_push(&heap, {2, 4});

  // Pop should return in sorted order (min first).
  EXPECT_EQ(iree_tokenizer_bpe_heap_pop(&heap).rank, 1u);
  EXPECT_EQ(iree_tokenizer_bpe_heap_pop(&heap).rank, 2u);
  EXPECT_EQ(iree_tokenizer_bpe_heap_pop(&heap).rank, 3u);
  EXPECT_EQ(iree_tokenizer_bpe_heap_pop(&heap).rank, 5u);
  EXPECT_EQ(iree_tokenizer_bpe_heap_pop(&heap).rank, 7u);
  EXPECT_TRUE(iree_tokenizer_bpe_heap_is_empty(&heap));
}

TEST(BpeHeapTest, Peek) {
  iree_tokenizer_bpe_heap_entry_t storage[16];
  iree_tokenizer_bpe_heap_t heap;
  iree_tokenizer_bpe_heap_initialize(&heap, storage, 16);

  iree_tokenizer_bpe_heap_push(&heap, {5, 0});
  iree_tokenizer_bpe_heap_push(&heap, {1, 1});

  // Peek should return minimum without removing.
  auto entry = iree_tokenizer_bpe_heap_peek(&heap);
  EXPECT_EQ(entry.rank, 1u);
  EXPECT_EQ(iree_tokenizer_bpe_heap_size(&heap), 2u);

  // Peek again should return same value.
  entry = iree_tokenizer_bpe_heap_peek(&heap);
  EXPECT_EQ(entry.rank, 1u);
}

TEST(BpeHeapTest, Reset) {
  iree_tokenizer_bpe_heap_entry_t storage[16];
  iree_tokenizer_bpe_heap_t heap;
  iree_tokenizer_bpe_heap_initialize(&heap, storage, 16);

  iree_tokenizer_bpe_heap_push(&heap, {1, 0});
  iree_tokenizer_bpe_heap_push(&heap, {2, 1});
  EXPECT_EQ(iree_tokenizer_bpe_heap_size(&heap), 2u);

  iree_tokenizer_bpe_heap_reset(&heap);
  EXPECT_TRUE(iree_tokenizer_bpe_heap_is_empty(&heap));
  EXPECT_EQ(iree_tokenizer_bpe_heap_size(&heap), 0u);

  // Should be able to push again after reset.
  iree_tokenizer_bpe_heap_push(&heap, {3, 0});
  EXPECT_EQ(iree_tokenizer_bpe_heap_size(&heap), 1u);
}

TEST(BpeHeapTest, BytePositionPreserved) {
  iree_tokenizer_bpe_heap_entry_t storage[16];
  iree_tokenizer_bpe_heap_t heap;
  iree_tokenizer_bpe_heap_initialize(&heap, storage, 16);

  // Push entries with same rank but different byte positions.
  iree_tokenizer_bpe_heap_push(&heap, {1, 100});
  iree_tokenizer_bpe_heap_push(&heap, {1, 200});
  iree_tokenizer_bpe_heap_push(&heap, {1, 50});

  // All should have rank 1, byte positions should be preserved (order may
  // vary).
  auto e1 = iree_tokenizer_bpe_heap_pop(&heap);
  auto e2 = iree_tokenizer_bpe_heap_pop(&heap);
  auto e3 = iree_tokenizer_bpe_heap_pop(&heap);

  EXPECT_EQ(e1.rank, 1u);
  EXPECT_EQ(e2.rank, 1u);
  EXPECT_EQ(e3.rank, 1u);

  // Verify all byte positions were preserved (in some order).
  std::set<uint32_t> byte_positions = {e1.left_start_byte, e2.left_start_byte,
                                       e3.left_start_byte};
  EXPECT_TRUE(byte_positions.count(50));
  EXPECT_TRUE(byte_positions.count(100));
  EXPECT_TRUE(byte_positions.count(200));
}

// Test that exercises the full capacity of the heap.
// The proven bound is 3L - 3 where L = max_token_length.
// For L = 128, that's 381 entries maximum.
TEST(BpeHeapTest, FullCapacity) {
  constexpr size_t kCapacity = 384;  // 3 * 128
  std::vector<iree_tokenizer_bpe_heap_entry_t> storage(kCapacity);
  iree_tokenizer_bpe_heap_t heap;
  iree_tokenizer_bpe_heap_initialize(&heap, storage.data(), kCapacity);

  // Fill to capacity.
  for (size_t i = 0; i < kCapacity; ++i) {
    iree_tokenizer_bpe_heap_push(&heap, {static_cast<uint32_t>(i), 0});
  }
  EXPECT_EQ(iree_tokenizer_bpe_heap_size(&heap), kCapacity);

  // Pop all, verify order.
  for (size_t i = 0; i < kCapacity; ++i) {
    auto entry = iree_tokenizer_bpe_heap_pop(&heap);
    EXPECT_EQ(entry.rank, static_cast<uint32_t>(i));
  }
  EXPECT_TRUE(iree_tokenizer_bpe_heap_is_empty(&heap));
}

// Worst-case pattern: interleaved push and pop simulating apply_pending_merges.
// Each merge pops 1 entry and pushes up to 2 new entries (net +1).
// This exercises the peak = H + (W - 1) scenario.
TEST(BpeHeapTest, InterleavedPushPop) {
  constexpr size_t kCapacity = 256;
  std::vector<iree_tokenizer_bpe_heap_entry_t> storage(kCapacity);
  iree_tokenizer_bpe_heap_t heap;
  iree_tokenizer_bpe_heap_initialize(&heap, storage.data(), kCapacity);

  // Simulate: start with L entries, then do L merges each adding 2 entries.
  constexpr size_t L = 64;

  // Phase 1: Push L entries (simulating byte loop accumulation).
  for (size_t i = 0; i < L; ++i) {
    iree_tokenizer_bpe_heap_push(&heap, {static_cast<uint32_t>(i), 0});
  }
  EXPECT_EQ(iree_tokenizer_bpe_heap_size(&heap), L);

  // Phase 2: Simulate apply_pending_merges with maximum merge cascade.
  // Each iteration: pop 1, push 2 (net +1).
  uint32_t next_rank = static_cast<uint32_t>(L);
  for (size_t i = 0; i < L; ++i) {
    iree_tokenizer_bpe_heap_pop(&heap);
    iree_tokenizer_bpe_heap_push(&heap, {next_rank++, 0});
    iree_tokenizer_bpe_heap_push(&heap, {next_rank++, 0});
  }

  // Peak should be L + L = 2L entries.
  EXPECT_EQ(iree_tokenizer_bpe_heap_size(&heap), 2 * L);

  // Drain the heap.
  while (!iree_tokenizer_bpe_heap_is_empty(&heap)) {
    iree_tokenizer_bpe_heap_pop(&heap);
  }
}

// Test heap behavior after reset and reuse.
// Simulates processing multiple segments.
TEST(BpeHeapTest, MultipleResetCycles) {
  constexpr size_t kCapacity = 128;
  std::vector<iree_tokenizer_bpe_heap_entry_t> storage(kCapacity);
  iree_tokenizer_bpe_heap_t heap;
  iree_tokenizer_bpe_heap_initialize(&heap, storage.data(), kCapacity);

  for (int cycle = 0; cycle < 10; ++cycle) {
    // Fill partially.
    for (size_t i = 0; i < 50; ++i) {
      iree_tokenizer_bpe_heap_push(&heap,
                                   {static_cast<uint32_t>(cycle * 100 + i), 0});
    }
    EXPECT_EQ(iree_tokenizer_bpe_heap_size(&heap), 50u);

    // Drain completely.
    while (!iree_tokenizer_bpe_heap_is_empty(&heap)) {
      iree_tokenizer_bpe_heap_pop(&heap);
    }

    // Reset explicitly.
    iree_tokenizer_bpe_heap_reset(&heap);
    EXPECT_TRUE(iree_tokenizer_bpe_heap_is_empty(&heap));
  }
}

}  // namespace
