// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/atomic_freelist.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "iree/base/internal/memory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Helper to manage slots array lifetime. Uses aligned allocation for atomics.
class SlotsArray {
 public:
  // Creates a SlotsArray of the given count, or returns an error status.
  static iree::StatusOr<SlotsArray> Create(size_t count) {
    if (count == 0) {
      return SlotsArray(nullptr, 0);
    }
    void* ptr = nullptr;
    iree_status_t status =
        iree_aligned_alloc(alignof(iree_atomic_freelist_slot_t),
                           count * sizeof(iree_atomic_freelist_slot_t), &ptr);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    return SlotsArray(static_cast<iree_atomic_freelist_slot_t*>(ptr), count);
  }

  // Move-only.
  SlotsArray(SlotsArray&& other) noexcept
      : slots_(other.slots_), count_(other.count_) {
    other.slots_ = nullptr;
    other.count_ = 0;
  }
  SlotsArray& operator=(SlotsArray&& other) noexcept {
    if (this != &other) {
      if (slots_) iree_aligned_free(slots_);
      slots_ = other.slots_;
      count_ = other.count_;
      other.slots_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }
  SlotsArray(const SlotsArray&) = delete;
  SlotsArray& operator=(const SlotsArray&) = delete;

  ~SlotsArray() {
    if (slots_) iree_aligned_free(slots_);
  }
  iree_atomic_freelist_slot_t* data() { return slots_; }
  size_t size() const { return count_; }

 private:
  SlotsArray(iree_atomic_freelist_slot_t* slots, size_t count)
      : slots_(slots), count_(count) {}

  iree_atomic_freelist_slot_t* slots_ = nullptr;
  size_t count_ = 0;
};

TEST(AtomicFreelist, EmptyInit) {
  iree_atomic_freelist_t freelist;
  IREE_ASSERT_OK_AND_ASSIGN(auto slots, SlotsArray::Create(0));
  IREE_ASSERT_OK(iree_atomic_freelist_initialize(slots.data(), 0, &freelist));

  // Empty freelist should have count 0 and pop should fail.
  EXPECT_EQ(0u, iree_atomic_freelist_count(&freelist));

  uint16_t index;
  EXPECT_FALSE(iree_atomic_freelist_try_pop(&freelist, slots.data(), &index));

  iree_atomic_freelist_deinitialize(&freelist);
}

TEST(AtomicFreelist, CountExceedsMaximum) {
  iree_atomic_freelist_t freelist;
  // Allocate 1 slot - sufficient for error path testing since validation
  // happens before slot access.
  IREE_ASSERT_OK_AND_ASSIGN(auto slots, SlotsArray::Create(1));

  // One over maximum should fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_atomic_freelist_initialize(
          slots.data(), IREE_ATOMIC_FREELIST_MAX_COUNT + 1, &freelist));

  // Way over maximum should also fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_atomic_freelist_initialize(slots.data(), 100000, &freelist));
}

TEST(AtomicFreelist, SingleEntry) {
  iree_atomic_freelist_t freelist;
  IREE_ASSERT_OK_AND_ASSIGN(auto slots, SlotsArray::Create(1));
  IREE_ASSERT_OK(iree_atomic_freelist_initialize(slots.data(), 1, &freelist));

  EXPECT_EQ(1u, iree_atomic_freelist_count(&freelist));

  // Pop the single entry.
  uint16_t index;
  EXPECT_TRUE(iree_atomic_freelist_try_pop(&freelist, slots.data(), &index));
  EXPECT_EQ(0u, index);
  EXPECT_EQ(0u, iree_atomic_freelist_count(&freelist));

  // Now empty.
  EXPECT_FALSE(iree_atomic_freelist_try_pop(&freelist, slots.data(), &index));

  // Push it back.
  iree_atomic_freelist_push(&freelist, slots.data(), 0);
  EXPECT_EQ(1u, iree_atomic_freelist_count(&freelist));

  // Pop again.
  EXPECT_TRUE(iree_atomic_freelist_try_pop(&freelist, slots.data(), &index));
  EXPECT_EQ(0u, index);

  iree_atomic_freelist_deinitialize(&freelist);
}

TEST(AtomicFreelist, MultipleEntries) {
  constexpr size_t kCount = 10;
  iree_atomic_freelist_t freelist;
  IREE_ASSERT_OK_AND_ASSIGN(auto slots, SlotsArray::Create(kCount));
  IREE_ASSERT_OK(
      iree_atomic_freelist_initialize(slots.data(), kCount, &freelist));

  EXPECT_EQ(kCount, iree_atomic_freelist_count(&freelist));

  // Pop all entries - should get indices 0-9 in LIFO order.
  // After init: head=0->1->2->...->9->EMPTY
  // Pop returns 0, then 1, then 2, etc.
  std::vector<uint16_t> popped;
  uint16_t index;
  while (iree_atomic_freelist_try_pop(&freelist, slots.data(), &index)) {
    popped.push_back(index);
  }

  EXPECT_EQ(kCount, popped.size());
  EXPECT_EQ(0u, iree_atomic_freelist_count(&freelist));

  // Should have gotten all indices.
  std::vector<uint16_t> expected(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    expected[i] = static_cast<uint16_t>(i);
  }
  std::sort(popped.begin(), popped.end());
  EXPECT_EQ(expected, popped);

  // Push them all back in reverse order.
  for (size_t i = kCount; i > 0; --i) {
    iree_atomic_freelist_push(&freelist, slots.data(),
                              static_cast<uint16_t>(i - 1));
  }
  EXPECT_EQ(kCount, iree_atomic_freelist_count(&freelist));

  // Pop again - should get all indices.
  popped.clear();
  while (iree_atomic_freelist_try_pop(&freelist, slots.data(), &index)) {
    popped.push_back(index);
  }
  EXPECT_EQ(kCount, popped.size());

  iree_atomic_freelist_deinitialize(&freelist);
}

TEST(AtomicFreelist, LIFOOrder) {
  constexpr size_t kCount = 5;
  iree_atomic_freelist_t freelist;
  IREE_ASSERT_OK_AND_ASSIGN(auto slots, SlotsArray::Create(kCount));
  IREE_ASSERT_OK(
      iree_atomic_freelist_initialize(slots.data(), kCount, &freelist));

  // Pop all.
  std::vector<uint16_t> popped;
  uint16_t index;
  while (iree_atomic_freelist_try_pop(&freelist, slots.data(), &index)) {
    popped.push_back(index);
  }

  // Push in order: 4, 3, 2, 1, 0.
  for (auto it = popped.rbegin(); it != popped.rend(); ++it) {
    iree_atomic_freelist_push(&freelist, slots.data(), *it);
  }

  // Pop should return them in LIFO order: 0, 1, 2, 3, 4 (reverse of push).
  std::vector<uint16_t> repopped;
  while (iree_atomic_freelist_try_pop(&freelist, slots.data(), &index)) {
    repopped.push_back(index);
  }

  // The last pushed (0) should be first popped.
  EXPECT_EQ(5u, repopped.size());
  EXPECT_EQ(0u, repopped[0]);
  EXPECT_EQ(1u, repopped[1]);
  EXPECT_EQ(2u, repopped[2]);
  EXPECT_EQ(3u, repopped[3]);
  EXPECT_EQ(4u, repopped[4]);

  iree_atomic_freelist_deinitialize(&freelist);
}

// Concurrent test: multiple threads acquiring and releasing.
TEST(AtomicFreelist, ConcurrentAcquireRelease) {
  constexpr size_t kCount = 1000;
  constexpr size_t kThreads = 8;
  constexpr size_t kIterations = 10000;

  iree_atomic_freelist_t freelist;
  IREE_ASSERT_OK_AND_ASSIGN(auto slots, SlotsArray::Create(kCount));
  IREE_ASSERT_OK(
      iree_atomic_freelist_initialize(slots.data(), kCount, &freelist));

  std::atomic<uint64_t> total_acquired{0};
  std::atomic<uint64_t> total_released{0};

  auto worker = [&](int thread_id) {
    uint64_t acquired = 0;
    uint64_t released = 0;
    std::vector<uint16_t> held;
    held.reserve(100);

    for (size_t i = 0; i < kIterations; ++i) {
      // Try to acquire some.
      uint16_t index;
      while (held.size() < 50 &&
             iree_atomic_freelist_try_pop(&freelist, slots.data(), &index)) {
        held.push_back(index);
        ++acquired;
      }

      // Release some.
      while (held.size() > 25) {
        iree_atomic_freelist_push(&freelist, slots.data(), held.back());
        held.pop_back();
        ++released;
      }
    }

    // Release all held at end.
    for (uint16_t idx : held) {
      iree_atomic_freelist_push(&freelist, slots.data(), idx);
      ++released;
    }

    total_acquired.fetch_add(acquired, std::memory_order_relaxed);
    total_released.fetch_add(released, std::memory_order_relaxed);
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < kThreads; ++i) {
    threads.emplace_back(worker, static_cast<int>(i));
  }
  for (auto& thread : threads) {
    thread.join();
  }

  // All indices should be back in the freelist.
  EXPECT_EQ(kCount, iree_atomic_freelist_count(&freelist));
  EXPECT_EQ(total_acquired.load(), total_released.load());

  // Verify we can pop all indices and they're all valid.
  std::vector<bool> seen(kCount, false);
  uint16_t index;
  size_t count = 0;
  while (iree_atomic_freelist_try_pop(&freelist, slots.data(), &index)) {
    ASSERT_LT(index, kCount) << "Invalid index returned: " << index;
    ASSERT_FALSE(seen[index]) << "Duplicate index returned: " << index;
    seen[index] = true;
    ++count;
  }
  EXPECT_EQ(kCount, count);

  iree_atomic_freelist_deinitialize(&freelist);
}

// Stress test: high contention with many threads.
TEST(AtomicFreelist, HighContention) {
  constexpr size_t kCount = 64;  // Small pool = high contention.
  constexpr size_t kThreads = 16;
  constexpr size_t kIterations = 100000;

  iree_atomic_freelist_t freelist;
  IREE_ASSERT_OK_AND_ASSIGN(auto slots, SlotsArray::Create(kCount));
  IREE_ASSERT_OK(
      iree_atomic_freelist_initialize(slots.data(), kCount, &freelist));

  std::atomic<uint64_t> operations{0};

  auto worker = [&]() {
    uint64_t ops = 0;
    for (size_t i = 0; i < kIterations; ++i) {
      uint16_t index;
      if (iree_atomic_freelist_try_pop(&freelist, slots.data(), &index)) {
        ++ops;
        // Immediately release back.
        iree_atomic_freelist_push(&freelist, slots.data(), index);
        ++ops;
      }
    }
    operations.fetch_add(ops, std::memory_order_relaxed);
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < kThreads; ++i) {
    threads.emplace_back(worker);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  // All indices should be back.
  EXPECT_EQ(kCount, iree_atomic_freelist_count(&freelist));
  EXPECT_GT(operations.load(), 0u);

  iree_atomic_freelist_deinitialize(&freelist);
}

TEST(AtomicFreelist, MaxCount) {
  // Test with maximum supported count (but not all 65534 for test speed).
  constexpr size_t kCount = 1000;
  iree_atomic_freelist_t freelist;
  IREE_ASSERT_OK_AND_ASSIGN(auto slots, SlotsArray::Create(kCount));
  IREE_ASSERT_OK(
      iree_atomic_freelist_initialize(slots.data(), kCount, &freelist));

  EXPECT_EQ(kCount, iree_atomic_freelist_count(&freelist));

  // Pop all.
  size_t count = 0;
  uint16_t index;
  while (iree_atomic_freelist_try_pop(&freelist, slots.data(), &index)) {
    ++count;
  }
  EXPECT_EQ(kCount, count);
  EXPECT_EQ(0u, iree_atomic_freelist_count(&freelist));

  iree_atomic_freelist_deinitialize(&freelist);
}

}  // namespace
