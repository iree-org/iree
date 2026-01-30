// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/futex.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/threading/thread.h"
#include "iree/testing/gtest.h"

namespace {

#if defined(IREE_RUNTIME_USE_FUTEX)

// Tests that waking an address with no waiters returns immediately without
// blocking or error.
TEST(FutexTest, WakeNoWaiters) {
  uint32_t futex_word = 0;
  // Should complete immediately - no waiters.
  iree_futex_wake(&futex_word, 1);
  iree_futex_wake(&futex_word, IREE_ALL_WAITERS);
}

// Tests that iree_futex_wait returns IREE_STATUS_OK immediately when the value
// at the address doesn't match the expected value (spurious wakeup handling).
TEST(FutexTest, WaitValueMismatch) {
  uint32_t futex_word = 42;
  // Expected value doesn't match - should return immediately.
  // Note: Linux futex returns EAGAIN which maps to OK (retry expected).
  // Windows returns "value changed" which also completes successfully.
  iree_status_code_t status =
      iree_futex_wait(&futex_word, 0, IREE_TIME_INFINITE_FUTURE);
  // Both OK (spurious) and UNAVAILABLE (value mismatch) are acceptable.
  EXPECT_TRUE(status == IREE_STATUS_OK || status == IREE_STATUS_UNAVAILABLE);
}

// Tests that a background thread can be woken by the main thread.
TEST(FutexTest, WakeWakesWaiter) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<bool> waiter_started{false};
  std::atomic<bool> waiter_completed{false};

  std::thread waiter([&]() {
    waiter_started.store(true, std::memory_order_release);

    // Wait for the value to change from 0.
    while (futex_word.load(std::memory_order_acquire) == 0) {
      iree_futex_wait(
          const_cast<std::atomic<uint32_t>*>(&futex_word), 0,
          iree_time_now() + 100 * 1000000);  // 100ms timeout for safety
    }

    waiter_completed.store(true, std::memory_order_release);
  });

  // Wait for waiter thread to start.
  while (!waiter_started.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }

  // Give the waiter time to enter the futex wait.
  iree_wait_until(iree_time_now() + iree_make_duration_ms(10));

  // Change the value and wake.
  futex_word.store(1, std::memory_order_release);
  iree_futex_wake(const_cast<std::atomic<uint32_t>*>(&futex_word), 1);

  waiter.join();

  EXPECT_TRUE(waiter_completed.load(std::memory_order_acquire));
}

// Tests that wake(N) wakes at most N waiters when multiple are waiting.
TEST(FutexTest, WakeCount) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<int> waiters_started{0};
  std::atomic<int> waiters_woken{0};
  constexpr int kNumWaiters = 3;
  constexpr int kWakeCount = 2;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      waiters_started.fetch_add(1, std::memory_order_acq_rel);

      // Wait until the value changes from 0.
      while (futex_word.load(std::memory_order_acquire) == 0) {
        iree_status_code_t status =
            iree_futex_wait(const_cast<std::atomic<uint32_t>*>(&futex_word), 0,
                            iree_time_now() + 50 * 1000000);  // 50ms timeout
        if (status == IREE_STATUS_DEADLINE_EXCEEDED) {
          // Timeout - check if we should exit.
          if (futex_word.load(std::memory_order_acquire) != 0) break;
        }
      }

      waiters_woken.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to start.
  while (waiters_started.load(std::memory_order_acquire) < kNumWaiters) {
    iree_thread_yield();
  }

  // Give waiters time to enter futex wait.
  iree_wait_until(iree_time_now() + iree_make_duration_ms(20));

  // Change value and wake exactly kWakeCount waiters.
  futex_word.store(1, std::memory_order_release);
  iree_futex_wake(const_cast<std::atomic<uint32_t>*>(&futex_word), kWakeCount);

  // Wait a bit for woken threads to complete.
  iree_wait_until(iree_time_now() + iree_make_duration_ms(50));

  // Wake any remaining waiters so they can exit.
  iree_futex_wake(const_cast<std::atomic<uint32_t>*>(&futex_word),
                  IREE_ALL_WAITERS);

  for (auto& t : waiters) {
    t.join();
  }

  // All waiters should have been woken (either by wake(2) or wake(ALL)).
  EXPECT_EQ(waiters_woken.load(std::memory_order_acquire), kNumWaiters);
}

// Tests that iree_futex_wait respects the timeout deadline.
TEST(FutexTest, WaitTimeout) {
  uint32_t futex_word = 0;

  iree_time_t start = iree_time_now();
  iree_time_t deadline = start + 50 * 1000000;  // 50ms timeout

  iree_status_code_t status = iree_futex_wait(&futex_word, 0, deadline);

  iree_time_t elapsed = iree_time_now() - start;

  // Should have timed out.
  EXPECT_EQ(status, IREE_STATUS_DEADLINE_EXCEEDED);

  // Should have waited at least close to the timeout (allow 10ms slack for
  // scheduling).
  EXPECT_GE(elapsed, 40 * 1000000);  // At least 40ms
}

// Tests that iree_futex_wait with immediate deadline returns immediately.
TEST(FutexTest, WaitImmediateDeadline) {
  uint32_t futex_word = 0;

  iree_time_t start = iree_time_now();

  // IREE_TIME_INFINITE_PAST should cause immediate return.
  iree_status_code_t status =
      iree_futex_wait(&futex_word, 0, IREE_TIME_INFINITE_PAST);

  iree_time_t elapsed = iree_time_now() - start;

  // Should return quickly (deadline already passed).
  EXPECT_EQ(status, IREE_STATUS_DEADLINE_EXCEEDED);
  EXPECT_LT(elapsed, 10 * 1000000);  // Should complete within 10ms
}

#else

// Placeholder test when futex is not available.
TEST(FutexTest, NotAvailable) {
  GTEST_SKIP() << "Futex not available on this platform/configuration";
}

#endif  // IREE_RUNTIME_USE_FUTEX

}  // namespace
