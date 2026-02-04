// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/notification.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

//==============================================================================
// Basic notification tests
//==============================================================================

TEST(NotificationTest, Lifetime) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, PostNoWaiters) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  // Should not block or crash when there are no waiters.
  iree_notification_post(&notification, 1);
  iree_notification_post(&notification, IREE_ALL_WAITERS);

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, TimeoutImmediate) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  iree_time_t start_ns = iree_time_now();

  EXPECT_FALSE(iree_notification_await(
      &notification,
      +[](void* entry_arg) -> bool {
        return false;  // Condition is never true.
      },
      NULL, iree_immediate_timeout()));

  iree_duration_t delta_ns = iree_time_now() - start_ns;
  iree_duration_t delta_ms = delta_ns / 1000000;
  EXPECT_LT(delta_ms, 50);  // Should return quickly.

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, Timeout) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  iree_time_t start_ns = iree_time_now();

  EXPECT_FALSE(iree_notification_await(
      &notification,
      +[](void* entry_arg) -> bool {
        return false;  // Condition is never true.
      },
      NULL, iree_make_timeout_ms(100)));

  iree_duration_t delta_ns = iree_time_now() - start_ns;
  iree_duration_t delta_ms = delta_ns / 1000000;
  EXPECT_GE(delta_ms, 50);  // Should wait at least some time.

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, AwaitConditionAlreadyTrue) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  // Condition already true - should return immediately.
  bool result = iree_notification_await(
      &notification, +[](void*) { return true; }, nullptr,
      iree_immediate_timeout());
  EXPECT_TRUE(result);

  iree_notification_deinitialize(&notification);
}

//==============================================================================
// Post and await tests
//==============================================================================

TEST(NotificationTest, PostAndAwait) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::atomic<bool> ready{false};
  std::atomic<bool> waiter_started{false};

  std::thread waiter([&]() {
    waiter_started.store(true, std::memory_order_release);
    iree_notification_await(
        &notification,
        +[](void* arg) {
          return static_cast<std::atomic<bool>*>(arg)->load(
              std::memory_order_acquire);
        },
        &ready, iree_infinite_timeout());
  });

  // Wait for waiter thread to start (not necessarily in kernel wait, but that's
  // fine - the condition check in await handles the race).
  while (!waiter_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  ready.store(true, std::memory_order_release);
  iree_notification_post(&notification, IREE_ALL_WAITERS);
  waiter.join();

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, MultipleWaiters) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::atomic<bool> ready{false};
  std::atomic<int> started_count{0};
  std::atomic<int> woken_count{0};
  constexpr int kNumWaiters = 4;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      started_count.fetch_add(1, std::memory_order_release);
      iree_notification_await(
          &notification,
          +[](void* arg) {
            return static_cast<std::atomic<bool>*>(arg)->load(
                std::memory_order_acquire);
          },
          &ready, iree_infinite_timeout());
      woken_count.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to start.
  while (started_count.load(std::memory_order_acquire) < kNumWaiters) {
    std::this_thread::yield();
  }

  ready.store(true, std::memory_order_release);
  iree_notification_post(&notification, IREE_ALL_WAITERS);

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(kNumWaiters, woken_count.load(std::memory_order_acquire));

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, PostSingleWaiter) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::atomic<int> phase{0};
  std::atomic<int> started_count{0};
  std::atomic<int> woken_count{0};
  constexpr int kNumWaiters = 4;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      started_count.fetch_add(1, std::memory_order_release);
      // Wait until phase >= 1.
      iree_notification_await(
          &notification,
          +[](void* arg) {
            return static_cast<std::atomic<int>*>(arg)->load(
                       std::memory_order_acquire) >= 1;
          },
          &phase, iree_infinite_timeout());
      woken_count.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to start.
  while (started_count.load(std::memory_order_acquire) < kNumWaiters) {
    std::this_thread::yield();
  }

  // Post to wake just one waiter.
  phase.store(1, std::memory_order_release);
  iree_notification_post(&notification, 1);

  // Wait for at least one waiter to wake.
  while (woken_count.load(std::memory_order_acquire) < 1) {
    std::this_thread::yield();
  }

  int initial_woken = woken_count.load(std::memory_order_acquire);
  // At least 1 should have woken (could be more due to spurious wakeups).
  EXPECT_GE(initial_woken, 1);

  // Wake remaining waiters.
  iree_notification_post(&notification, IREE_ALL_WAITERS);

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(kNumWaiters, woken_count.load(std::memory_order_acquire));

  iree_notification_deinitialize(&notification);
}

//==============================================================================
// Low-level prepare/commit/cancel API tests
//==============================================================================

TEST(NotificationTest, PrepareAndCancelWait) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  iree_wait_token_t token = iree_notification_prepare_wait(&notification);
  (void)token;  // We only care that prepare doesn't crash.
  iree_notification_cancel_wait(&notification);
  // Should not deadlock or crash.

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, PrepareCommitWaitAlreadySignaled) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  // Prepare wait (captures epoch).
  iree_wait_token_t token = iree_notification_prepare_wait(&notification);

  // Post before commit - should cause commit to return quickly.
  iree_notification_post(&notification, IREE_ALL_WAITERS);

  // Commit should return true (already signaled since epoch changed).
  bool result = iree_notification_commit_wait(
      &notification, token,
      /*spin_ns=*/0, iree_time_now() + 100 * 1000000);  // 100ms timeout.
  EXPECT_TRUE(result);

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, PrepareCommitWaitTimeout) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  // Prepare wait.
  iree_wait_token_t token = iree_notification_prepare_wait(&notification);

  iree_time_t start = iree_time_now();

  // Commit with short timeout - should timeout since no post.
  bool result = iree_notification_commit_wait(&notification, token,
                                              /*spin_ns=*/0,
                                              iree_time_now() + 50 * 1000000);

  iree_time_t elapsed = iree_time_now() - start;

  EXPECT_FALSE(result);
  EXPECT_GE(elapsed, 40 * 1000000);  // Should have waited close to 50ms.

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, PrepareCommitWaitWoken) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::atomic<bool> waiter_prepared{false};
  std::atomic<bool> waiter_result{false};

  std::thread waiter([&]() {
    iree_wait_token_t token = iree_notification_prepare_wait(&notification);
    waiter_prepared.store(true, std::memory_order_release);

    bool result = iree_notification_commit_wait(
        &notification, token,
        /*spin_ns=*/0, iree_time_now() + 1000000000);  // 1s timeout.
    waiter_result.store(result, std::memory_order_release);
  });

  // Wait for waiter to prepare.
  while (!waiter_prepared.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  // Post to wake the waiter. Even if commit_wait hasn't entered the kernel yet,
  // the epoch will have advanced and commit_wait will return immediately.
  iree_notification_post(&notification, IREE_ALL_WAITERS);

  waiter.join();

  EXPECT_TRUE(waiter_result.load(std::memory_order_acquire));

  iree_notification_deinitialize(&notification);
}

//==============================================================================
// Stress tests
//==============================================================================

TEST(NotificationTest, RapidPostAwait) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::atomic<int> counter{0};
  constexpr int kIterations = 100;

  std::thread producer([&]() {
    for (int i = 0; i < kIterations; ++i) {
      counter.fetch_add(1, std::memory_order_release);
      iree_notification_post(&notification, IREE_ALL_WAITERS);
      std::this_thread::yield();
    }
  });

  std::thread consumer([&]() {
    int last_seen = 0;
    while (last_seen < kIterations) {
      // Use a struct on the stack instead of heap allocation.
      struct await_state_t {
        std::atomic<int>* counter;
        int last_seen;
      } state = {&counter, last_seen};
      iree_notification_await(
          &notification,
          +[](void* arg) {
            auto* state = static_cast<await_state_t*>(arg);
            return state->counter->load(std::memory_order_acquire) >
                   state->last_seen;
          },
          &state, iree_make_timeout_ms(100));
      last_seen = counter.load(std::memory_order_acquire);
    }
  });

  producer.join();
  consumer.join();

  EXPECT_EQ(kIterations, counter.load(std::memory_order_acquire));

  iree_notification_deinitialize(&notification);
}

}  // namespace
