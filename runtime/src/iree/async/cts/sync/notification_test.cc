// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async notification operations.
//
// Notifications are lightweight synchronization primitives for proactor-
// integrated thread wakeup. Unlike events (edge-triggered, one signal per
// wait), notifications use an epoch counter that allows multiple signals
// to coalesce before a wait.
//
// Implementation varies by platform and capability:
//   - io_uring 6.7+: Uses futex word with FUTEX_WAIT/WAKE operations
//   - io_uring <6.7: Uses eventfd with linked POLL_ADD+READ pattern
//   - Other platforms: Platform-specific implementations

#include "iree/async/notification.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

namespace iree::async::cts {

class NotificationTest : public CtsTestBase<> {
 protected:
  // Initializes a NOTIFICATION_WAIT operation.
  static void InitNotificationWaitOp(
      iree_async_notification_wait_operation_t* operation,
      iree_async_notification_t* notification,
      iree_async_completion_fn_t callback, void* user_data) {
    memset(operation, 0, sizeof(*operation));
    operation->base.type = IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT;
    operation->base.completion_fn = callback;
    operation->base.user_data = user_data;
    operation->notification = notification;
    // wait_token is set by the proactor at submit time.
  }

  // Initializes a NOTIFICATION_SIGNAL operation.
  static void InitNotificationSignalOp(
      iree_async_notification_signal_operation_t* operation,
      iree_async_notification_t* notification, int32_t wake_count,
      iree_async_completion_fn_t callback, void* user_data) {
    memset(operation, 0, sizeof(*operation));
    operation->base.type = IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL;
    operation->base.completion_fn = callback;
    operation->base.user_data = user_data;
    operation->notification = notification;
    operation->wake_count = wake_count;
  }
};

// Create notification, retain/release, verify proper lifecycle.
TEST_P(NotificationTest, RetainRelease) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  // Initial ref count is 1.
  iree_async_notification_retain(notification);
  // Now ref count is 2.

  iree_async_notification_release(notification);
  // Now ref count is 1.

  iree_async_notification_release(notification);
  // Now ref count is 0, notification is destroyed.
}

// Signal notification with no waiters - should complete without error.
TEST_P(NotificationTest, SignalNoWaiters) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  // Signal with no waiters - this should be a no-op that succeeds.
  iree_async_notification_signal(notification, 1);

  // Verify notification is still usable.
  iree_async_notification_signal(notification, INT32_MAX);

  iree_async_notification_release(notification);
}

// Synchronous wait with signal from another thread.
//
// Epoch-based notifications coalesce signals that arrive before the waiter
// captures the epoch (via the internal prepare_wait). There is no way to
// observe from outside when prepare_wait has executed, so a single signal
// is not guaranteed to wake the worker. Signal continuously until the waiter
// completes — if a signal coalesces, the next one lands after prepare_wait.
TEST_P(NotificationTest, SyncWaitCrossThread) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<bool> wait_completed{false};

  std::thread waiter([&]() {
    bool result =
        iree_async_notification_wait(notification, iree_make_timeout_ms(5000));
    wait_completed.store(result, std::memory_order_release);
  });

  // Signal continuously until the waiter completes.
  while (!wait_completed.load(std::memory_order_acquire)) {
    iree_async_notification_signal(notification, 1);
    iree_thread_yield();
  }

  waiter.join();

  EXPECT_TRUE(wait_completed.load(std::memory_order_acquire));

  iree_async_notification_release(notification);
}

// Synchronous wait timeout.
TEST_P(NotificationTest, SyncWaitTimeout) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  // Wait with short timeout - should return false.
  iree_time_t start = iree_time_now();
  bool result =
      iree_async_notification_wait(notification, iree_make_timeout_ms(50));
  iree_time_t elapsed = iree_time_now() - start;

  EXPECT_FALSE(result);
  // Should have waited at least 10ms (generous slack for system scheduling).
  EXPECT_GE(elapsed, iree_make_duration_ms(10));

  iree_async_notification_release(notification);
}

// Multiple signals coalesce - waiter wakes exactly once regardless of how
// many signals arrive during a single wait.
TEST_P(NotificationTest, MultipleSignalsWhileWaiting) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<bool> wait_completed{false};

  std::thread waiter([&]() {
    iree_async_notification_wait(notification, iree_make_timeout_ms(5000));
    wait_completed.store(true, std::memory_order_release);
  });

  // Signal continuously until the waiter completes. Multiple signals coalesce
  // into one epoch advance per wait — the waiter wakes exactly once.
  while (!wait_completed.load(std::memory_order_acquire)) {
    iree_async_notification_signal(notification, 1);
    iree_thread_yield();
  }

  waiter.join();

  // Epoch advanced by more than 1 (multiple signals), but the waiter only
  // woke once from a single wait() call.
  EXPECT_GE(iree_async_notification_query_epoch(notification), 1u);

  iree_async_notification_release(notification);
}

// Repeated wait/signal cycles work correctly.
//
// Epoch-based notifications coalesce signals that arrive before the waiter
// captures the epoch (via the internal prepare_wait). There is no way to
// observe from outside when prepare_wait has executed, so a single signal
// is not guaranteed to wake the worker. Instead, the main thread signals
// continuously until the worker acknowledges each cycle. This makes progress
// unconditional: if a signal coalesces, the next one lands after prepare_wait.
TEST_P(NotificationTest, RepeatedWaitSignalCycles) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<int> cycles_completed{0};
  std::atomic<int> worker_cycle{0};
  constexpr int kCycles = 3;

  // Worker thread waits for signals in a loop.
  std::thread worker([&]() {
    for (int i = 0; i < kCycles; ++i) {
      // Publish which cycle we are about to wait on. The main thread uses
      // this to avoid racing ahead to cycle N+1 before we start cycle N.
      worker_cycle.store(i + 1, std::memory_order_release);

      bool result = iree_async_notification_wait(notification,
                                                 iree_make_timeout_ms(5000));
      if (result) {
        cycles_completed.fetch_add(1, std::memory_order_acq_rel);
      }
    }
  });

  for (int i = 0; i < kCycles; ++i) {
    // Wait for the worker to begin this cycle.
    while (worker_cycle.load(std::memory_order_acquire) < i + 1) {
      iree_thread_yield();
    }
    // Signal continuously until the worker completes this cycle.
    while (cycles_completed.load(std::memory_order_acquire) < i + 1) {
      iree_async_notification_signal(notification, 1);
      iree_thread_yield();
    }
  }

  worker.join();

  EXPECT_EQ(cycles_completed.load(std::memory_order_acquire), kCycles);

  iree_async_notification_release(notification);
}

// Multiple waiters all wake when signaled.
TEST_P(NotificationTest, MultipleWaitersPartialWake) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<int> waiters_woken{0};
  constexpr int kNumWaiters = 3;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      bool result = iree_async_notification_wait(notification,
                                                 iree_make_timeout_ms(5000));
      if (result) {
        waiters_woken.fetch_add(1, std::memory_order_acq_rel);
      }
    });
  }

  // Signal continuously until all waiters complete.
  while (waiters_woken.load(std::memory_order_acquire) < kNumWaiters) {
    iree_async_notification_signal(notification, INT32_MAX);
    iree_thread_yield();
  }

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(waiters_woken.load(std::memory_order_acquire), kNumWaiters);

  iree_async_notification_release(notification);
}

// Broadcast wake (INT32_MAX) wakes all waiters.
TEST_P(NotificationTest, BroadcastWake) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<int> waiters_woken{0};
  constexpr int kNumWaiters = 3;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      bool result = iree_async_notification_wait(notification,
                                                 iree_make_timeout_ms(5000));
      if (result) {
        waiters_woken.fetch_add(1, std::memory_order_acq_rel);
      }
    });
  }

  // Signal continuously until all waiters wake.
  while (waiters_woken.load(std::memory_order_acquire) < kNumWaiters) {
    iree_async_notification_signal(notification, INT32_MAX);
    iree_thread_yield();
  }

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(waiters_woken.load(std::memory_order_acquire), kNumWaiters);

  iree_async_notification_release(notification);
}

// Async NOTIFICATION_WAIT operation via proactor.
TEST_P(NotificationTest, AsyncWait) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  CompletionTracker tracker;
  iree_async_notification_wait_operation_t wait_op;
  InitNotificationWaitOp(&wait_op, notification, CompletionTracker::Callback,
                         &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Signal the notification from another thread.
  std::thread signaler(
      [notification]() { iree_async_notification_signal(notification, 1); });

  // Poll until the wait completes.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  signaler.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_notification_release(notification);
}

// Async NOTIFICATION_SIGNAL operation via proactor.
TEST_P(NotificationTest, AsyncSignal) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<bool> waiter_completed{false};

  // Background thread waits synchronously.
  std::thread waiter([&]() {
    bool result =
        iree_async_notification_wait(notification, iree_make_timeout_ms(5000));
    waiter_completed.store(result, std::memory_order_release);
  });

  // Submit async signal operations continuously until the waiter completes.
  // Each signal op is polled to completion before submitting the next.
  while (!waiter_completed.load(std::memory_order_acquire)) {
    CompletionTracker tracker;
    iree_async_notification_signal_operation_t signal_op;
    InitNotificationSignalOp(&signal_op, notification, 1,
                             CompletionTracker::Callback, &tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &signal_op.base));

    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));

    IREE_EXPECT_OK(tracker.ConsumeStatus());
  }

  waiter.join();

  EXPECT_TRUE(waiter_completed.load(std::memory_order_acquire));

  iree_async_notification_release(notification);
}

// Chain: NOTIFICATION_WAIT -> NOP, verify order.
TEST_P(NotificationTest, ChainedWaitNop) {
  if (!iree_any_bit_set(capabilities_,
                        IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS)) {
    GTEST_SKIP() << "backend lacks linked operations capability";
  }

  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  struct OrderTracker {
    std::vector<int> order;
    static void WaitCallback(void* u, iree_async_operation_t* o,
                             iree_status_t s, iree_async_completion_flags_t f) {
      static_cast<OrderTracker*>(u)->order.push_back(0);
      iree_status_ignore(s);
    }
    static void NopCallback(void* u, iree_async_operation_t* o, iree_status_t s,
                            iree_async_completion_flags_t f) {
      static_cast<OrderTracker*>(u)->order.push_back(1);
      iree_status_ignore(s);
    }
  };

  OrderTracker tracker;

  iree_async_notification_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT;
  wait_op.base.completion_fn = OrderTracker::WaitCallback;
  wait_op.base.user_data = &tracker;
  wait_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  wait_op.notification = notification;

  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop.base.completion_fn = OrderTracker::NopCallback;
  nop.base.user_data = &tracker;

  iree_async_operation_t* ops[] = {&wait_op.base, &nop.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Signal the notification from another thread.
  std::thread signaler(
      [notification]() { iree_async_notification_signal(notification, 1); });

  // Poll until both complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  signaler.join();

  ASSERT_EQ(tracker.order.size(), 2u);
  EXPECT_EQ(tracker.order[0], 0);  // Wait first.
  EXPECT_EQ(tracker.order[1], 1);  // NOP second.

  iree_async_notification_release(notification);
}

// Round-trip: async wait completes, callback wakes main thread.
// Uses iree_notification_t (base threading primitive) for callback -> main
// communication since the sync wait API captures epoch at call time.
TEST_P(NotificationTest, RoundTrip) {
  iree_async_notification_t* async_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &async_notification));

  struct RoundTripContext {
    bool callback_fired = false;
  };
  RoundTripContext context = {false};

  iree_async_notification_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT;
  wait_op.base.completion_fn = [](void* user_data, iree_async_operation_t* op,
                                  iree_status_t status,
                                  iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<RoundTripContext*>(user_data);
    ctx->callback_fired = true;
    iree_status_ignore(status);
  };
  wait_op.base.user_data = &context;
  wait_op.notification = async_notification;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Background thread signals the async notification.
  iree_notification_t ready;
  iree_notification_initialize(&ready);

  std::thread signaler([&]() {
    iree_notification_await(
        &ready, [](void*) { return true; }, nullptr, iree_infinite_timeout());
    iree_async_notification_signal(async_notification, 1);
  });

  // Signal the background thread to proceed.
  iree_notification_post(&ready, 1);

  // Main thread polls the proactor until the callback fires.
  // The callback posts to main_wakeup, which we can check with try_wait.
  while (!context.callback_fired) {
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_make_timeout_ms(100), &completed);
    if (!iree_status_is_deadline_exceeded(status)) {
      IREE_ASSERT_OK(status);
    } else {
      iree_status_ignore(status);
    }
  }

  signaler.join();

  EXPECT_TRUE(context.callback_fired);

  iree_notification_deinitialize(&ready);
  iree_async_notification_release(async_notification);
}

CTS_REGISTER_TEST_SUITE(NotificationTest);

}  // namespace iree::async::cts
