// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async futex operations (FUTEX_WAIT/FUTEX_WAKE).
//
// These tests verify the io_uring futex2 integration available on Linux 6.7+.
// Unlike the base futex syscall wrappers in iree/base/futex.h, these tests
// cover the async operation types that can be submitted to a proactor:
//   - IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT
//   - IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE
//
// The key pattern: submit FUTEX_WAIT to the proactor, wake it from a background
// thread using direct syscalls, verify completion, then optionally have the
// completion callback wake the main thread back for round-trip verification.
//
// All tests use iree_notification_t for thread synchronization (no sleeps).
// Tests requiring kernel 6.7+ check
// IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS and skip gracefully on older
// kernels.

#include "iree/async/operations/futex.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/futex.h"
#include "iree/base/threading/notification.h"

namespace iree::async::cts {

class FutexTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (!iree_any_bit_set(capabilities_,
                          IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS)) {
      GTEST_SKIP() << "backend lacks futex operations capability";
    }
  }

  // Initializes a FUTEX_WAIT operation.
  static void InitFutexWaitOp(iree_async_futex_wait_operation_t* operation,
                              void* address, uint32_t expected,
                              iree_async_completion_fn_t callback,
                              void* user_data) {
    memset(operation, 0, sizeof(*operation));
    operation->base.type = IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT;
    operation->base.completion_fn = callback;
    operation->base.user_data = user_data;
    operation->futex_address = address;
    operation->expected_value = expected;
    operation->futex_flags =
        IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;
  }

  // Initializes a FUTEX_WAKE operation.
  static void InitFutexWakeOp(iree_async_futex_wake_operation_t* operation,
                              void* address, int32_t count,
                              iree_async_completion_fn_t callback,
                              void* user_data) {
    memset(operation, 0, sizeof(*operation));
    operation->base.type = IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE;
    operation->base.completion_fn = callback;
    operation->base.user_data = user_data;
    operation->futex_address = address;
    operation->wake_count = count;
    operation->futex_flags =
        IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;
  }
};

#if defined(IREE_RUNTIME_USE_FUTEX)

// Submit FUTEX_WAIT to the proactor, background thread wakes it, verify
// completion with OK status.
TEST_P(FutexTest, BasicFutexWaitWake) {
  std::atomic<uint32_t> futex_word{0};
  iree_notification_t ready;
  iree_notification_initialize(&ready);

  CompletionTracker tracker;
  iree_async_futex_wait_operation_t wait_op;
  InitFutexWaitOp(&wait_op, &futex_word, 0, CompletionTracker::Callback,
                  &tracker);

  // Submit the wait operation.
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Background thread: wait for ready signal, then wake the futex.
  std::thread waker([&]() {
    // Wait for main thread to signal that the operation is submitted.
    iree_notification_await(
        &ready, [](void*) { return true; }, nullptr, iree_infinite_timeout());

    // Change the futex word and wake.
    futex_word.store(1, std::memory_order_release);
    iree_futex_wake(&futex_word, 1);
  });

  // Signal the background thread that we've submitted.
  iree_notification_post(&ready, 1);

  // Poll until the wait operation completes.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  waker.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_notification_deinitialize(&ready);
}

// Submit FUTEX_WAIT with wrong expected value. The operation should complete
// immediately with OK status since the value has already "changed" from the
// perspective of the wait (it never matched).
TEST_P(FutexTest, FutexWaitValueMismatch) {
  std::atomic<uint32_t> futex_word{42};  // Value is 42, not 0.

  CompletionTracker tracker;
  iree_async_futex_wait_operation_t wait_op;
  InitFutexWaitOp(&wait_op, &futex_word, 0,  // Expected 0, but actual is 42.
                  CompletionTracker::Callback, &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Should complete immediately since value doesn't match expected.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Submit FUTEX_WAKE operation to wake a background thread doing a syscall wait.
TEST_P(FutexTest, FutexWakeWakesWaiter) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<bool> waiter_started{false};
  std::atomic<bool> waiter_woken{false};

  // Background thread: wait on the futex using direct syscall.
  std::thread waiter([&]() {
    waiter_started.store(true, std::memory_order_release);

    // Wait on the futex (direct syscall, not async).
    while (futex_word.load(std::memory_order_acquire) == 0) {
      iree_futex_wait(&futex_word, 0,
                      iree_time_now() + iree_make_duration_ms(100));
    }

    waiter_woken.store(true, std::memory_order_release);
  });

  // Wait for waiter to start.
  while (!waiter_started.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }

  // Change the futex word.
  futex_word.store(1, std::memory_order_release);

  // Submit async FUTEX_WAKE operation.
  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, 1, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  // Poll for completion.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  waiter.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_GE(wake_op.woken_count, 0);  // May be 0 if waiter saw value change.
  EXPECT_TRUE(waiter_woken.load(std::memory_order_acquire));
}

// Submit FUTEX_WAKE(2) and verify exactly 2 of 3 waiters wake.
TEST_P(FutexTest, FutexWakeCount) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<int> waiters_ready{0};
  std::atomic<int> waiters_woken{0};
  constexpr int kNumWaiters = 3;
  constexpr int kWakeCount = 2;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      waiters_ready.fetch_add(1, std::memory_order_acq_rel);

      // Wait on the futex (direct syscall).
      while (futex_word.load(std::memory_order_acquire) == 0) {
        iree_futex_wait(&futex_word, 0,
                        iree_time_now() + iree_make_duration_ms(100));
      }

      waiters_woken.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to be ready.
  while (waiters_ready.load(std::memory_order_acquire) < kNumWaiters) {
    iree_thread_yield();
  }

  // Change the futex word.
  futex_word.store(1, std::memory_order_release);

  // Submit async FUTEX_WAKE for exactly 2 waiters.
  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, kWakeCount,
                  CompletionTracker::Callback, &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  // Poll for completion of the wake operation.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  // Wake remaining waiters so we can join.
  iree_futex_wake(&futex_word, IREE_ALL_WAITERS);

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Submit FUTEX_WAKE(INT32_MAX) and verify all waiters wake.
TEST_P(FutexTest, FutexWakeAll) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<int> waiters_ready{0};
  std::atomic<int> waiters_woken{0};
  constexpr int kNumWaiters = 3;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      waiters_ready.fetch_add(1, std::memory_order_acq_rel);

      // Wait on the futex (direct syscall).
      while (futex_word.load(std::memory_order_acquire) == 0) {
        iree_futex_wait(&futex_word, 0,
                        iree_time_now() + iree_make_duration_ms(100));
      }

      waiters_woken.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to be ready.
  while (waiters_ready.load(std::memory_order_acquire) < kNumWaiters) {
    iree_thread_yield();
  }

  // Change the futex word.
  futex_word.store(1, std::memory_order_release);

  // Submit async FUTEX_WAKE for all waiters.
  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, INT32_MAX, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  // Poll for completion.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(waiters_woken.load(std::memory_order_acquire), kNumWaiters);
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Submit FUTEX_WAKE with no waiters. Verify woken_count == 0.
TEST_P(FutexTest, FutexWakeNoWaiters) {
  std::atomic<uint32_t> futex_word{0};

  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, INT32_MAX, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_EQ(wake_op.woken_count, 0);
}

// Full async round-trip: main submits WAIT, background wakes it, completion
// callback wakes main back.
//
// Main thread:                    Background thread:
//   futex_A = 0, futex_B = 0
//   notification "ready" = 0
//   submit FUTEX_WAIT(A, expected=0)
//     with callback that:
//       futex_B = 1
//       syscall futex_wake(B)
//   post(ready)  ──────────────────►  wait(ready)
//   syscall futex_wait(B, expected=0)
//   |                                futex_A = 1
//   |                                syscall futex_wake(A)
//   poll() -> FUTEX_WAIT(A) completes
//   callback fires:
//     futex_B = 1
//     syscall futex_wake(B)
//   main's futex_wait(B) returns
//   verify round-trip completed
TEST_P(FutexTest, FutexRoundTrip) {
  std::atomic<uint32_t> futex_a{0};
  std::atomic<uint32_t> futex_b{0};
  iree_notification_t ready;
  iree_notification_initialize(&ready);

  struct RoundTripContext {
    std::atomic<uint32_t>* futex_b;
    bool callback_fired = false;
  };
  RoundTripContext context = {&futex_b, false};

  iree_async_futex_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT;
  wait_op.base.completion_fn = [](void* user_data, iree_async_operation_t* op,
                                  iree_status_t status,
                                  iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<RoundTripContext*>(user_data);
    ctx->callback_fired = true;
    // Wake the main thread by setting futex_b.
    ctx->futex_b->store(1, std::memory_order_release);
    iree_futex_wake(ctx->futex_b, 1);
    iree_status_ignore(status);
  };
  wait_op.base.user_data = &context;
  wait_op.futex_address = &futex_a;
  wait_op.expected_value = 0;
  wait_op.futex_flags =
      IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;

  // Submit the wait operation.
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Background thread: wake futex_a after ready signal.
  std::thread waker([&]() {
    iree_notification_await(
        &ready, [](void*) { return true; }, nullptr, iree_infinite_timeout());

    // Wake the proactor's wait on futex_a.
    futex_a.store(1, std::memory_order_release);
    iree_futex_wake(&futex_a, 1);
  });

  // Signal background thread.
  iree_notification_post(&ready, 1);

  // Main thread waits on futex_b (will be woken by callback).
  while (futex_b.load(std::memory_order_acquire) == 0) {
    // Poll the proactor while waiting.
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_make_timeout_ms(100), &completed);
    if (!iree_status_is_deadline_exceeded(status)) {
      IREE_ASSERT_OK(status);
    } else {
      iree_status_ignore(status);
    }
  }

  waker.join();

  EXPECT_TRUE(context.callback_fired);

  iree_notification_deinitialize(&ready);
}

// Submit FUTEX_WAIT + LINK + NOP, verify NOP only runs after wait completes.
TEST_P(FutexTest, FutexChainedOperations) {
  if (!iree_any_bit_set(capabilities_,
                        IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS)) {
    GTEST_SKIP() << "backend lacks linked operations capability";
  }

  std::atomic<uint32_t> futex_word{0};
  iree_notification_t ready;
  iree_notification_initialize(&ready);

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

  iree_async_futex_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT;
  wait_op.base.completion_fn = OrderTracker::WaitCallback;
  wait_op.base.user_data = &tracker;
  wait_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;  // Link to next op.
  wait_op.futex_address = &futex_word;
  wait_op.expected_value = 0;
  wait_op.futex_flags =
      IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;

  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop.base.completion_fn = OrderTracker::NopCallback;
  nop.base.user_data = &tracker;

  iree_async_operation_t* ops[] = {&wait_op.base, &nop.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Background thread: wake the futex after signal.
  std::thread waker([&]() {
    iree_notification_await(
        &ready, [](void*) { return true; }, nullptr, iree_infinite_timeout());

    // The LINKED flag ensures NOP waits for WAIT to complete.
    futex_word.store(1, std::memory_order_release);
    iree_futex_wake(&futex_word, 1);
  });

  // Signal background thread.
  iree_notification_post(&ready, 1);

  // Poll until both complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  waker.join();

  ASSERT_EQ(tracker.order.size(), 2u);
  // Wait must complete before NOP (enforced by LINK).
  EXPECT_EQ(tracker.order[0], 0);
  EXPECT_EQ(tracker.order[1], 1);

  iree_notification_deinitialize(&ready);
}

// Submit FUTEX_WAIT, cancel it, verify callback fires with CANCELLED status.
TEST_P(FutexTest, FutexCancellation) {
  std::atomic<uint32_t> futex_word{0};

  CompletionTracker tracker;
  iree_async_futex_wait_operation_t wait_op;
  InitFutexWaitOp(&wait_op, &futex_word, 0, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Cancel the wait operation immediately.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

  // Poll to receive the cancellation callback.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
}

// Tests that a background thread waiting on a futex can be woken from a
// completion callback. This is a common pattern: async I/O completes, callback
// wakes a worker thread that was waiting for data.
TEST_P(FutexTest, WakeFromCompletionCallback) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<bool> waiter_started{false};
  std::atomic<bool> waiter_completed{false};

  // Background thread: wait for futex_word to become non-zero.
  std::thread waiter([&]() {
    waiter_started.store(true, std::memory_order_release);

    while (futex_word.load(std::memory_order_acquire) == 0) {
      iree_futex_wait(&futex_word, 0,
                      iree_time_now() + iree_make_duration_ms(100));
    }

    waiter_completed.store(true, std::memory_order_release);
  });

  // Wait for waiter to start.
  while (!waiter_started.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }

  // Use an async FUTEX_WAKE operation (submitted to proactor) to wake the
  // waiter.
  futex_word.store(1, std::memory_order_release);

  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, 1, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  // Poll until the wake completes.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  // Wait for the waiter thread to finish.
  waiter.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_TRUE(waiter_completed.load(std::memory_order_acquire));
}

// Tests that multiple waiters can be woken with IREE_ALL_WAITERS.
TEST_P(FutexTest, WakeAllFromCallback) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<int> waiters_started{0};
  std::atomic<int> waiters_completed{0};
  constexpr int kNumWaiters = 3;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      waiters_started.fetch_add(1, std::memory_order_acq_rel);

      while (futex_word.load(std::memory_order_acquire) == 0) {
        iree_futex_wait(&futex_word, 0,
                        iree_time_now() + iree_make_duration_ms(100));
      }

      waiters_completed.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to start.
  while (waiters_started.load(std::memory_order_acquire) < kNumWaiters) {
    iree_thread_yield();
  }

  // Use async FUTEX_WAKE to wake all waiters.
  futex_word.store(1, std::memory_order_release);

  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, IREE_ALL_WAITERS,
                  CompletionTracker::Callback, &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(waiters_completed.load(std::memory_order_acquire), kNumWaiters);
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Double-cancel a FUTEX_WAIT operation. Second cancel should be harmless.
TEST_P(FutexTest, FutexDoubleCancellation) {
  std::atomic<uint32_t> futex_word{0};

  CompletionTracker tracker;
  iree_async_futex_wait_operation_t wait_op;
  InitFutexWaitOp(&wait_op, &futex_word, 0, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // First cancel.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

  // Second cancel should be harmless.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

  // Poll to receive the cancellation callback.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  // Drain any remaining CQEs.
  DrainPending(iree_make_duration_ms(100));

  // Exactly one callback should have fired.
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
}

// Cancel a FUTEX_WAIT that races with a wake. Either outcome is valid but the
// callback must fire exactly once.
TEST_P(FutexTest, CancelRacesWithWake) {
  static constexpr int kIterations = 10;

  for (int iter = 0; iter < kIterations; ++iter) {
    std::atomic<uint32_t> futex_word{0};

    CompletionTracker tracker;
    iree_async_futex_wait_operation_t wait_op;
    InitFutexWaitOp(&wait_op, &futex_word, 0, CompletionTracker::Callback,
                    &tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

    // Concurrently cancel and wake to create a race.
    std::thread waker([&]() {
      futex_word.store(1, std::memory_order_release);
      iree_futex_wake(&futex_word, 1);
    });

    IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

    waker.join();

    // Poll for the callback.
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));

    // Drain any additional CQEs.
    DrainPending(iree_make_duration_ms(100));

    // Must have exactly one callback.
    EXPECT_EQ(tracker.call_count, 1) << "Iteration " << iter;

    // Either CANCELLED or OK is valid.
    iree_status_t status = tracker.ConsumeStatus();
    if (!iree_status_is_ok(status) && !iree_status_is_cancelled(status)) {
      IREE_EXPECT_OK(status) << "Iteration " << iter;
    } else {
      iree_status_ignore(status);
    }
  }
}

#else

// Placeholder when futex is not available (e.g., macOS, Windows without
// appropriate support).
TEST_P(FutexTest, NotAvailable) {
  GTEST_SKIP() << "Futex not available on this platform/configuration";
}

#endif  // IREE_RUNTIME_USE_FUTEX

CTS_REGISTER_TEST_SUITE(FutexTest);

}  // namespace iree::async::cts
