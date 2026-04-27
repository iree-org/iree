// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/process.h"

#include <cstring>
#include <thread>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Test drain functions
//===----------------------------------------------------------------------===//

// Drain function that completes immediately.
static iree_status_t drain_immediate(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* result) {
  result->completed = true;
  result->did_work = true;
  return iree_ok_status();
}

// Drain function that does work but never completes (for testing wake/cancel).
static iree_status_t drain_never_completes(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* result) {
  result->completed = false;
  result->did_work = true;
  return iree_ok_status();
}

// Drain function that counts calls via user_data (int32_t counter).
static iree_status_t drain_counting(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* result) {
  iree_atomic_int32_t* counter =
      reinterpret_cast<iree_atomic_int32_t*>(process->user_data);
  int32_t count =
      iree_atomic_fetch_add(counter, 1, iree_memory_order_relaxed) + 1;
  // Complete after 3 calls.
  result->completed = (count >= 3);
  result->did_work = true;
  return iree_ok_status();
}

// Drain function that returns an error.
static iree_status_t drain_failing(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* result) {
  result->completed = true;
  result->did_work = true;
  return iree_make_status(IREE_STATUS_INTERNAL, "drain failed");
}

// Completion callback that records whether it was called and with what status.
struct CompletionRecord {
  bool called = false;
  iree_status_code_t status_code = IREE_STATUS_OK;
};
static void completion_record(iree_task_process_t* process,
                              iree_status_t status) {
  CompletionRecord* record =
      reinterpret_cast<CompletionRecord*>(process->user_data);
  record->called = true;
  record->status_code = iree_status_code(status);
  iree_status_free(status);
}

//===----------------------------------------------------------------------===//
// Initialization tests
//===----------------------------------------------------------------------===//

TEST(ProcessTest, InitializeSuspended) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/1,
                               /*wake_budget=*/1, &process);
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_SUSPENDED);
  EXPECT_FALSE(iree_task_process_is_terminal(&process));
  EXPECT_FALSE(iree_task_process_has_error(&process));
  EXPECT_EQ(iree_task_process_wake_budget(&process), 1);
}

TEST(ProcessTest, InitializeRunnable) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0,
                               /*wake_budget=*/4, &process);
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_RUNNABLE);
  EXPECT_EQ(iree_task_process_wake_budget(&process), 4);
}

TEST(ProcessTest, WarmRetainerAdmissionCapsToLimit) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0,
                               /*wake_budget=*/2, &process);

  EXPECT_TRUE(iree_task_process_try_retain_warm(&process, 2));
  EXPECT_TRUE(iree_task_process_try_retain_warm(&process, 2));
  EXPECT_FALSE(iree_task_process_try_retain_warm(&process, 2));
  EXPECT_EQ(iree_task_process_warm_retainer_count(&process), 2);

  EXPECT_FALSE(iree_task_process_try_retain_warm(&process, 1));
  EXPECT_EQ(iree_task_process_warm_retainer_count(&process), 2);

  iree_atomic_fetch_sub(&process.warm_retainers, 1, iree_memory_order_acq_rel);
  EXPECT_FALSE(iree_task_process_try_retain_warm(&process, 1));
  iree_atomic_fetch_sub(&process.warm_retainers, 1, iree_memory_order_acq_rel);
  EXPECT_TRUE(iree_task_process_try_retain_warm(&process, 1));
  EXPECT_EQ(iree_task_process_warm_retainer_count(&process), 1);
}

//===----------------------------------------------------------------------===//
// Wake tests
//===----------------------------------------------------------------------===//

TEST(ProcessTest, WakeSingle) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/1, 1,
                               &process);

  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_SUSPENDED);
  EXPECT_TRUE(iree_task_process_wake(&process));
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_RUNNABLE);
}

TEST(ProcessTest, WakeMultiple) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/3, 1,
                               &process);

  // First two wakes don't activate.
  EXPECT_FALSE(iree_task_process_wake(&process));
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_SUSPENDED);
  EXPECT_FALSE(iree_task_process_wake(&process));
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_SUSPENDED);

  // Third wake activates.
  EXPECT_TRUE(iree_task_process_wake(&process));
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_RUNNABLE);
}

TEST(ProcessTest, WakeCancelledReturnsFalse) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/1, 1,
                               &process);

  iree_task_process_t* head = nullptr;
  iree_task_process_t* tail = nullptr;
  iree_task_process_cancel(&process, &head, &tail);
  EXPECT_FALSE(iree_task_process_wake(&process));
  // State should still be CANCELLED.
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_CANCELLED);
}

//===----------------------------------------------------------------------===//
// Completion tests
//===----------------------------------------------------------------------===//

TEST(ProcessTest, CompleteSuccess) {
  CompletionRecord record;
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0, 1,
                               &process);
  process.completion_fn = completion_record;
  process.user_data = &record;

  iree_task_process_t* activated_head = nullptr;
  iree_task_process_t* activated_tail = nullptr;
  iree_task_process_complete(&process, &activated_head, &activated_tail);

  EXPECT_TRUE(record.called);
  EXPECT_EQ(record.status_code, IREE_STATUS_OK);
  EXPECT_TRUE(iree_task_process_is_terminal(&process));
  EXPECT_EQ(activated_head, nullptr);
  EXPECT_EQ(activated_tail, nullptr);
}

TEST(ProcessTest, CompleteWithError) {
  CompletionRecord record;
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0, 1,
                               &process);
  process.completion_fn = completion_record;
  process.user_data = &record;

  iree_task_process_report_error(
      &process, iree_make_status(IREE_STATUS_INTERNAL, "something broke"));

  iree_task_process_t* activated_head = nullptr;
  iree_task_process_t* activated_tail = nullptr;
  iree_task_process_complete(&process, &activated_head, &activated_tail);

  EXPECT_TRUE(record.called);
  EXPECT_EQ(record.status_code, IREE_STATUS_INTERNAL);
  EXPECT_TRUE(iree_task_process_is_terminal(&process));
}

TEST(ProcessTest, CompleteWithoutCallback) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0, 1,
                               &process);
  // No completion_fn set — should not crash.

  iree_task_process_t* activated_head = nullptr;
  iree_task_process_t* activated_tail = nullptr;
  iree_task_process_complete(&process, &activated_head, &activated_tail);

  EXPECT_TRUE(iree_task_process_is_terminal(&process));
}

//===----------------------------------------------------------------------===//
// Dependent resolution tests
//===----------------------------------------------------------------------===//

TEST(ProcessTest, CompletionActivatesDependents) {
  // Set up: A completes -> B and C should be woken.
  iree_task_process_t process_a;
  iree_task_process_t process_b;
  iree_task_process_t process_c;

  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0, 1,
                               &process_a);
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/1, 1,
                               &process_b);
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/1, 1,
                               &process_c);

  iree_task_process_t* dependents[] = {&process_b, &process_c};
  process_a.dependents = dependents;
  process_a.dependent_count = 2;

  iree_task_process_t* activated_head = nullptr;
  iree_task_process_t* activated_tail = nullptr;
  iree_task_process_complete(&process_a, &activated_head, &activated_tail);

  // Both B and C should be in the activated list.
  EXPECT_EQ(activated_head, &process_b);
  EXPECT_EQ(activated_tail, &process_c);
  EXPECT_EQ(iree_task_process_state(&process_b),
            IREE_TASK_PROCESS_STATE_RUNNABLE);
  EXPECT_EQ(iree_task_process_state(&process_c),
            IREE_TASK_PROCESS_STATE_RUNNABLE);

  // Walk the list to verify linking.
  EXPECT_EQ(iree_task_process_slist_get_next(activated_head), &process_c);
  EXPECT_EQ(iree_task_process_slist_get_next(activated_tail), nullptr);
}

TEST(ProcessTest, DependentStillSuspended) {
  // B has suspend_count=2, A is one of two dependencies.
  iree_task_process_t process_a;
  iree_task_process_t process_b;

  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0, 1,
                               &process_a);
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/2, 1,
                               &process_b);

  iree_task_process_t* dependents[] = {&process_b};
  process_a.dependents = dependents;
  process_a.dependent_count = 1;

  iree_task_process_t* activated_head = nullptr;
  iree_task_process_t* activated_tail = nullptr;
  iree_task_process_complete(&process_a, &activated_head, &activated_tail);

  // B should NOT be activated — it still has one remaining dependency.
  EXPECT_EQ(activated_head, nullptr);
  EXPECT_EQ(activated_tail, nullptr);
  EXPECT_EQ(iree_task_process_state(&process_b),
            IREE_TASK_PROCESS_STATE_SUSPENDED);
}

TEST(ProcessTest, DiamondDependency) {
  //     A
  //    / \
  //   B   C
  //    \ /
  //     D
  iree_task_process_t a, b, c, d;
  iree_task_process_initialize(drain_immediate, 0, 1, &a);
  iree_task_process_initialize(drain_immediate, 1, 1, &b);
  iree_task_process_initialize(drain_immediate, 1, 1, &c);
  iree_task_process_initialize(drain_immediate, 2, 1, &d);

  iree_task_process_t* a_deps[] = {&b, &c};
  a.dependents = a_deps;
  a.dependent_count = 2;

  iree_task_process_t* b_deps[] = {&d};
  b.dependents = b_deps;
  b.dependent_count = 1;

  iree_task_process_t* c_deps[] = {&d};
  c.dependents = c_deps;
  c.dependent_count = 1;

  // Complete A -> activates B and C.
  iree_task_process_t* head = nullptr;
  iree_task_process_t* tail = nullptr;
  iree_task_process_complete(&a, &head, &tail);
  EXPECT_EQ(head, &b);
  EXPECT_EQ(tail, &c);

  // Complete B -> D still suspended (suspend_count was 2, now 1).
  iree_task_process_complete(&b, &head, &tail);
  EXPECT_EQ(head, nullptr);
  EXPECT_EQ(iree_task_process_state(&d), IREE_TASK_PROCESS_STATE_SUSPENDED);

  // Complete C -> D activated.
  iree_task_process_complete(&c, &head, &tail);
  EXPECT_EQ(head, &d);
  EXPECT_EQ(tail, &d);
  EXPECT_EQ(iree_task_process_state(&d), IREE_TASK_PROCESS_STATE_RUNNABLE);
}

//===----------------------------------------------------------------------===//
// Error reporting tests
//===----------------------------------------------------------------------===//

TEST(ProcessTest, ReportError) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0, 1,
                               &process);

  EXPECT_FALSE(iree_task_process_has_error(&process));
  iree_task_process_report_error(
      &process, iree_make_status(IREE_STATUS_INTERNAL, "oops"));
  EXPECT_TRUE(iree_task_process_has_error(&process));
  EXPECT_TRUE(iree_task_process_is_terminal(&process));

  // Complete to consume the error status (avoids leak).
  iree_task_process_t* head = nullptr;
  iree_task_process_t* tail = nullptr;
  iree_task_process_complete(&process, &head, &tail);
}

TEST(ProcessTest, FirstErrorWins) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0, 1,
                               &process);

  iree_task_process_report_error(
      &process, iree_make_status(IREE_STATUS_INTERNAL, "first"));
  iree_task_process_report_error(
      &process,
      iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "second (dropped)"));

  // Complete and check that the first error is what the callback receives.
  CompletionRecord record;
  process.completion_fn = completion_record;
  process.user_data = &record;

  iree_task_process_t* head = nullptr;
  iree_task_process_t* tail = nullptr;
  iree_task_process_complete(&process, &head, &tail);

  EXPECT_TRUE(record.called);
  EXPECT_EQ(record.status_code, IREE_STATUS_INTERNAL);
}

//===----------------------------------------------------------------------===//
// Cancellation tests
//===----------------------------------------------------------------------===//

TEST(ProcessTest, CancelSuspended) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/1, 1,
                               &process);

  iree_task_process_t* head = nullptr;
  iree_task_process_t* tail = nullptr;
  iree_task_process_cancel(&process, &head, &tail);
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_CANCELLED);
  EXPECT_EQ(head, nullptr);
  EXPECT_EQ(tail, nullptr);
}

TEST(ProcessTest, CancelRunnable) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0, 1,
                               &process);

  iree_task_process_t* head = nullptr;
  iree_task_process_t* tail = nullptr;
  iree_task_process_cancel(&process, &head, &tail);
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_CANCELLED);
  // RUNNABLE cancel leaves error for the worker to consume via complete().
  EXPECT_TRUE(iree_task_process_has_error(&process));
  EXPECT_EQ(head, nullptr);
  EXPECT_EQ(tail, nullptr);
}

TEST(ProcessTest, CancelAlreadyCompleted) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0, 1,
                               &process);

  iree_task_process_t* head = nullptr;
  iree_task_process_t* tail = nullptr;
  iree_task_process_complete(&process, &head, &tail);
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_COMPLETED);

  // Cancel after completion is a no-op.
  iree_task_process_t* cancel_head = nullptr;
  iree_task_process_t* cancel_tail = nullptr;
  iree_task_process_cancel(&process, &cancel_head, &cancel_tail);
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_COMPLETED);
  EXPECT_EQ(cancel_head, nullptr);
  EXPECT_EQ(cancel_tail, nullptr);
}

TEST(ProcessTest, CancelPreservesExistingError) {
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0, 1,
                               &process);

  // Report a real error first.
  iree_task_process_report_error(
      &process, iree_make_status(IREE_STATUS_INTERNAL, "real error"));

  // Then cancel — the original error should be preserved.
  iree_task_process_t* cancel_head = nullptr;
  iree_task_process_t* cancel_tail = nullptr;
  iree_task_process_cancel(&process, &cancel_head, &cancel_tail);

  CompletionRecord record;
  process.completion_fn = completion_record;
  process.user_data = &record;

  iree_task_process_t* head = nullptr;
  iree_task_process_t* tail = nullptr;
  iree_task_process_complete(&process, &head, &tail);
  EXPECT_TRUE(record.called);
  EXPECT_EQ(record.status_code, IREE_STATUS_INTERNAL);
}

TEST(ProcessTest, CancelSuspendedResolvesDependents) {
  iree_task_process_t a, b, c;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/1, 1, &a);
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/1, 1, &b);
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/1, 1, &c);

  iree_task_process_t* a_deps[] = {&b, &c};
  a.dependents = a_deps;
  a.dependent_count = 2;

  iree_task_process_t* head = nullptr;
  iree_task_process_t* tail = nullptr;
  iree_task_process_cancel(&a, &head, &tail);

  // Cancel on SUSPENDED resolves inline: dependents are woken.
  EXPECT_EQ(iree_task_process_state(&a), IREE_TASK_PROCESS_STATE_CANCELLED);
  EXPECT_EQ(head, &b);
  EXPECT_EQ(tail, &c);
  EXPECT_EQ(iree_task_process_state(&b), IREE_TASK_PROCESS_STATE_RUNNABLE);
  EXPECT_EQ(iree_task_process_state(&c), IREE_TASK_PROCESS_STATE_RUNNABLE);
  EXPECT_EQ(iree_task_process_slist_get_next(head), &c);
  EXPECT_EQ(iree_task_process_slist_get_next(tail), nullptr);
}

TEST(ProcessTest, CancelSuspendedCallsCompletion) {
  CompletionRecord record;
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/1, 1,
                               &process);
  process.completion_fn = completion_record;
  process.user_data = &record;

  iree_task_process_t* head = nullptr;
  iree_task_process_t* tail = nullptr;
  iree_task_process_cancel(&process, &head, &tail);

  EXPECT_TRUE(record.called);
  EXPECT_EQ(record.status_code, IREE_STATUS_CANCELLED);
}

//===----------------------------------------------------------------------===//
// Concurrent wake tests
//===----------------------------------------------------------------------===//

TEST(ProcessTest, ConcurrentWake) {
  // Multiple threads racing to wake a process with suspend_count=N.
  // Exactly one should succeed (return true).
  static constexpr int kWakerCount = 8;
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate,
                               /*suspend_count=*/kWakerCount, 1, &process);

  iree_atomic_int32_t activation_count = IREE_ATOMIC_VAR_INIT(0);
  std::thread threads[kWakerCount];
  for (int i = 0; i < kWakerCount; ++i) {
    threads[i] = std::thread([&]() {
      if (iree_task_process_wake(&process)) {
        iree_atomic_fetch_add(&activation_count, 1, iree_memory_order_relaxed);
      }
    });
  }
  for (auto& t : threads) t.join();

  // Exactly one thread should have activated the process.
  EXPECT_EQ(iree_atomic_load(&activation_count, iree_memory_order_relaxed), 1);
  EXPECT_EQ(iree_task_process_state(&process),
            IREE_TASK_PROCESS_STATE_RUNNABLE);
}

TEST(ProcessTest, ConcurrentErrorReporting) {
  // Multiple threads race to report errors. Only the first should stick.
  static constexpr int kReporterCount = 8;
  iree_task_process_t process;
  iree_task_process_initialize(drain_immediate, /*suspend_count=*/0, 1,
                               &process);

  std::thread threads[kReporterCount];
  for (int i = 0; i < kReporterCount; ++i) {
    threads[i] = std::thread([&, i]() {
      iree_task_process_report_error(
          &process,
          iree_make_status(IREE_STATUS_INTERNAL, "error from thread %d", i));
    });
  }
  for (auto& t : threads) t.join();

  EXPECT_TRUE(iree_task_process_has_error(&process));
  EXPECT_TRUE(iree_task_process_is_terminal(&process));

  // Complete to clean up the error status.
  CompletionRecord record;
  process.completion_fn = completion_record;
  process.user_data = &record;
  iree_task_process_t* head = nullptr;
  iree_task_process_t* tail = nullptr;
  iree_task_process_complete(&process, &head, &tail);
  EXPECT_TRUE(record.called);
  EXPECT_EQ(record.status_code, IREE_STATUS_INTERNAL);
}

}  // namespace
