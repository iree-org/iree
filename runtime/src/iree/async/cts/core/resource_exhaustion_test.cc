// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for resource exhaustion handling.
//
// Tests verify that the proactor handles resource limits gracefully:
// - Submit returns appropriate errors when resources are exhausted
// - Operations complete correctly even under heavy load
// - No resource leaks after all operations complete

#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/operations/scheduling.h"

namespace iree::async::cts {

class ResourceExhaustionTest : public CtsTestBase<> {};

//===----------------------------------------------------------------------===//
// Concurrent operations stress tests
//===----------------------------------------------------------------------===//

// Submit many timers concurrently and verify all complete correctly.
// This tests that the proactor handles high concurrency without leaks.
TEST_P(ResourceExhaustionTest, ManyConcurrentTimers) {
  constexpr int kNumTimers = 100;

  std::vector<iree_async_timer_operation_t> timers(kNumTimers);
  std::vector<CompletionTracker> trackers(kNumTimers);

  // Submit all timers with immediate deadline.
  for (int i = 0; i < kNumTimers; ++i) {
    memset(&timers[i], 0, sizeof(timers[i]));
    timers[i].base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
    timers[i].deadline_ns = iree_time_now();  // Immediate.
    timers[i].base.completion_fn = CompletionTracker::Callback;
    timers[i].base.user_data = &trackers[i];

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timers[i].base));
  }

  // Poll until all complete.
  PollUntil(/*min_completions=*/kNumTimers,
            /*total_budget=*/iree_make_duration_ms(10000));

  // Verify all completed successfully.
  int completed_count = 0;
  for (int i = 0; i < kNumTimers; ++i) {
    if (trackers[i].call_count > 0) {
      ++completed_count;
      IREE_EXPECT_OK(trackers[i].ConsumeStatus()) << "Timer " << i << " failed";
    }
  }
  EXPECT_EQ(completed_count, kNumTimers) << "Not all timers completed";
}

// Submit and immediately cancel many operations.
// This tests that cancel under load doesn't leak resources.
TEST_P(ResourceExhaustionTest, ManyConcurrentCancellations) {
  constexpr int kNumTimers = 50;

  std::vector<iree_async_timer_operation_t> timers(kNumTimers);
  std::vector<CompletionTracker> trackers(kNumTimers);

  // Submit timers with future deadline (so they're pending).
  for (int i = 0; i < kNumTimers; ++i) {
    memset(&timers[i], 0, sizeof(timers[i]));
    timers[i].base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
    timers[i].deadline_ns = iree_time_now() + iree_make_duration_ms(60000);
    timers[i].base.completion_fn = CompletionTracker::Callback;
    timers[i].base.user_data = &trackers[i];

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timers[i].base));
  }

  // Cancel all of them.
  for (int i = 0; i < kNumTimers; ++i) {
    IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &timers[i].base));
  }

  // Poll until all complete (with CANCELLED status).
  PollUntil(/*min_completions=*/kNumTimers,
            /*total_budget=*/iree_make_duration_ms(10000));

  // Verify all completed (with any status - OK or CANCELLED depending on race).
  int completed_count = 0;
  for (int i = 0; i < kNumTimers; ++i) {
    if (trackers[i].call_count > 0) {
      ++completed_count;
      // Consume status to avoid leak - could be OK or CANCELLED.
      iree_status_ignore(trackers[i].ConsumeStatus());
    }
  }
  EXPECT_EQ(completed_count, kNumTimers)
      << "Not all cancelled timers completed";
}

// Submit NOP operations in rapid succession.
// NOPs are the simplest operations and stress the submit/complete path.
TEST_P(ResourceExhaustionTest, RapidNopSubmissions) {
  constexpr int kNumNops = 200;

  std::vector<iree_async_nop_operation_t> nops(kNumNops);
  std::vector<CompletionTracker> trackers(kNumNops);

  // Submit all NOPs.
  for (int i = 0; i < kNumNops; ++i) {
    memset(&nops[i], 0, sizeof(nops[i]));
    nops[i].base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
    nops[i].base.completion_fn = CompletionTracker::Callback;
    nops[i].base.user_data = &trackers[i];

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &nops[i].base));
  }

  // Poll until all complete.
  PollUntil(/*min_completions=*/kNumNops,
            /*total_budget=*/iree_make_duration_ms(10000));

  // Verify all completed.
  int completed_count = 0;
  for (int i = 0; i < kNumNops; ++i) {
    if (trackers[i].call_count > 0) {
      ++completed_count;
      IREE_EXPECT_OK(trackers[i].ConsumeStatus()) << "NOP " << i << " failed";
    }
  }
  EXPECT_EQ(completed_count, kNumNops) << "Not all NOPs completed";
}

// Interleaved submit and poll - simulates realistic workload.
TEST_P(ResourceExhaustionTest, InterleavedSubmitPoll) {
  constexpr int kIterations = 50;
  constexpr int kOpsPerIteration = 5;

  int total_submitted = 0;
  int total_completed = 0;

  for (int iter = 0; iter < kIterations; ++iter) {
    // Submit a batch of timers.
    std::vector<iree_async_timer_operation_t> timers(kOpsPerIteration);
    std::vector<CompletionTracker> trackers(kOpsPerIteration);

    for (int i = 0; i < kOpsPerIteration; ++i) {
      memset(&timers[i], 0, sizeof(timers[i]));
      timers[i].base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
      timers[i].deadline_ns = iree_time_now();  // Immediate.
      timers[i].base.completion_fn = CompletionTracker::Callback;
      timers[i].base.user_data = &trackers[i];

      IREE_ASSERT_OK(
          iree_async_proactor_submit_one(proactor_, &timers[i].base));
      ++total_submitted;
    }

    // Poll to process completions.
    PollUntil(/*min_completions=*/kOpsPerIteration,
              /*total_budget=*/iree_make_duration_ms(1000));

    // Count completions.
    for (int i = 0; i < kOpsPerIteration; ++i) {
      if (trackers[i].call_count > 0) {
        ++total_completed;
        IREE_EXPECT_OK(trackers[i].ConsumeStatus());
      }
    }
  }

  EXPECT_EQ(total_completed, total_submitted)
      << "Some operations did not complete";
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(ResourceExhaustionTest);

}  // namespace iree::async::cts
