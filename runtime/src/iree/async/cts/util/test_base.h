// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base class for proactor backend tests.
//
// Provides the test fixture and helper utilities shared by all proactor tests.
// Test logic lives in separate .cc files (socket/lifecycle_test.cc, etc.)
// which use CTS_REGISTER_TEST_SUITE() for self-registration.
//
// Key design:
//   - Fresh proactor per test via SetUp/TearDown (no state leakage)
//   - Capability-gated GTEST_SKIP instead of external EXCLUDED_TESTS lists
//   - Time-budgeted polling prevents slow-CI flakiness
//   - Link-time composition: test suites + backends linked together
//
// For socket tests, use SocketTestBase (in socket_test_base.h) which extends
// this class with socket-specific helpers like CreateListener().

#ifndef IREE_ASYNC_CTS_UTIL_TEST_BASE_H_
#define IREE_ASYNC_CTS_UTIL_TEST_BASE_H_

#include <functional>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Callback test utilities
//===----------------------------------------------------------------------===//

// Tracks completion callbacks for test assertions.
// Use with operations that fire a single final callback.
//
// IMPORTANT: Use ConsumeStatus() with IREE_EXPECT_OK or IREE_EXPECT_STATUS_IS
// to verify the completion status. This preserves error messages for debugging.
//
// Example:
//   CompletionTracker tracker;
//   // ... submit operation with tracker ...
//   PollUntil(1, ...);
//   IREE_EXPECT_OK(tracker.ConsumeStatus());
//   // or: IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
//   tracker.ConsumeStatus());
struct CompletionTracker {
  int call_count = 0;
  iree_status_t last_status = iree_ok_status();
  iree_async_completion_flags_t last_flags = 0;
  void* last_operation = nullptr;

  CompletionTracker() = default;
  ~CompletionTracker() { iree_status_ignore(last_status); }

  // Non-copyable (status ownership).
  CompletionTracker(const CompletionTracker&) = delete;
  CompletionTracker& operator=(const CompletionTracker&) = delete;

  // Returns and transfers ownership of the last status for testing.
  // The status matchers (IREE_EXPECT_OK, IREE_EXPECT_STATUS_IS) take ownership.
  // After calling this, last_status is reset to OK.
  iree_status_t ConsumeStatus() {
    iree_status_t result = last_status;
    last_status = iree_ok_status();
    return result;
  }

  // Resets the tracker for reuse (e.g., in a loop that resubmits operations).
  // Ignores any unconsumed status - call ConsumeStatus() first if you need to
  // verify the previous completion.
  void Reset() {
    last_status = iree_status_ignore(last_status);
    call_count = 0;
    last_flags = 0;
    last_operation = nullptr;
  }

  static void Callback(void* user_data, iree_async_operation_t* operation,
                       iree_status_t status,
                       iree_async_completion_flags_t flags) {
    auto* tracker = static_cast<CompletionTracker*>(user_data);
    ++tracker->call_count;
    // Free previous status before storing new one.
    tracker->last_status = iree_status_ignore(tracker->last_status);
    tracker->last_status = status;  // Take ownership - don't ignore!
    tracker->last_flags = flags;
    tracker->last_operation = operation;
  }
};

// Tracks ordered sequence of completion events (for multishot tests).
// Records each callback's status, flags, and bytes_transferred.
//
// IMPORTANT: Use ConsumeStatus(index) with IREE_EXPECT_OK or
// IREE_EXPECT_STATUS_IS to verify completion statuses. This preserves error
// messages for debugging.
struct CompletionLog {
  struct Entry {
    iree_status_t status = iree_ok_status();
    iree_async_completion_flags_t flags = 0;
    iree_host_size_t bytes_transferred = 0;  // For recv/send operations.

    Entry() = default;
    ~Entry() { iree_status_ignore(status); }

    // Non-copyable (status ownership).
    Entry(const Entry&) = delete;
    Entry& operator=(const Entry&) = delete;

    // Move constructor for vector storage.
    Entry(Entry&& other) noexcept
        : status(other.status),
          flags(other.flags),
          bytes_transferred(other.bytes_transferred) {
      other.status = iree_ok_status();
    }
    Entry& operator=(Entry&& other) noexcept {
      if (this != &other) {
        status = iree_status_ignore(status);
        status = other.status;
        flags = other.flags;
        bytes_transferred = other.bytes_transferred;
        other.status = iree_ok_status();
      }
      return *this;
    }
  };
  std::vector<Entry> entries;
  bool final_received = false;

  // Returns and transfers ownership of the status at |index| for testing.
  iree_status_t ConsumeStatus(size_t index) {
    if (index >= entries.size()) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "completion log index %" PRIu64
                              " out of range (size=%" PRIu64 ")",
                              (uint64_t)index, (uint64_t)entries.size());
    }
    iree_status_t result = entries[index].status;
    entries[index].status = iree_ok_status();
    return result;
  }

  static void Callback(void* user_data, iree_async_operation_t* operation,
                       iree_status_t status,
                       iree_async_completion_flags_t flags) {
    auto* log = static_cast<CompletionLog*>(user_data);
    Entry entry;
    entry.status = status;  // Take ownership - don't ignore!
    entry.flags = flags;
    entry.bytes_transferred = 0;
    log->entries.push_back(std::move(entry));
    if (!(flags & IREE_ASYNC_COMPLETION_FLAG_MORE)) {
      log->final_received = true;
    }
  }
};

//===----------------------------------------------------------------------===//
// Epoch-based notification wait
//===----------------------------------------------------------------------===//

// Context for epoch-based notification waits using iree_notification_await().
//
// Enables race-free cross-thread observation of async notification signals:
// the caller captures the baseline epoch BEFORE spawning the waiter thread,
// and the waiter uses iree_notification_await() with a predicate that checks
// whether the epoch has advanced. The iree_notification_t's prepare/commit/
// cancel protocol ensures no signal is missed regardless of thread scheduling.
//
// Usage:
//   iree_notification_t gate;
//   iree_notification_initialize(&gate);
//   EpochWaitContext context = {sink_notification,
//       iree_async_notification_query_epoch(sink_notification)};
//   std::thread waiter([&]() {
//       bool woken = iree_notification_await(&gate, epoch_advanced, &context,
//                                            timeout);
//   });
//   // ... trigger signal that will bump sink_notification's epoch ...
//   iree_notification_post(&gate, IREE_ALL_WAITERS);
//   waiter.join();
struct EpochWaitContext {
  iree_async_notification_t* notification;
  uint32_t baseline_epoch;
};

// Predicate for iree_notification_await: returns true when the async
// notification's epoch has advanced past the baseline captured at context
// creation time. Compatible with iree_condition_fn_t.
static bool epoch_advanced(void* arg) {
  auto* context = static_cast<EpochWaitContext*>(arg);
  return iree_async_notification_query_epoch(context->notification) !=
         context->baseline_epoch;
}

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

// Base class for all CTS tests. Parameterized on BackendInfo.
// Creates a fresh proactor in SetUp(), destroys in TearDown().
template <typename BaseType = ::testing::TestWithParam<BackendInfo>>
class CtsTestBase : public BaseType {
 protected:
  void SetUp() override {
    BackendInfo backend = this->GetParam();
    auto result = backend.factory();
    if (!result.ok() &&
        result.status().code() == iree::StatusCode::kUnavailable) {
      GTEST_SKIP() << "Backend '" << backend.name
                   << "' unavailable on this system";
    }
    IREE_ASSERT_OK_AND_ASSIGN(proactor_, std::move(result));
    capabilities_ = iree_async_proactor_query_capabilities(proactor_);
  }

  void TearDown() override {
    if (proactor_) {
      // Drain any in-flight operations before releasing. Without this,
      // operations referencing stack-local storage (common in tests) would
      // have dangling pointers when their callbacks fire during proactor
      // destruction.
      DrainPending();
      iree_async_proactor_release(proactor_);
      proactor_ = nullptr;
    }
  }

  // Poll until at least |min_completions| callbacks fire or |total_budget|
  // wall-clock time elapses. Uses a time-budget approach rather than iteration
  // count: this prevents spurious failures on slow CI (which might need many
  // poll rounds for kernel scheduling) while still failing fast when something
  // is genuinely broken.
  void PollUntil(iree_host_size_t min_completions,
                 iree_duration_t total_budget) {
    iree_host_size_t total = 0;
    // Convert to absolute deadline once at entry - no drift from here on.
    iree_time_t deadline_ns = iree_time_now() + total_budget;
    iree_timeout_t timeout = iree_make_deadline(deadline_ns);
    while (total < min_completions) {
      if (iree_time_now() >= deadline_ns) break;
      iree_host_size_t completed = 0;
      iree_status_t status =
          iree_async_proactor_poll(proactor_, timeout, &completed);
      if (!iree_status_is_deadline_exceeded(status)) {
        IREE_ASSERT_OK(status);
      } else {
        iree_status_ignore(status);
      }
      total += completed;
    }
    ASSERT_GE(total, min_completions)
        << "Expected at least " << min_completions << " completions but got "
        << total << " within budget of " << total_budget << " ns";
  }

  // Convenience: poll once with a short timeout, draining whatever is ready.
  void PollOnce() {
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_make_timeout_ms(100), &completed);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  // Drains all pending operations. Call this before releasing the proactor to
  // avoid dangling callbacks referencing stack operations. This is especially
  // important for multishot operations that may have completions in flight.
  //
  // Uses immediate timeout for each poll. Continues polling while CQEs are
  // available, stops when DEADLINE_EXCEEDED indicates no work is pending.
  // The budget limits total drain time for pathological cases.
  void DrainPending(iree_duration_t budget = 100 * 1000000ll) {  // 100ms max
    iree_time_t deadline_ns = iree_time_now() + budget;
    for (;;) {
      if (iree_time_now() >= deadline_ns) break;
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_immediate_timeout(), &completed);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        break;  // Nothing pending (no CQEs ready, no internal ops in-flight).
      } else if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        break;  // Unexpected error.
      }
      // Continue: OK means CQEs were processed or internal ops are pending.
    }
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_proactor_capabilities_t capabilities_ = 0;
};

}  // namespace iree::async::cts

#endif  // IREE_ASYNC_CTS_UTIL_TEST_BASE_H_
