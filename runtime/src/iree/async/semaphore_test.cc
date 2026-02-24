// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/semaphore.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

//===----------------------------------------------------------------------===//
// Test helpers
//===----------------------------------------------------------------------===//

// Test callback that records invocations.
struct TimepointCallback {
  std::atomic<int> call_count{0};
  std::atomic<uint64_t> last_value{0};
  iree_status_code_t last_status_code{IREE_STATUS_OK};

  static void Invoke(void* user_data,
                     iree_async_semaphore_timepoint_t* timepoint,
                     iree_status_t status) {
    auto* self = static_cast<TimepointCallback*>(user_data);
    self->call_count++;
    self->last_value = timepoint->minimum_value;
    self->last_status_code = iree_status_code(status);
    iree_status_free(status);
  }
};

// Creates a frontier with the given entries.
class FrontierBuilder {
 public:
  FrontierBuilder() = default;

  FrontierBuilder& Add(iree_async_axis_t axis, uint64_t epoch) {
    entries_.push_back({axis, epoch});
    return *this;
  }

  // Returns a pointer to a stack-allocated frontier.
  // Only valid until the next call or destruction.
  iree_async_frontier_t* Build() {
    // Sort by axis.
    std::sort(entries_.begin(), entries_.end(),
              [](const auto& a, const auto& b) { return a.axis < b.axis; });

    // Allocate space for header + entries.
    buffer_.resize(sizeof(iree_async_frontier_t) +
                   entries_.size() * sizeof(iree_async_frontier_entry_t));
    auto* frontier = reinterpret_cast<iree_async_frontier_t*>(buffer_.data());
    iree_async_frontier_initialize(frontier,
                                   static_cast<uint8_t>(entries_.size()));
    for (size_t i = 0; i < entries_.size(); ++i) {
      frontier->entries[i] = entries_[i];
    }
    return frontier;
  }

 private:
  std::vector<iree_async_frontier_entry_t> entries_;
  std::vector<uint8_t> buffer_;
};

//===----------------------------------------------------------------------===//
// Create / Destroy
//===----------------------------------------------------------------------===//

TEST(SemaphoreTest, CreateWithInitialValue) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      42, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));
  EXPECT_EQ(iree_async_semaphore_query(sem), 42u);
  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, CreateWithZeroInitialValue) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));
  EXPECT_EQ(iree_async_semaphore_query(sem), 0u);
  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, RetainRelease) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));
  iree_async_semaphore_retain(sem);
  iree_async_semaphore_release(sem);  // First release.
  iree_async_semaphore_release(sem);  // Second release, destroys.
}

//===----------------------------------------------------------------------===//
// Signal / Query
//===----------------------------------------------------------------------===//

TEST(SemaphoreTest, SignalAdvancesValue) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, 5, nullptr));
  EXPECT_EQ(iree_async_semaphore_query(sem), 5u);

  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, 10, nullptr));
  EXPECT_EQ(iree_async_semaphore_query(sem), 10u);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, SignalLessThanCurrentFails) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      10, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  iree_status_t status = iree_async_semaphore_signal(sem, 5, nullptr);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);

  // Value unchanged.
  EXPECT_EQ(iree_async_semaphore_query(sem), 10u);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, SignalEqualToCurrentFails) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      10, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  iree_status_t status = iree_async_semaphore_signal(sem, 10, nullptr);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, SignalLargeJump) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, UINT64_MAX, nullptr));
  EXPECT_EQ(iree_async_semaphore_query(sem), UINT64_MAX);

  iree_async_semaphore_release(sem);
}

//===----------------------------------------------------------------------===//
// Timepoints
//===----------------------------------------------------------------------===//

TEST(SemaphoreTest, TimepointImmediatelySatisfied) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      10, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  TimepointCallback callback;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointCallback::Invoke;
  timepoint.user_data = &callback;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 5, &timepoint));

  // Callback fired immediately (value 10 >= 5).
  EXPECT_EQ(callback.call_count, 1);
  EXPECT_EQ(callback.last_status_code, IREE_STATUS_OK);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, TimepointPendsThenSatisfied) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  TimepointCallback callback;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointCallback::Invoke;
  timepoint.user_data = &callback;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 10, &timepoint));

  // Not yet satisfied.
  EXPECT_EQ(callback.call_count, 0);

  // Signal to 5 — still not satisfied.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, 5, nullptr));
  EXPECT_EQ(callback.call_count, 0);

  // Signal to 10 — now satisfied.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, 10, nullptr));
  EXPECT_EQ(callback.call_count, 1);
  EXPECT_EQ(callback.last_status_code, IREE_STATUS_OK);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, TimepointOvershoot) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  TimepointCallback callback;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointCallback::Invoke;
  timepoint.user_data = &callback;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 10, &timepoint));
  EXPECT_EQ(callback.call_count, 0);

  // Signal to 100 — overshoots the target, still fires.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, 100, nullptr));
  EXPECT_EQ(callback.call_count, 1);
  EXPECT_EQ(callback.last_status_code, IREE_STATUS_OK);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, MultipleTimepoints) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  TimepointCallback cb1, cb2, cb3;
  iree_async_semaphore_timepoint_t tp1, tp2, tp3;
  tp1.callback = TimepointCallback::Invoke;
  tp1.user_data = &cb1;
  tp2.callback = TimepointCallback::Invoke;
  tp2.user_data = &cb2;
  tp3.callback = TimepointCallback::Invoke;
  tp3.user_data = &cb3;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 5, &tp1));
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 10, &tp2));
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 15, &tp3));

  EXPECT_EQ(cb1.call_count, 0);
  EXPECT_EQ(cb2.call_count, 0);
  EXPECT_EQ(cb3.call_count, 0);

  // Signal to 7 — only tp1 satisfied.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, 7, nullptr));
  EXPECT_EQ(cb1.call_count, 1);
  EXPECT_EQ(cb2.call_count, 0);
  EXPECT_EQ(cb3.call_count, 0);

  // Signal to 15 — tp2 and tp3 satisfied.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, 15, nullptr));
  EXPECT_EQ(cb1.call_count, 1);
  EXPECT_EQ(cb2.call_count, 1);
  EXPECT_EQ(cb3.call_count, 1);

  iree_async_semaphore_release(sem);
}

//===----------------------------------------------------------------------===//
// Cancel
//===----------------------------------------------------------------------===//

TEST(SemaphoreTest, CancelPendingTimepoint) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  TimepointCallback callback;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointCallback::Invoke;
  timepoint.user_data = &callback;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 10, &timepoint));
  EXPECT_EQ(callback.call_count, 0);

  // Cancel before satisfaction.
  iree_async_semaphore_cancel_timepoint(sem, &timepoint);

  // Signal past the target — callback should NOT fire.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, 100, nullptr));
  EXPECT_EQ(callback.call_count, 0);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, CancelAlreadyFiredIsNoOp) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      10, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  TimepointCallback callback;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointCallback::Invoke;
  timepoint.user_data = &callback;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 5, &timepoint));
  EXPECT_EQ(callback.call_count, 1);  // Already fired.

  // Cancel after firing — should be a no-op, no crash.
  iree_async_semaphore_cancel_timepoint(sem, &timepoint);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, DoubleCancelIsSafe) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  TimepointCallback callback;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointCallback::Invoke;
  timepoint.user_data = &callback;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 10, &timepoint));
  iree_async_semaphore_cancel_timepoint(sem, &timepoint);
  iree_async_semaphore_cancel_timepoint(sem, &timepoint);  // Second cancel.

  EXPECT_EQ(callback.call_count, 0);

  iree_async_semaphore_release(sem);
}

//===----------------------------------------------------------------------===//
// Failure
//===----------------------------------------------------------------------===//

TEST(SemaphoreTest, FailDispatchesPendingTimepoints) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  TimepointCallback cb1, cb2;
  iree_async_semaphore_timepoint_t tp1, tp2;
  tp1.callback = TimepointCallback::Invoke;
  tp1.user_data = &cb1;
  tp2.callback = TimepointCallback::Invoke;
  tp2.user_data = &cb2;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 10, &tp1));
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 20, &tp2));

  EXPECT_EQ(cb1.call_count, 0);
  EXPECT_EQ(cb2.call_count, 0);

  // Fail the semaphore.
  iree_async_semaphore_fail(
      sem, iree_make_status(IREE_STATUS_ABORTED, "test failure"));

  // Both timepoints fired with failure.
  EXPECT_EQ(cb1.call_count, 1);
  EXPECT_EQ(cb1.last_status_code, IREE_STATUS_ABORTED);
  EXPECT_EQ(cb2.call_count, 1);
  EXPECT_EQ(cb2.last_status_code, IREE_STATUS_ABORTED);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, FailThenAcquireTimepoint) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  // Fail first.
  iree_async_semaphore_fail(sem,
                            iree_make_status(IREE_STATUS_ABORTED, "failed"));

  // Then acquire timepoint — should fire immediately with failure.
  TimepointCallback callback;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointCallback::Invoke;
  timepoint.user_data = &callback;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 10, &timepoint));
  EXPECT_EQ(callback.call_count, 1);
  EXPECT_EQ(callback.last_status_code, IREE_STATUS_ABORTED);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, FirstFailureWins) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  // First failure.
  iree_async_semaphore_fail(sem,
                            iree_make_status(IREE_STATUS_ABORTED, "first"));

  // Second failure (should be ignored).
  iree_async_semaphore_fail(sem,
                            iree_make_status(IREE_STATUS_CANCELLED, "second"));

  // Acquire timepoint — should get the first failure status.
  TimepointCallback callback;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointCallback::Invoke;
  timepoint.user_data = &callback;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 10, &timepoint));
  EXPECT_EQ(callback.last_status_code, IREE_STATUS_ABORTED);  // First failure.

  iree_async_semaphore_release(sem);
}

//===----------------------------------------------------------------------===//
// Destroy with pending timepoints
//===----------------------------------------------------------------------===//

TEST(SemaphoreTest, DestroyWithPendingTimepoints) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  TimepointCallback callback;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointCallback::Invoke;
  timepoint.user_data = &callback;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(sem, 10, &timepoint));
  EXPECT_EQ(callback.call_count, 0);

  // Destroy while timepoint is pending.
  iree_async_semaphore_release(sem);

  // Callback fired with CANCELLED.
  EXPECT_EQ(callback.call_count, 1);
  EXPECT_EQ(callback.last_status_code, IREE_STATUS_CANCELLED);
}

//===----------------------------------------------------------------------===//
// Frontier tracking
//===----------------------------------------------------------------------===//

TEST(SemaphoreTest, SignalWithFrontier) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  iree_async_axis_t axis_a = iree_async_axis_make_queue(1, 0, 0, 0);
  iree_async_axis_t axis_b = iree_async_axis_make_queue(1, 0, 1, 0);

  FrontierBuilder fb;
  IREE_ASSERT_OK(
      iree_async_semaphore_signal(sem, 1, fb.Add(axis_a, 10).Build()));

  // Query the accumulated frontier.
  uint8_t storage[sizeof(iree_async_frontier_t) +
                  2 * sizeof(iree_async_frontier_entry_t)];
  auto* out = reinterpret_cast<iree_async_frontier_t*>(storage);
  uint8_t count = iree_async_semaphore_query_frontier(sem, out, 2);

  EXPECT_EQ(count, 1u);
  EXPECT_EQ(out->entry_count, 1u);
  EXPECT_EQ(out->entries[0].axis, axis_a);
  EXPECT_EQ(out->entries[0].epoch, 10u);

  // Signal with another frontier — should merge.
  IREE_ASSERT_OK(
      iree_async_semaphore_signal(sem, 2, fb.Add(axis_b, 5).Build()));
  count = iree_async_semaphore_query_frontier(sem, out, 2);

  EXPECT_EQ(count, 2u);
  // Entries are sorted by axis.
  EXPECT_EQ(out->entries[0].axis, axis_a);
  EXPECT_EQ(out->entries[0].epoch, 10u);
  EXPECT_EQ(out->entries[1].axis, axis_b);
  EXPECT_EQ(out->entries[1].epoch, 5u);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, FrontierMergeMaxEpoch) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 0, 0);

  FrontierBuilder fb1;
  IREE_ASSERT_OK(
      iree_async_semaphore_signal(sem, 1, fb1.Add(axis, 10).Build()));

  FrontierBuilder fb2;
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, 2, fb2.Add(axis, 5).Build()));

  // Merged frontier should have max(10, 5) = 10.
  uint8_t storage[sizeof(iree_async_frontier_t) +
                  sizeof(iree_async_frontier_entry_t)];
  auto* out = reinterpret_cast<iree_async_frontier_t*>(storage);
  iree_async_semaphore_query_frontier(sem, out, 1);

  EXPECT_EQ(out->entries[0].epoch, 10u);  // Max of 10 and 5.

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, SignalWithNullFrontier) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, 1, nullptr));

  // Frontier should be empty.
  EXPECT_EQ(iree_async_semaphore_query_frontier(sem, nullptr, 0), 0u);

  iree_async_semaphore_release(sem);
}

//===----------------------------------------------------------------------===//
// Tainting
//===----------------------------------------------------------------------===//

TEST(SemaphoreTest, InitialValueIsUntainted) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      10, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  EXPECT_EQ(iree_async_semaphore_query_untainted_value(sem), 10u);
  EXPECT_FALSE(iree_async_semaphore_is_value_tainted(sem, 10));
  EXPECT_FALSE(iree_async_semaphore_is_value_tainted(sem, 5));

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, RegularSignalDoesNotAdvanceUntainted) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  IREE_ASSERT_OK(iree_async_semaphore_signal(sem, 10, nullptr));

  // Regular signal doesn't advance untainted watermark.
  EXPECT_EQ(iree_async_semaphore_query_untainted_value(sem), 0u);
  EXPECT_TRUE(iree_async_semaphore_is_value_tainted(sem, 10));
  EXPECT_TRUE(iree_async_semaphore_is_value_tainted(sem, 1));
  EXPECT_FALSE(iree_async_semaphore_is_value_tainted(sem, 0));

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, SignalUntaintedAdvancesWatermark) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  IREE_ASSERT_OK(iree_async_semaphore_signal_untainted(sem, 10, nullptr));

  EXPECT_EQ(iree_async_semaphore_query_untainted_value(sem), 10u);
  EXPECT_FALSE(iree_async_semaphore_is_value_tainted(sem, 10));
  EXPECT_FALSE(iree_async_semaphore_is_value_tainted(sem, 5));
  EXPECT_TRUE(iree_async_semaphore_is_value_tainted(sem, 11));

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, MarkTaintedAbove) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  // Advance timeline and untainted together.
  IREE_ASSERT_OK(iree_async_semaphore_signal_untainted(sem, 10, nullptr));
  EXPECT_EQ(iree_async_semaphore_query_untainted_value(sem), 10u);

  // Mark tainted above 5 (reduces the untainted watermark).
  iree_async_semaphore_mark_tainted_above(sem, 5);
  EXPECT_EQ(iree_async_semaphore_query_untainted_value(sem), 5u);

  EXPECT_FALSE(iree_async_semaphore_is_value_tainted(sem, 5));
  EXPECT_TRUE(iree_async_semaphore_is_value_tainted(sem, 6));
  EXPECT_TRUE(iree_async_semaphore_is_value_tainted(sem, 10));

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, MarkTaintedAboveOnlyDecreases) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  // Start at untainted 10.
  IREE_ASSERT_OK(iree_async_semaphore_signal_untainted(sem, 10, nullptr));

  // Try to mark tainted above 15 — should have no effect (15 > 10).
  iree_async_semaphore_mark_tainted_above(sem, 15);
  EXPECT_EQ(iree_async_semaphore_query_untainted_value(sem), 10u);

  iree_async_semaphore_release(sem);
}

//===----------------------------------------------------------------------===//
// Export primitive
//===----------------------------------------------------------------------===//

TEST(SemaphoreTest, ExportPrimitiveUnavailable) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  iree_async_primitive_t primitive;
  iree_status_t status =
      iree_async_semaphore_export_primitive(sem, 10, &primitive);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNAVAILABLE);
  iree_status_free(status);

  iree_async_semaphore_release(sem);
}

//===----------------------------------------------------------------------===//
// Concurrency
//===----------------------------------------------------------------------===//

TEST(SemaphoreTest, ConcurrentSignals) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  constexpr int kNumThreads = 8;
  constexpr int kSignalsPerThread = 1000;

  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([sem, t, kSignalsPerThread]() {
      for (int i = 0; i < kSignalsPerThread; ++i) {
        uint64_t value = t * kSignalsPerThread + i + 1;
        // Try to signal — may fail if another thread signaled higher.
        iree_status_t status = iree_async_semaphore_signal(sem, value, nullptr);
        iree_status_free(status);  // Ignore errors.
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Final value should be somewhere in the range, and query should work.
  uint64_t final_value = iree_async_semaphore_query(sem);
  EXPECT_GT(final_value, 0u);

  iree_async_semaphore_release(sem);
}

TEST(SemaphoreTest, ConcurrentTimepointAcquisitionAndSignal) {
  iree_async_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem));

  constexpr int kNumTimepoints = 100;

  struct ThreadData {
    TimepointCallback callback;
    iree_async_semaphore_timepoint_t timepoint;
  };
  std::vector<ThreadData> data(kNumTimepoints);

  // Start threads that acquire timepoints.
  std::vector<std::thread> acquire_threads;
  for (int i = 0; i < kNumTimepoints; ++i) {
    data[i].timepoint.callback = TimepointCallback::Invoke;
    data[i].timepoint.user_data = &data[i].callback;
    acquire_threads.emplace_back([sem, &data, i]() {
      IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
          sem, i + 1, &data[i].timepoint));
    });
  }

  // Signal thread.
  std::thread signal_thread([sem, kNumTimepoints]() {
    for (int i = 1; i <= kNumTimepoints; ++i) {
      iree_status_t status = iree_async_semaphore_signal(sem, i, nullptr);
      // May fail if we signal out of order, that's OK.
      iree_status_free(status);
    }
  });

  for (auto& t : acquire_threads) {
    t.join();
  }
  signal_thread.join();

  // Ensure signal is past all timepoints.
  iree_status_t status =
      iree_async_semaphore_signal(sem, kNumTimepoints + 1, nullptr);
  iree_status_free(status);

  // All timepoints should have fired.
  for (int i = 0; i < kNumTimepoints; ++i) {
    EXPECT_EQ(data[i].callback.call_count, 1);
  }

  iree_async_semaphore_release(sem);
}

}  // namespace
}  // namespace iree
