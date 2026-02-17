// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async semaphore synchronous operations.
//
// Tests the semaphore API directly without proactor operations. These tests
// exercise timeline semantics, timepoint callbacks, frontier merging, and
// failure propagation using the software semaphore implementation.
//
// For proactor-based semaphore operations (SEMAPHORE_WAIT, SEMAPHORE_SIGNAL),
// see semaphore_async_test.cc.

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/frontier.h"
#include "iree/async/semaphore.h"

namespace iree::async::cts {

class SemaphoreSyncTest : public CtsTestBase<> {
 protected:
  // Helper to track timepoint callbacks.
  struct TimepointTracker {
    std::atomic<int> call_count{0};
    iree_status_t last_status = iree_ok_status();
    iree_async_semaphore_timepoint_t* last_timepoint = nullptr;

    ~TimepointTracker() { iree_status_ignore(last_status); }

    // Returns and transfers ownership of the last status for testing.
    iree_status_t ConsumeStatus() {
      iree_status_t result = last_status;
      last_status = iree_ok_status();
      return result;
    }

    void Reset() {
      last_status = iree_status_ignore(last_status);
      call_count.store(0, std::memory_order_release);
      last_timepoint = nullptr;
    }

    static void Callback(void* user_data,
                         iree_async_semaphore_timepoint_t* timepoint,
                         iree_status_t status) {
      auto* tracker = static_cast<TimepointTracker*>(user_data);
      tracker->call_count.fetch_add(1, std::memory_order_acq_rel);
      tracker->last_status = iree_status_ignore(tracker->last_status);
      tracker->last_status = status;
      tracker->last_timepoint = timepoint;
    }
  };
};

// Create semaphore at initial value, query returns it, release destroys.
TEST_P(SemaphoreSyncTest, CreateQueryRelease) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/42, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  EXPECT_EQ(iree_async_semaphore_query(semaphore), 42u);

  iree_async_semaphore_release(semaphore);
}

// Signal advances the timeline value.
TEST_P(SemaphoreSyncTest, SignalAdvancesValue) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  EXPECT_EQ(iree_async_semaphore_query(semaphore), 0u);

  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 5, /*frontier=*/NULL));
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 5u);

  IREE_ASSERT_OK(
      iree_async_semaphore_signal(semaphore, 100, /*frontier=*/NULL));
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 100u);

  iree_async_semaphore_release(semaphore);
}

// Signal with value <= current returns INVALID_ARGUMENT.
TEST_P(SemaphoreSyncTest, SignalNonMonotonicFails) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/10, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Signal to same value fails.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_semaphore_signal(semaphore, 10, NULL));

  // Signal to lower value fails.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_semaphore_signal(semaphore, 5, NULL));

  // Value should still be 10.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 10u);

  iree_async_semaphore_release(semaphore);
}

// Signal with frontier merges entries, query_frontier returns them.
TEST_P(SemaphoreSyncTest, SignalWithFrontier) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Create a frontier with some entries (must be sorted by axis).
  uint8_t frontier_storage[sizeof(iree_async_frontier_t) +
                           2 * sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* frontier = (iree_async_frontier_t*)frontier_storage;
  iree_async_frontier_initialize(frontier, 2);
  frontier->entries[0].axis = 0x1234;
  frontier->entries[0].epoch = 100;
  frontier->entries[1].axis = 0x5678;
  frontier->entries[1].epoch = 200;

  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 1, frontier));

  // Query the frontier back.
  uint8_t result_storage[sizeof(iree_async_frontier_t) +
                         4 * sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* result = (iree_async_frontier_t*)result_storage;
  uint8_t count =
      iree_async_semaphore_query_frontier(semaphore, result, /*capacity=*/4);

  EXPECT_EQ(count, 2u);
  EXPECT_EQ(result->entry_count, 2u);
  EXPECT_EQ(result->entries[0].axis, 0x1234u);
  EXPECT_EQ(result->entries[0].epoch, 100u);
  EXPECT_EQ(result->entries[1].axis, 0x5678u);
  EXPECT_EQ(result->entries[1].epoch, 200u);

  iree_async_semaphore_release(semaphore);
}

// Frontier exceeding capacity returns RESOURCE_EXHAUSTED.
TEST_P(SemaphoreSyncTest, FrontierCapacityExhausted) {
  // Create semaphore with capacity for only 2 entries.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, /*frontier_capacity=*/2, iree_allocator_system(),
      &semaphore));

  // First signal with 2 entries succeeds (must be sorted by axis).
  uint8_t frontier1_storage[sizeof(iree_async_frontier_t) +
                            2 * sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* frontier1 = (iree_async_frontier_t*)frontier1_storage;
  iree_async_frontier_initialize(frontier1, 2);
  frontier1->entries[0].axis = 0x1;
  frontier1->entries[0].epoch = 1;
  frontier1->entries[1].axis = 0x2;
  frontier1->entries[1].epoch = 2;

  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 1, frontier1));

  // Second signal with a new axis exceeds capacity.
  uint8_t frontier2_storage[sizeof(iree_async_frontier_t) +
                            1 * sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* frontier2 = (iree_async_frontier_t*)frontier2_storage;
  iree_async_frontier_initialize(frontier2, 1);
  frontier2->entries[0].axis = 0x3;  // New axis
  frontier2->entries[0].epoch = 3;

  // Signal still advances the timeline, but returns RESOURCE_EXHAUSTED.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_async_semaphore_signal(semaphore, 2, frontier2));

  // Timeline still advanced despite frontier merge failure.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 2u);

  iree_async_semaphore_release(semaphore);
}

// Fail semaphore, subsequent waits get failure, second fail is no-op.
TEST_P(SemaphoreSyncTest, FailStickyStatus) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Set up a pending timepoint.
  TimepointTracker tracker;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointTracker::Callback;
  timepoint.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      semaphore, /*minimum_value=*/10, &timepoint));

  // Timepoint should be pending.
  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 0);

  // Fail the semaphore.
  iree_async_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_ABORTED, "test failure"));

  // Timepoint should have fired with the failure.
  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, tracker.ConsumeStatus());

  // New timepoint should immediately get the failure.
  tracker.Reset();
  iree_async_semaphore_timepoint_t timepoint2;
  timepoint2.callback = TimepointTracker::Callback;
  timepoint2.user_data = &tracker;
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      semaphore, /*minimum_value=*/20, &timepoint2));

  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, tracker.ConsumeStatus());

  // Second fail is ignored (no-op).
  iree_async_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_INTERNAL, "second failure"));

  iree_async_semaphore_release(semaphore);
}

// acquire_timepoint on already-reached value fires synchronously.
TEST_P(SemaphoreSyncTest, TimepointImmediateSatisfaction) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/10, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  TimepointTracker tracker;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointTracker::Callback;
  timepoint.user_data = &tracker;

  // Acquire timepoint for value already reached.
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      semaphore, /*minimum_value=*/5, &timepoint));

  // Should have fired immediately.
  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // Same for exact match.
  tracker.Reset();
  iree_async_semaphore_timepoint_t timepoint2;
  timepoint2.callback = TimepointTracker::Callback;
  timepoint2.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      semaphore, /*minimum_value=*/10, &timepoint2));

  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// acquire_timepoint pending, then signal fires it.
TEST_P(SemaphoreSyncTest, TimepointPendingThenSignal) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  TimepointTracker tracker;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointTracker::Callback;
  timepoint.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      semaphore, /*minimum_value=*/10, &timepoint));

  // Should be pending.
  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 0);

  // Signal to below target - still pending.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 5, NULL));
  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 0);

  // Signal to target - fires.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 10, NULL));
  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// Cancel pending timepoint, callback doesn't fire.
TEST_P(SemaphoreSyncTest, TimepointCancel) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  TimepointTracker tracker;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointTracker::Callback;
  timepoint.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      semaphore, /*minimum_value=*/10, &timepoint));

  // Should be pending.
  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 0);

  // Cancel the timepoint.
  iree_async_semaphore_cancel_timepoint(semaphore, &timepoint);

  // Callback should NOT fire.
  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 0);

  // Signal past the value - callback still shouldn't fire.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 20, NULL));
  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 0);

  iree_async_semaphore_release(semaphore);
}

// Cancel already-fired timepoint is no-op.
TEST_P(SemaphoreSyncTest, TimepointCancelAfterFire) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/10, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  TimepointTracker tracker;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointTracker::Callback;
  timepoint.user_data = &tracker;

  // Acquire for already-satisfied value - fires immediately.
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      semaphore, /*minimum_value=*/5, &timepoint));

  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // Cancel after fire should be no-op (not crash).
  iree_async_semaphore_cancel_timepoint(semaphore, &timepoint);

  // Call count should still be 1.
  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 1);

  iree_async_semaphore_release(semaphore);
}

// Multiple timepoints at different values, signal dispatches correct ones.
TEST_P(SemaphoreSyncTest, MultipleTimepoints) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  TimepointTracker tracker1, tracker2, tracker3;

  iree_async_semaphore_timepoint_t timepoint1;
  timepoint1.callback = TimepointTracker::Callback;
  timepoint1.user_data = &tracker1;

  iree_async_semaphore_timepoint_t timepoint2;
  timepoint2.callback = TimepointTracker::Callback;
  timepoint2.user_data = &tracker2;

  iree_async_semaphore_timepoint_t timepoint3;
  timepoint3.callback = TimepointTracker::Callback;
  timepoint3.user_data = &tracker3;

  // Register timepoints for values 5, 10, 15.
  IREE_ASSERT_OK(
      iree_async_semaphore_acquire_timepoint(semaphore, 5, &timepoint1));
  IREE_ASSERT_OK(
      iree_async_semaphore_acquire_timepoint(semaphore, 10, &timepoint2));
  IREE_ASSERT_OK(
      iree_async_semaphore_acquire_timepoint(semaphore, 15, &timepoint3));

  // All pending.
  EXPECT_EQ(tracker1.call_count.load(std::memory_order_acquire), 0);
  EXPECT_EQ(tracker2.call_count.load(std::memory_order_acquire), 0);
  EXPECT_EQ(tracker3.call_count.load(std::memory_order_acquire), 0);

  // Signal to 7: only timepoint1 (value 5) should fire.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 7, NULL));
  EXPECT_EQ(tracker1.call_count.load(std::memory_order_acquire), 1);
  EXPECT_EQ(tracker2.call_count.load(std::memory_order_acquire), 0);
  EXPECT_EQ(tracker3.call_count.load(std::memory_order_acquire), 0);
  IREE_EXPECT_OK(tracker1.ConsumeStatus());

  // Signal to 12: only timepoint2 (value 10) should fire.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 12, NULL));
  EXPECT_EQ(tracker2.call_count.load(std::memory_order_acquire), 1);
  EXPECT_EQ(tracker3.call_count.load(std::memory_order_acquire), 0);
  IREE_EXPECT_OK(tracker2.ConsumeStatus());

  // Signal to 20: timepoint3 (value 15) should fire.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 20, NULL));
  EXPECT_EQ(tracker3.call_count.load(std::memory_order_acquire), 1);
  IREE_EXPECT_OK(tracker3.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// signal_untainted advances watermark, is_value_tainted returns correct values.
TEST_P(SemaphoreSyncTest, TaintingWatermark) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Initial value 0 is untainted.
  EXPECT_FALSE(iree_async_semaphore_is_value_tainted(semaphore, 0));
  EXPECT_EQ(iree_async_semaphore_query_untainted_value(semaphore), 0u);

  // Regular signal (potentially tainted).
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 5, NULL));
  EXPECT_TRUE(iree_async_semaphore_is_value_tainted(semaphore, 5));
  EXPECT_EQ(iree_async_semaphore_query_untainted_value(semaphore), 0u);

  // signal_untainted advances the watermark.
  IREE_ASSERT_OK(iree_async_semaphore_signal_untainted(semaphore, 10, NULL));
  EXPECT_FALSE(iree_async_semaphore_is_value_tainted(semaphore, 10));
  EXPECT_EQ(iree_async_semaphore_query_untainted_value(semaphore), 10u);

  // Values below watermark are also untainted.
  EXPECT_FALSE(iree_async_semaphore_is_value_tainted(semaphore, 5));

  // Values above watermark are tainted.
  EXPECT_TRUE(iree_async_semaphore_is_value_tainted(semaphore, 11));

  // mark_tainted_above can lower the watermark.
  iree_async_semaphore_mark_tainted_above(semaphore, 7);
  EXPECT_EQ(iree_async_semaphore_query_untainted_value(semaphore), 7u);
  EXPECT_FALSE(iree_async_semaphore_is_value_tainted(semaphore, 7));
  EXPECT_TRUE(iree_async_semaphore_is_value_tainted(semaphore, 8));

  iree_async_semaphore_release(semaphore);
}

// Retain/release reference counting works correctly.
TEST_P(SemaphoreSyncTest, RetainRelease) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Initial ref count is 1.
  iree_async_semaphore_retain(semaphore);
  // Now ref count is 2.

  iree_async_semaphore_release(semaphore);
  // Now ref count is 1.

  // Can still use the semaphore.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 0u);

  iree_async_semaphore_release(semaphore);
  // Now ref count is 0, semaphore is destroyed.
}

// Timepoint fires correctly when signaled from another thread.
TEST_P(SemaphoreSyncTest, TimepointCrossThread) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  TimepointTracker tracker;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointTracker::Callback;
  timepoint.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      semaphore, /*minimum_value=*/10, &timepoint));

  // Signal from another thread.
  std::thread signaler([semaphore]() {
    iree_wait_until(iree_time_now() + iree_make_duration_ms(10));
    iree_status_t status = iree_async_semaphore_signal(semaphore, 10, NULL);
    IREE_CHECK_OK(status);
  });

  // Wait for the timepoint to fire (with generous budget).
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(5000);
  while (tracker.call_count.load(std::memory_order_acquire) == 0 &&
         iree_time_now() < deadline) {
    iree_thread_yield();
  }

  signaler.join();

  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// Semaphore destruction cancels pending timepoints.
TEST_P(SemaphoreSyncTest, DestroyWithPendingTimepoints) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  TimepointTracker tracker;
  iree_async_semaphore_timepoint_t timepoint;
  timepoint.callback = TimepointTracker::Callback;
  timepoint.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      semaphore, /*minimum_value=*/10, &timepoint));

  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 0);

  // Release (destroy) the semaphore with pending timepoint.
  iree_async_semaphore_release(semaphore);

  // Timepoint should have fired with CANCELLED.
  EXPECT_EQ(tracker.call_count.load(std::memory_order_acquire), 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
}

CTS_REGISTER_TEST_SUITE(SemaphoreSyncTest);

}  // namespace iree::async::cts
