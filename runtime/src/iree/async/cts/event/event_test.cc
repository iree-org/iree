// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for event operations.
//
// Events are the building block for cross-thread signaling. These tests verify
// that event creation, signaling, waiting, and reset work correctly across
// single-threaded and multi-threaded scenarios.

#include "iree/async/event.h"

#include <thread>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/operations/scheduling.h"

namespace iree::async::cts {

class EventTest : public CtsTestBase<> {};

// Create event, signal from same thread, poll - callback fires.
TEST_P(EventTest, SameThreadSignal) {
  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));

  iree_async_event_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
  wait_op.event = event;

  CompletionTracker tracker;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Signal the event from the same thread.
  IREE_ASSERT_OK(iree_async_event_set(event));

  // Poll should pick up the signaled event.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(100));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_EQ(tracker.last_operation, &wait_op.base);

  iree_async_event_release(event);
}

// Create event, signal from another thread, poll - callback fires.
TEST_P(EventTest, CrossThreadSignal) {
  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));

  iree_async_event_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
  wait_op.event = event;

  CompletionTracker tracker;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Signal the event from another thread. The submit is synchronous, so the
  // wait is already registered by the time the signaler starts.
  std::thread signaler(
      [event]() { iree_status_ignore(iree_async_event_set(event)); });

  // Poll should pick up the signaled event.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(200));

  signaler.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_event_release(event);
}

// Multiple events, signal subset, verify correct callbacks.
TEST_P(EventTest, MultipleEventsPartialSignal) {
  constexpr int kEventCount = 3;
  iree_async_event_t* events[kEventCount] = {};
  iree_async_event_wait_operation_t wait_ops[kEventCount];
  CompletionTracker trackers[kEventCount];

  // Create events and wait operations.
  for (int i = 0; i < kEventCount; ++i) {
    IREE_ASSERT_OK(iree_async_event_create(proactor_, &events[i]));
    memset(&wait_ops[i], 0, sizeof(wait_ops[i]));
    wait_ops[i].base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
    wait_ops[i].event = events[i];
    wait_ops[i].base.completion_fn = CompletionTracker::Callback;
    wait_ops[i].base.user_data = &trackers[i];
  }

  // Submit all wait operations.
  iree_async_operation_t* ops[] = {&wait_ops[0].base, &wait_ops[1].base,
                                   &wait_ops[2].base};
  iree_async_operation_list_t list = {ops, kEventCount};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Signal only events 0 and 2, leaving event 1 unsignaled.
  IREE_ASSERT_OK(iree_async_event_set(events[0]));
  IREE_ASSERT_OK(iree_async_event_set(events[2]));

  // Poll should pick up exactly 2 completions.
  PollUntil(/*min_completions=*/2, /*total_budget=*/iree_make_duration_ms(100));

  EXPECT_EQ(trackers[0].call_count, 1);
  IREE_EXPECT_OK(trackers[0].ConsumeStatus());
  EXPECT_EQ(trackers[1].call_count, 0);  // Not signaled.
  EXPECT_EQ(trackers[2].call_count, 1);
  IREE_EXPECT_OK(trackers[2].ConsumeStatus());

  // Now signal event 1.
  IREE_ASSERT_OK(iree_async_event_set(events[1]));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(100));

  EXPECT_EQ(trackers[1].call_count, 1);
  IREE_EXPECT_OK(trackers[1].ConsumeStatus());

  for (int i = 0; i < kEventCount; ++i) {
    iree_async_event_release(events[i]);
  }
}

// Event reset after signal, re-wait works.
TEST_P(EventTest, ResetAndReWait) {
  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));

  // First wait/signal cycle.
  {
    iree_async_event_wait_operation_t wait_op;
    memset(&wait_op, 0, sizeof(wait_op));
    wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
    wait_op.event = event;

    CompletionTracker tracker;
    wait_op.base.completion_fn = CompletionTracker::Callback;
    wait_op.base.user_data = &tracker;

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));
    IREE_ASSERT_OK(iree_async_event_set(event));
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(100));

    EXPECT_EQ(tracker.call_count, 1);
    IREE_EXPECT_OK(tracker.ConsumeStatus());

    // Event is automatically drained by the proactor when the wait completes
    // (e.g., via linked POLL_ADD+READ on io_uring).
  }

  // Second wait/signal cycle should work identically.
  {
    iree_async_event_wait_operation_t wait_op;
    memset(&wait_op, 0, sizeof(wait_op));
    wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
    wait_op.event = event;

    CompletionTracker tracker;
    wait_op.base.completion_fn = CompletionTracker::Callback;
    wait_op.base.user_data = &tracker;

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));
    IREE_ASSERT_OK(iree_async_event_set(event));
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(100));

    EXPECT_EQ(tracker.call_count, 1);
    IREE_EXPECT_OK(tracker.ConsumeStatus());
  }

  iree_async_event_release(event);
}

// Pre-signaled event: signal before wait submission still completes.
TEST_P(EventTest, PreSignaledEvent) {
  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));

  // Signal BEFORE submitting the wait.
  IREE_ASSERT_OK(iree_async_event_set(event));

  iree_async_event_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
  wait_op.event = event;

  CompletionTracker tracker;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Should complete immediately since the event was already signaled.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(100));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_event_release(event);
}

// Retain/release pair on event doesn't crash.
TEST_P(EventTest, RetainRelease) {
  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));

  // Retain bumps refcount.
  iree_async_event_retain(event);

  // Release decrements but shouldn't destroy (create holds a ref).
  iree_async_event_release(event);

  // Event should still be usable.
  IREE_ASSERT_OK(iree_async_event_set(event));

  // Final release destroys.
  iree_async_event_release(event);
}

CTS_REGISTER_TEST_SUITE(EventTest);

}  // namespace iree::async::cts
