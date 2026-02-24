// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for operation cancellation.
//
// These tests verify the cancel() API behavior:
// - Cancelled operations receive CANCELLED status in their callback
// - The callback ALWAYS fires (never lost) after cancel returns
// - Double-cancel is harmless (second cancel silently succeeds)
// - Cancel of already-completed operation is harmless

#include <thread>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/event.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"

namespace iree::async::cts {

class CancellationTest : public SocketTestBase<> {};

//===----------------------------------------------------------------------===//
// Timer cancellation tests
//===----------------------------------------------------------------------===//

// Cancel a pending timer before it fires.
TEST_P(CancellationTest, CancelPendingTimer) {
  iree_async_timer_operation_t timer;
  memset(&timer, 0, sizeof(timer));
  timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;

  // Set deadline far in the future so we have time to cancel.
  timer.deadline_ns = iree_time_now() + iree_make_duration_ms(10000);  // 10s

  CompletionTracker tracker;
  timer.base.completion_fn = CompletionTracker::Callback;
  timer.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timer.base));

  // Cancel the timer immediately.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &timer.base));

  // Poll to receive the cancellation callback.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  // The callback must have fired with CANCELLED status.
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
}

// Cancel an already-completed timer is harmless.
TEST_P(CancellationTest, CancelCompletedTimer) {
  iree_async_timer_operation_t timer;
  memset(&timer, 0, sizeof(timer));
  timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;

  // Immediate deadline - will complete right away.
  timer.deadline_ns = iree_time_now() - 1000000LL;  // 1ms ago

  CompletionTracker tracker;
  timer.base.completion_fn = CompletionTracker::Callback;
  timer.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timer.base));

  // Wait for completion.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));
  ASSERT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // Cancel after completion - should be harmless.
  // The cancel targets by user_data, but the operation is no longer pending,
  // so the kernel returns -ENOENT which we silently handle.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &timer.base));

  // Drain any pending CQEs (the cancel CQE itself).
  DrainPending(iree_make_duration_ms(100));

  // No additional callbacks should have occurred.
  EXPECT_EQ(tracker.call_count, 1);
}

// Double-cancel a timer - second cancel is harmless.
TEST_P(CancellationTest, DoubleCancelTimer) {
  iree_async_timer_operation_t timer;
  memset(&timer, 0, sizeof(timer));
  timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;

  // Far future deadline.
  timer.deadline_ns = iree_time_now() + iree_make_duration_ms(10000);  // 10s

  CompletionTracker tracker;
  timer.base.completion_fn = CompletionTracker::Callback;
  timer.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timer.base));

  // First cancel.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &timer.base));

  // Second cancel - should succeed (kernel returns -ENOENT/-EALREADY).
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &timer.base));

  // Poll to receive the cancellation callback.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  // Drain any remaining CQEs.
  DrainPending(iree_make_duration_ms(100));

  // Exactly one callback should have fired.
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
}

//===----------------------------------------------------------------------===//
// Event wait cancellation tests
//===----------------------------------------------------------------------===//

// Cancel a pending event wait before the event is signaled.
TEST_P(CancellationTest, CancelPendingEventWait) {
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

  // Cancel immediately (event never signaled).
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

  // Poll to receive the cancellation callback.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());

  iree_async_event_release(event);
}

// Cancel an already-completed event wait is harmless.
TEST_P(CancellationTest, CancelCompletedEventWait) {
  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));

  // Signal before submitting the wait — it will complete immediately.
  IREE_ASSERT_OK(iree_async_event_set(event));

  iree_async_event_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
  wait_op.event = event;

  CompletionTracker tracker;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Wait for completion.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));
  ASSERT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // Cancel after completion — should be harmless.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

  // Drain any pending CQEs from the cancel.
  DrainPending(iree_make_duration_ms(100));

  // No additional callbacks should have occurred.
  EXPECT_EQ(tracker.call_count, 1);

  iree_async_event_release(event);
}

// Double-cancel an event wait — second cancel is harmless.
TEST_P(CancellationTest, DoubleCancelEventWait) {
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

  // First cancel.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

  // Second cancel — should succeed harmlessly.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

  // Poll to receive the cancellation callback.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  // Drain any remaining CQEs.
  DrainPending(iree_make_duration_ms(100));

  // Exactly one callback should have fired.
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());

  iree_async_event_release(event);
}

// Cancel an event wait, then verify the event is still usable for a new wait.
TEST_P(CancellationTest, CancelEventWaitEventStillUsable) {
  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));

  // First wait — cancel it.
  {
    iree_async_event_wait_operation_t wait_op;
    memset(&wait_op, 0, sizeof(wait_op));
    wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
    wait_op.event = event;

    CompletionTracker tracker;
    wait_op.base.completion_fn = CompletionTracker::Callback;
    wait_op.base.user_data = &tracker;

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));
    IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
    ASSERT_EQ(tracker.call_count, 1);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
  }

  // Second wait on the same event — should complete normally when signaled.
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
              /*total_budget=*/iree_make_duration_ms(500));
    EXPECT_EQ(tracker.call_count, 1);
    IREE_EXPECT_OK(tracker.ConsumeStatus());
  }

  iree_async_event_release(event);
}

// Cancel races with event signal — either outcome is valid.
TEST_P(CancellationTest, CancelEventWaitRacesWithSignal) {
  static constexpr int kIterations = 10;

  for (int iter = 0; iter < kIterations; ++iter) {
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

    // Signal and cancel back-to-back — race between completion and cancel.
    IREE_ASSERT_OK(iree_async_event_set(event));
    IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(500));
    DrainPending(iree_make_duration_ms(100));

    // Exactly one callback, either CANCELLED or OK.
    EXPECT_EQ(tracker.call_count, 1) << "Iteration " << iter;
    {
      iree_status_t status = tracker.ConsumeStatus();
      if (!iree_status_is_ok(status) && !iree_status_is_cancelled(status)) {
        IREE_EXPECT_OK(status) << "Iteration " << iter;
      } else {
        iree_status_ignore(status);
      }
    }

    iree_async_event_release(event);
  }
}

// Cancel from a background thread while the main thread is polling.
// This tests that cancel() wakes a blocking poll() and that the cancelled
// operation callback fires promptly.
TEST_P(CancellationTest, CancelEventWaitFromBackgroundThread) {
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

  // Poll once to register the wait (drain the pending queue).
  PollOnce();

  // Cancel from a background thread while the main thread polls.
  std::thread canceler([this, &wait_op]() {
    // Brief delay to let the main thread enter poll().
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    iree_status_ignore(iree_async_proactor_cancel(proactor_, &wait_op.base));
  });

  // This will block until the background cancel wakes us.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(2000));

  canceler.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());

  iree_async_event_release(event);
}

//===----------------------------------------------------------------------===//
// Socket recv cancellation tests
//===----------------------------------------------------------------------===//

// Cancel a pending recv operation.
TEST_P(CancellationTest, CancelPendingRecv) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Submit a recv that will block (no data sent yet).
  char recv_buffer[128];
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  // Cancel the recv.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &recv_op.base));

  // Poll for the cancellation callback.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  EXPECT_EQ(recv_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, recv_tracker.ConsumeStatus());

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// Cancel a pending recv, then verify socket is still usable.
TEST_P(CancellationTest, CancelRecvSocketStillUsable) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Submit a recv that will block.
  char recv_buffer[128];
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  // Cancel the recv.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &recv_op.base));

  // Wait for cancellation.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));
  ASSERT_EQ(recv_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, recv_tracker.ConsumeStatus());

  // Now send data from client and receive it on server - socket should work.
  const char* message = "After cancel";
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)message, strlen(message));

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Submit new recv.
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_socket_recv_operation_t recv_op2;
  CompletionTracker recv_tracker2;
  InitRecvOperation(&recv_op2, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker2);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op2.base));

  // Poll for both send and recv.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(2000));

  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(recv_tracker2.call_count, 1);
  IREE_EXPECT_OK(recv_tracker2.ConsumeStatus());
  EXPECT_EQ(recv_op2.bytes_received, strlen(message));
  EXPECT_EQ(memcmp(recv_buffer, message, strlen(message)), 0);

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Socket accept cancellation tests
//===----------------------------------------------------------------------===//

// Cancel a pending accept operation.
TEST_P(CancellationTest, CancelPendingAccept) {
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Submit accept - no client connecting, so it will block.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Cancel the accept.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &accept_op.base));

  // Poll for cancellation callback.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  EXPECT_EQ(accept_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, accept_tracker.ConsumeStatus());
  EXPECT_EQ(accept_op.accepted_socket, nullptr);

  iree_async_socket_release(listener);
}

// Cancel accept, then verify listener is still usable.
TEST_P(CancellationTest, CancelAcceptListenerStillUsable) {
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Submit accept that will block.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Cancel the accept.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &accept_op.base));

  // Wait for cancellation.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));
  ASSERT_EQ(accept_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, accept_tracker.ConsumeStatus());

  // Submit a new accept.
  iree_async_socket_accept_operation_t accept_op2;
  CompletionTracker accept_tracker2;
  InitAcceptOperation(&accept_op2, listener, CompletionTracker::Callback,
                      &accept_tracker2);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op2.base));

  // Create a client and connect.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &client));

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  // Poll for both accept and connect.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(2000));

  EXPECT_EQ(accept_tracker2.call_count, 1);
  IREE_EXPECT_OK(accept_tracker2.ConsumeStatus());
  EXPECT_NE(accept_op2.accepted_socket, nullptr);

  EXPECT_EQ(connect_tracker.call_count, 1);
  IREE_EXPECT_OK(connect_tracker.ConsumeStatus());

  iree_async_socket_release(accept_op2.accepted_socket);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Connect cancellation tests
//===----------------------------------------------------------------------===//

// Cancel a pending connect operation.
// Note: Connect cancellation behavior may vary - the connection might complete
// before the cancel takes effect, or the cancel might succeed.
TEST_P(CancellationTest, CancelPendingConnect) {
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Create client but don't submit accept yet - connect will block.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &client));

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  // Cancel the connect.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &connect_op.base));

  // Poll for callback. Due to TCP handshake timing, the connect might complete
  // before the cancel takes effect. Either outcome is valid.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(2000));

  EXPECT_EQ(connect_tracker.call_count, 1);
  // Accept either CANCELLED (cancel won) or OK (connect completed first).
  {
    iree_status_t status = connect_tracker.ConsumeStatus();
    if (!iree_status_is_ok(status) && !iree_status_is_cancelled(status)) {
      IREE_EXPECT_OK(status);
    } else {
      iree_status_ignore(status);
    }
  }

  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Callback guarantee tests
//===----------------------------------------------------------------------===//

// Verify that cancelled operations always receive exactly one callback.
// This is a critical invariant: after submit(), the callback will fire exactly
// once, whether the operation completes normally, errors, or is cancelled.
TEST_P(CancellationTest, CallbackAlwaysFires) {
  // Test multiple operation types to verify the invariant broadly.
  static constexpr int kNumOperations = 5;

  iree_async_timer_operation_t timers[kNumOperations];
  CompletionTracker trackers[kNumOperations];

  for (int i = 0; i < kNumOperations; ++i) {
    memset(&timers[i], 0, sizeof(timers[i]));
    timers[i].base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
    timers[i].deadline_ns =
        iree_time_now() + iree_make_duration_ms(10000);  // 10s
    timers[i].base.completion_fn = CompletionTracker::Callback;
    timers[i].base.user_data = &trackers[i];
  }

  // Submit all timers.
  for (int i = 0; i < kNumOperations; ++i) {
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timers[i].base));
  }

  // Cancel all of them.
  for (int i = 0; i < kNumOperations; ++i) {
    IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &timers[i].base));
  }

  // Poll until all callbacks fire.
  PollUntil(/*min_completions=*/kNumOperations,
            /*total_budget=*/iree_make_duration_ms(2000));

  // Drain any remaining.
  DrainPending(iree_make_duration_ms(500));

  // Verify each callback fired exactly once.
  for (int i = 0; i < kNumOperations; ++i) {
    EXPECT_EQ(trackers[i].call_count, 1)
        << "Timer " << i << " callback count mismatch";
    IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, trackers[i].ConsumeStatus())
        << "Timer " << i << " expected CANCELLED";
  }
}

// Verify callback fires even when cancel races with completion.
// Submit a timer with short deadline and cancel immediately - either outcome
// is valid, but the callback must fire exactly once.
TEST_P(CancellationTest, CancelRacesWithCompletion) {
  static constexpr int kIterations = 10;

  for (int iter = 0; iter < kIterations; ++iter) {
    iree_async_timer_operation_t timer;
    memset(&timer, 0, sizeof(timer));
    timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;

    // Very short deadline to create a race condition.
    timer.deadline_ns = iree_time_now() + iree_make_duration_ms(1);  // 1ms

    CompletionTracker tracker;
    timer.base.completion_fn = CompletionTracker::Callback;
    timer.base.user_data = &tracker;

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timer.base));

    // Cancel immediately - may or may not win the race.
    IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &timer.base));

    // Poll for the callback.
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(500));

    // Drain any additional CQEs.
    DrainPending(iree_make_duration_ms(100));

    // Must have exactly one callback.
    EXPECT_EQ(tracker.call_count, 1) << "Iteration " << iter;

    // Either CANCELLED or OK is valid.
    {
      iree_status_t status = tracker.ConsumeStatus();
      if (!iree_status_is_ok(status) && !iree_status_is_cancelled(status)) {
        IREE_EXPECT_OK(status) << "Iteration " << iter;
      } else {
        iree_status_ignore(status);
      }
    }
  }
}

CTS_REGISTER_TEST_SUITE(CancellationTest);

}  // namespace iree::async::cts
