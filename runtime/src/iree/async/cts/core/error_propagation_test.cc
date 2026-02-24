// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for error propagation semantics.
//
// Tests verify:
// - Socket sticky failure: after I/O error, socket records failure status
// - Sticky failure persists: subsequent operations on failed socket fail
// - Operation after error: new operations on a failed socket still complete
// - Proactor resilience: one socket failure doesn't affect other sockets
// - Multishot error handling: final callback fires on error (no MORE flag)
// - Callback status codes: errors carry correct status information
// - Connect failure in LINKED chain: continuation is cancelled

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"

namespace iree::async::cts {

class ErrorPropagationTest : public SocketTestBase<> {};

//===----------------------------------------------------------------------===//
// Socket sticky failure tests
//===----------------------------------------------------------------------===//

// When send fails due to peer closing connection, the socket enters failed
// state. Subsequent queries to iree_async_socket_query_failure() return the
// error status.
TEST_P(ErrorPropagationTest, SendOnClosedConnectionSetsStickyFailure) {
  // Establish connected pair.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Close server side - this will cause sends from client to fail.
  iree_async_socket_release(server);

  // Send data from client - should fail because peer is closed.
  char data[] = "hello";
  iree_async_span_t span = iree_async_span_from_ptr(data, sizeof(data));

  // First send might succeed (buffered) or fail - keep sending until failure.
  CompletionTracker tracker;
  bool got_error = false;
  for (int attempt = 0; attempt < 100 && !got_error; ++attempt) {
    iree_async_socket_send_operation_t send_op;
    tracker.Reset();
    InitSendOperation(&send_op, client, &span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &tracker);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));

    iree_status_t status = tracker.ConsumeStatus();
    if (!iree_status_is_ok(status)) {
      got_error = true;
      iree_status_ignore(status);
    }
  }
  ASSERT_TRUE(got_error) << "Expected send to fail after peer close";

  // Socket should now have sticky failure.
  IREE_EXPECT_NOT_OK(iree_async_socket_query_failure(client))
      << "Socket should have sticky failure after I/O error";

  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// After a socket enters failed state, subsequent send operations fail
// immediately with the same (or related) error.
TEST_P(ErrorPropagationTest, StickyFailurePersistsAcrossOperations) {
  // Establish connected pair.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Close server side to trigger client failures.
  iree_async_socket_release(server);

  char data[] = "test data for send";
  iree_async_span_t span = iree_async_span_from_ptr(data, sizeof(data));

  // Send repeatedly until we get an error.
  CompletionTracker tracker;
  bool first_error_seen = false;
  for (int attempt = 0; attempt < 100 && !first_error_seen; ++attempt) {
    iree_async_socket_send_operation_t send_op;
    tracker.Reset();
    InitSendOperation(&send_op, client, &span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &tracker);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));

    iree_status_t status = tracker.ConsumeStatus();
    if (!iree_status_is_ok(status)) {
      first_error_seen = true;
      iree_status_ignore(status);
    }
  }
  ASSERT_TRUE(first_error_seen) << "Expected send to fail after peer close";

  // Now verify that subsequent sends also fail.
  for (int i = 0; i < 3; ++i) {
    iree_async_socket_send_operation_t send_op;
    tracker.Reset();
    InitSendOperation(&send_op, client, &span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &tracker);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));

    IREE_EXPECT_NOT_OK(tracker.ConsumeStatus())
        << "Subsequent send " << i << " should also fail";
  }

  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// One socket failing doesn't affect other sockets on the same proactor.
// The proactor continues to process operations for healthy sockets.
TEST_P(ErrorPropagationTest, ProactorContinuesAfterSocketFailure) {
  // Create two independent connections.
  iree_async_socket_t* client1 = nullptr;
  iree_async_socket_t* server1 = nullptr;
  iree_async_socket_t* listener1 = nullptr;
  EstablishConnection(&client1, &server1, &listener1);

  iree_async_socket_t* client2 = nullptr;
  iree_async_socket_t* server2 = nullptr;
  iree_async_socket_t* listener2 = nullptr;
  EstablishConnection(&client2, &server2, &listener2);

  // Kill connection 1's server to make client1 fail.
  iree_async_socket_release(server1);

  // Send on client1 until it fails.
  char data[] = "breaking data";
  iree_async_span_t span = iree_async_span_from_ptr(data, sizeof(data));
  CompletionTracker tracker;
  bool client1_failed = false;
  for (int attempt = 0; attempt < 100 && !client1_failed; ++attempt) {
    iree_async_socket_send_operation_t send_op;
    tracker.Reset();
    InitSendOperation(&send_op, client1, &span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &tracker);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));

    iree_status_t status = tracker.ConsumeStatus();
    if (!iree_status_is_ok(status)) {
      client1_failed = true;
      iree_status_ignore(status);
    }
  }
  ASSERT_TRUE(client1_failed) << "Expected client1 to fail";

  // Now verify client2 still works fine.
  // Submit a recv on server2 and send from client2.
  char recv_buffer[64] = {0};
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));
  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server2, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  char send_data[] = "client2 still works";
  iree_async_span_t send_span =
      iree_async_span_from_ptr(send_data, sizeof(send_data));
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client2, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Both operations on client2/server2 should succeed.
  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());

  EXPECT_EQ(recv_tracker.call_count, 1);
  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());

  // Clean up.
  iree_async_socket_release(client1);
  iree_async_socket_release(listener1);
  iree_async_socket_release(client2);
  iree_async_socket_release(server2);
  iree_async_socket_release(listener2);
}

//===----------------------------------------------------------------------===//
// Multishot error handling tests
//===----------------------------------------------------------------------===//

// When a multishot recv encounters peer close, it delivers a final callback
// WITHOUT the MORE flag, indicating the operation is done.
// This is similar to MultishotRecv_ConnectionClose in multishot_test.cc but
// focuses on error propagation semantics.
TEST_P(ErrorPropagationTest, MultishotRecvPeerCloseDeliversFinalCallback) {
  if (!iree_any_bit_set(capabilities_,
                        IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
    GTEST_SKIP() << "backend lacks multishot capability";
  }

  // Establish connected pair.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Submit multishot recv on server.
  char buffer[256] = {0};
  iree_async_span_t span = iree_async_span_from_ptr(buffer, sizeof(buffer));
  iree_async_socket_recv_operation_t recv_op;
  CompletionLog log;
  InitMultishotRecvOperation(&recv_op, server, &span, 1,
                             CompletionLog::Callback, &log);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  // Send some data from client.
  char data1[] = "test message";
  iree_async_span_t send_span1 = iree_async_span_from_ptr(data1, sizeof(data1));
  iree_async_socket_send_operation_t send_op1;
  CompletionTracker send_tracker1;
  InitSendOperation(&send_op1, client, &send_span1, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker1);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op1.base));

  // Wait for send completion.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(2000));

  // Close client using close operation (not just release) for proper shutdown.
  iree_async_socket_close_operation_t close_op;
  CompletionTracker close_tracker;
  InitCloseOperation(&close_op, client, CompletionTracker::Callback,
                     &close_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &close_op.base));

  // Wait for close completion.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(2000));

  // Drain pending completions. The peer close should terminate multishot recv.
  DrainPending(iree_make_duration_ms(1000));

  // Verify multishot terminated (final_received should be true).
  EXPECT_TRUE(log.final_received)
      << "Multishot recv should have terminated after peer close";

  // The final entry should NOT have MORE flag.
  if (!log.entries.empty()) {
    size_t final_idx = log.entries.size() - 1;
    EXPECT_FALSE(log.entries[final_idx].flags & IREE_ASYNC_COMPLETION_FLAG_MORE)
        << "Final callback should NOT have MORE flag";
    // Final status could be OK (graceful close with 0 bytes) or error.
    iree_status_ignore(log.ConsumeStatus(final_idx));
  }

  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Callback status code tests
//===----------------------------------------------------------------------===//

// Connect to a non-listening port fails with an appropriate error status.
TEST_P(ErrorPropagationTest, ConnectFailureCarriesCorrectStatus) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));

  iree_async_address_t address = CreateRefusedAddress();

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker tracker;
  InitConnectOperation(&connect_op, socket, address,
                       CompletionTracker::Callback, &tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(tracker.call_count, 1);
  // Should be an error - typically CONNECTION_REFUSED or similar.
  IREE_EXPECT_NOT_OK(tracker.ConsumeStatus())
      << "Connect to non-listening port should fail";

  iree_async_socket_release(socket);
}

// After a send fails on a socket (peer closed), a subsequent recv on the same
// socket completes with an error. The proactor's internal fd tracking must not
// get corrupted by the first failure such that the second operation is silently
// lost.
TEST_P(ErrorPropagationTest, OperationAfterSocketError) {
  // Establish connected pair.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Close server side to trigger client failures.
  iree_async_socket_release(server);

  // Send repeatedly until we get an error.
  char data[] = "test data for send";
  iree_async_span_t span = iree_async_span_from_ptr(data, sizeof(data));
  CompletionTracker send_tracker;
  bool send_error_seen = false;
  for (int attempt = 0; attempt < 100 && !send_error_seen; ++attempt) {
    iree_async_socket_send_operation_t send_op;
    send_tracker.Reset();
    InitSendOperation(&send_op, client, &span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &send_tracker);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));

    iree_status_t status = send_tracker.ConsumeStatus();
    if (!iree_status_is_ok(status)) {
      send_error_seen = true;
      iree_status_ignore(status);
    }
  }
  ASSERT_TRUE(send_error_seen) << "Expected send to fail after peer close";

  // Now submit a recv on the same socket. The peer is gone, so this should
  // complete with an error — not hang indefinitely.
  char recv_buffer[64] = {0};
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));
  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, client, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(recv_tracker.call_count, 1)
      << "Recv on a broken socket should still complete";
  iree_status_t recv_status = recv_tracker.ConsumeStatus();
  // The recv should either return an error or return OK with 0 bytes (EOF).
  // Either is acceptable — what matters is that it completes.
  iree_status_ignore(recv_status);

  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// A LINKED connect to a port with no listener fails, and the continuation
// (send) is cancelled. Every submitted operation must eventually complete —
// the proactor must not silently drop operations when connect fails
// asynchronously.
TEST_P(ErrorPropagationTest, ConnectFailurePropagatesThroughLinkedChain) {
  if (!iree_any_bit_set(capabilities_,
                        IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS)) {
    GTEST_SKIP() << "backend lacks linked operations capability";
  }

  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));

  iree_async_address_t address = CreateRefusedAddress();

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, socket, address,
                       CompletionTracker::Callback, &connect_tracker);
  connect_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;

  // The send operation is the continuation — it should be CANCELLED when
  // connect fails.
  char data[] = "should never be sent";
  iree_async_span_t span = iree_async_span_from_ptr(data, sizeof(data));
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, socket, &span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  iree_async_operation_t* operations[] = {&connect_op.base, &send_op.base};
  iree_async_operation_list_t list = {operations, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Connect should have failed.
  EXPECT_EQ(connect_tracker.call_count, 1);
  IREE_EXPECT_NOT_OK(connect_tracker.ConsumeStatus())
      << "Connect to non-listening port should fail";

  // Send should have been cancelled as a LINKED continuation.
  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, send_tracker.ConsumeStatus());

  iree_async_socket_release(socket);
}

// Timer with deadline in the past completes immediately with OK status.
// (This isn't an error case, but verifies callback fires correctly.)
TEST_P(ErrorPropagationTest, TimerDeadlineInPastCompletesOk) {
  iree_async_timer_operation_t timer;
  memset(&timer, 0, sizeof(timer));
  timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer.deadline_ns = iree_time_now() - 1000000LL;  // 1ms ago

  CompletionTracker tracker;
  timer.base.completion_fn = CompletionTracker::Callback;
  timer.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timer.base));
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(ErrorPropagationTest);

}  // namespace iree::async::cts
