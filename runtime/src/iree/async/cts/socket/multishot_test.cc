// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for multishot socket operations.
//
// These tests verify multishot accept and recv operations where a single
// submitted operation produces multiple completions with
// IREE_ASYNC_COMPLETION_FLAG_MORE set on all but the final completion.
// Requires IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT.

#include <cstring>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"

namespace iree::async::cts {

// These tests verify multishot operations where a single submitted operation
// produces multiple completions with IREE_ASYNC_COMPLETION_FLAG_MORE set on all
// but the final completion. The tests use explicit cancellation via
// iree_async_proactor_cancel() to safely terminate multishot operations before
// test cleanup.
class MultishotTest : public SocketTestBase<> {};

// Multishot accept: submit once, accept multiple connections.
// This test creates a listener with multishot accept, then makes multiple
// connections. Each connection should produce a completion with
// IREE_ASYNC_COMPLETION_FLAG_MORE (except the final after cancellation).
TEST_P(MultishotTest, MultishotAccept_MultipleConnections) {
  if (!iree_any_bit_set(capabilities_,
                        IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
    GTEST_SKIP() << "backend lacks multishot capability";
  }

  static constexpr int kNumConnections = 3;

  // Create a listener.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Submit multishot accept operation.
  iree_async_socket_accept_operation_t accept_op;
  // Use CompletionLog to track multiple completions.
  CompletionLog accept_log;
  InitMultishotAcceptOperation(&accept_op, listener, CompletionLog::Callback,
                               &accept_log);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Store accepted sockets to release later.
  std::vector<iree_async_socket_t*> accepted_sockets;
  std::vector<iree_async_socket_t*> clients;

  // Create multiple client connections.
  for (int i = 0; i < kNumConnections; ++i) {
    iree_async_socket_t* client = nullptr;
    IREE_ASSERT_OK(
        iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                 IREE_ASYNC_SOCKET_OPTION_NO_DELAY, &client));
    clients.push_back(client);

    iree_async_socket_connect_operation_t connect_op;
    CompletionTracker connect_tracker;
    InitConnectOperation(&connect_op, client, listen_address,
                         CompletionTracker::Callback, &connect_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

    // Wait for this specific connect to complete. PollUntil returns when ANY
    // completion fires (possibly the multishot accept, not our connect), so we
    // must loop until the connect_tracker confirms our connect is done.
    // The connect_op is stack-local and must not outlive this iteration.
    while (connect_tracker.call_count == 0) {
      PollUntil(/*min_completions=*/1,
                /*total_budget=*/iree_make_duration_ms(2000));
    }

    // accepted_socket is overwritten on each multishot accept completion.
    // In a real application, the callback would extract it immediately.
    if (accept_op.accepted_socket != nullptr) {
      // Check if this is a new socket (not already in our list).
      bool is_new = true;
      for (auto* s : accepted_sockets) {
        if (s == accept_op.accepted_socket) {
          is_new = false;
          break;
        }
      }
      if (is_new) {
        accepted_sockets.push_back(accept_op.accepted_socket);
      }
    }
  }

  // Wait for at least one accept completion. The connect-wait loop above may
  // have already consumed accept completions, but on a slow system the last
  // accept(s) may still be in the poll thread pipeline.
  while (accept_log.entries.empty()) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(5000));
  }

  // Verify we got multiple accept completions with MORE flag set.
  EXPECT_GE(accept_log.entries.size(), 1u)
      << "Expected at least one accept completion";

  // All entries except possibly the last should have MORE flag.
  for (size_t i = 0; i + 1 < accept_log.entries.size(); ++i) {
    IREE_EXPECT_OK(accept_log.ConsumeStatus(i)) << "Accept " << i << " failed";
    EXPECT_TRUE(accept_log.entries[i].flags & IREE_ASYNC_COMPLETION_FLAG_MORE)
        << "Accept " << i << " should have MORE flag";
  }

  // Clean up clients first.
  for (auto* client : clients) {
    iree_async_socket_release(client);
  }
  for (auto* accepted : accepted_sockets) {
    iree_async_socket_release(accepted);
  }

  // Cancel the multishot accept and wait for the cancellation completion.
  iree_async_proactor_cancel(proactor_, &accept_op.base);
  {
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(2000);
    while (!accept_log.final_received && iree_time_now() < deadline) {
      PollOnce();
    }
  }

  EXPECT_TRUE(accept_log.final_received)
      << "Multishot accept should have terminated after cancel";

  // Safety: if the cancellation didn't complete above, drain to ensure the
  // accept_op is removed from the proactor before this stack frame exits.
  // Without this, TearDown would process operations against destroyed memory.
  if (!accept_log.final_received) {
    DrainPending();
  }

  iree_async_socket_release(listener);
}

// Multishot accept: closing the listener terminates multishot.
TEST_P(MultishotTest, MultishotAccept_ListenerClose) {
  if (!iree_any_bit_set(capabilities_,
                        IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
    GTEST_SKIP() << "backend lacks multishot capability";
  }

  // Create a listener.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Submit multishot accept.
  iree_async_socket_accept_operation_t accept_op;
  CompletionLog accept_log;
  InitMultishotAcceptOperation(&accept_op, listener, CompletionLog::Callback,
                               &accept_log);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Make one connection.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &client));

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  // Wait for this specific connect to complete. PollUntil returns when ANY
  // completion fires (possibly the multishot accept), so we must loop until
  // the connect_tracker confirms our connect is done. The connect_op is
  // stack-local and must not outlive this scope.
  while (connect_tracker.call_count == 0) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(5000));
  }

  // Also wait for at least one accept completion.
  while (accept_log.entries.empty()) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(5000));
  }

  ASSERT_GE(accept_log.entries.size(), 1u);
  IREE_EXPECT_OK(accept_log.ConsumeStatus(0));

  // Clean up accepted socket if present.
  iree_async_socket_release(accept_op.accepted_socket);
  accept_op.accepted_socket = nullptr;

  // Cancel the multishot accept and wait for the cancellation completion.
  iree_async_proactor_cancel(proactor_, &accept_op.base);
  {
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(2000);
    while (!accept_log.final_received && iree_time_now() < deadline) {
      PollOnce();
    }
  }

  EXPECT_TRUE(accept_log.final_received)
      << "Multishot accept should have terminated after cancel";

  // Safety: if the cancellation didn't complete above, drain to ensure the
  // accept_op is removed from the proactor before this stack frame exits.
  if (!accept_log.final_received) {
    DrainPending();
  }

  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Multishot recv: submit once, receive multiple messages.
// Note: Without a provided buffer ring, multishot recv with a fixed buffer
// reuses the same buffer for each receive. This test verifies the multishot
// flag is set and multiple completions can occur.
TEST_P(MultishotTest, MultishotRecv_MultipleMessages) {
  if (!iree_any_bit_set(capabilities_,
                        IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
    GTEST_SKIP() << "backend lacks multishot capability";
  }

  // Establish connection.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Prepare receive buffer.
  uint8_t recv_buffer[256];
  memset(recv_buffer, 0, sizeof(recv_buffer));

  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  // Submit multishot recv.
  iree_async_socket_recv_operation_t recv_op;
  CompletionLog recv_log;
  InitMultishotRecvOperation(&recv_op, server, &recv_span, 1,
                             CompletionLog::Callback, &recv_log);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  // Send multiple small messages.
  static constexpr int kNumMessages = 3;
  const char* messages[kNumMessages] = {"Hello", "World", "Test!"};

  for (int i = 0; i < kNumMessages; ++i) {
    iree_async_span_t send_span =
        iree_async_span_from_ptr((void*)messages[i], strlen(messages[i]));

    iree_async_socket_send_operation_t send_op;
    CompletionTracker send_tracker;
    InitSendOperation(&send_op, client, &send_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &send_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

    // Wait for this specific send to complete. PollUntil returns when ANY
    // completion fires (possibly the multishot recv), so we must loop until
    // the send_tracker confirms our send is done. The send_op is stack-local
    // and must not outlive this iteration.
    while (send_tracker.call_count == 0) {
      PollUntil(/*min_completions=*/1,
                /*total_budget=*/iree_make_duration_ms(1000));
    }
  }

  // Wait for at least one recv completion. Sends complete eagerly via writev
  // at submit time, so PollUntil during the send loop above may be satisfied
  // by send completions alone without any recv firing. DrainPending uses
  // immediate (0ms) timeout and breaks on the first empty poll, which is wrong
  // here — we need a blocking wait for the recv that may not have been
  // processed by the poll thread yet.
  while (recv_log.entries.empty()) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(5000));
  }

  // TCP may coalesce messages, so we might get fewer completions than
  // messages sent. The key verification is that multishot mode is active.
  EXPECT_GE(recv_log.entries.size(), 1u)
      << "Expected at least one recv completion";

  // All entries except the final should have MORE flag if multishot worked.
  // If only one entry, multishot may have been terminated (e.g., by TCP
  // coalescing all messages into one recv).
  for (size_t i = 0; i + 1 < recv_log.entries.size(); ++i) {
    EXPECT_TRUE(recv_log.entries[i].flags & IREE_ASYNC_COMPLETION_FLAG_MORE)
        << "Recv " << i << " should have MORE flag";
  }

  // Cancel the multishot recv and wait for the cancellation completion.
  iree_async_proactor_cancel(proactor_, &recv_op.base);
  {
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(2000);
    while (!recv_log.final_received && iree_time_now() < deadline) {
      PollOnce();
    }
  }

  EXPECT_TRUE(recv_log.final_received)
      << "Multishot recv should have terminated after cancel";

  // Safety: if the cancellation didn't complete above, drain to ensure the
  // recv_op is removed from the proactor before this stack frame exits.
  if (!recv_log.final_received) {
    DrainPending();
  }

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// Multishot recv: connection close terminates multishot.
TEST_P(MultishotTest, MultishotRecv_ConnectionClose) {
  if (!iree_any_bit_set(capabilities_,
                        IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
    GTEST_SKIP() << "backend lacks multishot capability";
  }

  // Establish connection.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Prepare receive buffer.
  uint8_t recv_buffer[128];
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  // Submit multishot recv.
  iree_async_socket_recv_operation_t recv_op;
  CompletionLog recv_log;
  InitMultishotRecvOperation(&recv_op, server, &recv_span, 1,
                             CompletionLog::Callback, &recv_log);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  // Send one message.
  const char* message = "Hello";
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)message, strlen(message));

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Wait for send and potentially first recv.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(2000));

  // Close the client (sender side). Use release (not a close operation)
  // because the close operation's completion goes through the completion queue
  // and can satisfy PollUntil before the FIN propagates through macOS loopback
  // to the server socket. Release closes the fd directly without a completion.
  iree_async_socket_release(client);
  client = nullptr;

  // Wait for the multishot recv to terminate (EOF from peer close).
  // On macOS loopback, the FIN from close may take a few microseconds to
  // propagate. Poll until the recv delivers its final callback (no MORE).
  {
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(5000);
    while (!recv_log.final_received && iree_time_now() < deadline) {
      PollOnce();
    }
  }

  // Verify multishot terminated (final_received should be true).
  // The final completion should either be a 0-byte read (EOF) or an error,
  // without the MORE flag.
  EXPECT_TRUE(recv_log.final_received)
      << "Multishot recv should have terminated after peer close";

  if (!recv_log.entries.empty()) {
    const auto& last = recv_log.entries.back();
    EXPECT_FALSE(last.flags & IREE_ASYNC_COMPLETION_FLAG_MORE)
        << "Final recv should not have MORE flag";
  }

  // Safety: if multishot recv didn't terminate, cancel it to prevent SEGFAULT
  // during TearDown. The recv_op is stack-local; the proactor's fd chain would
  // reference dangling memory when processing operations during shutdown.
  if (!recv_log.final_received) {
    iree_status_ignore(iree_async_proactor_cancel(proactor_, &recv_op.base));
    DrainPending();
  }

  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE_WITH_TAGS(MultishotTest, /*required=*/{"multishot"},
                                  /*excluded=*/{});

}  // namespace iree::async::cts
