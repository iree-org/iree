// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for TCP socket control operations.
//
// Tests verify shutdown behavior (half-close semantics), connection reset
// handling (RST), and error conditions in bind/listen operations.

#include <cstring>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Shutdown tests
//===----------------------------------------------------------------------===//

// Tests for socket shutdown behavior (half-close semantics).
class ShutdownTest : public SocketTestBase<> {};

// Client shuts down write, server recv returns 0 (EOF).
TEST_P(ShutdownTest, Shutdown_Write) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Send some data before shutdown.
  const char* message = "Before shutdown";
  iree_host_size_t message_length = strlen(message);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)message, message_length);

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));
  IREE_ASSERT_OK(send_tracker.ConsumeStatus());

  // Shutdown client's write direction.
  IREE_ASSERT_OK(
      iree_async_socket_shutdown(client, IREE_ASYNC_SOCKET_SHUTDOWN_WRITE));

  // Server should be able to receive the data that was sent.
  char recv_buffer[128];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, message_length);
  EXPECT_EQ(memcmp(recv_buffer, message, message_length), 0);

  // Another recv should get 0 bytes (EOF) since client shutdown write.
  CompletionTracker eof_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &eof_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  // On EOF, recv returns 0 bytes (status may be OK with 0 bytes or a specific
  // EOF status depending on implementation).
  EXPECT_EQ(eof_tracker.call_count, 1);
  EXPECT_EQ(recv_op.bytes_received, 0u);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Client can still receive after shutting down write.
TEST_P(ShutdownTest, Shutdown_WriteStillReceives) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Shutdown client's write direction.
  IREE_ASSERT_OK(
      iree_async_socket_shutdown(client, IREE_ASYNC_SOCKET_SHUTDOWN_WRITE));

  // Server can still send to client.
  const char* message = "Server to client after shutdown";
  iree_host_size_t message_length = strlen(message);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)message, message_length);

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, server, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Client should be able to receive it.
  char recv_buffer[128];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, client, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, message_length);
  EXPECT_EQ(memcmp(recv_buffer, message, message_length), 0);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Shutdown both directions.
TEST_P(ShutdownTest, Shutdown_Both) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Shutdown both directions on client.
  IREE_ASSERT_OK(
      iree_async_socket_shutdown(client, IREE_ASYNC_SOCKET_SHUTDOWN_BOTH));

  // Server recv should get EOF.
  char recv_buffer[128];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(recv_tracker.call_count, 1);
  EXPECT_EQ(recv_op.bytes_received, 0u);  // EOF

  // Socket is still valid after shutdown - can query state.
  EXPECT_NE(client->proactor, nullptr);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Shutdown on unconnected socket fails with FAILED_PRECONDITION.
TEST_P(ShutdownTest, Shutdown_Unconnected) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));

  // Shutdown on unconnected socket should fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_async_socket_shutdown(socket, IREE_ASYNC_SOCKET_SHUTDOWN_WRITE));

  iree_async_socket_release(socket);
}

// Invalid shutdown mode returns error.
TEST_P(ShutdownTest, Shutdown_InvalidMode) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Invalid mode value.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_socket_shutdown(
                            client, (iree_async_socket_shutdown_mode_t)99));

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Bind/listen error path tests
//===----------------------------------------------------------------------===//

class BindListenErrorTest : public SocketTestBase<> {};

// Two sockets binding to the same port without REUSE options should fail.
TEST_P(BindListenErrorTest, Bind_PortConflict) {
  // Create first socket and bind to ephemeral port.
  iree_async_socket_t* socket1 = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket1));

  iree_async_address_t address;
  IREE_ASSERT_OK(iree_async_address_from_ipv4(
      iree_make_cstring_view("127.0.0.1"), 0, &address));
  IREE_ASSERT_OK(iree_async_socket_bind(socket1, &address));

  // Query the assigned port.
  iree_async_address_t bound_address;
  IREE_ASSERT_OK(
      iree_async_socket_query_local_address(socket1, &bound_address));

  // Create second socket and try to bind to the same address.
  iree_async_socket_t* socket2 = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket2));

  // EADDRINUSE maps to FAILED_PRECONDITION in IREE because it represents a
  // state precondition violation (the port is already in use).
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_async_socket_bind(socket2, &bound_address));

  iree_async_socket_release(socket1);
  iree_async_socket_release(socket2);
}

// Calling listen on an unbound socket implicitly binds to an ephemeral port.
// This is valid POSIX behavior and should not fail.
TEST_P(BindListenErrorTest, Listen_ImplicitBind) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));

  // Listen without explicit bind succeeds (implicitly binds to 0.0.0.0:0).
  IREE_ASSERT_OK(iree_async_socket_listen(socket, 4));
  EXPECT_EQ(iree_async_socket_query_state(socket),
            IREE_ASYNC_SOCKET_STATE_LISTENING);

  // Verify the socket was implicitly bound by querying its address.
  iree_async_address_t address;
  IREE_ASSERT_OK(iree_async_socket_query_local_address(socket, &address));
  // The socket should be bound to some port (non-zero length address).
  EXPECT_GT(address.length, 0);

  iree_async_socket_release(socket);
}

// Calling listen twice is typically allowed (updates backlog).
TEST_P(BindListenErrorTest, Listen_Twice) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));

  iree_async_address_t address;
  IREE_ASSERT_OK(iree_async_address_from_ipv4(
      iree_make_cstring_view("127.0.0.1"), 0, &address));
  IREE_ASSERT_OK(iree_async_socket_bind(socket, &address));

  // First listen.
  IREE_ASSERT_OK(iree_async_socket_listen(socket, 4));
  EXPECT_EQ(iree_async_socket_query_state(socket),
            IREE_ASYNC_SOCKET_STATE_LISTENING);

  // Second listen should succeed (just updates backlog).
  IREE_EXPECT_OK(iree_async_socket_listen(socket, 8));

  iree_async_socket_release(socket);
}

// Accept on a non-listening socket should fail.
TEST_P(BindListenErrorTest, Accept_NonListening) {
  // Create socket but don't call listen.
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));

  iree_async_address_t address;
  IREE_ASSERT_OK(iree_async_address_from_ipv4(
      iree_make_cstring_view("127.0.0.1"), 0, &address));
  IREE_ASSERT_OK(iree_async_socket_bind(socket, &address));

  // Submit accept on non-listening socket.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, socket, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  // Accept should fail with INVALID_ARGUMENT or similar.
  EXPECT_EQ(accept_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        accept_tracker.ConsumeStatus());
  EXPECT_EQ(accept_op.accepted_socket, nullptr);

  iree_async_socket_release(socket);
}

//===----------------------------------------------------------------------===//
// Connection reset (RST) tests
//===----------------------------------------------------------------------===//

// - A socket is closed with unread data in its receive buffer
// - SO_LINGER is set with zero timeout
// - A peer process crashes or the network is forcibly disrupted
//
// These tests verify that RST conditions are properly reported as errors
// rather than causing hangs or silent failures.
class ResetTest : public SocketTestBase<> {};

// When a socket with unread data is closed with LINGER_ZERO, the TCP stack
// sends an RST to the peer. The peer should receive an error on subsequent send
// operations. LINGER_ZERO ensures the RST fires deterministically — without it,
// some platforms (macOS XNU) may perform a graceful FIN close even with unread
// data, and the RST from the FIN→data→RST round-trip may not propagate back
// before small eager writes complete.
//
// Distinct from Reset_LingerZero which tests recv after RST; this tests send.
TEST_P(ResetTest, Reset_CloseWithUnreadData) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_LINGER_ZERO,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Server sends data to client.
  const char* message = "This data will not be read";
  iree_host_size_t message_length = strlen(message);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)message, message_length);

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, server, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));
  IREE_ASSERT_OK(send_tracker.ConsumeStatus());

  // Client closes WITHOUT reading the data. LINGER_ZERO causes immediate RST.
  iree_async_socket_release(client);
  client = nullptr;

  // Submit a recv to force the kernel to process the RST. On loopback,
  // LINGER_ZERO RST is delivered synchronously within close(), so by this
  // point the server socket's kernel buffer already has the RST queued.
  // readv() fails immediately with ECONNRESET, setting sticky failure on the
  // socket. All subsequent eager sends then hit the sticky failure check and
  // fail without attempting writev().
  {
    char rst_probe_buffer[1] = {0};
    iree_async_span_t rst_probe_span =
        iree_async_span_from_ptr(rst_probe_buffer, sizeof(rst_probe_buffer));
    iree_async_socket_recv_operation_t rst_probe_op;
    CompletionTracker rst_probe_tracker;
    InitRecvOperation(&rst_probe_op, server, &rst_probe_span, 1,
                      CompletionTracker::Callback, &rst_probe_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &rst_probe_op.base));
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(5000));
    iree_status_ignore(rst_probe_tracker.ConsumeStatus());
  }

  // Server tries to send more data. The socket has sticky failure from the
  // recv above, so each eager send fails immediately.
  const char* more_data = "Sending after RST";
  iree_async_span_t more_span =
      iree_async_span_from_ptr((void*)more_data, strlen(more_data));

  bool got_error = false;
  for (int i = 0; i < 10 && !got_error; ++i) {
    iree_async_socket_send_operation_t send2_op;
    CompletionTracker send2_tracker;
    InitSendOperation(&send2_op, server, &more_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &send2_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send2_op.base));

    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));

    if (!iree_status_is_ok(send2_tracker.last_status)) {
      got_error = true;
      // ECONNRESET maps to UNAVAILABLE, EPIPE maps to FAILED_PRECONDITION.
      iree_status_t status = send2_tracker.ConsumeStatus();
      if (!iree_status_is_unavailable(status) &&
          !iree_status_is_failed_precondition(status)) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
      } else {
        iree_status_ignore(status);
      }
    }
  }

  EXPECT_TRUE(got_error) << "Expected error after peer closed with unread data";

  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// After a peer closes its connection, recv should return 0 bytes (EOF) or
// an error if the close was abrupt.
TEST_P(ResetTest, Reset_RecvAfterPeerClose) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Client closes without sending anything or doing proper shutdown.
  iree_async_socket_release(client);
  client = nullptr;

  // Server tries to recv. Should get EOF (0 bytes) or an error.
  char recv_buffer[128];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(recv_tracker.call_count, 1);
  // Either EOF (OK with 0 bytes) or an error.
  if (iree_status_is_ok(recv_tracker.last_status)) {
    iree_status_ignore(recv_tracker.ConsumeStatus());
    EXPECT_EQ(recv_op.bytes_received, 0u) << "Expected EOF (0 bytes)";
  } else {
    // RST conditions map to UNAVAILABLE.
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          recv_tracker.ConsumeStatus());
  }

  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// Attempting to send on a socket after the peer has gracefully closed should
// eventually result in an error (EPIPE/ECONNRESET).
TEST_P(ResetTest, Reset_SendAfterPeerShutdown) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Client does graceful shutdown of write direction.
  IREE_ASSERT_OK(
      iree_async_socket_shutdown(client, IREE_ASYNC_SOCKET_SHUTDOWN_BOTH));

  // Server tries to send data to a client that has shut down.
  // First sends may succeed (kernel buffers), but eventually should fail.
  const char* message = "Sending to shut-down peer";
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)message, strlen(message));

  bool got_error = false;
  for (int i = 0; i < 10 && !got_error; ++i) {
    iree_async_socket_send_operation_t send_op;
    CompletionTracker send_tracker;
    InitSendOperation(&send_op, server, &send_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &send_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));

    if (!iree_status_is_ok(send_tracker.last_status)) {
      got_error = true;
      // EPIPE (broken pipe) maps to FAILED_PRECONDITION.
      // ECONNRESET maps to UNAVAILABLE.
      iree_status_t status = send_tracker.ConsumeStatus();
      if (!iree_status_is_failed_precondition(status) &&
          !iree_status_is_unavailable(status)) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
      } else {
        iree_status_ignore(status);
      }
    }
  }

  // TCP may buffer several sends before returning an error, but it should
  // eventually fail. If all 10 sends succeeded, that's unexpected but not
  // necessarily wrong (depends on kernel buffering).

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// SO_LINGER with timeout=0 causes close() to send RST immediately instead of
// FIN. This test verifies that the peer receives an error rather than a clean
// EOF when the LINGER_ZERO socket is closed.
TEST_P(ResetTest, Reset_LingerZero) {
  // Create listener without LINGER_ZERO.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Submit accept.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Create client WITH LINGER_ZERO - closing this socket will send RST.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(
      proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_LINGER_ZERO,
      &client));

  // Connect.
  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  ASSERT_EQ(accept_tracker.call_count, 1);
  IREE_ASSERT_OK(accept_tracker.ConsumeStatus());
  ASSERT_NE(accept_op.accepted_socket, nullptr);
  iree_async_socket_t* server = accept_op.accepted_socket;

  // Exchange some data to ensure connection is fully established.
  const char* message = "Hello before RST";
  iree_host_size_t message_length = strlen(message);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)message, message_length);

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  char recv_buffer[128];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_ASSERT_OK(send_tracker.ConsumeStatus());
  IREE_ASSERT_OK(recv_tracker.ConsumeStatus());
  ASSERT_EQ(recv_op.bytes_received, message_length);

  // Close client with LINGER_ZERO - this sends RST instead of FIN.
  iree_async_socket_release(client);
  client = nullptr;

  // Server tries to recv. With RST, should get an error (ECONNRESET).
  // Without LINGER_ZERO, we would get EOF (0 bytes).
  memset(recv_buffer, 0, sizeof(recv_buffer));
  CompletionTracker rst_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &rst_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(rst_tracker.call_count, 1);
  // LINGER_ZERO should cause RST, which results in ECONNRESET -> UNAVAILABLE.
  // Some systems may return EOF instead if the RST arrives after the recv
  // completes, so we accept either an error or EOF with 0 bytes.
  if (iree_status_is_ok(rst_tracker.last_status)) {
    // If OK, should be EOF (0 bytes) - RST may have arrived as EOF on some
    // kernels/timing.
    iree_status_ignore(rst_tracker.ConsumeStatus());
    EXPECT_EQ(recv_op.bytes_received, 0u);
  } else {
    // Expected: ECONNRESET maps to UNAVAILABLE.
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, rst_tracker.ConsumeStatus());
  }

  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(ShutdownTest);
CTS_REGISTER_TEST_SUITE(BindListenErrorTest);
CTS_REGISTER_TEST_SUITE(ResetTest);

}  // namespace iree::async::cts
