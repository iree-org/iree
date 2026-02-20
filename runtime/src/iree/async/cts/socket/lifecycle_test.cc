// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for socket lifecycle: create, bind, listen, IPv6.

#include <cstring>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"

namespace iree::async::cts {

class SocketTest : public SocketTestBase<> {};

//===----------------------------------------------------------------------===//
// Lifecycle tests: create, bind, listen
//===----------------------------------------------------------------------===//

// Create a TCP socket and verify initial state.
TEST_P(SocketTest, CreateSocket_TCP) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));

  ASSERT_NE(socket, nullptr);
  EXPECT_EQ(iree_async_socket_query_state(socket),
            IREE_ASYNC_SOCKET_STATE_CREATED);
  IREE_EXPECT_OK(iree_async_socket_query_failure(socket));

  iree_async_socket_release(socket);
}

// Create a UDP socket and verify initial state.
TEST_P(SocketTest, CreateSocket_UDP) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UDP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));

  ASSERT_NE(socket, nullptr);
  EXPECT_EQ(iree_async_socket_query_state(socket),
            IREE_ASYNC_SOCKET_STATE_CREATED);

  iree_async_socket_release(socket);
}

// Create a Unix stream socket and verify initial state.
TEST_P(SocketTest, CreateSocket_Unix) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &socket));

  ASSERT_NE(socket, nullptr);
  EXPECT_EQ(iree_async_socket_query_state(socket),
            IREE_ASYNC_SOCKET_STATE_CREATED);

  iree_async_socket_release(socket);
}

//===----------------------------------------------------------------------===//
// IPv6 tests
//===----------------------------------------------------------------------===//

// IPv6 socket tests use IREE_ASYNC_SOCKET_TYPE_TCP6/UDP6 with AF_INET6.
// These are separate types from IPv4 to avoid platform-specific dual-stack
// (IPV6_V6ONLY) behavior differences.

// Verify IPv6 address parsing and formatting works.
TEST_P(SocketTest, IPv6_AddressParsing) {
  // Parse loopback address.
  iree_async_address_t loopback;
  IREE_ASSERT_OK(iree_async_address_from_ipv6(iree_make_cstring_view("::1"),
                                              8080, &loopback));

  // Format and verify.
  char buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t formatted;
  IREE_ASSERT_OK(
      iree_async_address_format(&loopback, sizeof(buffer), buffer, &formatted));

  // IPv6 addresses are formatted with brackets: [::1]:port
  std::string addr_str(formatted.data, formatted.size);
  EXPECT_NE(addr_str.find("[::1]:8080"), std::string::npos)
      << "Expected [::1]:8080 but got: " << addr_str;
}

// Verify IPv6 any address (::) parsing.
TEST_P(SocketTest, IPv6_AnyAddressParsing) {
  // Parse any address (empty string = in6addr_any).
  iree_async_address_t any_addr;
  IREE_ASSERT_OK(
      iree_async_address_from_ipv6(iree_string_view_empty(), 9090, &any_addr));

  // Format and verify.
  char buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t formatted;
  IREE_ASSERT_OK(
      iree_async_address_format(&any_addr, sizeof(buffer), buffer, &formatted));

  // Should contain port and brackets.
  std::string addr_str(formatted.data, formatted.size);
  EXPECT_NE(addr_str.find("]:9090"), std::string::npos)
      << "Expected ]:9090 but got: " << addr_str;
}

// Create a TCP6 socket and verify it can bind to an IPv6 address.
TEST_P(SocketTest, IPv6_CreateAndBind) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP6,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &socket));
  ASSERT_NE(socket, nullptr);
  EXPECT_EQ(socket->type, IREE_ASYNC_SOCKET_TYPE_TCP6);

  // Bind to IPv6 loopback on ephemeral port.
  iree_async_address_t address;
  IREE_ASSERT_OK(
      iree_async_address_from_ipv6(iree_make_cstring_view("::1"), 0, &address));
  IREE_EXPECT_OK(iree_async_socket_bind(socket, &address));

  // Query bound address.
  iree_async_address_t bound_addr;
  IREE_ASSERT_OK(iree_async_socket_query_local_address(socket, &bound_addr));

  // Format and verify it's an IPv6 address.
  char buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t formatted;
  IREE_ASSERT_OK(iree_async_address_format(&bound_addr, sizeof(buffer), buffer,
                                           &formatted));
  std::string addr_str(formatted.data, formatted.size);
  EXPECT_NE(addr_str.find("[::1]:"), std::string::npos)
      << "Expected IPv6 address but got: " << addr_str;

  iree_async_socket_release(socket);
}

// Create a UDP6 socket and verify it can bind.
TEST_P(SocketTest, IPv6_CreateUDP6) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UDP6,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &socket));
  ASSERT_NE(socket, nullptr);
  EXPECT_EQ(socket->type, IREE_ASYNC_SOCKET_TYPE_UDP6);

  // Bind to IPv6 any address.
  iree_async_address_t address;
  IREE_ASSERT_OK(
      iree_async_address_from_ipv6(iree_string_view_empty(), 0, &address));
  IREE_EXPECT_OK(iree_async_socket_bind(socket, &address));

  iree_async_socket_release(socket);
}

// Full IPv6 TCP loopback: connect, accept, send, recv over [::1].
TEST_P(SocketTest, IPv6_LoopbackSendRecv) {
  // Create an IPv6 listener socket.
  iree_async_socket_t* listener = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP6,
                               IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR, &listener));

  // Bind to IPv6 loopback on ephemeral port.
  iree_async_address_t bind_addr;
  IREE_ASSERT_OK(iree_async_address_from_ipv6(iree_make_cstring_view("::1"), 0,
                                              &bind_addr));
  IREE_ASSERT_OK(iree_async_socket_bind(listener, &bind_addr));
  IREE_ASSERT_OK(iree_async_socket_listen(listener, 4));

  // Query bound address.
  iree_async_address_t listen_address;
  IREE_ASSERT_OK(
      iree_async_socket_query_local_address(listener, &listen_address));

  // Submit accept operation.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Create a client socket and connect over IPv6.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP6,
                               IREE_ASYNC_SOCKET_OPTION_NO_DELAY, &client));

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  // Poll until both connect and accept complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  ASSERT_EQ(accept_tracker.call_count, 1);
  IREE_ASSERT_OK(accept_tracker.ConsumeStatus());
  ASSERT_NE(accept_op.accepted_socket, nullptr);

  ASSERT_EQ(connect_tracker.call_count, 1);
  IREE_ASSERT_OK(connect_tracker.ConsumeStatus());

  iree_async_socket_t* server = accept_op.accepted_socket;

  // Prepare send data.
  const char* send_data = "Hello over IPv6!";
  iree_host_size_t send_length = strlen(send_data);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)send_data, send_length);

  // Submit send operation from client.
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Prepare recv buffer on server.
  char recv_buffer[256];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  // Submit recv operation on server.
  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  // Poll until send and recv complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, send_length);

  EXPECT_EQ(recv_tracker.call_count, 1);
  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, send_length);

  // Verify data matches.
  EXPECT_EQ(memcmp(recv_buffer, send_data, send_length), 0);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Verify retain/release reference counting works correctly.
TEST_P(SocketTest, SocketRetainRelease) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));

  // Retain twice.
  iree_async_socket_retain(socket);
  iree_async_socket_retain(socket);

  // Release three times (original + two retains).
  iree_async_socket_release(socket);
  iree_async_socket_release(socket);
  iree_async_socket_release(socket);  // Final release destroys socket.
}

// Bind a socket to the loopback address with ephemeral port.
TEST_P(SocketTest, BindLoopback) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR,
                                          &socket));

  iree_async_address_t address;
  IREE_ASSERT_OK(iree_async_address_from_ipv4(
      iree_make_cstring_view("127.0.0.1"), 0, &address));

  IREE_EXPECT_OK(iree_async_socket_bind(socket, &address));

  iree_async_socket_release(socket);
}

// Bind to ephemeral port and verify query_local_address returns the assigned
// port.
TEST_P(SocketTest, BindEphemeralPort) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR,
                                          &socket));

  iree_async_address_t bind_address;
  IREE_ASSERT_OK(
      iree_async_address_from_ipv4(iree_string_view_empty(), 0, &bind_address));
  IREE_ASSERT_OK(iree_async_socket_bind(socket, &bind_address));

  // Query the assigned address.
  iree_async_address_t local_address;
  IREE_ASSERT_OK(iree_async_socket_query_local_address(socket, &local_address));

  // Format the address and verify it contains a port separator.
  char buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t formatted;
  IREE_ASSERT_OK(iree_async_address_format(&local_address, sizeof(buffer),
                                           buffer, &formatted));
  EXPECT_GT(formatted.size, 0u);

  // The formatted address should contain a colon (separating IP and port).
  bool has_port_separator = false;
  for (iree_host_size_t i = 0; i < formatted.size; ++i) {
    if (formatted.data[i] == ':') {
      has_port_separator = true;
      break;
    }
  }
  EXPECT_TRUE(has_port_separator)
      << "Address should contain port: "
      << std::string(formatted.data, formatted.size);

  iree_async_socket_release(socket);
}

// Listen with a backlog and verify state transitions to LISTENING.
TEST_P(SocketTest, ListenWithBacklog) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR,
                                          &socket));

  iree_async_address_t address;
  IREE_ASSERT_OK(
      iree_async_address_from_ipv4(iree_string_view_empty(), 0, &address));
  IREE_ASSERT_OK(iree_async_socket_bind(socket, &address));

  EXPECT_EQ(iree_async_socket_query_state(socket),
            IREE_ASYNC_SOCKET_STATE_CREATED);

  IREE_EXPECT_OK(iree_async_socket_listen(socket, /*backlog=*/32));

  EXPECT_EQ(iree_async_socket_query_state(socket),
            IREE_ASYNC_SOCKET_STATE_LISTENING);

  iree_async_socket_release(socket);
}

// Binding the same socket twice should fail.
TEST_P(SocketTest, BindTwice_Fails) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR,
                                          &socket));

  iree_async_address_t address;
  IREE_ASSERT_OK(
      iree_async_address_from_ipv4(iree_string_view_empty(), 0, &address));

  // First bind should succeed.
  IREE_ASSERT_OK(iree_async_socket_bind(socket, &address));

  // Second bind on the same socket should fail (EINVAL on Linux).
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_socket_bind(socket, &address));

  iree_async_socket_release(socket);
}

//===----------------------------------------------------------------------===//
// Sticky failure tests
//===----------------------------------------------------------------------===//

// Verify that iree_async_socket_query_failure() returns OK for a fresh socket.
TEST_P(SocketTest, StickyFailure_InitiallyOk) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));

  // Fresh socket should have no failure status.
  IREE_EXPECT_OK(iree_async_socket_query_failure(socket));

  iree_async_socket_release(socket);
}

// Socket can still be released cleanly after entering failed state.
TEST_P(SocketTest, StickyFailure_ReleaseAfterError) {
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &client));

  // Trigger failure via connect to a port with no listener.
  iree_async_address_t address = CreateRefusedAddress();

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker tracker;
  InitConnectOperation(&connect_op, client, address,
                       CompletionTracker::Callback, &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Retain and release multiple times - should not crash even in failed state.
  iree_async_socket_retain(client);
  iree_async_socket_release(client);

  // Final release - should cleanup properly.
  iree_async_socket_release(client);
}

//===----------------------------------------------------------------------===//
// Async operation tests: connect, accept, send, recv
//===----------------------------------------------------------------------===//

// Connect to a listening socket - callback fires with success.
TEST_P(SocketTest, ConnectSuccess) {
  // Create a listener.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Create a client socket.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &client));

  // Submit connect operation.
  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  // Poll until connect completes.
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(connect_tracker.call_count, 1);
  IREE_EXPECT_OK(connect_tracker.ConsumeStatus());
  EXPECT_EQ(iree_async_socket_query_state(client),
            IREE_ASYNC_SOCKET_STATE_CONNECTED);

  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Accept an incoming connection - callback fires with accepted socket.
TEST_P(SocketTest, AcceptSuccess) {
  // Create a listener.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Submit accept operation.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Create a client socket and connect.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &client));

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  // Poll until both connect and accept complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(accept_tracker.call_count, 1);
  IREE_EXPECT_OK(accept_tracker.ConsumeStatus());
  ASSERT_NE(accept_op.accepted_socket, nullptr);
  EXPECT_EQ(iree_async_socket_query_state(accept_op.accepted_socket),
            IREE_ASYNC_SOCKET_STATE_CONNECTED);

  EXPECT_EQ(connect_tracker.call_count, 1);
  IREE_EXPECT_OK(connect_tracker.ConsumeStatus());

  iree_async_socket_release(accept_op.accepted_socket);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Full loopback: connect, accept, send, recv with data verification.
TEST_P(SocketTest, LoopbackSendRecv) {
  // Create a listener.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Submit accept operation.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Create a client socket and connect.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &client));

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  // Poll until both connect and accept complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  ASSERT_EQ(accept_tracker.call_count, 1);
  IREE_ASSERT_OK(accept_tracker.ConsumeStatus());
  ASSERT_NE(accept_op.accepted_socket, nullptr);

  ASSERT_EQ(connect_tracker.call_count, 1);
  IREE_ASSERT_OK(connect_tracker.ConsumeStatus());

  iree_async_socket_t* server = accept_op.accepted_socket;

  // Prepare send data.
  const char* send_data = "Hello from client!";
  iree_host_size_t send_length = strlen(send_data);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)send_data, send_length);

  // Submit send operation from client.
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Prepare recv buffer on server.
  char recv_buffer[256];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  // Submit recv operation on server.
  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  // Poll until send and recv complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, send_length);

  EXPECT_EQ(recv_tracker.call_count, 1);
  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, send_length);

  // Verify data matches.
  EXPECT_EQ(memcmp(recv_buffer, send_data, send_length), 0);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Connect to a non-listening address - callback fires with error.
TEST_P(SocketTest, ConnectRefused) {
  // Create a client socket.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &client));

  // Connect to a port with no listener — should fail with ECONNREFUSED.
  iree_async_address_t address = CreateRefusedAddress();

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  // Poll until connect completes (should fail).
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(connect_tracker.call_count, 1);
  // Should be UNAVAILABLE (ECONNREFUSED).
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        connect_tracker.ConsumeStatus());

  iree_async_socket_release(client);
}

// Multiple sequential send/recv exchanges on the same connection.
TEST_P(SocketTest, MultipleExchanges) {
  // Create a listener.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Submit accept operation.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Create a client socket and connect.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &client));

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  // Poll until both connect and accept complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  ASSERT_NE(accept_op.accepted_socket, nullptr);
  iree_async_socket_t* server = accept_op.accepted_socket;

  // Do 3 exchanges.
  for (int round = 0; round < 3; ++round) {
    // Prepare send data with round number embedded.
    char send_data[64];
    snprintf(send_data, sizeof(send_data), "Message round %d", round);
    iree_host_size_t send_length = strlen(send_data);
    iree_async_span_t send_span =
        iree_async_span_from_ptr((void*)send_data, send_length);

    // Submit send operation from client.
    iree_async_socket_send_operation_t send_op;
    CompletionTracker send_tracker;
    InitSendOperation(&send_op, client, &send_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &send_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

    // Prepare recv buffer on server.
    char recv_buffer[256];
    memset(recv_buffer, 0, sizeof(recv_buffer));
    iree_async_span_t recv_span =
        iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

    // Submit recv operation on server.
    iree_async_socket_recv_operation_t recv_op;
    CompletionTracker recv_tracker;
    InitRecvOperation(&recv_op, server, &recv_span, 1,
                      CompletionTracker::Callback, &recv_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

    // Poll until send and recv complete.
    PollUntil(/*min_completions=*/2,
              /*total_budget=*/iree_make_duration_ms(5000));

    EXPECT_EQ(send_tracker.call_count, 1);
    IREE_EXPECT_OK(send_tracker.ConsumeStatus());
    EXPECT_EQ(send_op.bytes_sent, send_length);

    EXPECT_EQ(recv_tracker.call_count, 1);
    IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
    EXPECT_EQ(recv_op.bytes_received, send_length);

    // Verify data matches.
    EXPECT_EQ(memcmp(recv_buffer, send_data, send_length), 0)
        << "Data mismatch in round " << round;
  }

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(SocketTest);

}  // namespace iree::async::cts
