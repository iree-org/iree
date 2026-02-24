// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Extended test base for socket CTS tests.
//
// Provides CtsTestBase with additional socket-specific helpers like
// CreateListener(), EstablishConnection(), etc. Socket tests should inherit
// from SocketTestBase; non-socket tests (core, futex, etc.) inherit directly
// from CtsTestBase.
//
// This separation keeps test_base.h free of socket dependencies, allowing
// non-socket tests to compile on platforms without full socket support.

#ifndef IREE_ASYNC_CTS_UTIL_SOCKET_TEST_BASE_H_
#define IREE_ASYNC_CTS_UTIL_SOCKET_TEST_BASE_H_

#include "iree/async/cts/util/socket_test_util.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/socket.h"

namespace iree::async::cts {

// Extended test fixture for socket tests.
// Adds helper methods for common socket test patterns.
template <typename BaseType = ::testing::TestWithParam<BackendInfo>>
class SocketTestBase : public CtsTestBase<BaseType> {
 protected:
  // Creates a TCP listener socket bound to localhost on an ephemeral port.
  // Shared by SocketTest, MultishotTest, ErrorPropagationTest, etc.
  // Returns the listener socket; writes the bound address to |out_address|.
  iree_async_socket_t* CreateListener(iree_async_address_t* out_address) {
    return CreateListenerWithOptions(out_address,
                                     IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR);
  }

  // Creates a TCP listener socket with custom options.
  // Options like IREE_ASYNC_SOCKET_OPTION_ZERO_COPY propagate to accepted
  // sockets.
  iree_async_socket_t* CreateListenerWithOptions(
      iree_async_address_t* out_address, iree_async_socket_options_t options) {
    iree_async_socket_t* listener = nullptr;
    IREE_CHECK_OK(iree_async_socket_create(
        this->proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
        options | IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR, &listener));
    iree_async_address_t bind_address;
    IREE_CHECK_OK(
        iree_async_address_from_ipv4(IREE_SV("127.0.0.1"), 0, &bind_address));
    IREE_CHECK_OK(iree_async_socket_bind(listener, &bind_address));
    IREE_CHECK_OK(iree_async_socket_listen(listener, /*backlog=*/16));
    IREE_CHECK_OK(iree_async_socket_query_local_address(listener, out_address));
    return listener;
  }

  // Returns a loopback address where nothing is listening, guaranteed to
  // produce ECONNREFUSED on any platform. Works by binding a listener to an
  // ephemeral port, recording the address, then closing the listener. The
  // kernel knows this port was just in LISTEN state and sends RST immediately.
  //
  // This is portable across firewalls, macOS stealth mode, Docker/bwrap
  // sandboxes, and FreeBSD tcp.blackhole — environments where connecting to a
  // hardcoded well-known port (like port 1) may silently drop SYN packets
  // instead of sending RST, causing the connect to hang.
  iree_async_address_t CreateRefusedAddress() {
    iree_async_address_t address;
    iree_async_socket_t* listener = CreateListener(&address);
    iree_async_socket_release(listener);
    return address;
  }

  // Establishes a connected client/server pair via loopback.
  // Creates a listener, submits accept+connect, and polls until both complete.
  // Caller must release all three sockets when done.
  void EstablishConnection(iree_async_socket_t** out_client,
                           iree_async_socket_t** out_server,
                           iree_async_socket_t** out_listener) {
    EstablishConnectionWithOptions(out_client, out_server, out_listener,
                                   IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                   IREE_ASYNC_SOCKET_OPTION_NONE);
  }

  // Establishes a connected client/server pair with custom socket options.
  // |client_options| are applied to the client socket at creation.
  // |listener_options| are applied to the listener and inherited by accepted
  // sockets (e.g., IREE_ASYNC_SOCKET_OPTION_ZERO_COPY propagates to server).
  void EstablishConnectionWithOptions(
      iree_async_socket_t** out_client, iree_async_socket_t** out_server,
      iree_async_socket_t** out_listener,
      iree_async_socket_options_t client_options,
      iree_async_socket_options_t listener_options) {
    iree_async_address_t listen_address;
    *out_listener =
        CreateListenerWithOptions(&listen_address, listener_options);

    iree_async_socket_accept_operation_t accept_op;
    CompletionTracker accept_tracker;
    InitAcceptOperation(&accept_op, *out_listener, CompletionTracker::Callback,
                        &accept_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(this->proactor_, &accept_op.base));

    IREE_ASSERT_OK(iree_async_socket_create(this->proactor_,
                                            IREE_ASYNC_SOCKET_TYPE_TCP,
                                            client_options, out_client));

    iree_async_socket_connect_operation_t connect_op;
    CompletionTracker connect_tracker;
    InitConnectOperation(&connect_op, *out_client, listen_address,
                         CompletionTracker::Callback, &connect_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(this->proactor_, &connect_op.base));

    this->PollUntil(/*min_completions=*/2,
                    /*total_budget=*/iree_make_duration_ms(5000));

    ASSERT_NE(accept_op.accepted_socket, nullptr);
    *out_server = accept_op.accepted_socket;
  }

  // Receives up to |expected_length| bytes into |buffer|, returning actual
  // bytes received. Returns early on EOF (recv returns 0 bytes).
  //
  // This helper properly waits for each recv operation to complete using
  // operation-specific tracking, rather than relying on global completion
  // counts (which can miscount when sends complete concurrently).
  iree_host_size_t RecvAll(
      iree_async_socket_t* socket, uint8_t* buffer,
      iree_host_size_t expected_length,
      iree_duration_t timeout = iree_make_duration_ms(5000)) {
    iree_host_size_t total_received = 0;
    while (total_received < expected_length) {
      iree_async_span_t recv_span = iree_async_span_from_ptr(
          buffer + total_received, expected_length - total_received);

      iree_async_socket_recv_operation_t recv_op;
      CompletionTracker recv_tracker;
      InitRecvOperation(&recv_op, socket, &recv_span, 1,
                        CompletionTracker::Callback, &recv_tracker);

      IREE_CHECK_OK(
          iree_async_proactor_submit_one(this->proactor_, &recv_op.base));

      // Wait for this specific recv to complete. PollUntil counts all
      // completions globally, so we must check the tracker rather than
      // assuming the completion we wait for is our recv.
      while (recv_tracker.call_count == 0) {
        this->PollUntil(/*min_completions=*/1, /*total_budget=*/timeout);
      }

      if (recv_op.bytes_received == 0) break;  // EOF.
      total_received += recv_op.bytes_received;
    }
    return total_received;
  }
};

}  // namespace iree::async::cts

#endif  // IREE_ASYNC_CTS_UTIL_SOCKET_TEST_BASE_H_
