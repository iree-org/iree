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

#include <string.h>

#if defined(IREE_PLATFORM_WINDOWS)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#else
#include <sys/socket.h>
#endif  // IREE_PLATFORM_WINDOWS

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

  // Poll budget for connect attempts to CreateRefusedAddress() addresses.
  //
  // On POSIX loopback the refusal is effectively immediate: the RST (Reset
  // segment - "there is no connection here, stop immediately") elicited by
  // the SYN surfaces as ECONNREFUSED on the first poll. Windows is slower by
  // design, in two independent ways:
  //
  // 1. Retry-after-RST: Winsock does not fail the connect on the first RST;
  //    it retransmits the SYN and reports WSAECONNREFUSED only once the
  //    connect retry budget is exhausted. Because the RST proves the target
  //    is reachable, the retry timeout is not doubled between attempts. Per
  //    MS KB175523 "INFO: Winsock TCP Connection Performance to Unused
  //    Ports" (archived): "As long as an ACK/RST packet from an unused port
  //    is received, the time-out value will not increase and the process
  //    will repeat until the maximum retry value is reached."
  //    https://mskb.pkisolutions.com/kb/175523
  //    Net effect: a refused loopback connect takes on the order of seconds,
  //    not microseconds.
  //
  // 2. Unanswered-SYN backoff: if a SYN or its RST reply is dropped outright
  //    (loaded CI runners), the connect falls back to retransmission after
  //    the RTO (Retransmission TimeOut - how long TCP waits for an ACK
  //    before retransmitting a segment) with exponential backoff. With the
  //    defaults documented for TcpMaxConnectRetransmissions (initial timeout
  //    3s, "doubled with each successive retransmission", default 2
  //    retransmissions) the worst case is 3+6+12 = ~21 seconds:
  //    https://learn.microsoft.com/en-us/troubleshoot/windows-client/networking/tcpip-and-nbt-configuration-parameters
  //    Modern Windows exposes the equivalent knobs via Get/Set-NetTCPSetting
  //    as InitialRtoMs ("the period ... before connect, or SYN, retransmit")
  //    and MaxSynRetransmissions ("the maximum number of times the computer
  //    sends SYN packets without receiving a response"):
  //    https://learn.microsoft.com/en-us/powershell/module/nettcpip/set-nettcpsetting
  //
  // 30s covers the Windows worst case with margin. This does not slow down
  // healthy runs: PollUntil() returns as soon as the completion arrives; the
  // budget only bounds how long a genuinely broken run takes to fail.
  static constexpr iree_duration_t kRefusedConnectBudget =
      30ll * 1000 * 1000 * 1000;  // 30s

  // Returns a loopback address guaranteed to refuse TCP connections
  // (ECONNREFUSED / WSAECONNREFUSED) for as long as |*out_guard| stays alive.
  //
  // Implementation: binds a TCP socket to an ephemeral loopback port and
  // never calls listen(). RFC 9293 3.5.2 (Reset Generation, group 1:
  // connection does not exist) requires: "a reset is sent in response to any
  // incoming segment except another reset. A SYN segment that does not match
  // an existing connection is rejected by this means."
  // https://www.rfc-editor.org/rfc/rfc9293.html#section-3.5.2
  // The RFC speaks of connections, not sockets; the sockets-API mapping is
  // that bind() alone creates neither a connection nor a listener (only
  // listen() moves a socket to LISTEN), so an incoming SYN matches nothing
  // and the port is CLOSED as far as segment matching is concerned. Linux,
  // macOS, and Windows all implement this by answering the SYN with RST -
  // identical to a port with no socket at all. The bound-but-not-listening
  // socket merely reserves the port number so nothing else can claim it.
  //
  // Keeping the bound socket alive (the guard) is what makes this race-free:
  // the port cannot be handed back to the ephemeral allocator and claimed by
  // a concurrent test (or another process) whose listener would accept our
  // connect instead of refusing it. An earlier revision closed a listener
  // and connected to its stale port, which raced exactly that way; the
  // Windows variant co-bound a SO_REUSEADDR guard, but SO_REUSEADDR itself
  // permits any other SO_REUSEADDR socket to bind the same port and start
  // listening, so even the guarded port was hijackable:
  // https://learn.microsoft.com/en-us/windows/win32/winsock/using-so-reuseaddr-and-so-exclusiveaddruse
  // The bind-only guard requests no SO_REUSEADDR, so the port is exclusively
  // owned until the guard is released.
  //
  // NOTE: the refusal is deterministic but not necessarily fast; use
  // kRefusedConnectBudget when polling for the connect completion and
  // ReapIfPending() before releasing the sockets (see kRefusedConnectBudget
  // for the Windows SYN-retry latency details).
  //
  // The guard is returned via |out_guard| on all platforms and MUST be
  // released by the caller after the connect attempt completes.
  iree_async_address_t CreateRefusedAddress(iree_async_socket_t** out_guard) {
    iree_async_socket_t* guard = nullptr;
    IREE_CHECK_OK(
        iree_async_socket_create(this->proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                 IREE_ASYNC_SOCKET_OPTION_NONE, &guard));
    iree_async_address_t bind_address;
    IREE_CHECK_OK(
        iree_async_address_from_ipv4(IREE_SV("127.0.0.1"), 0, &bind_address));
    IREE_CHECK_OK(iree_async_socket_bind(guard, &bind_address));
    iree_async_address_t address;
    IREE_CHECK_OK(iree_async_socket_query_local_address(guard, &address));
    *out_guard = guard;
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

  // Releases a socket with LINGER_ZERO (forcing RST instead of FIN) and
  // submits a recv on the peer to detect the RST and set sticky failure.
  // After this call, |peer_socket| has sticky failure set and any subsequent
  // send operations will fail immediately via the eager-send sticky check.
  //
  // Accepted sockets inherit the default linger behavior (graceful FIN on
  // close). When a test needs deterministic error detection after closing one
  // end of a connection, this helper ensures close sends RST, which the peer
  // detects via a recv probe. Without this, eager sends can complete
  // successfully (writev deposits data in the kernel buffer) before the FIN→
  // RST roundtrip propagates back — making error detection non-deterministic.
  void ReleaseWithRst(iree_async_socket_t* socket_to_close,
                      iree_async_socket_t* peer_socket) {
    // Force LINGER_ZERO so close() sends RST deterministically.
    struct linger linger_opt;
    memset(&linger_opt, 0, sizeof(linger_opt));
    linger_opt.l_onoff = 1;
    linger_opt.l_linger = 0;
#if defined(IREE_PLATFORM_WINDOWS)
    setsockopt((SOCKET)socket_to_close->primitive.value.win32_handle,
               SOL_SOCKET, SO_LINGER, (const char*)&linger_opt,
               sizeof(linger_opt));
#else
    setsockopt(socket_to_close->primitive.value.fd, SOL_SOCKET, SO_LINGER,
               &linger_opt, sizeof(linger_opt));
#endif  // IREE_PLATFORM_WINDOWS

    iree_async_socket_release(socket_to_close);

    // Submit recv on peer to detect RST. On loopback, LINGER_ZERO RST is
    // delivered synchronously within close(), so readv() fails immediately
    // with ECONNRESET — setting sticky failure on the peer socket.
    char rst_probe_buffer[1] = {0};
    iree_async_span_t rst_probe_span =
        iree_async_span_from_ptr(rst_probe_buffer, sizeof(rst_probe_buffer));
    iree_async_socket_recv_operation_t rst_probe_op;
    CompletionTracker rst_probe_tracker;
    InitRecvOperation(&rst_probe_op, peer_socket, &rst_probe_span, 1,
                      CompletionTracker::Callback, &rst_probe_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(this->proactor_, &rst_probe_op.base));
    this->PollUntil(/*min_completions=*/1,
                    /*total_budget=*/iree_make_duration_ms(5000));
    iree_status_ignore(rst_probe_tracker.ConsumeStatus());
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
