// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// POSIX-specific socket CTS tests.
//
// Tests that require POSIX socket APIs (sys/socket.h, etc.) for setup or
// verification. These tests validate platform-specific behavior like fd import,
// Unix domain abstract namespace sockets, and socket option verification.

#include "iree/base/api.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID) || \
    defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"

namespace iree::async::cts {

class SocketPosixTest : public SocketTestBase<> {
 protected:
  // Returns a unique abstract socket name incorporating PID and a counter to
  // avoid collisions when test binaries run in parallel (--runs_per_test).
  // Abstract namespace sockets are process-global on Linux and persist until
  // all references are closed, so hardcoded names collide across processes.
  std::string UniqueAbstractName(const char* base_name) {
    static std::atomic<int> counter{0};
    char buffer[108];  // sun_path max
    snprintf(buffer, sizeof(buffer), "@iree_cts_%s_%d_%d", base_name,
             (int)getpid(), counter.fetch_add(1, std::memory_order_relaxed));
    return std::string(buffer);
  }
};

//===----------------------------------------------------------------------===//
// Import tests: wrapping raw file descriptors
//===----------------------------------------------------------------------===//

// Import a raw POSIX socket fd and verify it works with the proactor.
TEST_P(SocketPosixTest, ImportSocket_FromRawFd) {
  // Create a raw TCP socket using POSIX APIs.
  int raw_fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(raw_fd, 0) << "Failed to create raw socket: " << strerror(errno);

  // Import into the proactor.
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_import(
      proactor_, iree_async_primitive_from_fd(raw_fd),
      IREE_ASYNC_SOCKET_TYPE_TCP, IREE_ASYNC_SOCKET_FLAG_NONE, &socket));

  ASSERT_NE(socket, nullptr);
  EXPECT_EQ(iree_async_socket_query_state(socket),
            IREE_ASYNC_SOCKET_STATE_CREATED);

  // Verify the imported socket can be used for bind.
  iree_async_address_t address;
  IREE_ASSERT_OK(
      iree_async_address_from_ipv4(iree_string_view_empty(), 0, &address));
  IREE_EXPECT_OK(iree_async_socket_bind(socket, &address));

  // Release closes the underlying fd.
  iree_async_socket_release(socket);
}

// Import with invalid fd (-1) should fail.
TEST_P(SocketPosixTest, ImportSocket_InvalidFd) {
  iree_async_socket_t* socket = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_socket_import(proactor_, iree_async_primitive_from_fd(-1),
                               IREE_ASYNC_SOCKET_TYPE_TCP,
                               IREE_ASYNC_SOCKET_FLAG_NONE, &socket));
  EXPECT_EQ(socket, nullptr);
}

// Import a closed fd should fail gracefully.
TEST_P(SocketPosixTest, ImportSocket_ClosedFd) {
  // Create and immediately close a socket to get a stale fd.
  int raw_fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(raw_fd, 0);
  close(raw_fd);

  // Import should fail (fd is no longer valid).
  iree_async_socket_t* socket = nullptr;
  iree_status_t status = iree_async_socket_import(
      proactor_, iree_async_primitive_from_fd(raw_fd),
      IREE_ASYNC_SOCKET_TYPE_TCP, IREE_ASYNC_SOCKET_OPTION_NONE, &socket);

  // Either INVALID_ARGUMENT (detected bad fd) or the import succeeds but
  // subsequent operations fail. Either behavior is acceptable.
  if (iree_status_is_ok(status)) {
    // Import succeeded - socket should fail on first operation.
    iree_async_address_t address;
    IREE_ASSERT_OK(
        iree_async_address_from_ipv4(iree_string_view_empty(), 0, &address));
    IREE_EXPECT_NOT_OK(iree_async_socket_bind(socket, &address));
    iree_async_socket_release(socket);
  } else {
    // Import failed as expected.
    iree_status_ignore(status);
    EXPECT_EQ(socket, nullptr);
  }
}

//===----------------------------------------------------------------------===//
// Unix domain socket abstract namespace (Linux-only)
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)

// Bind to an abstract namespace Unix socket and verify it works.
TEST_P(SocketPosixTest, UnixSocket_AbstractNamespace) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &socket));

  // Abstract namespace path (@ prefix). Name must be unique per process to
  // avoid collisions under parallel test execution.
  std::string name = UniqueAbstractName("abstract_test");
  iree_async_address_t address;
  IREE_ASSERT_OK(iree_async_address_from_unix(
      iree_make_cstring_view(name.c_str()), &address));

  IREE_EXPECT_OK(iree_async_socket_bind(socket, &address));
  IREE_EXPECT_OK(iree_async_socket_listen(socket, 4));

  EXPECT_EQ(iree_async_socket_query_state(socket),
            IREE_ASYNC_SOCKET_STATE_LISTENING);

  iree_async_socket_release(socket);
}

// Connect to an abstract namespace Unix socket server.
TEST_P(SocketPosixTest, UnixSocket_AbstractConnect) {
  // Create server socket.
  iree_async_socket_t* server = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &server));

  std::string name = UniqueAbstractName("abstract_connect");
  iree_async_address_t address;
  IREE_ASSERT_OK(iree_async_address_from_unix(
      iree_make_cstring_view(name.c_str()), &address));
  IREE_ASSERT_OK(iree_async_socket_bind(server, &address));
  IREE_ASSERT_OK(iree_async_socket_listen(server, 4));

  // Submit accept operation.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, server, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Create client and connect.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &client));

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  // Poll until both complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(1000));

  EXPECT_EQ(accept_tracker.call_count, 1);
  IREE_EXPECT_OK(accept_tracker.ConsumeStatus());
  EXPECT_EQ(connect_tracker.call_count, 1);
  IREE_EXPECT_OK(connect_tracker.ConsumeStatus());

  iree_async_socket_release(accept_op.accepted_socket);
  iree_async_socket_release(client);
  iree_async_socket_release(server);
}

#else  // !Linux/Android

// On non-Linux POSIX platforms, abstract namespace should be unavailable.
TEST_P(SocketPosixTest, UnixSocket_AbstractNamespace_Unavailable) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_async_address_from_unix(
          iree_make_cstring_view("@abstract_not_supported"), &address));
}

#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID

//===----------------------------------------------------------------------===//
// Socket option verification via getsockopt()
//===----------------------------------------------------------------------===//

// Helper to get a boolean socket option.
static bool GetSocketOptBool(int fd, int level, int optname) {
  int value = 0;
  socklen_t len = sizeof(value);
  if (getsockopt(fd, level, optname, &value, &len) < 0) {
    return false;
  }
  return value != 0;
}

// Verify SO_REUSEADDR is applied to the underlying socket.
TEST_P(SocketPosixTest, VerifyOption_ReuseAddr) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR,
                                          &socket));

  int fd = socket->primitive.value.fd;
  EXPECT_TRUE(GetSocketOptBool(fd, SOL_SOCKET, SO_REUSEADDR))
      << "SO_REUSEADDR should be set";

  iree_async_socket_release(socket);
}

// Verify SO_REUSEPORT is applied to the underlying socket.
TEST_P(SocketPosixTest, VerifyOption_ReusePort) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_REUSE_PORT,
                                          &socket));

  int fd = socket->primitive.value.fd;
  EXPECT_TRUE(GetSocketOptBool(fd, SOL_SOCKET, SO_REUSEPORT))
      << "SO_REUSEPORT should be set";

  iree_async_socket_release(socket);
}

// Verify TCP_NODELAY is applied to the underlying socket.
TEST_P(SocketPosixTest, VerifyOption_NoDelay) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &socket));

  int fd = socket->primitive.value.fd;
  EXPECT_TRUE(GetSocketOptBool(fd, IPPROTO_TCP, TCP_NODELAY))
      << "TCP_NODELAY should be set";

  iree_async_socket_release(socket);
}

// Verify SO_KEEPALIVE is applied to the underlying socket.
TEST_P(SocketPosixTest, VerifyOption_KeepAlive) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_KEEP_ALIVE,
                                          &socket));

  int fd = socket->primitive.value.fd;
  EXPECT_TRUE(GetSocketOptBool(fd, SOL_SOCKET, SO_KEEPALIVE))
      << "SO_KEEPALIVE should be set";

  iree_async_socket_release(socket);
}

// Verify LINGER_ZERO option is applied correctly (SO_LINGER with l_linger=0).
TEST_P(SocketPosixTest, VerifyOption_LingerZero) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_LINGER_ZERO,
                                          &socket));

  int fd = socket->primitive.value.fd;
  struct linger linger_opt;
  socklen_t linger_len = sizeof(linger_opt);
  ASSERT_EQ(getsockopt(fd, SOL_SOCKET, SO_LINGER, &linger_opt, &linger_len), 0)
      << "getsockopt SO_LINGER failed: " << strerror(errno);

  EXPECT_EQ(linger_opt.l_onoff, 1) << "l_onoff should be 1 (enabled)";
  EXPECT_EQ(linger_opt.l_linger, 0) << "l_linger should be 0 (immediate RST)";

  iree_async_socket_release(socket);
}

//===----------------------------------------------------------------------===//
// Unix datagram socket tests (Linux/Android only - uses abstract namespace)
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)

// Create a Unix datagram socket and verify initial state.
TEST_P(SocketPosixTest, CreateSocket_UnixDgram) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &socket));

  ASSERT_NE(socket, nullptr);
  EXPECT_EQ(iree_async_socket_query_state(socket),
            IREE_ASYNC_SOCKET_STATE_CREATED);

  iree_async_socket_release(socket);
}

// Unix datagram loopback using connected sockets (abstract namespace).
TEST_P(SocketPosixTest, UnixDgram_ConnectedLoopback) {
  // Create two Unix datagram sockets.
  iree_async_socket_t* socket_a = nullptr;
  iree_async_socket_t* socket_b = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &socket_a));
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &socket_b));

  // Bind both to abstract namespace paths. Names must be unique per process.
  std::string name_a = UniqueAbstractName("dgram_a");
  std::string name_b = UniqueAbstractName("dgram_b");
  iree_async_address_t addr_a, addr_b;
  IREE_ASSERT_OK(iree_async_address_from_unix(
      iree_make_cstring_view(name_a.c_str()), &addr_a));
  IREE_ASSERT_OK(iree_async_address_from_unix(
      iree_make_cstring_view(name_b.c_str()), &addr_b));

  IREE_ASSERT_OK(iree_async_socket_bind(socket_a, &addr_a));
  IREE_ASSERT_OK(iree_async_socket_bind(socket_b, &addr_b));

  // Connect each to the other (sets default destination for send).
  iree_async_socket_connect_operation_t connect_a, connect_b;
  CompletionTracker connect_a_tracker, connect_b_tracker;
  InitConnectOperation(&connect_a, socket_a, addr_b,
                       CompletionTracker::Callback, &connect_a_tracker);
  InitConnectOperation(&connect_b, socket_b, addr_a,
                       CompletionTracker::Callback, &connect_b_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_a.base));
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_b.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(1000));

  IREE_ASSERT_OK(connect_a_tracker.ConsumeStatus());
  IREE_ASSERT_OK(connect_b_tracker.ConsumeStatus());

  // Send from A to B.
  const char* message = "Unix datagram test";
  iree_host_size_t message_length = strlen(message);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)message, message_length);

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, socket_a, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  char recv_buffer[256];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, socket_b, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, message_length);

  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, message_length);
  EXPECT_EQ(memcmp(recv_buffer, message, message_length), 0);

  iree_async_socket_release(socket_a);
  iree_async_socket_release(socket_b);
}

// Unix datagram preserves message boundaries.
TEST_P(SocketPosixTest, UnixDgram_MessageBoundary) {
  iree_async_socket_t* socket_a = nullptr;
  iree_async_socket_t* socket_b = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &socket_a));
  IREE_ASSERT_OK(
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &socket_b));

  std::string name_a = UniqueAbstractName("dgram_boundary_a");
  std::string name_b = UniqueAbstractName("dgram_boundary_b");
  iree_async_address_t addr_a, addr_b;
  IREE_ASSERT_OK(iree_async_address_from_unix(
      iree_make_cstring_view(name_a.c_str()), &addr_a));
  IREE_ASSERT_OK(iree_async_address_from_unix(
      iree_make_cstring_view(name_b.c_str()), &addr_b));

  IREE_ASSERT_OK(iree_async_socket_bind(socket_a, &addr_a));
  IREE_ASSERT_OK(iree_async_socket_bind(socket_b, &addr_b));

  // Connect A to B.
  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, socket_a, addr_b,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  // Send two separate datagrams.
  const char* msg1 = "First";
  const char* msg2 = "Second";

  iree_async_span_t span1 = iree_async_span_from_ptr((void*)msg1, strlen(msg1));
  iree_async_span_t span2 = iree_async_span_from_ptr((void*)msg2, strlen(msg2));

  iree_async_socket_send_operation_t send1, send2;
  CompletionTracker send1_tracker, send2_tracker;
  InitSendOperation(&send1, socket_a, &span1, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send1_tracker);
  InitSendOperation(&send2, socket_a, &span2, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send2_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send1.base));
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send2.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Receive - each recv should get exactly one datagram.
  char recv1_buffer[256], recv2_buffer[256];
  memset(recv1_buffer, 0, sizeof(recv1_buffer));
  memset(recv2_buffer, 0, sizeof(recv2_buffer));

  iree_async_span_t recv1_span =
      iree_async_span_from_ptr(recv1_buffer, sizeof(recv1_buffer));
  iree_async_span_t recv2_span =
      iree_async_span_from_ptr(recv2_buffer, sizeof(recv2_buffer));

  iree_async_socket_recv_operation_t recv1, recv2;
  CompletionTracker recv1_tracker, recv2_tracker;
  InitRecvOperation(&recv1, socket_b, &recv1_span, 1,
                    CompletionTracker::Callback, &recv1_tracker);
  InitRecvOperation(&recv2, socket_b, &recv2_span, 1,
                    CompletionTracker::Callback, &recv2_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv1.base));
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv2.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(recv1_tracker.ConsumeStatus());
  IREE_EXPECT_OK(recv2_tracker.ConsumeStatus());

  // Unix datagrams are reliable and ordered, so we expect msg1 then msg2.
  EXPECT_EQ(recv1.bytes_received, strlen(msg1));
  EXPECT_EQ(memcmp(recv1_buffer, msg1, strlen(msg1)), 0);
  EXPECT_EQ(recv2.bytes_received, strlen(msg2));
  EXPECT_EQ(memcmp(recv2_buffer, msg2, strlen(msg2)), 0);

  iree_async_socket_release(socket_a);
  iree_async_socket_release(socket_b);
}

#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(SocketPosixTest);

}  // namespace iree::async::cts

#endif  // POSIX platforms
