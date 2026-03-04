// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for the SHM cross-process handshake.
//
// Creates a connected socket pair in-process, then runs the server and client
// handshakes concurrently on separate threads. This validates the complete
// exchange: handle passing (SCM_RIGHTS on POSIX, DuplicateHandle on Windows),
// SHM region creation/mapping, ring initialization, and carrier parameter
// assembly.
//
// POSIX: uses socketpair(AF_UNIX) for the connected pair.
// Windows: creates a TCP loopback pair via bind/listen/connect/accept since
//   socketpair is not available.

#include "iree/net/carrier/shm/handshake.h"

#if !defined(IREE_PLATFORM_WINDOWS)
#include <sys/socket.h>
#else
// clang-format off
#include <winsock2.h>
#include <ws2tcpip.h>
// clang-format on
#endif  // !IREE_PLATFORM_WINDOWS

#include <thread>

#include "iree/async/proactor_platform.h"
#include "iree/net/carrier/shm/carrier.h"
#include "iree/net/carrier/shm/shared_wake.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class HandshakeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_status_t status = iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_);
    if (iree_status_is_unavailable(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "Platform proactor unavailable";
    }
    IREE_ASSERT_OK(status);

    IREE_ASSERT_OK(iree_net_shm_shared_wake_create_shared(
        proactor_, iree_allocator_system(), &server_wake_));
    IREE_ASSERT_OK(iree_net_shm_shared_wake_create_shared(
        proactor_, iree_allocator_system(), &client_wake_));
  }

  void TearDown() override {
    if (client_wake_) {
      iree_net_shm_shared_wake_release(client_wake_);
      client_wake_ = nullptr;
    }
    if (server_wake_) {
      iree_net_shm_shared_wake_release(server_wake_);
      server_wake_ = nullptr;
    }
    if (proactor_) {
      iree_async_proactor_release(proactor_);
      proactor_ = nullptr;
    }
  }

  // Creates a connected socket pair and wraps both ends as primitives.
  // The handshake functions take ownership and close the sockets on return.
  void CreateSocketPair(iree_async_primitive_t* out_server,
                        iree_async_primitive_t* out_client) {
#if !defined(IREE_PLATFORM_WINDOWS)
    int fds[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0)
        << "socketpair: " << strerror(errno);
    *out_server = iree_async_primitive_from_fd(fds[0]);
    *out_client = iree_async_primitive_from_fd(fds[1]);
#else
    // Windows lacks socketpair. Create a connected TCP loopback pair manually:
    // bind a listener to 127.0.0.1:0, connect to the ephemeral port, accept,
    // and close the listener. The proactor's WSAStartup handles initialization.
    SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    ASSERT_NE(listener, INVALID_SOCKET)
        << "listener socket: " << WSAGetLastError();

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;  // Ephemeral.
    ASSERT_EQ(bind(listener, (struct sockaddr*)&addr, sizeof(addr)), 0)
        << "bind: " << WSAGetLastError();

    int addr_length = sizeof(addr);
    ASSERT_EQ(getsockname(listener, (struct sockaddr*)&addr, &addr_length), 0)
        << "getsockname: " << WSAGetLastError();

    ASSERT_EQ(listen(listener, 1), 0) << "listen: " << WSAGetLastError();

    SOCKET client = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    ASSERT_NE(client, INVALID_SOCKET)
        << "client socket: " << WSAGetLastError();
    ASSERT_EQ(connect(client, (struct sockaddr*)&addr, sizeof(addr)), 0)
        << "connect: " << WSAGetLastError();

    SOCKET server = accept(listener, NULL, NULL);
    ASSERT_NE(server, INVALID_SOCKET)
        << "accept: " << WSAGetLastError();

    closesocket(listener);

    *out_server = iree_async_primitive_from_win32_handle((uintptr_t)server);
    *out_client = iree_async_primitive_from_win32_handle((uintptr_t)client);
#endif  // !IREE_PLATFORM_WINDOWS
  }

  // Runs both sides of the handshake concurrently and collects results.
  struct HandshakePairResult {
    iree_status_t server_status;
    iree_status_t client_status;
    iree_net_shm_handshake_result_t server;
    iree_net_shm_handshake_result_t client;
  };

  HandshakePairResult RunHandshake(
      iree_net_shm_carrier_options_t options =
          iree_net_shm_carrier_options_default()) {
    iree_async_primitive_t server_socket, client_socket;
    CreateSocketPair(&server_socket, &client_socket);

    HandshakePairResult result = {};
    memset(&result.server, 0, sizeof(result.server));
    memset(&result.client, 0, sizeof(result.client));

    std::thread server_thread([&] {
      result.server_status = iree_net_shm_handshake_server(
          server_socket, server_wake_, options, proactor_,
          iree_allocator_system(), &result.server);
    });
    std::thread client_thread([&] {
      result.client_status = iree_net_shm_handshake_client(
          client_socket, client_wake_, proactor_,
          iree_allocator_system(), &result.client);
    });

    server_thread.join();
    client_thread.join();
    return result;
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_net_shm_shared_wake_t* server_wake_ = nullptr;
  iree_net_shm_shared_wake_t* client_wake_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Success path
//===----------------------------------------------------------------------===//

TEST_F(HandshakeTest, BasicExchangeSucceeds) {
  auto result = RunHandshake();
  IREE_ASSERT_OK(result.server_status);
  IREE_ASSERT_OK(result.client_status);

  // Server params: not a client, has valid queues and armed flags.
  const auto& server_params = result.server.carrier_params;
  EXPECT_FALSE(server_params.is_client);
  EXPECT_NE(server_params.tx_queue.header, nullptr);
  EXPECT_NE(server_params.rx_queue.header, nullptr);
  EXPECT_NE(server_params.our_armed, nullptr);
  EXPECT_NE(server_params.peer_armed, nullptr);
  EXPECT_EQ(server_params.shared_wake, server_wake_);
  EXPECT_NE(server_params.peer_wake_notification, nullptr);
  EXPECT_NE(server_params.release_context, nullptr);
  EXPECT_NE(server_params.release_context_fn, nullptr);
  ASSERT_NE(result.server.context, nullptr);

  // Client params: is a client, has valid queues and armed flags.
  const auto& client_params = result.client.carrier_params;
  EXPECT_TRUE(client_params.is_client);
  EXPECT_NE(client_params.tx_queue.header, nullptr);
  EXPECT_NE(client_params.rx_queue.header, nullptr);
  EXPECT_NE(client_params.our_armed, nullptr);
  EXPECT_NE(client_params.peer_armed, nullptr);
  EXPECT_EQ(client_params.shared_wake, client_wake_);
  EXPECT_NE(client_params.peer_wake_notification, nullptr);
  EXPECT_NE(client_params.release_context, nullptr);
  EXPECT_NE(client_params.release_context_fn, nullptr);
  ASSERT_NE(result.client.context, nullptr);

  // TX and RX queues should be distinct (Ring A vs Ring B).
  EXPECT_NE(server_params.tx_queue.header, server_params.rx_queue.header);
  EXPECT_NE(client_params.tx_queue.header, client_params.rx_queue.header);

  // Our armed and peer armed should be distinct (consumer_a vs consumer_b).
  EXPECT_NE(server_params.our_armed, server_params.peer_armed);
  EXPECT_NE(client_params.our_armed, client_params.peer_armed);

  iree_net_shm_xproc_context_release(result.server.context);
  iree_net_shm_xproc_context_release(result.client.context);
}

TEST_F(HandshakeTest, CarriersCanBeCreatedFromResults) {
  auto result = RunHandshake();
  IREE_ASSERT_OK(result.server_status);
  IREE_ASSERT_OK(result.client_status);

  // Creating carriers validates that the params are fully well-formed.
  // The carrier takes ownership of the release_context, so we don't need
  // to release the xproc_context separately.
  iree_net_carrier_callback_t null_callback = {nullptr, nullptr};

  iree_net_carrier_t* server_carrier = nullptr;
  IREE_ASSERT_OK(iree_net_shm_carrier_create(
      &result.server.carrier_params, null_callback,
      iree_allocator_system(), &server_carrier));

  iree_net_carrier_t* client_carrier = nullptr;
  IREE_ASSERT_OK(iree_net_shm_carrier_create(
      &result.client.carrier_params, null_callback,
      iree_allocator_system(), &client_carrier));

  EXPECT_NE(server_carrier, nullptr);
  EXPECT_NE(client_carrier, nullptr);

  // Release both carriers. This triggers xproc_context cleanup (unmap SHM,
  // release peer notification, close peer signal primitive).
  iree_net_carrier_release(client_carrier);
  iree_net_carrier_release(server_carrier);
}

TEST_F(HandshakeTest, CustomRingCapacity) {
  iree_net_shm_carrier_options_t options =
      iree_net_shm_carrier_options_default();
  options.ring_capacity = 16384;  // 16 KiB instead of default.

  auto result = RunHandshake(options);
  IREE_ASSERT_OK(result.server_status);
  IREE_ASSERT_OK(result.client_status);

  // Verify the queue capacity matches what we requested.
  EXPECT_EQ(result.server.carrier_params.tx_queue.capacity, 16384u);
  EXPECT_EQ(result.server.carrier_params.rx_queue.capacity, 16384u);
  EXPECT_EQ(result.client.carrier_params.tx_queue.capacity, 16384u);
  EXPECT_EQ(result.client.carrier_params.rx_queue.capacity, 16384u);

  iree_net_shm_xproc_context_release(result.server.context);
  iree_net_shm_xproc_context_release(result.client.context);
}

//===----------------------------------------------------------------------===//
// Validation errors
//===----------------------------------------------------------------------===//

TEST_F(HandshakeTest, InvalidRingCapacityFailsValidation) {
  iree_net_shm_carrier_options_t options =
      iree_net_shm_carrier_options_default();
  options.ring_capacity = 5;  // Not a power of two.

  // The server validates ring_capacity before touching the socket, so we
  // don't need a real connected pair. A dummy socket that gets closed is fine.
  iree_async_primitive_t server_socket, client_socket;
  CreateSocketPair(&server_socket, &client_socket);

  iree_net_shm_handshake_result_t server_result = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_server(
          server_socket, server_wake_, options, proactor_,
          iree_allocator_system(), &server_result));

  // Server closed its socket. Clean up the client socket.
  iree_async_primitive_close(&client_socket);
}

TEST_F(HandshakeTest, NoneSocketFails) {
  iree_net_shm_handshake_result_t result = {};

  // Server with NONE socket should fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_handshake_server(
          iree_async_primitive_none(), server_wake_,
          iree_net_shm_carrier_options_default(), proactor_,
          iree_allocator_system(), &result));
}

}  // namespace
