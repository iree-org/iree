// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for the SHM cross-process handshake.
//
// Creates a connected channel pair in-process, then runs the server and client
// handshakes concurrently on separate threads. This validates the complete
// exchange: handle passing (SCM_RIGHTS on POSIX, DuplicateHandle on Windows),
// SHM region creation/mapping, ring initialization, and carrier parameter
// assembly.
//
// POSIX: uses socketpair(AF_UNIX) for the connected pair.
// Windows: creates a named pipe pair (CreateNamedPipeW + CreateFileW) since
//   the handshake uses overlapped ReadFile/WriteFile on pipes.

#include "iree/net/carrier/shm/handshake.h"

#if !defined(IREE_PLATFORM_WINDOWS)
#include <sys/socket.h>
#else
#include <windows.h>
#endif  // !IREE_PLATFORM_WINDOWS

#include <atomic>
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

  // Creates a connected channel pair and wraps both ends as primitives.
  // The handshake functions take ownership and close the channels on return.
  void CreateChannelPair(iree_async_primitive_t* out_server,
                         iree_async_primitive_t* out_client) {
#if !defined(IREE_PLATFORM_WINDOWS)
    int fds[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0)
        << "socketpair: " << strerror(errno);
    *out_server = iree_async_primitive_from_fd(fds[0]);
    *out_client = iree_async_primitive_from_fd(fds[1]);
#else
    // Create a named pipe pair. The server creates the pipe, the client opens
    // it with CreateFile. Both sides use FILE_FLAG_OVERLAPPED because the
    // handshake uses overlapped ReadFile/WriteFile with timeout support.
    static std::atomic<int> pipe_counter{0};
    DWORD pid = GetCurrentProcessId();
    int instance = pipe_counter.fetch_add(1, std::memory_order_relaxed);

    WCHAR pipe_name[MAX_PATH];
    swprintf_s(pipe_name, MAX_PATH, L"\\\\.\\pipe\\iree-handshake-test-%lu-%d",
               (unsigned long)pid, instance);

    HANDLE server_pipe =
        CreateNamedPipeW(pipe_name, PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                         PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                         1,  // Single instance (sufficient for test).
                         4096, 4096, 0, NULL);
    ASSERT_NE(server_pipe, INVALID_HANDLE_VALUE)
        << "CreateNamedPipeW: " << GetLastError();

    HANDLE client_pipe =
        CreateFileW(pipe_name, GENERIC_READ | GENERIC_WRITE, 0, NULL,
                    OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    ASSERT_NE(client_pipe, INVALID_HANDLE_VALUE)
        << "CreateFileW: " << GetLastError();

    // ConnectNamedPipe should return ERROR_PIPE_CONNECTED since the client
    // already connected above. We must provide an OVERLAPPED because the
    // pipe was created with FILE_FLAG_OVERLAPPED.
    HANDLE connect_event = CreateEventW(NULL, /*bManualReset=*/TRUE,
                                        /*bInitialState=*/FALSE, NULL);
    ASSERT_NE(connect_event, (HANDLE)NULL) << "CreateEvent: " << GetLastError();
    OVERLAPPED connect_overlapped = {};
    connect_overlapped.hEvent = connect_event;
    BOOL connected = ConnectNamedPipe(server_pipe, &connect_overlapped);
    if (!connected) {
      DWORD error = GetLastError();
      if (error == ERROR_IO_PENDING) {
        // Shouldn't happen since client already connected, but handle it.
        WaitForSingleObject(connect_event, INFINITE);
      } else {
        ASSERT_EQ(error, (DWORD)ERROR_PIPE_CONNECTED)
            << "ConnectNamedPipe unexpected error: " << error;
      }
    }
    CloseHandle(connect_event);

    *out_server =
        iree_async_primitive_from_win32_handle((uintptr_t)server_pipe);
    *out_client =
        iree_async_primitive_from_win32_handle((uintptr_t)client_pipe);
#endif  // !IREE_PLATFORM_WINDOWS
  }

  // Runs both sides of the handshake concurrently and collects results.
  struct HandshakePairResult {
    iree_status_t server_status;
    iree_status_t client_status;
    iree_net_shm_handshake_result_t server;
    iree_net_shm_handshake_result_t client;
  };

  HandshakePairResult RunHandshake(iree_net_shm_carrier_options_t options =
                                       iree_net_shm_carrier_options_default()) {
    iree_async_primitive_t server_channel, client_channel;
    CreateChannelPair(&server_channel, &client_channel);

    HandshakePairResult result = {};
    memset(&result.server, 0, sizeof(result.server));
    memset(&result.client, 0, sizeof(result.client));

    std::thread server_thread([&] {
      result.server_status = iree_net_shm_handshake_server(
          server_channel, server_wake_, options, proactor_,
          iree_allocator_system(), &result.server);
    });
    std::thread client_thread([&] {
      result.client_status = iree_net_shm_handshake_client(
          client_channel, client_wake_, proactor_, iree_allocator_system(),
          &result.client);
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
  IREE_ASSERT_OK(
      iree_net_shm_carrier_create(&result.server.carrier_params, null_callback,
                                  iree_allocator_system(), &server_carrier));

  iree_net_carrier_t* client_carrier = nullptr;
  IREE_ASSERT_OK(
      iree_net_shm_carrier_create(&result.client.carrier_params, null_callback,
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

  // The server validates ring_capacity before touching the channel, so we
  // don't need a real connected pair. A dummy channel that gets closed is fine.
  iree_async_primitive_t server_channel, client_channel;
  CreateChannelPair(&server_channel, &client_channel);

  iree_net_shm_handshake_result_t server_result = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_shm_handshake_server(
                            server_channel, server_wake_, options, proactor_,
                            iree_allocator_system(), &server_result));

  // Server closed its channel. Clean up the client channel.
  iree_async_primitive_close(&client_channel);
}

TEST_F(HandshakeTest, NoneChannelFails) {
  iree_net_shm_handshake_result_t result = {};

  // Server with NONE channel should fail.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_shm_handshake_server(
                            iree_async_primitive_none(), server_wake_,
                            iree_net_shm_carrier_options_default(), proactor_,
                            iree_allocator_system(), &result));
}

}  // namespace
