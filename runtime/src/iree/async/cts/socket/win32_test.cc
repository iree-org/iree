// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Windows-specific socket CTS tests.
//
// Tests that require Windows socket APIs (winsock2.h, etc.) for setup or
// verification. These tests validate platform-specific behavior like SOCKET
// handle import and socket option verification via Windows getsockopt().

#include "iree/base/api.h"

#if defined(IREE_PLATFORM_WINDOWS)

// Windows headers must be included in specific order.
// clang-format off
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
// clang-format on

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/socket.h"

namespace iree::async::cts {

class SocketWin32Test : public SocketTestBase<> {
 protected:
  // Ensures Winsock is initialized before tests run.
  static void SetUpTestSuite() {
    WSADATA wsa_data;
    int result = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    ASSERT_EQ(result, 0) << "WSAStartup failed with error: " << result;
  }

  static void TearDownTestSuite() { WSACleanup(); }
};

//===----------------------------------------------------------------------===//
// Import tests: wrapping Winsock SOCKET handles
//===----------------------------------------------------------------------===//

// Import a raw Winsock SOCKET and verify it works with the proactor.
TEST_P(SocketWin32Test, ImportSocket_FromWinsock) {
  // Create a raw TCP socket using Winsock APIs.
  SOCKET raw_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  ASSERT_NE(raw_socket, INVALID_SOCKET)
      << "Failed to create raw socket: " << WSAGetLastError();

  // Import into the proactor.
  // On Windows, SOCKET is stored as win32_handle (both are pointer-sized).
  iree_async_primitive_t primitive;
  primitive.type = IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE;
  primitive.value.win32_handle = (uintptr_t)raw_socket;

  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(
      iree_async_socket_import(proactor_, primitive, IREE_ASYNC_SOCKET_TYPE_TCP,
                               IREE_ASYNC_SOCKET_FLAG_NONE, &socket));

  ASSERT_NE(socket, nullptr);
  EXPECT_EQ(iree_async_socket_query_state(socket),
            IREE_ASYNC_SOCKET_STATE_CREATED);

  // Verify the imported socket can be used for bind.
  iree_async_address_t address;
  IREE_ASSERT_OK(
      iree_async_address_from_ipv4(iree_string_view_empty(), 0, &address));
  IREE_EXPECT_OK(iree_async_socket_bind(socket, &address));

  // Release closes the underlying socket.
  iree_async_socket_release(socket);
}

// Import with INVALID_SOCKET should fail.
TEST_P(SocketWin32Test, ImportSocket_InvalidHandle) {
  iree_async_primitive_t primitive;
  primitive.type = IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE;
  primitive.value.win32_handle = (uintptr_t)INVALID_SOCKET;

  iree_async_socket_t* socket = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_socket_import(proactor_, primitive, IREE_ASYNC_SOCKET_TYPE_TCP,
                               IREE_ASYNC_SOCKET_FLAG_NONE, &socket));
  EXPECT_EQ(socket, nullptr);
}

// Import a closed socket handle should fail gracefully.
TEST_P(SocketWin32Test, ImportSocket_ClosedHandle) {
  // Create and immediately close a socket to get a stale handle.
  SOCKET raw_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  ASSERT_NE(raw_socket, INVALID_SOCKET);
  closesocket(raw_socket);

  iree_async_primitive_t primitive;
  primitive.type = IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE;
  primitive.value.win32_handle = (uintptr_t)raw_socket;

  // Import should fail (handle is no longer valid).
  iree_async_socket_t* socket = nullptr;
  iree_status_t status =
      iree_async_socket_import(proactor_, primitive, IREE_ASYNC_SOCKET_TYPE_TCP,
                               IREE_ASYNC_SOCKET_FLAG_NONE, &socket);

  // Either INVALID_ARGUMENT (detected bad handle) or the import succeeds but
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
// Socket option verification via getsockopt()
//===----------------------------------------------------------------------===//

// Helper to get a boolean socket option on Windows.
static bool GetSocketOptBool(SOCKET sock, int level, int optname) {
  char value = 0;
  int len = sizeof(value);
  if (getsockopt(sock, level, optname, &value, &len) == SOCKET_ERROR) {
    return false;
  }
  return value != 0;
}

// Verify SO_REUSEADDR is applied to the underlying socket.
TEST_P(SocketWin32Test, VerifyOption_ReuseAddr) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR,
                                          &socket));

  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;
  EXPECT_TRUE(GetSocketOptBool(sock, SOL_SOCKET, SO_REUSEADDR))
      << "SO_REUSEADDR should be set";

  iree_async_socket_release(socket);
}

// Note: SO_REUSEPORT is not available on Windows. The closest equivalent is
// SO_REUSEADDR which has different semantics. We don't test REUSE_PORT on
// Windows since the option flag may be ignored or mapped differently.

// Verify TCP_NODELAY is applied to the underlying socket.
TEST_P(SocketWin32Test, VerifyOption_NoDelay) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &socket));

  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;
  EXPECT_TRUE(GetSocketOptBool(sock, IPPROTO_TCP, TCP_NODELAY))
      << "TCP_NODELAY should be set";

  iree_async_socket_release(socket);
}

// Verify SO_KEEPALIVE is applied to the underlying socket.
TEST_P(SocketWin32Test, VerifyOption_KeepAlive) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_KEEP_ALIVE,
                                          &socket));

  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;
  EXPECT_TRUE(GetSocketOptBool(sock, SOL_SOCKET, SO_KEEPALIVE))
      << "SO_KEEPALIVE should be set";

  iree_async_socket_release(socket);
}

// Verify LINGER_ZERO option is applied correctly (SO_LINGER with l_linger=0).
TEST_P(SocketWin32Test, VerifyOption_LingerZero) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_LINGER_ZERO,
                                          &socket));

  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;
  struct linger linger_opt;
  int linger_len = sizeof(linger_opt);
  ASSERT_EQ(
      getsockopt(sock, SOL_SOCKET, SO_LINGER, (char*)&linger_opt, &linger_len),
      0)
      << "getsockopt SO_LINGER failed: " << WSAGetLastError();

  EXPECT_NE(linger_opt.l_onoff, 0) << "l_onoff should be non-zero (enabled)";
  EXPECT_EQ(linger_opt.l_linger, 0) << "l_linger should be 0 (immediate RST)";

  iree_async_socket_release(socket);
}

CTS_REGISTER_TEST_SUITE(SocketWin32Test);

}  // namespace iree::async::cts

#endif  // IREE_PLATFORM_WINDOWS
