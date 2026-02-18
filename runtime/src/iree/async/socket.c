// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/socket.h"

#include "iree/async/proactor.h"

//===----------------------------------------------------------------------===//
// Platform abstractions
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

// Windows uses SOCKET (unsigned) and closesocket.
typedef SOCKET iree_socket_t;
#define IREE_INVALID_SOCKET INVALID_SOCKET
#define iree_close_socket closesocket

static inline iree_status_t iree_status_from_socket_error(void) {
  int error = WSAGetLastError();
  return iree_make_status(iree_status_code_from_win32_error(error),
                          "socket operation failed (WSA error %d)", error);
}

#else  // POSIX

#include <errno.h>
#include <sys/socket.h>
#include <unistd.h>

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
#include <linux/sockios.h>
#include <sys/ioctl.h>
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID

// POSIX uses int and close.
typedef int iree_socket_t;
#define IREE_INVALID_SOCKET (-1)
#define iree_close_socket close

static inline iree_status_t iree_status_from_socket_error(void) {
  int error = errno;
  return iree_make_status(iree_status_code_from_errno(error),
                          "socket operation failed (errno %d)", error);
}

#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// Socket lifecycle
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_socket_create(
    iree_async_proactor_t* proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_socket);
  *out_socket = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      proactor->vtable->create_socket(proactor, type, options, out_socket);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_async_socket_import(
    iree_async_proactor_t* proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_socket);
  *out_socket = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = proactor->vtable->import_socket(
      proactor, primitive, type, flags, out_socket);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Destroys the socket, releasing backend resources and closing the underlying
// platform handle. Routed through the proactor that created this socket.
// Must not be called while operations referencing this socket are in flight.
static void iree_async_socket_destroy(iree_async_socket_t* socket) {
  IREE_TRACE_ZONE_BEGIN(z0);
  socket->proactor->vtable->destroy_socket(socket->proactor, socket);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_async_socket_retain(iree_async_socket_t* socket) {
  if (socket) {
    iree_atomic_ref_count_inc(&socket->ref_count);
  }
}

IREE_API_EXPORT void iree_async_socket_release(iree_async_socket_t* socket) {
  if (socket && iree_atomic_ref_count_dec(&socket->ref_count) == 1) {
    iree_async_socket_destroy(socket);
  }
}

//===----------------------------------------------------------------------===//
// Socket operations
//===----------------------------------------------------------------------===//

static inline iree_socket_t iree_socket_from_primitive(
    iree_async_primitive_t primitive) {
#if defined(IREE_PLATFORM_WINDOWS)
  // On Windows, SOCKET is stored as win32_handle (both are pointer-sized).
  return (iree_socket_t)primitive.value.win32_handle;
#else
  return primitive.value.fd;
#endif  // IREE_PLATFORM_WINDOWS
}

IREE_API_EXPORT iree_status_t iree_async_socket_bind(
    iree_async_socket_t* socket, const iree_async_address_t* address) {
  // Check sticky failure state.
  iree_status_t failure = iree_async_socket_query_failure(socket);
  if (!iree_status_is_ok(failure)) {
    return iree_status_clone(failure);
  }

  iree_socket_t sock = iree_socket_from_primitive(socket->primitive);
  const struct sockaddr* sa = (const struct sockaddr*)address->storage;

#if defined(IREE_PLATFORM_WINDOWS)
  int result = bind(sock, sa, (int)address->length);
#else
  int result = bind(sock, sa, (socklen_t)address->length);
#endif  // IREE_PLATFORM_WINDOWS

  if (result != 0) {
    return iree_status_from_socket_error();
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_socket_listen(
    iree_async_socket_t* socket, iree_host_size_t backlog) {
  // Check sticky failure state.
  iree_status_t failure = iree_async_socket_query_failure(socket);
  if (!iree_status_is_ok(failure)) {
    return iree_status_clone(failure);
  }

  iree_socket_t sock = iree_socket_from_primitive(socket->primitive);

#if defined(IREE_PLATFORM_WINDOWS)
  // Windows requires explicit bind before listen. POSIX implicitly binds to
  // an ephemeral port when listen() is called on an unbound socket. Replicate
  // that behavior by detecting the unbound state (getsockname returns
  // WSAEINVAL) and binding to INADDR_ANY:0 or in6addr_any:0.
  {
    struct sockaddr_storage probe_address;
    int probe_length = sizeof(probe_address);
    if (getsockname(sock, (struct sockaddr*)&probe_address, &probe_length) ==
        SOCKET_ERROR) {
      // getsockname fails with WSAEINVAL on an unbound socket.
      struct sockaddr_storage bind_addr;
      int bind_addr_length = 0;
      memset(&bind_addr, 0, sizeof(bind_addr));
      switch (socket->type) {
        case IREE_ASYNC_SOCKET_TYPE_TCP:
        case IREE_ASYNC_SOCKET_TYPE_UDP: {
          struct sockaddr_in* addr4 = (struct sockaddr_in*)&bind_addr;
          addr4->sin_family = AF_INET;
          addr4->sin_addr.s_addr = INADDR_ANY;
          addr4->sin_port = 0;
          bind_addr_length = sizeof(struct sockaddr_in);
          break;
        }
        case IREE_ASYNC_SOCKET_TYPE_TCP6:
        case IREE_ASYNC_SOCKET_TYPE_UDP6: {
          struct sockaddr_in6* addr6 = (struct sockaddr_in6*)&bind_addr;
          addr6->sin6_family = AF_INET6;
          addr6->sin6_addr = in6addr_any;
          addr6->sin6_port = 0;
          bind_addr_length = sizeof(struct sockaddr_in6);
          break;
        }
        default:
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "implicit bind not supported for socket type %d", socket->type);
      }
      if (bind(sock, (struct sockaddr*)&bind_addr, bind_addr_length) != 0) {
        return iree_status_from_socket_error();
      }
    }
  }
#endif  // IREE_PLATFORM_WINDOWS

  // Use SOMAXCONN as default if backlog is 0.
  int backlog_int = (backlog == 0) ? SOMAXCONN : (int)backlog;
  if (backlog_int > SOMAXCONN) {
    backlog_int = SOMAXCONN;  // Clamp to system maximum.
  }

  int result = listen(sock, backlog_int);
  if (result != 0) {
    return iree_status_from_socket_error();
  }

  // Update diagnostic state.
  socket->state = IREE_ASYNC_SOCKET_STATE_LISTENING;

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_socket_query_local_address(
    const iree_async_socket_t* socket, iree_async_address_t* out_address) {
  memset(out_address, 0, sizeof(*out_address));

  iree_socket_t sock = iree_socket_from_primitive(socket->primitive);

#if defined(IREE_PLATFORM_WINDOWS)
  int len = (int)sizeof(out_address->storage);
  int result = getsockname(sock, (struct sockaddr*)out_address->storage, &len);
#else
  socklen_t len = (socklen_t)sizeof(out_address->storage);
  int result = getsockname(sock, (struct sockaddr*)out_address->storage, &len);
#endif  // IREE_PLATFORM_WINDOWS

  if (result != 0) {
    return iree_status_from_socket_error();
  }

  out_address->length = (iree_host_size_t)len;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_socket_shutdown(
    iree_async_socket_t* socket, iree_async_socket_shutdown_mode_t mode) {
  // No sticky failure check here: shutdown is a teardown operation that must
  // execute regardless of prior errors. Callers use shutdown(SHUT_RD) to force
  // pending recv operations to complete during deactivation/drain, so blocking
  // it on a prior send error (e.g., EPIPE) would prevent cleanup.

  iree_socket_t sock = iree_socket_from_primitive(socket->primitive);
  if (sock == IREE_INVALID_SOCKET) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "socket has invalid handle");
  }

  // Map shutdown mode to platform constant.
  int how;
  switch (mode) {
    case IREE_ASYNC_SOCKET_SHUTDOWN_READ:
#if defined(IREE_PLATFORM_WINDOWS)
      how = SD_RECEIVE;
#else
      how = SHUT_RD;
#endif  // IREE_PLATFORM_WINDOWS
      break;
    case IREE_ASYNC_SOCKET_SHUTDOWN_WRITE:
#if defined(IREE_PLATFORM_WINDOWS)
      how = SD_SEND;
#else
      how = SHUT_WR;
#endif  // IREE_PLATFORM_WINDOWS
      break;
    case IREE_ASYNC_SOCKET_SHUTDOWN_BOTH:
#if defined(IREE_PLATFORM_WINDOWS)
      how = SD_BOTH;
#else
      how = SHUT_RDWR;
#endif  // IREE_PLATFORM_WINDOWS
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid shutdown mode %d", mode);
  }

  int result = shutdown(sock, how);
  if (result != 0) {
#if !defined(IREE_PLATFORM_WINDOWS)
    // ENOTCONN is expected for unconnected sockets; map to precondition.
    if (errno == ENOTCONN) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "socket is not connected");
    }
#endif  // !IREE_PLATFORM_WINDOWS
    return iree_status_from_socket_error();
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_socket_query_send_space(
    iree_async_socket_t* socket, iree_host_size_t* out_space) {
  *out_space = IREE_HOST_SIZE_MAX;

  // Check sticky failure state.
  iree_status_t failure = iree_async_socket_query_failure(socket);
  if (!iree_status_is_ok(failure)) {
    return failure;
  }

  iree_socket_t sock = iree_socket_from_primitive(socket->primitive);
  if (sock == IREE_INVALID_SOCKET) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "invalid socket");
  }

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
  // Linux: Use SIOCOUTQ to query unsent bytes in the send queue.
  // This gives us an accurate picture of available send buffer space.
  int unsent = 0;
  if (ioctl(sock, SIOCOUTQ, &unsent) == 0) {
    // Get the send buffer size.
    int sndbuf = 0;
    socklen_t len = sizeof(sndbuf);
    if (getsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, &len) == 0) {
      // SO_SNDBUF returns 2x the actual buffer size on Linux (kernel doubles
      // it), but SIOCOUTQ uses the actual buffer size. Divide by 2 to match.
      int actual_buffer = sndbuf / 2;
      if (actual_buffer > unsent) {
        *out_space = (iree_host_size_t)(actual_buffer - unsent);
      } else {
        *out_space = 0;  // Buffer is full.
      }
      return iree_ok_status();
    }
  }
  // ioctl or getsockopt failed - fall through to unknown.
#else
  // Windows/macOS: No equivalent to SIOCOUTQ.
  // We cannot determine available send space without platform-specific APIs.
  // Returning SO_SNDBUF would be misleading since it's total capacity, not
  // current free space.
  (void)sock;
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID

  // Unknown - caller must use send() return codes for backpressure.
  // out_space is already set to IREE_HOST_SIZE_MAX.
  return iree_ok_status();
}
