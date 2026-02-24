// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/iocp/socket.h"

#include "iree/async/platform/iocp/proactor.h"

#if defined(IREE_PLATFORM_WINDOWS)

// Windows headers — winsock2.h must precede windows.h to avoid conflicts.
// clang-format off
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h>
#include <windows.h>
// clang-format on

#include <string.h>

//===----------------------------------------------------------------------===//
// Socket type mapping
//===----------------------------------------------------------------------===//

static int iree_async_iocp_socket_type_to_domain(
    iree_async_socket_type_t type) {
  switch (type) {
    case IREE_ASYNC_SOCKET_TYPE_TCP:
    case IREE_ASYNC_SOCKET_TYPE_UDP:
      return AF_INET;
    case IREE_ASYNC_SOCKET_TYPE_TCP6:
    case IREE_ASYNC_SOCKET_TYPE_UDP6:
      return AF_INET6;
    case IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM:
    case IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM:
      return AF_UNIX;
    default:
      return -1;
  }
}

static int iree_async_iocp_socket_type_to_socktype(
    iree_async_socket_type_t type) {
  switch (type) {
    case IREE_ASYNC_SOCKET_TYPE_TCP:
    case IREE_ASYNC_SOCKET_TYPE_TCP6:
    case IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM:
      return SOCK_STREAM;
    case IREE_ASYNC_SOCKET_TYPE_UDP:
    case IREE_ASYNC_SOCKET_TYPE_UDP6:
    case IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM:
      return SOCK_DGRAM;
    default:
      return -1;
  }
}

static int iree_async_iocp_socket_type_to_protocol(
    iree_async_socket_type_t type) {
  switch (type) {
    case IREE_ASYNC_SOCKET_TYPE_TCP:
    case IREE_ASYNC_SOCKET_TYPE_TCP6:
      return IPPROTO_TCP;
    case IREE_ASYNC_SOCKET_TYPE_UDP:
    case IREE_ASYNC_SOCKET_TYPE_UDP6:
      return IPPROTO_UDP;
    case IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM:
    case IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM:
      return 0;
    default:
      return -1;
  }
}

//===----------------------------------------------------------------------===//
// Socket options
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_iocp_socket_apply_options(
    SOCKET sock, iree_async_socket_type_t type,
    iree_async_socket_options_t options) {
  int optval = 1;

  if (iree_any_bit_set(options, IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR)) {
    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&optval,
                   sizeof(optval)) == SOCKET_ERROR) {
      int wsa_error = WSAGetLastError();
      return iree_make_status(iree_status_code_from_win32_error(wsa_error),
                              "setsockopt SO_REUSEADDR failed (WSA error %d)",
                              wsa_error);
    }
  }

  if (iree_any_bit_set(options, IREE_ASYNC_SOCKET_OPTION_REUSE_PORT)) {
    // Windows has no equivalent to SO_REUSEPORT. SO_REUSEADDR on Windows has
    // different semantics than on POSIX (allows hijacking by default), so we
    // cannot safely emulate this.
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "SO_REUSEPORT is not available on Windows");
  }

  if (iree_any_bit_set(options, IREE_ASYNC_SOCKET_OPTION_NO_DELAY) &&
      (type == IREE_ASYNC_SOCKET_TYPE_TCP ||
       type == IREE_ASYNC_SOCKET_TYPE_TCP6)) {
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&optval,
                   sizeof(optval)) == SOCKET_ERROR) {
      int wsa_error = WSAGetLastError();
      return iree_make_status(iree_status_code_from_win32_error(wsa_error),
                              "setsockopt TCP_NODELAY failed (WSA error %d)",
                              wsa_error);
    }
  }

  if (iree_any_bit_set(options, IREE_ASYNC_SOCKET_OPTION_KEEP_ALIVE)) {
    if (setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (const char*)&optval,
                   sizeof(optval)) == SOCKET_ERROR) {
      int wsa_error = WSAGetLastError();
      return iree_make_status(iree_status_code_from_win32_error(wsa_error),
                              "setsockopt SO_KEEPALIVE failed (WSA error %d)",
                              wsa_error);
    }
  }

  // ZERO_COPY: Windows has no SO_ZEROCOPY equivalent. On POSIX, the setsockopt
  // succeeds and sets the socket flag; the send path then uses zero-copy
  // (SEND_ZC) or regular send transparently based on proactor capability. Here
  // we skip the setsockopt (nothing to set) but still allow the option — the
  // flag is recorded on the socket via flags_from_options, and sends use
  // regular WSASend (the copy path) transparently.

  if (iree_any_bit_set(options, IREE_ASYNC_SOCKET_OPTION_LINGER_ZERO)) {
    struct linger linger_opt;
    linger_opt.l_onoff = 1;
    linger_opt.l_linger = 0;
    if (setsockopt(sock, SOL_SOCKET, SO_LINGER, (const char*)&linger_opt,
                   sizeof(linger_opt)) == SOCKET_ERROR) {
      int wsa_error = WSAGetLastError();
      return iree_make_status(iree_status_code_from_win32_error(wsa_error),
                              "setsockopt SO_LINGER failed (WSA error %d)",
                              wsa_error);
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Socket struct initialization
//===----------------------------------------------------------------------===//

static void iree_async_iocp_socket_initialize(
    iree_async_socket_t* socket, iree_async_proactor_iocp_t* proactor,
    SOCKET sock, iree_async_socket_type_t type,
    iree_async_socket_flags_t flags) {
  iree_atomic_ref_count_init(&socket->ref_count);
  socket->proactor = &proactor->base;
  socket->primitive = iree_async_primitive_from_win32_handle((uintptr_t)sock);
  socket->fixed_file_index = -1;
  socket->type = type;
  socket->state = IREE_ASYNC_SOCKET_STATE_CREATED;
  socket->flags = flags;
  iree_atomic_store(&socket->failure_status, (intptr_t)iree_ok_status(),
                    iree_memory_order_release);
  IREE_TRACE({
    snprintf(socket->debug_label, sizeof(socket->debug_label), "socket:%llu",
             (unsigned long long)sock);
  });
}

static iree_async_socket_flags_t iree_async_iocp_socket_flags_from_options(
    iree_async_socket_options_t options) {
  iree_async_socket_flags_t flags = IREE_ASYNC_SOCKET_FLAG_NONE;
  if (iree_any_bit_set(options, IREE_ASYNC_SOCKET_OPTION_ZERO_COPY)) {
    flags |= IREE_ASYNC_SOCKET_FLAG_ZERO_COPY;
  }
  return flags;
}

//===----------------------------------------------------------------------===//
// IOCP association
//===----------------------------------------------------------------------===//

// Associates a socket with the proactor's IOCP completion port. All future
// overlapped I/O completions on this socket will be delivered to this port.
static iree_status_t iree_async_iocp_socket_associate(
    iree_async_proactor_iocp_t* proactor, SOCKET sock) {
  HANDLE result = CreateIoCompletionPort(
      (HANDLE)sock, (HANDLE)proactor->completion_port, /*CompletionKey=*/0,
      /*NumberOfConcurrentThreads=*/0);
  if (result == NULL) {
    DWORD error = GetLastError();
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "CreateIoCompletionPort for socket failed "
                            "(error %lu)",
                            (unsigned long)error);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// WSA extension function loading
//===----------------------------------------------------------------------===//

// Loads AcceptEx, ConnectEx, and GetAcceptExSockaddrs function pointers via
// WSAIoctl. These are not directly linked — they must be loaded per-process
// using well-known GUIDs from mswsock.h. The function pointers are the same
// for all socket families, so we load them once using any socket.
static iree_status_t iree_async_iocp_load_wsa_extensions(
    iree_async_proactor_iocp_t* proactor, SOCKET sock) {
  if (proactor->wsa_extensions.loaded) return iree_ok_status();

  DWORD bytes_returned = 0;

  // Load AcceptEx.
  GUID accept_ex_guid = WSAID_ACCEPTEX;
  int result = WSAIoctl(
      sock, SIO_GET_EXTENSION_FUNCTION_POINTER, &accept_ex_guid,
      sizeof(accept_ex_guid), &proactor->wsa_extensions.AcceptEx,
      sizeof(proactor->wsa_extensions.AcceptEx), &bytes_returned, NULL, NULL);
  if (result == SOCKET_ERROR) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "WSAIoctl SIO_GET_EXTENSION_FUNCTION_POINTER for AcceptEx failed "
        "(WSA error %d)",
        WSAGetLastError());
  }

  // Load ConnectEx.
  GUID connect_ex_guid = WSAID_CONNECTEX;
  result = WSAIoctl(
      sock, SIO_GET_EXTENSION_FUNCTION_POINTER, &connect_ex_guid,
      sizeof(connect_ex_guid), &proactor->wsa_extensions.ConnectEx,
      sizeof(proactor->wsa_extensions.ConnectEx), &bytes_returned, NULL, NULL);
  if (result == SOCKET_ERROR) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "WSAIoctl SIO_GET_EXTENSION_FUNCTION_POINTER for ConnectEx failed "
        "(WSA error %d)",
        WSAGetLastError());
  }

  // Load GetAcceptExSockaddrs.
  GUID get_acceptex_sockaddrs_guid = WSAID_GETACCEPTEXSOCKADDRS;
  result = WSAIoctl(sock, SIO_GET_EXTENSION_FUNCTION_POINTER,
                    &get_acceptex_sockaddrs_guid,
                    sizeof(get_acceptex_sockaddrs_guid),
                    &proactor->wsa_extensions.GetAcceptExSockaddrs,
                    sizeof(proactor->wsa_extensions.GetAcceptExSockaddrs),
                    &bytes_returned, NULL, NULL);
  if (result == SOCKET_ERROR) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "WSAIoctl SIO_GET_EXTENSION_FUNCTION_POINTER for "
                            "GetAcceptExSockaddrs failed (WSA error %d)",
                            WSAGetLastError());
  }

  proactor->wsa_extensions.loaded = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_async_iocp_socket_create(
    iree_async_proactor_iocp_t* proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_socket);
  *out_socket = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  int domain = iree_async_iocp_socket_type_to_domain(type);
  int socktype = iree_async_iocp_socket_type_to_socktype(type);
  int protocol = iree_async_iocp_socket_type_to_protocol(type);
  if (domain < 0 || socktype < 0 || protocol < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid socket type %d", type);
  }

  // WSA_FLAG_OVERLAPPED is required for all IOCP overlapped I/O.
  SOCKET sock =
      WSASocketW(domain, socktype, protocol, NULL, 0, WSA_FLAG_OVERLAPPED);
  if (sock == INVALID_SOCKET) {
    int wsa_error = WSAGetLastError();
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_win32_error(wsa_error),
                            "WSASocket failed (WSA error %d)", wsa_error);
  }

  iree_status_t status =
      iree_async_iocp_socket_apply_options(sock, type, options);

  // Associate with IOCP completion port.
  if (iree_status_is_ok(status)) {
    status = iree_async_iocp_socket_associate(proactor, sock);
  }

  // Load WSA extension function pointers on first socket creation.
  if (iree_status_is_ok(status)) {
    status = iree_async_iocp_load_wsa_extensions(proactor, sock);
  }

  iree_async_socket_t* socket = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(proactor->base.allocator, sizeof(*socket),
                                   (void**)&socket);
  }

  if (iree_status_is_ok(status)) {
    memset(socket, 0, sizeof(*socket));
    iree_async_socket_flags_t flags =
        iree_async_iocp_socket_flags_from_options(options);
    iree_async_iocp_socket_initialize(socket, proactor, sock, type, flags);
    *out_socket = socket;
  } else {
    closesocket(sock);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_async_iocp_socket_import(
    iree_async_proactor_iocp_t* proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_socket);
  *out_socket = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected WIN32_HANDLE primitive for socket import");
  }

  SOCKET sock = (SOCKET)primitive.value.win32_handle;
  if (sock == INVALID_SOCKET) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid socket");
  }

  // Associate with IOCP completion port.
  iree_status_t status = iree_async_iocp_socket_associate(proactor, sock);

  // Load WSA extension function pointers if not already loaded.
  if (iree_status_is_ok(status)) {
    status = iree_async_iocp_load_wsa_extensions(proactor, sock);
  }

  iree_async_socket_t* socket = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(proactor->base.allocator, sizeof(*socket),
                                   (void**)&socket);
  }

  if (iree_status_is_ok(status)) {
    memset(socket, 0, sizeof(*socket));
    iree_async_iocp_socket_initialize(socket, proactor, sock, type, flags);
    *out_socket = socket;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_async_iocp_socket_destroy(iree_async_proactor_iocp_t* proactor,
                                    iree_async_socket_t* socket) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(socket);
  IREE_TRACE_ZONE_BEGIN(z0);

  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;
  if (sock != INVALID_SOCKET) {
    closesocket(sock);
  }

  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &socket->failure_status, iree_memory_order_acquire);
  iree_status_ignore(failure);

  iree_allocator_free(proactor->base.allocator, socket);

  IREE_TRACE_ZONE_END(z0);
}

#endif  // IREE_PLATFORM_WINDOWS
