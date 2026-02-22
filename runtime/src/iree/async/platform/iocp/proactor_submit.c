// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Submit path for the IOCP proactor.
//
// This module handles operation submission for all operation types. Per-type
// submit helpers issue overlapped I/O calls (WSARecv, WSASend, AcceptEx,
// ConnectEx, ReadFile, WriteFile) or route non-I/O operations through the
// pending_queue for poll-thread processing. The main submit() function builds
// LINKED chains and dispatches chain heads through submit_operation().

#include <string.h>

#include "iree/async/buffer_pool.h"
#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/notification.h"
#include "iree/async/operation.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/message.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/platform/iocp/proactor.h"
#include "iree/async/proactor.h"
#include "iree/async/semaphore.h"
#include "iree/async/span.h"
#include "iree/async/util/message_pool.h"
#include "iree/async/util/sequence_emulation.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/memory.h"

#if defined(IREE_PLATFORM_WINDOWS)

// Windows headers — winsock2.h must precede windows.h to avoid conflicts.
// clang-format off
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h>
#include <windows.h>
// clang-format on

//===----------------------------------------------------------------------===//
// LINKED chain dispatch helpers
//===----------------------------------------------------------------------===//

// Forward declaration for submit_operation (used by submit_continuation_chain).
static iree_status_t iree_async_proactor_iocp_submit_operation(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* operation);

// Cancels a continuation chain by directly invoking callbacks with CANCELLED.
// Cancelled continuations were never submitted, so no resources were retained.
// Returns the number of callbacks invoked.
iree_host_size_t iree_async_proactor_iocp_cancel_continuation_chain(
    iree_async_operation_t* chain_head) {
  if (!chain_head) return 0;
  iree_host_size_t cancelled_count = 0;
  iree_async_operation_t* operation = chain_head;
  while (operation) {
    iree_async_operation_t* next = operation->linked_next;
    operation->linked_next = NULL;
    if (operation->completion_fn) {
      operation->completion_fn(operation->user_data, operation,
                               iree_status_from_code(IREE_STATUS_CANCELLED),
                               IREE_ASYNC_COMPLETION_FLAG_NONE);
      ++cancelled_count;
    }
    operation = next;
  }
  return cancelled_count;
}

// Submits a continuation chain head. On submit failure, fires the chain head's
// callback with the error and cancels remaining continuations.
void iree_async_proactor_iocp_submit_continuation_chain(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* chain_head) {
  if (!chain_head) return;

  iree_status_t status =
      iree_async_proactor_iocp_submit_operation(proactor, chain_head);
  if (iree_status_is_ok(status)) return;

  // Submit failed: fire the failed operation's callback and cancel the rest.
  iree_async_operation_t* rest = chain_head->linked_next;
  chain_head->linked_next = NULL;
  if (chain_head->completion_fn) {
    chain_head->completion_fn(chain_head->user_data, chain_head, status,
                              IREE_ASYNC_COMPLETION_FLAG_NONE);
  } else {
    iree_status_ignore(status);
  }
  iree_async_proactor_iocp_cancel_continuation_chain(rest);
}

// Dispatches a linked_next continuation chain based on the trigger's status.
// On success: submits the chain for execution.
// On failure: cancels the chain by directly invoking callbacks with CANCELLED.
// Returns the number of directly-invoked callbacks (for completion counting).
iree_host_size_t iree_async_proactor_iocp_dispatch_linked_continuation(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* operation,
    iree_status_t trigger_status) {
  iree_async_operation_t* continuation = operation->linked_next;
  if (!continuation) return 0;

  // Detach the chain before potentially recursive submit.
  operation->linked_next = NULL;

  if (iree_status_is_ok(trigger_status)) {
    iree_async_proactor_iocp_submit_continuation_chain(proactor, continuation);
    return 0;  // Submitted ops produce completions counted by the drain loop.
  } else {
    return iree_async_proactor_iocp_cancel_continuation_chain(continuation);
  }
}

//===----------------------------------------------------------------------===//
// Carrier allocation and submit failure helpers
//===----------------------------------------------------------------------===//

// Allocates and initializes a carrier for socket I/O. The carrier is zeroed
// and configured with the given type, completion port, operation, and socket
// handle. The caller fills in the type-specific data union members.
static iree_status_t iree_async_proactor_iocp_allocate_carrier(
    iree_async_proactor_iocp_t* proactor,
    iree_async_iocp_carrier_type_t carrier_type,
    iree_async_operation_t* operation, uintptr_t io_handle,
    iree_async_iocp_carrier_t** out_carrier) {
  iree_async_iocp_carrier_t* carrier = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      proactor->base.allocator, sizeof(*carrier), (void**)&carrier));
  memset(carrier, 0, sizeof(*carrier));
  carrier->type = carrier_type;
  carrier->completion_port = proactor->completion_port;
  carrier->operation = operation;
  carrier->io_handle = io_handle;
  iree_atomic_fetch_add(&proactor->outstanding_carrier_count, 1,
                        iree_memory_order_relaxed);
  *out_carrier = carrier;
  return iree_ok_status();
}

// Releases a carrier that was allocated but never successfully posted to IOCP.
// Decrements the outstanding_carrier_count and frees the carrier memory.
// This is the symmetric counterpart to allocate_carrier for error paths where
// the overlapped I/O call fails synchronously (error != *_IO_PENDING).
static void iree_async_proactor_iocp_release_carrier(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier) {
  iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                        iree_memory_order_relaxed);
  iree_allocator_free(proactor->base.allocator, carrier);
}

// Builds a WSABUF array from a span list. Returns the number of buffers.
static DWORD iree_async_proactor_iocp_build_wsabuf(
    WSABUF* wsabuf, iree_async_span_list_t buffers) {
  DWORD count =
      (DWORD)(buffers.count > IREE_ASYNC_IOCP_MAX_SCATTER_GATHER_BUFFERS
                  ? IREE_ASYNC_IOCP_MAX_SCATTER_GATHER_BUFFERS
                  : buffers.count);
  for (DWORD i = 0; i < count; ++i) {
    wsabuf[i].buf = (char*)iree_async_span_ptr(buffers.values[i]);
    wsabuf[i].len = (ULONG)buffers.values[i].length;
  }
  return count;
}

// Posts a synthetic failure completion for an operation whose overlapped I/O
// call failed synchronously (error != ERROR_IO_PENDING / WSA_IO_PENDING). The
// proactor API contract requires submit() to succeed and deliver errors through
// the completion callback on the poll thread. This posts the operation as a
// "direct completion" with the Win32 error code encoded in
// dwNumberOfBytesTransferred for the poll thread to convert to iree_status_t.
static void iree_async_proactor_iocp_post_submit_failure(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* operation,
    int error_code) {
  PostQueuedCompletionStatus((HANDLE)proactor->completion_port,
                             (DWORD)error_code, (ULONG_PTR)operation, NULL);
}

// Posts a direct completion with a pre-computed iree_status_t. The status is
// stashed in operation->next (available as scratch for non-carrier operations)
// and retrieved by the poll thread via the sentinel in bytes_transferred.
static void iree_async_proactor_iocp_post_stashed_status(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* operation,
    iree_status_t op_status) {
  operation->next = (iree_async_operation_t*)(uintptr_t)op_status;
  PostQueuedCompletionStatus((HANDLE)proactor->completion_port,
                             IREE_ASYNC_IOCP_STASHED_STATUS_SENTINEL,
                             (ULONG_PTR)operation, NULL);
}

//===----------------------------------------------------------------------===//
// Timer and event wait submit
//===----------------------------------------------------------------------===//

// Submits a timer operation. All timers go through the pending_queue
// regardless of whether the deadline has already passed. The drain inserts
// into the timer list, and the timer expiry scan completes already-expired
// entries. This avoids PostQueuedCompletionStatus, which would require a
// GQCS round-trip and prevent same-poll-iteration completion when timers
// are submitted from callbacks.
static iree_status_t iree_async_proactor_iocp_submit_timer(
    iree_async_proactor_iocp_t* proactor,
    iree_async_timer_operation_t* timer_operation) {
  iree_async_proactor_iocp_push_pending(proactor, &timer_operation->base);
  return iree_ok_status();
}

// Submits an EVENT_WAIT by deferring to the poll thread. push_pending retains
// the event reference; release_operation_resources releases it on completion.
static iree_status_t iree_async_proactor_iocp_submit_event_wait(
    iree_async_proactor_iocp_t* proactor,
    iree_async_event_wait_operation_t* event_wait) {
  iree_async_proactor_iocp_push_pending(proactor, &event_wait->base);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Socket submit helpers
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_iocp_submit_socket_accept(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_accept_operation_t* accept_op) {
  iree_async_socket_t* listen_socket = accept_op->listen_socket;
  SOCKET listen_sock = (SOCKET)listen_socket->primitive.value.win32_handle;

  // Retain the listen socket before issuing overlapped I/O.
  iree_async_operation_retain_resources(&accept_op->base);

  // AcceptEx requires a pre-created accept socket of the same family/type.
  // MSDN only documents AcceptEx for AF_INET/AF_INET6. AF_UNIX support is
  // undocumented but works on Windows 10 1809+ where the AF_UNIX Winsock
  // provider routes AcceptEx through its internal accept path.
  int domain = AF_INET;
  int socktype = SOCK_STREAM;
  int protocol = IPPROTO_TCP;
  switch (listen_socket->type) {
    case IREE_ASYNC_SOCKET_TYPE_TCP6:
      domain = AF_INET6;
      protocol = IPPROTO_TCP;
      break;
    case IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM:
      domain = AF_UNIX;
      protocol = 0;
      break;
    default:
      break;
  }

  SOCKET accept_sock =
      WSASocketW(domain, socktype, protocol, NULL, 0, WSA_FLAG_OVERLAPPED);
  if (accept_sock == INVALID_SOCKET) {
    int wsa_error = WSAGetLastError();
    iree_async_operation_release_resources(&accept_op->base);
    return iree_make_status(iree_status_code_from_win32_error(wsa_error),
                            "WSASocket for accept failed (WSA error %d)",
                            wsa_error);
  }

  // Associate accept socket with IOCP port.
  HANDLE result = CreateIoCompletionPort(
      (HANDLE)accept_sock, (HANDLE)proactor->completion_port, 0, 0);
  if (result == NULL) {
    DWORD error = GetLastError();
    closesocket(accept_sock);
    iree_async_operation_release_resources(&accept_op->base);
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "CreateIoCompletionPort for accept socket failed "
                            "(error %lu)",
                            (unsigned long)error);
  }

  // Allocate carrier.
  iree_async_iocp_carrier_t* carrier = NULL;
  iree_status_t status = iree_async_proactor_iocp_allocate_carrier(
      proactor, IREE_ASYNC_IOCP_CARRIER_ACCEPT, &accept_op->base,
      (uintptr_t)listen_sock, &carrier);
  if (!iree_status_is_ok(status)) {
    closesocket(accept_sock);
    iree_async_operation_release_resources(&accept_op->base);
    return status;
  }

  carrier->data.accept.accept_socket = accept_sock;
  // Each address slot needs the maximum address size plus 16 bytes of
  // padding (AcceptEx requirement: at least 16 bytes more than the maximum
  // address length for the transport protocol). sockaddr_storage covers all
  // address families including AF_UNIX (sockaddr_un is 110 bytes on Windows).
  carrier->data.accept.local_address_length =
      (DWORD)(sizeof(struct sockaddr_storage) + 16);
  carrier->data.accept.remote_address_length =
      (DWORD)(sizeof(struct sockaddr_storage) + 16);

  // Issue AcceptEx. dwReceiveDataLength = 0 means accept-only (no initial
  // data reception).
  DWORD bytes_received = 0;
  BOOL accepted = proactor->wsa_extensions.AcceptEx(
      listen_sock, accept_sock, carrier->data.accept.address_buffer,
      /*dwReceiveDataLength=*/0, carrier->data.accept.local_address_length,
      carrier->data.accept.remote_address_length, &bytes_received,
      &carrier->overlapped);

  if (!accepted) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      closesocket(accept_sock);
      iree_async_proactor_iocp_release_carrier(proactor, carrier);
      iree_async_proactor_iocp_post_submit_failure(proactor, &accept_op->base,
                                                   wsa_error);
      return iree_ok_status();
    }
    // WSA_IO_PENDING: I/O is pending, completion will arrive via GQCS.
  }

  // Store carrier backpointer in operation->next for cancellation.
  accept_op->base.next = (iree_async_operation_t*)carrier;
  return iree_ok_status();
}

// Binds an unbound socket to INADDR_ANY:0 (or in6addr_any:0 for IPv6).
// ConnectEx requires the socket to be bound before it can be called, and UDP
// connect also needs a bound socket. This is invisible to the caller.
//
// Detection: getsockname() returns WSAEINVAL on an unbound Windows socket.
// If getsockname() succeeds, the socket is already bound and no action is
// needed. This is more reliable than checking socket->state because
// iree_async_socket_bind() does not update the state field.
static iree_status_t iree_async_proactor_iocp_auto_bind_if_needed(
    iree_async_socket_t* socket, SOCKET sock) {
  struct sockaddr_storage probe_address;
  int probe_length = sizeof(probe_address);
  if (getsockname(sock, (struct sockaddr*)&probe_address, &probe_length) !=
      SOCKET_ERROR) {
    return iree_ok_status();  // Already bound.
  }

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
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "auto-bind not supported for socket type %d",
                              socket->type);
  }

  if (bind(sock, (struct sockaddr*)&bind_addr, bind_addr_length) ==
      SOCKET_ERROR) {
    int wsa_error = WSAGetLastError();
    return iree_make_status(iree_status_code_from_win32_error(wsa_error),
                            "auto-bind failed (WSA error %d)", wsa_error);
  }
  return iree_ok_status();
}

// Returns true if the socket type is connectionless (UDP/UDP6).
static bool iree_async_socket_type_is_datagram(iree_async_socket_type_t type) {
  return type == IREE_ASYNC_SOCKET_TYPE_UDP ||
         type == IREE_ASYNC_SOCKET_TYPE_UDP6;
}

static iree_status_t iree_async_proactor_iocp_submit_socket_connect(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_connect_operation_t* connect_op) {
  iree_async_socket_t* socket = connect_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;

  // Auto-bind if needed (ConnectEx requires a bound socket; UDP connect also
  // needs it when the socket hasn't been explicitly bound yet).
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_iocp_auto_bind_if_needed(socket, sock));

  // UDP connect is synchronous: it just sets the default destination address
  // in the socket. ConnectEx is TCP-only (connection-oriented). For UDP, use
  // plain connect() and post a direct completion.
  if (iree_async_socket_type_is_datagram(socket->type)) {
    const struct sockaddr* target_addr =
        (const struct sockaddr*)connect_op->address.storage;
    int addr_length = (int)connect_op->address.length;

    iree_async_operation_retain_resources(&connect_op->base);

    int result = connect(sock, target_addr, addr_length);
    if (result == SOCKET_ERROR) {
      int wsa_error = WSAGetLastError();
      iree_async_proactor_iocp_post_submit_failure(proactor, &connect_op->base,
                                                   wsa_error);
      return iree_ok_status();
    }

    socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTED;

    // Post a direct completion (NULL overlapped, operation as CompletionKey,
    // bytes=0 for success).
    PostQueuedCompletionStatus((HANDLE)proactor->completion_port, 0,
                               (ULONG_PTR)&connect_op->base, NULL);
    return iree_ok_status();
  }

  // TCP: use ConnectEx for overlapped (asynchronous) connection.
  // MSDN only documents ConnectEx for AF_INET/AF_INET6. AF_UNIX stream sockets
  // would reach this path but are currently blocked by auto_bind_if_needed
  // (which rejects non-TCP/UDP types). If AF_UNIX connect support is added,
  // note that ConnectEx with AF_UNIX is undocumented but appears to work on
  // Windows 10 1809+ via the AF_UNIX Winsock provider.
  iree_async_operation_retain_resources(&connect_op->base);

  iree_async_iocp_carrier_t* carrier = NULL;
  iree_status_t status = iree_async_proactor_iocp_allocate_carrier(
      proactor, IREE_ASYNC_IOCP_CARRIER_CONNECT, &connect_op->base,
      (uintptr_t)sock, &carrier);
  if (!iree_status_is_ok(status)) {
    iree_async_operation_release_resources(&connect_op->base);
    return status;
  }

  const struct sockaddr* target_addr =
      (const struct sockaddr*)connect_op->address.storage;
  int addr_length = (int)connect_op->address.length;

  BOOL connected = proactor->wsa_extensions.ConnectEx(
      sock, target_addr, addr_length, NULL, 0, NULL, &carrier->overlapped);

  if (!connected) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      iree_async_proactor_iocp_release_carrier(proactor, carrier);
      iree_async_proactor_iocp_post_submit_failure(proactor, &connect_op->base,
                                                   wsa_error);
      return iree_ok_status();
    }
  }

  socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTING;
  connect_op->base.next = (iree_async_operation_t*)carrier;
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_iocp_submit_socket_recv(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_recv_operation_t* recv_op) {
  iree_async_socket_t* socket = recv_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;

  iree_async_operation_retain_resources(&recv_op->base);

  iree_async_iocp_carrier_t* carrier = NULL;
  iree_status_t status = iree_async_proactor_iocp_allocate_carrier(
      proactor, IREE_ASYNC_IOCP_CARRIER_SOCKET_IO, &recv_op->base,
      (uintptr_t)sock, &carrier);
  if (!iree_status_is_ok(status)) {
    iree_async_operation_release_resources(&recv_op->base);
    return status;
  }

  carrier->data.socket_io.buffer_count = iree_async_proactor_iocp_build_wsabuf(
      carrier->data.socket_io.wsabuf, recv_op->buffers);
  carrier->data.socket_io.flags = 0;

  int result =
      WSARecv(sock, carrier->data.socket_io.wsabuf,
              carrier->data.socket_io.buffer_count, NULL,
              &carrier->data.socket_io.flags, &carrier->overlapped, NULL);
  if (result == SOCKET_ERROR) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      iree_async_proactor_iocp_release_carrier(proactor, carrier);
      iree_async_proactor_iocp_post_submit_failure(proactor, &recv_op->base,
                                                   wsa_error);
      return iree_ok_status();
    }
  }

  recv_op->base.next = (iree_async_operation_t*)carrier;
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_iocp_submit_socket_send(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_send_operation_t* send_op) {
  iree_async_socket_t* socket = send_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;

  iree_async_operation_retain_resources(&send_op->base);

  iree_async_iocp_carrier_t* carrier = NULL;
  iree_status_t status = iree_async_proactor_iocp_allocate_carrier(
      proactor, IREE_ASYNC_IOCP_CARRIER_SOCKET_IO, &send_op->base,
      (uintptr_t)sock, &carrier);
  if (!iree_status_is_ok(status)) {
    iree_async_operation_release_resources(&send_op->base);
    return status;
  }

  carrier->data.socket_io.buffer_count = iree_async_proactor_iocp_build_wsabuf(
      carrier->data.socket_io.wsabuf, send_op->buffers);
  // MSG_MORE: silently ignored on Windows (no equivalent; TCP_NODELAY controls
  // coalescing at the socket level).
  DWORD flags = 0;

  int result = WSASend(sock, carrier->data.socket_io.wsabuf,
                       carrier->data.socket_io.buffer_count, NULL, flags,
                       &carrier->overlapped, NULL);
  if (result == SOCKET_ERROR) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      iree_async_proactor_iocp_release_carrier(proactor, carrier);
      iree_async_proactor_iocp_post_submit_failure(proactor, &send_op->base,
                                                   wsa_error);
      return iree_ok_status();
    }
  }

  send_op->base.next = (iree_async_operation_t*)carrier;
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_iocp_submit_socket_sendto(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_sendto_operation_t* sendto_op) {
  iree_async_socket_t* socket = sendto_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;

  iree_async_operation_retain_resources(&sendto_op->base);

  iree_async_iocp_carrier_t* carrier = NULL;
  iree_status_t status = iree_async_proactor_iocp_allocate_carrier(
      proactor, IREE_ASYNC_IOCP_CARRIER_SOCKET_IO, &sendto_op->base,
      (uintptr_t)sock, &carrier);
  if (!iree_status_is_ok(status)) {
    iree_async_operation_release_resources(&sendto_op->base);
    return status;
  }

  carrier->data.socket_io.buffer_count = iree_async_proactor_iocp_build_wsabuf(
      carrier->data.socket_io.wsabuf, sendto_op->buffers);

  const struct sockaddr* dest_addr =
      (const struct sockaddr*)sendto_op->destination.storage;
  int dest_length = (int)sendto_op->destination.length;

  int result = WSASendTo(sock, carrier->data.socket_io.wsabuf,
                         carrier->data.socket_io.buffer_count, NULL, 0,
                         dest_addr, dest_length, &carrier->overlapped, NULL);
  if (result == SOCKET_ERROR) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      iree_async_proactor_iocp_release_carrier(proactor, carrier);
      iree_async_proactor_iocp_post_submit_failure(proactor, &sendto_op->base,
                                                   wsa_error);
      return iree_ok_status();
    }
  }

  sendto_op->base.next = (iree_async_operation_t*)carrier;
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_iocp_submit_socket_recvfrom(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_recvfrom_operation_t* recvfrom_op) {
  iree_async_socket_t* socket = recvfrom_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;

  iree_async_operation_retain_resources(&recvfrom_op->base);

  iree_async_iocp_carrier_t* carrier = NULL;
  iree_status_t status = iree_async_proactor_iocp_allocate_carrier(
      proactor, IREE_ASYNC_IOCP_CARRIER_SOCKET_IO, &recvfrom_op->base,
      (uintptr_t)sock, &carrier);
  if (!iree_status_is_ok(status)) {
    iree_async_operation_release_resources(&recvfrom_op->base);
    return status;
  }

  carrier->data.socket_io.buffer_count = iree_async_proactor_iocp_build_wsabuf(
      carrier->data.socket_io.wsabuf, recvfrom_op->buffers);
  carrier->data.socket_io.flags = 0;

  // WSARecvFrom writes the actual sender address length asynchronously at
  // completion time. The length must be stored in the carrier (heap-allocated,
  // persists until completion), not a stack local.
  carrier->data.socket_io.sender_address_length =
      (int)sizeof(recvfrom_op->sender.storage);

  int result = WSARecvFrom(sock, carrier->data.socket_io.wsabuf,
                           carrier->data.socket_io.buffer_count, NULL,
                           &carrier->data.socket_io.flags,
                           (struct sockaddr*)recvfrom_op->sender.storage,
                           &carrier->data.socket_io.sender_address_length,
                           &carrier->overlapped, NULL);
  if (result == SOCKET_ERROR) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      iree_async_proactor_iocp_release_carrier(proactor, carrier);
      iree_async_proactor_iocp_post_submit_failure(proactor, &recvfrom_op->base,
                                                   wsa_error);
      return iree_ok_status();
    }
  }

  recvfrom_op->base.next = (iree_async_operation_t*)carrier;
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_iocp_submit_socket_recv_pool(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_recv_pool_operation_t* recv_pool_op) {
  iree_async_socket_t* socket = recv_pool_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;

  // Validate the buffer pool's region before proceeding. RECV_POOL writes
  // received data into pool buffers, so the region must have WRITE access.
  iree_async_region_t* region =
      iree_async_buffer_pool_region(recv_pool_op->pool);
  if (IREE_UNLIKELY(!iree_any_bit_set(region->access_flags,
                                      IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SOCKET_RECV_POOL requires a buffer pool with WRITE access; "
        "this pool was registered with read-only access");
  }
  if (IREE_UNLIKELY(region->proactor != &proactor->base)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SOCKET_RECV_POOL buffer pool was registered with a different "
        "proactor");
  }

  recv_pool_op->bytes_received = 0;
  memset(&recv_pool_op->lease, 0, sizeof(recv_pool_op->lease));

  // Acquire buffer from pool before allocating carrier.
  iree_status_t status =
      iree_async_buffer_pool_acquire(recv_pool_op->pool, &recv_pool_op->lease);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  iree_async_operation_retain_resources(&recv_pool_op->base);

  iree_async_iocp_carrier_t* carrier = NULL;
  status = iree_async_proactor_iocp_allocate_carrier(
      proactor, IREE_ASYNC_IOCP_CARRIER_RECV_POOL, &recv_pool_op->base,
      (uintptr_t)sock, &carrier);
  if (!iree_status_is_ok(status)) {
    iree_async_buffer_lease_release(&recv_pool_op->lease);
    iree_async_operation_release_resources(&recv_pool_op->base);
    return status;
  }

  carrier->data.recv_pool.wsabuf.buf =
      (char*)iree_async_span_ptr(recv_pool_op->lease.span);
  carrier->data.recv_pool.wsabuf.len = (ULONG)recv_pool_op->lease.span.length;
  carrier->data.recv_pool.flags = 0;

  int result =
      WSARecv(sock, &carrier->data.recv_pool.wsabuf, 1, NULL,
              &carrier->data.recv_pool.flags, &carrier->overlapped, NULL);
  if (result == SOCKET_ERROR) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      // Release pool lease here because the direct completion path has no
      // carrier-type-specific handling to do it.
      iree_async_buffer_lease_release(&recv_pool_op->lease);
      iree_async_proactor_iocp_release_carrier(proactor, carrier);
      iree_async_proactor_iocp_post_submit_failure(
          proactor, &recv_pool_op->base, wsa_error);
      return iree_ok_status();
    }
  }

  recv_pool_op->base.next = (iree_async_operation_t*)carrier;
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_iocp_submit_socket_close(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_close_operation_t* close_op) {
  iree_async_socket_t* socket = close_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;

  // Close is synchronous, no carrier needed. Close the underlying socket,
  // then post an immediate completion.
  if (sock != INVALID_SOCKET) {
    closesocket(sock);
  }
  // Invalidate the socket primitive so destroy doesn't double-close.
  socket->primitive = iree_async_primitive_none();
  socket->state = IREE_ASYNC_SOCKET_STATE_CLOSED;

  // Post immediate completion. Resources are not retained (close consumes
  // the caller's reference via release_operation_resources).
  PostQueuedCompletionStatus((HANDLE)proactor->completion_port, 0,
                             (ULONG_PTR)&close_op->base, NULL);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Semaphore operations
//===----------------------------------------------------------------------===//

void iree_async_proactor_iocp_semaphore_wait_enqueue_completion(
    iree_async_iocp_semaphore_wait_tracker_t* tracker) {
  // Guard against double-enqueue: success callbacks (remaining_or_satisfied),
  // error callbacks, and cancel all independently decide to enqueue. Only the
  // first to set the flag actually pushes.
  int32_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(&tracker->enqueued, &expected, 1,
                                           iree_memory_order_acq_rel,
                                           iree_memory_order_relaxed)) {
    return;
  }
  iree_atomic_slist_push(&tracker->proactor->pending_semaphore_waits,
                         &tracker->slist_entry);
  iree_async_proactor_iocp_wake(&tracker->proactor->base);
}

// Timepoint callback for SEMAPHORE_WAIT operations.
// Called under the semaphore's internal lock — must be fast and non-blocking.
// Decodes the tracker and index from user_data, then tracks ALL/ANY
// satisfaction. When the wait is complete, pushes the tracker to the
// pending_semaphore_waits MPSC slist for the poll thread to drain.
static void iree_async_proactor_iocp_semaphore_wait_timepoint_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  uintptr_t encoded = (uintptr_t)user_data;
  iree_async_iocp_semaphore_wait_tracker_t* tracker =
      (iree_async_iocp_semaphore_wait_tracker_t*)(encoded &
                                                  0x00FFFFFFFFFFFFFFull);
  iree_host_size_t index = (iree_host_size_t)(encoded >> 56);

  if (!iree_status_is_ok(status)) {
    // Failure or cancellation. Store the error status (first one wins).
    intptr_t expected = (intptr_t)iree_ok_status();
    if (!iree_atomic_compare_exchange_strong(
            &tracker->completion_status, &expected, (intptr_t)status,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      iree_status_ignore(status);
    }
    // Enqueue for completion regardless of whether we won the status race.
    iree_async_proactor_iocp_semaphore_wait_enqueue_completion(tracker);
    return;
  }

  if (tracker->operation->mode == IREE_ASYNC_WAIT_MODE_ANY) {
    // ANY mode: first satisfied index wins via CAS.
    int32_t expected = -1;
    if (iree_atomic_compare_exchange_strong(
            &tracker->remaining_or_satisfied, &expected, (int32_t)index,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      iree_async_proactor_iocp_semaphore_wait_enqueue_completion(tracker);
    }
  } else {
    // ALL mode: decrement remaining count.
    int32_t remaining = iree_atomic_fetch_sub(&tracker->remaining_or_satisfied,
                                              1, iree_memory_order_acq_rel) -
                        1;
    if (remaining == 0) {
      iree_async_proactor_iocp_semaphore_wait_enqueue_completion(tracker);
    }
  }
}

static iree_status_t iree_async_proactor_iocp_submit_semaphore_signal(
    iree_async_proactor_iocp_t* proactor,
    iree_async_semaphore_signal_operation_t* signal_op) {
  // Signal all semaphores synchronously. On first error, break.
  iree_status_t op_status = iree_ok_status();
  for (iree_host_size_t i = 0; i < signal_op->count; ++i) {
    iree_status_t status = iree_async_semaphore_signal(
        signal_op->semaphores[i], signal_op->values[i], signal_op->frontier);
    if (!iree_status_is_ok(status)) {
      op_status = status;
      break;
    }
  }

  // Retain resources and post direct completion for poll-thread dispatch.
  iree_async_operation_retain_resources(&signal_op->base);

  if (iree_status_is_ok(op_status)) {
    PostQueuedCompletionStatus((HANDLE)proactor->completion_port, 0,
                               (ULONG_PTR)&signal_op->base, NULL);
  } else {
    // Stash the error status for the poll thread. Cannot use bytes_transferred
    // encoding (WSA error code) because the status from semaphore_signal
    // carries a rich message that would be lost in the conversion.
    iree_async_proactor_iocp_post_stashed_status(proactor, &signal_op->base,
                                                 op_status);
  }
  return iree_ok_status();
}

// Submits a SEMAPHORE_WAIT by checking for immediate satisfaction, then
// allocating a tracker with embedded timepoints and registering callbacks
// on each semaphore. The timepoint callbacks fire from the signaling thread
// and push the tracker to the pending_semaphore_waits MPSC slist, which is
// drained by the poll thread.
static iree_status_t iree_async_proactor_iocp_submit_semaphore_wait(
    iree_async_proactor_iocp_t* proactor,
    iree_async_semaphore_wait_operation_t* wait_op) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // The timepoint callback encodes {tracker_pointer, semaphore_index} in a
  // single pointer using a 56-bit/8-bit split. The 8-bit index field limits
  // the maximum number of semaphores per wait to 255.
  if (IREE_UNLIKELY(wait_op->count > 255)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "semaphore wait count exceeds the maximum of 255 "
                            "(limited by timepoint user_data encoding)");
  }

  // Check for immediate satisfaction before allocating a tracker.
  bool immediately_satisfied = false;
  bool all_satisfied = true;
  for (iree_host_size_t i = 0; i < wait_op->count; ++i) {
    uint64_t current = iree_async_semaphore_query(wait_op->semaphores[i]);
    if (current >= wait_op->values[i]) {
      if (wait_op->mode == IREE_ASYNC_WAIT_MODE_ANY) {
        wait_op->satisfied_index = i;
        immediately_satisfied = true;
        break;
      }
    } else {
      all_satisfied = false;
    }
  }
  if (!immediately_satisfied && all_satisfied) {
    immediately_satisfied = true;
  }
  if (immediately_satisfied) {
    IREE_TRACE_ZONE_END(z0);
    // Retain, post direct completion for poll-thread dispatch.
    iree_async_operation_retain_resources(&wait_op->base);
    PostQueuedCompletionStatus((HANDLE)proactor->completion_port, 0,
                               (ULONG_PTR)&wait_op->base, NULL);
    return iree_ok_status();
  }

  // Calculate allocation size with overflow checking.
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_async_iocp_semaphore_wait_tracker_t), &total_size,
              IREE_STRUCT_FIELD_FAM(wait_op->count,
                                    iree_async_semaphore_timepoint_t)));

  // Allocate tracker with embedded timepoints.
  iree_async_iocp_semaphore_wait_tracker_t* tracker = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(proactor->base.allocator, total_size,
                                (void**)&tracker));
  memset(tracker, 0, total_size);

  tracker->operation = wait_op;
  tracker->proactor = proactor;
  tracker->allocator = proactor->base.allocator;
  tracker->count = wait_op->count;
  tracker->registered_count = 0;
  iree_atomic_store(&tracker->completion_status, (intptr_t)iree_ok_status(),
                    iree_memory_order_release);
  iree_atomic_store(&tracker->enqueued, 0, iree_memory_order_release);

  // Transfer LINKED continuation chain from operation to tracker.
  tracker->continuation_head = wait_op->base.linked_next;
  wait_op->base.linked_next = NULL;

  if (wait_op->mode == IREE_ASYNC_WAIT_MODE_ALL) {
    iree_atomic_store(&tracker->remaining_or_satisfied, (int32_t)wait_op->count,
                      iree_memory_order_release);
  } else {
    // ANY mode: -1 indicates not yet satisfied.
    iree_atomic_store(&tracker->remaining_or_satisfied, -1,
                      iree_memory_order_release);
  }

  // Store tracker in operation for cancel to find it.
  wait_op->base.next = (iree_async_operation_t*)tracker;

  // Register timepoints for each semaphore. Use status chaining so that
  // on failure we cancel only the successfully registered timepoints.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < wait_op->count && iree_status_is_ok(status);
       ++i) {
    iree_async_semaphore_timepoint_t* timepoint = &tracker->timepoints[i];
    timepoint->callback =
        iree_async_proactor_iocp_semaphore_wait_timepoint_callback;
    // Encode tracker pointer + index in user_data using a 56-bit/8-bit split
    // for LA57 (5-level paging) safety: userspace pointers use at most 56 bits
    // on x86-64, leaving 8 bits for the index (max 255).
    // The count <= 255 check above guarantees this shift is safe.
    timepoint->user_data = (void*)((uintptr_t)tracker | ((uintptr_t)i << 56));
    status = iree_async_semaphore_acquire_timepoint(
        wait_op->semaphores[i], wait_op->values[i], timepoint);
    if (iree_status_is_ok(status)) {
      tracker->registered_count = i + 1;
    }
  }

  // Single exit point: on failure, cancel registered timepoints and free.
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < tracker->registered_count; ++i) {
      iree_async_semaphore_cancel_timepoint(wait_op->semaphores[i],
                                            &tracker->timepoints[i]);
    }
    wait_op->base.next = NULL;
    iree_allocator_free(tracker->allocator, tracker);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Notification operation submit
//===----------------------------------------------------------------------===//

// NOTIFICATION_WAIT: captured epoch token, routed through pending_queue so
// the poll thread can either complete immediately (epoch already advanced)
// or link into the notification's pending_waits list.
static iree_status_t iree_async_proactor_iocp_submit_notification_wait(
    iree_async_proactor_iocp_t* proactor,
    iree_async_notification_wait_operation_t* wait_op) {
  // Capture epoch token at submit time. The poll thread completes the
  // operation when the epoch advances past this value.
  wait_op->wait_token = (uint32_t)iree_atomic_load(
      &wait_op->notification->epoch, iree_memory_order_acquire);
  iree_async_proactor_iocp_push_pending(proactor, &wait_op->base);
  return iree_ok_status();
}

// NOTIFICATION_SIGNAL: signal synchronously and post direct completion.
static iree_status_t iree_async_proactor_iocp_submit_notification_signal(
    iree_async_proactor_iocp_t* proactor,
    iree_async_notification_signal_operation_t* signal_op) {
  // Perform the signal synchronously.
  iree_async_notification_signal(signal_op->notification,
                                 signal_op->wake_count);
  // woken_count is not precisely available from the Windows API, so report
  // the requested count.
  signal_op->woken_count = signal_op->wake_count;

  // Retain and post direct completion for poll-thread dispatch.
  iree_async_operation_retain_resources(&signal_op->base);
  PostQueuedCompletionStatus((HANDLE)proactor->completion_port, 0,
                             (ULONG_PTR)&signal_op->base, NULL);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Message operation submit
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_iocp_submit_message(
    iree_async_proactor_iocp_t* proactor,
    iree_async_message_operation_t* message) {
  iree_async_proactor_t* target = message->target;
  if (!target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MESSAGE target proactor is NULL");
  }
  if (target->vtable != &iree_async_proactor_iocp_vtable) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "MESSAGE target must be an IOCP proactor from the same backend; "
        "cross-backend messaging is not supported via operations");
  }
  iree_async_proactor_iocp_t* target_iocp =
      iree_async_proactor_iocp_cast(target);

  // Deliver message to the target proactor's pool and wake its poll thread.
  iree_status_t send_status = iree_async_message_pool_send(
      &target_iocp->message_pool, message->message_data);
  if (iree_status_is_ok(send_status)) {
    target->vtable->wake(target);
  }

  // Handle source-side completion.
  if (iree_any_bit_set(message->message_flags,
                       IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION)) {
    // Fire-and-forget: no source completion callback. Dispatch linked
    // continuation directly with the send status.
    iree_async_proactor_iocp_dispatch_linked_continuation(
        proactor, &message->base, send_status);
    iree_status_ignore(send_status);
    return iree_ok_status();
  }

  // Post source completion through the completion port for poll-thread
  // dispatch (consistent with all other operation types).
  iree_async_operation_retain_resources(&message->base);
  if (iree_status_is_ok(send_status)) {
    PostQueuedCompletionStatus((HANDLE)proactor->completion_port, 0,
                               (ULONG_PTR)&message->base, NULL);
  } else {
    iree_async_proactor_iocp_post_stashed_status(proactor, &message->base,
                                                 send_status);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// File I/O submit handlers
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_iocp_submit_file_open(
    iree_async_proactor_iocp_t* proactor,
    iree_async_file_open_operation_t* open_op) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Translate open flags to Windows CreateFileW parameters.
  DWORD desired_access = 0;
  DWORD share_mode = FILE_SHARE_READ;
  DWORD creation_disposition = OPEN_EXISTING;
  DWORD flags_and_attributes = FILE_FLAG_OVERLAPPED;

  if (iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_READ)) {
    desired_access |= GENERIC_READ;
  }
  if (iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_WRITE)) {
    desired_access |= GENERIC_WRITE;
    share_mode = 0;  // Exclusive write access.
  }
  if (iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_APPEND)) {
    desired_access |= FILE_APPEND_DATA;
    share_mode = 0;
  }

  // Determine creation disposition based on flags.
  bool create =
      iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_CREATE);
  bool truncate =
      iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_TRUNCATE);
  if (create && truncate) {
    creation_disposition = CREATE_ALWAYS;
  } else if (create) {
    creation_disposition = OPEN_ALWAYS;
  } else if (truncate) {
    creation_disposition = TRUNCATE_EXISTING;
  } else {
    creation_disposition = OPEN_EXISTING;
  }

  if (iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_DIRECT)) {
    flags_and_attributes |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH;
  }

  // Convert path to wide string. Two-pass MultiByteToWideChar: first call
  // determines the required buffer size, second call performs the conversion.
  // This avoids MAX_PATH (260 char) limitation and supports long paths.
  int wide_length = MultiByteToWideChar(CP_UTF8, 0, open_op->path, -1, NULL, 0);
  if (wide_length == 0) {
    DWORD error = GetLastError();
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "failed to determine wide string length for path "
                            "(error %lu)",
                            (unsigned long)error);
  }
  WCHAR* wide_path = (WCHAR*)iree_alloca(wide_length * sizeof(WCHAR));
  int converted_length = MultiByteToWideChar(CP_UTF8, 0, open_op->path, -1,
                                             wide_path, wide_length);
  if (converted_length == 0) {
    DWORD error = GetLastError();
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "failed to convert path to wide string (error %lu)",
                            (unsigned long)error);
  }

  HANDLE file_handle =
      CreateFileW(wide_path, desired_access, share_mode, NULL,
                  creation_disposition, flags_and_attributes, NULL);
  if (file_handle == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    IREE_TRACE_ZONE_END(z0);
    // Post failure as direct completion for poll-thread delivery.
    iree_async_proactor_iocp_post_submit_failure(proactor, &open_op->base,
                                                 (int)error);
    return iree_ok_status();
  }

  // Import the file handle (associates with IOCP port).
  iree_async_primitive_t primitive =
      iree_async_primitive_from_win32_handle((uintptr_t)file_handle);
  iree_status_t status =
      iree_async_file_import(&proactor->base, primitive, &open_op->opened_file);
  if (!iree_status_is_ok(status)) {
    CloseHandle(file_handle);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Post immediate completion (open is synchronous).
  PostQueuedCompletionStatus((HANDLE)proactor->completion_port, 0,
                             (ULONG_PTR)&open_op->base, NULL);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_iocp_submit_file_read(
    iree_async_proactor_iocp_t* proactor,
    iree_async_file_read_operation_t* read_op) {
  iree_async_file_t* file = read_op->file;
  HANDLE file_handle = (HANDLE)file->primitive.value.win32_handle;

  iree_async_operation_retain_resources(&read_op->base);

  iree_async_iocp_carrier_t* carrier = NULL;
  iree_status_t status = iree_async_proactor_iocp_allocate_carrier(
      proactor, IREE_ASYNC_IOCP_CARRIER_FILE_IO, &read_op->base,
      (uintptr_t)file_handle, &carrier);
  if (!iree_status_is_ok(status)) {
    iree_async_operation_release_resources(&read_op->base);
    return status;
  }

  // Encode file offset in the OVERLAPPED structure.
  carrier->overlapped.Offset = (DWORD)(read_op->offset & 0xFFFFFFFF);
  carrier->overlapped.OffsetHigh = (DWORD)(read_op->offset >> 32);

  void* buffer_ptr = iree_async_span_ptr(read_op->buffer);
  DWORD buffer_length = (DWORD)read_op->buffer.length;

  BOOL read_ok = ReadFile(file_handle, buffer_ptr, buffer_length, NULL,
                          &carrier->overlapped);
  if (!read_ok) {
    DWORD error = GetLastError();
    if (error != ERROR_IO_PENDING) {
      iree_async_proactor_iocp_release_carrier(proactor, carrier);
      iree_async_proactor_iocp_post_submit_failure(proactor, &read_op->base,
                                                   (int)error);
      return iree_ok_status();
    }
  }

  read_op->base.next = (iree_async_operation_t*)carrier;
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_iocp_submit_file_write(
    iree_async_proactor_iocp_t* proactor,
    iree_async_file_write_operation_t* write_op) {
  iree_async_file_t* file = write_op->file;
  HANDLE file_handle = (HANDLE)file->primitive.value.win32_handle;

  iree_async_operation_retain_resources(&write_op->base);

  iree_async_iocp_carrier_t* carrier = NULL;
  iree_status_t status = iree_async_proactor_iocp_allocate_carrier(
      proactor, IREE_ASYNC_IOCP_CARRIER_FILE_IO, &write_op->base,
      (uintptr_t)file_handle, &carrier);
  if (!iree_status_is_ok(status)) {
    iree_async_operation_release_resources(&write_op->base);
    return status;
  }

  // Encode file offset in the OVERLAPPED structure.
  // For APPEND mode, the caller opened the file with FILE_APPEND_DATA.
  // Windows handles append semantics at the kernel level — the offset
  // is still specified but the kernel atomically appends.
  carrier->overlapped.Offset = (DWORD)(write_op->offset & 0xFFFFFFFF);
  carrier->overlapped.OffsetHigh = (DWORD)(write_op->offset >> 32);

  const void* buffer_ptr = iree_async_span_ptr(write_op->buffer);
  DWORD buffer_length = (DWORD)write_op->buffer.length;

  BOOL write_ok = WriteFile(file_handle, buffer_ptr, buffer_length, NULL,
                            &carrier->overlapped);
  if (!write_ok) {
    DWORD error = GetLastError();
    if (error != ERROR_IO_PENDING) {
      iree_async_proactor_iocp_release_carrier(proactor, carrier);
      iree_async_proactor_iocp_post_submit_failure(proactor, &write_op->base,
                                                   (int)error);
      return iree_ok_status();
    }
  }

  write_op->base.next = (iree_async_operation_t*)carrier;
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_iocp_submit_file_close(
    iree_async_proactor_iocp_t* proactor,
    iree_async_file_close_operation_t* close_op) {
  iree_async_file_t* file = close_op->file;
  HANDLE file_handle = (HANDLE)file->primitive.value.win32_handle;

  // Close is synchronous, no carrier needed.
  BOOL close_ok = TRUE;
  if (file_handle != NULL && file_handle != INVALID_HANDLE_VALUE) {
    close_ok = CloseHandle(file_handle);
  }
  // Invalidate the handle to prevent double-close in destroy.
  file->primitive.value.win32_handle = 0;

  // Post completion. The caller's file reference is consumed by
  // release_operation_resources during completion dispatch (no prior retain).
  if (!close_ok) {
    DWORD error = GetLastError();
    iree_async_proactor_iocp_post_submit_failure(proactor, &close_op->base,
                                                 (int)error);
    return iree_ok_status();
  }

  PostQueuedCompletionStatus((HANDLE)proactor->completion_port, 0,
                             (ULONG_PTR)&close_op->base, NULL);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Submit dispatcher
//===----------------------------------------------------------------------===//

// Dispatches a single operation through the per-type submit handlers.
static iree_status_t iree_async_proactor_iocp_submit_operation(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_NOP:
      // NOP: route through pending_queue for inline completion during drain.
      // PostQueuedCompletionStatus would require a GQCS round-trip, which
      // prevents same-poll-iteration completion when NOPs are submitted from
      // callbacks (e.g., sequence emulation step advancement).
      iree_async_proactor_iocp_push_pending(proactor, operation);
      return iree_ok_status();

    case IREE_ASYNC_OPERATION_TYPE_TIMER:
      return iree_async_proactor_iocp_submit_timer(
          proactor, (iree_async_timer_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT:
      return iree_async_proactor_iocp_submit_event_wait(
          proactor, (iree_async_event_wait_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SEQUENCE: {
      iree_async_sequence_operation_t* sequence =
          (iree_async_sequence_operation_t*)operation;
      if (!sequence->step_fn) {
        return iree_async_sequence_submit_as_linked(&proactor->base, sequence);
      } else {
        return iree_async_sequence_emulation_begin(&proactor->sequence_emulator,
                                                   sequence);
      }
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
      return iree_async_proactor_iocp_submit_socket_accept(
          proactor, (iree_async_socket_accept_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
      return iree_async_proactor_iocp_submit_socket_connect(
          proactor, (iree_async_socket_connect_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
      return iree_async_proactor_iocp_submit_socket_recv(
          proactor, (iree_async_socket_recv_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
      return iree_async_proactor_iocp_submit_socket_send(
          proactor, (iree_async_socket_send_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
      return iree_async_proactor_iocp_submit_socket_sendto(
          proactor, (iree_async_socket_sendto_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      return iree_async_proactor_iocp_submit_socket_recvfrom(
          proactor, (iree_async_socket_recvfrom_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      return iree_async_proactor_iocp_submit_socket_recv_pool(
          proactor, (iree_async_socket_recv_pool_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL:
      return iree_async_proactor_iocp_submit_semaphore_signal(
          proactor, (iree_async_semaphore_signal_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT:
      return iree_async_proactor_iocp_submit_semaphore_wait(
          proactor, (iree_async_semaphore_wait_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT:
      return iree_async_proactor_iocp_submit_notification_wait(
          proactor, (iree_async_notification_wait_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL:
      return iree_async_proactor_iocp_submit_notification_signal(
          proactor, (iree_async_notification_signal_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_MESSAGE:
      return iree_async_proactor_iocp_submit_message(
          proactor, (iree_async_message_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE:
      return iree_async_proactor_iocp_submit_socket_close(
          proactor, (iree_async_socket_close_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
      return iree_async_proactor_iocp_submit_file_open(
          proactor, (iree_async_file_open_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_FILE_READ:
      return iree_async_proactor_iocp_submit_file_read(
          proactor, (iree_async_file_read_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE:
      return iree_async_proactor_iocp_submit_file_write(
          proactor, (iree_async_file_write_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE:
      return iree_async_proactor_iocp_submit_file_close(
          proactor, (iree_async_file_close_operation_t*)operation);

    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "IOCP proactor: operation type %d not yet implemented",
          (int)operation->type);
  }
}

//===----------------------------------------------------------------------===//
// Submit
//===----------------------------------------------------------------------===//

iree_status_t iree_async_proactor_iocp_submit(
    iree_async_proactor_t* base_proactor,
    iree_async_operation_list_t operations) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);

  if (iree_atomic_load(&proactor->shutdown_requested,
                       iree_memory_order_acquire)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_ABORTED, "proactor is shutting down");
  }

  // Build linked_next chains from LINKED flags and validate.
  // Operations with LINKED flag point to the next operation in the batch.
  // Only "chain heads" (operations not preceded by a LINKED operation) are
  // submitted to their backends; continuations stay in linked_next and are
  // dispatched when the predecessor completes.
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    iree_async_operation_t* operation = operations.values[i];
    operation->linked_next = NULL;
    if (!iree_any_bit_set(operation->flags, IREE_ASYNC_OPERATION_FLAG_LINKED)) {
      continue;
    }
    // LINKED on last operation is a contract violation.
    if (i + 1 >= operations.count) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LINKED flag set on last operation in batch (no successor)");
    }
    operation->linked_next = operations.values[i + 1];
  }

  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    iree_async_operation_t* operation = operations.values[i];

    // Skip continuation operations — they are held in the predecessor's
    // linked_next and will be submitted when it completes.
    if (i > 0 && iree_any_bit_set(operations.values[i - 1]->flags,
                                  IREE_ASYNC_OPERATION_FLAG_LINKED)) {
      continue;
    }

    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_async_proactor_iocp_submit_operation(proactor, operation));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_WINDOWS
