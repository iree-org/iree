// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Async socket abstraction.
//
// An iree_async_socket_t wraps a platform socket (TCP, UDP, Unix) and is bound
// to a specific proactor for async I/O operations. Sockets are created via
// iree_async_socket_create() / iree_async_socket_import() and used as targets
// for async accept, connect, recv, and send operations.
//
// Sockets are ref-counted. The proactor retains a reference while operations
// are in flight. The caller's reference controls logical ownership; releasing
// it after all operations complete (or are cancelled) triggers cleanup.
//
// Sticky failure: once a socket encounters an error, it enters a permanently
// failed state. Subsequent operations on a failed socket complete immediately
// with the recorded failure status.

#ifndef IREE_ASYNC_SOCKET_H_
#define IREE_ASYNC_SOCKET_H_

#include "iree/async/address.h"
#include "iree/async/primitive.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_t iree_async_proactor_t;

//===----------------------------------------------------------------------===//
// Socket
//===----------------------------------------------------------------------===//

// The kind of socket to create.
// IPv4 and IPv6 variants are separate types (no dual-stack) to avoid
// platform-specific IPV6_V6ONLY default behavior differences.
enum iree_async_socket_type_e {
  IREE_ASYNC_SOCKET_TYPE_TCP = 0u,
  IREE_ASYNC_SOCKET_TYPE_UDP,
  IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
  IREE_ASYNC_SOCKET_TYPE_UNIX_DGRAM,
  // IPv6 variants use AF_INET6 exclusively.
  IREE_ASYNC_SOCKET_TYPE_TCP6,
  IREE_ASYNC_SOCKET_TYPE_UDP6,
};
typedef uint8_t iree_async_socket_type_t;

// Socket options set at creation time. Immutable after the socket is created.
// The proactor applies these to the underlying platform handle during
// create_socket and may use them to select backend-specific optimizations
// (e.g., enabling kernel zero-copy paths, configuring io_uring features).
//
// For imported sockets (import_socket), options are set to NONE — the caller
// is responsible for pre-configuring the platform handle before import.
//
// Queried via iree_async_socket_query_options() after creation.
enum iree_async_socket_option_bits_e {
  IREE_ASYNC_SOCKET_OPTION_NONE = 0u,

  // SO_REUSEADDR: allows binding to a port that was recently in use.
  // Required for servers that restart without waiting for TIME_WAIT to expire.
  IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR = 1u << 0,

  // SO_REUSEPORT: allows multiple sockets to bind to the same port.
  // The kernel load-balances incoming connections across them. Useful for
  // multi-proactor server architectures where each proactor thread has its own
  // listening socket on the same port.
  IREE_ASYNC_SOCKET_OPTION_REUSE_PORT = 1u << 1,

  // TCP_NODELAY: disables Nagle's algorithm. Sends data immediately without
  // waiting to coalesce small writes. Required for low-latency protocols.
  IREE_ASYNC_SOCKET_OPTION_NO_DELAY = 1u << 2,

  // SO_KEEPALIVE: enables TCP keepalive probes. Detects dead connections
  // without application-level heartbeats. The kernel sends probes after an
  // idle period and closes the connection if the peer is unreachable.
  IREE_ASYNC_SOCKET_OPTION_KEEP_ALIVE = 1u << 3,

  // Hint to use kernel zero-copy send path when available.
  //
  // On Linux, this calls setsockopt(SO_ZEROCOPY) on the socket. The send path
  // then uses zero-copy (SEND_ZC on io_uring 6.0+) or falls back to regular
  // send transparently based on whether the proactor has
  // IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND. On platforms without
  // SO_ZEROCOPY (Windows, macOS), the option is accepted silently and sends
  // always use the regular copy path.
  //
  // Accepted sockets inherit this option from their listening socket. The
  // proactor calls setsockopt(SO_ZEROCOPY) on each accepted fd since the
  // kernel does not inherit this socket option across accept().
  IREE_ASYNC_SOCKET_OPTION_ZERO_COPY = 1u << 4,

  // SO_LINGER with timeout=0: forces immediate RST on close instead of
  // graceful FIN. When set, closing a socket discards any unsent data and
  // sends RST to the peer, avoiding TIME_WAIT state. Useful for:
  // - Aborting connections that are known to be broken
  // - Avoiding TIME_WAIT accumulation in high-churn scenarios
  // - Testing RST handling paths
  IREE_ASYNC_SOCKET_OPTION_LINGER_ZERO = 1u << 5,
};
typedef uint32_t iree_async_socket_options_t;

// Runtime behavior flags stored on the socket.
// Unlike options (which configure the socket via setsockopt at creation and
// are then discarded), flags control proactor behavior for operations on
// this socket.
//
// For create_socket: flags are derived from options (e.g., ZERO_COPY option
// sets ZERO_COPY flag).
// For import_socket: caller declares which flags apply to the imported socket.
enum iree_async_socket_flag_bits_e {
  IREE_ASYNC_SOCKET_FLAG_NONE = 0u,

  // Send operations prefer zero-copy path (SEND_ZC on io_uring 6.0+) when the
  // proactor has IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND; otherwise sends
  // use the regular copy path transparently.
  // For create_socket: set when IREE_ASYNC_SOCKET_OPTION_ZERO_COPY is
  // requested. For import_socket: caller must set this if they configured
  // SO_ZEROCOPY.
  IREE_ASYNC_SOCKET_FLAG_ZERO_COPY = 1u << 0,
};
typedef uint32_t iree_async_socket_flags_t;

// Diagnostic connection state. This tracks the socket's logical lifecycle but
// is not used for protocol logic — operations determine actual behavior.
enum iree_async_socket_state_e {
  IREE_ASYNC_SOCKET_STATE_CREATED = 0u,
  IREE_ASYNC_SOCKET_STATE_CONNECTING,
  IREE_ASYNC_SOCKET_STATE_CONNECTED,
  IREE_ASYNC_SOCKET_STATE_LISTENING,
  IREE_ASYNC_SOCKET_STATE_CLOSING,
  IREE_ASYNC_SOCKET_STATE_CLOSED,
};
typedef uint8_t iree_async_socket_state_t;

// A proactor-managed socket. Created via iree_async_socket_create() or
// iree_async_socket_import().
typedef struct iree_async_socket_t {
  iree_atomic_ref_count_t ref_count;

  // The proactor this socket is bound to. All operations on this socket must
  // be submitted to this proactor. Not retained (proactor outlives sockets).
  iree_async_proactor_t* proactor;

  // Underlying platform handle.
  iree_async_primitive_t primitive;

  // io_uring fixed file index for reduced syscall overhead (-1 if not
  // registered). Backend-specific optimization; ignored on other platforms.
  int32_t fixed_file_index;

  // Socket type (TCP, UDP, Unix stream/dgram).
  iree_async_socket_type_t type;

  // Current diagnostic state.
  iree_async_socket_state_t state;

  // Runtime behavior flags. Immutable after creation.
  // Controls proactor behavior for operations on this socket.
  iree_async_socket_flags_t flags;

  // Sticky failure status. Once set, all operations on this socket fail with
  // this status. Stored as an atomic for thread-safe access.
  iree_atomic_intptr_t failure_status;

  IREE_TRACE(char debug_label[64];)
} iree_async_socket_t;

// Creates a new socket of the given type with the specified options.
//
// The socket is bound to |proactor| and must be used only with it for all async
// operations. The returned socket has one reference; caller must release it
// when done (or submit a close operation, which consumes the ref).
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Options consumption:
//   |options| are applied to the platform handle during creation via setsockopt
//   and are not stored—they are consumed and discarded. The socket does not
//   remember what options were requested.
//
// Options are best-effort:
//   Options that require kernel support (e.g., ZERO_COPY requiring
//   SO_ZEROCOPY) are applied via setsockopt when available. On platforms
//   without the underlying support, the option is accepted silently and the
//   corresponding behavior degrades to the regular path. The ZERO_COPY flag
//   is still recorded on the socket so the send path can attempt zero-copy
//   when the proactor has IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND.
//
// Returns:
//   IREE_STATUS_OK: Socket created successfully.
//   IREE_STATUS_RESOURCE_EXHAUSTED: System socket limit reached.
IREE_API_EXPORT iree_status_t iree_async_socket_create(
    iree_async_proactor_t* proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket);

// Imports an existing platform handle as a proactor-managed socket.
//
// Use this to bring externally-created sockets (e.g., from accept() or
// inherited from a parent process) under proactor management.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Ownership:
//   The proactor takes logical ownership of the handle and will close it when
//   the socket is released or a close operation completes. The caller must not
//   close the handle after a successful import.
//
// No mutation:
//   The proactor does NOT modify the socket (no setsockopt, no fcntl). The
//   caller is responsible for pre-configuring nonblocking mode, socket options,
//   etc. before import. This allows importing sockets with non-default
//   configurations that the proactor doesn't know about.
//
// Flags parameter:
//   The |flags| parameter declares runtime behavior for the imported socket.
//   For zero-copy: caller must have set SO_ZEROCOPY via setsockopt before
//   import and pass IREE_ASYNC_SOCKET_FLAG_ZERO_COPY in |flags|.
//
// Platform handles:
//   - POSIX: int fd (socket file descriptor)
//   - Windows: SOCKET (cast to uintptr_t in primitive.handle)
//
// Returns:
//   IREE_STATUS_OK: Socket imported successfully.
//   IREE_STATUS_INVALID_ARGUMENT: Invalid handle or type mismatch.
IREE_API_EXPORT iree_status_t iree_async_socket_import(
    iree_async_proactor_t* proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket);

// Increments the reference count.
IREE_API_EXPORT void iree_async_socket_retain(iree_async_socket_t* socket);

// Decrements the reference count and destroys if it reaches zero.
IREE_API_EXPORT void iree_async_socket_release(iree_async_socket_t* socket);

// Returns the socket's current diagnostic state.
static inline iree_async_socket_state_t iree_async_socket_query_state(
    const iree_async_socket_t* socket) {
  return socket->state;
}

// Returns the sticky failure status, or iree_ok_status() if not failed.
// The returned status is NOT owned by the caller (peek semantics).
static inline iree_status_t iree_async_socket_query_failure(
    const iree_async_socket_t* socket) {
  intptr_t value =
      iree_atomic_load((iree_atomic_intptr_t*)&socket->failure_status,
                       iree_memory_order_acquire);
  return (iree_status_t)value;
}

// Sets the socket's sticky failure status if not already failed.
// Uses atomic compare-exchange so the first error wins: if the socket already
// has a failure, |status| is discarded and the original failure preserved.
//
// Only the status code is stored (no storage allocation). The full error
// context is delivered to the caller via the operation's completion callback;
// the sticky failure is a persistent "is this socket still usable?" indicator
// that supports peek semantics in iree_async_socket_query_failure().
static inline void iree_async_socket_set_failure(iree_async_socket_t* socket,
                                                 iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  intptr_t failure_code =
      (intptr_t)iree_status_from_code(iree_status_code(status));
  intptr_t expected = (intptr_t)iree_ok_status();
  iree_atomic_compare_exchange_strong(&socket->failure_status, &expected,
                                      failure_code, iree_memory_order_acq_rel,
                                      iree_memory_order_acquire);
}

//===----------------------------------------------------------------------===//
// Socket lifecycle (synchronous, pre-async-I/O setup)
//===----------------------------------------------------------------------===//

// Synchronous configuration calls that prepare a socket for async I/O.
// These operate directly on the platform handle and do not involve the
// proactor's event loop. They are called once during socket setup — the socket
// is then ready for async operations (accept, connect, recv, send).
//
// Server setup:
//   iree_async_socket_create(proactor, TCP, options, &socket);
//   iree_async_socket_bind(socket, &address);
//   iree_async_socket_listen(socket, backlog);
//   // Submit multishot accept operations via the proactor.
//
// Client setup:
//   iree_async_socket_create(proactor, TCP, options, &socket);
//   // Submit async connect operation via the proactor.

// Binds the socket to a local address. Must be called before listen.
// Returns IREE_STATUS_ALREADY_EXISTS if the address is in use (and
// REUSE_ADDR/REUSE_PORT were not enabled at creation time).
IREE_API_EXPORT iree_status_t iree_async_socket_bind(
    iree_async_socket_t* socket, const iree_async_address_t* address);

// Puts the socket into listening state with the given backlog queue depth.
// Must be called after bind. The socket's diagnostic state transitions to
// LISTENING. |backlog| is a hint for the kernel's accept queue size; 0 uses
// the system default (typically 128 on Linux).
IREE_API_EXPORT iree_status_t
iree_async_socket_listen(iree_async_socket_t* socket, iree_host_size_t backlog);

// Queries the local address the socket is bound to (getsockname).
// Useful after binding to port 0 (ephemeral) to discover the assigned port.
// Returns IREE_STATUS_FAILED_PRECONDITION if the socket is not yet bound.
IREE_API_EXPORT iree_status_t iree_async_socket_query_local_address(
    const iree_async_socket_t* socket, iree_async_address_t* out_address);

// Shutdown mode for iree_async_socket_shutdown().
enum iree_async_socket_shutdown_mode_e {
  // Disables further receive operations. Data already in the receive buffer
  // can still be read. The peer's subsequent sends may fail with
  // EPIPE/ECONNRESET.
  IREE_ASYNC_SOCKET_SHUTDOWN_READ = 0,

  // Disables further send operations and sends FIN to the peer (half-close).
  // The peer's recv will return 0 (EOF) after reading any buffered data.
  // The socket can still receive data until the peer also closes.
  IREE_ASYNC_SOCKET_SHUTDOWN_WRITE = 1,

  // Disables both send and receive. Equivalent to calling shutdown with both
  // modes, but the socket remains valid for querying state (unlike close).
  IREE_ASYNC_SOCKET_SHUTDOWN_BOTH = 2,
};
typedef uint8_t iree_async_socket_shutdown_mode_t;

// Shuts down part or all of the socket's communication.
// This is a synchronous operation that affects the underlying socket
// immediately. The socket remains valid after shutdown; use release to destroy
// it.
//
// Shutdown modes:
//   SHUTDOWN_READ: Peer's sends will fail; local recv returns 0 or error.
//   SHUTDOWN_WRITE: Sends FIN; peer recv returns 0 (EOF). Local can still recv.
//   SHUTDOWN_BOTH: Disables both directions; socket still valid for queries.
//
// Returns IREE_STATUS_FAILED_PRECONDITION if the socket is not connected.
IREE_API_EXPORT iree_status_t iree_async_socket_shutdown(
    iree_async_socket_t* socket, iree_async_socket_shutdown_mode_t mode);

// Queries available send buffer space for backpressure tracking.
//
// Returns an estimate of how many bytes can be sent without blocking or hitting
// flow control. This is useful for implementing application-level backpressure.
//
// Platform behavior:
//   - Linux: Uses ioctl(SIOCOUTQ) to query unsent bytes in the send queue,
//     then returns SO_SNDBUF - unsent. This is accurate but queries current
//     state (not a guarantee of future capacity).
//   - macOS/Windows: No equivalent to SIOCOUTQ exists. Returns
//   IREE_HOST_SIZE_MAX
//     (unknown). Callers must rely on send() return codes (EAGAIN/EWOULDBLOCK)
//     for backpressure.
//
// On success, |out_space| is set to:
//   - Estimated available send space in bytes on Linux.
//   - IREE_HOST_SIZE_MAX if the query is not supported on this platform.
//   - 0 if the send buffer is currently full.
//
// Returns an error if the socket is in a failed state.
IREE_API_EXPORT iree_status_t iree_async_socket_query_send_space(
    iree_async_socket_t* socket, iree_host_size_t* out_space);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_SOCKET_H_
