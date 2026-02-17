// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Network operations for async socket I/O.
//
// All operations in this file require a proactor-managed socket obtained via
// iree_async_socket_create() or iree_async_socket_import().
//
// Availability (all operations):
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Performance characteristics vary by backend and operation; see individual
// operation documentation for multishot availability and zero-copy paths.

#ifndef IREE_ASYNC_OPERATIONS_NET_H_
#define IREE_ASYNC_OPERATIONS_NET_H_

#include "iree/async/buffer_pool.h"
#include "iree/async/operation.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Scatter-gather constants
//===----------------------------------------------------------------------===//

// Maximum number of scatter-gather buffers supported in send/recv operations.
// This limit applies to both IREE's span list and the underlying platform
// vectored I/O (struct iovec on POSIX, WSABUF on Windows).
#define IREE_ASYNC_SOCKET_SCATTER_GATHER_MAX_BUFFERS 8

// Platform storage sizing for scatter-gather operations.
// These are generous upper bounds that work across all supported platforms.
// struct iovec: 16 bytes on 64-bit (void* iov_base + size_t iov_len)
// struct msghdr: ~56 bytes on Linux 64-bit, we round up to 64 for alignment
#define IREE_ASYNC_SOCKET_PLATFORM_IOVEC_SIZE 16
#define IREE_ASYNC_SOCKET_PLATFORM_MSGHDR_SIZE 64

// Computed storage sizes for platform structs.
#define IREE_ASYNC_SOCKET_PLATFORM_IOVEC_STORAGE  \
  (IREE_ASYNC_SOCKET_SCATTER_GATHER_MAX_BUFFERS * \
   IREE_ASYNC_SOCKET_PLATFORM_IOVEC_SIZE)

//===----------------------------------------------------------------------===//
// Accept
//===----------------------------------------------------------------------===//

// Accepts an incoming connection on a listening socket.
//
// On success, |accepted_socket| is set to the new connection (caller must
// release it). The |peer_address| is populated with the remote endpoint.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Multishot availability:
//   generic | io_uring | IOCP | kqueue
//   emul    | 5.19+    | emul | emul
//
// Multishot mode (IREE_ASYNC_OPERATION_FLAG_MULTISHOT):
//   When enabled, the operation delivers repeated completions for each new
//   connection until explicitly cancelled. Each completion sets a new
//   |accepted_socket| and |peer_address|. The callback receives
//   IREE_ASYNC_COMPLETION_FLAG_MORE for all but the final completion.
//
//   io_uring (5.19+): Uses IORING_OP_ACCEPT with IORING_ACCEPT_MULTISHOT
//     for kernel-side accept loop with minimal syscall overhead.
//   Other backends: Emulated by resubmitting after each accept.
//
// Threading model:
//   Callback fires on the poll thread. The accepted socket is ready for
//   immediate use (already associated with the proactor).
//
// Example:
//   iree_async_socket_accept_operation_t accept_op = {0};
//   accept_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT;
//   accept_op.base.flags = IREE_ASYNC_OPERATION_FLAG_MULTISHOT;
//   accept_op.base.completion_fn = on_accept;
//   accept_op.listen_socket = server_socket;
//   iree_async_proactor_submit_one(proactor, &accept_op.base);
typedef struct iree_async_socket_accept_operation_t {
  iree_async_operation_t base;

  // The listening socket to accept from.
  iree_async_socket_t* listen_socket;

  // Result: the newly accepted socket (new reference, caller must release).
  iree_async_socket_t* accepted_socket;

  // Result: peer address of the accepted connection.
  iree_async_address_t peer_address;
} iree_async_socket_accept_operation_t;

//===----------------------------------------------------------------------===//
// Connect
//===----------------------------------------------------------------------===//

// Initiates an outbound connection to a remote address.
//
// On success, the socket transitions to CONNECTED state and is ready for
// send/recv operations. On failure, the socket remains in CREATED state
// and may be reused for another connect attempt.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Threading model:
//   Callback fires on the poll thread when the TCP handshake completes
//   (or fails). The socket should not be used for I/O until the connect
//   callback fires with success.
//
// Returns (via callback status):
//   IREE_STATUS_OK: Connection established.
//   IREE_STATUS_UNAVAILABLE: Host unreachable or connection refused.
//   IREE_STATUS_DEADLINE_EXCEEDED: Connection timeout (TCP level).
typedef struct iree_async_socket_connect_operation_t {
  iree_async_operation_t base;

  // The socket to connect (must be in CREATED state).
  iree_async_socket_t* socket;

  // Target address.
  iree_async_address_t address;
} iree_async_socket_connect_operation_t;

//===----------------------------------------------------------------------===//
// Recv
//===----------------------------------------------------------------------===//

// Receives data from a connected socket into one or more buffers (scatter).
//
// On success, |bytes_received| indicates how many bytes were written total.
// A zero |bytes_received| with OK status indicates graceful connection close
// (EOF).
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Multishot availability:
//   generic | io_uring | IOCP | kqueue
//   emul    | 5.19+    | emul | emul
//
// Scatter-gather semantics:
//   For multiple buffers, data fills buffers in order: the first buffer fills
//   completely before any data goes to the second, etc. This is standard
//   vectored I/O (readv/recvmsg) semantics. Maximum scatter count is
//   IREE_ASYNC_SOCKET_RECV_MAX_BUFFERS.
//
// Multishot mode (IREE_ASYNC_OPERATION_FLAG_MULTISHOT):
//   Delivers repeated completions for each received chunk until cancelled
//   or the connection closes. Each completion updates |bytes_received|.
//   The same buffer(s) are reused for each receive—process data before
//   returning from the callback.
//
// Variable-size allocation:
//   Use iree_async_socket_recv_operation_size() for slab allocation with
//   inline span list storage, or set |buffers| to a caller-managed span list.
//
// Threading model:
//   Callback fires on the poll thread. Buffer contents are valid only
//   during the callback—copy out if needed after return.
typedef struct iree_async_socket_recv_operation_t {
  iree_async_operation_t base;

  // The socket to receive from.
  iree_async_socket_t* socket;

  // Scatter buffer list. The values pointer may reference trailing slab
  // data or caller-managed storage.
  iree_async_span_list_t buffers;

  // Result: total bytes received across all buffer entries.
  iree_host_size_t bytes_received;

  // Platform-specific storage for scatter-gather I/O.
  // Opaque to callers; initialized by the proactor.
  // Alignment ensures platform structs (msghdr, iovec) can be safely cast.
  union {
    // POSIX vectored I/O (all POSIX-based backends: poll, epoll, kqueue,
    // io_uring). Contains struct msghdr and struct iovec storage.
    struct {
      // Storage for struct msghdr used by RECVMSG.
      iree_alignas(iree_max_align_t) uint8_t
          msg_header[IREE_ASYNC_SOCKET_PLATFORM_MSGHDR_SIZE];
      // Storage for struct iovec array.
      uint8_t iovecs[IREE_ASYNC_SOCKET_PLATFORM_IOVEC_STORAGE];
    } posix;
  } platform;
} iree_async_socket_recv_operation_t;

// Maximum number of scatter-gather buffers supported in a single recv.
#define IREE_ASYNC_SOCKET_RECV_MAX_BUFFERS \
  IREE_ASYNC_SOCKET_SCATTER_GATHER_MAX_BUFFERS

// Computes the total allocation size for a recv operation with |buffer_count|
// scatter-gather entries using overflow-checked arithmetic.
static inline iree_status_t iree_async_socket_recv_operation_size(
    iree_host_size_t buffer_count, iree_host_size_t* out_size) {
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_async_socket_recv_operation_t), out_size,
      IREE_STRUCT_FIELD_FAM(buffer_count, iree_async_span_t));
}

// Initializes a slab-allocated recv operation. Sets buffers.values to point at
// trailing data within the slab.
static inline void iree_async_socket_recv_operation_initialize(
    iree_async_socket_recv_operation_t* operation,
    iree_host_size_t buffer_count) {
  operation->buffers.values =
      (iree_async_span_t*)((uint8_t*)operation + sizeof(*operation));
  operation->buffers.count = buffer_count;
}

//===----------------------------------------------------------------------===//
// Recv (pool-based)
//===----------------------------------------------------------------------===//

// Receives data from a connected socket into a buffer acquired from a pool.
//
// The proactor (or kernel, for io_uring) selects a buffer from the pool at
// receive time. On completion, |lease| identifies which buffer received data
// and |bytes_received| indicates how many bytes were written.
//
// This operation is designed for high-throughput receive patterns where the
// kernel selects buffers, eliminating head-of-line blocking from slow buffer
// processing.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | 5.19+    | yes  | yes
//
// Multishot availability:
//   generic | io_uring | IOCP | kqueue
//   emul    | 5.19+    | emul | emul
//
// Multishot mode (IREE_ASYNC_OPERATION_FLAG_MULTISHOT):
//   Delivers repeated completions until cancelled, each completion carrying
//   a fresh lease. The caller must return each lease (via
//   iree_async_buffer_pool_release) after processing the data.
//
// Pool exhaustion behavior:
//   If the pool is exhausted (no buffers available), the operation pauses
//   and resumes when buffers are returned:
//     io_uring: Stops delivering until the ring is replenished.
//     Other backends: May queue internally or return EAGAIN.
//
// Optimal path (io_uring 5.19+):
//   Uses provided buffer rings (PBUF_RING) for kernel-managed buffer
//   selection with zero syscalls per receive after initial setup.
//
// Threading model:
//   Callback fires on the poll thread. The lease is valid until released.
//   Release must happen from any thread (pool is thread-safe).
typedef struct iree_async_socket_recv_pool_operation_t {
  iree_async_operation_t base;

  // The socket to receive from.
  iree_async_socket_t* socket;

  // The buffer pool to acquire receive buffers from.
  iree_async_buffer_pool_t* pool;

  // Result: lease for the buffer that received data.
  // Valid only on successful completion. The caller owns the lease and must
  // return it via iree_async_buffer_pool_release() after processing.
  iree_async_buffer_lease_t lease;

  // Result: number of bytes received into the leased buffer.
  iree_host_size_t bytes_received;
} iree_async_socket_recv_pool_operation_t;

//===----------------------------------------------------------------------===//
// Send
//===----------------------------------------------------------------------===//

// Behavioral flags for send operations.
//
// Flag support by backend:
//   Flag   generic | io_uring | IOCP | kqueue
//   ─────────────────────────────────────────────
//   MORE   yes     | yes      | yes  | yes
//
// Note: Zero-copy send is controlled at socket creation time via
// IREE_ASYNC_SOCKET_OPTION_ZERO_COPY, not per-send. This matches the kernel
// model where SO_ZEROCOPY is a socket option, and simplifies the API by
// avoiding accidental performance pessimization (e.g., ZC on loopback).
//
enum iree_async_socket_send_flag_bits_e {
  IREE_ASYNC_SOCKET_SEND_FLAG_NONE = 0u,

  // Hint that more data follows (MSG_MORE / TCP_CORK).
  // The kernel may delay sending to coalesce with subsequent data.
  // Use this when sending multiple logical units that should share a packet.
  IREE_ASYNC_SOCKET_SEND_FLAG_MORE = 1u << 0,
};
typedef uint32_t iree_async_socket_send_flags_t;

// Sends data from one or more buffers (scatter-gather) to a connected socket.
//
// On success, |bytes_sent| indicates how many bytes were transmitted.
// Partial sends are possible under memory pressure or socket buffer limits.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Zero-copy path:
//   Create the socket with IREE_ASYNC_SOCKET_OPTION_ZERO_COPY and use
//   registered buffers (via iree_async_proactor_register_slab) for DMA directly
//   from application memory. The buffer must remain valid until the callback
//   fires. Zero-copy is a socket-level option, not a per-send flag.
//
// Scatter-gather semantics:
//   Buffers are sent in order as a single logical message. Maximum scatter
//   count is IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS. Use this to send headers
//   and payloads without copying into a contiguous buffer.
//
// Variable-size allocation:
//   Use iree_async_socket_send_operation_size() for slab allocation with
//   inline span list storage, or set |buffers| to a caller-managed span list.
//
// Threading model:
//   Callback fires on the poll thread when the send completes. For
//   zero-copy sends, "completes" means the kernel has finished DMA—the
//   buffer is now safe to modify or free.
typedef struct iree_async_socket_send_operation_t {
  iree_async_operation_t base;

  // The socket to send on.
  iree_async_socket_t* socket;

  // Scatter-gather buffer list. The values pointer may reference trailing slab
  // data or caller-managed storage.
  iree_async_span_list_t buffers;

  // Behavioral flags (zero-copy, cork, etc.).
  iree_async_socket_send_flags_t send_flags;

  // Result: total bytes sent across all buffer entries.
  iree_host_size_t bytes_sent;

  // Platform-specific storage for scatter-gather I/O.
  // Opaque to callers; initialized by the proactor.
  // Alignment ensures platform structs (msghdr, iovec) can be safely cast.
  union {
    // POSIX vectored I/O (all POSIX-based backends: poll, epoll, kqueue,
    // io_uring). Contains struct msghdr and struct iovec storage.
    struct {
      // Storage for struct msghdr used by SENDMSG.
      iree_alignas(iree_max_align_t) uint8_t
          msg_header[IREE_ASYNC_SOCKET_PLATFORM_MSGHDR_SIZE];
      // Storage for struct iovec array.
      uint8_t iovecs[IREE_ASYNC_SOCKET_PLATFORM_IOVEC_STORAGE];
    } posix;
  } platform;
} iree_async_socket_send_operation_t;

// Maximum number of scatter-gather buffers supported in a single send.
#define IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS \
  IREE_ASYNC_SOCKET_SCATTER_GATHER_MAX_BUFFERS

// Computes the total allocation size for a send operation with |buffer_count|
// scatter-gather entries using overflow-checked arithmetic.
static inline iree_status_t iree_async_socket_send_operation_size(
    iree_host_size_t buffer_count, iree_host_size_t* out_size) {
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_async_socket_send_operation_t), out_size,
      IREE_STRUCT_FIELD_FAM(buffer_count, iree_async_span_t));
}

// Initializes a slab-allocated send operation. Sets buffers.values to point at
// trailing data within the slab.
static inline void iree_async_socket_send_operation_initialize(
    iree_async_socket_send_operation_t* operation,
    iree_host_size_t buffer_count, iree_async_socket_send_flags_t send_flags) {
  operation->buffers.values =
      (iree_async_span_t*)((uint8_t*)operation + sizeof(*operation));
  operation->buffers.count = buffer_count;
  operation->send_flags = send_flags;
  operation->bytes_sent = 0;
}

//===----------------------------------------------------------------------===//
// Sendto (unconnected)
//===----------------------------------------------------------------------===//

// Sends data from one or more buffers to a specified destination address.
//
// Unlike SOCKET_SEND, this operation does not require the socket to be
// connected. The destination address is specified per-operation, making this
// suitable for:
//   - UDP servers responding to multiple clients from one socket
//   - Multicast/broadcast (where destination varies)
//   - Discovery protocols
//
// On success, |bytes_sent| indicates how many bytes were transmitted.
// Partial sends are possible under memory pressure or socket buffer limits.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Multishot: Not supported. Each send is a discrete operation.
//
// Zero-copy: Supported when socket has IREE_ASYNC_SOCKET_OPTION_ZERO_COPY and
// the proactor has IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND.
//
// Scatter-gather semantics:
//   Buffers are sent in order as a single logical datagram. Maximum scatter
//   count is IREE_ASYNC_SOCKET_SENDTO_MAX_BUFFERS.
//
// Threading model:
//   Callback fires on the poll thread when the send completes.
typedef struct iree_async_socket_sendto_operation_t {
  iree_async_operation_t base;

  // The socket to send on (need not be connected).
  iree_async_socket_t* socket;

  // Scatter-gather buffer list. The values pointer may reference trailing slab
  // data or caller-managed storage.
  iree_async_span_list_t buffers;

  // Behavioral flags (cork, etc.).
  iree_async_socket_send_flags_t send_flags;

  // Destination address (INPUT).
  iree_async_address_t destination;

  // Result: total bytes sent across all buffer entries.
  iree_host_size_t bytes_sent;

  // Platform-specific storage for scatter-gather I/O.
  // Opaque to callers; initialized by the proactor.
  // Alignment ensures platform structs (msghdr, iovec) can be safely cast.
  union {
    // POSIX vectored I/O (all POSIX-based backends: poll, epoll, kqueue,
    // io_uring). Contains struct msghdr and struct iovec storage.
    struct {
      // Storage for struct msghdr used by SENDMSG.
      iree_alignas(iree_max_align_t) uint8_t
          msg_header[IREE_ASYNC_SOCKET_PLATFORM_MSGHDR_SIZE];
      // Storage for struct iovec array.
      uint8_t iovecs[IREE_ASYNC_SOCKET_PLATFORM_IOVEC_STORAGE];
    } posix;
  } platform;
} iree_async_socket_sendto_operation_t;

// Maximum number of scatter-gather buffers supported in a single sendto.
#define IREE_ASYNC_SOCKET_SENDTO_MAX_BUFFERS \
  IREE_ASYNC_SOCKET_SCATTER_GATHER_MAX_BUFFERS

// Computes the total allocation size for a sendto operation with |buffer_count|
// scatter-gather entries using overflow-checked arithmetic.
static inline iree_status_t iree_async_socket_sendto_operation_size(
    iree_host_size_t buffer_count, iree_host_size_t* out_size) {
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_async_socket_sendto_operation_t), out_size,
      IREE_STRUCT_FIELD_FAM(buffer_count, iree_async_span_t));
}

// Initializes a slab-allocated sendto operation. Sets buffers.values to point
// at trailing data within the slab.
static inline void iree_async_socket_sendto_operation_initialize(
    iree_async_socket_sendto_operation_t* operation,
    iree_host_size_t buffer_count, iree_async_socket_send_flags_t send_flags,
    const iree_async_address_t* destination) {
  operation->buffers.values =
      (iree_async_span_t*)((uint8_t*)operation + sizeof(*operation));
  operation->buffers.count = buffer_count;
  operation->send_flags = send_flags;
  operation->destination = *destination;
  operation->bytes_sent = 0;
}

//===----------------------------------------------------------------------===//
// Recvfrom (unconnected)
//===----------------------------------------------------------------------===//

// Receives data from an unconnected socket, capturing the sender's address.
//
// Unlike SOCKET_RECV, this operation returns the source address of each
// received datagram in |sender|. Use this for:
//   - UDP servers handling multiple clients from one socket
//   - Protocols where sender identity matters
//   - Reply-to-sender patterns
//
// On success, |bytes_received| indicates how many bytes were written total.
// A zero |bytes_received| with OK status indicates graceful connection close
// (EOF) for stream sockets, or an empty datagram for datagram sockets.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Multishot: Not supported for single-shot recvfrom. For high-throughput
// unconnected receives, consider RECV_POOL with a per-buffer address scheme
// (future work).
//
// Scatter-gather semantics:
//   For multiple buffers, data fills buffers in order. Maximum scatter count is
//   IREE_ASYNC_SOCKET_RECVFROM_MAX_BUFFERS.
//
// Threading model:
//   Callback fires on the poll thread. |sender| is populated with the peer
//   address. Buffer contents and sender address are valid only during the
//   callback for multishot (not currently supported).
typedef struct iree_async_socket_recvfrom_operation_t {
  iree_async_operation_t base;

  // The socket to receive from.
  iree_async_socket_t* socket;

  // Scatter buffer list. The values pointer may reference trailing slab
  // data or caller-managed storage.
  iree_async_span_list_t buffers;

  // Result: sender address (OUTPUT).
  // Populated by the kernel with the source address of the received datagram.
  // The length field is updated to reflect the actual address size.
  iree_async_address_t sender;

  // Result: total bytes received across all buffer entries.
  iree_host_size_t bytes_received;

  // Platform-specific storage for scatter-gather I/O.
  // Opaque to callers; initialized by the proactor.
  // Alignment ensures platform structs (msghdr, iovec) can be safely cast.
  union {
    // POSIX vectored I/O (all POSIX-based backends: poll, epoll, kqueue,
    // io_uring). Contains struct msghdr and struct iovec storage.
    struct {
      // Storage for struct msghdr used by RECVMSG.
      iree_alignas(iree_max_align_t) uint8_t
          msg_header[IREE_ASYNC_SOCKET_PLATFORM_MSGHDR_SIZE];
      // Storage for struct iovec array.
      uint8_t iovecs[IREE_ASYNC_SOCKET_PLATFORM_IOVEC_STORAGE];
    } posix;
  } platform;
} iree_async_socket_recvfrom_operation_t;

// Maximum number of scatter-gather buffers supported in a single recvfrom.
#define IREE_ASYNC_SOCKET_RECVFROM_MAX_BUFFERS \
  IREE_ASYNC_SOCKET_SCATTER_GATHER_MAX_BUFFERS

// Computes the total allocation size for a recvfrom operation with
// |buffer_count| scatter-gather entries using overflow-checked arithmetic.
static inline iree_status_t iree_async_socket_recvfrom_operation_size(
    iree_host_size_t buffer_count, iree_host_size_t* out_size) {
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_async_socket_recvfrom_operation_t), out_size,
      IREE_STRUCT_FIELD_FAM(buffer_count, iree_async_span_t));
}

// Initializes a slab-allocated recvfrom operation. Sets buffers.values to point
// at trailing data within the slab. The sender address is cleared and will be
// populated on completion.
static inline void iree_async_socket_recvfrom_operation_initialize(
    iree_async_socket_recvfrom_operation_t* operation,
    iree_host_size_t buffer_count) {
  operation->buffers.values =
      (iree_async_span_t*)((uint8_t*)operation + sizeof(*operation));
  operation->buffers.count = buffer_count;
  memset(&operation->sender, 0, sizeof(operation->sender));
  operation->bytes_received = 0;
}

//===----------------------------------------------------------------------===//
// Socket close
//===----------------------------------------------------------------------===//

// Closes a socket asynchronously.
//
// Consumes the caller's reference: the socket must not be accessed after
// submit. On completion, the socket is destroyed and the callback reports
// whether the close succeeded.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Linger behavior:
//   If SO_LINGER is set on the socket, close may block until pending data
//   is transmitted or the linger timeout expires. Errors during this period
//   (e.g., connection reset) are reported via the callback status.
//
//   For non-blocking close without waiting for pending data, disable linger
//   before importing the socket or use socket options at creation time.
//
// Returns (via callback status):
//   IREE_STATUS_OK: Socket closed successfully.
//   IREE_STATUS_DEADLINE_EXCEEDED: SO_LINGER timeout expired.
//   Other: Network error during graceful close.
typedef struct iree_async_socket_close_operation_t {
  iree_async_operation_t base;

  // The socket to close. Consumed on submit.
  iree_async_socket_t* socket;
} iree_async_socket_close_operation_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_OPERATIONS_NET_H_
