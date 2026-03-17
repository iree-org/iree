// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Carrier abstraction: transport-agnostic byte/message movement.
//
// A carrier is the thing that moves bytes between endpoints. It is NOT a
// protocol - it doesn't understand streams, reliability, or semantics. It
// just moves data and reports completions.
//
// Carriers are the integration point for different network technologies:
//   - TCP: Reliable ordered byte stream
//   - UDP: Unreliable unordered datagrams
//   - RDMA: Kernel-bypass with one-sided operations
//   - devmem TCP: GPU-direct network I/O
//   - Shared memory: In-process testing and IPC
//
// ## Composability Rule
//
// Carriers never call syscalls directly. They submit operations to the
// proactor and receive completions. This keeps the I/O scheduling unified
// regardless of transport technology.
//
// ## Capability-Based Selection
//
// Callers query carrier capabilities to select optimal paths:
//   - RELIABLE + ORDERED: Can skip ARQ logic
//   - ZERO_COPY_TX/RX: Can avoid copies for large transfers
//   - DIRECT_WRITE/READ: Can use one-sided RDMA operations
//
// ## Backpressure
//
// Carriers expose backpressure via query_send_budget(). When budget is
// exhausted, callers should wait for send completions before submitting
// more data. This prevents buffer exhaustion and RNR errors (RDMA).

#ifndef IREE_NET_CARRIER_H_
#define IREE_NET_CARRIER_H_

#include "iree/async/api.h"
#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Carrier properties and capabilities
//===----------------------------------------------------------------------===//

// Carrier lifecycle state for deactivate-before-destroy enforcement.
typedef enum iree_net_carrier_state_e {
  IREE_NET_CARRIER_STATE_CREATED = 0,      // Not yet activated.
  IREE_NET_CARRIER_STATE_ACTIVE = 1,       // Recv handler receiving data.
  IREE_NET_CARRIER_STATE_DRAINING = 2,     // Deactivate in progress.
  IREE_NET_CARRIER_STATE_DEACTIVATED = 3,  // Drained, safe to destroy.
} iree_net_carrier_state_t;

// Capability flags describing what a carrier can do.
// Query with iree_net_carrier_capabilities().
typedef enum iree_net_carrier_capability_bits_e {
  IREE_NET_CARRIER_CAPABILITY_NONE = 0u,

  // Carrier guarantees delivery (TCP, RDMA RC).
  // When set, callers can skip implementing ARQ/retransmission.
  IREE_NET_CARRIER_CAPABILITY_RELIABLE = 1u << 0,

  // Carrier guarantees in-order delivery (TCP).
  // When set, sequence numbers are for correlation, not reordering.
  IREE_NET_CARRIER_CAPABILITY_ORDERED = 1u << 1,

  // Carrier supports zero-copy send (io_uring SEND_ZC, RDMA).
  // Buffer must remain valid until send completion notification.
  IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_TX = 1u << 2,

  // Carrier supports zero-copy receive (io_uring buffer rings, RDMA).
  // Data arrives in pre-registered buffers from the carrier's pool.
  IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_RX = 1u << 3,

  // Carrier can use registered memory regions for optimal transfer.
  // Spans referencing registered regions get zero-copy treatment.
  IREE_NET_CARRIER_CAPABILITY_REGISTERED_REGIONS = 1u << 4,

  // Carrier supports one-sided write (RDMA WRITE, devmem TCP).
  // Caller provides remote handle; data lands without remote CPU.
  IREE_NET_CARRIER_CAPABILITY_DIRECT_WRITE = 1u << 5,

  // Carrier supports one-sided read (RDMA READ).
  // Caller provides remote handle; data fetched without remote CPU.
  IREE_NET_CARRIER_CAPABILITY_DIRECT_READ = 1u << 6,

  // Carrier supports remote atomic operations (RDMA atomics).
  // fetch-add, compare-and-swap on remote memory.
  IREE_NET_CARRIER_CAPABILITY_DIRECT_ATOMIC = 1u << 7,

  // Carrier is datagram-based (UDP, RDMA UD).
  // Message boundaries are preserved; MTU limits apply.
  IREE_NET_CARRIER_CAPABILITY_DATAGRAM = 1u << 8,

  // Carrier supports multicast (UDP multicast, RDMA UD multicast).
  IREE_NET_CARRIER_CAPABILITY_MULTICAST = 1u << 9,

  // Carrier can send from CPU-inaccessible device memory (devmem TCP, RDMA).
  // When set, spans with base_ptr == NULL can be used for send operations.
  // When not set, all send spans must be CPU-accessible.
  // Note: This is orthogonal to ZERO_COPY_TX (which avoids kernel copies but
  // may still require CPU access for io_uring SEND_ZC on host memory).
  IREE_NET_CARRIER_CAPABILITY_DEVICE_MEMORY_TX = 1u << 10,

  // Carrier can receive into CPU-inaccessible device memory (devmem TCP, RDMA).
  // When set, receive buffers with base_ptr == NULL can be used.
  // When not set, all receive buffers must be CPU-accessible.
  IREE_NET_CARRIER_CAPABILITY_DEVICE_MEMORY_RX = 1u << 11,
} iree_net_carrier_capability_bits_t;
typedef uint32_t iree_net_carrier_capabilities_t;

//===----------------------------------------------------------------------===//
// Remote memory handle (for RDMA and devmem)
//===----------------------------------------------------------------------===//

// Opaque handle to remote memory. Interpretation is carrier-specific:
//   - RDMA: (rkey, remote_addr)
//   - Shared memory: offset in shared region
//   - Emulated: buffer ID that triggers receiver response
//
// Handles are exchanged via control channel during buffer registration.
typedef struct iree_net_remote_handle_t {
  uint64_t opaque[2];
} iree_net_remote_handle_t;

// Returns an empty/invalid remote handle.
static inline iree_net_remote_handle_t iree_net_remote_handle_null(void) {
  iree_net_remote_handle_t handle = {{0, 0}};
  return handle;
}

// Checks if handle is null/invalid.
static inline bool iree_net_remote_handle_is_null(
    iree_net_remote_handle_t handle) {
  return handle.opaque[0] == 0 && handle.opaque[1] == 0;
}

//===----------------------------------------------------------------------===//
// Send/recv parameters
//===----------------------------------------------------------------------===//

// Flags for send operations.
typedef enum iree_net_send_flag_bits_e {
  IREE_NET_SEND_FLAG_NONE = 0u,

  // Request zero-copy if carrier supports it (avoids kernel buffer copy).
  // All sends require buffers valid until completion; this flag additionally
  // prevents the kernel from copying data to its own buffers, so the DMA
  // engine reads directly from the caller's buffer.
  IREE_NET_SEND_FLAG_ZERO_COPY = 1u << 0,

  // This is the last segment of a message (for datagram carriers).
  IREE_NET_SEND_FLAG_END_OF_MESSAGE = 1u << 1,

  // High priority (may use separate queue/channel).
  IREE_NET_SEND_FLAG_HIGH_PRIORITY = 1u << 2,
} iree_net_send_flag_bits_t;
typedef uint32_t iree_net_send_flags_t;

// Parameters for send operations.
typedef struct iree_net_send_params_t {
  // Data to send. Spans may reference registered regions for zero-copy.
  iree_async_span_list_t data;

  // Flags controlling send behavior.
  iree_net_send_flags_t flags;

  // User data passed to completion callback.
  uint64_t user_data;
} iree_net_send_params_t;

// Flags for direct write operations.
typedef enum iree_net_direct_write_flag_bits_e {
  IREE_NET_DIRECT_WRITE_FLAG_NONE = 0u,

  // Generates a completion on the receiver (RDMA WRITE WITH IMMEDIATE).
  // When set, the receiver's proactor wakes and |immediate| is delivered.
  // This consumes a pre-posted RECV work queue entry on the receiver.
  // Required for signaling; use for frontier advancement in collectives.
  IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER = 1u << 0,
} iree_net_direct_write_flag_bits_t;
typedef uint32_t iree_net_direct_write_flags_t;

// Parameters for direct write operations (RDMA WRITE, etc.).
typedef struct iree_net_direct_write_params_t {
  // Local source data.
  iree_async_span_t local;

  // Remote destination handle.
  iree_net_remote_handle_t remote;

  // Flags controlling write behavior.
  iree_net_direct_write_flags_t flags;

  // Immediate value delivered to receiver when SIGNAL_RECEIVER is set.
  // For RDMA, this is the 32-bit immediate in RDMA_WRITE_WITH_IMM.
  // Typically encodes stream_id + sequence for frontier tracking.
  uint32_t immediate;

  // User data passed to completion callback.
  uint64_t user_data;
} iree_net_direct_write_params_t;

// Parameters for direct read operations (RDMA READ).
typedef struct iree_net_direct_read_params_t {
  // Local destination buffer.
  iree_async_span_t local;

  // Remote source handle.
  iree_net_remote_handle_t remote;

  // User data passed to completion callback.
  uint64_t user_data;
} iree_net_direct_read_params_t;

//===----------------------------------------------------------------------===//
// Backpressure / flow control
//===----------------------------------------------------------------------===//

// Send budget for backpressure management.
// Carriers report both byte budget and operation slot budget because they
// have different limiting factors:
//   - TCP: Limited by socket buffer space (bytes)
//   - RDMA: Limited by send queue depth (slots) to avoid RNR errors
//   - UDP: Limited by socket buffer and MTU
//
// Callers should check both dimensions before submitting operations.
typedef struct iree_net_carrier_send_budget_t {
  // Bytes available for sending. 0 means backpressured on bandwidth.
  // For TCP, this reflects socket send buffer space.
  // For RDMA, this is typically large (limited by slots, not bytes).
  iree_host_size_t bytes;

  // Operation slots available. 0 means backpressured on queue depth.
  // For TCP, this is effectively unlimited (use UINT32_MAX).
  // For RDMA, this reflects send queue depth minus outstanding WRs.
  // For RDMA WRITE WITH IMMEDIATE, also limited by receiver's posted RECVs.
  uint32_t slots;
} iree_net_carrier_send_budget_t;

//===----------------------------------------------------------------------===//
// Callbacks and handlers
//===----------------------------------------------------------------------===//

// Recv handler invoked when data is received on an activated carrier.
//
// Called from the proactor thread for each received message/chunk. The handler
// receives a view into the receive buffer and optionally a lease that can be
// retained if the data is needed beyond the callback.
//
// |lease| may be NULL for carriers that don't use buffer pools (e.g., loopback
// where data comes from the sender's buffer). When non-NULL, the carrier
// releases the lease after this callback returns - handlers that need to retain
// the buffer must copy the data. Call iree_async_buffer_lease_release() to
// release early and return buffers to the pool sooner.
//
// Return iree_ok_status() to indicate successful processing. Returning an
// error causes the carrier to report the error and may trigger deactivation.
typedef iree_status_t (*iree_net_carrier_recv_handler_fn_t)(
    void* user_data, iree_async_span_t data, iree_async_buffer_lease_t* lease);

typedef struct iree_net_carrier_recv_handler_t {
  iree_net_carrier_recv_handler_fn_t fn;
  void* user_data;
} iree_net_carrier_recv_handler_t;

// Callback invoked when carrier deactivation completes (all operations
// drained). After this callback, the carrier is in DEACTIVATED state and safe
// to release.
typedef void (*iree_net_carrier_deactivate_callback_fn_t)(void* user_data);

// Completion callback for send operations and legacy recv operations.
//
// Invoked from proactor poll() on the polling thread. Callbacks for a single
// carrier are never concurrent - they're serialized in completion order.
// Implementations should be non-blocking.
//
// |operation_user_data| is echoed from the send/recv params for correlation.
// |status| is OK on success, error on failure.
// |bytes_transferred| is the number of bytes actually sent/received.
// |recv_lease| is non-NULL only for successful pooled recv completions;
// caller must release via iree_async_buffer_lease_release() after processing.
typedef void (*iree_net_carrier_completion_fn_t)(
    void* callback_user_data, uint64_t operation_user_data,
    iree_status_t status, iree_host_size_t bytes_transferred,
    iree_async_buffer_lease_t* recv_lease);

typedef struct iree_net_carrier_callback_t {
  iree_net_carrier_completion_fn_t fn;
  void* user_data;
} iree_net_carrier_callback_t;

//===----------------------------------------------------------------------===//
// iree_net_carrier_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_carrier_t iree_net_carrier_t;
typedef struct iree_net_carrier_vtable_t iree_net_carrier_vtable_t;

// Carrier implementation vtable.
// See the corresponding iree_net_carrier_*() inline functions for detailed
// documentation of each operation.
struct iree_net_carrier_vtable_t {
  void (*destroy)(iree_net_carrier_t* carrier);

  // Handler-driven recv model (new).
  void (*set_recv_handler)(iree_net_carrier_t* carrier,
                           iree_net_carrier_recv_handler_t handler);
  iree_status_t (*activate)(iree_net_carrier_t* carrier);
  iree_status_t (*deactivate)(
      iree_net_carrier_t* carrier,
      iree_net_carrier_deactivate_callback_fn_t callback, void* user_data);

  // Data transfer.
  iree_net_carrier_send_budget_t (*query_send_budget)(
      iree_net_carrier_t* carrier);
  iree_status_t (*send)(iree_net_carrier_t* carrier,
                        const iree_net_send_params_t* params);
  iree_status_t (*shutdown)(iree_net_carrier_t* carrier);

  // One-sided RDMA operations (optional).
  iree_status_t (*direct_write)(iree_net_carrier_t* carrier,
                                const iree_net_direct_write_params_t* params);
  iree_status_t (*direct_read)(iree_net_carrier_t* carrier,
                               const iree_net_direct_read_params_t* params);
  iree_status_t (*register_buffer)(iree_net_carrier_t* carrier,
                                   iree_async_region_t* region,
                                   iree_net_remote_handle_t* out_handle);
  void (*unregister_buffer)(iree_net_carrier_t* carrier,
                            iree_net_remote_handle_t handle);
};

// Statistics for a carrier. Cumulative since carrier creation.
typedef struct iree_net_carrier_stats_t {
  // Total bytes sent (successfully completed send operations).
  int64_t bytes_sent;
  // Total bytes received (successfully completed recv operations).
  int64_t bytes_received;
} iree_net_carrier_stats_t;

// Base carrier structure. Concrete implementations embed this.
struct iree_net_carrier_t {
  iree_atomic_ref_count_t ref_count;
  const iree_net_carrier_vtable_t* vtable;

  // Lifecycle state - implementations update via iree_net_carrier_set_state().
  iree_atomic_int32_t state;  // iree_net_carrier_state_t

  // Immutable carrier properties (set at creation, no vtable dispatch).
  iree_net_carrier_capabilities_t capabilities;
  iree_host_size_t mtu;      // 0 for stream carriers.
  iree_host_size_t max_iov;  // Max scatter-gather entries per operation.

  // Handler for received data (handler-driven model).
  // Set via set_recv_handler() before activate().
  iree_net_carrier_recv_handler_t recv_handler;

  // Completion callback for send operations and legacy recv.
  iree_net_carrier_callback_t callback;

  iree_allocator_t host_allocator;

  // Number of operations currently in flight (submitted but not completed).
  // Used to determine when it's safe to release the carrier after shutdown.
  iree_atomic_int32_t pending_operations;

  // Cumulative statistics. Updated atomically as operations complete.
  iree_atomic_int64_t bytes_sent;
  iree_atomic_int64_t bytes_received;
};

// Initializes base carrier fields. Called by carrier implementations.
//
// The carrier starts in CREATED state with no recv handler. Before activation,
// callers must set a recv handler via iree_net_carrier_set_recv_handler().
//
// Parameters:
//   vtable: Implementation vtable.
//   capabilities: Carrier capabilities (RELIABLE, ORDERED, ZERO_COPY_*, etc.).
//   mtu: Maximum transfer unit for datagram carriers, 0 for stream carriers.
//   max_iov: Maximum scatter-gather entries per operation.
//   callback: Completion callback for send operations.
//   host_allocator: Allocator for carrier and internal allocations.
//   out_carrier: Carrier to initialize.
static inline void iree_net_carrier_initialize(
    const iree_net_carrier_vtable_t* vtable,
    iree_net_carrier_capabilities_t capabilities, iree_host_size_t mtu,
    iree_host_size_t max_iov, iree_net_carrier_callback_t callback,
    iree_allocator_t host_allocator, iree_net_carrier_t* out_carrier) {
  iree_atomic_ref_count_init(&out_carrier->ref_count);
  out_carrier->vtable = vtable;
  iree_atomic_store(&out_carrier->state, IREE_NET_CARRIER_STATE_CREATED,
                    iree_memory_order_relaxed);
  out_carrier->capabilities = capabilities;
  out_carrier->mtu = mtu;
  out_carrier->max_iov = max_iov;
  out_carrier->recv_handler.fn = NULL;
  out_carrier->recv_handler.user_data = NULL;
  out_carrier->callback = callback;
  out_carrier->host_allocator = host_allocator;
  iree_atomic_store(&out_carrier->pending_operations, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&out_carrier->bytes_sent, 0, iree_memory_order_relaxed);
  iree_atomic_store(&out_carrier->bytes_received, 0, iree_memory_order_relaxed);
}

static inline void iree_net_carrier_retain(iree_net_carrier_t* carrier) {
  iree_atomic_ref_count_inc(&carrier->ref_count);
}

static inline void iree_net_carrier_release(iree_net_carrier_t* carrier) {
  if (iree_atomic_ref_count_dec(&carrier->ref_count) == 1) {
    carrier->vtable->destroy(carrier);
  }
}

// Returns the current lifecycle state of the carrier.
static inline iree_net_carrier_state_t iree_net_carrier_state(
    iree_net_carrier_t* carrier) {
  return (iree_net_carrier_state_t)iree_atomic_load(&carrier->state,
                                                    iree_memory_order_acquire);
}

// Updates the carrier lifecycle state. Called by carrier implementations as
// they transition through CREATED -> ACTIVE -> DRAINING -> DEACTIVATED.
static inline void iree_net_carrier_set_state(iree_net_carrier_t* carrier,
                                              iree_net_carrier_state_t state) {
  iree_atomic_store(&carrier->state, state, iree_memory_order_release);
}

// Sets the handler for received data. Must be called before activate().
//
// The handler is invoked from the proactor thread for each received message or
// chunk. Unlike the legacy recv() model where callers post explicit receive
// operations, the handler-driven model automatically posts receives after
// activation and delivers data to the handler.
//
// The handler receives:
//   - data: View of received bytes (valid only during callback).
//   - lease: If non-NULL, can be retained to keep the buffer beyond callback.
//            Release via iree_async_buffer_lease_release() when done.
//
// Return iree_ok_status() from the handler to continue receiving. Returning an
// error causes the carrier to report the error and begin deactivation.
static inline void iree_net_carrier_set_recv_handler(
    iree_net_carrier_t* carrier, iree_net_carrier_recv_handler_t handler) {
  carrier->vtable->set_recv_handler(carrier, handler);
}

// Activates the carrier to begin receiving data.
//
// After activation, the carrier auto-posts receives and delivers data to the
// recv handler. The carrier transitions to ACTIVE state.
//
// Prerequisites:
//   - Recv handler must be set via set_recv_handler().
//   - Carrier must be in CREATED state.
//
// Returns IREE_STATUS_FAILED_PRECONDITION if prerequisites aren't met.
static inline iree_status_t iree_net_carrier_activate(
    iree_net_carrier_t* carrier) {
  return carrier->vtable->activate(carrier);
}

// Begins graceful deactivation of the carrier.
//
// This drains outstanding operations and stops auto-posting new receives. The
// carrier transitions through DRAINING -> DEACTIVATED. The callback fires when
// deactivation completes and the carrier is safe to release.
//
// Unlike shutdown() which only stops the send direction, deactivate() stops
// both directions and guarantees all pending operations have completed before
// the callback fires.
//
// Typical sequence:
//   1. iree_net_carrier_deactivate() - begin draining
//   2. Continue polling proactor (operations complete, recv handler may fire)
//   3. Callback fires when all drained - carrier now in DEACTIVATED state
//   4. iree_net_carrier_release() - safe to release
static inline iree_status_t iree_net_carrier_deactivate(
    iree_net_carrier_t* carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
  return carrier->vtable->deactivate(carrier, callback, user_data);
}

// Returns the capabilities of this carrier.
// O(1) direct field access, no vtable dispatch.
static inline iree_net_carrier_capabilities_t iree_net_carrier_capabilities(
    const iree_net_carrier_t* carrier) {
  return carrier->capabilities;
}

// Returns the MTU (0 for stream carriers).
// O(1) direct field access, no vtable dispatch.
static inline iree_host_size_t iree_net_carrier_mtu(
    const iree_net_carrier_t* carrier) {
  return carrier->mtu;
}

// Returns the maximum scatter-gather entries per operation.
// O(1) direct field access, no vtable dispatch.
static inline iree_host_size_t iree_net_carrier_max_iov(
    const iree_net_carrier_t* carrier) {
  return carrier->max_iov;
}

// Returns the number of operations currently in flight.
// Use this to determine when it's safe to release the carrier after shutdown.
//
// Safe release sequence:
//   1. iree_net_carrier_shutdown() - stop sending, signal EOF to peer
//   2. Drain pending recv operations (process incoming data)
//   3. Poll proactor until pending_count() == 0
//   4. iree_net_carrier_release() - safe once no operations pending
static inline int32_t iree_net_carrier_pending_count(
    iree_net_carrier_t* carrier) {
  return iree_atomic_load(&carrier->pending_operations,
                          iree_memory_order_acquire);
}

// Returns current statistics for the carrier.
// Statistics are cumulative since carrier creation. Thread-safe.
static inline iree_net_carrier_stats_t iree_net_carrier_query_stats(
    iree_net_carrier_t* carrier) {
  iree_net_carrier_stats_t stats;
  stats.bytes_sent =
      iree_atomic_load(&carrier->bytes_sent, iree_memory_order_relaxed);
  stats.bytes_received =
      iree_atomic_load(&carrier->bytes_received, iree_memory_order_relaxed);
  return stats;
}

// Queries the send budget for backpressure management.
//
// Returns both byte budget and operation slot budget. Different carriers have
// different bottlenecks:
//   - TCP: Limited by socket send buffer space (bytes).
//   - RDMA: Limited by send queue depth (slots) to avoid RNR errors.
//   - UDP: Limited by socket buffer and MTU.
//
// When either dimension reaches 0, the carrier is backpressured and callers
// should wait for send completions before submitting more operations.
static inline iree_net_carrier_send_budget_t iree_net_carrier_query_send_budget(
    iree_net_carrier_t* carrier) {
  return carrier->vtable->query_send_budget(carrier);
}

// Submits a send operation.
//
// |params| specifies the data to send (scatter-gather list), flags, and
// user_data for correlation. See iree_net_send_params_t for flag options
// including ZERO_COPY and HIGH_PRIORITY.
//
// The data buffers referenced by |params->data| must remain valid until the
// completion callback fires. This is the standard async I/O contract — the
// proactor may not consume the data during the send() call itself (e.g.,
// io_uring defers SQE processing). Callers that need to send from transient
// storage (stack buffers, temporary allocations) must copy the data into
// stable storage before calling send().
//
// Prerequisites:
//   - Carrier must be activated (ACTIVE state).
//
// Returns IREE_STATUS_FAILED_PRECONDITION if carrier is not activated.
static inline iree_status_t iree_net_carrier_send(
    iree_net_carrier_t* carrier, const iree_net_send_params_t* params) {
  return carrier->vtable->send(carrier, params);
}

// Initiates graceful shutdown of the carrier (send direction only).
//
// After shutdown, no new send operations will succeed. Pending operations
// continue to completion. For TCP, this sends FIN to the peer.
//
// Shutdown is one-way: the carrier can still receive data until the peer also
// closes or the carrier is released. Use deactivate() when done receiving.
static inline iree_status_t iree_net_carrier_shutdown(
    iree_net_carrier_t* carrier) {
  return carrier->vtable->shutdown(carrier);
}

// Writes local data to remote memory without involving the remote CPU.
//
// This is an RDMA WRITE operation: data moves directly from local memory to
// remote memory via DMA. The remote side is not notified unless
// IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER is set, which uses RDMA WRITE
// WITH IMMEDIATE to wake the remote proactor.
//
// Only available if IREE_NET_CARRIER_CAPABILITY_DIRECT_WRITE is set.
// Returns IREE_STATUS_UNIMPLEMENTED for carriers that don't support one-sided
// writes (TCP, UDP, loopback).
static inline iree_status_t iree_net_carrier_direct_write(
    iree_net_carrier_t* carrier, const iree_net_direct_write_params_t* params) {
  if (!carrier->vtable->direct_write) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "carrier does not support direct_write");
  }
  return carrier->vtable->direct_write(carrier, params);
}

// Reads remote memory into a local buffer without involving the remote CPU.
//
// This is an RDMA READ operation: data moves directly from remote memory to
// local memory via DMA. The remote side is not notified.
//
// Only available if IREE_NET_CARRIER_CAPABILITY_DIRECT_READ is set.
// Returns IREE_STATUS_UNIMPLEMENTED for carriers that don't support one-sided
// reads (TCP, UDP, loopback).
static inline iree_status_t iree_net_carrier_direct_read(
    iree_net_carrier_t* carrier, const iree_net_direct_read_params_t* params) {
  if (!carrier->vtable->direct_read) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "carrier does not support direct_read");
  }
  return carrier->vtable->direct_read(carrier, params);
}

// Registers a local buffer for remote access (RDMA memory registration).
//
// Returns a handle that can be sent to peers for direct_write/read operations.
// The peer uses this handle as the remote destination/source for one-sided
// DMA operations.
//
// For RDMA carriers, this performs ibv_reg_mr or ibv_reg_dmabuf_mr to pin
// memory and get an rkey. The region must remain valid and registered until
// iree_net_carrier_unregister_buffer() is called.
//
// Only meaningful if IREE_NET_CARRIER_CAPABILITY_DIRECT_WRITE or
// IREE_NET_CARRIER_CAPABILITY_DIRECT_READ is set.
// Returns IREE_STATUS_UNIMPLEMENTED for carriers that don't support remote
// memory access.
static inline iree_status_t iree_net_carrier_register_buffer(
    iree_net_carrier_t* carrier, iree_async_region_t* region,
    iree_net_remote_handle_t* out_handle) {
  if (!carrier->vtable->register_buffer) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "carrier does not support buffer registration");
  }
  return carrier->vtable->register_buffer(carrier, region, out_handle);
}

// Unregisters a previously registered buffer.
//
// The handle becomes invalid after this call. Any in-flight direct_write/read
// operations targeting this handle may fail or produce undefined behavior.
// Ensure all operations using the handle have completed before unregistering.
//
// For carriers that don't support buffer registration, this is a no-op.
static inline void iree_net_carrier_unregister_buffer(
    iree_net_carrier_t* carrier, iree_net_remote_handle_t handle) {
  if (carrier->vtable->unregister_buffer) {
    carrier->vtable->unregister_buffer(carrier, handle);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_H_
