// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/tcp/carrier.h"

#include "iree/async/operations/net.h"
#include "iree/async/socket.h"
#include "iree/base/internal/atomics.h"

//===----------------------------------------------------------------------===//
// Send slot management
//===----------------------------------------------------------------------===//

typedef enum iree_net_tcp_send_slot_state_e {
  IREE_NET_TCP_SEND_SLOT_STATE_FREE = 0,
  IREE_NET_TCP_SEND_SLOT_STATE_IN_USE = 1,
} iree_net_tcp_send_slot_state_t;

// Inline storage capacity for unregistered span data in each send slot.
// Covers TCP frame headers (16 bytes) and small metadata. Unregistered spans
// (region == NULL) use raw pointers that may reference caller stack frames.
// The io_uring backend defers kernel data reads until io_uring_enter, so data
// must survive beyond the submit call. This buffer provides stable storage.
#define IREE_NET_TCP_SEND_SLOT_INLINE_DATA_CAPACITY 64

// Pre-allocated send operation slot.
// Each slot tracks one in-flight send operation and provides inline storage
// for the span list and small unregistered span data. This ensures all data
// referenced by io_uring SQEs survives until the kernel processes them.
typedef struct iree_net_tcp_send_slot_t {
  // The async send operation submitted to the proactor.
  iree_async_socket_send_operation_t operation;

  // User data from iree_net_send_params_t, echoed to completion callback.
  uint64_t user_data;

  // Slot state for debugging and leak detection.
  iree_atomic_int32_t state;

  // Slot-local copy of the span list values. The caller's span array may be
  // stack-allocated, so we copy it here to ensure operation.buffers.values
  // references stable memory.
  iree_async_span_t inline_spans[IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS];

  // Inline data storage for unregistered span payloads. When a span has
  // region == NULL, its data pointer may reference transient memory (e.g. a
  // stack-allocated frame header). Small payloads are copied here so that
  // iovec data pointers in the platform storage reference heap memory.
  uint8_t inline_data[IREE_NET_TCP_SEND_SLOT_INLINE_DATA_CAPACITY];
} iree_net_tcp_send_slot_t;

//===----------------------------------------------------------------------===//
// Deactivate callback storage
//===----------------------------------------------------------------------===//

// Bundles deactivate callback function with user data.
typedef struct iree_net_tcp_deactivate_callback_t {
  iree_net_carrier_deactivate_callback_fn_t fn;
  void* user_data;
} iree_net_tcp_deactivate_callback_t;

//===----------------------------------------------------------------------===//
// TCP carrier structure
//===----------------------------------------------------------------------===//

typedef struct iree_net_tcp_carrier_t {
  // Base carrier (must be first for safe upcasting).
  iree_net_carrier_t base;

  // Proactor this carrier is bound to. Not owned.
  iree_async_proactor_t* proactor;

  // Socket owned by this carrier.
  iree_async_socket_t* socket;

  // Buffer pool for receive operations. Not owned - caller must ensure it
  // outlives the carrier.
  iree_async_buffer_pool_t* recv_pool;

  // Send slot ring buffer.
  struct {
    // Number of slots (power of 2 for efficient masking).
    uint32_t slot_count;

    // Index mask: (slot_count - 1).
    uint32_t slot_mask;

    // Next slot to use for submission. Atomically incremented by senders.
    // Uses unsigned type for well-defined wraparound behavior.
    iree_atomic_uint32_t head;

    // Next slot to complete. Updated by proactor thread with release semantics;
    // read by sender threads with acquire semantics for proper visibility.
    // Uses unsigned type for well-defined wraparound behavior.
    iree_atomic_uint32_t tail;

    // Pre-allocated operations. Indexed by (slot_index & slot_mask).
    iree_net_tcp_send_slot_t* slots;
  } send;

  // Receive operations.
  struct {
    // True if using multishot recv with PBUF_RING (io_uring 5.19+).
    bool multishot_enabled;

    // For multishot: single operation that stays posted.
    // For single-shot: ring of recv operations.
    union {
      struct {
        iree_async_socket_recv_pool_operation_t operation;
      } multishot;
      struct {
        // Number of single-shot recv operations (power of 2).
        uint32_t operation_count;
        // The recv operations, re-posted after each completion.
        iree_async_socket_recv_pool_operation_t* operations;
        // Number of currently active (posted) recv operations.
        iree_atomic_int32_t active_count;
      } single_shot;
    };
  } recv;

  // Sticky failure status. Initially NULL (no error).
  // Set via atomic CAS - first error wins, subsequent errors are ignored.
  iree_atomic_intptr_t failure_status;

  // Callback to invoke when deactivation completes.
  iree_net_tcp_deactivate_callback_t deactivate_callback;
} iree_net_tcp_carrier_t;

// Casts from base carrier to TCP carrier.
static inline iree_net_tcp_carrier_t* iree_net_tcp_carrier_cast(
    iree_net_carrier_t* base_carrier) {
  return (iree_net_tcp_carrier_t*)base_carrier;
}

//===----------------------------------------------------------------------===//
// Sticky failure status helpers
//===----------------------------------------------------------------------===//

// Sets the sticky failure status (first error wins).
static void iree_net_tcp_carrier_set_failure_status(
    iree_net_tcp_carrier_t* carrier, iree_status_t status) {
  intptr_t expected = 0;
  intptr_t desired = (intptr_t)status;
  if (!iree_atomic_compare_exchange_strong(&carrier->failure_status, &expected,
                                           desired, iree_memory_order_release,
                                           iree_memory_order_relaxed)) {
    // Another error was already captured. Ignore this one.
    iree_status_ignore(status);
  }
}

// Returns a CLONE of the sticky failure status, or iree_ok_status() if none.
// Caller takes ownership of the returned status.
static iree_status_t iree_net_tcp_carrier_get_failure_status(
    iree_net_tcp_carrier_t* carrier) {
  intptr_t stored =
      iree_atomic_load(&carrier->failure_status, iree_memory_order_acquire);
  if (stored == 0) {
    return iree_ok_status();
  }
  // Clone to avoid double-free: the stored status remains owned by carrier.
  return iree_status_clone((iree_status_t)stored);
}

// Consumes the sticky failure status (called once during destruction).
static void iree_net_tcp_carrier_consume_failure_status(
    iree_net_tcp_carrier_t* carrier) {
  intptr_t stored = iree_atomic_exchange(&carrier->failure_status, 0,
                                         iree_memory_order_acq_rel);
  if (stored != 0) {
    iree_status_ignore((iree_status_t)stored);
  }
}

//===----------------------------------------------------------------------===//
// Deactivation completion check
//===----------------------------------------------------------------------===//

// Checks if deactivation has completed and invokes callback if so.
static void iree_net_tcp_carrier_maybe_complete_deactivation(
    iree_net_tcp_carrier_t* carrier) {
  // Check state is DRAINING.
  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  if (state != IREE_NET_CARRIER_STATE_DRAINING) {
    return;
  }

  // Check all pending operations have completed.
  int32_t pending = iree_atomic_load(&carrier->base.pending_operations,
                                     iree_memory_order_acquire);
  if (pending > 0) {
    return;
  }

  // All operations drained - transition to DEACTIVATED.
  iree_net_carrier_set_state(&carrier->base,
                             IREE_NET_CARRIER_STATE_DEACTIVATED);

  // Invoke deactivate callback if set.
  if (carrier->deactivate_callback.fn) {
    carrier->deactivate_callback.fn(carrier->deactivate_callback.user_data);
  }
}

//===----------------------------------------------------------------------===//
// Recv completion handlers
//===----------------------------------------------------------------------===//

// Processes received data by invoking the recv handler and handling errors.
// Returns true if the carrier should continue receiving, false if receiving
// should stop (EOF, error, or deactivating).
static bool iree_net_tcp_carrier_process_recv(
    iree_net_tcp_carrier_t* carrier, iree_status_t status,
    iree_host_size_t bytes_received, iree_async_buffer_lease_t* lease) {
  // Check for errors from the recv operation itself.
  if (!iree_status_is_ok(status)) {
    iree_net_tcp_carrier_set_failure_status(carrier, status);
    return false;
  }

  // EOF: bytes_received == 0 with OK status means graceful close.
  if (bytes_received == 0) {
    return false;
  }

  // Build span from the lease, adjusted for actual bytes received.
  // The lease's span covers the full buffer; we narrow to received bytes.
  iree_async_span_t data = iree_async_span_make(
      lease->span.region, lease->span.offset, bytes_received);

  // Invoke the recv handler.
  iree_status_t handler_status = carrier->base.recv_handler.fn(
      carrier->base.recv_handler.user_data, data, lease);

  // Capture handler errors as sticky failure status.
  if (!iree_status_is_ok(handler_status)) {
    iree_net_tcp_carrier_set_failure_status(carrier, handler_status);
    return false;
  }

  // Check if we're still active (handler might have triggered deactivation).
  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  return state == IREE_NET_CARRIER_STATE_ACTIVE;
}

// Completion callback for multishot recv operations.
// Fires repeatedly until EOF, error, or cancellation.
static void iree_net_tcp_carrier_recv_completion_multishot(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_tcp_carrier_t* carrier = (iree_net_tcp_carrier_t*)user_data;
  iree_async_socket_recv_pool_operation_t* recv_op =
      (iree_async_socket_recv_pool_operation_t*)operation;

  bool is_final = !iree_all_bits_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE);

  // ENOBUFS (RESOURCE_EXHAUSTED) with no MORE flag means the kernel terminated
  // the multishot recv because the provided buffer ring was empty when data
  // arrived. The data is still in the kernel's TCP receive buffer — re-arm the
  // multishot recv so it resumes once buffers are recycled back to the ring.
  // This must be handled before process_recv to avoid setting a sticky error.
  if (is_final && iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED &&
      iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_status_ignore(status);
    memset(recv_op, 0, sizeof(*recv_op));
    iree_async_operation_initialize(
        &recv_op->base, IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL,
        IREE_ASYNC_OPERATION_FLAG_MULTISHOT,
        iree_net_tcp_carrier_recv_completion_multishot, carrier);
    recv_op->socket = carrier->socket;
    recv_op->pool = carrier->recv_pool;
    iree_status_t rearm_status =
        iree_async_proactor_submit_one(carrier->proactor, &recv_op->base);
    if (!iree_status_is_ok(rearm_status)) {
      iree_net_tcp_carrier_set_failure_status(carrier, rearm_status);
      iree_atomic_fetch_sub(&carrier->base.pending_operations, 1,
                            iree_memory_order_release);
      iree_net_tcp_carrier_maybe_complete_deactivation(carrier);
    }
    return;
  }

  // Save status info before process_recv. process_recv takes ownership of
  // |status| when it is non-OK (passes to set_failure_status), so |status|
  // must not be read after this call.
  bool status_ok = iree_status_is_ok(status);

  // Process the received data.
  bool continue_recv = iree_net_tcp_carrier_process_recv(
      carrier, status, recv_op->bytes_received, &recv_op->lease);
  // NOTE: |status| is consumed and must not be used after this point.

  // Release the lease (we've either copied the data or passed it to handler).
  iree_async_buffer_lease_release(&recv_op->lease);

  // Update statistics on success.
  if (status_ok && recv_op->bytes_received > 0) {
    iree_atomic_fetch_add(&carrier->base.bytes_received,
                          (int64_t)recv_op->bytes_received,
                          iree_memory_order_relaxed);
  }

  if (is_final) {
    // Permanent termination (connection reset, EOF, cancelled, etc.).
    iree_atomic_fetch_sub(&carrier->base.pending_operations, 1,
                          iree_memory_order_release);
    iree_net_tcp_carrier_maybe_complete_deactivation(carrier);
  } else if (!continue_recv) {
    // Handler signaled stop (error or application request) but multishot is
    // still armed. Shut down socket read side to cancel the multishot operation
    // and trigger the final completion with an error.
    iree_status_ignore(iree_async_socket_shutdown(
        carrier->socket, IREE_ASYNC_SOCKET_SHUTDOWN_READ));
  }
}

// Submits a single-shot recv operation.
static iree_status_t iree_net_tcp_carrier_submit_single_shot_recv(
    iree_net_tcp_carrier_t* carrier,
    iree_async_socket_recv_pool_operation_t* recv_op);

// Completion callback for single-shot recv operations.
// Re-submits the operation after processing unless stopping.
static void iree_net_tcp_carrier_recv_completion_single_shot(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)flags;  // Single-shot never has MORE flag.
  iree_net_tcp_carrier_t* carrier = (iree_net_tcp_carrier_t*)user_data;
  iree_async_socket_recv_pool_operation_t* recv_op =
      (iree_async_socket_recv_pool_operation_t*)operation;

  // Save status info before process_recv. process_recv takes ownership of
  // |status| when it is non-OK (passes to set_failure_status), so |status|
  // must not be read after this call.
  bool status_ok = iree_status_is_ok(status);

  // Process the received data.
  bool continue_recv = iree_net_tcp_carrier_process_recv(
      carrier, status, recv_op->bytes_received, &recv_op->lease);
  // NOTE: |status| is consumed and must not be used after this point.

  // Release the lease.
  iree_async_buffer_lease_release(&recv_op->lease);

  // Update statistics on success.
  if (status_ok && recv_op->bytes_received > 0) {
    iree_atomic_fetch_add(&carrier->base.bytes_received,
                          (int64_t)recv_op->bytes_received,
                          iree_memory_order_relaxed);
  }

  if (continue_recv) {
    // Re-submit the recv operation.
    iree_status_t submit_status =
        iree_net_tcp_carrier_submit_single_shot_recv(carrier, recv_op);
    if (!iree_status_is_ok(submit_status)) {
      // Failed to resubmit - capture error and decrement active count.
      iree_net_tcp_carrier_set_failure_status(carrier, submit_status);
      int32_t remaining_active =
          iree_atomic_fetch_sub(&carrier->recv.single_shot.active_count, 1,
                                iree_memory_order_release);
      iree_atomic_fetch_sub(&carrier->base.pending_operations, 1,
                            iree_memory_order_release);

      // Check for zombie state: all recv operations failed to resubmit while
      // carrier is still ACTIVE. Shut down socket to prevent silent hang.
      if (remaining_active == 1) {  // We just decremented from 1 to 0.
        iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
        if (state == IREE_NET_CARRIER_STATE_ACTIVE) {
          // No more recv operations and not draining - zombie state.
          // Shut down socket read side to signal the problem to any pending
          // sends and prevent the carrier from appearing healthy.
          iree_status_ignore(iree_async_socket_shutdown(
              carrier->socket, IREE_ASYNC_SOCKET_SHUTDOWN_READ));
        }
      }
      iree_net_tcp_carrier_maybe_complete_deactivation(carrier);
    }
    // On successful resubmit, pending_operations stays the same.
  } else {
    // Not continuing - decrement active count and pending operations.
    iree_atomic_fetch_sub(&carrier->recv.single_shot.active_count, 1,
                          iree_memory_order_release);
    iree_atomic_fetch_sub(&carrier->base.pending_operations, 1,
                          iree_memory_order_release);
    iree_net_tcp_carrier_maybe_complete_deactivation(carrier);
  }
}

// Submits a single-shot recv operation (implementation).
static iree_status_t iree_net_tcp_carrier_submit_single_shot_recv(
    iree_net_tcp_carrier_t* carrier,
    iree_async_socket_recv_pool_operation_t* recv_op) {
  // Initialize the operation for reuse.
  memset(recv_op, 0, sizeof(*recv_op));
  iree_async_operation_initialize(
      &recv_op->base, IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_net_tcp_carrier_recv_completion_single_shot, carrier);
  recv_op->socket = carrier->socket;
  recv_op->pool = carrier->recv_pool;

  return iree_async_proactor_submit_one(carrier->proactor, &recv_op->base);
}

//===----------------------------------------------------------------------===//
// Send completion handler
//===----------------------------------------------------------------------===//

// Completion callback for send operations.
static void iree_net_tcp_carrier_send_completion(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)flags;
  iree_net_tcp_carrier_t* carrier = (iree_net_tcp_carrier_t*)user_data;
  iree_async_socket_send_operation_t* send_op =
      (iree_async_socket_send_operation_t*)operation;

  // Find the slot containing this operation.
  // The slot's operation is at the beginning of the slot struct.
  iree_net_tcp_send_slot_t* slot = (iree_net_tcp_send_slot_t*)send_op;

  // Update statistics on success.
  if (iree_status_is_ok(status)) {
    iree_atomic_fetch_add(&carrier->base.bytes_sent,
                          (int64_t)send_op->bytes_sent,
                          iree_memory_order_relaxed);
  } else {
    // Capture send errors as sticky failure status.
    iree_net_tcp_carrier_set_failure_status(carrier, iree_status_clone(status));
  }

  // Invoke user callback if set.
  if (carrier->base.callback.fn) {
    carrier->base.callback.fn(carrier->base.callback.user_data, slot->user_data,
                              status, send_op->bytes_sent, NULL);
  } else {
    // No callback - we must consume the status.
    iree_status_ignore(status);
  }

  // Release the slot.
  iree_atomic_store(&slot->state, IREE_NET_TCP_SEND_SLOT_STATE_FREE,
                    iree_memory_order_release);

  // Advance tail to make slot available for reuse.
  // Note: We use relaxed ordering here because the slot state store above
  // already has release semantics, and consumers acquire via the state.
  iree_atomic_fetch_add(&carrier->send.tail, 1, iree_memory_order_relaxed);

  // Decrement pending operations.
  iree_atomic_fetch_sub(&carrier->base.pending_operations, 1,
                        iree_memory_order_release);

  // Check for deactivation completion.
  iree_net_tcp_carrier_maybe_complete_deactivation(carrier);
}

//===----------------------------------------------------------------------===//
// Vtable implementations
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_tcp_carrier_deactivate(
    iree_net_carrier_t* base_carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data);

// Final free after all io_uring operations have been cancelled and completed.
static void iree_net_tcp_carrier_free(iree_net_tcp_carrier_t* carrier) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Consume sticky failure status.
  iree_net_tcp_carrier_consume_failure_status(carrier);

  // Release socket - null-safe.
  iree_async_socket_release(carrier->socket);

  // NOTE: recv_pool is not owned by carrier. Caller must ensure it outlives us.

  // Free carrier memory.
  iree_allocator_t allocator = carrier->base.host_allocator;
  iree_allocator_free(allocator, carrier);
  IREE_TRACE_ZONE_END(z0);
}

// Deactivation callback for deferred destroy: called when all pending io_uring
// operations have completed their cancellation CQEs.
static void iree_net_tcp_carrier_deferred_destroy(void* user_data) {
  iree_net_tcp_carrier_t* carrier = (iree_net_tcp_carrier_t*)user_data;
  iree_net_tcp_carrier_free(carrier);
}

static void iree_net_tcp_carrier_destroy(iree_net_carrier_t* base_carrier) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);

  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);

  if (state == IREE_NET_CARRIER_STATE_ACTIVE) {
    // Carrier was never properly deactivated (caller released without
    // shutting down). Start async deactivation to cancel pending io_uring
    // operations (multishot recv, in-flight sends). The carrier stays alive
    // until all cancellation CQEs have been processed, at which point
    // iree_net_tcp_carrier_deferred_destroy fires the actual free.
    //
    // Clear recv handler to prevent delivery of stale data during drain.
    base_carrier->recv_handler.fn = NULL;
    base_carrier->recv_handler.user_data = NULL;
    iree_status_ignore(iree_net_tcp_carrier_deactivate(
        base_carrier, iree_net_tcp_carrier_deferred_destroy, carrier));
    return;
  }

  if (state == IREE_NET_CARRIER_STATE_DRAINING) {
    // Already deactivating — replace the callback so we get notified when
    // draining completes. The previous callback holder has already released.
    carrier->deactivate_callback.fn = iree_net_tcp_carrier_deferred_destroy;
    carrier->deactivate_callback.user_data = carrier;
    return;
  }

  // DEACTIVATED or CREATED — safe to free immediately.
  IREE_ASSERT(state == IREE_NET_CARRIER_STATE_DEACTIVATED ||
              state == IREE_NET_CARRIER_STATE_CREATED);
  iree_net_tcp_carrier_free(carrier);
}

static void iree_net_tcp_carrier_set_recv_handler(
    iree_net_carrier_t* base_carrier, iree_net_carrier_recv_handler_t handler) {
  base_carrier->recv_handler = handler;
}

static iree_status_t iree_net_tcp_carrier_activate(
    iree_net_carrier_t* base_carrier) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify state is CREATED.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_CREATED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in CREATED state to activate");
  }

  // Verify recv handler is set.
  if (!base_carrier->recv_handler.fn) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "recv handler must be set before activation");
  }

  // Transition to ACTIVE state.
  iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_ACTIVE);

  iree_status_t status = iree_ok_status();

  if (carrier->recv.multishot_enabled) {
    // Multishot: submit single recv_pool operation with MULTISHOT flag.
    iree_async_socket_recv_pool_operation_t* recv_op =
        &carrier->recv.multishot.operation;
    memset(recv_op, 0, sizeof(*recv_op));
    iree_async_operation_initialize(
        &recv_op->base, IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL,
        IREE_ASYNC_OPERATION_FLAG_MULTISHOT,
        iree_net_tcp_carrier_recv_completion_multishot, carrier);
    recv_op->socket = carrier->socket;
    recv_op->pool = carrier->recv_pool;

    // Track pending operation before submit.
    iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                          iree_memory_order_relaxed);

    status = iree_async_proactor_submit_one(carrier->proactor, &recv_op->base);
    if (!iree_status_is_ok(status)) {
      // Rollback pending_operations on submit failure.
      iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                            iree_memory_order_relaxed);
    }
  } else {
    // Single-shot: submit all recv operations.
    uint32_t operation_count = carrier->recv.single_shot.operation_count;
    for (uint32_t i = 0; i < operation_count && iree_status_is_ok(status);
         ++i) {
      iree_async_socket_recv_pool_operation_t* recv_op =
          &carrier->recv.single_shot.operations[i];

      // Track pending operation before submit.
      iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                            iree_memory_order_relaxed);
      iree_atomic_fetch_add(&carrier->recv.single_shot.active_count, 1,
                            iree_memory_order_relaxed);

      status = iree_net_tcp_carrier_submit_single_shot_recv(carrier, recv_op);
      if (!iree_status_is_ok(status)) {
        // Rollback counts on submit failure.
        iree_atomic_fetch_sub(&carrier->recv.single_shot.active_count, 1,
                              iree_memory_order_relaxed);
        iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                              iree_memory_order_relaxed);
      }
    }
  }

  // On failure, handle partial activation state.
  if (!iree_status_is_ok(status)) {
    int32_t pending = iree_atomic_load(&base_carrier->pending_operations,
                                       iree_memory_order_acquire);
    if (pending > 0) {
      // Some operations were successfully submitted before failure.
      // Stay in ACTIVE state and shut down socket to trigger their completion.
      // Caller MUST call deactivate() before destroy() to drain operations.
      // Set sticky failure status so subsequent send() calls fail fast.
      iree_net_tcp_carrier_set_failure_status(carrier,
                                              iree_status_clone(status));
      iree_status_ignore(iree_async_socket_shutdown(
          carrier->socket, IREE_ASYNC_SOCKET_SHUTDOWN_READ));
    } else {
      // No operations were submitted - safe to reset to CREATED.
      iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_CREATED);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_net_tcp_carrier_deactivate(
    iree_net_carrier_t* base_carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify state is ACTIVE or CREATED.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE &&
      state != IREE_NET_CARRIER_STATE_CREATED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "carrier must be in ACTIVE or CREATED state to deactivate");
  }

  // Store callback for when deactivation completes.
  carrier->deactivate_callback.fn = callback;
  carrier->deactivate_callback.user_data = user_data;

  // If never activated, skip directly to DEACTIVATED (no async work to drain).
  if (state == IREE_NET_CARRIER_STATE_CREATED) {
    iree_net_carrier_set_state(base_carrier,
                               IREE_NET_CARRIER_STATE_DEACTIVATED);
    if (callback) {
      callback(user_data);
    }
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Transition to DRAINING.
  iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_DRAINING);

  // Cancel recv operations by shutting down the socket read side and explicitly
  // cancelling pending recv operations.
  //
  // On POSIX, shutdown(SHUT_RD) causes pending recv operations to complete with
  // 0 bytes (EOF). On Windows, shutdown(SD_RECEIVE) does NOT cancel pending
  // overlapped WSARecv operations — they remain pending until CancelIoEx() or
  // closesocket(). The explicit cancel calls below dispatch to CancelIoEx() on
  // IOCP, ensuring recv operations complete on all platforms.
  //
  // We keep the shutdown call as well: on POSIX it provides the primary
  // cancellation signal, and on Windows it prevents new data from arriving
  // (which could otherwise wake a recv that was just being cancelled).
  iree_status_ignore(iree_async_socket_shutdown(
      carrier->socket, IREE_ASYNC_SOCKET_SHUTDOWN_READ));

  // Explicitly cancel pending recv operations. Safe to call on operations that
  // already completed (returns NOT_FOUND, which we ignore).
  if (carrier->recv.multishot_enabled) {
    iree_status_ignore(iree_async_proactor_cancel(
        carrier->proactor, &carrier->recv.multishot.operation.base));
  } else {
    uint32_t operation_count = carrier->recv.single_shot.operation_count;
    for (uint32_t i = 0; i < operation_count; ++i) {
      iree_status_ignore(iree_async_proactor_cancel(
          carrier->proactor, &carrier->recv.single_shot.operations[i].base));
    }
  }

  // Check if we can complete immediately (no pending operations).
  iree_net_tcp_carrier_maybe_complete_deactivation(carrier);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_net_carrier_send_budget_t iree_net_tcp_carrier_query_send_budget(
    iree_net_carrier_t* base_carrier) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);

  // Load head (relaxed - just an estimate).
  uint32_t head =
      iree_atomic_load(&carrier->send.head, iree_memory_order_relaxed);
  // Load tail (acquire - synchronize with completion releases).
  uint32_t tail =
      iree_atomic_load(&carrier->send.tail, iree_memory_order_acquire);

  // Available slots = total - in_flight.
  // Unsigned arithmetic gives well-defined wraparound behavior.
  uint32_t in_flight = head - tail;
  uint32_t available = carrier->send.slot_count - in_flight;

  iree_net_carrier_send_budget_t budget;
  budget.slots = available;
  // For TCP, byte budget is effectively unlimited (kernel handles buffering).
  // Use SIZE_MAX to indicate "no byte limit, limited by slots only".
  budget.bytes = IREE_HOST_SIZE_MAX;
  return budget;
}

static iree_status_t iree_net_tcp_carrier_send(
    iree_net_carrier_t* base_carrier, const iree_net_send_params_t* params) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);

  // Validate scatter-gather count early (doesn't need pending_operations).
  if (params->data.count > IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "too many scatter-gather buffers: %" PRIhsz " > %d",
                            params->data.count,
                            IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS);
  }

  // Calculate total size and reject empty sends.
  iree_host_size_t total_size = 0;
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    total_size += params->data.values[i].length;
  }
  if (total_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty sends are not allowed");
  }

  // Increment pending_operations FIRST to prevent TOCTOU race with deactivate.
  // This ensures deactivate sees our operation before completing.
  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);

  // Now verify state is ACTIVE. If not, rollback and return error.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_tcp_carrier_maybe_complete_deactivation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE state to send");
  }

  // Check for sticky failure status.
  iree_status_t failure = iree_net_tcp_carrier_get_failure_status(carrier);
  if (!iree_status_is_ok(failure)) {
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_tcp_carrier_maybe_complete_deactivation(carrier);
    return failure;
  }

  // Claim a send slot using CAS loop.
  // We atomically increment head, then check if slot is available.
  uint32_t slot_index;
  for (;;) {
    uint32_t head =
        iree_atomic_load(&carrier->send.head, iree_memory_order_relaxed);
    uint32_t tail =
        iree_atomic_load(&carrier->send.tail, iree_memory_order_acquire);

    // Check if slots are available.
    // Unsigned arithmetic gives well-defined wraparound behavior.
    uint32_t in_flight = head - tail;
    if (in_flight >= carrier->send.slot_count) {
      // Rollback pending_operations increment.
      iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                            iree_memory_order_release);
      iree_net_tcp_carrier_maybe_complete_deactivation(carrier);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "no send slots available");
    }

    // Try to claim the next slot.
    if (iree_atomic_compare_exchange_strong(&carrier->send.head, &head,
                                            head + 1, iree_memory_order_acq_rel,
                                            iree_memory_order_relaxed)) {
      slot_index = head & carrier->send.slot_mask;
      break;
    }
    // CAS failed - another thread claimed the slot, retry.
  }

  // Get the slot and initialize the operation.
  iree_net_tcp_send_slot_t* slot = &carrier->send.slots[slot_index];

  // Verify slot is free (debug check).
  IREE_ASSERT(iree_atomic_load(&slot->state, iree_memory_order_acquire) ==
              IREE_NET_TCP_SEND_SLOT_STATE_FREE);

  // Mark slot as in use.
  iree_atomic_store(&slot->state, IREE_NET_TCP_SEND_SLOT_STATE_IN_USE,
                    iree_memory_order_release);

  // Store user data for completion callback.
  slot->user_data = params->user_data;

  // Initialize the send operation.
  iree_async_socket_send_operation_t* send_op = &slot->operation;
  memset(send_op, 0, sizeof(*send_op));
  iree_async_operation_initialize(
      &send_op->base, IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_tcp_carrier_send_completion,
      carrier);
  send_op->socket = carrier->socket;
  send_op->send_flags = IREE_ASYNC_SOCKET_SEND_FLAG_NONE;

  // Copy span list into slot-local storage. The caller's span array and the
  // data behind unregistered spans may be stack-allocated. Under io_uring,
  // the kernel reads iovec data at io_uring_enter time (during poll), which
  // is after the caller returns. We must ensure all references are stable.
  memcpy(slot->inline_spans, params->data.values,
         params->data.count * sizeof(iree_async_span_t));
  send_op->buffers.values = slot->inline_spans;
  send_op->buffers.count = params->data.count;

  // Copy small unregistered span data into slot-local inline storage.
  // Unregistered spans (region == NULL) store raw pointers via
  // iree_async_span_from_ptr that may reference transient caller memory.
  // Registered spans have managed lifetime through the region and are
  // left as-is.
  iree_host_size_t inline_data_offset = 0;
  for (iree_host_size_t i = 0; i < send_op->buffers.count; ++i) {
    if (slot->inline_spans[i].region != NULL) continue;
    iree_host_size_t length = slot->inline_spans[i].length;
    if (length == 0) continue;
    if (inline_data_offset + length >
        IREE_NET_TCP_SEND_SLOT_INLINE_DATA_CAPACITY) {
      // Data exceeds inline capacity. The caller is responsible for keeping
      // unregistered data alive until send completion. This is only safe when
      // the data is in stable memory (heap, static, or a stack frame that
      // won't unwind before poll). Large stack-allocated sends are a bug.
      break;
    }
    void* source = iree_async_span_ptr(slot->inline_spans[i]);
    memcpy(slot->inline_data + inline_data_offset, source, length);
    slot->inline_spans[i] = iree_async_span_from_ptr(
        slot->inline_data + inline_data_offset, length);
    inline_data_offset += length;
  }

  // Submit to proactor.
  iree_status_t status =
      iree_async_proactor_submit_one(carrier->proactor, &send_op->base);

  if (!iree_status_is_ok(status)) {
    // Rollback: release slot, advance tail.
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_atomic_store(&slot->state, IREE_NET_TCP_SEND_SLOT_STATE_FREE,
                      iree_memory_order_release);
    // Advance tail to keep ring buffer accounting consistent.
    // Without this, (head - tail) would permanently overcount by 1 per failure.
    iree_atomic_fetch_add(&carrier->send.tail, 1, iree_memory_order_release);
    // Check if deactivation was waiting for this operation.
    iree_net_tcp_carrier_maybe_complete_deactivation(carrier);
    return status;
  }

  return iree_ok_status();
}

static iree_status_t iree_net_tcp_carrier_shutdown(
    iree_net_carrier_t* base_carrier) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);

  // Shutdown is valid in ACTIVE or DRAINING state.
  // In ACTIVE: normal graceful close initiation.
  // In DRAINING: shutdown write side while waiting for recv to drain.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE &&
      state != IREE_NET_CARRIER_STATE_DRAINING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE or DRAINING state to "
                            "shutdown");
  }

  // Shut down the write side of the socket. This sends FIN to the peer,
  // causing their recv to return EOF. The carrier can still receive data
  // until deactivate is called.
  return iree_async_socket_shutdown(carrier->socket,
                                    IREE_ASYNC_SOCKET_SHUTDOWN_WRITE);
}

// Vtable for TCP carrier. RDMA operations are not supported.
static const iree_net_carrier_vtable_t iree_net_tcp_carrier_vtable = {
    .destroy = iree_net_tcp_carrier_destroy,
    .set_recv_handler = iree_net_tcp_carrier_set_recv_handler,
    .activate = iree_net_tcp_carrier_activate,
    .deactivate = iree_net_tcp_carrier_deactivate,
    .query_send_budget = iree_net_tcp_carrier_query_send_budget,
    .send = iree_net_tcp_carrier_send,
    .shutdown = iree_net_tcp_carrier_shutdown,
    .direct_write = NULL,  // TCP does not support RDMA.
    .direct_read = NULL,
    .register_buffer = NULL,
    .unregister_buffer = NULL,
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_tcp_carrier_allocate(
    iree_async_proactor_t* proactor, iree_async_socket_t* socket,
    iree_async_buffer_pool_t* recv_pool, iree_net_tcp_carrier_options_t options,
    iree_net_carrier_callback_t callback, iree_allocator_t host_allocator,
    iree_net_carrier_t** out_carrier) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(socket);
  IREE_ASSERT_ARGUMENT(recv_pool);
  IREE_ASSERT_ARGUMENT(out_carrier);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_carrier = NULL;

  // Validate options.
  if (!iree_is_power_of_two_uint64(options.send_slot_count)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send_slot_count must be power of 2, got %" PRIu32,
                            options.send_slot_count);
  }
  if (!iree_is_power_of_two_uint64(options.single_shot_recv_count)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "single_shot_recv_count must be power of 2, got %" PRIu32,
        options.single_shot_recv_count);
  }

  // Query proactor capabilities.
  iree_async_proactor_capabilities_t capabilities =
      iree_async_proactor_query_capabilities(proactor);
  bool use_multishot =
      options.prefer_multishot_recv &&
      iree_any_bit_set(capabilities, IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT);
  bool use_zero_copy_send =
      options.prefer_zero_copy_send &&
      iree_any_bit_set(capabilities,
                       IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND);

  // Compute allocation size with overflow checking.
  // For multishot, recv_operation_count is 0 so that array contributes nothing.
  iree_host_size_t recv_operation_count =
      use_multishot ? 0 : options.single_shot_recv_count;
  iree_host_size_t total_size = 0;
  iree_host_size_t send_slots_offset = 0;
  iree_host_size_t recv_operations_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_net_tcp_carrier_t), &total_size,
              IREE_STRUCT_FIELD(options.send_slot_count,
                                iree_net_tcp_send_slot_t, &send_slots_offset),
              IREE_STRUCT_FIELD(recv_operation_count,
                                iree_async_socket_recv_pool_operation_t,
                                &recv_operations_offset)));

  // Allocate carrier.
  iree_net_tcp_carrier_t* carrier = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&carrier));
  memset(carrier, 0, total_size);

  // Compute capabilities.
  iree_net_carrier_capabilities_t carrier_capabilities =
      IREE_NET_CARRIER_CAPABILITY_RELIABLE |
      IREE_NET_CARRIER_CAPABILITY_ORDERED;
  if (use_zero_copy_send) {
    carrier_capabilities |= IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_TX;
  }
  if (use_multishot) {
    carrier_capabilities |= IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_RX;
  }

  // Initialize base carrier.
  iree_net_carrier_initialize(&iree_net_tcp_carrier_vtable,
                              carrier_capabilities,
                              /*mtu=*/0,  // Stream carrier, no MTU.
                              IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS, callback,
                              host_allocator, &carrier->base);

  // Initialize TCP carrier fields.
  carrier->proactor = proactor;
  carrier->socket = socket;
  carrier->recv_pool = recv_pool;
  // NOTE: recv_pool is not ref-counted. Caller must ensure it outlives carrier.

  // Initialize send slot ring.
  carrier->send.slot_count = options.send_slot_count;
  carrier->send.slot_mask = options.send_slot_count - 1;
  iree_atomic_store(&carrier->send.head, 0, iree_memory_order_relaxed);
  iree_atomic_store(&carrier->send.tail, 0, iree_memory_order_relaxed);

  // Point to trailing send slot storage.
  carrier->send.slots =
      (iree_net_tcp_send_slot_t*)((uint8_t*)carrier + send_slots_offset);

  // Initialize send slots to FREE state.
  for (uint32_t i = 0; i < options.send_slot_count; ++i) {
    iree_atomic_store(&carrier->send.slots[i].state,
                      IREE_NET_TCP_SEND_SLOT_STATE_FREE,
                      iree_memory_order_relaxed);
  }

  // Initialize recv operations.
  carrier->recv.multishot_enabled = use_multishot;
  if (!use_multishot) {
    carrier->recv.single_shot.operation_count = options.single_shot_recv_count;
    carrier->recv.single_shot.operations =
        (iree_async_socket_recv_pool_operation_t*)((uint8_t*)carrier +
                                                   recv_operations_offset);
    iree_atomic_store(&carrier->recv.single_shot.active_count, 0,
                      iree_memory_order_relaxed);
  }

  // Initialize error handling.
  iree_atomic_store(&carrier->failure_status, 0, iree_memory_order_relaxed);

  // Initialize deactivate callback to empty.
  carrier->deactivate_callback.fn = NULL;
  carrier->deactivate_callback.user_data = NULL;

  *out_carrier = &carrier->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
