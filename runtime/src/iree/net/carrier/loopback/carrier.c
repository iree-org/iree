// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/loopback/carrier.h"

#include "iree/async/operations/scheduling.h"
#include "iree/base/internal/math.h"

// Number of concurrent send operations per carrier. Must be power of 2
// and at most 32 (tracked by a uint32_t bitmap).
#define IREE_NET_LOOPBACK_SEND_SLOT_COUNT 32
static_assert(IREE_NET_LOOPBACK_SEND_SLOT_COUNT <= 32,
              "send slot count exceeds uint32_t bitmap capacity");

//===----------------------------------------------------------------------===//
// Send slot
//===----------------------------------------------------------------------===//

// A pending send operation awaiting delivery during the next poll() cycle.
// Each slot holds a NOP operation that, when completed by the proactor,
// triggers data delivery to the peer's recv handler.
typedef struct iree_net_loopback_send_slot_t {
  // NOP operation submitted to the proactor. Must be the first field so that
  // the completion callback can cast (iree_net_loopback_send_slot_t*)operation.
  iree_async_nop_operation_t nop;

  // Data to deliver to the peer's recv handler when the NOP completes.
  // Always points at coalesce_buffer (data is copied during send).
  iree_async_span_t delivery_span;

  // Heap-allocated copy of the sender's data. Owned by this slot and freed
  // in the NOP completion callback.
  uint8_t* coalesce_buffer;

  // Total byte count for statistics and completion callback.
  iree_host_size_t total_size;

  // User data from iree_net_send_params_t, echoed to send completion callback.
  uint64_t user_data;
} iree_net_loopback_send_slot_t;

//===----------------------------------------------------------------------===//
// Loopback carrier structure
//===----------------------------------------------------------------------===//

typedef struct iree_net_loopback_carrier_t {
  // Base carrier (must be first for safe upcasting).
  iree_net_carrier_t base;

  // Proactor for async delivery via NOP operations. Retained.
  iree_async_proactor_t* proactor;

  // Peer carrier (the other end of the pair). Not retained.
  // Set at pair creation, cleared on deactivate or peer destruction.
  struct iree_net_loopback_carrier_t* peer;

  // True if shutdown() was called (future sends fail).
  bool shutdown_initiated;

  // Send slot bitmap. Bit i set = slot i is free.
  // Claim: find first set bit (ctz), CAS-clear it.
  // Release: atomic OR to set bit.
  // No ordering dependency between slots — out-of-order completion is correct.
  struct {
    uint32_t slot_count;
    iree_atomic_uint32_t free_bitmap;
    iree_net_loopback_send_slot_t* slots;  // Points into trailing allocation.
  } send;

  // Deactivation callback (stored during deactivate, fired when drained).
  struct {
    iree_net_carrier_deactivate_callback_fn_t fn;
    void* user_data;
  } deactivate_callback;

  // Handler invoked when the peer carrier disconnects (deactivates or is
  // destroyed). Set by the endpoint adapter to propagate transport errors.
  // This provides the loopback equivalent of TCP's ECONNRESET notification.
  iree_net_loopback_carrier_disconnect_handler_t peer_disconnect_handler;
} iree_net_loopback_carrier_t;

static inline iree_net_loopback_carrier_t* iree_net_loopback_carrier_cast(
    iree_net_carrier_t* base_carrier) {
  return (iree_net_loopback_carrier_t*)base_carrier;
}

//===----------------------------------------------------------------------===//
// Forward declarations
//===----------------------------------------------------------------------===//

static void iree_net_loopback_carrier_destroy(iree_net_carrier_t* base_carrier);
static void iree_net_loopback_carrier_set_recv_handler(
    iree_net_carrier_t* base_carrier, iree_net_carrier_recv_handler_t handler);
static iree_status_t iree_net_loopback_carrier_activate(
    iree_net_carrier_t* base_carrier);
static iree_status_t iree_net_loopback_carrier_deactivate(
    iree_net_carrier_t* base_carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data);
static iree_net_carrier_send_budget_t
iree_net_loopback_carrier_query_send_budget(iree_net_carrier_t* base_carrier);
static iree_status_t iree_net_loopback_carrier_send(
    iree_net_carrier_t* base_carrier, const iree_net_send_params_t* params);
static iree_status_t iree_net_loopback_carrier_begin_send(
    iree_net_carrier_t* base_carrier, iree_host_size_t size, void** out_ptr,
    iree_net_carrier_send_handle_t* out_handle);
static iree_status_t iree_net_loopback_carrier_commit_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle);
static void iree_net_loopback_carrier_abort_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle);
static iree_status_t iree_net_loopback_carrier_shutdown(
    iree_net_carrier_t* base_carrier);
static void iree_net_loopback_carrier_maybe_complete_deactivation(
    iree_net_loopback_carrier_t* carrier);

//===----------------------------------------------------------------------===//
// Deactivation completion check
//===----------------------------------------------------------------------===//

// Checks if deactivation has completed and invokes callback if so.
// Called after every pending_operations decrement.
static void iree_net_loopback_carrier_maybe_complete_deactivation(
    iree_net_loopback_carrier_t* carrier) {
  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  if (state != IREE_NET_CARRIER_STATE_DRAINING) return;

  int32_t pending = iree_atomic_load(&carrier->base.pending_operations,
                                     iree_memory_order_acquire);
  if (pending > 0) return;

  iree_net_carrier_set_state(&carrier->base,
                             IREE_NET_CARRIER_STATE_DEACTIVATED);
  if (carrier->deactivate_callback.fn) {
    carrier->deactivate_callback.fn(carrier->deactivate_callback.user_data);
  }
}

//===----------------------------------------------------------------------===//
// Peer disconnect notification
//===----------------------------------------------------------------------===//

// Deferred notification delivered to the surviving peer when the other side
// of a loopback pair deactivates or is destroyed. The NOP fires on the next
// proactor poll cycle, invoking the peer's disconnect handler.
typedef struct iree_net_loopback_disconnect_notify_t {
  iree_async_nop_operation_t nop;
  iree_net_loopback_carrier_t* peer;  // Retained.
} iree_net_loopback_disconnect_notify_t;

static void iree_net_loopback_carrier_disconnect_notify_completion(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)user_data;
  (void)flags;
  iree_status_ignore(status);
  iree_net_loopback_disconnect_notify_t* notify =
      (iree_net_loopback_disconnect_notify_t*)operation;
  iree_net_loopback_carrier_t* peer = notify->peer;
  iree_allocator_free(peer->base.host_allocator, notify);

  // Fire the handler if the peer hasn't already been deactivated.
  iree_net_carrier_state_t state = iree_net_carrier_state(&peer->base);
  if (state != IREE_NET_CARRIER_STATE_DEACTIVATED &&
      peer->peer_disconnect_handler.fn) {
    peer->peer_disconnect_handler.fn(
        peer->peer_disconnect_handler.user_data,
        iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected"));
  }

  iree_net_carrier_release(&peer->base);
}

// Notifies the surviving peer that this carrier is departing. Submits a NOP
// to deliver the notification asynchronously. Must be called BEFORE clearing
// the peer link (carrier->peer).
static void iree_net_loopback_carrier_notify_peer_disconnect(
    iree_net_loopback_carrier_t* carrier) {
  iree_net_loopback_carrier_t* peer = carrier->peer;
  if (!peer || !peer->peer_disconnect_handler.fn) return;

  // Don't notify if peer is already shutting down.
  iree_net_carrier_state_t peer_state = iree_net_carrier_state(&peer->base);
  if (peer_state == IREE_NET_CARRIER_STATE_DEACTIVATED) return;

  iree_net_loopback_disconnect_notify_t* notify = NULL;
  iree_status_t status = iree_allocator_malloc(
      peer->base.host_allocator, sizeof(*notify), (void**)&notify);
  if (iree_status_is_ok(status)) {
    memset(notify, 0, sizeof(*notify));
    notify->peer = peer;
    iree_net_carrier_retain(&peer->base);
    iree_async_operation_initialize(
        &notify->nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_net_loopback_carrier_disconnect_notify_completion, notify);
    status = iree_async_proactor_submit_one(peer->proactor, &notify->nop.base);
    if (!iree_status_is_ok(status)) {
      iree_net_carrier_release(&peer->base);
      iree_allocator_free(peer->base.host_allocator, notify);
    }
  }
  if (!iree_status_is_ok(status)) {
    // Synchronous fallback on OOM or submit failure. Safe because the handler
    // operates on the surviving peer, not the carrier being torn down.
    iree_status_ignore(status);
    peer->peer_disconnect_handler.fn(
        peer->peer_disconnect_handler.user_data,
        iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected"));
  }
}

//===----------------------------------------------------------------------===//
// NOP completion callback
//===----------------------------------------------------------------------===//

// Fires from within iree_async_proactor_poll() on the proactor thread.
// Delivers data to the peer's recv handler and fires the sender's send
// completion callback.
static void iree_net_loopback_carrier_nop_completion(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)flags;  // NOP is never multishot.
  iree_net_loopback_carrier_t* carrier =
      (iree_net_loopback_carrier_t*)user_data;
  iree_net_loopback_send_slot_t* slot =
      (iree_net_loopback_send_slot_t*)operation;

  // Consume NOP status (always OK for NOP, but be correct about ownership).
  iree_status_ignore(status);

  // Deliver data to peer's recv handler if peer is still alive and active.
  // If the peer departed between send() and this completion (deactivated or
  // destroyed while NOP was in flight), report an error through the sender's
  // completion callback. This mirrors TCP/SHM carriers where the OS reports
  // EPIPE/ECONNRESET when the peer closes the connection.
  //
  // The recv handler may trigger peer destruction (e.g., GOAWAY processing
  // destroys the peer's session, which deactivates and releases the peer
  // carrier). Retain the peer so the statistics update below doesn't access
  // freed memory.
  iree_status_t delivery_status = iree_ok_status();
  iree_net_loopback_carrier_t* peer = carrier->peer;
  if (peer) iree_net_carrier_retain(&peer->base);
  if (peer &&
      iree_net_carrier_state(&peer->base) == IREE_NET_CARRIER_STATE_ACTIVE &&
      peer->base.recv_handler.fn) {
    delivery_status = peer->base.recv_handler.fn(
        peer->base.recv_handler.user_data, slot->delivery_span, NULL);
  } else {
    delivery_status =
        iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected");
  }

  // Free coalesce buffer if allocated (scatter-gather sends).
  iree_allocator_free(carrier->base.host_allocator, slot->coalesce_buffer);
  slot->coalesce_buffer = NULL;

  // Update statistics on successful delivery.
  if (iree_status_is_ok(delivery_status)) {
    iree_atomic_fetch_add(&carrier->base.bytes_sent, (int64_t)slot->total_size,
                          iree_memory_order_relaxed);
    if (peer) {
      iree_atomic_fetch_add(&peer->base.bytes_received,
                            (int64_t)slot->total_size,
                            iree_memory_order_relaxed);
    }
  }

  // Release peer ref acquired above.
  if (peer) iree_net_carrier_release(&peer->base);

  // Fire sender's send completion callback if set.
  if (carrier->base.callback.fn) {
    carrier->base.callback.fn(carrier->base.callback.user_data, slot->user_data,
                              delivery_status, slot->total_size, NULL);
  } else {
    iree_status_ignore(delivery_status);
  }

  // Release send slot by setting its bit in the free bitmap.
  int slot_index =
      (int)((iree_net_loopback_send_slot_t*)operation - carrier->send.slots);
  iree_atomic_fetch_or(&carrier->send.free_bitmap, (uint32_t)1 << slot_index,
                       iree_memory_order_release);

  // Decrement pending operations.
  iree_atomic_fetch_sub(&carrier->base.pending_operations, 1,
                        iree_memory_order_release);

  // Check for deactivation completion.
  iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
}

//===----------------------------------------------------------------------===//
// Destroy
//===----------------------------------------------------------------------===//

static void iree_net_loopback_carrier_destroy(
    iree_net_carrier_t* base_carrier) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Assert state is DEACTIVATED or CREATED (never activated).
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  IREE_ASSERT(state == IREE_NET_CARRIER_STATE_DEACTIVATED ||
              state == IREE_NET_CARRIER_STATE_CREATED);

  // Notify surviving peer before clearing the link. This delivers a transport
  // error to the peer's session so it can transition to ERROR state and be
  // cleaned up by the server.
  iree_net_loopback_carrier_notify_peer_disconnect(carrier);

  // Clear peer link if still set (peer may have already been destroyed).
  if (carrier->peer) {
    carrier->peer->peer = NULL;
    carrier->peer = NULL;
  }

  // Defensive cleanup of any remaining coalesce buffers in slots.
  for (uint32_t i = 0; i < carrier->send.slot_count; ++i) {
    iree_allocator_free(carrier->base.host_allocator,
                        carrier->send.slots[i].coalesce_buffer);
  }

  // Release proactor reference.
  iree_async_proactor_release(carrier->proactor);

  // Free carrier memory (single allocation includes trailing slots).
  iree_allocator_t allocator = carrier->base.host_allocator;
  iree_allocator_free(allocator, carrier);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Recv handler
//===----------------------------------------------------------------------===//

static void iree_net_loopback_carrier_set_recv_handler(
    iree_net_carrier_t* base_carrier, iree_net_carrier_recv_handler_t handler) {
  base_carrier->recv_handler = handler;
}

//===----------------------------------------------------------------------===//
// Activate / Deactivate
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_loopback_carrier_activate(
    iree_net_carrier_t* base_carrier) {
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

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_net_loopback_carrier_deactivate(
    iree_net_carrier_t* base_carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);
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

  // Store callback for deferred invocation when all operations drain.
  carrier->deactivate_callback.fn = callback;
  carrier->deactivate_callback.user_data = user_data;

  // Notify surviving peer before clearing the link. This delivers a transport
  // error so the peer's session can detect the disconnect.
  iree_net_loopback_carrier_notify_peer_disconnect(carrier);

  // Clear peer link so in-flight NOP completions skip delivery and new sends
  // fail immediately.
  if (carrier->peer) {
    carrier->peer->peer = NULL;
    carrier->peer = NULL;
  }

  // If never activated, transition directly to DEACTIVATED.
  if (state == IREE_NET_CARRIER_STATE_CREATED) {
    iree_net_carrier_set_state(base_carrier,
                               IREE_NET_CARRIER_STATE_DEACTIVATED);
    if (callback) callback(user_data);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Transition to DRAINING. In-flight NOP completions will check this state.
  iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_DRAINING);

  // If no pending operations, complete immediately.
  iree_net_loopback_carrier_maybe_complete_deactivation(carrier);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Send budget
//===----------------------------------------------------------------------===//

static iree_net_carrier_send_budget_t
iree_net_loopback_carrier_query_send_budget(iree_net_carrier_t* base_carrier) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);

  // If peer is gone, no budget available.
  if (!carrier->peer) {
    iree_net_carrier_send_budget_t budget = {0, 0};
    return budget;
  }

  // Available slots = popcount of free bitmap.
  uint32_t bitmap =
      iree_atomic_load(&carrier->send.free_bitmap, iree_memory_order_acquire);
  uint32_t available = iree_math_count_ones_u32(bitmap);

  iree_net_carrier_send_budget_t budget;
  budget.slots = available;
  budget.bytes = available > 0 ? IREE_HOST_SIZE_MAX : 0;
  return budget;
}

//===----------------------------------------------------------------------===//
// Send
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_loopback_carrier_send(
    iree_net_carrier_t* base_carrier, const iree_net_send_params_t* params) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);

  // Calculate total size and reject empty sends (before pending_operations
  // increment to avoid unnecessary rollback).
  iree_host_size_t total_size = 0;
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    total_size += params->data.values[i].length;
  }
  if (total_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty sends are not allowed");
  }

  // Increment pending_operations FIRST to prevent TOCTOU race with deactivate.
  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);

  // Verify state is ACTIVE after incrementing.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE state to send");
  }

  // Check if shutdown was initiated.
  if (carrier->shutdown_initiated) {
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier has been shut down for sending");
  }

  // Check peer is still connected.
  if (!carrier->peer) {
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected");
  }

  // Claim a free send slot from the bitmap.
  uint32_t slot_index;
  uint32_t bitmap =
      iree_atomic_load(&carrier->send.free_bitmap, iree_memory_order_acquire);
  for (;;) {
    if (bitmap == 0) {
      iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                            iree_memory_order_release);
      iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "no send slots available");
    }
    slot_index = (uint32_t)iree_math_count_trailing_zeros_u32(bitmap);
    uint32_t cleared = bitmap & ~((uint32_t)1 << slot_index);
    if (iree_atomic_compare_exchange_weak(&carrier->send.free_bitmap, &bitmap,
                                          cleared, iree_memory_order_acq_rel,
                                          iree_memory_order_acquire)) {
      break;
    }
  }

  // Get the slot and populate it.
  iree_net_loopback_send_slot_t* slot = &carrier->send.slots[slot_index];

  // Copy data into a carrier-owned buffer. This ensures the sender's buffer
  // can be freed immediately after send() returns. Loopback delivers
  // asynchronously via the proactor's progress callback, so the original
  // span data is not available at delivery time.
  iree_status_t alloc_status = iree_allocator_malloc(
      carrier->base.host_allocator, total_size, (void**)&slot->coalesce_buffer);
  if (!iree_status_is_ok(alloc_status)) {
    iree_atomic_fetch_or(&carrier->send.free_bitmap, (uint32_t)1 << slot_index,
                         iree_memory_order_release);
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
    return alloc_status;
  }
  uint8_t* write_ptr = slot->coalesce_buffer;
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    memcpy(write_ptr, iree_async_span_ptr(params->data.values[i]),
           params->data.values[i].length);
    write_ptr += params->data.values[i].length;
  }
  slot->delivery_span =
      iree_async_span_from_ptr(slot->coalesce_buffer, total_size);

  slot->total_size = total_size;
  slot->user_data = params->user_data;

  // Initialize NOP operation.
  memset(&slot->nop, 0, sizeof(slot->nop));
  iree_async_operation_initialize(
      &slot->nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_loopback_carrier_nop_completion,
      carrier);

  // Submit NOP to proactor. Completes on the next poll() cycle.
  iree_status_t submit_status =
      iree_async_proactor_submit_one(carrier->proactor, &slot->nop.base);
  if (!iree_status_is_ok(submit_status)) {
    // Rollback everything.
    iree_allocator_free(carrier->base.host_allocator, slot->coalesce_buffer);
    slot->coalesce_buffer = NULL;
    iree_atomic_fetch_or(&carrier->send.free_bitmap, (uint32_t)1 << slot_index,
                         iree_memory_order_release);
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
    return submit_status;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Begin/Commit/Abort Send
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_loopback_carrier_begin_send(
    iree_net_carrier_t* base_carrier, iree_host_size_t size, void** out_ptr,
    iree_net_carrier_send_handle_t* out_handle) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);

  // Reject empty sends.
  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty sends are not allowed");
  }

  // Increment pending_operations FIRST to prevent TOCTOU race with deactivate.
  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);

  // Verify state is ACTIVE after incrementing.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE state to send");
  }

  // Check if shutdown was initiated.
  if (carrier->shutdown_initiated) {
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier has been shut down for sending");
  }

  // Check peer is still connected.
  if (!carrier->peer) {
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected");
  }

  // Claim a free send slot from the bitmap.
  uint32_t slot_index;
  uint32_t bitmap =
      iree_atomic_load(&carrier->send.free_bitmap, iree_memory_order_acquire);
  for (;;) {
    if (bitmap == 0) {
      iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                            iree_memory_order_release);
      iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "no send slots available");
    }
    slot_index = (uint32_t)iree_math_count_trailing_zeros_u32(bitmap);
    uint32_t cleared = bitmap & ~((uint32_t)1 << slot_index);
    if (iree_atomic_compare_exchange_weak(&carrier->send.free_bitmap, &bitmap,
                                          cleared, iree_memory_order_acq_rel,
                                          iree_memory_order_acquire)) {
      break;
    }
  }

  // Allocate coalesce buffer for the caller to write into.
  iree_net_loopback_send_slot_t* slot = &carrier->send.slots[slot_index];
  iree_status_t alloc_status = iree_allocator_malloc(
      carrier->base.host_allocator, size, (void**)&slot->coalesce_buffer);
  if (!iree_status_is_ok(alloc_status)) {
    iree_atomic_fetch_or(&carrier->send.free_bitmap, (uint32_t)1 << slot_index,
                         iree_memory_order_release);
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
    return alloc_status;
  }

  // Store size in the slot for commit_send. The handle carries only the slot
  // index — packing size into the handle truncates at 32 bits.
  slot->total_size = size;
  *out_ptr = slot->coalesce_buffer;
  *out_handle = (iree_net_carrier_send_handle_t)slot_index;
  return iree_ok_status();
}

static iree_status_t iree_net_loopback_carrier_commit_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);

  uint32_t slot_index = (uint32_t)handle;

  // Read size from the slot (stored by begin_send).
  iree_net_loopback_send_slot_t* slot = &carrier->send.slots[slot_index];
  iree_host_size_t size = slot->total_size;
  slot->delivery_span = iree_async_span_from_ptr(slot->coalesce_buffer, size);
  slot->user_data = 0;

  // Initialize NOP operation.
  memset(&slot->nop, 0, sizeof(slot->nop));
  iree_async_operation_initialize(
      &slot->nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_loopback_carrier_nop_completion,
      carrier);

  // Submit NOP to proactor. Completes on the next poll() cycle.
  iree_status_t submit_status =
      iree_async_proactor_submit_one(carrier->proactor, &slot->nop.base);
  if (!iree_status_is_ok(submit_status)) {
    iree_allocator_free(carrier->base.host_allocator, slot->coalesce_buffer);
    slot->coalesce_buffer = NULL;
    iree_atomic_fetch_or(&carrier->send.free_bitmap, (uint32_t)1 << slot_index,
                         iree_memory_order_release);
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
    return submit_status;
  }

  return iree_ok_status();
}

static void iree_net_loopback_carrier_abort_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);

  uint32_t slot_index = (uint32_t)handle;
  iree_net_loopback_send_slot_t* slot = &carrier->send.slots[slot_index];

  // Free the coalesce buffer allocated in begin_send.
  iree_allocator_free(carrier->base.host_allocator, slot->coalesce_buffer);
  slot->coalesce_buffer = NULL;

  // Release send slot by setting its bit in the free bitmap.
  iree_atomic_fetch_or(&carrier->send.free_bitmap, (uint32_t)1 << slot_index,
                       iree_memory_order_release);

  // Decrement pending operations.
  iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                        iree_memory_order_release);

  // Check for deactivation completion.
  iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
}

//===----------------------------------------------------------------------===//
// RDMA stubs (unsupported)
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_loopback_carrier_direct_write(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_write_params_t* params) {
  (void)base_carrier;
  (void)params;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "loopback carrier does not support direct_write");
}

static iree_status_t iree_net_loopback_carrier_direct_read(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_read_params_t* params) {
  (void)base_carrier;
  (void)params;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "loopback carrier does not support direct_read");
}

static iree_status_t iree_net_loopback_carrier_register_buffer(
    iree_net_carrier_t* base_carrier, iree_async_region_t* region,
    iree_net_remote_handle_t* out_handle) {
  (void)base_carrier;
  (void)region;
  (void)out_handle;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "loopback carrier does not support register_buffer");
}

static void iree_net_loopback_carrier_unregister_buffer(
    iree_net_carrier_t* base_carrier, iree_net_remote_handle_t handle) {
  (void)base_carrier;
  (void)handle;
}

//===----------------------------------------------------------------------===//
// Shutdown
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_loopback_carrier_shutdown(
    iree_net_carrier_t* base_carrier) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);

  // Verify state is ACTIVE.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE state to shutdown");
  }

  // Mark shutdown initiated — future sends will fail.
  carrier->shutdown_initiated = true;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_net_carrier_vtable_t iree_net_loopback_carrier_vtable = {
    .destroy = iree_net_loopback_carrier_destroy,
    .set_recv_handler = iree_net_loopback_carrier_set_recv_handler,
    .activate = iree_net_loopback_carrier_activate,
    .deactivate = iree_net_loopback_carrier_deactivate,
    .query_send_budget = iree_net_loopback_carrier_query_send_budget,
    .send = iree_net_loopback_carrier_send,
    .begin_send = iree_net_loopback_carrier_begin_send,
    .commit_send = iree_net_loopback_carrier_commit_send,
    .abort_send = iree_net_loopback_carrier_abort_send,
    .shutdown = iree_net_loopback_carrier_shutdown,
    .direct_write = iree_net_loopback_carrier_direct_write,
    .direct_read = iree_net_loopback_carrier_direct_read,
    .register_buffer = iree_net_loopback_carrier_register_buffer,
    .unregister_buffer = iree_net_loopback_carrier_unregister_buffer,
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Initializes a single loopback carrier with trailing send slot storage.
static void iree_net_loopback_carrier_init(
    iree_net_loopback_carrier_t* carrier, iree_host_size_t total_size,
    iree_host_size_t send_slots_offset, iree_async_proactor_t* proactor,
    iree_net_carrier_capabilities_t capabilities,
    iree_net_carrier_callback_t callback, iree_allocator_t host_allocator) {
  memset(carrier, 0, total_size);
  iree_net_carrier_initialize(&iree_net_loopback_carrier_vtable, capabilities,
                              0,         // No MTU (stream-like).
                              SIZE_MAX,  // Unlimited scatter-gather.
                              callback, host_allocator, &carrier->base);
  carrier->proactor = proactor;
  iree_async_proactor_retain(proactor);
  carrier->shutdown_initiated = false;
  carrier->send.slot_count = IREE_NET_LOOPBACK_SEND_SLOT_COUNT;
  // All slots start free.
  iree_atomic_store(&carrier->send.free_bitmap, UINT32_MAX,
                    iree_memory_order_relaxed);
  carrier->send.slots =
      (iree_net_loopback_send_slot_t*)((uint8_t*)carrier + send_slots_offset);
}

IREE_API_EXPORT void iree_net_loopback_carrier_set_peer_disconnect_handler(
    iree_net_carrier_t* base_carrier,
    iree_net_loopback_carrier_disconnect_handler_t handler) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);
  carrier->peer_disconnect_handler = handler;
}

IREE_API_EXPORT iree_status_t iree_net_loopback_carrier_create_pair(
    iree_async_proactor_t* proactor, iree_net_carrier_callback_t callback,
    iree_allocator_t host_allocator, iree_net_carrier_t** out_client,
    iree_net_carrier_t** out_server) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_client);
  IREE_ASSERT_ARGUMENT(out_server);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_client = NULL;
  *out_server = NULL;

  // Calculate allocation size with trailing send slot array.
  iree_host_size_t total_size = 0;
  iree_host_size_t send_slots_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(sizeof(iree_net_loopback_carrier_t), &total_size,
                             IREE_STRUCT_FIELD_ALIGNED(
                                 IREE_NET_LOOPBACK_SEND_SLOT_COUNT,
                                 iree_net_loopback_send_slot_t,
                                 iree_alignof(iree_net_loopback_send_slot_t),
                                 &send_slots_offset)));

  // Allocate both carriers.
  iree_net_loopback_carrier_t* client = NULL;
  iree_net_loopback_carrier_t* server = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&client);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size, (void**)&server);
  }

  if (iree_status_is_ok(status)) {
    // Capabilities: reliable, ordered. Data is copied during send() (no
    // zero-copy TX) to match real carrier behavior where the sender's buffer
    // can be freed immediately after send() returns.
    iree_net_carrier_capabilities_t capabilities =
        IREE_NET_CARRIER_CAPABILITY_RELIABLE |
        IREE_NET_CARRIER_CAPABILITY_ORDERED;

    // Initialize both carriers.
    iree_net_loopback_carrier_init(client, total_size, send_slots_offset,
                                   proactor, capabilities, callback,
                                   host_allocator);
    iree_net_loopback_carrier_init(server, total_size, send_slots_offset,
                                   proactor, capabilities, callback,
                                   host_allocator);

    // Set up peer links.
    client->peer = server;
    server->peer = client;

    *out_client = &client->base;
    *out_server = &server->base;
  } else {
    // Cleanup on allocation failure. If client was allocated, it was
    // initialized so we need to release its proactor reference.
    if (client) {
      iree_async_proactor_release(proactor);
    }
    iree_allocator_free(host_allocator, client);
    iree_allocator_free(host_allocator, server);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
