// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/queue.h"

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/hal/remote/client/buffer.h"
#include "iree/hal/remote/client/command_buffer.h"
#include "iree/hal/remote/client/executable.h"
#include "iree/hal/remote/client/semaphore.h"
#include "iree/hal/remote/protocol/queue.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/channel/util/frame_sender.h"

//===----------------------------------------------------------------------===//
// Pending signal batch
//===----------------------------------------------------------------------===//

// Batch header for pending signal contexts. A single allocation holds this
// header followed by N iree_hal_remote_pending_signal_t entries. The atomic
// counter tracks how many entries are still live; the last callback to
// decrement frees the entire batch.
typedef struct iree_hal_remote_pending_signal_batch_t {
  iree_atomic_int32_t remaining;
  iree_allocator_t host_allocator;
  // Trailing: iree_hal_remote_pending_signal_t entries[count]
} iree_hal_remote_pending_signal_batch_t;

// Per-semaphore signal context within a batch. Each entry holds a frontier
// waiter that fires when the server's ADVANCE echoes the submission epoch.
typedef struct iree_hal_remote_pending_signal_t {
  iree_async_frontier_waiter_t waiter;
  iree_hal_semaphore_t* semaphore;  // retained
  uint64_t value;
  iree_hal_remote_pending_signal_batch_t* batch;
  iree_async_single_frontier_t frontier;
} iree_hal_remote_pending_signal_t;

// Fired by the frontier tracker when the signal frontier is satisfied.
// Signals the proxy semaphore to the target value. The last callback to
// complete frees the entire batch allocation.
static void iree_hal_remote_pending_signal_callback(void* user_data,
                                                    iree_status_t status) {
  iree_hal_remote_pending_signal_t* pending =
      (iree_hal_remote_pending_signal_t*)user_data;
  if (iree_status_is_ok(status)) {
    iree_status_t signal_status =
        iree_hal_semaphore_signal(pending->semaphore, pending->value);
    iree_status_ignore(signal_status);
  } else {
    // Frontier wait failed (axis error). Propagate by failing the semaphore.
    iree_hal_semaphore_fail(pending->semaphore, status);
  }
  iree_hal_semaphore_release(pending->semaphore);
  iree_hal_remote_pending_signal_batch_t* batch = pending->batch;
  if (iree_atomic_fetch_sub(&batch->remaining, 1, iree_memory_order_acq_rel) ==
      1) {
    iree_allocator_free(batch->host_allocator, batch);
  }
}

//===----------------------------------------------------------------------===//
// Queue operations
//===----------------------------------------------------------------------===//

// Registers frontier waiters for each signal semaphore. Each waiter fires when
// the server echoes the submission epoch in an ADVANCE frame, signaling the
// proxy semaphore to its target value.
//
// All per-semaphore entries are allocated in a single batch with an atomic ref
// count. The submitter holds a ref during registration; each successfully
// registered waiter adds a ref. The last ref to be released frees the batch.
// On partial failure, |*out_registered_count| reflects how many waiters were
// successfully registered (for error-path semaphore failure by the caller).
static iree_status_t iree_hal_remote_client_device_register_signal_waiters(
    iree_hal_remote_client_device_t* device,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_async_axis_t axis, uint64_t epoch,
    iree_host_size_t* out_registered_count) {
  *out_registered_count = 0;
  if (signal_semaphore_list.count == 0) return iree_ok_status();

  iree_hal_remote_pending_signal_batch_t* batch = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t entries_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*batch), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          signal_semaphore_list.count, iree_hal_remote_pending_signal_t,
          iree_alignof(iree_hal_remote_pending_signal_t), &entries_offset)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(device->host_allocator, total_size,
                                             (void**)&batch));

  iree_atomic_store(&batch->remaining, 1, iree_memory_order_relaxed);
  batch->host_allocator = device->host_allocator;

  iree_hal_remote_pending_signal_t* entries =
      (iree_hal_remote_pending_signal_t*)((uint8_t*)batch + entries_offset);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    iree_hal_remote_pending_signal_t* pending = &entries[i];
    pending->semaphore = signal_semaphore_list.semaphores[i];
    iree_hal_semaphore_retain(pending->semaphore);
    pending->value = signal_semaphore_list.payload_values[i];
    pending->batch = batch;

    iree_async_single_frontier_initialize(&pending->frontier, axis, epoch);

    // Add a ref for this waiter before registration. If registration fails
    // we undo the ref and release the semaphore.
    iree_atomic_fetch_add(&batch->remaining, 1, iree_memory_order_relaxed);

    status = iree_async_frontier_tracker_wait(
        device->frontier_tracker,
        iree_async_single_frontier_as_frontier(&pending->frontier),
        iree_hal_remote_pending_signal_callback, pending, &pending->waiter);
    if (!iree_status_is_ok(status)) {
      iree_atomic_fetch_sub(&batch->remaining, 1, iree_memory_order_relaxed);
      iree_hal_semaphore_release(pending->semaphore);
      break;
    }

    // Record the (value → axis, epoch) mapping on the semaphore AFTER the
    // waiter is successfully registered. Recording before registration would
    // leave a stale mapping if registration fails — a subsequent operation
    // would encode the stale epoch in its wait frontier, and the server would
    // fail with NOT_FOUND because the COMMAND was never sent.
    iree_hal_remote_client_semaphore_record_epoch(pending->semaphore,
                                                  pending->value, axis, epoch);

    ++*out_registered_count;
  }

  // Release the submitter hold. If no waiters were registered (or all
  // failed), this is the last ref and frees the batch immediately.
  if (iree_atomic_fetch_sub(&batch->remaining, 1, iree_memory_order_acq_rel) ==
      1) {
    iree_allocator_free(batch->host_allocator, batch);
  }

  return status;
}

// Builds a wait frontier from a HAL wait_semaphore_list by looking up
// epoch mappings on each proxy semaphore. Populates |out_entries| with up to
// |max_entries| frontier entries. Semaphores that are already satisfied
// (current value >= wait value) are skipped. Unsatisfied semaphores without
// an epoch mapping (host-signaled, cross-device) are reported via
// |out_first_gate|: the first such semaphore and value that must be locally
// gated before the COMMAND can be sent.
static iree_status_t iree_hal_remote_client_device_build_wait_frontier(
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_async_frontier_entry_t* out_entries, iree_host_size_t max_entries,
    iree_host_size_t* out_entry_count, iree_hal_semaphore_t** out_first_gate,
    uint64_t* out_first_gate_value) {
  *out_entry_count = 0;
  *out_first_gate = NULL;
  *out_first_gate_value = 0;
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    iree_hal_semaphore_t* semaphore = wait_semaphore_list.semaphores[i];
    uint64_t value = wait_semaphore_list.payload_values[i];

    // If the semaphore is already satisfied, no wait needed.
    uint64_t current_value = 0;
    iree_status_t query_status =
        iree_hal_semaphore_query(semaphore, &current_value);
    if (iree_status_is_ok(query_status) && current_value >= value) continue;
    iree_status_ignore(query_status);

    // Look up the epoch mapping on the proxy semaphore.
    iree_async_axis_t axis = 0;
    uint64_t epoch = 0;
    if (!iree_hal_remote_client_semaphore_lookup_epoch(semaphore, value, &axis,
                                                       &epoch)) {
      // No epoch mapping: this semaphore will be signaled by something
      // other than a prior queue operation on this device (host signal,
      // cross-device dependency, deferred operation whose epoch hasn't
      // been assigned yet). Report it as a gate.
      if (!*out_first_gate) {
        *out_first_gate = semaphore;
        *out_first_gate_value = value;
      }
      continue;
    }

    if (*out_entry_count >= max_entries) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "wait frontier exceeds max entry count %" PRIhsz,
                              max_entries);
    }
    out_entries[*out_entry_count].axis = axis;
    out_entries[*out_entry_count].epoch = epoch;
    ++*out_entry_count;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Deferred submit
//===----------------------------------------------------------------------===//

// Context for a deferred queue operation submission. Allocated when
// submit_queue_op encounters an unsatisfied wait semaphore without an epoch
// mapping (host→device dependency, cross-device dependency, or dependency on
// a deferred operation whose epoch hasn't been assigned yet).
//
// A timepoint is registered on the gate semaphore. When it fires (semaphore
// satisfied), the submit is retried with the saved arguments. The previously-
// gated semaphore is now satisfied and gets skipped. If more gates remain,
// another deferred context is allocated (recursive, converges after at most
// N retries for N gate semaphores).
typedef struct iree_hal_remote_deferred_submit_t {
  iree_async_semaphore_timepoint_t timepoint;
  iree_hal_remote_client_device_t* device;  // borrowed (device outlives us)
  iree_allocator_t host_allocator;

  // Deep-copied wait and signal semaphore lists. Semaphores are retained.
  iree_host_size_t wait_count;
  iree_host_size_t signal_count;
  // Trailing layout:
  //   iree_hal_semaphore_t* wait_semaphores[wait_count]
  //   uint64_t wait_values[wait_count]
  //   iree_hal_semaphore_t* signal_semaphores[signal_count]
  //   uint64_t signal_values[signal_count]
  //   uint8_t payload_data[payload_length]
  iree_host_size_t payload_length;
} iree_hal_remote_deferred_submit_t;

// Forward declaration — submit_queue_op and the deferred callback are
// mutually recursive.
static iree_status_t iree_hal_remote_client_device_submit_queue_op(
    iree_hal_remote_client_device_t* device,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const uint8_t* payload_data, iree_host_size_t payload_length);

static void iree_hal_remote_deferred_submit_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_remote_deferred_submit_t* deferred =
      (iree_hal_remote_deferred_submit_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Unpack the trailing layout.
  uint8_t* base = (uint8_t*)(deferred + 1);
  iree_hal_semaphore_t** wait_semaphores = (iree_hal_semaphore_t**)base;
  uint64_t* wait_values =
      (uint64_t*)(base + deferred->wait_count * sizeof(iree_hal_semaphore_t*));
  iree_hal_semaphore_t** signal_semaphores =
      (iree_hal_semaphore_t**)(wait_values + deferred->wait_count);
  uint64_t* signal_values =
      (uint64_t*)((uint8_t*)signal_semaphores +
                  deferred->signal_count * sizeof(iree_hal_semaphore_t*));
  uint8_t* payload_data = (uint8_t*)(signal_values + deferred->signal_count);

  if (iree_status_is_ok(status)) {
    // Re-invoke submit. The previously-gated semaphore is now satisfied
    // and will be skipped by build_wait_frontier.
    iree_hal_semaphore_list_t wait_list = {
        .count = deferred->wait_count,
        .semaphores = wait_semaphores,
        .payload_values = wait_values,
    };
    iree_hal_semaphore_list_t signal_list = {
        .count = deferred->signal_count,
        .semaphores = signal_semaphores,
        .payload_values = signal_values,
    };
    status = iree_hal_remote_client_device_submit_queue_op(
        deferred->device, wait_list, signal_list, payload_data,
        deferred->payload_length);
  }

  if (!iree_status_is_ok(status)) {
    // Gate failed or re-submit failed. Fail all signal semaphores.
    iree_hal_semaphore_t** sigs =
        (iree_hal_semaphore_t**)(wait_values + deferred->wait_count);
    for (iree_host_size_t i = 0; i < deferred->signal_count; ++i) {
      iree_hal_semaphore_fail(sigs[i], iree_status_clone(status));
    }
    iree_status_ignore(status);
  }

  // Release retained semaphores.
  for (iree_host_size_t i = 0; i < deferred->wait_count; ++i) {
    iree_hal_semaphore_release(wait_semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < deferred->signal_count; ++i) {
    iree_hal_semaphore_release(signal_semaphores[i]);
  }
  iree_allocator_free(deferred->host_allocator, deferred);
  IREE_TRACE_ZONE_END(z0);
}

// Creates a deferred submit context and registers a timepoint on |gate|.
// Deep-copies the wait/signal semaphore lists and op payload. Returns OK
// immediately; the actual COMMAND send happens when the gate fires.
static iree_status_t iree_hal_remote_client_device_defer_submit(
    iree_hal_remote_client_device_t* device,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const uint8_t* payload_data, iree_host_size_t payload_length,
    iree_hal_semaphore_t* gate, uint64_t gate_value) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Compute trailing layout size with overflow checking.
  iree_host_size_t total_size = sizeof(iree_hal_remote_deferred_submit_t);
  // wait semaphore pointers + values
  if (!iree_host_size_checked_mul_add(total_size, wait_semaphore_list.count,
                                      sizeof(iree_hal_semaphore_t*),
                                      &total_size) ||
      !iree_host_size_checked_mul_add(total_size, wait_semaphore_list.count,
                                      sizeof(uint64_t), &total_size) ||
      // signal semaphore pointers + values
      !iree_host_size_checked_mul_add(total_size, signal_semaphore_list.count,
                                      sizeof(iree_hal_semaphore_t*),
                                      &total_size) ||
      !iree_host_size_checked_mul_add(total_size, signal_semaphore_list.count,
                                      sizeof(uint64_t), &total_size) ||
      // payload data
      !iree_host_size_checked_add(total_size, payload_length, &total_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "deferred submit allocation size overflow");
  }

  iree_hal_remote_deferred_submit_t* deferred = NULL;
  iree_status_t status = iree_allocator_malloc(device->host_allocator,
                                               total_size, (void**)&deferred);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(deferred, 0, sizeof(*deferred));
  deferred->device = device;
  deferred->host_allocator = device->host_allocator;
  deferred->wait_count = wait_semaphore_list.count;
  deferred->signal_count = signal_semaphore_list.count;
  deferred->payload_length = payload_length;

  // Copy and retain semaphore lists.
  uint8_t* base = (uint8_t*)(deferred + 1);
  iree_hal_semaphore_t** wait_sems = (iree_hal_semaphore_t**)base;
  uint64_t* wait_vals = (uint64_t*)(base + wait_semaphore_list.count *
                                               sizeof(iree_hal_semaphore_t*));
  iree_hal_semaphore_t** sig_sems =
      (iree_hal_semaphore_t**)(wait_vals + wait_semaphore_list.count);
  uint64_t* sig_vals =
      (uint64_t*)((uint8_t*)sig_sems +
                  signal_semaphore_list.count * sizeof(iree_hal_semaphore_t*));
  uint8_t* payload_dst = (uint8_t*)(sig_vals + signal_semaphore_list.count);

  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    wait_sems[i] = wait_semaphore_list.semaphores[i];
    iree_hal_semaphore_retain(wait_sems[i]);
    wait_vals[i] = wait_semaphore_list.payload_values[i];
  }
  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    sig_sems[i] = signal_semaphore_list.semaphores[i];
    iree_hal_semaphore_retain(sig_sems[i]);
    sig_vals[i] = signal_semaphore_list.payload_values[i];
  }
  if (payload_length > 0) {
    memcpy(payload_dst, payload_data, payload_length);
  }

  // Register timepoint on the gate semaphore.
  deferred->timepoint.callback = iree_hal_remote_deferred_submit_callback;
  deferred->timepoint.user_data = deferred;
  status = iree_async_semaphore_acquire_timepoint(
      (iree_async_semaphore_t*)gate, gate_value, &deferred->timepoint);

  if (!iree_status_is_ok(status)) {
    // Timepoint registration failed. Clean up.
    for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
      iree_hal_semaphore_release(wait_sems[i]);
    }
    for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
      iree_hal_semaphore_release(sig_sems[i]);
    }
    iree_allocator_free(device->host_allocator, deferred);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Common submission path
//===----------------------------------------------------------------------===//

// Flattens an op payload from span list into a contiguous buffer.
// Returns the total length.
static iree_host_size_t iree_hal_remote_flatten_payload(
    iree_async_span_list_t op_payload, uint8_t* out_buffer) {
  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < op_payload.count; ++i) {
    const void* src = (const void*)(uintptr_t)op_payload.values[i].offset;
    memcpy(out_buffer + offset, src, op_payload.values[i].length);
    offset += op_payload.values[i].length;
  }
  return offset;
}

static iree_host_size_t iree_hal_remote_payload_total_length(
    iree_async_span_list_t op_payload) {
  iree_host_size_t total = 0;
  for (iree_host_size_t i = 0; i < op_payload.count; ++i) {
    total += op_payload.values[i].length;
  }
  return total;
}

// Common submission path for all queue operations. Handles:
//   - Wait frontier encoding from wait_semaphore_list
//   - Gate detection and deferred send for host/cross-device dependencies
//   - Epoch assignment and signal waiter registration
//   - Dekker-pattern queue channel access
//   - COMMAND frame send with wait + signal frontiers + op payload
//   - Error-path semaphore failure
//
// The payload is passed as a contiguous buffer (not a span list) so that
// deferred sends can deep-copy it. Callers with span lists should flatten
// first. The span list variant below is the public entry point.
static iree_status_t iree_hal_remote_client_device_submit_queue_op(
    iree_hal_remote_client_device_t* device,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const uint8_t* payload_data, iree_host_size_t payload_length) {
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Build wait frontier from wait_semaphore_list.
  iree_async_frontier_entry_t wait_entries[8];
  iree_host_size_t wait_entry_count = 0;
  iree_hal_semaphore_t* gate = NULL;
  uint64_t gate_value = 0;
  iree_status_t status = iree_hal_remote_client_device_build_wait_frontier(
      wait_semaphore_list, wait_entries, IREE_ARRAYSIZE(wait_entries),
      &wait_entry_count, &gate, &gate_value);

  // If there's a gate semaphore (unsatisfied, no epoch mapping), defer the
  // entire submission until the gate fires. Epoch assignment and signal waiter
  // registration happen on retry (when the gate is satisfied).
  if (iree_status_is_ok(status) && gate) {
    status = iree_hal_remote_client_device_defer_submit(
        device, wait_semaphore_list, signal_semaphore_list, payload_data,
        payload_length, gate, gate_value);
    if (!iree_status_is_ok(status)) {
      // Deferred submission failed (OOM, timepoint registration failure).
      // Fail all signal semaphores so callers don't hang.
      iree_hal_semaphore_list_fail(signal_semaphore_list,
                                   iree_status_clone(status));
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Assign epoch on the remote queue axis (atomic — deferred callbacks may
  // run on the proactor thread concurrently with app-thread submissions).
  uint64_t epoch = (uint64_t)iree_atomic_fetch_add(
      &device->next_submission_epoch, 1, iree_memory_order_relaxed);

  // Register frontier waiters for signal semaphores.
  iree_host_size_t registered_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_register_signal_waiters(
        device, signal_semaphore_list, device->remote_queue_axis, epoch,
        &registered_count);
  }

  // Build frontier structs for the wire and send.
  if (iree_status_is_ok(status)) {
    iree_async_single_frontier_t signal_frontier_storage;
    iree_async_single_frontier_initialize(&signal_frontier_storage,
                                          device->remote_queue_axis, epoch);
    iree_async_frontier_t* signal_frontier =
        iree_async_single_frontier_as_frontier(&signal_frontier_storage);

    // Build wait frontier (NULL if no entries). Stack-allocated with inline
    // storage for up to 8 entries, layout-compatible with
    // iree_async_frontier_t.
    struct {
      uint8_t entry_count;
      uint8_t reserved[7];
      iree_async_frontier_entry_t entries[8];
    } wait_frontier_storage;
    iree_async_frontier_t* wait_frontier = NULL;
    if (wait_entry_count > 0) {
      memset(&wait_frontier_storage, 0, sizeof(wait_frontier_storage));
      wait_frontier_storage.entry_count = (uint8_t)wait_entry_count;
      memcpy(wait_frontier_storage.entries, wait_entries,
             wait_entry_count * sizeof(iree_async_frontier_entry_t));
      wait_frontier = (iree_async_frontier_t*)&wait_frontier_storage;
    }

    // Build op payload span from contiguous buffer.
    iree_async_span_t payload_span =
        iree_async_span_from_ptr((void*)payload_data, payload_length);
    iree_async_span_list_t payload = {
        payload_length > 0 ? &payload_span : NULL,
        payload_length > 0 ? 1 : 0,
    };

    // Dekker pattern: increment channel_users then load queue_channel.
    iree_atomic_fetch_add(&device->channel_users, 1, iree_memory_order_seq_cst);
    iree_net_queue_channel_t* queue_channel =
        (iree_net_queue_channel_t*)iree_atomic_load(&device->queue_channel,
                                                    iree_memory_order_seq_cst);
    if (!queue_channel) {
      iree_atomic_fetch_sub(&device->channel_users, 1,
                            iree_memory_order_release);
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "queue channel not available");
    } else {
      status = iree_net_queue_channel_send_command(
          queue_channel, /*stream_id=*/0, wait_frontier, signal_frontier,
          payload, /*operation_user_data=*/epoch);
      iree_atomic_fetch_sub(&device->channel_users, 1,
                            iree_memory_order_release);
    }
  }

  if (!iree_status_is_ok(status) && signal_semaphore_list.count > 0) {
    // Fail ALL signal semaphores, not just the registered ones. If
    // register_signal_waiters failed partway through, unregistered
    // semaphores would hang forever. Failing an already-failed semaphore
    // is a no-op (monotonic failure).
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Entry point for queue operations that have span-list payloads.
// Flattens the spans into a contiguous stack buffer and delegates to
// the contiguous-payload submit path.
static iree_status_t iree_hal_remote_client_device_submit_queue_op_spans(
    iree_hal_remote_client_device_t* device,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_async_span_list_t op_payload) {
  // Flatten spans into a stack buffer. Queue op payloads are small (max ~64KB
  // for buffer_update, but typically <100 bytes for fill/copy). For large
  // payloads we heap-allocate.
  iree_host_size_t total_length =
      iree_hal_remote_payload_total_length(op_payload);
  uint8_t stack_buffer[256];
  uint8_t* payload_data = stack_buffer;
  bool heap_allocated = false;
  if (total_length > sizeof(stack_buffer)) {
    iree_status_t status = iree_allocator_malloc(
        device->host_allocator, total_length, (void**)&payload_data);
    if (!iree_status_is_ok(status)) return status;
    heap_allocated = true;
  }
  iree_hal_remote_flatten_payload(op_payload, payload_data);

  iree_status_t status = iree_hal_remote_client_device_submit_queue_op(
      device, wait_semaphore_list, signal_semaphore_list, payload_data,
      total_length);

  if (heap_allocated) {
    iree_allocator_free(device->host_allocator, payload_data);
  }
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!command_buffer) {
    // Barrier: delegate to the common submit path with empty payload.
    iree_status_t status = iree_hal_remote_client_device_submit_queue_op(
        device, wait_semaphore_list, signal_semaphore_list,
        /*payload_data=*/NULL, /*payload_length=*/0);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Command buffer execution. Build a COMMAND_BUFFER_EXECUTE op with either
  // inline command stream (one-shot) or cached resource ID (reusable).
  bool is_one_shot = iree_all_bits_set(command_buffer->mode,
                                       IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT);
  iree_const_byte_span_t stream =
      is_one_shot ? iree_hal_remote_client_command_buffer_stream(command_buffer)
                  : iree_const_byte_span_empty();

  // Compute total payload size: header + binding table + inline stream.
  iree_host_size_t header_size =
      sizeof(iree_hal_remote_command_buffer_execute_op_t);
  iree_host_size_t bindings_size = 0;
  if (!iree_host_size_checked_mul(binding_table.count,
                                  sizeof(iree_hal_remote_binding_t),
                                  &bindings_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command buffer execute bindings size overflow");
  }
  iree_host_size_t total_payload = 0;
  if (!iree_host_size_checked_add(header_size, bindings_size, &total_payload) ||
      !iree_host_size_checked_add(total_payload, stream.data_length,
                                  &total_payload)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command buffer execute payload size overflow");
  }

  // Allocate contiguous payload.
  uint8_t stack_buffer[512];
  uint8_t* payload_data = stack_buffer;
  bool heap_allocated = false;
  if (total_payload > sizeof(stack_buffer)) {
    iree_status_t alloc_status = iree_allocator_malloc(
        device->host_allocator, total_payload, (void**)&payload_data);
    if (!iree_status_is_ok(alloc_status)) {
      IREE_TRACE_ZONE_END(z0);
      return alloc_status;
    }
    heap_allocated = true;
  }
  memset(payload_data, 0, total_payload);

  // Fill the execute op header.
  iree_hal_remote_command_buffer_execute_op_t* op =
      (iree_hal_remote_command_buffer_execute_op_t*)payload_data;
  op->header.type = IREE_HAL_REMOTE_QUEUE_OP_COMMAND_BUFFER_EXECUTE;
  if (is_one_shot) {
    op->header.flags = IREE_HAL_REMOTE_EXECUTE_FLAG_INLINE_COMMAND_STREAM;
  } else {
    op->command_buffer_id =
        iree_hal_remote_client_command_buffer_resource_id(command_buffer);
  }
  op->binding_count = (uint16_t)binding_table.count;
  op->execute_flags = flags;

  // Serialize binding table.
  iree_hal_remote_binding_t* wire_bindings =
      (iree_hal_remote_binding_t*)(payload_data + header_size);
  for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
    if (binding_table.bindings[i].buffer) {
      wire_bindings[i].buffer_id = iree_hal_remote_client_buffer_resource_id(
          binding_table.bindings[i].buffer);
    }
    wire_bindings[i].offset = binding_table.bindings[i].offset;
    wire_bindings[i].length = binding_table.bindings[i].length;
  }

  // Append inline command stream for one-shot.
  if (stream.data_length > 0) {
    memcpy(payload_data + header_size + bindings_size, stream.data,
           stream.data_length);
  }

  iree_status_t status = iree_hal_remote_client_device_submit_queue_op(
      device, wait_semaphore_list, signal_semaphore_list, payload_data,
      total_payload);

  if (heap_allocated) {
    iree_allocator_free(device->host_allocator, payload_data);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);

  // Assign a unique provisional resource ID for this allocation.
  uint16_t generation = (uint16_t)iree_atomic_fetch_add(
      &device->next_provisional_generation, 1, iree_memory_order_relaxed);
  iree_hal_remote_resource_id_t provisional_id =
      IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, generation);

  // Create a buffer proxy with the provisional ID. The buffer is immediately
  // usable in subsequent queue ops (frontier ordering guarantees that the
  // server processes the alloca before any op that references this buffer).
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_remote_client_buffer_create(
      device, provisional_id, &params, allocation_size, device->host_allocator,
      &buffer);

  // Register in the provisional buffer table so on_advance can resolve the
  // provisional_id to the server's canonical ID.
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_register_provisional(
        device, provisional_id, buffer);
  }

  // Build and send the COMMAND frame.
  if (iree_status_is_ok(status)) {
    iree_hal_remote_buffer_alloca_op_t op;
    memset(&op, 0, sizeof(op));
    op.header.type = IREE_HAL_REMOTE_QUEUE_OP_BUFFER_ALLOCA;
    op.pool = pool;
    op.params.usage = params.usage;
    op.params.access = (uint16_t)params.access;
    op.params.type = params.type;
    op.params.queue_affinity = params.queue_affinity;
    op.params.min_alignment = (uint64_t)params.min_alignment;
    op.allocation_size = (uint64_t)allocation_size;
    op.alloca_flags = flags;
    op.provisional_buffer_id = provisional_id;

    iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
    iree_async_span_list_t payload = {&span, 1};
    status = iree_hal_remote_client_device_submit_queue_op_spans(
        device, wait_semaphore_list, signal_semaphore_list, payload);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_buffer_dealloca_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_BUFFER_DEALLOCA;
  op.buffer_id = iree_hal_remote_client_buffer_resource_id(buffer);
  op.dealloca_flags = flags;

  iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
  iree_async_span_list_t payload = {&span, 1};
  iree_status_t status = iree_hal_remote_client_device_submit_queue_op_spans(
      device, wait_semaphore_list, signal_semaphore_list, payload);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_ASSERT_ARGUMENT(pattern);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  iree_hal_remote_buffer_fill_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_BUFFER_FILL;
  op.target_buffer_id =
      iree_hal_remote_client_buffer_resource_id(target_buffer);
  op.target_offset = target_offset;
  op.length = length;
  op.pattern_length = (uint8_t)pattern_length;
  op.fill_flags = flags;
  // Copy pattern (1, 2, or 4 bytes) into the 4-byte field, zero-extended.
  memcpy(&op.pattern, pattern, pattern_length);

  iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
  iree_async_span_list_t payload = {&span, 1};
  iree_status_t status = iree_hal_remote_client_device_submit_queue_op_spans(
      device, wait_semaphore_list, signal_semaphore_list, payload);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  iree_hal_remote_buffer_update_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_BUFFER_UPDATE;
  op.target_buffer_id =
      iree_hal_remote_client_buffer_resource_id(target_buffer);
  op.target_offset = target_offset;
  op.length = length;
  op.update_flags = flags;

  // The inline source data follows the op struct. Build two spans: the op
  // header and the source data. Padding to 8-byte alignment is handled by
  // the queue channel's frame builder.
  iree_async_span_t spans[2] = {
      iree_async_span_from_ptr(&op, sizeof(op)),
      iree_async_span_from_ptr(
          (void*)((const uint8_t*)source_buffer + source_offset),
          (iree_host_size_t)length),
  };
  iree_async_span_list_t payload = {spans, 2};
  iree_status_t status = iree_hal_remote_client_device_submit_queue_op_spans(
      device, wait_semaphore_list, signal_semaphore_list, payload);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  iree_hal_remote_buffer_copy_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_BUFFER_COPY;
  op.source_buffer_id =
      iree_hal_remote_client_buffer_resource_id(source_buffer);
  op.source_offset = source_offset;
  op.target_buffer_id =
      iree_hal_remote_client_buffer_resource_id(target_buffer);
  op.target_offset = target_offset;
  op.length = length;
  op.copy_flags = flags;

  iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
  iree_async_span_list_t payload = {&span, 1};
  iree_status_t status = iree_hal_remote_client_device_submit_queue_op_spans(
      device, wait_semaphore_list, signal_semaphore_list, payload);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_read not yet implemented");
}

iree_status_t iree_hal_remote_client_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_write not yet implemented");
}

iree_status_t iree_hal_remote_client_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "queue host calls not supported on remote device; host calls "
      "require local execution with buffer contents transferred");
}

iree_status_t iree_hal_remote_client_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Build the dispatch op with variable-length constants + bindings.
  uint16_t constant_count =
      (uint16_t)(constants.data_length / sizeof(uint32_t));
  uint16_t binding_count = (uint16_t)bindings.count;

  iree_host_size_t constants_size = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)constant_count,
                                  sizeof(uint32_t), &constants_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch constants size overflow");
  }
  iree_host_size_t constants_padded = iree_host_align(constants_size, 8);
  iree_host_size_t bindings_size = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)binding_count,
                                  sizeof(iree_hal_remote_binding_t),
                                  &bindings_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch bindings size overflow");
  }
  iree_host_size_t total_payload = 0;
  if (!iree_host_size_checked_add(sizeof(iree_hal_remote_dispatch_op_t),
                                  constants_padded, &total_payload) ||
      !iree_host_size_checked_add(total_payload, bindings_size,
                                  &total_payload)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch payload size overflow");
  }

  // Allocate contiguous payload on the stack for small dispatches, heap
  // for large ones.
  uint8_t stack_buffer[512];
  uint8_t* payload_data = stack_buffer;
  bool heap_allocated = false;
  if (total_payload > sizeof(stack_buffer)) {
    iree_status_t alloc_status = iree_allocator_malloc(
        device->host_allocator, total_payload, (void**)&payload_data);
    if (!iree_status_is_ok(alloc_status)) {
      IREE_TRACE_ZONE_END(z0);
      return alloc_status;
    }
    heap_allocated = true;
  }
  memset(payload_data, 0, total_payload);

  // Fill the dispatch op header.
  iree_hal_remote_dispatch_op_t* op =
      (iree_hal_remote_dispatch_op_t*)payload_data;
  op->header.type = IREE_HAL_REMOTE_QUEUE_OP_DISPATCH;
  op->executable_id = iree_hal_remote_client_executable_resource_id(executable);
  op->export_ordinal = export_ordinal;
  memcpy(op->config.workgroup_size, config.workgroup_size,
         sizeof(config.workgroup_size));
  memcpy(op->config.workgroup_count, config.workgroup_count,
         sizeof(config.workgroup_count));
  op->config.dynamic_workgroup_local_memory =
      config.dynamic_workgroup_local_memory;
  op->constant_count = constant_count;
  op->binding_count = binding_count;
  op->dispatch_flags = flags;

  // Copy constants (padded to 8-byte alignment).
  uint8_t* constants_dst = payload_data + sizeof(iree_hal_remote_dispatch_op_t);
  if (constants_size > 0) {
    memcpy(constants_dst, constants.data, constants_size);
  }

  // Serialize bindings.
  iree_hal_remote_binding_t* bindings_dst =
      (iree_hal_remote_binding_t*)(constants_dst + constants_padded);
  for (uint16_t i = 0; i < binding_count; ++i) {
    const iree_hal_buffer_ref_t* ref = &bindings.values[i];
    if (ref->buffer) {
      bindings_dst[i].buffer_id =
          iree_hal_remote_client_buffer_resource_id(ref->buffer);
    }
    bindings_dst[i].offset = ref->offset;
    bindings_dst[i].length = ref->length;
    bindings_dst[i].buffer_slot = ref->buffer_slot;
  }

  iree_status_t status = iree_hal_remote_client_device_submit_queue_op(
      device, wait_semaphore_list, signal_semaphore_list, payload_data,
      total_payload);

  if (heap_allocated) {
    iree_allocator_free(device->host_allocator, payload_data);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  // All sends are immediate (no batching). Nothing to flush.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Queue channel callbacks
//===----------------------------------------------------------------------===//

// Client receives ADVANCE frames when server-side operations complete.
// Advances the frontier tracker for each entry in the signal frontier, which
// dispatches any waiters whose frontiers are now satisfied.
static iree_status_t iree_hal_remote_client_device_on_advance(
    void* user_data, const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t advance_data, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!signal_frontier || signal_frontier->entry_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ADVANCE frame with empty signal frontier");
  }

  // Process resolution entries BEFORE advancing the frontier. This ensures
  // that when the frontier advance fires semaphore signals and the
  // application wakes, all buffer proxies have their resolved resource_ids.
  if (advance_data.data_length >= sizeof(iree_hal_remote_advance_payload_t)) {
    const iree_hal_remote_advance_payload_t* advance_payload =
        (const iree_hal_remote_advance_payload_t*)advance_data.data;
    iree_host_size_t entries_size =
        (iree_host_size_t)advance_payload->resolution_count *
        sizeof(iree_hal_remote_resolution_entry_t);
    if (advance_data.data_length >=
        sizeof(iree_hal_remote_advance_payload_t) + entries_size) {
      const iree_hal_remote_resolution_entry_t* entries =
          (const iree_hal_remote_resolution_entry_t*)(advance_payload + 1);
      for (uint16_t i = 0; i < advance_payload->resolution_count; ++i) {
        iree_hal_buffer_t* buffer =
            iree_hal_remote_client_device_resolve_provisional(
                device, entries[i].provisional_id);
        if (buffer) {
          iree_hal_remote_client_buffer_set_resource_id(buffer,
                                                        entries[i].resolved_id);
        }
      }
    }
  }

  // Now advance the frontier tracker (fires semaphore signals).
  for (uint8_t i = 0; i < signal_frontier->entry_count; ++i) {
    iree_async_frontier_tracker_advance(device->frontier_tracker,
                                        signal_frontier->entries[i].axis,
                                        signal_frontier->entries[i].epoch);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Client does not receive COMMAND frames (only servers do).
static iree_status_t iree_hal_remote_client_device_on_command(
    void* user_data, uint32_t stream_id,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t command_data, iree_async_buffer_lease_t* lease) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "client does not accept COMMAND frames");
}

// Transport error on the queue channel endpoint.
static void iree_hal_remote_client_device_on_queue_transport_error(
    void* user_data, iree_status_t status) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_client_device_store_state(
      device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
  if (device->options.error_callback.fn) {
    device->options.error_callback.fn(device->options.error_callback.user_data,
                                      status);
  } else {
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
}

// Called when the queue endpoint is ready after session bootstrap.
// Creates the header pool, queue channel, and activates it.
void iree_hal_remote_client_device_on_queue_endpoint_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_store_state(
        device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
    // Fire the connect callback with error so the application doesn't hang.
    iree_hal_remote_client_device_connected_callback_t callback =
        device->connect_callback;
    memset(&device->connect_callback, 0, sizeof(device->connect_callback));
    if (callback.fn) {
      callback.fn(callback.user_data,
                  iree_make_status(IREE_STATUS_INTERNAL,
                                   "failed to open queue endpoint"));
    } else if (device->options.error_callback.fn) {
      device->options.error_callback.fn(
          device->options.error_callback.user_data,
          iree_make_status(IREE_STATUS_INTERNAL,
                           "failed to open queue endpoint"));
    }
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Create header pool and queue channel into locals first, then publish
  // atomically to device->queue_channel.
  iree_async_buffer_pool_t* header_pool = NULL;
  iree_net_queue_channel_t* queue_channel = NULL;

  // Create header pool for queue frame header + frontier encoding.
  status = iree_hal_remote_create_queue_header_pool(
      IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_COUNT,
      IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_SIZE, device->host_allocator,
      &header_pool);

  // Create queue channel with client-side callbacks.
  if (iree_status_is_ok(status)) {
    iree_net_queue_channel_callbacks_t callbacks = {
        .on_command = iree_hal_remote_client_device_on_command,
        .on_advance = iree_hal_remote_client_device_on_advance,
        .on_transport_error =
            iree_hal_remote_client_device_on_queue_transport_error,
        .user_data = device,
    };

    status = iree_net_queue_channel_create(
        endpoint, IREE_NET_FRAME_SENDER_MAX_SPANS, header_pool, callbacks,
        device->host_allocator, &queue_channel);
  }

  // Activate the channel to begin receiving frames.
  if (iree_status_is_ok(status)) {
    status = iree_net_queue_channel_activate(queue_channel);
  }

  if (iree_status_is_ok(status)) {
    // Publish atomically. The hot path (queue_execute) loads with seq_cst
    // after incrementing channel_users, establishing the Dekker ordering.
    iree_atomic_store(&device->queue_channel, (intptr_t)queue_channel,
                      iree_memory_order_release);

    // Queue channel is ready. Fire the deferred connect callback so the
    // application can immediately submit queue operations.
    iree_hal_remote_client_device_connected_callback_t callback =
        device->connect_callback;
    memset(&device->connect_callback, 0, sizeof(device->connect_callback));
    if (callback.fn) {
      callback.fn(callback.user_data, iree_ok_status());
    }
  } else {
    // Cleanup on failure. Channel owns the pool if it was created
    // successfully; otherwise we must free the pool ourselves.
    if (queue_channel) {
      iree_net_queue_channel_release(queue_channel);
    } else {
      iree_async_buffer_pool_free(header_pool);
    }

    iree_hal_remote_client_device_store_state(
        device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
    // Fire connect callback with error so the application doesn't hang.
    iree_hal_remote_client_device_connected_callback_t callback =
        device->connect_callback;
    memset(&device->connect_callback, 0, sizeof(device->connect_callback));
    if (callback.fn) {
      callback.fn(callback.user_data, status);
    } else if (device->options.error_callback.fn) {
      device->options.error_callback.fn(
          device->options.error_callback.user_data, status);
    } else {
      iree_status_ignore(status);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}
