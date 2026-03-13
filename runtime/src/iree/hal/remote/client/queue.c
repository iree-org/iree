// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/queue.h"

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
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

  // Only barrier operations are currently supported (no command buffer).
  if (command_buffer) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "remote command buffer execution not yet "
                            "implemented (barrier only)");
  }

  // Wait frontiers are not yet supported.
  if (wait_semaphore_list.count > 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "remote wait semaphores not yet implemented");
  }

  // Assign epoch N on the remote queue axis.
  uint64_t epoch = device->next_submission_epoch++;

  // Register frontier waiters for signal semaphores. Each waiter will signal
  // its proxy semaphore when the ADVANCE frame arrives from the server.
  iree_host_size_t registered_count = 0;
  iree_status_t status = iree_hal_remote_client_device_register_signal_waiters(
      device, signal_semaphore_list, device->remote_queue_axis, epoch,
      &registered_count);

  // Build and send the COMMAND frame with signal frontier, empty payload.
  //
  // Uses the Dekker pattern: increment channel_users (seq_cst) then load
  // queue_channel (seq_cst). If the channel is NULL (either not yet ready or
  // torn down by goaway/error), bail and decrement. The teardown side zeroes
  // queue_channel (seq_cst) then drains channel_users, so the seq_cst total
  // order guarantees at least one side sees the other's update.
  if (iree_status_is_ok(status)) {
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
      // Stack-allocate signal frontier for the wire.
      iree_async_single_frontier_t signal_frontier_storage;
      iree_async_single_frontier_initialize(&signal_frontier_storage,
                                            device->remote_queue_axis, epoch);
      iree_async_frontier_t* signal_frontier =
          iree_async_single_frontier_as_frontier(&signal_frontier_storage);

      iree_async_span_list_t empty_payload = {NULL, 0};
      status = iree_net_queue_channel_send_command(
          queue_channel, /*stream_id=*/0,
          /*wait_frontier=*/NULL, signal_frontier, empty_payload,
          /*operation_user_data=*/epoch);
      iree_atomic_fetch_sub(&device->channel_users, 1,
                            iree_memory_order_release);
    }
  }

  if (!iree_status_is_ok(status) && registered_count > 0) {
    // Fail all signal semaphores that had waiters registered. The COMMAND was
    // either never sent or failed to send, so the epoch will never advance
    // and those waiters would hang forever. Failing the semaphores ensures
    // the application sees the error. The waiter callbacks will still fire
    // when/if a future submission advances past this epoch, but signaling a
    // failed semaphore is a no-op.
    for (iree_host_size_t i = 0; i < registered_count; ++i) {
      iree_hal_semaphore_fail(signal_semaphore_list.semaphores[i],
                              iree_status_clone(status));
    }
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
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_alloca not yet implemented");
}

iree_status_t iree_hal_remote_client_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_dealloca not yet implemented");
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
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_fill not yet implemented");
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
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_update not yet implemented");
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
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_copy not yet implemented");
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
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_dispatch not yet implemented");
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
    if (device->options.error_callback.fn) {
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
    if (device->options.error_callback.fn) {
      device->options.error_callback.fn(
          device->options.error_callback.user_data, status);
    } else {
      iree_status_ignore(status);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}
