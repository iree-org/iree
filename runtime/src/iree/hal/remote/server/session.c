// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/session.h"

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier.h"
#include "iree/async/semaphore.h"
#include "iree/hal/remote/protocol/common.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/protocol/queue.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/status_wire.h"

//===----------------------------------------------------------------------===//
// Session slot helpers
//===----------------------------------------------------------------------===//

// Finds the slot holding the given session. Returns -1 if not found.
static int32_t iree_hal_remote_server_find_session_slot(
    iree_hal_remote_server_t* server, iree_net_session_t* session) {
  for (uint32_t i = 0; i < server->options.max_connections; ++i) {
    if (server->sessions[i].session == session) return (int32_t)i;
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// Session removal
//===----------------------------------------------------------------------===//

void iree_hal_remote_server_remove_session(iree_hal_remote_server_t* server,
                                           iree_net_session_t* session) {
  iree_net_queue_channel_t* queue_channel = NULL;
  iree_hal_remote_server_stopped_callback_t stopped_callback;
  memset(&stopped_callback, 0, sizeof(stopped_callback));

  // Snapshot the resource table and epoch mapping for cleanup outside the lock.
  iree_hal_remote_resource_table_t resource_table;
  memset(&resource_table, 0, sizeof(resource_table));
  uint64_t* epoch_map_epochs = NULL;
  iree_hal_semaphore_t** epoch_map_semaphores = NULL;
  iree_host_size_t epoch_map_count = 0;
  iree_hal_remote_resource_id_t* prov_map_provisionals = NULL;
  iree_hal_remote_resource_id_t* prov_map_resolved = NULL;

  iree_slim_mutex_lock(&server->session_mutex);
  int32_t slot = iree_hal_remote_server_find_session_slot(server, session);
  if (slot >= 0) {
    // Snapshot references to clean up outside the lock.
    queue_channel = server->sessions[slot].queue_channel;
    server->sessions[slot].queue_channel = NULL;

    resource_table = server->sessions[slot].resource_table;
    memset(&server->sessions[slot].resource_table, 0,
           sizeof(server->sessions[slot].resource_table));

    epoch_map_epochs = server->sessions[slot].epoch_semaphore_map.epochs;
    epoch_map_semaphores =
        server->sessions[slot].epoch_semaphore_map.semaphores;
    epoch_map_count = server->sessions[slot].epoch_semaphore_map.count;
    memset(&server->sessions[slot].epoch_semaphore_map, 0,
           sizeof(server->sessions[slot].epoch_semaphore_map));

    prov_map_provisionals =
        server->sessions[slot].provisional_map.provisional_ids;
    prov_map_resolved = server->sessions[slot].provisional_map.resolved_ids;
    memset(&server->sessions[slot].provisional_map, 0,
           sizeof(server->sessions[slot].provisional_map));

    server->sessions[slot].session = NULL;
    server->sessions[slot].session_id = 0;
    --server->active_session_count;

    // Check if shutdown is now complete.
    if (server->state == IREE_HAL_REMOTE_SERVER_STATE_STOPPING &&
        server->active_session_count == 0 && !server->listener) {
      server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;
      stopped_callback = server->stopped_callback;
    }
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  if (slot < 0) return;  // Already removed (e.g., double callback).

  // Release all resources in the table.
  iree_hal_remote_resource_table_deinitialize(&resource_table,
                                              server->host_allocator);

  // Release all local semaphores in the epoch mapping.
  for (iree_host_size_t i = 0; i < epoch_map_count; ++i) {
    iree_hal_semaphore_release(epoch_map_semaphores[i]);
  }
  iree_allocator_free(server->host_allocator, epoch_map_epochs);
  iree_allocator_free(server->host_allocator, epoch_map_semaphores);

  // Free provisional mapping arrays.
  iree_allocator_free(server->host_allocator, prov_map_provisionals);
  iree_allocator_free(server->host_allocator, prov_map_resolved);

  // Detach the queue channel from its endpoint before releasing the session.
  // Command completions may hold retained references to the channel that
  // outlive the session. Detach clears the endpoint callbacks (safe while the
  // endpoint is alive) and zeroes the endpoint reference so that the eventual
  // channel destroy does not UAF on the freed endpoint.
  iree_net_queue_channel_detach(queue_channel);
  iree_net_queue_channel_release(queue_channel);
  iree_net_session_release(session);

  if (stopped_callback.fn) {
    stopped_callback.fn(stopped_callback.user_data);
  }
}

//===----------------------------------------------------------------------===//
// Command completion
//===----------------------------------------------------------------------===//

// Context for a pending command completion on the server. Heap-allocated and
// freed in the timepoint callback after sending the ADVANCE frame. Used for
// all queue operations (barrier, fill, copy, update, alloca, dealloca).
typedef struct iree_hal_remote_server_command_completion_t {
  iree_async_semaphore_timepoint_t timepoint;
  iree_net_queue_channel_t* queue_channel;  // retained
  iree_hal_semaphore_t* local_semaphore;    // retained
  iree_allocator_t host_allocator;
  iree_async_single_frontier_t signal_frontier;
  // Resolution entry for BUFFER_ALLOCA completions. The ADVANCE frame
  // piggybacks this (provisional → resolved) mapping so the client can
  // update its buffer proxy. resolution_count is 0 for non-alloca ops.
  uint16_t resolution_count;
  uint16_t resolution_padding[3];
  iree_hal_remote_resolution_entry_t resolution;
} iree_hal_remote_server_command_completion_t;

// Fired by the local device's semaphore when the queue operation completes.
// Sends ADVANCE back to the client and cleans up.
static void iree_hal_remote_server_on_command_complete(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_remote_server_command_completion_t* completion =
      (iree_hal_remote_server_command_completion_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_status_is_ok(status)) {
    iree_async_frontier_t* signal_frontier =
        iree_async_single_frontier_as_frontier(&completion->signal_frontier);

    if (completion->resolution_count > 0) {
      // BUFFER_ALLOCA: include resolution entries in the ADVANCE payload.
      iree_hal_remote_advance_payload_t advance_header;
      memset(&advance_header, 0, sizeof(advance_header));
      advance_header.resolution_count = completion->resolution_count;
      iree_async_span_t spans[2] = {
          iree_async_span_from_ptr(&advance_header, sizeof(advance_header)),
          iree_async_span_from_ptr(&completion->resolution,
                                   sizeof(completion->resolution)),
      };
      iree_async_span_list_t payload = {spans, 2};
      iree_status_t send_status = iree_net_queue_channel_send_advance(
          completion->queue_channel, signal_frontier, payload,
          /*operation_user_data=*/0);
      iree_status_ignore(send_status);
    } else {
      // Non-alloca: empty ADVANCE payload.
      iree_async_span_list_t empty_payload = {NULL, 0};
      iree_status_t send_status = iree_net_queue_channel_send_advance(
          completion->queue_channel, signal_frontier, empty_payload,
          /*operation_user_data=*/0);
      iree_status_ignore(send_status);
    }
  } else {
    iree_status_ignore(status);
  }

  iree_net_queue_channel_release(completion->queue_channel);
  iree_hal_semaphore_release(completion->local_semaphore);
  iree_allocator_free(completion->host_allocator, completion);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Epoch→semaphore mapping
//===----------------------------------------------------------------------===//

// Stores a mapping from signal frontier epoch to local semaphore.
// Retains the semaphore. Epochs are appended in monotonically increasing order.
static iree_status_t iree_hal_remote_server_store_epoch_semaphore(
    iree_hal_remote_server_session_t* session_slot, uint64_t epoch,
    iree_hal_semaphore_t* semaphore, iree_allocator_t host_allocator) {
  // Grow both parallel arrays together so they share a single capacity.
  // iree_allocator_grow_array handles the doubling strategy internally.
  iree_host_size_t minimum_capacity =
      session_slot->epoch_semaphore_map.count + 1;
  if (minimum_capacity > session_slot->epoch_semaphore_map.capacity) {
    // Save capacity before first grow — both arrays must grow to the same
    // target. Grow epochs first, then grow semaphores to the same capacity.
    iree_host_size_t epochs_capacity =
        session_slot->epoch_semaphore_map.capacity;
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        host_allocator, minimum_capacity, sizeof(uint64_t), &epochs_capacity,
        (void**)&session_slot->epoch_semaphore_map.epochs));
    iree_host_size_t semaphores_capacity =
        session_slot->epoch_semaphore_map.capacity;
    iree_status_t status = iree_allocator_grow_array(
        host_allocator, minimum_capacity, sizeof(iree_hal_semaphore_t*),
        &semaphores_capacity,
        (void**)&session_slot->epoch_semaphore_map.semaphores);
    if (!iree_status_is_ok(status)) {
      // Epochs grew but semaphores didn't. The epochs array is larger than
      // needed but still valid — capacity reflects the smaller of the two.
      // Don't update capacity; the next call will try to grow semaphores
      // again.
      return status;
    }
    // Both grew successfully. Use the minimum of the two (they should be
    // equal since grow_array uses the same doubling strategy, but be safe).
    session_slot->epoch_semaphore_map.capacity =
        iree_min(epochs_capacity, semaphores_capacity);
  }

  iree_host_size_t index = session_slot->epoch_semaphore_map.count++;
  session_slot->epoch_semaphore_map.epochs[index] = epoch;
  session_slot->epoch_semaphore_map.semaphores[index] = semaphore;
  iree_hal_semaphore_retain(semaphore);
  return iree_ok_status();
}

// Looks up the local semaphore for a given epoch. Returns NULL if not found.
// The returned pointer is borrowed — the epoch mapping retains it.
static iree_hal_semaphore_t* iree_hal_remote_server_lookup_epoch_semaphore(
    iree_hal_remote_server_session_t* session_slot, uint64_t epoch) {
  // Binary search (epochs are monotonically increasing).
  iree_host_size_t low = 0;
  iree_host_size_t high = session_slot->epoch_semaphore_map.count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    if (session_slot->epoch_semaphore_map.epochs[mid] < epoch) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  if (low < session_slot->epoch_semaphore_map.count &&
      session_slot->epoch_semaphore_map.epochs[low] == epoch) {
    return session_slot->epoch_semaphore_map.semaphores[low];
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Provisional→resolved resource ID mapping
//===----------------------------------------------------------------------===//

// Stores a provisional→resolved resource ID mapping. Used by BUFFER_ALLOCA
// so that subsequent commands referencing the provisional ID can be resolved.
static iree_status_t iree_hal_remote_server_store_provisional(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t provisional_id,
    iree_hal_remote_resource_id_t resolved_id,
    iree_allocator_t host_allocator) {
  iree_host_size_t minimum_capacity = session_slot->provisional_map.count + 1;
  if (minimum_capacity > session_slot->provisional_map.capacity) {
    iree_host_size_t prov_capacity = session_slot->provisional_map.capacity;
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        host_allocator, minimum_capacity, sizeof(iree_hal_remote_resource_id_t),
        &prov_capacity,
        (void**)&session_slot->provisional_map.provisional_ids));
    iree_host_size_t res_capacity = session_slot->provisional_map.capacity;
    iree_status_t status = iree_allocator_grow_array(
        host_allocator, minimum_capacity, sizeof(iree_hal_remote_resource_id_t),
        &res_capacity, (void**)&session_slot->provisional_map.resolved_ids);
    if (!iree_status_is_ok(status)) {
      session_slot->provisional_map.capacity = prov_capacity;
      return status;
    }
    session_slot->provisional_map.capacity =
        iree_min(prov_capacity, res_capacity);
  }
  iree_host_size_t index = session_slot->provisional_map.count++;
  session_slot->provisional_map.provisional_ids[index] = provisional_id;
  session_slot->provisional_map.resolved_ids[index] = resolved_id;
  return iree_ok_status();
}

// Resolves a resource ID that may be provisional. If the ID has the
// PROVISIONAL flag set, looks up the mapping and returns the resolved ID.
// If not provisional, returns the ID unchanged.
static iree_hal_remote_resource_id_t iree_hal_remote_server_resolve_resource_id(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t resource_id) {
  if (!IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(resource_id)) {
    return resource_id;
  }
  for (iree_host_size_t i = 0; i < session_slot->provisional_map.count; ++i) {
    if (session_slot->provisional_map.provisional_ids[i] == resource_id) {
      return session_slot->provisional_map.resolved_ids[i];
    }
  }
  return resource_id;  // Not found — return as-is (will fail in table lookup).
}

// Resolves a wire wait_frontier to a local wait semaphore list. For each
// (axis, epoch) entry in the wait frontier, looks up the corresponding local
// semaphore from the epoch mapping. All resolved semaphores use payload_value=1
// (each local semaphore signals to 1 on completion).
//
// |out_semaphores| and |out_values| are caller-provided arrays of |capacity|.
// Returns the number of resolved entries in |out_count|.
static iree_status_t iree_hal_remote_server_resolve_wait_frontier(
    iree_hal_remote_server_session_t* session_slot,
    const iree_async_frontier_t* wait_frontier,
    iree_hal_semaphore_t** out_semaphores, uint64_t* out_values,
    iree_host_size_t capacity, iree_host_size_t* out_count) {
  *out_count = 0;
  if (!wait_frontier) return iree_ok_status();
  for (uint8_t i = 0; i < wait_frontier->entry_count; ++i) {
    iree_hal_semaphore_t* local_semaphore =
        iree_hal_remote_server_lookup_epoch_semaphore(
            session_slot, wait_frontier->entries[i].epoch);
    if (!local_semaphore) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no local semaphore for wait frontier epoch "
                              "%" PRIu64,
                              wait_frontier->entries[i].epoch);
    }
    if (*out_count >= capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "wait frontier exceeds local semaphore capacity");
    }
    out_semaphores[*out_count] = local_semaphore;
    out_values[*out_count] = 1;
    ++*out_count;
  }
  return iree_ok_status();
}

// Common setup for a queue operation on the server: creates a local signal
// semaphore, stores the epoch mapping, submits the operation to the local
// device, and registers a timepoint to send ADVANCE on completion.
//
// |submit_fn| is called to perform the actual device queue operation, receiving
// the local wait and signal semaphore lists. The caller provides the specific
// device queue call (fill, copy, update, barrier) via this callback.
typedef iree_status_t (*iree_hal_remote_server_submit_fn_t)(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list, iree_hal_semaphore_list_t signal_list);

// Context passed to op submit callbacks. Carries the command payload and
// session slot reference. Alloca callbacks also populate resolution data
// that gets piggybacked on the ADVANCE frame.
typedef struct iree_hal_remote_server_op_context_t {
  iree_hal_remote_server_session_t* session_slot;
  iree_const_byte_span_t command_data;
  // Populated by BUFFER_ALLOCA callback to piggyback resolution on ADVANCE.
  uint16_t resolution_count;
  iree_hal_remote_resolution_entry_t resolution;
} iree_hal_remote_server_op_context_t;

// The submit_fn callback may populate resolution data on the op_context
// (e.g., BUFFER_ALLOCA stores provisional→resolved mapping). After the
// callback returns, submit_command checks the op_context for resolution
// data and copies it into the completion context for piggybacking on the
// ADVANCE frame. Non-alloca ops leave resolution_count=0.
static iree_status_t iree_hal_remote_server_submit_command(
    iree_hal_remote_server_session_t* session_slot,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_hal_remote_server_submit_fn_t submit_fn, void* submit_user_data) {
  iree_hal_remote_server_t* server = session_slot->server;
  iree_hal_device_t* local_device = server->devices[0];

  // Resolve wait frontier → local wait semaphore list.
  iree_hal_semaphore_t* wait_semaphores[8];
  uint64_t wait_values[8];
  iree_host_size_t wait_count = 0;
  iree_status_t status = iree_hal_remote_server_resolve_wait_frontier(
      session_slot, wait_frontier, wait_semaphores, wait_values,
      IREE_ARRAYSIZE(wait_semaphores), &wait_count);

  // Create local signal semaphore (initial_value=0).
  iree_hal_semaphore_t* local_semaphore = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &local_semaphore);
  }

  // Store epoch→semaphore mapping for future wait frontier resolution.
  if (iree_status_is_ok(status)) {
    IREE_ASSERT(signal_frontier && signal_frontier->entry_count == 1);
    status = iree_hal_remote_server_store_epoch_semaphore(
        session_slot, signal_frontier->entries[0].epoch, local_semaphore,
        server->host_allocator);
  }

  // Submit the operation to the local device.
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t wait_list = {
        .count = wait_count,
        .semaphores = wait_semaphores,
        .payload_values = wait_values,
    };
    iree_hal_semaphore_list_t signal_list = {
        .count = 1,
        .semaphores = &local_semaphore,
        .payload_values = (uint64_t[]){1},
    };
    status = submit_fn(submit_user_data, local_device, wait_list, signal_list);
  }

  // Allocate completion context and register timepoint.
  iree_hal_remote_server_command_completion_t* completion = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(server->host_allocator, sizeof(*completion),
                                   (void**)&completion);
  }

  if (iree_status_is_ok(status)) {
    memset(completion, 0, sizeof(*completion));
    completion->host_allocator = server->host_allocator;
    completion->queue_channel = session_slot->queue_channel;
    iree_net_queue_channel_retain(completion->queue_channel);
    completion->local_semaphore = local_semaphore;
    iree_hal_semaphore_retain(local_semaphore);

    // Deep copy the signal frontier.
    iree_async_single_frontier_initialize(&completion->signal_frontier,
                                          signal_frontier->entries[0].axis,
                                          signal_frontier->entries[0].epoch);

    // Copy resolution entry from the op context if the submit callback
    // populated one (e.g., BUFFER_ALLOCA stores provisional→resolved mapping).
    iree_hal_remote_server_op_context_t* op_context =
        (iree_hal_remote_server_op_context_t*)submit_user_data;
    if (op_context && op_context->resolution_count > 0) {
      completion->resolution_count = op_context->resolution_count;
      completion->resolution = op_context->resolution;
    }

    completion->timepoint.callback = iree_hal_remote_server_on_command_complete;
    completion->timepoint.user_data = completion;
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)local_semaphore, /*minimum_value=*/1,
        &completion->timepoint);
  }

  if (!iree_status_is_ok(status)) {
    if (completion) {
      iree_hal_semaphore_release(completion->local_semaphore);
      iree_net_queue_channel_release(completion->queue_channel);
      iree_allocator_free(server->host_allocator, completion);
    }
  }

  iree_hal_semaphore_release(local_semaphore);
  return status;
}

//===----------------------------------------------------------------------===//
// Queue operation handlers
//===----------------------------------------------------------------------===//

// Submit callback for queue_barrier (empty COMMAND).
static iree_status_t iree_hal_remote_server_submit_barrier(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  (void)user_data;
  return iree_hal_device_queue_barrier(local_device,
                                       IREE_HAL_QUEUE_AFFINITY_ANY, wait_list,
                                       signal_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_remote_server_submit_buffer_fill(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_buffer_fill_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_FILL command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_buffer_fill_op_t));
  }
  const iree_hal_remote_buffer_fill_op_t* op =
      (const iree_hal_remote_buffer_fill_op_t*)context->command_data.data;

  iree_hal_remote_resource_id_t target_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->target_buffer_id);
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, target_id);
  if (!buffer) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "buffer not found for BUFFER_FILL target");
  }

  return iree_hal_device_queue_fill(local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                    wait_list, signal_list, buffer,
                                    op->target_offset, op->length, &op->pattern,
                                    (iree_host_size_t)op->pattern_length,
                                    (iree_hal_fill_flags_t)op->fill_flags);
}

static iree_status_t iree_hal_remote_server_submit_buffer_copy(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_buffer_copy_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_COPY command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_buffer_copy_op_t));
  }
  const iree_hal_remote_buffer_copy_op_t* op =
      (const iree_hal_remote_buffer_copy_op_t*)context->command_data.data;

  iree_hal_remote_resource_id_t source_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->source_buffer_id);
  iree_hal_buffer_t* source_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, source_id);
  if (!source_buffer) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "buffer not found for BUFFER_COPY source");
  }

  iree_hal_remote_resource_id_t target_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->target_buffer_id);
  iree_hal_buffer_t* target_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, target_id);
  if (!target_buffer) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "buffer not found for BUFFER_COPY target");
  }

  return iree_hal_device_queue_copy(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      source_buffer, op->source_offset, target_buffer, op->target_offset,
      op->length, (iree_hal_copy_flags_t)op->copy_flags);
}

static iree_status_t iree_hal_remote_server_submit_buffer_update(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_buffer_update_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_UPDATE command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_buffer_update_op_t));
  }
  const iree_hal_remote_buffer_update_op_t* op =
      (const iree_hal_remote_buffer_update_op_t*)context->command_data.data;

  // Inline source data follows the op struct.
  iree_host_size_t inline_data_offset =
      sizeof(iree_hal_remote_buffer_update_op_t);
  if (context->command_data.data_length < inline_data_offset + op->length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "BUFFER_UPDATE inline data truncated: need "
        "%" PRIu64 " bytes, have %" PRIhsz,
        op->length, context->command_data.data_length - inline_data_offset);
  }
  const void* inline_data = context->command_data.data + inline_data_offset;

  iree_hal_remote_resource_id_t target_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->target_buffer_id);
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, target_id);
  if (!buffer) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "buffer not found for BUFFER_UPDATE target");
  }

  return iree_hal_device_queue_update(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      inline_data, /*source_offset=*/0, buffer, op->target_offset, op->length,
      (iree_hal_update_flags_t)op->update_flags);
}

static iree_status_t iree_hal_remote_server_submit_buffer_alloca(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_buffer_alloca_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_ALLOCA command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_buffer_alloca_op_t));
  }
  const iree_hal_remote_buffer_alloca_op_t* op =
      (const iree_hal_remote_buffer_alloca_op_t*)context->command_data.data;

  // Translate wire buffer params to HAL buffer params.
  iree_hal_buffer_params_t params = {0};
  params.usage = (iree_hal_buffer_usage_t)op->params.usage;
  params.access = (iree_hal_memory_access_t)op->params.access;
  params.type = (iree_hal_memory_type_t)op->params.type;
  params.queue_affinity = (iree_hal_queue_affinity_t)op->params.queue_affinity;
  params.min_alignment = (iree_device_size_t)op->params.min_alignment;

  // Allocate on the local device. queue_alloca returns a buffer handle
  // immediately (synchronous allocation, async queue ordering).
  iree_hal_buffer_t* local_buffer = NULL;
  iree_status_t status = iree_hal_device_queue_alloca(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      (iree_hal_allocator_pool_t)op->pool, params,
      (iree_device_size_t)op->allocation_size,
      (iree_hal_alloca_flags_t)op->alloca_flags, &local_buffer);
  if (!iree_status_is_ok(status)) return status;

  // Assign the buffer to the resource table to get a canonical resolved ID.
  iree_hal_remote_resource_id_t resolved_id = 0;
  status = iree_hal_remote_resource_table_assign(
      &context->session_slot->resource_table,
      IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, local_buffer, &resolved_id);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(local_buffer);
    return status;
  }

  // Store the provisional→resolved mapping so subsequent commands referencing
  // the provisional ID can be resolved.
  status = iree_hal_remote_server_store_provisional(
      context->session_slot, op->provisional_buffer_id, resolved_id,
      context->session_slot->server->host_allocator);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(local_buffer);
    return status;
  }

  // Store the resolution entry in the op context so submit_command can
  // populate the completion's resolution field.
  context->resolution_count = 1;
  context->resolution.provisional_id = op->provisional_buffer_id;
  context->resolution.resolved_id = resolved_id;

  iree_hal_buffer_release(local_buffer);
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_server_submit_buffer_dealloca(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_buffer_dealloca_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_DEALLOCA command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_buffer_dealloca_op_t));
  }
  const iree_hal_remote_buffer_dealloca_op_t* op =
      (const iree_hal_remote_buffer_dealloca_op_t*)context->command_data.data;

  // Resolve the buffer ID (may be provisional).
  iree_hal_remote_resource_id_t resolved_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->buffer_id);
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, resolved_id);
  if (!buffer) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "buffer not found for BUFFER_DEALLOCA");
  }

  return iree_hal_device_queue_dealloca(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list, buffer,
      (iree_hal_dealloca_flags_t)op->dealloca_flags);
}

//===----------------------------------------------------------------------===//
// Control channel dispatch
//===----------------------------------------------------------------------===//

// Sends a control channel response. Builds envelope + response_prefix + body
// on the stack and sends via the session. The request envelope is used to echo
// the request_id and message_type.
static iree_status_t iree_hal_remote_server_send_response(
    iree_net_session_t* session,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_code_t status_code, const void* body,
    iree_host_size_t body_length) {
  iree_host_size_t response_length =
      sizeof(iree_hal_remote_control_envelope_t) +
      sizeof(iree_hal_remote_control_response_prefix_t) + body_length;

  // Stack-allocate up to a reasonable limit. All current responses are small.
  uint8_t response_storage[256];
  if (response_length > sizeof(response_storage)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "control response too large for stack buffer: "
                            "%" PRIhsz " bytes",
                            response_length);
  }
  memset(response_storage, 0, response_length);

  // Envelope.
  iree_hal_remote_control_envelope_t* envelope =
      (iree_hal_remote_control_envelope_t*)response_storage;
  envelope->message_type = request_envelope->message_type;
  envelope->message_flags = IREE_HAL_REMOTE_CONTROL_FLAG_IS_RESPONSE;
  envelope->request_id = request_envelope->request_id;

  // Response prefix.
  iree_hal_remote_control_response_prefix_t* prefix =
      (iree_hal_remote_control_response_prefix_t*)(response_storage +
                                                   sizeof(
                                                       iree_hal_remote_control_envelope_t));
  prefix->status_code = (uint32_t)status_code;

  // Body.
  if (body && body_length > 0) {
    memcpy(response_storage + sizeof(iree_hal_remote_control_envelope_t) +
               sizeof(iree_hal_remote_control_response_prefix_t),
           body, body_length);
  }

  iree_async_span_t span =
      iree_async_span_from_ptr(response_storage, response_length);
  iree_async_span_list_t payload = {&span, 1};
  return iree_net_session_send_control_data(session, /*flags=*/0, payload,
                                            /*operation_user_data=*/0);
}

// Maximum serialized status size that fits in the stack response buffer.
// The stack buffer is 256 bytes; envelope(16) + prefix(8) leaves 232 for body.
#define IREE_HAL_REMOTE_MAX_STATUS_WIRE_SIZE 232

// Sends an error response with the full serialized status as body.
// Consumes |status| (the caller must not use it after this call).
// The response prefix carries the status code for fast-path checking; the body
// carries the full wire-format status (source location, message, annotations).
static iree_status_t iree_hal_remote_server_send_error_response(
    iree_net_session_t* session,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_t status) {
  iree_status_code_t code = iree_status_code(status);

  // Compute serialized size and serialize if it fits in the stack buffer.
  iree_host_size_t wire_size = 0;
  iree_net_status_wire_size(status, &wire_size);

  iree_status_t send_status;
  if (wire_size <= IREE_HAL_REMOTE_MAX_STATUS_WIRE_SIZE) {
    uint8_t wire_buffer[IREE_HAL_REMOTE_MAX_STATUS_WIRE_SIZE];
    iree_status_t serialize_status = iree_net_status_wire_serialize(
        status, iree_make_byte_span(wire_buffer, wire_size));
    if (iree_status_is_ok(serialize_status)) {
      send_status = iree_hal_remote_server_send_response(
          session, request_envelope, code, wire_buffer, wire_size);
    } else {
      // Serialization failed; send code-only response.
      iree_status_ignore(serialize_status);
      send_status = iree_hal_remote_server_send_response(
          session, request_envelope, code, NULL, 0);
    }
  } else {
    // Status too large for stack buffer; send code-only response.
    send_status = iree_hal_remote_server_send_response(
        session, request_envelope, code, NULL, 0);
  }

  iree_status_ignore(status);
  return send_status;
}

// Handles BUFFER_ALLOC: allocates a buffer on the local device and assigns
// a resource slot in the session's table.
static iree_status_t iree_hal_remote_server_handle_buffer_alloc(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_alloc_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_ALLOC body too small: %" PRIhsz " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_alloc_request_t* request =
      (const iree_hal_remote_buffer_alloc_request_t*)body;

  // Convert wire params to HAL params.
  iree_hal_buffer_params_t params = {
      .usage = (iree_hal_buffer_usage_t)request->params.usage,
      .access = (iree_hal_memory_access_t)request->params.access,
      .type = (iree_hal_memory_type_t)request->params.type,
      .queue_affinity =
          (iree_hal_queue_affinity_t)request->params.queue_affinity,
      .min_alignment = (iree_device_size_t)request->params.min_alignment,
  };

  iree_device_size_t allocation_size =
      (iree_device_size_t)request->allocation_size;

  // Allocate on the local device.
  iree_hal_remote_server_t* server = entry->server;
  iree_hal_device_t* local_device = server->devices[0];
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(local_device);

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      allocator, params, allocation_size, &buffer);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(entry->session, envelope,
                                                      status);
  }

  // Assign a slot in the session's resource table.
  iree_hal_remote_resource_id_t resolved_id = 0;
  status = iree_hal_remote_resource_table_assign(
      &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, buffer,
      &resolved_id);
  // The table retains the buffer; release our allocation reference.
  iree_hal_buffer_release(buffer);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(entry->session, envelope,
                                                      status);
  }

  // Send success response with the resolved resource_id.
  iree_hal_remote_buffer_alloc_response_t response = {
      .resolved_id = resolved_id,
  };
  return iree_hal_remote_server_send_response(
      entry->session, envelope, IREE_STATUS_OK, &response, sizeof(response));
}

// Handles BUFFER_QUERY_HEAPS: queries the local device's memory heap topology
// and sends the descriptions back to the client.
static iree_status_t iree_hal_remote_server_handle_buffer_query_heaps(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  iree_hal_remote_server_t* server = entry->server;
  iree_hal_device_t* local_device = server->devices[0];
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(local_device);

  // Query heap count first. The HAL contract returns OUT_OF_RANGE when
  // capacity < count (the standard pre-sizing pattern). We use capacity=0
  // to trigger this and read the count from out_count.
  iree_host_size_t heap_count = 0;
  iree_status_t status =
      iree_hal_allocator_query_memory_heaps(allocator, 0, NULL, &heap_count);
  if (iree_status_code(status) == IREE_STATUS_OUT_OF_RANGE) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(entry->session, envelope,
                                                      status);
  }

  // Query full heap descriptions (stack-allocated for reasonable counts).
  iree_hal_allocator_memory_heap_t heaps_storage[16];
  if (heap_count > IREE_ARRAYSIZE(heaps_storage)) {
    return iree_hal_remote_server_send_error_response(
        entry->session, envelope,
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "too many heaps: %" PRIhsz, heap_count));
  }
  status = iree_hal_allocator_query_memory_heaps(allocator, heap_count,
                                                 heaps_storage, &heap_count);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(entry->session, envelope,
                                                      status);
  }

  // Build response: header + wire heap descriptions.
  uint8_t response_body[sizeof(iree_hal_remote_buffer_query_heaps_response_t) +
                        16 * sizeof(iree_hal_remote_memory_heap_t)];
  memset(response_body, 0, sizeof(response_body));

  iree_hal_remote_buffer_query_heaps_response_t* response_header =
      (iree_hal_remote_buffer_query_heaps_response_t*)response_body;
  response_header->heap_count = (uint16_t)heap_count;

  iree_hal_remote_memory_heap_t* wire_heaps =
      (iree_hal_remote_memory_heap_t*)(response_body +
                                       sizeof(
                                           iree_hal_remote_buffer_query_heaps_response_t));
  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    wire_heaps[i].type = (uint32_t)heaps_storage[i].type;
    wire_heaps[i].allowed_usage = (uint32_t)heaps_storage[i].allowed_usage;
    wire_heaps[i].max_allocation_size =
        (uint64_t)heaps_storage[i].max_allocation_size;
    wire_heaps[i].min_alignment = (uint64_t)heaps_storage[i].min_alignment;
  }

  iree_host_size_t response_body_length =
      sizeof(iree_hal_remote_buffer_query_heaps_response_t) +
      heap_count * sizeof(iree_hal_remote_memory_heap_t);
  return iree_hal_remote_server_send_response(entry->session, envelope,
                                              IREE_STATUS_OK, response_body,
                                              response_body_length);
}

// Sends a control channel response with a variable-length data payload using
// scatter-gather. The header (envelope + prefix + body_header) is
// stack-allocated; the data payload points directly at the source buffer
// (avoiding a copy). Used for BUFFER_MAP responses with inline buffer data.
static iree_status_t iree_hal_remote_server_send_response_with_data(
    iree_net_session_t* session,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_code_t status_code, const void* body_header,
    iree_host_size_t body_header_length, const void* data,
    iree_host_size_t data_length) {
  // Build envelope + prefix + body header on the stack.
  uint8_t header_storage[256];
  iree_host_size_t header_length =
      sizeof(iree_hal_remote_control_envelope_t) +
      sizeof(iree_hal_remote_control_response_prefix_t) + body_header_length;
  if (header_length > sizeof(header_storage)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "response header too large for stack buffer");
  }
  memset(header_storage, 0, header_length);

  iree_hal_remote_control_envelope_t* envelope =
      (iree_hal_remote_control_envelope_t*)header_storage;
  envelope->message_type = request_envelope->message_type;
  envelope->message_flags = IREE_HAL_REMOTE_CONTROL_FLAG_IS_RESPONSE;
  envelope->request_id = request_envelope->request_id;

  iree_hal_remote_control_response_prefix_t* prefix =
      (iree_hal_remote_control_response_prefix_t*)(header_storage +
                                                   sizeof(*envelope));
  prefix->status_code = (uint32_t)status_code;

  if (body_header && body_header_length > 0) {
    memcpy(header_storage + sizeof(*envelope) + sizeof(*prefix), body_header,
           body_header_length);
  }

  // Scatter-gather: header span + data span.
  iree_async_span_t spans[2] = {
      iree_async_span_from_ptr(header_storage, header_length),
      iree_async_span_from_ptr((void*)data, data_length),
  };
  iree_host_size_t span_count = data_length > 0 ? 2 : 1;
  iree_async_span_list_t payload = {spans, span_count};
  return iree_net_session_send_control_data(session, /*flags=*/0, payload,
                                            /*operation_user_data=*/0);
}

// Handles BUFFER_MAP: maps a buffer on the local device, reads the requested
// region, and returns the data inline in the response. No persistent
// server-side mapping state is created — the map/unmap is scoped to this
// handler.
static iree_status_t iree_hal_remote_server_handle_buffer_map(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_map_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_MAP body too small: %" PRIhsz " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_map_request_t* request =
      (const iree_hal_remote_buffer_map_request_t*)body;

  // Look up the buffer in the resource table.
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
          request->buffer_id);
  if (!buffer) {
    return iree_hal_remote_server_send_error_response(
        entry->session, envelope,
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "BUFFER_MAP: buffer_id 0x%016" PRIx64 " not found",
                         request->buffer_id));
  }
  iree_hal_memory_access_t memory_access =
      (iree_hal_memory_access_t)request->memory_access;
  iree_device_size_t offset = (iree_device_size_t)request->offset;
  iree_device_size_t length = (iree_device_size_t)request->length;

  // Only perform the local map+read if READ access was requested.
  if (iree_all_bits_set(memory_access, IREE_HAL_MEMORY_ACCESS_READ)) {
    iree_hal_buffer_mapping_t mapping;
    iree_status_t status = iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        offset, length, &mapping);
    if (!iree_status_is_ok(status)) {
      return iree_hal_remote_server_send_error_response(entry->session,
                                                        envelope, status);
    }

    // Send response header + inline data via scatter-gather.
    iree_hal_remote_buffer_map_response_t response = {
        .mapped_offset = offset,
        .mapped_length = mapping.contents.data_length,
    };
    iree_status_t send_status = iree_hal_remote_server_send_response_with_data(
        entry->session, envelope, IREE_STATUS_OK, &response, sizeof(response),
        mapping.contents.data, mapping.contents.data_length);

    iree_status_ignore(iree_hal_buffer_unmap_range(&mapping));
    return send_status;
  }

  // WRITE-only or DISCARD: no data to send, just acknowledge.
  iree_hal_remote_buffer_map_response_t response = {
      .mapped_offset = offset,
      .mapped_length = length,
  };
  return iree_hal_remote_server_send_response(
      entry->session, envelope, IREE_STATUS_OK, &response, sizeof(response));
}

// Handles BUFFER_UNMAP: writes inline data from the client into a buffer on
// the local device. Maps the buffer, copies the data, and responds with status.
static iree_status_t iree_hal_remote_server_handle_buffer_unmap(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_unmap_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_UNMAP body too small: %" PRIhsz " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_unmap_request_t* request =
      (const iree_hal_remote_buffer_unmap_request_t*)body;
  iree_device_size_t offset = (iree_device_size_t)request->offset;
  iree_device_size_t length = (iree_device_size_t)request->length;

  // Validate inline data is present.
  const uint8_t* data = body + sizeof(iree_hal_remote_buffer_unmap_request_t);
  iree_host_size_t data_length =
      body_length - sizeof(iree_hal_remote_buffer_unmap_request_t);
  if (data_length < length) {
    return iree_hal_remote_server_send_error_response(
        entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_UNMAP data truncated: %" PRIhsz
                         " bytes, expected %" PRIdsz,
                         data_length, length));
  }

  // Look up the buffer in the resource table.
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
          request->buffer_id);
  if (!buffer) {
    return iree_hal_remote_server_send_error_response(
        entry->session, envelope,
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "BUFFER_UNMAP: buffer_id 0x%016" PRIx64 " not found",
                         request->buffer_id));
  }

  // Map the buffer for writing and copy the client data in.
  iree_hal_buffer_mapping_t mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, offset, length, &mapping);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(entry->session, envelope,
                                                      status);
  }

  memcpy(mapping.contents.data, data, length);
  iree_status_ignore(iree_hal_buffer_unmap_range(&mapping));

  // Status-only response (no body).
  return iree_hal_remote_server_send_response(entry->session, envelope,
                                              IREE_STATUS_OK, NULL, 0);
}

// Handles RESOURCE_RELEASE_BATCH: releases resources by ID. Fire-and-forget
// (no response sent).
static iree_status_t iree_hal_remote_server_handle_resource_release_batch(
    iree_hal_remote_server_session_t* entry, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_resource_release_batch_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RESOURCE_RELEASE_BATCH body too small: %" PRIhsz
                            " bytes",
                            body_length);
  }

  const iree_hal_remote_resource_release_batch_t* batch =
      (const iree_hal_remote_resource_release_batch_t*)body;
  uint32_t resource_count = batch->resource_count;

  iree_host_size_t expected_size =
      sizeof(iree_hal_remote_resource_release_batch_t) +
      resource_count * sizeof(iree_hal_remote_resource_id_t);
  if (body_length < expected_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RESOURCE_RELEASE_BATCH truncated: %" PRIhsz
                            " bytes, expected "
                            "%" PRIhsz " for %u resources",
                            body_length, expected_size, resource_count);
  }

  const iree_hal_remote_resource_id_t* resource_ids =
      (const iree_hal_remote_resource_id_t*)(body +
                                             sizeof(
                                                 iree_hal_remote_resource_release_batch_t));
  for (uint32_t i = 0; i < resource_count; ++i) {
    iree_hal_remote_resource_table_release(&entry->resource_table,
                                           resource_ids[i]);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Queue channel callbacks
//===----------------------------------------------------------------------===//

// Server receives COMMAND frames from clients and dispatches to the local
// device. On completion (via semaphore timepoint), sends ADVANCE back.
static iree_status_t iree_hal_remote_server_on_command(
    void* user_data, uint32_t stream_id,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t command_data, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_server_session_t* session_slot =
      (iree_hal_remote_server_session_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!signal_frontier || signal_frontier->entry_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "COMMAND frame must have a signal frontier for completion tracking");
  }

  iree_status_t status;

  if (command_data.data_length == 0) {
    // Barrier operation (empty payload).
    status = iree_hal_remote_server_submit_command(
        session_slot, wait_frontier, signal_frontier,
        iree_hal_remote_server_submit_barrier, NULL);
  } else if (command_data.data_length >=
             sizeof(iree_hal_remote_queue_op_header_t)) {
    const iree_hal_remote_queue_op_header_t* op_header =
        (const iree_hal_remote_queue_op_header_t*)command_data.data;
    iree_hal_remote_server_op_context_t op_context = {
        .session_slot = session_slot,
        .command_data = command_data,
        .resolution_count = 0,
    };
    switch (op_header->type) {
      case IREE_HAL_REMOTE_QUEUE_OP_BUFFER_ALLOCA:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_buffer_alloca, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_BUFFER_DEALLOCA:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_buffer_dealloca, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_BUFFER_FILL:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_buffer_fill, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_BUFFER_COPY:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_buffer_copy, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_BUFFER_UPDATE:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_buffer_update, &op_context);
        break;
      default:
        status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "queue op type 0x%04x not implemented",
                                  op_header->type);
        break;
    }
  } else {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "COMMAND payload too short for op header: "
                              "%" PRIhsz " bytes",
                              command_data.data_length);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Server does not receive ADVANCE frames (only clients do).
static iree_status_t iree_hal_remote_server_on_advance(
    void* user_data, const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t advance_data, iree_async_buffer_lease_t* lease) {
  (void)user_data;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "server does not accept ADVANCE frames");
}

static void iree_hal_remote_server_on_queue_transport_error(
    void* user_data, iree_status_t status) {
  (void)user_data;
  // TODO(benvanik): propagate queue channel transport error to session.
  iree_status_ignore(status);
}

// Context passed to the endpoint_ready callback to identify which session
// slot should receive the queue channel.
typedef struct iree_hal_remote_server_endpoint_context_t {
  iree_hal_remote_server_t* server;
  iree_net_session_t* session;  // retained
  iree_allocator_t host_allocator;
} iree_hal_remote_server_endpoint_context_t;

static void iree_hal_remote_server_on_queue_endpoint_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_hal_remote_server_endpoint_context_t* context =
      (iree_hal_remote_server_endpoint_context_t*)user_data;
  iree_hal_remote_server_t* server = context->server;
  iree_net_session_t* session = context->session;
  iree_allocator_t host_allocator = context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Free the context first (we've captured what we need).
  iree_allocator_free(host_allocator, context);
  context = NULL;

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_net_session_release(session);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Find the session slot (under lock — stop() may be iterating).
  iree_slim_mutex_lock(&server->session_mutex);
  int32_t slot = iree_hal_remote_server_find_session_slot(server, session);
  iree_slim_mutex_unlock(&server->session_mutex);
  if (slot < 0) {
    // Session was removed while endpoint was opening.
    iree_net_session_release(session);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Create header pool and queue channel outside the lock (allocation, no
  // shared state).
  iree_async_buffer_pool_t* header_pool = NULL;
  status = iree_hal_remote_create_queue_header_pool(
      IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_COUNT,
      IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_SIZE, host_allocator,
      &header_pool);

  iree_net_queue_channel_t* queue_channel = NULL;
  if (iree_status_is_ok(status)) {
    iree_net_queue_channel_callbacks_t callbacks = {
        .on_command = iree_hal_remote_server_on_command,
        .on_advance = iree_hal_remote_server_on_advance,
        .on_transport_error = iree_hal_remote_server_on_queue_transport_error,
        .user_data = &server->sessions[slot],
    };

    status = iree_net_queue_channel_create(
        endpoint, IREE_NET_FRAME_SENDER_MAX_SPANS, header_pool, callbacks,
        host_allocator, &queue_channel);
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_queue_channel_activate(queue_channel);
  }

  if (iree_status_is_ok(status)) {
    // Re-verify the session is still in its slot before storing the channel.
    // Between the slot lookup above and now, stop() → remove_session() could
    // have cleared this slot.
    iree_slim_mutex_lock(&server->session_mutex);
    if (server->sessions[slot].session == session) {
      server->sessions[slot].queue_channel = queue_channel;
      queue_channel = NULL;  // Ownership transferred.
    }
    iree_slim_mutex_unlock(&server->session_mutex);

    if (queue_channel) {
      // Session was removed while we were setting up the channel.
      iree_net_queue_channel_release(queue_channel);
    }
  } else {
    // Channel create failed or wasn't reached — channel owns the pool if
    // it was created successfully, otherwise we must free the pool ourselves.
    if (queue_channel) {
      iree_net_queue_channel_release(queue_channel);
    } else {
      iree_async_buffer_pool_free(header_pool);
    }
    // Shut down the session so it transitions to a terminal state and frees
    // its slot. A session without a queue channel cannot process commands.
    iree_status_t shutdown_status = iree_net_session_shutdown(
        session, /*reason_code=*/0,
        iree_make_cstring_view("queue channel setup failed"));
    iree_status_ignore(shutdown_status);
    iree_status_ignore(status);
  }

  iree_net_session_release(session);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Session callbacks
//===----------------------------------------------------------------------===//

void iree_hal_remote_server_on_session_ready(
    void* user_data, iree_net_session_t* session,
    const iree_net_session_topology_t* remote_topology) {
  iree_hal_remote_server_session_t* entry =
      (iree_hal_remote_server_session_t*)user_data;
  iree_hal_remote_server_t* server = entry->server;
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)remote_topology;

  // If we're shutting down, immediately GOAWAY this newly-ready session.
  iree_slim_mutex_lock(&server->session_mutex);
  bool is_stopping = server->state != IREE_HAL_REMOTE_SERVER_STATE_RUNNING;
  iree_slim_mutex_unlock(&server->session_mutex);
  if (is_stopping) {
    iree_status_t goaway_status = iree_net_session_shutdown(
        session, /*reason_code=*/0, iree_make_cstring_view("server stopping"));
    iree_status_ignore(goaway_status);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Open the queue endpoint for HAL command dispatch. The endpoint_ready
  // callback creates the queue channel and activates it.
  iree_hal_remote_server_endpoint_context_t* context = NULL;
  iree_status_t status = iree_allocator_malloc(
      server->host_allocator, sizeof(*context), (void**)&context);
  if (iree_status_is_ok(status)) {
    context->server = server;
    context->session = session;
    iree_net_session_retain(session);
    context->host_allocator = server->host_allocator;

    iree_net_endpoint_ready_callback_t endpoint_callback = {
        .fn = iree_hal_remote_server_on_queue_endpoint_ready,
        .user_data = context,
    };
    status = iree_net_session_open_endpoint(session, endpoint_callback);
    if (!iree_status_is_ok(status)) {
      iree_net_session_release(session);
      iree_allocator_free(server->host_allocator, context);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_remote_server_on_session_goaway(void* user_data,
                                              iree_net_session_t* session,
                                              uint32_t reason_code,
                                              iree_string_view_t message) {
  iree_hal_remote_server_session_t* entry =
      (iree_hal_remote_server_session_t*)user_data;
  iree_hal_remote_server_t* server = entry->server;
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)reason_code;
  (void)message;
  // Client initiated graceful shutdown. The session transitions to DRAINING
  // and will eventually reach CLOSED, at which point we remove it.
  // For now, remove immediately since we have no application endpoints to
  // drain.
  iree_hal_remote_server_remove_session(server, session);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_remote_server_on_session_error(void* user_data,
                                             iree_net_session_t* session,
                                             iree_status_t status) {
  iree_hal_remote_server_session_t* entry =
      (iree_hal_remote_server_session_t*)user_data;
  iree_hal_remote_server_t* server = entry->server;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Log and consume the error.
  iree_status_ignore(status);

  // Remove the failed session from tracking.
  iree_hal_remote_server_remove_session(server, session);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_remote_server_on_control_data(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_server_session_t* entry =
      (iree_hal_remote_server_session_t*)user_data;

  // Parse control envelope.
  if (payload.data_length < sizeof(iree_hal_remote_control_envelope_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control data too small for envelope: %" PRIhsz
                            " bytes",
                            payload.data_length);
  }
  const iree_hal_remote_control_envelope_t* envelope =
      (const iree_hal_remote_control_envelope_t*)payload.data;

  // Server should not receive responses.
  if (envelope->message_flags & IREE_HAL_REMOTE_CONTROL_FLAG_IS_RESPONSE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "server received unexpected IS_RESPONSE flag");
  }

  // Message body starts after the envelope.
  const uint8_t* body =
      payload.data + sizeof(iree_hal_remote_control_envelope_t);
  iree_host_size_t body_length =
      payload.data_length - sizeof(iree_hal_remote_control_envelope_t);

  // Dispatch by message type.
  switch (envelope->message_type) {
    case IREE_HAL_REMOTE_CONTROL_BUFFER_ALLOC:
      return iree_hal_remote_server_handle_buffer_alloc(entry, envelope, body,
                                                        body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_MAP:
      return iree_hal_remote_server_handle_buffer_map(entry, envelope, body,
                                                      body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_UNMAP:
      return iree_hal_remote_server_handle_buffer_unmap(entry, envelope, body,
                                                        body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_QUERY_HEAPS:
      return iree_hal_remote_server_handle_buffer_query_heaps(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_RESOURCE_RELEASE_BATCH:
      return iree_hal_remote_server_handle_resource_release_batch(entry, body,
                                                                  body_length);
    default:
      // For request/response messages, send an UNIMPLEMENTED error back.
      // For fire-and-forget messages, just return an error.
      if (!(envelope->message_flags &
            IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET)) {
        return iree_hal_remote_server_send_error_response(
            entry->session, envelope,
            iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "unhandled control message type 0x%04x",
                             envelope->message_type));
      }
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unhandled control message type 0x%04x",
                              envelope->message_type);
  }
}
