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

  // Snapshot the resource table for cleanup outside the lock.
  iree_hal_remote_resource_table_t resource_table;
  memset(&resource_table, 0, sizeof(resource_table));

  iree_slim_mutex_lock(&server->session_mutex);
  int32_t slot = iree_hal_remote_server_find_session_slot(server, session);
  if (slot >= 0) {
    // Snapshot references to clean up outside the lock.
    queue_channel = server->sessions[slot].queue_channel;
    server->sessions[slot].queue_channel = NULL;

    resource_table = server->sessions[slot].resource_table;
    memset(&server->sessions[slot].resource_table, 0,
           sizeof(server->sessions[slot].resource_table));

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

  // Detach the queue channel from its endpoint before releasing the session.
  // Barrier completions may hold retained references to the channel that
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
// Barrier completion
//===----------------------------------------------------------------------===//

// Context for a pending barrier completion on the server. Heap-allocated and
// freed in the timepoint callback after sending the ADVANCE frame.
typedef struct iree_hal_remote_server_barrier_completion_t {
  iree_async_semaphore_timepoint_t timepoint;
  iree_net_queue_channel_t* queue_channel;  // retained
  iree_hal_semaphore_t* local_semaphore;    // retained
  iree_allocator_t host_allocator;
  iree_async_single_frontier_t signal_frontier;
} iree_hal_remote_server_barrier_completion_t;

// Fired by the local-task device's semaphore when retire_cmd completes.
// Sends ADVANCE back to the client and cleans up.
static void iree_hal_remote_server_on_barrier_complete(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_remote_server_barrier_completion_t* completion =
      (iree_hal_remote_server_barrier_completion_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_status_is_ok(status)) {
    iree_async_frontier_t* signal_frontier =
        iree_async_single_frontier_as_frontier(&completion->signal_frontier);
    iree_async_span_list_t empty_payload = {NULL, 0};
    iree_status_t send_status = iree_net_queue_channel_send_advance(
        completion->queue_channel, signal_frontier, empty_payload,
        /*operation_user_data=*/0);
    iree_status_ignore(send_status);
  } else {
    iree_status_ignore(status);
  }

  iree_net_queue_channel_release(completion->queue_channel);
  iree_hal_semaphore_release(completion->local_semaphore);
  iree_allocator_free(completion->host_allocator, completion);

  IREE_TRACE_ZONE_END(z0);
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
  iree_hal_remote_server_t* server = session_slot->server;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Currently only barrier operations are supported (empty command_data).
  // The signal frontier tells us what epoch to echo back in the ADVANCE frame.
  if (!signal_frontier || signal_frontier->entry_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "COMMAND frame must have a signal frontier for completion tracking");
  }

  // TODO(benvanik): select device based on queue affinity from the command.
  iree_hal_device_t* local_device = server->devices[0];

  // Create a local HAL semaphore (initial_value=0) for completion tracking.
  iree_hal_semaphore_t* local_semaphore = NULL;
  iree_status_t status = iree_hal_semaphore_create(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &local_semaphore);

  // Execute barrier on the local device: signal local_semaphore to 1.
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t signal_list = {
        .count = 1,
        .semaphores = &local_semaphore,
        .payload_values = (uint64_t[]){1},
    };
    status =
        iree_hal_device_queue_barrier(local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                      iree_hal_semaphore_list_empty(),
                                      signal_list, IREE_HAL_EXECUTE_FLAG_NONE);
  }

  // Allocate completion context and acquire a timepoint on the local semaphore.
  // When retire_cmd signals the semaphore to 1, the timepoint fires and sends
  // ADVANCE back to the client.
  iree_hal_remote_server_barrier_completion_t* completion = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(server->host_allocator, sizeof(*completion),
                                   (void**)&completion);
  }

  if (iree_status_is_ok(status)) {
    memset(completion, 0, sizeof(*completion));
    completion->host_allocator = server->host_allocator;

    // Retain the queue channel from the session slot that dispatched this
    // callback. The completion context outlives the on_command call, so we
    // must hold a reference.
    completion->queue_channel = session_slot->queue_channel;
    iree_net_queue_channel_retain(completion->queue_channel);
  }

  if (iree_status_is_ok(status)) {
    completion->local_semaphore = local_semaphore;
    iree_hal_semaphore_retain(local_semaphore);

    // Deep copy the signal frontier into the completion context.
    // Barrier operations send single-entry frontiers; assert that invariant.
    IREE_ASSERT(signal_frontier->entry_count == 1);
    iree_async_single_frontier_initialize(&completion->signal_frontier,
                                          signal_frontier->entries[0].axis,
                                          signal_frontier->entries[0].epoch);

    // Acquire timepoint: callback fires when local_semaphore reaches 1.
    completion->timepoint.callback = iree_hal_remote_server_on_barrier_complete;
    completion->timepoint.user_data = completion;
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)local_semaphore, /*minimum_value=*/1,
        &completion->timepoint);
  }

  if (!iree_status_is_ok(status)) {
    // Cleanup on failure.
    if (completion) {
      iree_hal_semaphore_release(completion->local_semaphore);
      iree_net_queue_channel_release(completion->queue_channel);
      iree_allocator_free(server->host_allocator, completion);
    }
  }

  iree_hal_semaphore_release(local_semaphore);

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

    status = iree_net_session_open_endpoint(
        session, iree_hal_remote_server_on_queue_endpoint_ready, context);
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
