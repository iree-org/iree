// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/server.h"

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier.h"
#include "iree/async/semaphore.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/transport_factory.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_server_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_remote_server_options_initialize(
    iree_hal_remote_server_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
}

IREE_API_EXPORT iree_status_t iree_hal_remote_server_options_parse(
    iree_hal_remote_server_options_t* options, iree_string_pair_list_t params) {
  for (iree_host_size_t i = 0; i < params.count; ++i) {
    iree_string_view_t key = params.pairs[i].key;
    iree_string_view_t value = params.pairs[i].value;

    if (iree_string_view_equal(key, IREE_SV("bind"))) {
      options->bind_address = value;
    } else if (iree_string_view_equal(key, IREE_SV("max_connections"))) {
      uint32_t max_connections = 0;
      if (!iree_string_view_atoi_uint32(value, &max_connections)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid max_connections value");
      }
      options->max_connections = max_connections;
    } else if (iree_string_view_equal(key, IREE_SV("rdma"))) {
      if (iree_string_view_equal(value, IREE_SV("true"))) {
        options->flags |= IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA;
      } else if (iree_string_view_equal(value, IREE_SV("false"))) {
        options->flags &= ~IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rdma must be 'true' or 'false'");
      }
    } else if (iree_string_view_equal(key, IREE_SV("trace"))) {
      if (iree_string_view_equal(value, IREE_SV("true"))) {
        options->flags |= IREE_HAL_REMOTE_SERVER_FLAG_TRACE_SERVER_OPS;
      } else if (iree_string_view_equal(value, IREE_SV("false"))) {
        options->flags &= ~IREE_HAL_REMOTE_SERVER_FLAG_TRACE_SERVER_OPS;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "trace must be 'true' or 'false'");
      }
    }
    // Unknown parameters are ignored for forward compatibility.
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_server_options_verify(
    const iree_hal_remote_server_options_t* options) {
  if (!options->transport_factory) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "transport_factory is required");
  }
  if (iree_string_view_is_empty(options->bind_address)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bind_address is required");
  }
  if (!options->local_topology) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local_topology is required");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_remote_server_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_remote_server_create(
    const iree_hal_remote_server_options_t* options,
    iree_hal_device_t* const* devices, iree_host_size_t device_count,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_async_buffer_pool_t* recv_pool, iree_allocator_t host_allocator,
    iree_hal_remote_server_t** out_server) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(devices);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  IREE_ASSERT_ARGUMENT(recv_pool);
  IREE_ASSERT_ARGUMENT(out_server);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_server = NULL;

  if (device_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one device is required");
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_remote_server_options_verify(options));

  uint32_t max_connections = options->max_connections
                                 ? options->max_connections
                                 : IREE_HAL_REMOTE_DEFAULT_MAX_CONNECTIONS;
  uint32_t axis_count = options->local_topology->axis_count;

  // Calculate trailing storage layout.
  iree_host_size_t total_size = 0;
  iree_host_size_t bind_address_offset = 0;
  iree_host_size_t axes_offset = 0;
  iree_host_size_t epochs_offset = 0;
  iree_host_size_t devices_offset = 0;
  iree_host_size_t sessions_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          sizeof(iree_hal_remote_server_t), &total_size,
          IREE_STRUCT_FIELD(options->bind_address.size, char,
                            &bind_address_offset),
          IREE_STRUCT_FIELD_ALIGNED(axis_count, iree_async_axis_t,
                                    iree_alignof(iree_async_axis_t),
                                    &axes_offset),
          IREE_STRUCT_FIELD(axis_count, uint64_t, &epochs_offset),
          IREE_STRUCT_FIELD(device_count, iree_hal_device_t*, &devices_offset),
          IREE_STRUCT_FIELD(max_connections, iree_hal_remote_server_session_t,
                            &sessions_offset)));

  iree_hal_remote_server_t* server = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&server));

  iree_atomic_ref_count_init(&server->ref_count);
  server->host_allocator = host_allocator;

  // Copy options and bind_address to trailing storage.
  server->options = *options;
  server->options.max_connections = max_connections;
  iree_string_view_append_to_buffer(options->bind_address,
                                    &server->options.bind_address,
                                    (char*)server + bind_address_offset);

  // Copy topology arrays to trailing storage.
  iree_async_axis_t* local_axes =
      (iree_async_axis_t*)((uint8_t*)server + axes_offset);
  uint64_t* local_epochs = (uint64_t*)((uint8_t*)server + epochs_offset);
  memcpy(local_axes, options->local_topology->axes,
         axis_count * sizeof(iree_async_axis_t));
  memcpy(local_epochs, options->local_topology->current_epochs,
         axis_count * sizeof(uint64_t));
  server->local_topology.axes = local_axes;
  server->local_topology.current_epochs = local_epochs;
  server->local_topology.axis_count = axis_count;
  server->local_topology.machine_index = options->local_topology->machine_index;
  server->local_topology.session_epoch = options->local_topology->session_epoch;
  memset(server->local_topology.reserved, 0,
         sizeof(server->local_topology.reserved));

  // Clear topology pointer in options (we own the copy now, not the original).
  server->options.local_topology = NULL;

  // Retain transport factory.
  iree_net_transport_factory_retain(options->transport_factory);

  // Copy and retain devices.
  server->devices = (iree_hal_device_t**)((uint8_t*)server + devices_offset);
  server->device_count = device_count;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    server->devices[i] = devices[i];
    iree_hal_device_retain(devices[i]);
  }

  // Borrow infrastructure.
  server->proactor = proactor;
  server->frontier_tracker = frontier_tracker;
  server->recv_pool = recv_pool;

  // Initialize session tracking.
  server->sessions =
      (iree_hal_remote_server_session_t*)((uint8_t*)server + sessions_offset);
  memset(server->sessions, 0,
         max_connections * sizeof(iree_hal_remote_server_session_t));
  server->active_session_count = 0;
  server->next_session_id = 1;

  server->listener = NULL;
  server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;
  memset(&server->stopped_callback, 0, sizeof(server->stopped_callback));

  *out_server = server;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_remote_server_retain(
    iree_hal_remote_server_t* server) {
  if (IREE_LIKELY(server)) {
    iree_atomic_ref_count_inc(&server->ref_count);
  }
}

static void iree_hal_remote_server_destroy(iree_hal_remote_server_t* server) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = server->host_allocator;

  // Release all active sessions (should be empty if stop() was called).
  // Release queue channels before sessions (channels reference endpoints
  // owned by sessions). Clear each slot before releasing to prevent
  // re-entrancy: if release drops the last ref and the session's destructor
  // synchronously fires an error callback, on_session_error → remove_session
  // must not find the session still in the slot.
  for (uint32_t i = 0; i < server->options.max_connections; ++i) {
    iree_net_queue_channel_release(server->sessions[i].queue_channel);
    server->sessions[i].queue_channel = NULL;
    iree_async_buffer_pool_free(server->sessions[i].queue_header_pool);
    server->sessions[i].queue_header_pool = NULL;

    iree_net_session_t* session = server->sessions[i].session;
    server->sessions[i].session = NULL;
    iree_net_session_release(session);
  }

  // Free listener if still alive (shouldn't be if stop() was called).
  if (server->listener) {
    iree_net_listener_free(server->listener);
    server->listener = NULL;
  }

  // Release retained objects.
  iree_net_transport_factory_release(server->options.transport_factory);
  for (iree_host_size_t i = 0; i < server->device_count; ++i) {
    iree_hal_device_release(server->devices[i]);
  }

  iree_allocator_free(host_allocator, server);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_remote_server_release(
    iree_hal_remote_server_t* server) {
  if (IREE_LIKELY(server) &&
      iree_atomic_ref_count_dec(&server->ref_count) == 1) {
    iree_hal_remote_server_destroy(server);
  }
}

//===----------------------------------------------------------------------===//
// Session management
//===----------------------------------------------------------------------===//

// Finds a free session slot. Returns the slot index or -1 if full.
static int32_t iree_hal_remote_server_find_free_slot(
    iree_hal_remote_server_t* server) {
  for (uint32_t i = 0; i < server->options.max_connections; ++i) {
    if (!server->sessions[i].session) return (int32_t)i;
  }
  return -1;
}

// Finds the slot holding the given session. Returns -1 if not found.
static int32_t iree_hal_remote_server_find_session_slot(
    iree_hal_remote_server_t* server, iree_net_session_t* session) {
  for (uint32_t i = 0; i < server->options.max_connections; ++i) {
    if (server->sessions[i].session == session) return (int32_t)i;
  }
  return -1;
}

// Removes a session from the server's tracking. Called when a session reaches
// a terminal state (CLOSED or ERROR).
static void iree_hal_remote_server_remove_session(
    iree_hal_remote_server_t* server, iree_net_session_t* session) {
  int32_t slot = iree_hal_remote_server_find_session_slot(server, session);
  if (slot < 0) return;  // Already removed (e.g., double callback).

  // Release queue channel before session (channel references the endpoint
  // which is owned by the session's connection).
  iree_net_queue_channel_release(server->sessions[slot].queue_channel);
  server->sessions[slot].queue_channel = NULL;
  iree_async_buffer_pool_free(server->sessions[slot].queue_header_pool);
  server->sessions[slot].queue_header_pool = NULL;

  server->sessions[slot].session = NULL;
  server->sessions[slot].session_id = 0;
  --server->active_session_count;
  iree_net_session_release(session);

  // If we're stopping and this was the last session, check if shutdown is
  // complete.
  if (server->state == IREE_HAL_REMOTE_SERVER_STATE_STOPPING &&
      server->active_session_count == 0 && !server->listener) {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;
    if (server->stopped_callback.fn) {
      server->stopped_callback.fn(server->stopped_callback.user_data);
    }
  }
}

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

  // Find the session slot.
  int32_t slot = iree_hal_remote_server_find_session_slot(server, session);
  if (slot < 0) {
    // Session was removed while endpoint was opening.
    iree_net_session_release(session);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Create header pool.
  iree_async_buffer_pool_t* header_pool = NULL;
  status = iree_hal_remote_create_queue_header_pool(
      IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_COUNT,
      IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_SIZE, host_allocator,
      &header_pool);

  // Create queue channel.
  iree_net_queue_channel_t* queue_channel = NULL;
  if (iree_status_is_ok(status)) {
    iree_net_queue_channel_callbacks_t callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.on_command = iree_hal_remote_server_on_command;
    callbacks.on_advance = iree_hal_remote_server_on_advance;
    callbacks.on_transport_error =
        iree_hal_remote_server_on_queue_transport_error;
    callbacks.user_data = &server->sessions[slot];

    status = iree_net_queue_channel_create(
        endpoint, IREE_NET_FRAME_SENDER_MAX_SPANS, header_pool, callbacks,
        host_allocator, &queue_channel);
  }

  // Activate.
  if (iree_status_is_ok(status)) {
    status = iree_net_queue_channel_activate(queue_channel);
  }

  if (iree_status_is_ok(status)) {
    server->sessions[slot].queue_channel = queue_channel;
    server->sessions[slot].queue_header_pool = header_pool;
  } else {
    iree_net_queue_channel_release(queue_channel);
    iree_async_buffer_pool_free(header_pool);
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

static void iree_hal_remote_server_on_session_ready(
    void* user_data, iree_net_session_t* session,
    const iree_net_session_topology_t* remote_topology) {
  iree_hal_remote_server_t* server = (iree_hal_remote_server_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)remote_topology;

  // If we're shutting down, immediately GOAWAY this newly-ready session.
  if (server->state == IREE_HAL_REMOTE_SERVER_STATE_STOPPING) {
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

static void iree_hal_remote_server_on_session_goaway(
    void* user_data, iree_net_session_t* session, uint32_t reason_code,
    iree_string_view_t message) {
  iree_hal_remote_server_t* server = (iree_hal_remote_server_t*)user_data;
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

static void iree_hal_remote_server_on_session_error(void* user_data,
                                                    iree_net_session_t* session,
                                                    iree_status_t status) {
  iree_hal_remote_server_t* server = (iree_hal_remote_server_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Log and consume the error.
  iree_status_ignore(status);

  // Remove the failed session from tracking.
  iree_hal_remote_server_remove_session(server, session);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_remote_server_on_control_data(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  // HAL command dispatch will be wired here. For now, acknowledge receipt
  // without processing. This is the steady-state hot path for inline command
  // buffer recordings and device queries.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "HAL command dispatch not yet implemented");
}

static void iree_hal_remote_server_on_accept(
    void* user_data, iree_status_t status, iree_net_connection_t* connection) {
  iree_hal_remote_server_t* server = (iree_hal_remote_server_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reject if not running.
  if (iree_status_is_ok(status) &&
      server->state != IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  }

  // Find a free session slot.
  int32_t slot = -1;
  if (iree_status_is_ok(status)) {
    slot = iree_hal_remote_server_find_free_slot(server);
    if (slot < 0) {
      status = iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
  }

  // Assign a session ID and create the server-side session.
  iree_net_session_t* session = NULL;
  uint64_t session_id = 0;
  if (iree_status_is_ok(status)) {
    session_id = server->next_session_id++;

    iree_net_session_options_t session_options =
        iree_net_session_options_default();
    session_options.local_topology = server->local_topology;
    session_options.session_id = session_id;

    iree_net_session_callbacks_t callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.on_ready = iree_hal_remote_server_on_session_ready;
    callbacks.on_goaway = iree_hal_remote_server_on_session_goaway;
    callbacks.on_error = iree_hal_remote_server_on_session_error;
    callbacks.on_control_data = iree_hal_remote_server_on_control_data;
    callbacks.user_data = server;

    status = iree_net_session_accept(
        connection, server->proactor, server->frontier_tracker,
        &session_options, callbacks, server->host_allocator, &session);
  }

  // Release the accept callback's connection reference. The session retains it
  // internally if accept succeeded; connection is NULL when the transport
  // delivers an error status.
  iree_net_connection_release(connection);

  if (iree_status_is_ok(status)) {
    // Track the session.
    server->sessions[slot].server = server;
    server->sessions[slot].session = session;
    server->sessions[slot].session_id = session_id;
    ++server->active_session_count;
  } else {
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_remote_server_on_listener_stopped(void* user_data) {
  iree_hal_remote_server_t* server = (iree_hal_remote_server_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Listener has fully stopped — safe to free it.
  iree_net_listener_free(server->listener);
  server->listener = NULL;

  // Check if shutdown is complete (no listener and no active sessions).
  if (server->state == IREE_HAL_REMOTE_SERVER_STATE_STOPPING &&
      server->active_session_count == 0) {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;
    if (server->stopped_callback.fn) {
      server->stopped_callback.fn(server->stopped_callback.user_data);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t
iree_hal_remote_server_start(iree_hal_remote_server_t* server) {
  IREE_ASSERT_ARGUMENT(server);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (server->state != IREE_HAL_REMOTE_SERVER_STATE_STOPPED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "server must be in STOPPED state to start "
                            "(current state: %d)",
                            (int)server->state);
  }

  // Create the listener via the transport factory.
  iree_status_t status = iree_net_transport_factory_create_listener(
      server->options.transport_factory, server->options.bind_address,
      server->proactor, server->recv_pool, iree_hal_remote_server_on_accept,
      server, server->host_allocator, &server->listener);

  if (iree_status_is_ok(status)) {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_RUNNING;
  } else {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_ERROR;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_server_stop(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_stopped_callback_t callback) {
  IREE_ASSERT_ARGUMENT(server);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (server->state != IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "server must be in RUNNING state to stop "
                            "(current state: %d)",
                            (int)server->state);
  }

  server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPING;
  server->stopped_callback = callback;

  // Send GOAWAY to all active sessions.
  for (uint32_t i = 0; i < server->options.max_connections; ++i) {
    if (server->sessions[i].session) {
      iree_net_session_state_t session_state =
          iree_net_session_state(server->sessions[i].session);
      if (session_state == IREE_NET_SESSION_STATE_OPERATIONAL) {
        iree_status_t goaway_status = iree_net_session_shutdown(
            server->sessions[i].session, /*reason_code=*/0,
            iree_make_cstring_view("server stopping"));
        // Shutdown may fail if the session is already in a terminal state
        // (race with async error). That's fine — the session will be removed
        // when its error/goaway callback fires.
        iree_status_ignore(goaway_status);
      }
    }
  }

  // Stop the listener (no more accepts).
  if (server->listener) {
    iree_net_listener_stopped_callback_t listener_callback;
    listener_callback.fn = iree_hal_remote_server_on_listener_stopped;
    listener_callback.user_data = server;
    iree_status_t stop_status =
        iree_net_listener_stop(server->listener, listener_callback);
    if (!iree_status_is_ok(stop_status)) {
      // Listener stop failed — free it directly and proceed.
      iree_status_ignore(stop_status);
      iree_net_listener_free(server->listener);
      server->listener = NULL;
    }
  }

  // If there are no sessions and no listener, complete immediately.
  if (server->active_session_count == 0 && !server->listener) {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;
    if (server->stopped_callback.fn) {
      server->stopped_callback.fn(server->stopped_callback.user_data);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_remote_server_query_bound_address(
    iree_hal_remote_server_t* server, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  IREE_ASSERT_ARGUMENT(server);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_address);

  if (server->state != IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "server must be in RUNNING state to query address "
                            "(current state: %d)",
                            (int)server->state);
  }

  if (!server->listener) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "server is RUNNING but has no listener");
  }

  return iree_net_listener_query_bound_address(
      server->listener, buffer_capacity, buffer, out_address);
}
