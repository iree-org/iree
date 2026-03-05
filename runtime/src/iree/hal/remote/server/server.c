// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/server.h"

#include "iree/hal/remote/server/api.h"
#include "iree/net/transport_factory.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_server_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_remote_server_options_initialize(
    iree_hal_remote_server_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->transport_factory = NULL;
  out_options->bind_address = iree_string_view_empty();
  out_options->max_connections = 0;           // Use default (16).
  out_options->max_control_message_size = 0;  // Use default (64KB).
  out_options->max_queue_frame_size = 0;      // Use default (64KB).
  out_options->flags = IREE_HAL_REMOTE_SERVER_FLAG_NONE;
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
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_remote_server_t
//===----------------------------------------------------------------------===//

// Server state machine.
typedef enum iree_hal_remote_server_state_e {
  // Server created but not started.
  IREE_HAL_REMOTE_SERVER_STATE_STOPPED = 0,
  // Server is starting (binding to address).
  IREE_HAL_REMOTE_SERVER_STATE_STARTING,
  // Server is running and accepting connections.
  IREE_HAL_REMOTE_SERVER_STATE_RUNNING,
  // Server is stopping (closing connections).
  IREE_HAL_REMOTE_SERVER_STATE_STOPPING,
  // Server encountered an unrecoverable error.
  IREE_HAL_REMOTE_SERVER_STATE_ERROR,
} iree_hal_remote_server_state_t;

struct iree_hal_remote_server_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Configuration options.
  iree_hal_remote_server_options_t options;

  // The wrapped local device that handles actual operations.
  iree_hal_device_t* wrapped_device;

  // Current server state.
  iree_hal_remote_server_state_t state;

  // + trailing bind_address string storage
};

IREE_API_EXPORT iree_status_t iree_hal_remote_server_create(
    const iree_hal_remote_server_options_t* options,
    iree_hal_device_t* wrapped_device, iree_allocator_t host_allocator,
    iree_hal_remote_server_t** out_server) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(wrapped_device);
  IREE_ASSERT_ARGUMENT(out_server);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_server = NULL;

  // Verify options before allocating resources.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_remote_server_options_verify(options));

  // Calculate layout with trailing storage for strings.
  iree_hal_remote_server_t* server = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t bind_address_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(*server), &total_size,
              IREE_STRUCT_FIELD_ALIGNED(options->bind_address.size, char, 1,
                                        &bind_address_offset)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&server));

  iree_atomic_ref_count_init(&server->ref_count);
  server->host_allocator = host_allocator;

  // Copy options and strings to trailing storage.
  server->options = *options;
  iree_net_transport_factory_retain(options->transport_factory);
  iree_string_view_append_to_buffer(options->bind_address,
                                    &server->options.bind_address,
                                    (char*)server + bind_address_offset);

  // Retain the wrapped device.
  iree_hal_device_retain(wrapped_device);
  server->wrapped_device = wrapped_device;

  server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;

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

IREE_API_EXPORT void iree_hal_remote_server_release(
    iree_hal_remote_server_t* server) {
  if (IREE_LIKELY(server) &&
      iree_atomic_ref_count_dec(&server->ref_count) == 1) {
    iree_allocator_t host_allocator = server->host_allocator;
    IREE_TRACE_ZONE_BEGIN(z0);

    // Release the transport factory and wrapped device.
    iree_net_transport_factory_release(server->options.transport_factory);
    iree_hal_device_release(server->wrapped_device);

    iree_allocator_free(host_allocator, server);

    IREE_TRACE_ZONE_END(z0);
  }
}

IREE_API_EXPORT iree_status_t
iree_hal_remote_server_start(iree_hal_remote_server_t* server) {
  IREE_ASSERT_ARGUMENT(server);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (server->state == IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  if (server->state != IREE_HAL_REMOTE_SERVER_STATE_STOPPED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "server not in stopped state");
  }

  server->state = IREE_HAL_REMOTE_SERVER_STATE_STARTING;

  // Bind to the configured address and start listening.
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "server networking not yet "
                                          "implemented");

  if (iree_status_is_ok(status)) {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_RUNNING;
  } else {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_ERROR;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_server_stop(
    iree_hal_remote_server_t* server, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(server);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (server->state == IREE_HAL_REMOTE_SERVER_STATE_STOPPED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  if (server->state != IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "server not in running state");
  }

  server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPING;

  // Close all active connections and stop listening.
  // Block until all connections are closed or timeout expires.
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "server networking not yet "
                                          "implemented");

  if (iree_status_is_ok(status)) {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;
  } else {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_ERROR;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_remote_server_run(iree_hal_remote_server_t* server) {
  IREE_ASSERT_ARGUMENT(server);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Start the server if not already running.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_remote_server_start(server));

  // Run the proactor event loop until stopped.
  // The server is built on iree/async/proactor.h for fully async I/O.
  // Shutdown is handled through the proactor's cancellation mechanism.
  // TODO: integrate with proactor - for now this is a placeholder loop.
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         server->state == IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
    status = iree_hal_remote_server_poll(server, iree_infinite_timeout());
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_server_poll(
    iree_hal_remote_server_t* server, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(server);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (server->state != IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "server not running");
  }

  // Poll for network events:
  //   - New connections on the listener socket.
  //   - Incoming data on existing connections.
  //   - Connection closures or errors.
  //
  // Process events and dispatch operations to the wrapped device.
  // Return DEADLINE_EXCEEDED if no events occurred within the timeout.

  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "server networking not yet "
                                          "implemented");

  IREE_TRACE_ZONE_END(z0);
  return status;
}
