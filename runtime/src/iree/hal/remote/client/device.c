// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/device.h"

#include "iree/hal/remote/client/api.h"
#include "iree/net/session.h"
#include "iree/net/transport_factory.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_device_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_remote_client_device_options_initialize(
    iree_hal_remote_client_device_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_options_parse(
    iree_hal_remote_client_device_options_t* options,
    iree_string_pair_list_t params) {
  for (iree_host_size_t i = 0; i < params.count; ++i) {
    iree_string_view_t key = params.pairs[i].key;
    iree_string_view_t value = params.pairs[i].value;

    if (iree_string_view_equal(key, IREE_SV("server"))) {
      options->server_address = value;
    } else if (iree_string_view_equal(key, IREE_SV("connect_timeout"))) {
      uint32_t timeout_ms = 0;
      if (!iree_string_view_atoi_uint32(value, &timeout_ms)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid connect_timeout value");
      }
      options->connect_timeout_ns =
          (iree_duration_t)timeout_ms * 1000000;  // ms to ns.
    } else if (iree_string_view_equal(key, IREE_SV("rdma"))) {
      if (iree_string_view_equal(value, IREE_SV("true"))) {
        options->flags |= IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_ENABLE_RDMA;
      } else if (iree_string_view_equal(value, IREE_SV("false"))) {
        options->flags &= ~IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_ENABLE_RDMA;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rdma must be 'true' or 'false'");
      }
    } else if (iree_string_view_equal(key, IREE_SV("trace"))) {
      if (iree_string_view_equal(value, IREE_SV("true"))) {
        options->flags |= IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_TRACE_REMOTE_OPS;
      } else if (iree_string_view_equal(value, IREE_SV("false"))) {
        options->flags &= ~IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_TRACE_REMOTE_OPS;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "trace must be 'true' or 'false'");
      }
    }
    // Unknown parameters are ignored for forward compatibility.
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_device_options_verify(
    const iree_hal_remote_client_device_options_t* options) {
  if (!options->transport_factory) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "transport_factory is required");
  }
  if (iree_string_view_is_empty(options->server_address)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "server_address is required");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_device_t
//===----------------------------------------------------------------------===//

static const iree_hal_device_vtable_t iree_hal_remote_client_device_vtable;

typedef struct iree_hal_remote_client_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Device configuration options.
  iree_hal_remote_client_device_options_t options;

  // Borrowed infrastructure (must outlive the device).
  iree_async_proactor_t* proactor;
  iree_async_frontier_tracker_t* frontier_tracker;
  iree_async_buffer_pool_t* recv_pool;

  // Active session (NULL when disconnected).
  iree_net_session_t* session;

  // Current connection state.
  iree_hal_remote_client_device_state_t state;

  // Pending connect callback (valid during CONNECTING state).
  iree_hal_remote_client_device_connected_callback_t connect_callback;

  // Trailing storage layout:
  //   char identifier_storage[identifier.size]
  //   char server_address_storage[options.server_address.size]
} iree_hal_remote_client_device_t;

static iree_hal_remote_client_device_t* iree_hal_remote_client_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_remote_client_device_vtable);
  return (iree_hal_remote_client_device_t*)base_value;
}

//===----------------------------------------------------------------------===//
// Session callbacks
//===----------------------------------------------------------------------===//

static void iree_hal_remote_client_device_on_session_ready(
    void* user_data, iree_net_session_t* session,
    const iree_net_session_topology_t* remote_topology) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)session;
  (void)remote_topology;

  device->state = IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED;

  // Fire the pending connect callback.
  iree_hal_remote_client_device_connected_callback_t callback =
      device->connect_callback;
  memset(&device->connect_callback, 0, sizeof(device->connect_callback));
  if (callback.fn) {
    callback.fn(callback.user_data, iree_ok_status());
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_remote_client_device_on_session_goaway(
    void* user_data, iree_net_session_t* session, uint32_t reason_code,
    iree_string_view_t message) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)session;
  (void)reason_code;
  (void)message;

  iree_hal_remote_client_device_state_t previous_state = device->state;
  device->state = IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED;

  // Clear session before releasing to prevent re-entrancy.
  iree_net_session_t* device_session = device->session;
  device->session = NULL;
  iree_net_session_release(device_session);

  if (previous_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING) {
    // Server rejected during bootstrap.
    iree_hal_remote_client_device_connected_callback_t callback =
        device->connect_callback;
    memset(&device->connect_callback, 0, sizeof(device->connect_callback));
    if (callback.fn) {
      callback.fn(callback.user_data,
                  iree_make_status(IREE_STATUS_UNAVAILABLE,
                                   "server sent GOAWAY during connect"));
    }
  } else if (previous_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    // Server initiated graceful disconnect after we were connected.
    if (device->options.error_callback.fn) {
      device->options.error_callback.fn(
          device->options.error_callback.user_data,
          iree_make_status(IREE_STATUS_UNAVAILABLE, "server sent GOAWAY"));
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_remote_client_device_on_session_error(
    void* user_data, iree_net_session_t* session, iree_status_t status) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)session;

  iree_hal_remote_client_device_state_t previous_state = device->state;
  device->state = IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR;

  // Clear session before releasing to prevent re-entrancy.
  iree_net_session_t* device_session = device->session;
  device->session = NULL;
  iree_net_session_release(device_session);

  if (previous_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING) {
    // Connection or bootstrap failed.
    iree_hal_remote_client_device_connected_callback_t callback =
        device->connect_callback;
    memset(&device->connect_callback, 0, sizeof(device->connect_callback));
    if (callback.fn) {
      callback.fn(callback.user_data, status);
    } else {
      iree_status_ignore(status);
    }
  } else {
    // Post-connect session failure.
    if (device->options.error_callback.fn) {
      device->options.error_callback.fn(
          device->options.error_callback.user_data, status);
    } else {
      iree_status_ignore(status);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_remote_client_device_on_control_data(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  // HAL command dispatch will be wired here. For now, acknowledge receipt
  // without processing.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "HAL command dispatch not yet implemented");
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_create(
    iree_string_view_t identifier,
    const iree_hal_remote_client_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_async_buffer_pool_t* recv_pool, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  IREE_ASSERT_ARGUMENT(recv_pool);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

  (void)create_params;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_remote_client_device_options_verify(options));

  // Calculate trailing storage layout.
  iree_host_size_t total_size = 0;
  iree_host_size_t identifier_offset = 0;
  iree_host_size_t server_address_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_hal_remote_client_device_t), &total_size,
              IREE_STRUCT_FIELD(identifier.size, char, &identifier_offset),
              IREE_STRUCT_FIELD(options->server_address.size, char,
                                &server_address_offset)));

  iree_hal_remote_client_device_t* device = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_hal_resource_initialize(&iree_hal_remote_client_device_vtable,
                               &device->resource);

  // Copy strings to trailing storage.
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + identifier_offset);
  device->options = *options;
  iree_string_view_append_to_buffer(options->server_address,
                                    &device->options.server_address,
                                    (char*)device + server_address_offset);

  // Retain transport factory.
  iree_net_transport_factory_retain(options->transport_factory);

  device->host_allocator = host_allocator;
  device->channel_provider = NULL;
  device->device_allocator = NULL;

  // Borrow infrastructure.
  device->proactor = proactor;
  device->frontier_tracker = frontier_tracker;
  device->recv_pool = recv_pool;

  device->session = NULL;
  device->state = IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED;
  memset(&device->connect_callback, 0, sizeof(device->connect_callback));

  *out_device = (iree_hal_device_t*)device;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_connect(
    iree_hal_device_t* base_device,
    iree_hal_remote_client_device_connected_callback_t callback) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (device->state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "device is already connected");
  }
  if (device->state != IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device must be in DISCONNECTED state to connect "
                            "(current state: %d)",
                            (int)device->state);
  }

  device->state = IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING;
  device->connect_callback = callback;

  // Configure session options. The client provides no local topology (no
  // device queues to advertise) — the server will advertise its topology in
  // the HELLO_ACK.
  iree_net_session_options_t session_options =
      iree_net_session_options_default();
  if (device->options.connect_timeout_ns) {
    session_options.bootstrap_timeout_ns = device->options.connect_timeout_ns;
  }

  // Wire session callbacks.
  iree_net_session_callbacks_t session_callbacks;
  memset(&session_callbacks, 0, sizeof(session_callbacks));
  session_callbacks.on_ready = iree_hal_remote_client_device_on_session_ready;
  session_callbacks.on_goaway = iree_hal_remote_client_device_on_session_goaway;
  session_callbacks.on_error = iree_hal_remote_client_device_on_session_error;
  session_callbacks.on_control_data =
      iree_hal_remote_client_device_on_control_data;
  session_callbacks.user_data = device;

  // Initiate async connection + session bootstrap.
  iree_status_t status = iree_net_session_connect(
      device->options.transport_factory, device->options.server_address,
      device->proactor, device->recv_pool, device->frontier_tracker,
      &session_options, session_callbacks, device->host_allocator,
      &device->session);

  if (!iree_status_is_ok(status)) {
    device->state = IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR;
    memset(&device->connect_callback, 0, sizeof(device->connect_callback));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_hal_remote_client_device_state_t
iree_hal_remote_client_device_state(iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return device->state;
}

static void iree_hal_remote_client_device_destroy(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Clear session before releasing to prevent re-entrancy.
  iree_net_session_t* session = device->session;
  device->session = NULL;
  iree_net_session_release(session);

  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_net_transport_factory_release(device->options.transport_factory);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_remote_client_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_remote_client_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_remote_client_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_remote_client_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_remote_client_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_remote_client_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  if (device->device_allocator) {
    IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  *out_value = 0;

  // Device ID query can be answered locally.
  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  // Other queries require a connected session.
  if (device->state != IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device is not connected");
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote device query not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collective channels not supported on remote device");
}

static iree_status_t iree_hal_remote_client_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote command buffer not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote events not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote executable cache not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote file import not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote semaphore not yet implemented");
}

static iree_hal_semaphore_compatibility_t
iree_hal_remote_client_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
         IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_SIGNAL;
}

// All queue operations require the device to be connected.
#define IREE_HAL_REMOTE_REQUIRE_CONNECTED(device)                         \
  if ((device)->state != IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) { \
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,              \
                            "device is not connected");                   \
  }

static iree_status_t iree_hal_remote_client_device_queue_alloca(
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

static iree_status_t iree_hal_remote_client_device_queue_dealloca(
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

static iree_status_t iree_hal_remote_client_device_queue_fill(
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

static iree_status_t iree_hal_remote_client_device_queue_update(
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

static iree_status_t iree_hal_remote_client_device_queue_copy(
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

static iree_status_t iree_hal_remote_client_device_queue_read(
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

static iree_status_t iree_hal_remote_client_device_queue_write(
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

static iree_status_t iree_hal_remote_client_device_queue_host_call(
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

static iree_status_t iree_hal_remote_client_device_queue_dispatch(
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

static iree_status_t iree_hal_remote_client_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_execute not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  // Not connected means nothing to flush.
  if (device->state != IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_flush not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote profiling not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_profiling_flush(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  if (device->state != IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote profiling not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  if (device->state != IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote profiling not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_remote_client_device_topology_info(iree_hal_device_t* base_device) {
  return NULL;
}

static iree_status_t iree_hal_remote_client_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_remote_client_device_vtable = {
    .destroy = iree_hal_remote_client_device_destroy,
    .id = iree_hal_remote_client_device_id,
    .host_allocator = iree_hal_remote_client_device_host_allocator,
    .device_allocator = iree_hal_remote_client_device_allocator,
    .replace_device_allocator = iree_hal_remote_client_replace_device_allocator,
    .replace_channel_provider = iree_hal_remote_client_replace_channel_provider,
    .trim = iree_hal_remote_client_device_trim,
    .query_i64 = iree_hal_remote_client_device_query_i64,
    .query_capabilities = iree_hal_remote_client_device_query_capabilities,
    .topology_info = iree_hal_remote_client_device_topology_info,
    .refine_topology_edge = iree_hal_remote_client_device_refine_topology_edge,
    .assign_topology_info = iree_hal_remote_client_device_assign_topology_info,
    .create_channel = iree_hal_remote_client_device_create_channel,
    .create_command_buffer =
        iree_hal_remote_client_device_create_command_buffer,
    .create_event = iree_hal_remote_client_device_create_event,
    .create_executable_cache =
        iree_hal_remote_client_device_create_executable_cache,
    .import_file = iree_hal_remote_client_device_import_file,
    .create_semaphore = iree_hal_remote_client_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_remote_client_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_remote_client_device_queue_alloca,
    .queue_dealloca = iree_hal_remote_client_device_queue_dealloca,
    .queue_fill = iree_hal_remote_client_device_queue_fill,
    .queue_update = iree_hal_remote_client_device_queue_update,
    .queue_copy = iree_hal_remote_client_device_queue_copy,
    .queue_read = iree_hal_remote_client_device_queue_read,
    .queue_write = iree_hal_remote_client_device_queue_write,
    .queue_host_call = iree_hal_remote_client_device_queue_host_call,
    .queue_dispatch = iree_hal_remote_client_device_queue_dispatch,
    .queue_execute = iree_hal_remote_client_device_queue_execute,
    .queue_flush = iree_hal_remote_client_device_queue_flush,
    .profiling_begin = iree_hal_remote_client_device_profiling_begin,
    .profiling_flush = iree_hal_remote_client_device_profiling_flush,
    .profiling_end = iree_hal_remote_client_device_profiling_end,
};
