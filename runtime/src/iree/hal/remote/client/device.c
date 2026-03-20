// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/device.h"

#include "iree/async/frontier_tracker.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/processor.h"
#include "iree/hal/remote/client/allocator.h"
#include "iree/hal/remote/client/command_buffer.h"
#include "iree/hal/remote/client/executable_cache.h"
#include "iree/hal/remote/client/queue.h"
#include "iree/hal/remote/client/semaphore.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/status_wire.h"
#include "iree/net/transport_factory.h"

static const iree_hal_device_vtable_t iree_hal_remote_client_device_vtable;

iree_hal_remote_client_device_t* iree_hal_remote_client_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_remote_client_device_vtable);
  return (iree_hal_remote_client_device_t*)base_value;
}

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

// Pending control RPC state. Stack-allocated by the blocking caller, linked
// into the device's pending_rpcs list for the duration of the RPC.
typedef struct iree_hal_remote_pending_rpc_t {
  uint32_t request_id;
  iree_notification_t notification;
  // Written by the proactor thread with release ordering, read by the app
  // thread with acquire ordering. This is the C11 atomic bridge between the
  // proactor's writes to response fields and the app thread's reads. The
  // notification post/commit_wait also provides ordering, but the sentinel
  // check between prepare_wait and commit_wait requires explicit atomics.
  iree_atomic_int32_t response_status_code;
  iree_const_byte_span_t response_payload;  // points into retained lease
  iree_async_buffer_lease_t response_lease;
  struct iree_hal_remote_pending_rpc_t* next;
} iree_hal_remote_pending_rpc_t;

IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_create(
    iree_string_view_t identifier,
    const iree_hal_remote_client_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_hal_remote_recv_pool_t* recv_pool, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(recv_pool);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

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

  // Retain proactor from the pool, borrow frontier tracker, retain recv_pool.
  device->proactor = iree_hal_remote_recv_pool_proactor(recv_pool);
  iree_async_proactor_retain(device->proactor);
  device->frontier_tracker = create_params->frontier.tracker;
  device->recv_pool = recv_pool;
  iree_hal_remote_recv_pool_retain(recv_pool);

  device->session = NULL;
  iree_atomic_store(&device->queue_channel, 0, iree_memory_order_relaxed);
  iree_atomic_store(&device->channel_users, 0, iree_memory_order_relaxed);
  device->remote_queue_axis = 0;
  iree_atomic_store(&device->next_submission_epoch, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&device->next_provisional_generation, 1,
                    iree_memory_order_relaxed);
  iree_slim_mutex_initialize(&device->provisional_mutex);
  memset(&device->provisional_buffers, 0, sizeof(device->provisional_buffers));
  iree_atomic_store(&device->next_request_id, 1, iree_memory_order_relaxed);
  iree_slim_mutex_initialize(&device->rpc_mutex);
  device->pending_rpcs = NULL;
  iree_atomic_store(&device->state,
                    IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED,
                    iree_memory_order_relaxed);
  memset(&device->connect_callback, 0, sizeof(device->connect_callback));

  // Create the remote allocator proxy.
  iree_status_t allocator_status = iree_hal_remote_client_allocator_create(
      device, identifier, host_allocator, &device->device_allocator);
  if (!iree_status_is_ok(allocator_status)) {
    iree_slim_mutex_deinitialize(&device->rpc_mutex);
    iree_net_transport_factory_release(options->transport_factory);
    iree_allocator_free(host_allocator, device);
    IREE_TRACE_ZONE_END(z0);
    return allocator_status;
  }

  *out_device = (iree_hal_device_t*)device;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Atomically detaches the queue channel using the Dekker drain pattern.
// Zeroes device->queue_channel (seq_cst), then spins until all in-flight
// channel_users have drained. Returns the old channel pointer (or NULL if
// already detached). The caller must detach and release the returned channel.
static iree_net_queue_channel_t*
iree_hal_remote_client_device_drain_queue_channel(
    iree_hal_remote_client_device_t* device) {
  iree_net_queue_channel_t* queue_channel =
      (iree_net_queue_channel_t*)iree_atomic_exchange(
          &device->queue_channel, 0, iree_memory_order_seq_cst);
  // Drain in-flight users. The seq_cst exchange above is ordered before any
  // subsequent load of channel_users in the total order, so if a user
  // incremented channel_users before we zeroed queue_channel, we will see
  // their increment here.
  while (iree_atomic_load(&device->channel_users, iree_memory_order_acquire) !=
         0) {
    iree_processor_yield();
  }
  return queue_channel;
}

static void iree_hal_remote_client_device_destroy(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Fail the remote queue axis to resolve any remaining frontier waiters.
  // Typically a no-op (goaway/error already failed it), but handles the edge
  // case where destroy is called without a prior disconnect.
  if (iree_hal_remote_client_device_load_state(device) ==
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    iree_async_frontier_tracker_fail_axis(
        device->frontier_tracker, device->remote_queue_axis,
        iree_make_status(IREE_STATUS_CANCELLED,
                         "device destroyed while connected"));
  }

  // Detach and release queue channel before session. Detach clears endpoint
  // callbacks while the endpoint is still alive and zeroes the endpoint
  // reference, preventing UAF if retained references (barrier completions)
  // later drop the last channel ref after the session is freed.
  iree_net_queue_channel_t* queue_channel =
      iree_hal_remote_client_device_drain_queue_channel(device);
  iree_net_queue_channel_detach(queue_channel);
  iree_net_queue_channel_release(queue_channel);

  // Clear session before releasing to prevent re-entrancy.
  iree_net_session_t* session = device->session;
  device->session = NULL;
  iree_net_session_release(session);

  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_net_transport_factory_release(device->options.transport_factory);

  // Release any unresolved provisional buffers (shouldn't happen in normal
  // operation — all provisionals are resolved by ADVANCE before teardown).
  for (iree_host_size_t i = 0; i < device->provisional_buffers.count; ++i) {
    iree_hal_buffer_release(device->provisional_buffers.buffers[i]);
  }
  iree_allocator_free(host_allocator,
                      device->provisional_buffers.provisional_ids);
  iree_allocator_free(host_allocator, device->provisional_buffers.buffers);
  iree_slim_mutex_deinitialize(&device->provisional_mutex);

  iree_hal_remote_recv_pool_release(device->recv_pool);
  iree_async_proactor_release(device->proactor);

  iree_slim_mutex_deinitialize(&device->rpc_mutex);
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
  (void)base_device;
  *out_value = 0;

  // The remote device is a transparent proxy — it matches any device ID and
  // executable format pattern. The server validates actual compatibility at
  // executable upload time. This allows compiled modules to find the remote
  // device regardless of which backend the server wraps.
  if (iree_string_view_equal(category, IREE_SV("hal.device.id")) ||
      iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = 1;
    return iree_ok_status();
  }

  // Other queries return 0 (unsupported/unknown). This is the standard
  // behavior for queries the device doesn't recognize — not an error.
  *out_value = 0;
  return iree_ok_status();
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

static iree_status_t iree_hal_remote_client_device_query_string(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, iree_host_size_t out_string_size,
    char* out_string) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  if (out_string_size > 0) out_string[0] = '\0';

  IREE_HAL_REMOTE_REQUIRE_CONNECTED(device);

  // Build the request: envelope + query_string_request + category + key.
  // Category is padded to 8-byte alignment.
  uint16_t category_padded = (uint16_t)((category.size + 7) & ~7);
  iree_host_size_t total_size =
      sizeof(iree_hal_remote_control_envelope_t) +
      sizeof(iree_hal_remote_device_query_string_request_t) +
      category_padded + key.size;

  uint8_t request_buffer[512];
  if (total_size > sizeof(request_buffer)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "query_string request too large");
  }
  memset(request_buffer, 0, total_size);

  // Envelope.
  iree_hal_remote_control_envelope_t* envelope =
      (iree_hal_remote_control_envelope_t*)request_buffer;
  envelope->message_type = IREE_HAL_REMOTE_CONTROL_DEVICE_QUERY_STRING;

  // Request body.
  iree_hal_remote_device_query_string_request_t* request =
      (iree_hal_remote_device_query_string_request_t*)(request_buffer +
          sizeof(iree_hal_remote_control_envelope_t));
  request->category_length = (uint16_t)category.size;
  request->key_length = (uint16_t)key.size;

  // Copy category and key strings.
  uint8_t* strings_start = (uint8_t*)request +
      sizeof(iree_hal_remote_device_query_string_request_t);
  memcpy(strings_start, category.data, category.size);
  memcpy(strings_start + category_padded, key.data, key.size);

  // Send RPC and wait for response.
  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      device, iree_make_const_byte_span(request_buffer, total_size),
      &response_payload, &response_lease);

  if (iree_status_is_ok(status)) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_device_query_string_response_t)) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "query_string response too small");
    }
  }

  if (iree_status_is_ok(status)) {
    const iree_hal_remote_device_query_string_response_t* response =
        (const iree_hal_remote_device_query_string_response_t*)
            response_payload.data;
    uint16_t value_length = response->value_length;
    const char* value_data =
        (const char*)(response_payload.data +
            sizeof(iree_hal_remote_device_query_string_response_t));

    // Verify we have enough data.
    if (response_payload.data_length <
        sizeof(iree_hal_remote_device_query_string_response_t) +
            value_length) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "query_string response data truncated");
    } else if (out_string_size <= value_length) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output string too small");
    } else {
      memcpy(out_string, value_data, value_length);
      out_string[value_length] = '\0';
    }
  }

  iree_async_buffer_lease_release(&response_lease);
  return status;
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
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return iree_hal_remote_client_command_buffer_create(
      device, mode, command_categories, queue_affinity, binding_capacity,
      device->host_allocator, out_command_buffer);
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
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return iree_hal_remote_client_executable_cache_create(
      device, identifier, device->host_allocator, out_executable_cache);
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
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return iree_hal_remote_client_semaphore_create(
      device->proactor, initial_value, device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_remote_client_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
         IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_SIGNAL;
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
  if (iree_hal_remote_client_device_load_state(device) !=
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote profiling not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  if (iree_hal_remote_client_device_load_state(device) !=
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote profiling not yet implemented");
}

static void iree_hal_remote_client_device_fail_pending_rpcs(
    iree_hal_remote_client_device_t* device);

//===----------------------------------------------------------------------===//
// Session callbacks
//===----------------------------------------------------------------------===//

static void iree_hal_remote_client_device_on_session_ready(
    void* user_data, iree_net_session_t* session,
    const iree_net_session_topology_t* remote_topology) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Save the remote queue axis for building signal frontiers.
  // The server's topology must have at least one axis (the queue axis).
  if (remote_topology->axis_count > 0) {
    device->remote_queue_axis = remote_topology->axes[0];
  }
  iree_atomic_store(&device->next_submission_epoch, 1,
                    iree_memory_order_relaxed);
  iree_atomic_store(&device->next_provisional_generation, 1,
                    iree_memory_order_relaxed);

  iree_hal_remote_client_device_store_state(
      device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED);

  // The connect callback is NOT fired here — it is deferred to
  // on_queue_endpoint_ready so the application receives the callback only
  // after the queue channel is published and queue operations are usable.
  iree_net_endpoint_ready_callback_t endpoint_callback = {
      .fn = iree_hal_remote_client_device_on_queue_endpoint_ready,
      .user_data = device,
  };
  iree_status_t status =
      iree_net_session_open_endpoint(session, endpoint_callback);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_store_state(
        device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
    // Fire the connect callback with the error so the application doesn't
    // hang waiting for a connection that will never complete.
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

static void iree_hal_remote_client_device_on_session_goaway(
    void* user_data, iree_net_session_t* session, uint32_t reason_code,
    iree_string_view_t message) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)session;
  (void)reason_code;
  (void)message;

  iree_hal_remote_client_device_state_t previous_state =
      iree_hal_remote_client_device_load_state(device);
  iree_hal_remote_client_device_store_state(
      device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED);

  // Wake any pending control RPCs so they fail instead of blocking forever.
  iree_hal_remote_client_device_fail_pending_rpcs(device);

  // Fail the remote queue axis so pending frontier waiters (from queue_execute
  // signal semaphores) are resolved immediately. Without this, waiters for
  // epochs whose ADVANCE frames will never arrive would hang forever, leaking
  // their pending_signal_t allocations and retained semaphores.
  // This is a no-op if the axis was never registered (device never connected).
  if (previous_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    iree_async_frontier_tracker_fail_axis(
        device->frontier_tracker, device->remote_queue_axis,
        iree_make_status(IREE_STATUS_UNAVAILABLE,
                         "server sent GOAWAY; remote queue axis failed"));
  }

  // Drain in-flight channel users then detach and release. Detach while the
  // endpoint is still alive to clear callbacks safely.
  iree_net_queue_channel_t* queue_channel =
      iree_hal_remote_client_device_drain_queue_channel(device);
  iree_net_queue_channel_detach(queue_channel);
  iree_net_queue_channel_release(queue_channel);

  // Clear session before releasing to prevent re-entrancy.
  iree_net_session_t* device_session = device->session;
  device->session = NULL;
  iree_net_session_release(device_session);

  // If the connect callback is still pending (bootstrap hasn't fully
  // completed — on_queue_endpoint_ready hasn't fired yet), fire it with
  // error so the application doesn't hang waiting for a connection result.
  // This handles GOAWAY arriving between on_session_ready (CONNECTED state)
  // and on_queue_endpoint_ready (callback fire).
  iree_hal_remote_client_device_connected_callback_t callback =
      device->connect_callback;
  memset(&device->connect_callback, 0, sizeof(device->connect_callback));
  if (callback.fn) {
    callback.fn(callback.user_data,
                iree_make_status(IREE_STATUS_UNAVAILABLE,
                                 "server sent GOAWAY during connect"));
  } else if (previous_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    // Fully connected (callback already fired). Notify via error callback.
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

  iree_hal_remote_client_device_state_t previous_state =
      iree_hal_remote_client_device_load_state(device);
  iree_hal_remote_client_device_store_state(
      device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);

  // Wake any pending control RPCs so they fail instead of blocking forever.
  iree_hal_remote_client_device_fail_pending_rpcs(device);

  // Fail the remote queue axis (same rationale as goaway handler).
  if (previous_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    iree_async_frontier_tracker_fail_axis(device->frontier_tracker,
                                          device->remote_queue_axis,
                                          iree_status_clone(status));
  }

  // Drain in-flight channel users then detach and release.
  iree_net_queue_channel_t* queue_channel =
      iree_hal_remote_client_device_drain_queue_channel(device);
  iree_net_queue_channel_detach(queue_channel);
  iree_net_queue_channel_release(queue_channel);

  // Clear session before releasing to prevent re-entrancy.
  iree_net_session_t* device_session = device->session;
  device->session = NULL;
  iree_net_session_release(device_session);

  // If the connect callback is still pending, fire it with the error so the
  // application doesn't hang. This covers errors during bootstrap (CONNECTING
  // state) and errors between on_session_ready and on_queue_endpoint_ready
  // (CONNECTED state but callback not yet fired).
  iree_hal_remote_client_device_connected_callback_t callback =
      device->connect_callback;
  memset(&device->connect_callback, 0, sizeof(device->connect_callback));
  if (callback.fn) {
    callback.fn(callback.user_data, status);
  } else {
    // Fully connected (callback already fired). Notify via error callback.
    if (device->options.error_callback.fn) {
      device->options.error_callback.fn(
          device->options.error_callback.user_data, status);
    } else {
      iree_status_ignore(status);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

// Wakes all pending RPCs with an error status. Called when the session
// disconnects (goaway/error) while RPCs are in-flight.
static void iree_hal_remote_client_device_fail_pending_rpcs(
    iree_hal_remote_client_device_t* device) {
  iree_slim_mutex_lock(&device->rpc_mutex);
  iree_hal_remote_pending_rpc_t* pending = device->pending_rpcs;
  device->pending_rpcs = NULL;
  iree_slim_mutex_unlock(&device->rpc_mutex);

  while (pending) {
    iree_hal_remote_pending_rpc_t* next = pending->next;
    iree_atomic_store(&pending->response_status_code,
                      (int32_t)IREE_STATUS_UNAVAILABLE,
                      iree_memory_order_release);
    iree_notification_post(&pending->notification, IREE_ALL_WAITERS);
    pending = next;
  }
}

// Sends a control channel message and blocks until the response arrives.
// The request span must contain the full message ([envelope + body]).
// On success, |out_response_payload| points into the retained lease and
// |out_response_lease| holds the backing buffer. The caller must release
// the lease after processing the response.
iree_status_t iree_hal_remote_client_device_control_rpc(
    iree_hal_remote_client_device_t* device, iree_const_byte_span_t request,
    iree_const_byte_span_t* out_response_payload,
    iree_async_buffer_lease_t* out_response_lease) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_hal_remote_client_device_load_state(device) !=
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device is not connected");
  }
  if (!device->session) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "session is not available");
  }

  // Assign a unique request_id and patch it into the envelope.
  uint32_t request_id = (uint32_t)iree_atomic_fetch_add(
      &device->next_request_id, 1, iree_memory_order_relaxed);
  // The envelope is at the start of the request buffer. We cast away const
  // to patch the request_id — the caller allocated this on their stack.
  iree_hal_remote_control_envelope_t* envelope =
      (iree_hal_remote_control_envelope_t*)request.data;
  envelope->request_id = request_id;

  // Initialize the pending RPC entry on the stack.
  iree_hal_remote_pending_rpc_t pending;
  memset(&pending, 0, sizeof(pending));
  pending.request_id = request_id;
  iree_notification_initialize(&pending.notification);
  // IREE_STATUS_INTERNAL is the sentinel: "no response yet."
  iree_atomic_store(&pending.response_status_code, IREE_STATUS_INTERNAL,
                    iree_memory_order_relaxed);

  // Link into the pending list.
  iree_slim_mutex_lock(&device->rpc_mutex);
  pending.next = device->pending_rpcs;
  device->pending_rpcs = &pending;
  iree_slim_mutex_unlock(&device->rpc_mutex);

  // Send the request.
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)request.data, request.data_length);
  iree_async_span_list_t payload = {&span, 1};
  iree_status_t status = iree_net_session_send_control_data(
      device->session, /*flags=*/0, payload, /*operation_user_data=*/0);

  if (iree_status_is_ok(status)) {
    // Block until the proactor thread delivers the response.
    iree_wait_token_t token =
        iree_notification_prepare_wait(&pending.notification);
    // Check if the response already arrived (between send and prepare_wait).
    // Acquire pairs with the proactor thread's release store.
    if (iree_atomic_load(&pending.response_status_code,
                         iree_memory_order_acquire) == IREE_STATUS_INTERNAL) {
      // Still waiting — commit the wait.
      iree_notification_commit_wait(&pending.notification, token,
                                    IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    } else {
      iree_notification_cancel_wait(&pending.notification);
    }
  }

  // Unlink from the pending list.
  iree_slim_mutex_lock(&device->rpc_mutex);
  iree_hal_remote_pending_rpc_t** prev = &device->pending_rpcs;
  while (*prev && *prev != &pending) prev = &(*prev)->next;
  if (*prev == &pending) *prev = pending.next;
  iree_slim_mutex_unlock(&device->rpc_mutex);

  iree_notification_deinitialize(&pending.notification);

  if (!iree_status_is_ok(status)) {
    // Send failed — clean up any lease that might have been set.
    iree_async_buffer_lease_release(&pending.response_lease);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Check the response status. The acquire load is redundant after
  // commit_wait (which already provides acquire ordering), but explicit
  // for the cancel_wait early-return path and for TSAN visibility.
  iree_status_code_t response_code = (iree_status_code_t)iree_atomic_load(
      &pending.response_status_code, iree_memory_order_acquire);
  if (response_code != IREE_STATUS_OK) {
    // Try to deserialize the full status from the response body. The server
    // serializes the status using status_wire format as the body payload.
    iree_status_t error_status = iree_ok_status();
    if (pending.response_payload.data_length > 0) {
      iree_status_t deserialize_status = iree_net_status_wire_deserialize(
          pending.response_payload, &error_status);
      if (!iree_status_is_ok(deserialize_status)) {
        // Deserialization failed (truncated, version mismatch, etc.).
        // Propagate the deserialization failure annotated with the server's
        // response code so the caller sees both what went wrong on the
        // server AND that the detailed status couldn't be recovered.
        error_status = iree_status_annotate_f(
            deserialize_status, "server returned %s; status details lost",
            iree_status_code_string(response_code));
      }
    } else {
      // No body — the server sent only a status code (old server, or the
      // error path didn't serialize a body). Annotate so the caller knows
      // this came from a remote RPC.
      error_status =
          iree_make_status(response_code, "remote control RPC failed");
    }
    iree_async_buffer_lease_release(&pending.response_lease);
    IREE_TRACE_ZONE_END(z0);
    return error_status;
  }

  *out_response_payload = pending.response_payload;
  *out_response_lease = pending.response_lease;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Sends a fire-and-forget control message (no response expected).
iree_status_t iree_hal_remote_client_device_send_fire_and_forget(
    iree_hal_remote_client_device_t* device, iree_const_byte_span_t message) {
  if (!device->session) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "session is not available");
  }
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)message.data, message.data_length);
  iree_async_span_list_t payload = {&span, 1};
  return iree_net_session_send_control_data(device->session, /*flags=*/0,
                                            payload,
                                            /*operation_user_data=*/0);
}

// Returns the device's session (for use by the allocator module).
iree_net_session_t* iree_hal_remote_client_device_session(
    iree_hal_remote_client_device_t* device) {
  return device->session;
}

//===----------------------------------------------------------------------===//
// Provisional buffer tracking
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_remote_client_device_register_provisional(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t provisional_id, iree_hal_buffer_t* buffer) {
  iree_slim_mutex_lock(&device->provisional_mutex);

  iree_host_size_t minimum_capacity = device->provisional_buffers.count + 1;
  if (minimum_capacity > device->provisional_buffers.capacity) {
    iree_host_size_t ids_capacity = device->provisional_buffers.capacity;
    iree_status_t status = iree_allocator_grow_array(
        device->host_allocator, minimum_capacity,
        sizeof(iree_hal_remote_resource_id_t), &ids_capacity,
        (void**)&device->provisional_buffers.provisional_ids);
    if (iree_status_is_ok(status)) {
      iree_host_size_t bufs_capacity = device->provisional_buffers.capacity;
      status = iree_allocator_grow_array(
          device->host_allocator, minimum_capacity, sizeof(iree_hal_buffer_t*),
          &bufs_capacity, (void**)&device->provisional_buffers.buffers);
      device->provisional_buffers.capacity =
          iree_min(ids_capacity, bufs_capacity);
    } else {
      device->provisional_buffers.capacity = ids_capacity;
    }
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&device->provisional_mutex);
      return status;
    }
  }

  iree_host_size_t index = device->provisional_buffers.count++;
  device->provisional_buffers.provisional_ids[index] = provisional_id;
  device->provisional_buffers.buffers[index] = buffer;
  iree_hal_buffer_retain(buffer);

  iree_slim_mutex_unlock(&device->provisional_mutex);
  return iree_ok_status();
}

iree_hal_buffer_t* iree_hal_remote_client_device_resolve_provisional(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t provisional_id) {
  iree_slim_mutex_lock(&device->provisional_mutex);

  iree_hal_buffer_t* buffer = NULL;
  for (iree_host_size_t i = 0; i < device->provisional_buffers.count; ++i) {
    if (device->provisional_buffers.provisional_ids[i] == provisional_id) {
      buffer = device->provisional_buffers.buffers[i];
      // Remove by swapping with the last entry.
      iree_host_size_t last = device->provisional_buffers.count - 1;
      if (i != last) {
        device->provisional_buffers.provisional_ids[i] =
            device->provisional_buffers.provisional_ids[last];
        device->provisional_buffers.buffers[i] =
            device->provisional_buffers.buffers[last];
      }
      --device->provisional_buffers.count;
      // Release the tracking reference. The buffer stays alive via the
      // caller's reference from queue_alloca.
      iree_hal_buffer_release(buffer);
      break;
    }
  }

  iree_slim_mutex_unlock(&device->provisional_mutex);
  return buffer;
}

static iree_status_t iree_hal_remote_client_device_on_control_data(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;

  // Parse control envelope.
  if (payload.data_length < sizeof(iree_hal_remote_control_envelope_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control data too small for envelope: %" PRIhsz
                            " bytes",
                            payload.data_length);
  }
  const iree_hal_remote_control_envelope_t* envelope =
      (const iree_hal_remote_control_envelope_t*)payload.data;

  if (envelope->message_flags & IREE_HAL_REMOTE_CONTROL_FLAG_IS_RESPONSE) {
    // Response to a pending RPC. Match by request_id.
    const uint8_t* after_envelope =
        payload.data + sizeof(iree_hal_remote_control_envelope_t);
    iree_host_size_t remaining =
        payload.data_length - sizeof(iree_hal_remote_control_envelope_t);

    // Parse response prefix.
    if (remaining < sizeof(iree_hal_remote_control_response_prefix_t)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "control response too small for response prefix: %" PRIhsz " bytes",
          remaining);
    }
    const iree_hal_remote_control_response_prefix_t* prefix =
        (const iree_hal_remote_control_response_prefix_t*)after_envelope;
    const uint8_t* response_body =
        after_envelope + sizeof(iree_hal_remote_control_response_prefix_t);
    iree_host_size_t response_body_length =
        remaining - sizeof(iree_hal_remote_control_response_prefix_t);

    // Find the matching pending RPC.
    iree_slim_mutex_lock(&device->rpc_mutex);
    iree_hal_remote_pending_rpc_t* pending = device->pending_rpcs;
    while (pending && pending->request_id != envelope->request_id) {
      pending = pending->next;
    }
    iree_slim_mutex_unlock(&device->rpc_mutex);

    if (!pending) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no pending RPC with request_id=%u",
                              envelope->request_id);
    }

    // Populate the pending RPC with response data. The payload and lease
    // must be written before the status code (which is the release store
    // that makes them visible to the app thread's acquire load).
    pending->response_payload =
        iree_make_const_byte_span(response_body, response_body_length);
    // Steal ownership of the lease so the response payload data stays valid
    // after this callback returns. Zeroing the original makes the session's
    // post-callback release a no-op (release.fn == NULL).
    pending->response_lease = *lease;
    memset(lease, 0, sizeof(*lease));

    // Release store: makes all prior writes (payload, lease) visible to the
    // app thread when it does an acquire load of response_status_code.
    iree_atomic_store(&pending->response_status_code,
                      (int32_t)prefix->status_code, iree_memory_order_release);

    // Wake the blocked caller.
    iree_notification_post(&pending->notification, IREE_ALL_WAITERS);
    return iree_ok_status();
  }

  // Server-initiated notification (request_id == 0).
  // TODO: dispatch NOTIFY_RESOURCE_ERROR, NOTIFY_DEVICE_LOST, etc.
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_connect(
    iree_hal_device_t* base_device,
    iree_hal_remote_client_device_connected_callback_t callback) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_hal_remote_client_device_load_state(device) ==
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "device is already connected");
  }
  if (iree_hal_remote_client_device_load_state(device) !=
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "device must be in DISCONNECTED state to connect "
        "(current state: %d)",
        (int)iree_hal_remote_client_device_load_state(device));
  }

  iree_hal_remote_client_device_store_state(
      device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING);
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
  iree_net_session_callbacks_t session_callbacks = {
      .on_ready = iree_hal_remote_client_device_on_session_ready,
      .on_goaway = iree_hal_remote_client_device_on_session_goaway,
      .on_error = iree_hal_remote_client_device_on_session_error,
      .on_control_data = iree_hal_remote_client_device_on_control_data,
      .user_data = device,
  };

  // Initiate async connection + session bootstrap.
  iree_status_t status = iree_net_session_connect(
      device->options.transport_factory, device->options.server_address,
      device->proactor,
      iree_hal_remote_recv_pool_buffer_pool(device->recv_pool),
      device->frontier_tracker, &session_options, session_callbacks,
      device->host_allocator, &device->session);

  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_store_state(
        device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
    memset(&device->connect_callback, 0, sizeof(device->connect_callback));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_hal_remote_client_device_state_t
iree_hal_remote_client_device_state(iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return iree_hal_remote_client_device_load_state(device);
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
    .query_string = iree_hal_remote_client_device_query_string,
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
