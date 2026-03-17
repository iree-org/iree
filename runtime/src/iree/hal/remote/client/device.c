// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/device.h"

#include "iree/async/util/proactor_pool.h"
#include "iree/hal/remote/client/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_device_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_remote_client_device_options_initialize(
    iree_hal_remote_client_device_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->carrier = IREE_HAL_REMOTE_CLIENT_CARRIER_TCP;  // Default.
  out_options->server_address = iree_string_view_empty();
  out_options->connect_timeout_ns = 0;        // Use default (30 seconds).
  out_options->max_control_message_size = 0;  // Use default (64KB).
  out_options->max_queue_frame_size = 0;      // Use default (64KB).
  out_options->flags = IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_NONE;
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
  if (iree_string_view_is_empty(options->server_address)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "server_address is required");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_remote_client_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Proactor pool for async I/O (retained).
  iree_async_proactor_pool_t* proactor_pool;

  // Device configuration options.
  iree_hal_remote_client_device_options_t options;

  // Current connection state.
  iree_hal_remote_client_device_state_t state;

  // + trailing identifier string storage
  // + trailing server_address string storage
} iree_hal_remote_client_device_t;

static const iree_hal_device_vtable_t iree_hal_remote_client_device_vtable;

static iree_hal_remote_client_device_t* iree_hal_remote_client_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_remote_client_device_vtable);
  return (iree_hal_remote_client_device_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_create(
    iree_string_view_t identifier,
    const iree_hal_remote_client_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

  // Verify the parameters prior to creating resources.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_remote_client_device_options_verify(options));

  // Allocate device with trailing storage for identifier and server address.
  iree_hal_remote_client_device_t* device = NULL;
  iree_host_size_t total_size =
      sizeof(*device) + identifier.size + options->server_address.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_hal_resource_initialize(&iree_hal_remote_client_device_vtable,
                               &device->resource);

  // Copy identifier to trailing storage.
  char* string_storage = (char*)device + sizeof(*device);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    string_storage);
  string_storage += identifier.size;

  // Copy server address to trailing storage.
  device->options = *options;
  iree_string_view_append_to_buffer(
      options->server_address, &device->options.server_address, string_storage);

  device->host_allocator = host_allocator;
  device->channel_provider = NULL;
  device->proactor_pool = create_params ? create_params->proactor_pool : NULL;
  if (device->proactor_pool) {
    iree_async_proactor_pool_retain(device->proactor_pool);
  }
  device->state = IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED;

  // Create the proxy allocator that forwards allocation requests to the server.
  // This is created immediately (not on connect) so that the device allocator
  // is never NULL - callers can always get the allocator, though operations
  // on it will trigger connection if not already connected.
  // TODO: implement iree_hal_remote_client_allocator_create.
  device->device_allocator = NULL;

  *out_device = (iree_hal_device_t*)device;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_remote_client_device_connect(iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Already connected or connecting.
  if (device->state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  if (device->state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device is in error state; must be recreated");
  }

  // Mark as connecting.
  device->state = IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING;

  // Connection not yet implemented.
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "remote client connection not yet "
                                          "implemented");

  if (!iree_status_is_ok(status)) {
    device->state = IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR;
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

// Ensures the device is connected before performing an operation.
// Returns OK if connected, or attempts connection and returns the result.
static iree_status_t iree_hal_remote_client_device_ensure_connected(
    iree_hal_remote_client_device_t* device) {
  if (device->state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    return iree_ok_status();
  }
  return iree_hal_remote_client_device_connect((iree_hal_device_t*)device);
}

static void iree_hal_remote_client_device_destroy(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release any outstanding resources.
  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);

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

  // Trim local allocator caches.
  if (device->device_allocator) {
    IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  }

  // Remote trim would involve sending a message to the server to release any
  // cached resources on the server side as well.

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

  // Other queries require connection to the server to get remote device info.
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_ensure_connected(device));

  // Forward query to server.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote device query not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  // Collective channels are not supported for remote devices.
  // Cross-device collectives would need to be coordinated through the server
  // or use a separate collective transport.
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

  // Command buffers are recorded locally and streamed to the server on execute.
  // The client command buffer captures commands in a format suitable for
  // network transmission.
  (void)device;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote command buffer not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  // Events are device-local synchronization primitives.
  // Remote events would need server-side representation.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote events not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);

  // Executable cache manages executables loaded on the server.
  // Executables are uploaded to the server and cached there.
  (void)device;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote executable cache not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  // File import would need to transfer file contents to the server.
  // For large files, this may use the bulk transfer channel.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote file import not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);

  // Remote semaphores are proxy objects that track both local state and remote
  // server state. They use the frontier system for causal consistency.
  (void)device;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote semaphore not yet implemented");
}

static iree_hal_semaphore_compatibility_t
iree_hal_remote_client_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  (void)device;

  // Check if the semaphore is one of our remote proxy semaphores.
  // Only our own semaphores can be used for queue operations.
  // Host wait/signal is always possible as we track state locally.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
         IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_SIGNAL;
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
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_ensure_connected(device));

  // Queue-ordered allocation on the server.
  // Creates a local proxy buffer that references server-side memory.
  // The allocation request is sent via the queue channel with frontier
  // dependencies from wait_semaphore_list.
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
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_ensure_connected(device));

  // Queue-ordered deallocation on the server.
  // The server releases the memory after frontier dependencies are satisfied.
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
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_ensure_connected(device));

  // Fill is sent as a queue operation to the server.
  // The pattern data is small enough to include inline in the message.
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
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_ensure_connected(device));

  // Small updates can be sent inline in queue messages.
  // Large updates should use the bulk transfer channel.
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
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_ensure_connected(device));

  // Server-side copy between two remote buffers.
  // No data transfer over the network, just a queue command.
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
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_ensure_connected(device));

  // File read on the server side.
  // File handle must reference a server-accessible file.
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
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_ensure_connected(device));

  // File write on the server side.
  // File handle must reference a server-accessible file.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_write not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  // Host calls cannot be executed remotely; they require local execution.
  // The call would need to execute on the client with data transferred from
  // the server.
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
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_ensure_connected(device));

  // Dispatch is sent to the server referencing server-side executable and
  // buffers. Constants are sent inline. The operation executes entirely on
  // the server.
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
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_ensure_connected(device));

  // Command buffer contents are streamed to the server for execution.
  // The server reconstructs a command buffer on the wrapped device and
  // executes it with the specified frontier dependencies.
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

  // Flush any buffered queue messages to the server.
  // This ensures all pending operations have been sent over the network.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote queue_flush not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_ensure_connected(device));

  // Request profiling mode on the server.
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

  // Request profiling data flush from the server.
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

  // End profiling mode on the server.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote profiling not yet implemented");
}

static iree_status_t iree_hal_remote_client_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  // Capabilities must be queried from the remote server.
  // For now, report no capabilities until connected.
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_remote_client_device_topology_info(iree_hal_device_t* base_device) {
  // Topology info is populated during session handshake with the server.
  return NULL;
}

static iree_status_t iree_hal_remote_client_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  // Remote devices are always connected via network — topology edge reflects
  // the transport characteristics (latency, bandwidth) once connected.
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  // Topology info is assigned during session handshake.
  // For now, accept and ignore (the server will provide this).
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
