// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/testing/mock_device.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_mock_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_mock_device_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Identifier string (backed by trailing storage).
  iree_string_view_t identifier;

  // Capabilities returned by query_capabilities.
  iree_hal_device_capabilities_t capabilities;

  // Topology information assigned during group creation.
  iree_hal_device_topology_info_t topology_info;
} iree_hal_mock_device_t;

static const iree_hal_device_vtable_t iree_hal_mock_device_vtable;

static iree_hal_mock_device_t* iree_hal_mock_device_cast(
    iree_hal_device_t* base_device) {
  IREE_HAL_ASSERT_TYPE(base_device, &iree_hal_mock_device_vtable);
  return (iree_hal_mock_device_t*)base_device;
}

void iree_hal_mock_device_options_initialize(
    iree_hal_mock_device_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
}

iree_status_t iree_hal_mock_device_create(
    const iree_hal_mock_device_options_t* options,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;

  iree_hal_mock_device_t* device = NULL;
  iree_host_size_t total_size = sizeof(*device) + options->identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, sizeof(*device));
  iree_hal_resource_initialize(&iree_hal_mock_device_vtable, &device->resource);
  device->host_allocator = host_allocator;
  device->capabilities = options->capabilities;

  // Copy identifier into trailing storage.
  iree_string_view_append_to_buffer(
      options->identifier, &device->identifier,
      (char*)device + total_size - options->identifier.size);

  *out_device = (iree_hal_device_t*)device;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Implemented vtable methods
//===----------------------------------------------------------------------===//

static void iree_hal_mock_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  iree_allocator_t host_allocator = device->host_allocator;
  iree_allocator_free(host_allocator, device);
}

static iree_string_view_t iree_hal_mock_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_mock_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  return device->host_allocator;
}

static iree_status_t iree_hal_mock_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  *out_capabilities = device->capabilities;
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_mock_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_mock_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  // No refinement — the base edge from capabilities is used as-is.
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  device->topology_info = *topology_info;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Stub vtable methods (UNIMPLEMENTED)
//===----------------------------------------------------------------------===//
//
// These exist only to fill the vtable. Any call through them is a test bug.

static iree_hal_allocator_t* iree_hal_mock_device_allocator(
    iree_hal_device_t* base_device) {
  return NULL;
}

static void iree_hal_mock_device_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {}

static void iree_hal_mock_device_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
}

static iree_status_t iree_hal_mock_device_trim(iree_hal_device_t* base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_hal_semaphore_compatibility_t
iree_hal_mock_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;
}

static iree_status_t iree_hal_mock_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_hal_wait_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_profiling_flush(
    iree_hal_device_t* base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_profiling_end(
    iree_hal_device_t* base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_device_vtable_t iree_hal_mock_device_vtable = {
    .destroy = iree_hal_mock_device_destroy,
    .id = iree_hal_mock_device_id,
    .host_allocator = iree_hal_mock_device_host_allocator,
    .device_allocator = iree_hal_mock_device_allocator,
    .replace_device_allocator = iree_hal_mock_device_replace_device_allocator,
    .replace_channel_provider = iree_hal_mock_device_replace_channel_provider,
    .trim = iree_hal_mock_device_trim,
    .query_i64 = iree_hal_mock_device_query_i64,
    .query_capabilities = iree_hal_mock_device_query_capabilities,
    .topology_info = iree_hal_mock_device_topology_info,
    .refine_topology_edge = iree_hal_mock_device_refine_topology_edge,
    .assign_topology_info = iree_hal_mock_device_assign_topology_info,
    .create_channel = iree_hal_mock_device_create_channel,
    .create_command_buffer = iree_hal_mock_device_create_command_buffer,
    .create_event = iree_hal_mock_device_create_event,
    .create_executable_cache = iree_hal_mock_device_create_executable_cache,
    .import_file = iree_hal_mock_device_import_file,
    .create_semaphore = iree_hal_mock_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_mock_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_mock_device_queue_alloca,
    .queue_dealloca = iree_hal_mock_device_queue_dealloca,
    .queue_fill = iree_hal_mock_device_queue_fill,
    .queue_update = iree_hal_mock_device_queue_update,
    .queue_copy = iree_hal_mock_device_queue_copy,
    .queue_read = iree_hal_mock_device_queue_read,
    .queue_write = iree_hal_mock_device_queue_write,
    .queue_host_call = iree_hal_mock_device_queue_host_call,
    .queue_dispatch = iree_hal_mock_device_queue_dispatch,
    .queue_execute = iree_hal_mock_device_queue_execute,
    .queue_flush = iree_hal_mock_device_queue_flush,
    .wait_semaphores = iree_hal_mock_device_wait_semaphores,
    .profiling_begin = iree_hal_mock_device_profiling_begin,
    .profiling_flush = iree_hal_mock_device_profiling_flush,
    .profiling_end = iree_hal_mock_device_profiling_end,
};
