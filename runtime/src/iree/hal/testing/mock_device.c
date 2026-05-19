// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/testing/mock_device.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Mock executable support
//===----------------------------------------------------------------------===//

static const iree_string_view_t iree_hal_mock_executable_format =
    IREE_SVL("mock-executable");

typedef struct iree_hal_mock_executable_function_record_t {
  // Number of 32-bit constants reflected for the function.
  uint8_t constant_count;
  // Number of buffer bindings reflected for the function.
  uint8_t binding_count;
  // Executable function flags byte.
  uint8_t flags;
  // Static workgroup size reflected for the function.
  uint8_t workgroup_size[3];
  // Byte length of the function name in the trailing name storage.
  uint8_t name_length;
  // Reserved byte; must be zero.
  uint8_t reserved;
} iree_hal_mock_executable_function_record_t;

typedef struct iree_hal_mock_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_host_size_t function_count;
  iree_hal_executable_function_info_t functions[];
} iree_hal_mock_executable_t;

static const iree_hal_executable_vtable_t iree_hal_mock_executable_vtable;

static iree_hal_mock_executable_t* iree_hal_mock_executable_cast(
    iree_hal_executable_t* base_executable) {
  IREE_HAL_ASSERT_TYPE(base_executable, &iree_hal_mock_executable_vtable);
  return (iree_hal_mock_executable_t*)base_executable;
}

static iree_status_t iree_hal_mock_executable_create(
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  if (!iree_string_view_equal(executable_params->executable_format,
                              iree_hal_mock_executable_format)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported mock executable format '%.*s'",
                            (int)executable_params->executable_format.size,
                            executable_params->executable_format.data);
  }
  if (IREE_UNLIKELY(executable_params->executable_data.data_length <
                    sizeof(uint32_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mock executable data is too short");
  }
  uint32_t function_count = 0;
  memcpy(&function_count, executable_params->executable_data.data,
         sizeof(function_count));
  iree_const_byte_span_t function_data = iree_make_const_byte_span(
      executable_params->executable_data.data + sizeof(function_count),
      executable_params->executable_data.data_length - sizeof(function_count));
  iree_host_size_t function_record_data_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
                        function_count,
                        sizeof(iree_hal_mock_executable_function_record_t),
                        &function_record_data_length) ||
                    function_record_data_length > function_data.data_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "mock executable function metadata length mismatch");
  }
  iree_const_byte_span_t name_storage = iree_make_const_byte_span(
      function_data.data + function_record_data_length,
      function_data.data_length - function_record_data_length);

  const iree_hal_mock_executable_function_record_t* function_records =
      (const iree_hal_mock_executable_function_record_t*)function_data.data;
  iree_host_size_t expected_name_storage_length = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    const iree_hal_mock_executable_function_record_t* record =
        &function_records[i];
    if (IREE_UNLIKELY(record->reserved != 0)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "mock executable function metadata reserved byte "
                              "must be zero");
    }
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            expected_name_storage_length, record->name_length,
            &expected_name_storage_length))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "mock executable function name storage overflow");
    }
  }
  if (IREE_UNLIKELY(expected_name_storage_length != name_storage.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mock executable function name length mismatch");
  }

  iree_host_size_t function_info_size = 0;
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(
              function_count, sizeof(iree_hal_executable_function_info_t),
              &function_info_size) ||
          !iree_host_size_checked_add(sizeof(iree_hal_mock_executable_t),
                                      function_info_size, &total_size) ||
          !iree_host_size_checked_add(total_size, name_storage.data_length,
                                      &total_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mock executable metadata is too large");
  }
  iree_hal_mock_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  memset(executable, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_mock_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->function_count = function_count;

  char* executable_name_storage =
      (char*)executable + sizeof(*executable) + function_info_size;
  if (name_storage.data_length != 0) {
    memcpy(executable_name_storage, name_storage.data,
           name_storage.data_length);
  }
  iree_host_size_t name_offset = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    const iree_hal_mock_executable_function_record_t* record =
        &function_records[i];
    executable->functions[i].name = iree_make_string_view(
        executable_name_storage + name_offset, record->name_length);
    name_offset += record->name_length;
    executable->functions[i].flags = record->flags;
    executable->functions[i].constant_count = record->constant_count;
    executable->functions[i].binding_count = record->binding_count;
    executable->functions[i].parameter_count = 0;
    executable->functions[i].workgroup_size[0] = record->workgroup_size[0];
    executable->functions[i].workgroup_size[1] = record->workgroup_size[1];
    executable->functions[i].workgroup_size[2] = record->workgroup_size[2];
  }

  *out_executable = (iree_hal_executable_t*)executable;
  return iree_ok_status();
}

static void iree_hal_mock_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t iree_hal_mock_executable_function_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  return executable->function_count;
}

static iree_status_t iree_hal_mock_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
  }
  const uint32_t function_ordinal =
      iree_hal_executable_function_index(function);
  *out_info = executable->functions[function_ordinal];
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  (void)base_executable;
  (void)function;
  (void)capacity;
  (void)out_parameters;
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    if (iree_string_view_equal(executable->functions[i].name, name)) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  *out_function = iree_hal_executable_function_invalid();
  return iree_make_status(IREE_STATUS_NOT_FOUND);
}

static iree_status_t iree_hal_mock_executable_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  (void)base_executable;
  (void)name;
  (void)queue_affinity;
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static const iree_hal_executable_vtable_t iree_hal_mock_executable_vtable = {
    .destroy = iree_hal_mock_executable_destroy,
    .function_count = iree_hal_mock_executable_function_count,
    .function_info = iree_hal_mock_executable_function_info,
    .function_parameters = iree_hal_mock_executable_function_parameters,
    .lookup_function_by_name = iree_hal_mock_executable_lookup_function_by_name,
    .lookup_global_by_name = iree_hal_mock_executable_lookup_global_by_name,
};

typedef struct iree_hal_mock_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} iree_hal_mock_executable_cache_t;

static const iree_hal_executable_cache_vtable_t
    iree_hal_mock_executable_cache_vtable;

static iree_hal_mock_executable_cache_t* iree_hal_mock_executable_cache_cast(
    iree_hal_executable_cache_t* base_executable_cache) {
  IREE_HAL_ASSERT_TYPE(base_executable_cache,
                       &iree_hal_mock_executable_cache_vtable);
  return (iree_hal_mock_executable_cache_t*)base_executable_cache;
}

static iree_status_t iree_hal_mock_executable_cache_create(
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;
  iree_hal_mock_executable_cache_t* executable_cache = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executable_cache), (void**)&executable_cache));
  memset(executable_cache, 0, sizeof(*executable_cache));
  iree_hal_resource_initialize(&iree_hal_mock_executable_cache_vtable,
                               &executable_cache->resource);
  executable_cache->host_allocator = host_allocator;
  *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;
  return iree_ok_status();
}

static void iree_hal_mock_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  iree_hal_mock_executable_cache_t* executable_cache =
      iree_hal_mock_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->host_allocator;
  iree_allocator_free(host_allocator, executable_cache);
}

static iree_status_t iree_hal_mock_executable_cache_infer_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  (void)base_executable_cache;
  (void)caching_mode;
  (void)executable_data;
  if (iree_hal_mock_executable_format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  memcpy(executable_format, iree_hal_mock_executable_format.data,
         iree_hal_mock_executable_format.size + /*NUL*/ 1);
  *out_inferred_size = executable_data.data_length;
  return iree_ok_status();
}

static bool iree_hal_mock_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  (void)base_executable_cache;
  (void)caching_mode;
  return iree_string_view_equal(executable_format,
                                iree_hal_mock_executable_format);
}

static iree_status_t iree_hal_mock_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_mock_executable_cache_t* executable_cache =
      iree_hal_mock_executable_cache_cast(base_executable_cache);
  return iree_hal_mock_executable_create(
      executable_params, executable_cache->host_allocator, out_executable);
}

static const iree_hal_executable_cache_vtable_t
    iree_hal_mock_executable_cache_vtable = {
        .destroy = iree_hal_mock_executable_cache_destroy,
        .infer_format = iree_hal_mock_executable_cache_infer_format,
        .can_prepare_format = iree_hal_mock_executable_cache_can_prepare_format,
        .prepare_executable = iree_hal_mock_executable_cache_prepare_executable,
};

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

  // Status returned by assign_topology_info.
  iree_status_code_t assign_topology_info_status_code;

  // True when create_executable_cache returns mock executable caches.
  bool executable_cache_enabled;

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
  device->assign_topology_info_status_code =
      options->assign_topology_info_status_code;
  device->executable_cache_enabled = options->executable_cache_enabled;

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
  if (!topology_info) {
    memset(&device->topology_info, 0, sizeof(device->topology_info));
    return iree_ok_status();
  } else if (device->assign_topology_info_status_code != IREE_STATUS_OK) {
    return iree_make_status(device->assign_topology_info_status_code);
  }
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
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  if (device->executable_cache_enabled) {
    (void)identifier;
    return iree_hal_mock_executable_cache_create(device->host_allocator,
                                                 out_executable_cache);
  }
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

static iree_status_t iree_hal_mock_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
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
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
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

static iree_status_t iree_hal_mock_device_external_capture_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_external_capture_options_t* options) {
  (void)base_device;
  (void)options;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_external_capture_end(
    iree_hal_device_t* base_device) {
  (void)base_device;
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
    .query_queue_pool_backend = iree_hal_mock_device_query_queue_pool_backend,
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
    .profiling_begin = iree_hal_mock_device_profiling_begin,
    .profiling_flush = iree_hal_mock_device_profiling_flush,
    .profiling_end = iree_hal_mock_device_profiling_end,
    .external_capture_begin = iree_hal_mock_device_external_capture_begin,
    .external_capture_end = iree_hal_mock_device_external_capture_end,
};
