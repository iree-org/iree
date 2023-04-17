// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/plugins/system_library_plugin.h"

#include "iree/base/internal/dynamic_library.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_system_library_executable_plugin_t {
  iree_hal_executable_plugin_t base;
  iree_allocator_t host_allocator;
  iree_dynamic_library_t* handle;
} iree_hal_system_library_executable_plugin_t;

static const iree_hal_executable_plugin_vtable_t
    iree_hal_system_library_executable_plugin_vtable;

static iree_status_t iree_hal_system_library_executable_plugin_create(
    iree_dynamic_library_t* handle, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_executable_plugin_t** out_plugin) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_plugin);
  *out_plugin = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get the exported symbol used to get the plugin metadata.
  iree_hal_executable_plugin_query_fn_t query_fn = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_dynamic_library_lookup_symbol(
          handle, IREE_HAL_EXECUTABLE_PLUGIN_EXPORT_NAME, (void**)&query_fn));

  // Query the plugin interface.
  // This may fail if the version cannot be satisfied.
  const iree_hal_executable_plugin_header_t** header_ptr =
      query_fn(IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST, /*reserved=*/NULL);

  iree_hal_system_library_executable_plugin_t* plugin = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*plugin), (void**)&plugin));
  plugin->host_allocator = host_allocator;
  plugin->handle = handle;
  iree_dynamic_library_retain(plugin->handle);

  iree_status_t status = iree_hal_executable_plugin_initialize(
      &iree_hal_system_library_executable_plugin_vtable,
      IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_NONE, header_ptr, param_count, params,
      /*resolve_thunk=*/NULL, host_allocator, &plugin->base);

  if (iree_status_is_ok(status)) {
    *out_plugin = (iree_hal_executable_plugin_t*)plugin;
  } else {
    iree_hal_executable_plugin_release((iree_hal_executable_plugin_t*)plugin);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_system_library_executable_plugin_load_from_memory(
    iree_string_view_t identifier, iree_const_byte_span_t buffer,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator,
    iree_hal_executable_plugin_t** out_plugin) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Try to load the library from memory using the system loader.
  // This can fail for many reasons (incompatible, unsigned code, missing
  // required imports, etc).
  iree_dynamic_library_t* handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_dynamic_library_load_from_memory(identifier, buffer,
                                                IREE_DYNAMIC_LIBRARY_FLAG_NONE,
                                                host_allocator, &handle));

  // Create the wrapper and load the plugin.
  iree_status_t status = iree_hal_system_library_executable_plugin_create(
      handle, param_count, params, host_allocator, out_plugin);
  iree_dynamic_library_release(handle);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_system_library_executable_plugin_load_from_file(
    const char* path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_executable_plugin_t** out_plugin) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Try to load the library from file using the system loader.
  // This can fail for many reasons (file not found, not accessible,
  // incompatible, unsigned code, missing required imports, etc).
  iree_dynamic_library_t* handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_dynamic_library_load_from_file(
              path, IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &handle));

  // Create the wrapper and load the plugin.
  iree_status_t status = iree_hal_system_library_executable_plugin_create(
      handle, param_count, params, host_allocator, out_plugin);
  iree_dynamic_library_release(handle);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_system_library_executable_plugin_destroy(
    iree_hal_executable_plugin_t* base_plugin) {
  iree_hal_system_library_executable_plugin_t* plugin =
      (iree_hal_system_library_executable_plugin_t*)base_plugin;
  iree_allocator_t host_allocator = plugin->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_dynamic_library_release(plugin->handle);
  iree_allocator_free(host_allocator, plugin);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_executable_plugin_vtable_t
    iree_hal_system_library_executable_plugin_vtable = {
        .destroy = iree_hal_system_library_executable_plugin_destroy,
};
