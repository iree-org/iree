// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/plugins/embedded_elf_plugin.h"

#include "iree/base/tracing.h"
#include "iree/hal/local/elf/elf_module.h"

#if IREE_FILE_IO_ENABLE
#include "iree/base/internal/file_io.h"
#endif  // IREE_FILE_IO_ENABLE

//===----------------------------------------------------------------------===//
// iree_hal_memory_embedded_elf_executable_plugin_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_memory_embedded_elf_executable_plugin_t {
  iree_hal_executable_plugin_t base;
  iree_allocator_t host_allocator;
  iree_elf_module_t module;
} iree_hal_memory_embedded_elf_executable_plugin_t;

static const iree_hal_executable_plugin_vtable_t
    iree_hal_memory_embedded_elf_executable_plugin_vtable;

iree_status_t iree_hal_embedded_elf_executable_plugin_load_from_memory(
    iree_const_byte_span_t buffer, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_executable_plugin_t** out_plugin) {
  IREE_ASSERT_ARGUMENT(out_plugin);
  *out_plugin = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_memory_embedded_elf_executable_plugin_t* plugin = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*plugin), (void**)&plugin));
  plugin->host_allocator = host_allocator;

  // Attempt to load the ELF module.
  iree_status_t status = iree_elf_module_initialize_from_memory(
      buffer, /*import_table=*/NULL, host_allocator, &plugin->module);

  // Get the exported symbol used to get the plugin metadata.
  iree_hal_executable_plugin_query_fn_t query_fn = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_elf_module_lookup_export(
        &plugin->module, IREE_HAL_EXECUTABLE_PLUGIN_EXPORT_NAME,
        (void**)&query_fn);
  }

  if (iree_status_is_ok(status)) {
    // Query the plugin interface.
    // This may fail if the version cannot be satisfied.
    const iree_hal_executable_plugin_header_t** header_ptr =
        (const iree_hal_executable_plugin_header_t**)iree_elf_call_p_ip(
            query_fn, IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST,
            /*reserved=*/NULL);

    status = iree_hal_executable_plugin_initialize(
        &iree_hal_memory_embedded_elf_executable_plugin_vtable,
        IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_STANDALONE, header_ptr, param_count,
        params, /*resolve_thunk=*/
        (iree_hal_executable_plugin_resolve_thunk_t)iree_elf_call_p_ppp,
        host_allocator, &plugin->base);
  }

  if (iree_status_is_ok(status)) {
    *out_plugin = (iree_hal_executable_plugin_t*)plugin;
  } else {
    iree_hal_executable_plugin_release((iree_hal_executable_plugin_t*)plugin);
    status = iree_status_annotate_f(status, "loading plugin from memory");
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_memory_embedded_elf_executable_plugin_destroy(
    iree_hal_executable_plugin_t* base_plugin) {
  iree_hal_memory_embedded_elf_executable_plugin_t* plugin =
      (iree_hal_memory_embedded_elf_executable_plugin_t*)base_plugin;
  iree_allocator_t host_allocator = plugin->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_elf_module_deinitialize(&plugin->module);
  iree_allocator_free(host_allocator, plugin);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_executable_plugin_vtable_t
    iree_hal_memory_embedded_elf_executable_plugin_vtable = {
        .destroy = iree_hal_memory_embedded_elf_executable_plugin_destroy,
};

//===----------------------------------------------------------------------===//
// iree_hal_file_embedded_elf_executable_plugin_t
//===----------------------------------------------------------------------===//

#if IREE_FILE_IO_ENABLE

typedef struct iree_hal_file_embedded_elf_executable_plugin_t {
  iree_hal_executable_plugin_t base;
  iree_allocator_t host_allocator;
  iree_file_contents_t* file_contents;
  iree_elf_module_t module;
} iree_hal_file_embedded_elf_executable_plugin_t;

static const iree_hal_executable_plugin_vtable_t
    iree_hal_file_embedded_elf_executable_plugin_vtable;

iree_status_t iree_hal_embedded_elf_executable_plugin_load_from_file(
    const char* path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_executable_plugin_t** out_plugin) {
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(out_plugin);
  *out_plugin = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Try to load the file first, which is the most likely thing to fail.
  iree_file_contents_t* file_contents = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_file_read_contents(path, host_allocator, &file_contents));

  iree_hal_file_embedded_elf_executable_plugin_t* plugin = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*plugin), (void**)&plugin);
  if (!iree_status_is_ok(status)) {
    iree_file_contents_free(file_contents);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  plugin->host_allocator = host_allocator;
  plugin->file_contents = file_contents;

  // Attempt to load the ELF module.
  status = iree_elf_module_initialize_from_memory(
      file_contents->const_buffer, /*import_table=*/NULL, host_allocator,
      &plugin->module);

  // Get the exported symbol used to get the plugin metadata.
  iree_hal_executable_plugin_query_fn_t query_fn = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_elf_module_lookup_export(
        &plugin->module, IREE_HAL_EXECUTABLE_PLUGIN_EXPORT_NAME,
        (void**)&query_fn);
  }

  // Load the plugin.
  if (iree_status_is_ok(status)) {
    // Query the plugin interface.
    // This may fail if the version cannot be satisfied.
    const iree_hal_executable_plugin_header_t** header =
        (const iree_hal_executable_plugin_header_t**)iree_elf_call_p_ip(
            query_fn, IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST,
            /*reserved=*/NULL);

    status = iree_hal_executable_plugin_initialize(
        &iree_hal_file_embedded_elf_executable_plugin_vtable,
        IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_STANDALONE, header, param_count,
        params, /*resolve_thunk=*/
        (iree_hal_executable_plugin_resolve_thunk_t)iree_elf_call_p_ppp,
        host_allocator, &plugin->base);
  }

  if (iree_status_is_ok(status)) {
    *out_plugin = (iree_hal_executable_plugin_t*)plugin;
  } else {
    iree_hal_executable_plugin_release((iree_hal_executable_plugin_t*)plugin);
    status =
        iree_status_annotate_f(status, "loading plugin from file '%s'", path);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_file_embedded_elf_executable_plugin_destroy(
    iree_hal_executable_plugin_t* base_plugin) {
  iree_hal_file_embedded_elf_executable_plugin_t* plugin =
      (iree_hal_file_embedded_elf_executable_plugin_t*)base_plugin;
  iree_allocator_t host_allocator = plugin->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_elf_module_deinitialize(&plugin->module);
  iree_file_contents_free(plugin->file_contents);
  iree_allocator_free(host_allocator, plugin);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_executable_plugin_vtable_t
    iree_hal_file_embedded_elf_executable_plugin_vtable = {
        .destroy = iree_hal_file_embedded_elf_executable_plugin_destroy,
};

#endif  // IREE_FILE_IO_ENABLE
