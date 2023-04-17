// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/plugins/static_plugin.h"

#include "iree/base/tracing.h"

typedef struct iree_hal_static_executable_plugin_t {
  iree_hal_executable_plugin_t base;
  iree_allocator_t host_allocator;
} iree_hal_static_executable_plugin_t;

static const iree_hal_executable_plugin_vtable_t
    iree_hal_static_executable_plugin_vtable;

iree_status_t iree_hal_static_executable_plugin_create(
    iree_hal_executable_plugin_query_fn_t query_fn,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator,
    iree_hal_executable_plugin_t** out_plugin) {
  IREE_ASSERT_ARGUMENT(query_fn);
  IREE_ASSERT_ARGUMENT(out_plugin);
  *out_plugin = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_static_executable_plugin_t* plugin = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*plugin), (void**)&plugin));
  plugin->host_allocator = host_allocator;

  // Query the plugin interface.
  // This may fail if the version cannot be satisfied.
  const iree_hal_executable_plugin_header_t** header_ptr =
      query_fn(IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST, /*reserved=*/NULL);

  iree_status_t status = iree_hal_executable_plugin_initialize(
      &iree_hal_static_executable_plugin_vtable,
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

static void iree_hal_static_executable_plugin_destroy(
    iree_hal_executable_plugin_t* base_plugin) {
  iree_hal_static_executable_plugin_t* plugin =
      (iree_hal_static_executable_plugin_t*)base_plugin;
  iree_allocator_t host_allocator = plugin->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, plugin);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_executable_plugin_vtable_t
    iree_hal_static_executable_plugin_vtable = {
        .destroy = iree_hal_static_executable_plugin_destroy,
};
