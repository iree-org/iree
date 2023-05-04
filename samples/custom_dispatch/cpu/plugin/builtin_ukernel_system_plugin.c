// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/api.h"
#include "iree/hal/local/executable_plugin.h"

// Implementation of iree_uk_assert_fail failure is deferred to users code, i.e.
// to us here, as core ukernel/ code can't use the standard library.
// Limit that to assertions-enabled builds.
#if defined(NDEBUG)
void iree_uk_assert_fail(const char* file, int line, const char* function,
                         const char* condition) {}
#else  // not defined(NDEBUG)
#include <stdio.h>
#include <stdlib.h>
void iree_uk_assert_fail(const char* file, int line, const char* function,
                         const char* condition) {
  fflush(stdout);
  // Must be a single fprintf call (which must make a single write) - typically
  // called from multiple worker threads concurrently.
  fprintf(stderr, "%s:%d: %s: assertion failed: %s\n", file, line, function,
          condition);
  fflush(stderr);
  abort();
}
#endif  // not defined(NDEBUG)

typedef struct {
  iree_hal_executable_plugin_allocator_t host_allocator;
} system_plugin_t;

// Plugin entry points wrapping the actual ukernels.
static int iree_uk_mmt4d_plugin(void* context, void* params_ptr,
                                void* reserved) {
  iree_uk_mmt4d((const iree_uk_mmt4d_params_t*)params_ptr);
  return 0;
}

static int iree_uk_pack_plugin(void* context, void* params_ptr,
                                void* reserved) {
  iree_uk_pack((const iree_uk_pack_params_t*)params_ptr);
  return 0;
}

static int iree_uk_unpack_plugin(void* context, void* params_ptr,
                                void* reserved) {
  iree_uk_unpack((const iree_uk_unpack_params_t*)params_ptr);
  return 0;
}

static iree_hal_executable_plugin_status_t system_plugin_load(
    const iree_hal_executable_plugin_environment_v0_t* environment,
    size_t param_count, const iree_hal_executable_plugin_string_pair_t* params,
    void** out_self) {
  // Allocate the plugin state.
  system_plugin_t* plugin = NULL;
  iree_hal_executable_plugin_status_t status =
      iree_hal_executable_plugin_allocator_malloc(
          environment->host_allocator, sizeof(*plugin), (void**)&plugin);
  if (status) return status;
  plugin->host_allocator = environment->host_allocator;

  // Pass back the plugin instance that'll be passed to resolve.
  *out_self = plugin;
  return iree_hal_executable_plugin_ok_status();
}

// Called to free any plugin state allocated in load.
static void system_plugin_unload(void* self) {
  system_plugin_t* plugin = (system_plugin_t*)self;
  iree_hal_executable_plugin_allocator_t host_allocator =
      plugin->host_allocator;

  // Free the plugin state using the same allocator it came from.
  iree_hal_executable_plugin_allocator_free(host_allocator, plugin);
}

#define ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))

// Called to resolve one or more imports by symbol name.
// See the plugin API header for more information. Note that some of the
// functions may already be resolved and some may be optional.
static iree_hal_executable_plugin_status_t system_plugin_resolve(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution) {
  typedef struct {
    const char* symbol_name;
    const void* fn_ptr;
  } plugin_entry_point_t;
  static const plugin_entry_point_t entry_points[] = {
      {"uk.mmt4d", iree_uk_mmt4d_plugin},
      {"uk.pack", iree_uk_pack_plugin},
      {"uk.unpack", iree_uk_unpack_plugin},
  };
  system_plugin_t* plugin = (system_plugin_t*)self;
  *out_resolution = 0;
  bool any_required_not_found = false;
  for (size_t i = 0; i < params->count; ++i) {
    if (params->out_fn_ptrs[i]) continue;
    const char* symbol_name = params->symbol_names[i];
    bool is_optional =
        iree_hal_executable_plugin_import_is_optional(symbol_name);
    if (is_optional) ++symbol_name;
    bool found = false;
    for (int ep_idx = 0; ep_idx < ARRAYSIZE(entry_points); ++ep_idx) {
      const plugin_entry_point_t* entry_point = &entry_points[ep_idx];
      if (iree_hal_executable_plugin_strcmp(symbol_name,
                                            entry_point->symbol_name) == 0) {
        params->out_fn_ptrs[i] = (void*)(entry_point->fn_ptr);
        params->out_fn_contexts[i] = plugin;
        found = true;
        break;
      }
    }
    if (!found) {
      if (is_optional) {
        *out_resolution |=
            IREE_HAL_EXECUTABLE_PLUGIN_RESOLUTION_MISSING_OPTIONAL;
      } else {
        any_required_not_found = true;
      }
    }
  }
  return any_required_not_found
             ? iree_hal_executable_plugin_status_from_code(
                   IREE_HAL_EXECUTABLE_PLUGIN_STATUS_NOT_FOUND)
             : iree_hal_executable_plugin_ok_status();
}

// Exported on the shared library and used by the runtime to query the plugin
// interface. When statically linking the plugin this is just a function that
// can be called and can have any name to allow for multiple plugins. When
// dynamically linking the exported symbol must be exactly this with no C++
// name mangling.
IREE_HAL_EXECUTABLE_PLUGIN_EXPORT const iree_hal_executable_plugin_header_t**
iree_hal_executable_plugin_query(
    iree_hal_executable_plugin_version_t max_version, void* reserved) {
  static const iree_hal_executable_plugin_header_t header = {
      // Declares what library version is present: newer runtimes may support
      // loading older plugins but newer plugins cannot load on older runtimes.
      .version = IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST,
      // Name and description are used for tracing/logging/diagnostics.
      .name = "builtin_ukernel_system_plugin",
      .description = "builtin ukernels as system plugin (" __FILE__ ")",
      .features = 0,
      // Let the runtime know what sanitizer this plugin was compiled with.
      .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND,
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header = &header,
      .load = system_plugin_load,
      .unload = system_plugin_unload,
      .resolve = system_plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t**)&plugin
             : NULL;
}
