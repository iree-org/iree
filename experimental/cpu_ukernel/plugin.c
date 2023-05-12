// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/api.h"
#include "iree/hal/local/executable_plugin.h"

// Implementation of iree_uk_assert_fail failure is deferred to users code, i.e.
// to us here, as core ukernel/ code can't use the standard library.
#if defined(IREE_UK_STANDALONE)  // Building a standalone plugin.
void iree_uk_assert_fail(const char* file, int line, const char* function,
                         const char* condition) {
  // Doing nothing at the moment.
}
#else  // Building a system plugin.
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
#endif  // defined(IREE_UK_STANDALONE)

// Plugin entry points wrapping the actual ukernels.
static int iree_uk_plugin_mmt4d(void* context, void* params_ptr,
                                void* reserved) {
  iree_uk_mmt4d((const iree_uk_mmt4d_params_t*)params_ptr);
  return 0;
}

static int iree_uk_plugin_pack(void* context, void* params_ptr,
                               void* reserved) {
  iree_uk_pack((const iree_uk_pack_params_t*)params_ptr);
  return 0;
}

static int iree_uk_plugin_unpack(void* context, void* params_ptr,
                                 void* reserved) {
  iree_uk_unpack((const iree_uk_unpack_params_t*)params_ptr);
  return 0;
}

static iree_hal_executable_plugin_status_t iree_uk_plugin_load(
    const iree_hal_executable_plugin_environment_v0_t* environment,
    size_t param_count, const iree_hal_executable_plugin_string_pair_t* params,
    void** out_self) {
  *out_self = NULL;  // no state in this plugin
  return iree_hal_executable_plugin_ok_status();
}

// Called to free any plugin state allocated in load.
static void iree_uk_plugin_unload(void* self) {}

#define ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))

// Called to resolve one or more imports by symbol name.
// See the plugin API header for more information. Note that some of the
// functions may already be resolved and some may be optional.
static iree_hal_executable_plugin_status_t iree_uk_plugin_resolve(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution) {
  typedef struct {
    const char* symbol_name;
    const void* fn_ptr;
  } plugin_entry_point_t;
  static const plugin_entry_point_t entry_points[] = {
      {"iree_uk_mmt4d", iree_uk_plugin_mmt4d},
      {"iree_uk_pack", iree_uk_plugin_pack},
      {"iree_uk_unpack", iree_uk_plugin_unpack},
  };
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
        params->out_fn_contexts[i] = NULL;
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
#if defined(IREE_UK_STANDALONE)  // Building a standalone plugin.
    // Name and description are used for tracing/logging/diagnostics.
    .name = "builtin_ukernel_standalone_plugin",
    .description = "builtin ukernels as standalone plugin (" __FILE__ ")",
    // Standalone plugins must declare that they are standalone so that the
    // runtime can verify support.
    .features = IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_STANDALONE,
    // Standalone plugins don't support sanitizers.
    .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_NONE,
#else   // Building a system plugin.
    // Name and description are used for tracing/logging/diagnostics.
    .name = "builtin_ukernel_system_plugin",
    .description = "builtin ukernels as system plugin (" __FILE__ ")",
    .features = 0,
    // Let the runtime know what sanitizer this plugin was compiled with.
    .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND,
#endif  // defined(IREE_UK_STANDALONE)
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header = &header,
      .load = iree_uk_plugin_load,
      .unload = iree_uk_plugin_unload,
      .resolve = iree_uk_plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t**)&plugin
             : NULL;
}
