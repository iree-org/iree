// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demonstrates an mlp example with the implementation of MLP provided
// using system linked plugin exporting a single `mlp_external`
// function.  See samples/custom_dispatch/cpu/plugin/system_plugin.c
// for more information about system plugins and their caveats.

#include <inttypes.h>
#include <stdio.h>

// The only header required from IREE:
#include "iree/hal/local/executable_plugin.h"

// Stateful plugin instance.
// There may be multiple of these in a process at a time, each with its own
// load/unload pairing. We pass a pointer to this to all import calls via the
// context argument.
typedef struct {
  iree_hal_executable_plugin_allocator_t host_allocator;
  FILE* file;
} mlp_plugin_t;

// Helper function to resolve index [i][j] into location for given strides.
static size_t get_index(size_t i, size_t j, size_t offset, size_t stride) {
  return offset + i * stride + j;
}

// `ret = mlp(lhs, rhs)`
//
// Conforms to ABI:
// #hal.pipeline.layout<push_constants = 1, sets = [
//   <0, bindings = [
//       <0, storage_buffer, ReadOnly>,
//       <1, storage_buffer, ReadOnly>,
//       <2, storage_buffer>
//   ]>
// ]>
// With a workgroup size of 64x1x1.
//
// |context| is whatever was set in out_fn_contexts. This could point to shared
// state or each import can have its own context (pointer into some JIT lookup
// table, etc). In this sample we pass the sample plugin pointer to all imports.
//
// |params_ptr| points to a packed struct of all results followed by all args
// using native arch packing/alignment rules. Results should be set before
// returning.
//
// Expects a return of 0 on success and any other value indicates failure.
// Try not to fail!
static int mlp_external(void* params_ptr, void* context, void* reserved) {
  mlp_plugin_t* plugin = (mlp_plugin_t*)context;
  typedef struct {
    const float* restrict lhs;
    size_t lhs_offset;
    const float* restrict rhs;
    size_t rhs_offset;
    float* restrict result;
    size_t result_offset;
    int32_t M;
    int32_t N;
    int32_t K;
  } params_t;
  const params_t* params = (const params_t*)params_ptr;
  fprintf(plugin->file, "[Plugin]: M = %d, N = %d, K = %d\n", params->M,
          params->N, params->K);
  for (int32_t i = 0; i < params->M; i++) {
    for (int32_t j = 0; j < params->N; j++) {
      float curr_result = 0.0;
      for (int32_t k = 0; k < params->K; k++) {
        size_t lhs_index = get_index(i, k, params->lhs_offset, (size_t)params->K);
        size_t rhs_index = get_index(k, j, params->rhs_offset, (size_t)params->N);
        curr_result += params->lhs[lhs_index] * params->rhs[rhs_index];
      }
      curr_result = curr_result < 0.0 ? 0.0 : curr_result;
      size_t result_index =
          get_index(i, j, params->result_offset, params->N);
      params->result[result_index] = curr_result;
    }
  }
  return 0;
}

// Called once for each plugin load and paired with a future call to unload.
// Even in standalone mode we could allocate using environment->host_allocator,
// set an out_self pointer, and parse parameters but here in system mode we can
// do whatever we want.
//
// If any state is required it should be allocated and stored in |out_self|.
// This self value will be passed to all future calls related to the particular
// instance. Note that there may be multiple instances of a plugin in any
// particular process and this must be thread-safe.
static iree_hal_executable_plugin_status_t mlp_plugin_load(
    const iree_hal_executable_plugin_environment_v0_t* environment,
    size_t param_count, const iree_hal_executable_plugin_string_pair_t* params,
    void** out_self) {
  // Allocate the plugin state.
  mlp_plugin_t* plugin = NULL;
  iree_hal_executable_plugin_status_t status =
      iree_hal_executable_plugin_allocator_malloc(
          environment->host_allocator, sizeof(*plugin), (void**)&plugin);
  if (status) return status;
  plugin->host_allocator = environment->host_allocator;

  // "Open standard out" simulating us doing some syscalls or other expensive
  // stateful/side-effecting things.
  plugin->file = stdout;

  // Pass back the plugin instance that'll be passed to resolve.
  *out_self = plugin;
  return iree_hal_executable_plugin_ok_status();
}

// Called to free any plugin state allocated in load.
static void mlp_plugin_unload(void* self) {
  mlp_plugin_t* plugin = (mlp_plugin_t*)self;
  iree_hal_executable_plugin_allocator_t host_allocator =
      plugin->host_allocator;

  // "Close standard out" simulating us doing some syscalls and other expensive
  // stateful/side-effecting things.
  fflush(plugin->file);
  plugin->file = NULL;

  // Free the plugin state using the same allocator it came from.
  iree_hal_executable_plugin_allocator_free(host_allocator, plugin);
}

// Called to resolve one or more imports by symbol name.
// See the plugin API header for more information. Note that some of the
// functions may already be resolved and some may be optional.
static iree_hal_executable_plugin_status_t mlp_plugin_resolve(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution) {
  mlp_plugin_t* plugin = (mlp_plugin_t*)self;
  *out_resolution = 0;
  bool any_required_not_found = false;
  for (size_t i = 0; i < params->count; ++i) {
    if (params->out_fn_ptrs[i]) continue;
    const char* symbol_name = params->symbol_names[i];
    bool is_optional =
        iree_hal_executable_plugin_import_is_optional(symbol_name);
    if (is_optional) ++symbol_name;
    if (iree_hal_executable_plugin_strcmp(symbol_name, "mlp_external") == 0) {
      params->out_fn_ptrs[i] = mlp_external;
      params->out_fn_contexts[i] =
          plugin;  // passing plugin to each import call
    } else {
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
      .name = "sample_system",
      .description =
          "system plugin sample "
          "(custom_dispatch/cpu/plugin/mlp_plugin.c)",
      .features = 0,
      // Let the runtime know what sanitizer this plugin was compiled with.
      .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND,
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header = &header,
      .load = mlp_plugin_load,
      .unload = mlp_plugin_unload,
      .resolve = mlp_plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t**)&plugin
             : NULL;
}
