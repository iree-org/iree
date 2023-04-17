// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demonstrates a system linked plugin exporting a single `simple_mul_workgroup`
// function that also prints to stdout. This is not a great idea but shows how
// plugins can have side-effecting behavior - even if in most cases a standalone
// plugin can be used with much smaller code size and portability.
//
// The major use-case for a system linked plugin is JITs that may compile
// imports on-demand. Such plugins could either JIT everything on load time
// or defer JITting to the first call to a particular import. Performing JIT at
// load time is strongly preferred as it keeps all of the expensive work in one
// place before the program starts scheduling execution. Deferring will
// introduce first-run delays and require warmup steps. Since only the imports
// used by the program are present and most programs use all imports it's almost
// always going to be better to do things ahead of time.
//
// NOTE: when using the system loader all unsafe behavior is allowed: TLS,
// threads, mutable globals, syscalls, etc. Doing any of those things will
// likely break in interesting ways as the import functions are called from
// arbitrary threads concurrently. Be very careful and prefer standalone plugins
// instead except when debugging/profiling.

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
} system_plugin_t;

// `ret = lhs * rhs`
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
static int simple_mul_workgroup(void* context, void* params_ptr,
                                void* reserved) {
  system_plugin_t* plugin = (system_plugin_t*)context;
  typedef struct {
    // vvvv simplification pending (buffer + offset)
    const float* restrict binding0;
    const float* restrict binding0_aligned;
    size_t binding0_offset;
    size_t binding0_size;
    size_t binding0_stride;
    const float* restrict binding1;
    const float* restrict binding1_aligned;
    size_t binding1_offset;
    size_t binding1_size;
    size_t binding1_stride;
    float* restrict binding2;
    float* restrict binding2_aligned;
    size_t binding2_offset;
    size_t binding2_size;
    size_t binding2_stride;
    // ^^^^ simplification pending (buffer + offset)
    size_t dim;
    size_t tid;
    uint32_t processor_id;
    const uint64_t* restrict processor_data;
  } params_t;
  const params_t* params = (const params_t*)params_ptr;
  fprintf(plugin->file, "processor_id=%u\nprocessor_data[0]=%" PRIX64 "\n",
          params->processor_id, params->processor_data[0]);
  size_t end = params->tid + 64;
  if (end > params->dim) end = params->dim;
  for (size_t i = params->tid; i < end; ++i) {
    params->binding2[i] = params->binding0[i] * params->binding1[i];
    fprintf(plugin->file, "mul[%zu](%g * %g = %g)\n", i, params->binding0[i],
            params->binding1[i], params->binding2[i]);
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

  // "Open standard out" simulating us doing some syscalls or other expensive
  // stateful/side-effecting things.
  plugin->file = stdout;

  // Pass back the plugin instance that'll be passed to resolve.
  *out_self = plugin;
  return iree_hal_executable_plugin_ok_status();
}

// Called to free any plugin state allocated in load.
static void system_plugin_unload(void* self) {
  system_plugin_t* plugin = (system_plugin_t*)self;
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
static iree_hal_executable_plugin_status_t system_plugin_resolve(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution) {
  system_plugin_t* plugin = (system_plugin_t*)self;
  *out_resolution = 0;
  bool any_required_not_found = false;
  for (size_t i = 0; i < params->count; ++i) {
    if (params->out_fn_ptrs[i]) continue;
    const char* symbol_name = params->symbol_names[i];
    bool is_optional =
        iree_hal_executable_plugin_import_is_optional(symbol_name);
    if (is_optional) ++symbol_name;
    if (iree_hal_executable_plugin_strcmp(symbol_name,
                                          "simple_mul_workgroup") == 0) {
      params->out_fn_ptrs[i] = simple_mul_workgroup;
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
          "(custom_dispatch/cpu/plugin/system_plugin.c)",
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
