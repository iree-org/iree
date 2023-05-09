// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demonstrates a standalone plugin exporting a single `simple_mul_workgroup`
// function. This models kernel libraries and custom intrinsics where there are
// only stateless functions. Standalone plugins can be compiled to portable ELFs
// that allow the same plugin file to be used on all platforms
// (linux/windows/mac/ bare-metal, etc) without the need to recompile or have
// platform-specific toolchains. As much as possible plugins should try to be in
// this form - when getting called in the CPU task system performing syscalls,
// blocking, or using TLS are either unsupported or extremely bad ideas. System
// linked plugins allow all those things but don't make it any safer.
//
// NOTE: in standalone mode the plugin cannot have side-effects: no allocations
// outside of the iree_hal_executable_plugin_allocator_t, no syscalls, no rwdata
// globals, and no TLS.

// The only header required from IREE:
#include "iree/hal/local/executable_plugin.h"

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
// table, etc).
//
// |params_ptr| points to a packed struct of all results followed by all args
// using native arch packing/alignment rules. Results should be set before
// returning.
//
// Expects a return of 0 on success and any other value indicates failure.
// Try not to fail!
static int simple_mul_workgroup(void* context, void* params_ptr,
                                void* reserved) {
  typedef struct {
    const float* restrict binding0;
    size_t binding0_offset;
    const float* restrict binding1;
    size_t binding1_offset;
    float* restrict binding2;
    size_t binding2_offset;
    size_t size;
    size_t tid;
    uint32_t processor_id;
    const uint64_t* restrict processor_data;
  } params_t;
  const params_t* params = (const params_t*)params_ptr;
  // The operation `iree_codegen.ukernel.generic` always operates
  // on a slice of the inputs to produce a slice of the output,
  // so the loop here just needs to iterate from `0` to `size`,
  // where `size` is the size of the slice to be executed by this call.
  for (size_t i = 0; i < params->size; ++i) {
    // The operation `iree_codegen.ukernel.generic` takes a slice of
    // the inputs and outputs as operands. So the `pointer` and `offset`
    // passed into this function represent the starting location of
    // where to read the data from for this invocation of the function.
    params->binding2[params->binding2_offset + i] =
        params->binding0[params->binding0_offset + i] *
        params->binding1[params->binding2_offset + i];
  }
  return 0;
}

// Called once for each plugin load and paired with a future call to unload.
// We don't do anything special here as this plugin is meant to represent a
// pure/stateless kernel library. Even in standalone mode we could allocate
// using environment->host_allocator, set an out_self pointer, and parse
// parameters.
//
// If any state is required it should be allocated and stored in |out_self|.
// This self value will be passed to all future calls related to the particular
// instance. Note that there may be multiple instances of a plugin in any
// particular process and this must be thread-safe.
static iree_hal_executable_plugin_status_t standalone_plugin_load(
    const iree_hal_executable_plugin_environment_v0_t* environment,
    size_t param_count, const iree_hal_executable_plugin_string_pair_t* params,
    void** out_self) {
  *out_self = NULL;  // no state in this plugin
  return iree_hal_executable_plugin_ok_status();
}

// Called to free any plugin state allocated in load.
// In this sample it's a no-op as we don't have state.
static void standalone_plugin_unload(void* self) {}

// Called to resolve one or more imports by symbol name.
// See the plugin API header for more information. Note that some of the
// functions may already be resolved and some may be optional.
static iree_hal_executable_plugin_status_t standalone_plugin_resolve(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution) {
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
      params->out_fn_contexts[i] = NULL;  // no context used, could be self
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
      .name = "sample_standalone",
      .description =
          "standalone plugin sample "
          "(custom_dispatch/cpu/plugin/standalone_plugin.c)",
      // Standalone plugins must declare that they are standalone so that the
      // runtime can verify support.
      .features = IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_STANDALONE,
      // Standalone plugins don't support sanitizers.
      .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_NONE,
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header = &header,
      .load = standalone_plugin_load,
      .unload = standalone_plugin_unload,
      .resolve = standalone_plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t**)&plugin
             : NULL;
}
