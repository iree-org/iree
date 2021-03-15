// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/local/loaders/wasm_module_loader.h"

#include <math.h>

#include "iree/base/tracing.h"
#include "iree/hal/local/local_executable.h"
#include "wasm_export.h"

// flatcc schemas:
#include "iree/base/flatcc.h"
#include "iree/schemas/dylib_executable_def_reader.h"
#include "iree/schemas/dylib_executable_def_verifier.h"

float fmaf_impl(wasm_exec_env_t exec_env, float x, float y, float z) {
  return fmaf(x, y, z);
}

// TODO(scotttodd): remove flatbuffer utils after switching to system library

//===----------------------------------------------------------------------===//
// Verification and file utilities
//===----------------------------------------------------------------------===//

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_dylib_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data,
    iree_host_size_t expected_entry_point_count) {
  // Special handling for valid but mismatching flatbuffers.
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16 ||
      !flatbuffers_has_identifier(flatbuffer_data.data,
                                  iree_DyLibExecutableDef_file_identifier)) {
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_DyLibExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_DyLibExecutableDef_table_t executable_def =
      iree_DyLibExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_DyLibExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
  if (entry_point_count != expected_entry_point_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable provides %zu entry points but caller "
                            "provided %zu; must match",
                            entry_point_count, expected_entry_point_count);
  }

  for (size_t i = 0; i < entry_point_count; ++i) {
    if (!flatbuffers_string_len(
            flatbuffers_string_vec_at(entry_points_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  if (!flatbuffers_uint8_vec_len(
          iree_DyLibExecutableDef_library_embedded_get(executable_def))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable library_embedded is missing/empty");
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_wasm_executable_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_local_executable_t base;

  // Flatbuffer definition referencing the WASM module memory.
  iree_DyLibExecutableDef_table_t def;

  wasm_module_t wasm_module;
  wasm_module_inst_t wasm_module_inst;
  wasm_exec_env_t wasm_exec_env;

  // Resolved entry points from |wasm_module_inst|.
  iree_host_size_t entry_fn_count;
  wasm_function_inst_t entry_fns[];
} iree_hal_wasm_executable_t;

static const iree_hal_local_executable_vtable_t iree_hal_wasm_executable_vtable;

static iree_status_t iree_hal_wasm_executable_load(
    iree_hal_wasm_executable_t* executable, iree_allocator_t host_allocator) {
  RuntimeInitArgs init_args;
  memset(&init_args, 0, sizeof(RuntimeInitArgs));

  // TODO(scotttodd): use Alloc_With_Allocator and forward the host_allocator
  init_args.mem_alloc_type = Alloc_With_System_Allocator;
  // init_args.mem_alloc_option.allocator.malloc_func = NULL;
  //       (iree_allocator_alloc_fn_t + IREE_ALLOCATION_MODE_ZERO_CONTENTS)
  // init_args.mem_alloc_option.allocator.realloc_func = NULL;
  //       (iree_allocator_alloc_fn_t + IREE_ALLOCATION_MODE_TRY_REUSE_EXISTING)
  // init_args.mem_alloc_option.allocator.free_func = NULL;
  //       (iree_allocator_free_fn_t)

  // TODO(scotttodd): figure out where names come from...
  //   * a more specific name including "IREE" or a file name would be nice
  //   * "env" is load bearing for resolving native symbols:
  //      (import "env" "fmaf" (func $fmaf (type 0)))
  init_args.native_module_name = "env";

  // `fmaf` is still used by some models
  // TODO(scotttodd): enforce no imports, remove `--allow-undefined` linker flag
  static NativeSymbol native_symbols[] = {
      {.symbol = "fmaf",
       .func_ptr = fmaf_impl,
       .signature = "(fff)f",
       .attachment = NULL},
  };
  init_args.native_symbols = native_symbols;
  init_args.n_native_symbols = IREE_ARRAYSIZE(native_symbols);

  if (!wasm_runtime_full_init(&init_args)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "wasm_runtime_full_init failed");
  }

  flatbuffers_uint8_vec_t embedded_library_vec =
      iree_DyLibExecutableDef_library_embedded_get(executable->def);
  char error_buffer[128];
  executable->wasm_module = wasm_runtime_load(
      embedded_library_vec, flatbuffers_uint8_vec_len(embedded_library_vec),
      error_buffer, sizeof(error_buffer));
  if (!executable->wasm_module) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "wasm_runtime_load failed: %s", error_buffer);
  }

  // TODO(scotttodd): make this configurable
  uint32_t stack_size = 1073741824;  // 1 GB
  uint32_t heap_size = 1073741824;   // 1 GB
  executable->wasm_module_inst =
      wasm_runtime_instantiate(executable->wasm_module, stack_size, heap_size,
                               error_buffer, sizeof(error_buffer));
  if (!executable->wasm_module_inst) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "wasm_runtime_instantiate failed: %s",
                            error_buffer);
  }

  executable->wasm_exec_env =
      wasm_runtime_create_exec_env(executable->wasm_module_inst, stack_size);
  if (!executable->wasm_exec_env) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "wasm_runtime_create_exec_env failed");
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_wasm_executable_resolve_symbols(
    iree_hal_wasm_executable_t* executable) {
  flatbuffers_string_vec_t entry_points_vec =
      iree_DyLibExecutableDef_entry_points_get(executable->def);
  for (iree_host_size_t i = 0; i < executable->entry_fn_count; ++i) {
    flatbuffers_string_t entry_point_str =
        flatbuffers_string_vec_at(entry_points_vec, i);
    wasm_function_inst_t wasm_function_inst = wasm_runtime_lookup_function(
        executable->wasm_module_inst, entry_point_str, /*signature=*/NULL);
    if (!wasm_function_inst) {
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "symbol %s not exported by the WASM module, check visibility",
          entry_point_str);
    }
    executable->entry_fns[i] = wasm_function_inst;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_wasm_executable_create(
    iree_DyLibExecutableDef_table_t executable_def,
    iree_host_size_t executable_layout_count,
    iree_hal_executable_layout_t* const* executable_layouts,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_def);
  IREE_ASSERT_ARGUMENT(!executable_layout_count || executable_layouts);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  flatbuffers_string_vec_t entry_points_vec =
      iree_DyLibExecutableDef_entry_points_get(executable_def);
  iree_host_size_t entry_point_count =
      flatbuffers_string_vec_len(entry_points_vec);
  if (entry_point_count != executable_layout_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable provides %zu entry points but caller "
                            "provided %zu; must match",
                            entry_point_count, executable_layout_count);
  }

  iree_hal_wasm_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) + entry_point_count * sizeof(*executable->entry_fns) +
      executable_layout_count * sizeof(iree_hal_local_executable_layout_t);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_local_executable_layout_t** executable_layouts_ptr =
        (iree_hal_local_executable_layout_t**)(((uint8_t*)executable) +
                                               sizeof(*executable) +
                                               entry_point_count *
                                                   sizeof(
                                                       *executable->entry_fns));
    iree_hal_local_executable_initialize(
        &iree_hal_wasm_executable_vtable, executable_layout_count,
        executable_layouts, executable_layouts_ptr, host_allocator,
        &executable->base);
    executable->def = executable_def;
    executable->entry_fn_count = entry_point_count;
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_wasm_executable_load(executable, host_allocator);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_wasm_executable_resolve_symbols(executable);
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_release((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_wasm_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_wasm_executable_t* executable =
      (iree_hal_wasm_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(scotttodd): ref counting / sharing between threads?
  if (executable->wasm_exec_env) {
    wasm_runtime_destroy_exec_env(executable->wasm_exec_env);
  }
  if (executable->wasm_module_inst) {
    wasm_runtime_deinstantiate(executable->wasm_module_inst);
  }
  if (executable->wasm_module) {
    wasm_runtime_unload(executable->wasm_module);
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_wasm_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_vec3_t* workgroup_id) {
  iree_hal_wasm_executable_t* executable =
      (iree_hal_wasm_executable_t*)base_executable;

  // WebAssembly modules do not have access to any memory outside of their
  // sandboxed runtime unless explicitly allowed through module imports.

  // TODO(#4022): Use a smarter allocator / allocation scheme.
  //   This is just MVP placeholder code.

  // Deep clone dispatch_state and workgroup_id using WASM-accessible memory.
  // All pointers in these structs should be WASM virtual memory addresses, not
  // the typical host pointers or memory addresses.

  // Clone dispatch_state.
  void* native_buffer_dispatch_state = NULL;
  uint32_t wasm_buffer_dispatch_state = wasm_runtime_module_malloc(
      executable->wasm_module_inst,
      sizeof(iree_hal_executable_dispatch_state_v0_t),
      &native_buffer_dispatch_state);
  if (IREE_UNLIKELY(wasm_buffer_dispatch_state == 0)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "malloc failed");
  }
  iree_hal_executable_dispatch_state_v0_t* wasm_dispatch_state =
      (iree_hal_executable_dispatch_state_v0_t*)native_buffer_dispatch_state;
  // Workground count/size.
  wasm_dispatch_state->workgroup_count.x = dispatch_state->workgroup_count.x;
  wasm_dispatch_state->workgroup_count.y = dispatch_state->workgroup_count.y;
  wasm_dispatch_state->workgroup_count.z = dispatch_state->workgroup_count.z;
  wasm_dispatch_state->workgroup_size.x = dispatch_state->workgroup_size.x;
  wasm_dispatch_state->workgroup_size.y = dispatch_state->workgroup_size.y;
  wasm_dispatch_state->workgroup_size.z = dispatch_state->workgroup_size.z;
  // Push constants.
  wasm_dispatch_state->push_constant_count =
      dispatch_state->push_constant_count;
  void* native_buffer_push_constants = NULL;
  uint32_t wasm_buffer_push_constants = wasm_runtime_module_malloc(
      executable->wasm_module_inst,
      dispatch_state->push_constant_count * sizeof(int32_t*),
      &native_buffer_push_constants);
  if (IREE_UNLIKELY(wasm_buffer_push_constants == 0)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "malloc failed");
  }
  int32_t* wasm_push_constants = (int32_t*)native_buffer_push_constants;
  for (int i = 0; i < dispatch_state->push_constant_count; ++i) {
    wasm_push_constants[i] = dispatch_state->push_constants[i];
  }
  // TODO(scotttodd): fix:
  //   '=': 'const uint32_t *' differs in levels of indirection from 'uint32_t'
  wasm_dispatch_state->push_constants = wasm_buffer_push_constants;
  // Bindings.
  wasm_dispatch_state->binding_count = dispatch_state->binding_count;
  //   - binding_ptrs
  void* native_buffer_binding_ptrs = NULL;
  uint32_t wasm_buffer_binding_ptrs = wasm_runtime_module_malloc(
      executable->wasm_module_inst,
      dispatch_state->binding_count * sizeof(int32_t*),
      &native_buffer_binding_ptrs);
  if (IREE_UNLIKELY(wasm_buffer_binding_ptrs == 0)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "malloc failed");
  }
  int32_t* wasm_binding_ptrs = (int32_t*)native_buffer_binding_ptrs;
  // HACK: remember the last native buffer (usually the single output buffer)
  //   This is broken in the general case, but lets us memcpy results back
  //   from WASM to the host, without extra bookkeeping.
  //   Revisit when using a real allocation scheme.
  void* latest_native_buffer_binding = NULL;
  for (int i = 0; i < dispatch_state->binding_count; ++i) {
    void* native_buffer_binding = NULL;
    uint32_t wasm_buffer_binding = wasm_runtime_module_malloc(
        executable->wasm_module_inst, dispatch_state->binding_lengths[i],
        &native_buffer_binding);
    if (IREE_UNLIKELY(wasm_buffer_binding == 0)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "malloc failed");
    }
    memcpy(native_buffer_binding, dispatch_state->binding_ptrs[i],
           dispatch_state->binding_lengths[i]);
    latest_native_buffer_binding = native_buffer_binding;
    wasm_binding_ptrs[i] = wasm_buffer_binding;
  }
  // TODO(scotttodd): fix
  //   '=': 'void *const *' differs in levels of indirection from 'uint32_t'
  wasm_dispatch_state->binding_ptrs = wasm_buffer_binding_ptrs;
  //   - binding_lengths
  void* native_buffer_binding_lengths = NULL;
  uint32_t wasm_buffer_binding_lengths = wasm_runtime_module_malloc(
      executable->wasm_module_inst,
      dispatch_state->binding_count * sizeof(size_t*),
      &native_buffer_binding_lengths);
  if (IREE_UNLIKELY(wasm_buffer_binding_lengths == 0)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "malloc failed");
  }
  int32_t* wasm_binding_lengths = (int32_t*)native_buffer_binding_lengths;
  for (int i = 0; i < dispatch_state->binding_count; ++i) {
    wasm_binding_lengths[i] = dispatch_state->binding_lengths[i];
  }
  // TODO(scotttodd): fix
  //   '=': 'const size_t *' differs in levels of indirection from 'uint32_t'
  wasm_dispatch_state->binding_lengths = wasm_buffer_binding_lengths;
  // Imports.
  // TODO(scotttodd): implement import table if needed? fail if not-NULL?
  wasm_dispatch_state->imports = NULL;

  // Clone workgroup_id.
  void* native_buffer_workgroup_id = NULL;
  uint32_t wasm_buffer_workgroup_id = wasm_runtime_module_malloc(
      executable->wasm_module_inst, sizeof(iree_hal_vec3_t),
      &native_buffer_workgroup_id);
  if (IREE_UNLIKELY(wasm_buffer_workgroup_id == 0)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "malloc failed");
  }
  iree_hal_vec3_t* wasm_workgroup_id =
      (iree_hal_vec3_t*)native_buffer_workgroup_id;
  wasm_workgroup_id->x = workgroup_id->x;
  wasm_workgroup_id->y = workgroup_id->y;
  wasm_workgroup_id->z = workgroup_id->z;

  uint32_t argv[2];
  argv[0] = wasm_buffer_dispatch_state;
  argv[1] = wasm_buffer_workgroup_id;

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  flatbuffers_string_t entry_point_str = flatbuffers_string_vec_at(
      iree_DyLibExecutableDef_entry_points_get(executable->def), ordinal);
  iree_string_view_t entry_point_name = iree_make_string_view(
      entry_point_str, flatbuffers_string_len(entry_point_str));
  if (iree_string_view_is_empty(entry_point_name)) {
    entry_point_name = iree_make_cstring_view("unknown_wasm_call");
  }
  IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z0, entry_point_name.data,
                                      entry_point_name.size);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  if (!wasm_runtime_call_wasm(executable->wasm_exec_env,
                              executable->entry_fns[ordinal],
                              /*argc=*/2, argv)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL, "wasm_runtime_call_wasm failed: %s",
        wasm_runtime_get_exception(executable->wasm_module_inst));
  }

  IREE_TRACE_ZONE_END(z0);

  // Copy result(s) back from WASM.
  // HACK: See above. This isn't really correct, but works for simple cases.
  memcpy(dispatch_state->binding_ptrs[dispatch_state->binding_count - 1],
         latest_native_buffer_binding,
         dispatch_state->binding_lengths[dispatch_state->binding_count - 1]);

  wasm_runtime_module_free(executable->wasm_module_inst,
                           wasm_buffer_workgroup_id);
  wasm_runtime_module_free(executable->wasm_module_inst,
                           wasm_buffer_push_constants);
  for (int i = 0; i < dispatch_state->binding_count; ++i) {
    wasm_runtime_module_free(executable->wasm_module_inst,
                             wasm_binding_ptrs[i]);
  }
  wasm_runtime_module_free(executable->wasm_module_inst,
                           wasm_buffer_binding_ptrs);
  wasm_runtime_module_free(executable->wasm_module_inst,
                           wasm_buffer_binding_lengths);
  wasm_runtime_module_free(executable->wasm_module_inst,
                           wasm_buffer_dispatch_state);

  return iree_ok_status();
}

static const iree_hal_local_executable_vtable_t
    iree_hal_wasm_executable_vtable = {
        .base =
            {
                .destroy = iree_hal_wasm_executable_destroy,
            },
        .issue_call = iree_hal_wasm_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_wasm_module_loader_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
} iree_hal_wasm_module_loader_t;

static const iree_hal_executable_loader_vtable_t
    iree_hal_wasm_module_loader_vtable;

iree_status_t iree_hal_wasm_module_loader_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_wasm_module_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(&iree_hal_wasm_module_loader_vtable,
                                          &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_wasm_module_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_wasm_module_loader_t* executable_loader =
      (iree_hal_wasm_module_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_wasm_module_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_hal_executable_format_t executable_format) {
  return executable_format == iree_hal_make_executable_format("WASM");
}

static iree_status_t iree_hal_wasm_module_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable) {
  iree_hal_wasm_module_loader_t* executable_loader =
      (iree_hal_wasm_module_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(scotttodd): replace with system library code, no flatbuffer wrapper

  // Verify and fetch the executable flatbuffer wrapper.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_dylib_executable_flatbuffer_verify(
              executable_spec->executable_data,
              executable_spec->executable_layout_count));
  iree_DyLibExecutableDef_table_t executable_def =
      iree_DyLibExecutableDef_as_root(executable_spec->executable_data.data);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_wasm_executable_create(
              executable_def, executable_spec->executable_layout_count,
              executable_spec->executable_layouts,
              executable_loader->host_allocator, out_executable));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_hal_executable_loader_vtable_t
    iree_hal_wasm_module_loader_vtable = {
        .destroy = iree_hal_wasm_module_loader_destroy,
        .query_support = iree_hal_wasm_module_loader_query_support,
        .try_load = iree_hal_wasm_module_loader_try_load,
};
