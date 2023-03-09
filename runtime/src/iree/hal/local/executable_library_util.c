// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_library_util.h"

iree_status_t iree_hal_executable_library_verify(
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_executable_library_v0_t* library) {
  // Tooling and testing may disable verification to make it easier to define
  // libraries. The compiler should never produce anything that fails
  // verification, though, and should always have it enabled.
  const bool disable_verification =
      iree_all_bits_set(executable_params->caching_mode,
                        IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION);
  if (disable_verification) return iree_ok_status();

  // Check that there's one pipeline layout per export. Multiple exports may
  // share the same layout but it still needs to be declared.
  // NOTE: pipeline layouts are optional but if provided must be consistent.
  if (executable_params->pipeline_layout_count > 0) {
    if (library->exports.count != executable_params->pipeline_layout_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "executable provides %u entry points but caller "
                              "provided %zu; must match",
                              library->exports.count,
                              executable_params->pipeline_layout_count);
    }
  }

  // Check to make sure that the constant table has values for all constants.
  if (library->constants.count != executable_params->constant_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable requires %u constants but caller "
                            "provided %zu; must match",
                            library->constants.count,
                            executable_params->constant_count);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_executable_library_initialize_imports(
    iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_import_provider_t import_provider,
    const iree_hal_executable_import_table_v0_t* import_table,
    iree_hal_executable_import_thunk_v0_t import_thunk,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(import_thunk);
  if (!import_table || !import_table->count) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, import_table->count);

  // The thunk is used to give the loader a chance to intercept import calls
  // in cases where it needs to JIT, perform FFI/ABI conversion, etc.
  environment->import_thunk = import_thunk;

  // Allocate storage for the imports.
  iree_host_size_t import_funcs_size =
      iree_host_align(import_table->count * sizeof(*environment->import_funcs),
                      iree_max_align_t);
  iree_host_size_t import_contexts_size = iree_host_align(
      import_table->count * sizeof(*environment->import_contexts),
      iree_max_align_t);
  uint8_t* base_ptr = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator,
                                import_funcs_size + import_contexts_size,
                                (void**)&base_ptr));
  environment->import_funcs = (const iree_hal_executable_import_v0_t*)base_ptr;
  environment->import_contexts = (const void**)(base_ptr + import_funcs_size);

  // Try to resolve each import.
  // Will fail if any required import is not found.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_executable_import_provider_try_resolve(
              import_provider, import_table->count, import_table->symbols,
              (void**)environment->import_funcs,
              (void**)environment->import_contexts,
              /*out_resolution=*/NULL));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_executable_library_deinitialize_imports(
    iree_hal_executable_environment_v0_t* environment,
    iree_allocator_t host_allocator) {
  // NOTE: import_funcs and import_contexts are allocated as one block.
  if (environment->import_funcs != NULL) {
    iree_allocator_free(host_allocator, (void*)environment->import_funcs);
  }
  environment->import_funcs = NULL;
  environment->import_contexts = NULL;
  environment->import_thunk = NULL;
}

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

iree_zone_id_t iree_hal_executable_library_call_zone_begin(
    iree_string_view_t executable_identifier,
    const iree_hal_executable_library_v0_t* library, iree_host_size_t ordinal) {
  iree_string_view_t entry_point_name = iree_string_view_empty();
  if (library->exports.names != NULL) {
    entry_point_name = iree_make_cstring_view(library->exports.names[ordinal]);
  }
  if (iree_string_view_is_empty(entry_point_name)) {
    entry_point_name = iree_make_cstring_view("unknown_dylib_call");
  }

  const char* source_file = NULL;
  size_t source_file_length = 0;
  uint32_t source_line;
  if (library->exports.src_locs != NULL) {
    // We have source location data, so use it.
    source_file = library->exports.src_locs[ordinal].path;
    source_file_length = library->exports.src_locs[ordinal].path_length;
    source_line = library->exports.src_locs[ordinal].line;
  } else {
    // No source location data, so make do with what we have.
    source_file = executable_identifier.data;
    source_file_length = executable_identifier.size;
    source_line = ordinal;
  }

  IREE_TRACE_ZONE_BEGIN_EXTERNAL(z0, source_file, source_file_length,
                                 source_line, entry_point_name.data,
                                 entry_point_name.size, NULL, 0);

  if (library->exports.tags != NULL) {
    const char* tag = library->exports.tags[ordinal];
    if (tag) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, tag);
    }
  }

  return z0;
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
