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

  // Check to make sure that the constant table has values for all constants.
  if (library->constants.count != executable_params->constant_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable requires %u constants but caller "
                            "provided %" PRIhsz "; must match",
                            library->constants.count,
                            executable_params->constant_count);
  }

  // Validate that dispatch attributes are present.
  if (!library->exports.attrs) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable exports must provide dispatch attributes");
  }

  // Validate dispatch attributes are in range.
  for (uint32_t i = 0; i < library->exports.count; ++i) {
    const iree_hal_executable_dispatch_attrs_v0_t dispatch_attrs =
        library->exports.attrs[i];
    if (dispatch_attrs.constant_count >
        IREE_HAL_EXECUTABLE_MAX_CONSTANT_COUNT) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch requiring %u constants exceeds limit of %d",
          dispatch_attrs.constant_count,
          IREE_HAL_EXECUTABLE_MAX_CONSTANT_COUNT);
    }
    if (dispatch_attrs.binding_count > IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch requiring %u bindings exceeds limit of %d",
          dispatch_attrs.binding_count, IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT);
    }
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
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, import_table->count);

  // The thunk is used to give the loader a chance to intercept import calls
  // in cases where it needs to JIT, perform FFI/ABI conversion, etc.
  environment->import_thunk = import_thunk;

  // Allocate storage for the imports with overflow checking.
  iree_host_size_t total_size = 0;
  iree_host_size_t import_contexts_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &total_size,
              IREE_STRUCT_FIELD_ALIGNED(import_table->count,
                                        iree_hal_executable_import_v0_t,
                                        iree_max_align_t, NULL),
              IREE_STRUCT_FIELD_ALIGNED(import_table->count, const void*,
                                        iree_max_align_t,
                                        &import_contexts_offset)));
  uint8_t* base_ptr = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&base_ptr));
  environment->import_funcs = (const iree_hal_executable_import_v0_t*)base_ptr;
  environment->import_contexts =
      (const void**)(base_ptr + import_contexts_offset);

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

iree_host_size_t iree_hal_executable_library_export_count(
    const iree_hal_executable_library_v0_t* library) {
  IREE_ASSERT_ARGUMENT(library);
  return library->exports.count;
}

iree_status_t iree_hal_executable_library_export_info(
    const iree_hal_executable_library_v0_t* library,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(out_info);
  memset(out_info, 0, sizeof(*out_info));

  if (export_ordinal >= library->exports.count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %u out of range (count: %u)",
                            export_ordinal, library->exports.count);
  }

  // Set the name if available.
  if (library->exports.names && library->exports.names[export_ordinal]) {
    out_info->name =
        iree_make_cstring_view(library->exports.names[export_ordinal]);
  } else {
    out_info->name = iree_string_view_empty();
  }

  // Get dispatch attributes if available.
  if (library->exports.attrs) {
    const iree_hal_executable_dispatch_attrs_v0_t* attrs =
        &library->exports.attrs[export_ordinal];

    if (iree_any_bit_set(attrs->flags,
                         IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_SEQUENTIAL)) {
      out_info->flags |= IREE_HAL_EXECUTABLE_EXPORT_FLAG_SEQUENTIAL;
    }
    if (iree_any_bit_set(
            attrs->flags,
            IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_WORKGROUP_SIZE_DYNAMIC)) {
      out_info->flags |= IREE_HAL_EXECUTABLE_EXPORT_FLAG_WORKGROUP_SIZE_DYNAMIC;
    }

    out_info->constant_count = attrs->constant_count;
    out_info->binding_count = attrs->binding_count;
    out_info->parameter_count = attrs->parameter_count;

    out_info->workgroup_size[0] = attrs->workgroup_size_x;
    out_info->workgroup_size[1] = attrs->workgroup_size_y;
    out_info->workgroup_size[2] = attrs->workgroup_size_z;
  }

  // Occupancy info is not yet implemented.
  memset(&out_info->occupancy_info, 0, sizeof(out_info->occupancy_info));

  return iree_ok_status();
}

iree_status_t iree_hal_executable_library_export_parameters(
    const iree_hal_executable_library_v0_t* library,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(out_parameters || capacity == 0);

  if (export_ordinal >= library->exports.count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %u out of range (count: %u)",
                            export_ordinal, library->exports.count);
  }

  if (!library->exports.attrs || !library->exports.params ||
      !library->exports.params[export_ordinal]) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "reflection data not available");
  }

  const iree_hal_executable_dispatch_attrs_v0_t* attrs =
      &library->exports.attrs[export_ordinal];
  const iree_hal_executable_dispatch_parameter_v0_t** params =
      library->exports.params;
  const iree_hal_executable_dispatch_parameter_v0_t* export_params =
      params[export_ordinal];

  // Copy as many parameters as we can fit.
  iree_host_size_t count =
      iree_min(capacity, (iree_host_size_t)attrs->parameter_count);
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_hal_executable_dispatch_parameter_v0_t* src = &export_params[i];
    iree_hal_executable_export_parameter_t* dst = &out_parameters[i];
    switch (src->type) {
      case IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT:
        dst->type = IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_CONSTANT;
        break;
      case IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING:
        dst->type = IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING;
        break;
      case IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BUFFER_PTR:
        dst->type = IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BUFFER_PTR;
        break;
      default:
        dst->type = IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_CONSTANT;
        break;
    }
    dst->size = src->size;
    dst->flags = src->flags;
    dst->offset = src->offset;
    if (library->exports.parameter_names && src->name < UINT16_MAX &&
        library->exports.parameter_names[src->name]) {
      dst->name =
          iree_make_cstring_view(library->exports.parameter_names[src->name]);
    } else {
      dst->name = iree_string_view_empty();
    }
  }

  return iree_ok_status();
}

iree_status_t iree_hal_executable_library_lookup_export_by_name(
    const iree_hal_executable_library_v0_t* library, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(out_export_ordinal);

  // Names array must exist for lookup by name.
  if (!library->exports.names) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "export names not available in library");
  }

  // Linear search through export names.
  for (uint32_t i = 0; i < library->exports.count; ++i) {
    if (library->exports.names[i]) {
      iree_string_view_t export_name =
          iree_make_cstring_view(library->exports.names[i]);
      if (iree_string_view_equal(export_name, name)) {
        *out_export_ordinal = i;
        return iree_ok_status();
      }
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "export '%.*s' not found in library", (int)name.size,
                          name.data);
}

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

void iree_hal_executable_library_publish_source_files(
    const iree_hal_executable_library_v0_t* library) {
  for (uint32_t i = 0; i < library->sources.count; ++i) {
    const iree_hal_executable_source_file_v0_t* source_file =
        &library->sources.files[i];
    IREE_TRACE_PUBLISH_SOURCE_FILE(source_file->path, source_file->path_length,
                                   source_file->content,
                                   source_file->content_length);
  }
}

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
  uint32_t source_line = 0;
  if (library->exports.stage_locations != NULL) {
    for (uint32_t i = 0; i < library->exports.stage_locations->count; ++i) {
      // TODO(benvanik): a way to select what location is chosen. For now we
      // just pick the first one.
      // const char* name = library->exports.stage_locations->names[i];
      const iree_hal_executable_source_location_v0_t* location =
          &library->exports.stage_locations->locations[i];
      source_file = location->path;
      source_file_length = location->path_length;
      source_line = location->line;
      break;
    }
  }
  if (source_file == NULL) {
    if (library->exports.source_locations != NULL) {
      // We have source location data, so use it.
      const iree_hal_executable_source_location_v0_t* location =
          &library->exports.source_locations[ordinal];
      source_file = location->path;
      source_file_length = location->path_length;
      source_line = location->line;
    } else {
      // No source location data, so make do with what we have.
      source_file = executable_identifier.data;
      source_file_length = executable_identifier.size;
      source_line = ordinal;
    }
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
