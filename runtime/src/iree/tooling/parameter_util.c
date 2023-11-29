// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/parameter_util.h"

#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/base/internal/path.h"
#include "iree/io/formats/gguf/gguf_parser.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/io/formats/safetensors/safetensors_parser.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/scope_map.h"
#include "iree/modules/io/parameters/module.h"

//===----------------------------------------------------------------------===//
// Parameter file I/O
//===----------------------------------------------------------------------===//

IREE_FLAG(
    string, parameter_mode, "mmap",
    "A parameter I/O mode of ['preload', 'mmap'].\n"
    "  preload: read entire parameter files into wired memory on startup.\n"
    "  mmap: maps the parameter files into discardable memory - can increase\n"
    "        warm-up time and variance as mapped pages are swapped\n"
    "        by the OS.");

static void iree_file_contents_release_callback(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  iree_file_contents_t* file_contents = (iree_file_contents_t*)user_data;
  iree_file_contents_free(file_contents);
}

// Opens the parameter file at |path| with the mode specified by the
// --parameter_mode flag and returns its handle.
static iree_status_t iree_io_open_parameter_file(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_file_handle) {
  IREE_ASSERT_ARGUMENT(out_file_handle);
  *out_file_handle = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  char path_str[2048] = {0};
  iree_string_view_to_cstring(path, path_str, sizeof(path_str));
  iree_file_read_flags_t read_flags = 0;
  if (strcmp(FLAG_parameter_mode, "mmap") == 0) {
    read_flags |= IREE_FILE_READ_FLAG_MMAP;
  } else if (strcmp(FLAG_parameter_mode, "preload") == 0) {
    read_flags |= IREE_FILE_READ_FLAG_PRELOAD;
  } else {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized --parameter_mode= value '%s'",
                            FLAG_parameter_mode);
  }

  iree_file_contents_t* file_contents = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_file_read_contents(path_str, read_flags, host_allocator,
                                  &file_contents));

  iree_io_file_handle_release_callback_t release_callback = {
      .fn = iree_file_contents_release_callback,
      .user_data = file_contents,
  };
  iree_io_file_handle_t* file_handle = NULL;
  iree_status_t status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ, file_contents->buffer, release_callback,
      host_allocator, &file_handle);
  if (iree_status_is_ok(status)) {
    *out_file_handle = file_handle;
  } else {
    iree_file_contents_free(file_contents);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Parameter file format parsing
//===----------------------------------------------------------------------===//

IREE_FLAG_LIST(
    string, parameters,
    "Specifies a parameter file to make available to programs with either an\n"
    "anonymous global scope (`some_file.gguf`) or a named scope like\n"
    "`my_scope=some_file.gguf`.\n"
    "\n"
    "Supported formats:\n"
    "- .irpa (IREE parameter archive)\n"
    "- .gguf (https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)\n"
    "- .safetensors (https://github.com/huggingface/safetensors)");

// Appends the parameter file located at |path| to |index|.
static iree_status_t iree_io_append_parameter_file_to_index(
    iree_string_view_t path, iree_io_parameter_index_t* index,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Open the file.
  iree_io_file_handle_t* file_handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_open_parameter_file(path, host_allocator, &file_handle));

  // Index the file based on its (inferred) format.
  iree_status_t status = iree_ok_status();
  iree_string_view_t path_ext = iree_file_path_extension(path);
  if (iree_string_view_equal_case(path_ext, IREE_SV("irpa"))) {
    status = iree_io_parse_irpa_index(file_handle, index);
  } else if (iree_string_view_equal_case(path_ext, IREE_SV("gguf"))) {
    status = iree_io_parse_gguf_index(file_handle, index);
  } else if (iree_string_view_equal_case(path_ext, IREE_SV("safetensors"))) {
    status = iree_io_parse_safetensors_index(file_handle, index);
  } else {
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unhandled parameter file format: .%.*s",
                              (int)path_ext.size, path_ext.data);
  }

  // Release our file reference - it's still retained by the index if it had any
  // parameters in it.
  iree_io_file_handle_release(file_handle);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tooling_build_parameter_indices_from_flags(
    iree_io_scope_map_t* scope_map) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create one index per scope and add parameters to each.
  for (iree_host_size_t i = 0; i < FLAG_parameters_list().count; ++i) {
    // Parse the `scope=path` flag. Note that the scope is optional.
    iree_string_view_t flag = FLAG_parameters_list().values[i];
    iree_string_view_t scope, path;
    if (iree_string_view_split(flag, '=', &scope, &path) == -1) {
      // No scope provided (that's ok).
      path = scope;
      scope = iree_string_view_empty();
    }

    // Lookup (or create) the index for the given scope.
    iree_io_parameter_index_t* index = NULL;  // unowned
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_scope_map_lookup(scope_map, scope, &index));

    // Index the file.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_append_parameter_file_to_index(path, index,
                                                   scope_map->host_allocator));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_tooling_create_parameters_module_from_flags(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_scope_map_t scope_map;
  iree_io_scope_map_initialize(host_allocator, &scope_map);

  // Parse all parameter files and build out their indices.
  iree_status_t status =
      iree_tooling_build_parameter_indices_from_flags(&scope_map);

  // Create one provider per scope.
  iree_host_size_t provider_count = 0;
  iree_io_parameter_provider_t** providers =
      (iree_io_parameter_provider_t**)iree_alloca(
          scope_map.count * sizeof(iree_io_parameter_provider_t*));
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < scope_map.count; ++i) {
      status = iree_io_parameter_index_provider_create(
          scope_map.entries[i]->scope, scope_map.entries[i]->index,
          IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
          host_allocator, &providers[i]);
      if (!iree_status_is_ok(status)) break;
      ++provider_count;
    }
  }

  // Create the module with the list of providers.
  if (iree_status_is_ok(status)) {
    status = iree_io_parameters_module_create(
        instance, provider_count, providers, host_allocator, out_module);
  }

  // Cleanup (module owns providers which own indices/etc).
  for (iree_host_size_t i = 0; i < provider_count; ++i) {
    iree_io_parameter_provider_release(providers[i]);
  }
  iree_io_scope_map_deinitialize(&scope_map);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
