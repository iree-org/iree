// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/parameter_util.h"

#include "iree/base/internal/flags.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/parser_registry.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/scope_map.h"
#include "iree/modules/io/parameters/module.h"

//===----------------------------------------------------------------------===//
// Parameter file I/O
//===----------------------------------------------------------------------===//

IREE_FLAG(
    string, parameter_mode, "file",
    "A parameter I/O mode of ['preload', 'file'].\n"
    "  preload: read entire parameter files into wired memory on startup.\n"
    "  file: uses platform file APIs to read/write the file as needed.");

// Opens the parameter file at |path| with the mode specified by the
// --parameter_mode flag and returns its handle.
static iree_status_t iree_io_open_parameter_file(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_file_handle) {
  IREE_ASSERT_ARGUMENT(out_file_handle);
  *out_file_handle = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  iree_status_t status = iree_ok_status();
  iree_io_file_handle_t* file_handle = NULL;
  if (strcmp(FLAG_parameter_mode, "preload") == 0) {
    status = iree_io_file_handle_preload(IREE_IO_FILE_MODE_READ, path,
                                         host_allocator, &file_handle);
  } else if (strcmp(FLAG_parameter_mode, "file") == 0) {
    status = iree_io_file_handle_open(IREE_IO_FILE_MODE_READ, path,
                                      host_allocator, &file_handle);
  } else {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unrecognized --parameter_mode= value '%s'",
                              FLAG_parameter_mode);
  }

  if (iree_status_is_ok(status)) {
    *out_file_handle = file_handle;
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
  iree_status_t status =
      iree_io_parse_file_index(path, file_handle, index, host_allocator);

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
