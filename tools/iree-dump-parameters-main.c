// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dumps parsed parameter file information.
// We intentionally use the same flags and parsing behavior as the rest of the
// runtime tools so that users can pass their parameter flags and see exactly
// what the runtime tools would see post-indexing. We don't try to dump original
// file metadata as we only support parsing enough information for what we need.
//
// We also support basic extraction of individual parameters in cases where
// users want to dump them out. Since we don't parse or preserve metadata we
// can't easily dump them to
//
// # List all available parameters and their index information:
// $ iree-dump-parameters --parameters=my_scope=my_file.gguf [--parameters=...]
// # Extract parameter binary contents from a file:
// $ iree-dump-parameters ... --extract=scope::key0=file0.bin [--extract=...]

#include <ctype.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/io/parameter_index.h"
#include "iree/io/scope_map.h"
#include "iree/tooling/parameter_util.h"

//===----------------------------------------------------------------------===//
// Parameter extraction
//===----------------------------------------------------------------------===//

IREE_FLAG_LIST(string, extract,
               "Extracts a parameter to a file as `[scope::]key=file.bin`.");

static iree_status_t iree_tooling_extract_parameter(
    iree_io_scope_map_t* scope_map, iree_string_view_t scope,
    iree_string_view_t key, iree_string_view_t path,
    iree_allocator_t host_allocator) {
  // Lookup the index for the given scope.
  iree_io_parameter_index_t* index = NULL;  // unowned
  IREE_RETURN_IF_ERROR(iree_io_scope_map_lookup(scope_map, scope, &index));

  // Lookup the entry within the index.
  const iree_io_parameter_index_entry_t* entry = NULL;  // unowned
  IREE_RETURN_IF_ERROR(iree_io_parameter_index_lookup(index, key, &entry));

  fprintf(stdout, "Extracting parameter `");
  if (!iree_string_view_is_empty(scope)) {
    fprintf(stdout, "%.*s::", (int)scope.size, scope.data);
  }
  fprintf(stdout, "%.*s` (%" PRIu64 "b) to `%.*s`...\n", (int)key.size,
          key.data, entry->length, (int)path.size, path.data);

  if (entry->type != IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot extract parameters of type %d",
                            (int)entry->type);
  }

  // TODO(benvanik): support generic file handle IO instead of memory-only.
  if (iree_io_file_handle_type(entry->storage.file.handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "only host allocation file handles are supported today");
  }
  iree_byte_span_t file_contents =
      iree_io_file_handle_value(entry->storage.file.handle).host_allocation;
  iree_const_byte_span_t entry_contents = iree_make_const_byte_span(
      file_contents.data + entry->storage.file.offset, entry->length);
  char* path_str = (char*)iree_alloca(path.size + 1);
  memcpy(path_str, path.data, path.size);
  path_str[path.size] = 0;
  return iree_file_write_contents(path_str, entry_contents);
}

static iree_status_t iree_tooling_extract_parameters(
    iree_io_scope_map_t* scope_map, iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0; i < FLAG_extract_list().count; ++i) {
    iree_string_view_t flag = FLAG_extract_list().values[i];
    iree_string_view_t identifier, path;
    iree_string_view_split(flag, '=', &identifier, &path);

    iree_host_size_t separator_pos =
        iree_string_view_find_first_of(identifier, IREE_SV("::"), 0);
    iree_string_view_t scope = iree_string_view_empty();
    iree_string_view_t key = iree_string_view_empty();
    if (separator_pos != IREE_STRING_VIEW_NPOS) {
      scope = iree_string_view_substr(identifier, 0, separator_pos);
      key = iree_string_view_substr(identifier, separator_pos + 2,
                                    IREE_HOST_SIZE_MAX);
    } else {
      key = identifier;
    }

    IREE_RETURN_IF_ERROR(iree_tooling_extract_parameter(scope_map, scope, key,
                                                        path, host_allocator),
                         "extracting parameter with flag `%.*s`",
                         (int)flag.size, flag.data);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

int main(int argc, char** argv) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  // Parse command line flags.
  iree_flags_set_usage("iree-dump-parameters",
                       "Dumps information about parsed parameter files.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  if (argc > 1) {
      fprintf(stderr, "Error, no positional arguments expected. \n"
                      "Pass --parameters=flags\n");
      exit_code = EXIT_FAILURE;
  }

  iree_io_scope_map_t scope_map = {0};
  iree_io_scope_map_initialize(host_allocator, &scope_map);

  // Parse parameters using the common tooling flags.
  iree_status_t status =
      iree_tooling_build_parameter_indices_from_flags(&scope_map);

  // Dump parameter information.
  if (iree_status_is_ok(status)) {
    iree_string_builder_t sb;
    iree_string_builder_initialize(host_allocator, &sb);
    status = iree_io_scope_map_dump(&scope_map, &sb);
    if (iree_status_is_ok(status)) {
      fprintf(stdout, "%.*s", (int)iree_string_builder_size(&sb),
              iree_string_builder_buffer(&sb));
    }
    iree_string_builder_deinitialize(&sb);
  }

  // Extract parameters as requested, if any.
  if (iree_status_is_ok(status)) {
    status = iree_tooling_extract_parameters(&scope_map, host_allocator);
  }

  iree_io_scope_map_deinitialize(&scope_map);

  fflush(stdout);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);

  IREE_TRACE_ZONE_END(z0);
  return exit_code;
}
