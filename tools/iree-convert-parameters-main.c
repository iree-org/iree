// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Converts one or more parameter files into a single IREE Parameter Archive.
// Allows for stripping and renaming parameters as basic editing features.

#include <ctype.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/api.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/io/parameter_index.h"
#include "iree/io/scope_map.h"
#include "iree/tooling/parameter_util.h"

//===----------------------------------------------------------------------===//
// Parameter index logic
//===----------------------------------------------------------------------===//

IREE_FLAG_LIST(string, exclude,
               "Excludes a named parameter from the resulting file.");
IREE_FLAG_LIST(string, rename,
               "Renames a parameter when adding to the resulting file in the "
               "form of `--rename=old=new`.");
IREE_FLAG(bool, strip, false,
          "Strips all parameters by replacing them with zeros.");
IREE_FLAG_LIST(
    string, splat,
    "Turns a parameter into a splat of 0 (`--splat=name`) or a specific\n"
    "sequence of typed values (`--splat=name=i8=123`, `--splat=name=f32=4.5`,\n"
    "`--splat=name=x32=CAFEF00D`).");

static bool iree_tooling_is_parameter_excluded(iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < FLAG_exclude_list().count; ++i) {
    if (iree_string_view_equal(FLAG_exclude_list().values[i], name)) {
      return true;
    }
  }
  return false;
}

static iree_string_view_t iree_tooling_get_renamed_parameter_name(
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < FLAG_rename_list().count; ++i) {
    iree_string_view_t old_name, new_name;
    iree_string_view_split(FLAG_rename_list().values[i], '=', &old_name,
                           &new_name);
    if (iree_string_view_equal(old_name, name)) {
      return new_name;
    }
  }
  return name;
}

// Expects `type=value` consistent with the HAL.
static iree_status_t iree_tooling_parse_splat(iree_string_view_t splat_value,
                                              uint8_t* out_pattern_length,
                                              uint8_t* out_pattern) {
  if (iree_string_view_is_empty(splat_value)) {
    *out_pattern_length = 1;
    out_pattern[0] = 0;
    return iree_ok_status();
  }

  iree_string_view_t type_str, value_str;
  iree_string_view_split(splat_value, '=', &type_str, &value_str);

  iree_hal_element_type_t type = IREE_HAL_ELEMENT_TYPE_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_parse_element_type(type_str, &type));

  iree_device_size_t byte_count = iree_hal_element_dense_byte_count(type);
  if (byte_count > 16) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "element type size for %.*s out of range of splat patterns",
        (int)type_str.size, type_str.data);
  }
  *out_pattern_length = (uint8_t)byte_count;

  return iree_hal_parse_element(value_str, type,
                                iree_make_byte_span(out_pattern, 16));
}

static iree_status_t iree_tooling_replace_splatted_parameter(
    iree_io_parameter_index_entry_t* entry) {
  // Always favor specific splat values.
  for (iree_host_size_t i = 0; i < FLAG_splat_list().count; ++i) {
    iree_string_view_t name, splat_value;
    iree_string_view_split(FLAG_splat_list().values[i], '=', &name,
                           &splat_value);
    if (iree_string_view_equal(name, entry->key)) {
      entry->type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
      memset(&entry->storage, 0, sizeof(entry->storage));
      return iree_tooling_parse_splat(splat_value,
                                      &entry->storage.splat.pattern_length,
                                      entry->storage.splat.pattern);
    }
  }

  // If not specifically splatted then see if we are stripping and use that.
  if (FLAG_strip) {
    entry->type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
    memset(&entry->storage, 0, sizeof(entry->storage));
    entry->storage.splat.pattern_length = 1;
    entry->storage.splat.pattern[0] = 0;
    return iree_ok_status();
  }

  return iree_ok_status();
}

static iree_status_t iree_tooling_convert_parameter_index(
    iree_io_parameter_index_t* source_index,
    iree_io_parameter_index_t* target_index) {
  for (iree_host_size_t i = 0; i < iree_io_parameter_index_count(source_index);
       ++i) {
    // Get the existing entry we'll use as a template.
    const iree_io_parameter_index_entry_t* source_entry = NULL;
    IREE_RETURN_IF_ERROR(
        iree_io_parameter_index_get(source_index, i, &source_entry));
    iree_io_parameter_index_entry_t target_entry = *source_entry;

    // If the parameter is in the exclude list then we just skip it.
    if (iree_tooling_is_parameter_excluded(source_entry->key)) continue;

    // If the parameter is in the rename list we'll add it with the new name.
    target_entry.key =
        iree_tooling_get_renamed_parameter_name(source_entry->key);

    // If the parameter is turned into a splat we change its type. Note that it
    // may have already been a splat but the user may want to change the value.
    IREE_RETURN_IF_ERROR(
        iree_tooling_replace_splatted_parameter(&target_entry));

    // Add the entry (potentially modified) to the new index.
    IREE_RETURN_IF_ERROR(
        iree_io_parameter_index_add(target_index, &target_entry));
  }
  return iree_ok_status();
}

static iree_status_t iree_tooling_convert_parameters(
    iree_io_scope_map_t* scope_map, iree_io_parameter_index_t* target_index,
    iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0; i < scope_map->count; ++i) {
    IREE_RETURN_IF_ERROR(iree_tooling_convert_parameter_index(
        scope_map->entries[i]->index, target_index));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

IREE_FLAG(bool, quiet, false,
          "Silences additional stdout output when not needed.");

IREE_FLAG(string, output, "", "Output .irpa file path.");

static void iree_io_file_handle_release_mapping(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  iree_file_contents_free((iree_file_contents_t*)user_data);
}

typedef struct {
  iree_allocator_t host_allocator;
  const char* path;
} iree_tooling_open_params_t;
static iree_status_t iree_tooling_open_output_parameter_file(
    void* user_data, iree_io_physical_offset_t archive_offset,
    iree_io_physical_size_t archive_length,
    iree_io_file_handle_t** out_file_handle) {
  iree_tooling_open_params_t* params = (iree_tooling_open_params_t*)user_data;
  iree_file_contents_t* file_contents = NULL;
  IREE_RETURN_IF_ERROR(
      iree_file_create_mapped(params->path, archive_offset + archive_length,
                              archive_offset, (iree_host_size_t)archive_length,
                              params->host_allocator, &file_contents));
  iree_io_file_handle_release_callback_t release_callback = {
      .fn = iree_io_file_handle_release_mapping,
      .user_data = file_contents,
  };
  iree_status_t status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_WRITE, file_contents->buffer, release_callback,
      params->host_allocator, out_file_handle);
  if (!iree_status_is_ok(status)) {
    iree_file_contents_free(file_contents);
  }
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  // Parse command line flags.
  iree_flags_set_usage(
      "iree-convert-parameters",
      "Converts supported parameter file formats into IREE Parameter Archives\n"
      "(.irpa) files. Provide one or more input parameter files in the same\n"
      "form as expected by the iree-run-module tool (`--parameters=foo.gguf`)\n"
      "and an output file with `--output=file.irpa`.\n"
      "\n"
      "Example converting from safetensors to IRPA:\n"
      "  iree-convert-parameters \\\n"
      "    --parameters=input.safetensors \\\n"
      "    --output=output.irpa\n"
      "\n"
      "Example mutating parameters:\n"
      "  iree-convert-parameters \\\n"
      "    --parameters=a.gguf \\\n"
      "    --parameters=b.safetensors \\\n"
      "    --exclude=unneeded_param \\\n"
      "    --rename=old_name=new_name \\\n"
      "    --splat=some_name=f32=4.2 \\\n"
      "    --output=ab.irpa\n"
      "\n"
      "Example stripping all parameters and replacing them with zeros except\n"
      "for one that needs special handling:\n"
      "  iree-convert-parameters \\\n"
      "    --parameters=input.irpa \\\n"
      "    --strip \\\n"
      "    --splat=special_param=f32=1.0 \\\n"
      "    --output=output.irpa\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  // Load parameter indices as specified by command line flags.
  iree_io_scope_map_t scope_map = {0};
  iree_io_scope_map_initialize(host_allocator, &scope_map);
  iree_status_t status =
      iree_tooling_build_parameter_indices_from_flags(&scope_map);

  // Build the new combined/modified index in memory based on the inputs.
  iree_io_parameter_index_t* new_index = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_create(host_allocator, &new_index);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_tooling_convert_parameters(&scope_map, new_index, host_allocator);
  }
  iree_io_scope_map_deinitialize(&scope_map);

  iree_io_parameter_index_t* built_index = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_create(host_allocator, &built_index);
  }

  // Write out the new archive.
  if (iree_status_is_ok(status)) {
    iree_tooling_open_params_t open_params = {
        .host_allocator = host_allocator,
        .path = FLAG_output,
    };
    iree_io_parameter_archive_file_open_callback_t open_callback = {
        .fn = iree_tooling_open_output_parameter_file,
        .user_data = &open_params,
    };
    status = iree_io_build_parameter_archive(
        new_index, built_index, open_callback,
        /*target_file_offset=*/0, host_allocator);
  }

  // Dump the new index ala iree-dump-parameters to show the final file.
  if (iree_status_is_ok(status) && !FLAG_quiet) {
    status = iree_io_parameter_index_fprint(stdout, iree_string_view_empty(),
                                            built_index);
  }

  iree_io_parameter_index_release(built_index);
  iree_io_parameter_index_release(new_index);

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
