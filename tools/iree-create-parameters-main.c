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
#include "iree/io/stream.h"

//===----------------------------------------------------------------------===//
// Parameter builder logic
//===----------------------------------------------------------------------===//

IREE_FLAG_LIST(
    string, splat,
    "Declares a splat parameter in the form `name=shapextype=pattern`.\n"
    "Splat parameters have no storage on disk and can be used to mock real\n"
    "parameters that besides load-time performance and contents behave the\n"
    "same as if real parameters had been used. The shape and type are used to\n"
    "calculate the parameter size in the index and the value is interpreted\n"
    "based on the type. Note that splat parameters cannot be mutated at\n"
    "runtime and can only be used for constant values.\n");

// TODO(benvanik): support @file.bin syntax, numpy files, etc.
IREE_FLAG_LIST(
    string, data,
    "Declares a data parameter in the form `name=shapextype=pattern`.\n"
    "Data parameters have a storage reservation in the final archive with\n"
    "either zeroed contents or a specified repeating pattern value.\n"
    "The shape and type are used to calculate the parameter size in the index\n"
    "and the value is interpreted based on the type. Omitting the value\n"
    "will leave the parameter with zeroed contents on disk.");

// TODO(benvanik): support external entries (--external=file.bin#128-512).

IREE_FLAG(int32_t, alignment, IREE_IO_PARAMETER_ARCHIVE_DEFAULT_DATA_ALIGNMENT,
          "Storage data alignment relative to the header.");

typedef struct {
  iree_string_view_t name;
  uint64_t storage_size;
  uint64_t element_count;
  iree_hal_element_type_t element_type;
  struct {
    uint8_t pattern[16];
    uint8_t pattern_length;
  } splat;
} iree_io_parameter_info_t;

// Parses a `name=shapextype[=value]` flag string.
static iree_status_t iree_io_parameter_info_from_string(
    iree_string_view_t parameter_value, iree_io_parameter_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));

  iree_string_view_t spec;
  iree_string_view_split(parameter_value, '=', &out_info->name, &spec);

  iree_string_view_t shape_type, contents;
  iree_string_view_split(spec, '=', &shape_type, &contents);

  iree_host_size_t shape_rank = 0;
  iree_hal_dim_t shape[16] = {0};
  IREE_RETURN_IF_ERROR(iree_hal_parse_shape_and_element_type(
      shape_type, IREE_ARRAYSIZE(shape), &shape_rank, shape,
      &out_info->element_type));

  if (IREE_UNLIKELY(iree_hal_element_bit_count(out_info->element_type) == 0) ||
      IREE_UNLIKELY(
          !iree_hal_element_is_byte_aligned(out_info->element_type))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "opaque and sub-byte aligned element types cannot currently be used as "
        "splats; use a bitcast type (2xi4=1xi8, etc)");
  }

  out_info->storage_size =
      iree_hal_element_dense_byte_count(out_info->element_type);
  out_info->element_count = 1;
  for (iree_host_size_t i = 0; i < shape_rank; ++i) {
    out_info->storage_size *= shape[i];
    out_info->element_count *= shape[i];
  }

  // TODO(benvanik): support external files and such; for now we just assume
  // either empty string (0 splat) or a splat value.

  if (iree_string_view_is_empty(contents)) {
    out_info->splat.pattern_length = 1;
    memset(out_info->splat.pattern, 0, sizeof(out_info->splat.pattern));
  } else {
    iree_device_size_t byte_count =
        iree_hal_element_dense_byte_count(out_info->element_type);
    if (byte_count > sizeof(out_info->splat.pattern)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "element type size for %.*s out of range of splat patterns",
          (int)shape_type.size, shape_type.data);
    }
    out_info->splat.pattern_length = (uint8_t)byte_count;
    IREE_RETURN_IF_ERROR(iree_hal_parse_element(
        contents, out_info->element_type,
        iree_make_byte_span(out_info->splat.pattern, 16)));
  }
  return iree_ok_status();
}

// Declares parameter metadata for all parameters specified by flags.
static iree_status_t iree_tooling_declare_parameters(
    iree_io_parameter_archive_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Metadata-only parameters first; they have no storage and the fact that they
  // are not in-order with any data parameters means they don't impact runtime
  // access locality behavior.
  for (iree_host_size_t i = 0; i < FLAG_splat_list().count; ++i) {
    iree_io_parameter_info_t info;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_io_parameter_info_from_string(FLAG_splat_list().values[i], &info));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_io_parameter_archive_builder_add_splat_entry(
            builder, info.name, /*metadata=*/iree_const_byte_span_empty(),
            info.splat.pattern, info.splat.pattern_length, info.storage_size));
  }

  // Data parameters follow and will appear in storage in the order they were
  // declared with flags.
  for (iree_host_size_t i = 0; i < FLAG_data_list().count; ++i) {
    iree_io_parameter_info_t info;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_io_parameter_info_from_string(FLAG_data_list().values[i], &info));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_parameter_archive_builder_add_data_entry(
                builder, info.name, /*metadata=*/iree_const_byte_span_empty(),
                FLAG_alignment, info.storage_size));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Defines parameter storage for those that require it.
static iree_status_t iree_tooling_define_parameters(
    iree_io_parameter_index_t* target_index,
    iree_io_physical_offset_t target_file_offset,
    iree_io_stream_t* target_stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < FLAG_data_list().count; ++i) {
    iree_io_parameter_info_t info;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_io_parameter_info_from_string(FLAG_data_list().values[i], &info));
    const iree_io_parameter_index_entry_t* target_entry = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_io_parameter_index_lookup(target_index, info.name, &target_entry));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_stream_seek(
                target_stream, IREE_IO_STREAM_SEEK_SET,
                target_file_offset + target_entry->storage.file.offset));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_stream_fill(target_stream, info.element_count,
                                info.splat.pattern, info.splat.pattern_length));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

IREE_FLAG(bool, quiet, false,
          "Silences additional stdout output when not needed.");

// TODO(benvanik): add --append= to chain archive headers.
IREE_FLAG(string, output, "", "Output .irpa file path.");

static void iree_io_file_handle_release_mapping(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  iree_file_contents_free((iree_file_contents_t*)user_data);
}

static iree_status_t iree_tooling_open_output_parameter_file(
    iree_io_physical_offset_t archive_offset,
    iree_io_physical_size_t archive_length, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_file_handle) {
  iree_file_contents_t* file_contents = NULL;
  IREE_RETURN_IF_ERROR(iree_file_create_mapped(
      FLAG_output, archive_offset + archive_length, archive_offset,
      (iree_host_size_t)archive_length, host_allocator, &file_contents));
  iree_io_file_handle_release_callback_t release_callback = {
      .fn = iree_io_file_handle_release_mapping,
      .user_data = file_contents,
  };
  iree_status_t status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_WRITE, file_contents->buffer, release_callback,
      host_allocator, out_file_handle);
  if (!iree_status_is_ok(status)) {
    iree_file_contents_free(file_contents);
  }
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  // Parse command line flags.
  iree_flags_set_usage(
      "iree-create-parameters",
      "Creates IREE Parameter Archive (.irpa) files. Provide zero or more\n"
      "parameter value declarations and an output file with\n"
      "`--output=file.irpa` to produce a new file with zeroed or patterned\n"
      "contents.\n"
      "\n"
      "Parameter declarations take a shape and type in order to calculate the\n"
      "required storage size of the parameter at rest and at runtime. The\n"
      "shape and type need not match what the consuming program expects so\n"
      "long as the storage size is equivalent; for example, if the program\n"
      "expects a parameter of type `tensor<8x2xi4>` the parameter declaration\n"
      "can be `8xi8`, `1xi64`, `2xf32`, etc.\n"
      "\n"
      "Example creating a file with two embedded data parameters that have\n"
      "zeroed contents and one with a repeating pattern:\n"
      "  iree-create-parameters \\\n"
      "    --data=my.zeroed_param_1=4096xf32 \\\n"
      "    --data=my.zeroed_param_2=2x4096xi16 \\\n"
      "    --data=my.pattern_param_2=8x2xf32=2.1 \\\n"
      "    --output=output_with_storage.irpa\n"
      "\n"
      "Example creating a file with splatted values (no storage on disk):\n"
      "  iree-create-parameters \\\n"
      "    --splat=my.splat_param_1=4096xf32=4.1 \\\n"
      "    --splat=my.splat_param_2=2x4096xi16=123 \\\n"
      "    --output=output_without_storage.irpa\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_io_parameter_archive_builder_t builder;
  iree_io_parameter_archive_builder_initialize(host_allocator, &builder);

  // Declare parameters based on flags, populating the builder with the metadata
  // for each parameter without yet writing any data.
  iree_status_t status = iree_tooling_declare_parameters(&builder);

  // Open a file of sufficient size (now that we know it) for writing.
  iree_io_physical_offset_t target_file_offset = 0;
  iree_io_physical_offset_t archive_offset = iree_align_uint64(
      target_file_offset, IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT);
  iree_io_physical_size_t archive_length =
      iree_io_parameter_archive_builder_total_size(&builder);
  iree_io_file_handle_t* target_file_handle = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_open_output_parameter_file(
        archive_offset, archive_length, host_allocator, &target_file_handle);
  }

  // Wrap the target file in a stream.
  iree_io_stream_t* target_stream = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_open(IREE_IO_STREAM_MODE_WRITABLE, target_file_handle,
                            target_file_offset, host_allocator, &target_stream);
  }

  // Allocate an index we'll populate during building to allow us to get the
  // storage ranges of non-metadata parameters.
  iree_io_parameter_index_t* built_index = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_create(host_allocator, &built_index);
  }

  // Commit the archive header to the file and produce an index referencing it.
  // This will allow us to know where to copy file contents.
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_archive_builder_write(
        &builder, target_file_handle, target_file_offset, target_stream,
        built_index);
  }

  // Define non-metadata-only parameters that use the data storage segment.
  if (iree_status_is_ok(status)) {
    status = iree_tooling_define_parameters(built_index, target_file_offset,
                                            target_stream);
  }

  // Dump the new index ala iree-dump-parameters to show the final file.
  if (iree_status_is_ok(status) && !FLAG_quiet) {
    status = iree_io_parameter_index_fprint(stdout, iree_string_view_empty(),
                                            built_index);
  }

  iree_io_stream_release(target_stream);
  iree_io_file_handle_release(target_file_handle);
  iree_io_parameter_archive_builder_deinitialize(&builder);
  iree_io_parameter_index_release(built_index);

  fflush(stdout);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
