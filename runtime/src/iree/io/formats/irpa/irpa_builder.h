// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_FORMATS_IRPA_IRPA_BUILDER_H_
#define IREE_IO_FORMATS_IRPA_IRPA_BUILDER_H_

#include "iree/base/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/stream.h"
#include "iree/schemas/parameter_archive.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Utility to build IRPA (IREE Parameter Archive) headers in-memory.
//
// Example:
//  iree_io_parameter_archive_builder_t builder;
//  iree_io_parameter_archive_builder_initialize(host_allocator, &builder);
//  iree_io_parameter_archive_builder_add_data_entry(&builder, ...);
//  iree_io_parameter_archive_builder_add_data_entry(&builder, ...);
//  iree_io_parameter_archive_builder_add_data_entry(&builder, ...);
//  total_size = iree_io_parameter_archive_builder_total_size(&builder);
//  << create file of total_size, map into memory >>
//  iree_io_parameter_archive_builder_write(&builder, file, &target_index);
//  << file now contains the full archive header >>
//  << target_index now references the ranges in the file >>
//  << write parameter contents, or don't if leaving uninitialized >>
//  iree_io_parameter_archive_builder_deinitialize(&builder);
typedef struct iree_io_parameter_archive_builder_t {
  iree_allocator_t host_allocator;
  iree_io_parameter_index_t* index;
  iree_io_physical_size_t file_alignment;
  iree_io_physical_size_t entry_segment_size;
  iree_io_physical_size_t metadata_segment_size;
  iree_io_physical_size_t storage_segment_size;
  iree_io_physical_size_t storage_alignment;
} iree_io_parameter_archive_builder_t;

// Initializes a new parameter builder in |out_builder| for use.
// iree_io_parameter_archive_builder_deinitialize must be called to drop any
// resources retained while building.
IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_initialize(
    iree_allocator_t host_allocator,
    iree_io_parameter_archive_builder_t* out_builder);

// Deinitializes |builder| and drops all resources.
IREE_API_EXPORT void iree_io_parameter_archive_builder_deinitialize(
    iree_io_parameter_archive_builder_t* builder);

// Returns the total file size required to store the parameter archive header
// and contents of all added parameters. Adding new parameters will invalidate
// this value.
IREE_API_EXPORT iree_io_physical_size_t
iree_io_parameter_archive_builder_total_size(
    const iree_io_parameter_archive_builder_t* builder);

// Writes the parameter archive to the given |file_handle|. The file must have
// at least enough storage to fit iree_io_parameter_archive_builder_total_size.
// The archive will be written starting at the given |file_offset|.
// If an optional |target_index| is provided entries for all parameters will be
// appended to the index referencing the given |file_handle|.
IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_write(
    const iree_io_parameter_archive_builder_t* builder,
    iree_io_file_handle_t* file_handle, iree_io_physical_offset_t file_offset,
    iree_io_stream_t* stream, iree_io_parameter_index_t* target_index);

// Adds a new splat entry to |builder|.
// Splat entries have no physical storage and exist only in the header.
// |pattern| and |metadata| (if provided) are copied prior to returning.
// |pattern_length| must be <= 16 (enough for complex<f64>).
IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_add_splat_entry(
    iree_io_parameter_archive_builder_t* builder, iree_string_view_t name,
    iree_const_byte_span_t metadata, const void* pattern,
    uint8_t pattern_length, iree_io_physical_size_t data_length);

// Adds a new data entry to |builder|.
// |metadata| (if provided) is copied prior to returning.
// Physical storage will be allocated for |data_length| and it will be aligned
// to at least |minimum_alignment|.
IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_add_data_entry(
    iree_io_parameter_archive_builder_t* builder, iree_string_view_t name,
    iree_const_byte_span_t metadata, iree_io_physical_size_t minimum_alignment,
    iree_io_physical_size_t data_length);

// Callback for opening a file for writing.
// Implementations need to ensure that at least |archive_length| bytes are
// available in the file starting at |archive_offset|.
typedef iree_status_t(IREE_API_PTR* iree_io_parameter_archive_file_open_fn_t)(
    void* user_data, iree_io_physical_offset_t archive_offset,
    iree_io_physical_size_t archive_length,
    iree_io_file_handle_t** out_file_handle);

// A callback issued to open a file.
typedef struct {
  // Callback function pointer.
  iree_io_parameter_archive_file_open_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_io_parameter_archive_file_open_callback_t;

// Builds a parameter archive from the given |source_index| and returns a new
// index in |target_index| referencing the new archive file.
// The total size of the archive will be calculated and the provided
// |target_file_open| callback will be used to acquire a handle to a writeable
// file with enough capacity to fit the whole archive. All parameter contents
// will be written and flushed to the file prior to returning.
IREE_API_EXPORT iree_status_t iree_io_build_parameter_archive(
    iree_io_parameter_index_t* source_index,
    iree_io_parameter_index_t* target_index,
    iree_io_parameter_archive_file_open_callback_t target_file_open,
    iree_io_physical_offset_t target_file_offset,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_FORMATS_IRPA_IRPA_BUILDER_H_
