// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_READER_H_
#define IREE_TOOLING_PROFILE_READER_H_

#include "iree/io/file_contents.h"
#include "iree/tooling/profile/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_profile_file_t {
  // Mapped profile file contents owned by this wrapper.
  iree_io_file_contents_t* contents;
  // Parsed profile file header.
  iree_hal_profile_file_header_t header;
  // Byte offset of the first record after the file header.
  iree_host_size_t first_record_offset;
  // Logical byte length to parse, excluding any stale trailing storage.
  iree_host_size_t file_length;
} iree_profile_file_t;

// Function signature invoked once per parsed profile file record.
typedef iree_status_t (*iree_profile_file_record_callback_fn_t)(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index);

typedef struct iree_profile_file_record_callback_t {
  // Function invoked once per parsed profile file record.
  iree_profile_file_record_callback_fn_t fn;
  // User data passed to |fn|.
  void* user_data;
} iree_profile_file_record_callback_t;

typedef struct iree_profile_typed_record_t {
  // Source chunk containing this typed record.
  const iree_hal_profile_file_record_t* chunk;
  // Zero-based ordinal of this typed record within |chunk|.
  iree_host_size_t record_index;
  // Byte offset of this typed record within |chunk->payload|.
  iree_host_size_t payload_offset;
  // Minimum fixed header length requested by the parser.
  iree_host_size_t minimum_record_length;
  // Total byte length reported by the typed record prefix.
  iree_host_size_t record_length;
  // Full bytes covered by |record_length|.
  iree_const_byte_span_t contents;
  // Bytes after |minimum_record_length| and before |record_length|.
  iree_const_byte_span_t inline_payload;
  // Bytes after |record_length| through the end of the containing chunk.
  iree_const_byte_span_t following_payload;
} iree_profile_typed_record_t;

typedef struct iree_profile_typed_record_iterator_t {
  // Source chunk being iterated.
  const iree_hal_profile_file_record_t* chunk;
  // Minimum fixed header length to validate for each typed record.
  iree_host_size_t minimum_record_length;
  // Current byte offset into |chunk->payload|.
  iree_host_size_t payload_offset;
  // Zero-based ordinal for the next typed record.
  iree_host_size_t record_index;
} iree_profile_typed_record_iterator_t;

// Parses one typed record from |chunk->payload| at |payload_offset|.
iree_status_t iree_profile_typed_record_parse(
    const iree_hal_profile_file_record_t* chunk,
    iree_host_size_t payload_offset, iree_host_size_t minimum_record_length,
    iree_host_size_t record_index, iree_profile_typed_record_t* out_record);

// Parses the single executable trace artifact carried by |chunk|.
//
// Executable trace chunks are intentionally not generic packed typed-record
// streams: the fixed trace record has record_length == sizeof(record), and all
// bytes after that fixed record are the raw trace artifact. |out_trace_data|
// may be NULL when the caller only needs metadata validation.
iree_status_t iree_profile_executable_trace_record_parse(
    const iree_hal_profile_file_record_t* chunk,
    iree_hal_profile_executable_trace_record_t* out_record,
    iree_const_byte_span_t* out_trace_data);

// Initializes an iterator over packed typed records in |chunk->payload|.
void iree_profile_typed_record_iterator_initialize(
    const iree_hal_profile_file_record_t* chunk,
    iree_host_size_t minimum_record_length,
    iree_profile_typed_record_iterator_t* out_iterator);

// Advances |iterator| and returns the next typed record if present.
iree_status_t iree_profile_typed_record_iterator_next(
    iree_profile_typed_record_iterator_t* iterator,
    iree_profile_typed_record_t* out_record, bool* out_has_record);

// Opens and maps a HAL profile bundle for borrowed record iteration.
iree_status_t iree_profile_file_open(iree_string_view_t path,
                                     iree_allocator_t host_allocator,
                                     iree_profile_file_t* out_profile_file);

// Releases the mapping owned by |profile_file|.
void iree_profile_file_close(iree_profile_file_t* profile_file);

// Iterates all records in |profile_file| in file order.
iree_status_t iree_profile_file_for_each_record(
    const iree_profile_file_t* profile_file,
    iree_profile_file_record_callback_t callback);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_READER_H_
