// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_FILE_H_
#define IREE_HAL_REPLAY_FILE_H_

#include "iree/base/api.h"
#include "iree/hal/replay/format.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque append-only writer for `.ireereplay` files.
typedef struct iree_hal_replay_file_writer_t iree_hal_replay_file_writer_t;

// Metadata used to append one replay file record.
typedef struct iree_hal_replay_file_record_metadata_t {
  // Capture-order sequence ordinal assigned by the shared recorder.
  uint64_t sequence_ordinal;
  // Capture-time host thread identifier, or zero when unknown.
  uint64_t thread_id;
  // Session-local device object id associated with this record.
  iree_hal_replay_object_id_t device_id;
  // Primary session-local object id associated with this record.
  iree_hal_replay_object_id_t object_id;
  // Secondary session-local object id associated with this record.
  iree_hal_replay_object_id_t related_object_id;
  // Type of replay file record to append.
  iree_hal_replay_file_record_type_t record_type;
  // Flags specifying optional record behavior.
  iree_hal_replay_file_record_flags_t record_flags;
  // Producer-defined payload schema identifier.
  uint32_t payload_type;
  // HAL object kind for object records, or zero otherwise.
  iree_hal_replay_object_type_t object_type;
  // HAL operation kind for operation records, or zero otherwise.
  iree_hal_replay_operation_code_t operation_code;
  // IREE status code returned by the captured operation, or zero for OK/none.
  uint32_t status_code;
} iree_hal_replay_file_record_metadata_t;

// Borrowed view of a parsed replay file record.
//
// All pointers reference the original file contents passed to
// iree_hal_replay_file_parse_record and remain valid only for as long as the
// file contents remain mapped/allocated.
typedef struct iree_hal_replay_file_record_t {
  // Parsed record header value.
  iree_hal_replay_file_record_header_t header;
  // Payload bytes following |header|.
  iree_const_byte_span_t payload;
} iree_hal_replay_file_record_t;

// Creates a writer that writes an IREE HAL replay file to |file_handle|.
//
// The returned writer opens a writable stream at offset zero and immediately
// writes a placeholder file header. Callers must close the writer with
// iree_hal_replay_file_writer_close before freeing it so the final file length
// is written into the header.
IREE_API_EXPORT iree_status_t iree_hal_replay_file_writer_create(
    iree_io_file_handle_t* file_handle, iree_allocator_t host_allocator,
    iree_hal_replay_file_writer_t** out_writer);

// Closes |writer| and updates the file header with the final file length.
//
// Safe to call multiple times. Appending records after close fails.
IREE_API_EXPORT iree_status_t
iree_hal_replay_file_writer_close(iree_hal_replay_file_writer_t* writer);

// Frees |writer| without reporting close errors.
//
// Callers that need a complete executable replay artifact must call
// iree_hal_replay_file_writer_close and check its status before freeing.
IREE_API_EXPORT void iree_hal_replay_file_writer_free(
    iree_hal_replay_file_writer_t* writer);

// Appends one replay file record with payload bytes from |iovecs|.
//
// |out_payload_range|, if non-NULL, receives a replay-file range pointing to
// the payload bytes written for the record. Zero-length payloads still receive
// a valid range with length zero and an integrity digest of empty contents.
IREE_API_EXPORT iree_status_t iree_hal_replay_file_writer_append_record(
    iree_hal_replay_file_writer_t* writer,
    const iree_hal_replay_file_record_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs,
    iree_hal_replay_file_range_t* out_payload_range);

// Parses and validates the file header in |file_contents|.
//
// On success |out_header| contains the parsed header and |out_record_offset|
// points at the first record. Forward-compatible header extensions are skipped
// according to |header_length|. Callers should bound later record parsing to
// |out_header->file_length| bytes when |file_contents| may include unused tail
// storage beyond the completed replay file.
IREE_API_EXPORT iree_status_t
iree_hal_replay_file_parse_header(iree_const_byte_span_t file_contents,
                                  iree_hal_replay_file_header_t* out_header,
                                  iree_host_size_t* out_record_offset);

// Parses one record beginning at |record_offset| in |file_contents|.
//
// On success |out_record| contains borrowed views into |file_contents| and
// |out_next_record_offset| points at the next record, or file end when the
// parsed record was the final record.
IREE_API_EXPORT iree_status_t iree_hal_replay_file_parse_record(
    iree_const_byte_span_t file_contents, iree_host_size_t record_offset,
    iree_hal_replay_file_record_t* out_record,
    iree_host_size_t* out_next_record_offset);

// Validates that |range| is structurally valid and fits within |file_contents|.
//
// For supported digest types this also checks the stored digest against the
// referenced bytes.
IREE_API_EXPORT iree_status_t
iree_hal_replay_file_range_validate(iree_const_byte_span_t file_contents,
                                    const iree_hal_replay_file_range_t* range);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_FILE_H_
