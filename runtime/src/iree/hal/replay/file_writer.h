// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_FILE_WRITER_H_
#define IREE_HAL_REPLAY_FILE_WRITER_H_

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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_FILE_WRITER_H_
