// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_PROFILE_FILE_H_
#define IREE_HAL_UTILS_PROFILE_FILE_H_

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/profile_sink.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Magic header bytes for IREE HAL profile bundle files: "IRPF".
#define IREE_HAL_PROFILE_FILE_MAGIC 0x46505249u

// Major version of the IREE HAL profile bundle file format.
#define IREE_HAL_PROFILE_FILE_VERSION_MAJOR 1u

// Minor version of the IREE HAL profile bundle file format.
#define IREE_HAL_PROFILE_FILE_VERSION_MINOR 6u

// File header stored at byte 0 of every IREE HAL profile bundle.
typedef struct iree_hal_profile_file_header_t {
  // Magic bytes equal to IREE_HAL_PROFILE_FILE_MAGIC.
  uint32_t magic;
  // Major version of the file format.
  uint16_t version_major;
  // Minor version of the file format.
  uint16_t version_minor;
  // Size of this header in bytes.
  uint32_t header_length;
  // Reserved for future file-level flags; must be zero.
  uint32_t flags;
  // Logical byte length of the completed profile bundle, or zero if the bundle
  // has not been finalized with a session end record.
  uint64_t file_length;
} iree_hal_profile_file_header_t;

// Type of one record in an IREE HAL profile bundle file.
typedef uint16_t iree_hal_profile_file_record_type_t;
enum iree_hal_profile_file_record_type_e {
  IREE_HAL_PROFILE_FILE_RECORD_TYPE_NONE = 0u,
  // Profiling session begin marker with session metadata.
  IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN = 1u,
  // Profiling chunk emitted by iree_hal_profile_sink_write.
  IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK = 2u,
  // Profiling session end marker with terminal status code.
  IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END = 3u,
};

// Record header followed by content type bytes, name bytes, and payload bytes.
//
// All multi-byte fields are serialized in little-endian order. The current
// writer only supports little-endian hosts; a future big-endian implementation
// must byte-swap instead of changing the file format.
typedef struct iree_hal_profile_file_record_header_t {
  // Total byte length of this record including header, strings, and payload.
  uint64_t record_length;
  // Total payload byte length after content type and name strings.
  uint64_t payload_length;
  // Profiling session identifier from iree_hal_profile_chunk_metadata_t.
  uint64_t session_id;
  // Stream identifier from iree_hal_profile_chunk_metadata_t.
  uint64_t stream_id;
  // Event identifier from iree_hal_profile_chunk_metadata_t.
  uint64_t event_id;
  // Executable identifier from iree_hal_profile_chunk_metadata_t.
  uint64_t executable_id;
  // Command-buffer identifier from iree_hal_profile_chunk_metadata_t.
  uint64_t command_buffer_id;
  // Size of this record header in bytes.
  uint32_t header_length;
  // Byte length of the content type string following this header.
  uint32_t content_type_length;
  // Byte length of the name string following the content type string.
  uint32_t name_length;
  // Physical device ordinal from iree_hal_profile_chunk_metadata_t.
  uint32_t physical_device_ordinal;
  // Queue ordinal from iree_hal_profile_chunk_metadata_t.
  uint32_t queue_ordinal;
  // Chunk flags from iree_hal_profile_chunk_metadata_t for CHUNK records,
  // otherwise zero.
  iree_hal_profile_chunk_flags_t chunk_flags;
  // Dropped record count from iree_hal_profile_chunk_metadata_t for CHUNK
  // records, otherwise zero.
  uint64_t dropped_record_count;
  // Session end status code for SESSION_END records, otherwise zero.
  uint32_t session_status_code;
  // Type of this file record.
  iree_hal_profile_file_record_type_t record_type;
  // Reserved for future record-level flags; must be zero.
  uint16_t flags;
} iree_hal_profile_file_record_header_t;

static_assert(sizeof(iree_hal_profile_file_header_t) == 24,
              "profile file header layout must remain explicit");
static_assert(offsetof(iree_hal_profile_file_header_t, file_length) == 16,
              "profile file header file_length offset must remain explicit");
static_assert(sizeof(iree_hal_profile_file_record_header_t) == 104,
              "profile file record header layout must remain explicit");
static_assert(offsetof(iree_hal_profile_file_record_header_t, payload_length) ==
                  8,
              "profile file record payload_length offset must remain explicit");
static_assert(offsetof(iree_hal_profile_file_record_header_t, header_length) ==
                  56,
              "profile file record header_length offset must remain explicit");
static_assert(offsetof(iree_hal_profile_file_record_header_t, record_type) ==
                  100,
              "profile file record_type offset must remain explicit");

// Borrowed view of a parsed profile bundle record.
//
// All pointers reference the original file contents passed to
// iree_hal_profile_file_parse_record and remain valid only for as long as the
// file contents remain mapped/allocated.
typedef struct iree_hal_profile_file_record_t {
  // Parsed record header value.
  iree_hal_profile_file_record_header_t header;
  // Content type string following |header|.
  iree_string_view_t content_type;
  // Record name string following |content_type|.
  iree_string_view_t name;
  // Payload bytes following |name|.
  iree_const_byte_span_t payload;
} iree_hal_profile_file_record_t;

// Creates a sink that writes raw HAL profile chunks to |file_handle|.
//
// The caller supplies the target file handle so embedders can choose normal
// files, host allocations, temporary files, or platform-specific handles. The
// sink opens a writable stream at offset zero, writes a provisional file header
// during creation, then appends session begin/chunk/end records as callbacks
// arrive. A successful end_session patches the header with the logical file
// length. Consumers must ignore bytes after that logical length; this lets
// embedders write to fixed-size host allocations or existing file handles
// without stale trailing storage becoming parseable records.
//
// The returned sink is not internally synchronized. Callers must serialize
// callbacks for a single sink instance, which matches the HAL profiling API.
IREE_API_EXPORT iree_status_t iree_hal_profile_file_sink_create(
    iree_io_file_handle_t* file_handle, iree_allocator_t host_allocator,
    iree_hal_profile_sink_t** out_sink);

// Parses and validates the file header in |file_contents|.
//
// On success |out_header| contains the parsed header and |out_record_offset|
// points at the first record. If |out_header| has a nonzero |file_length|,
// callers should parse records only within that logical byte extent.
// Forward-compatible header extensions are skipped according to
// |header_length|.
IREE_API_EXPORT iree_status_t
iree_hal_profile_file_parse_header(iree_const_byte_span_t file_contents,
                                   iree_hal_profile_file_header_t* out_header,
                                   iree_host_size_t* out_record_offset);

// Parses one record beginning at |record_offset| in |file_contents|.
// |file_contents| should be capped to iree_hal_profile_file_header_t::
// file_length when nonzero so stale storage after the logical bundle end is
// not visible to low-level record parsing.
//
// On success |out_record| contains borrowed views into |file_contents| and
// |out_next_record_offset| points at the next record, or file end when the
// parsed record was the final record.
IREE_API_EXPORT iree_status_t iree_hal_profile_file_parse_record(
    iree_const_byte_span_t file_contents, iree_host_size_t record_offset,
    iree_hal_profile_file_record_t* out_record,
    iree_host_size_t* out_next_record_offset);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_PROFILE_FILE_H_
