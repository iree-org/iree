// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_FILE_READER_H_
#define IREE_HAL_REPLAY_FILE_READER_H_

#include "iree/base/api.h"
#include "iree/hal/replay/format.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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

#endif  // IREE_HAL_REPLAY_FILE_READER_H_
