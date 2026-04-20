// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/file_reader.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/replay/digest.h"

#if !defined(IREE_ENDIANNESS_LITTLE) || !IREE_ENDIANNESS_LITTLE
#error "IREE HAL replay file serialization requires little-endian hosts"
#endif  // !IREE_ENDIANNESS_LITTLE

IREE_API_EXPORT iree_status_t
iree_hal_replay_file_parse_header(iree_const_byte_span_t file_contents,
                                  iree_hal_replay_file_header_t* out_header,
                                  iree_host_size_t* out_record_offset) {
  IREE_ASSERT_ARGUMENT(out_header);
  IREE_ASSERT_ARGUMENT(out_record_offset);
  memset(out_header, 0, sizeof(*out_header));
  *out_record_offset = 0;

  if (IREE_UNLIKELY(file_contents.data_length > 0 && !file_contents.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay file contents data is required");
  }
  if (IREE_UNLIKELY(file_contents.data_length <
                    sizeof(iree_hal_replay_file_header_t))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file is too small for header");
  }

  iree_hal_replay_file_header_t header;
  memcpy(&header, file_contents.data, sizeof(header));
  if (IREE_UNLIKELY(header.magic != IREE_HAL_REPLAY_FILE_MAGIC)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid IREE HAL replay file magic");
  }
  if (IREE_UNLIKELY(header.version_major !=
                    IREE_HAL_REPLAY_FILE_VERSION_MAJOR)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported IREE HAL replay file version");
  }
  if (IREE_UNLIKELY(header.header_length < sizeof(header))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file header length is too small");
  }
  if (IREE_UNLIKELY(header.header_length > file_contents.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file header extends past file end");
  }
  if (IREE_UNLIKELY(header.flags != 0)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file reserved flags must be zero");
  }
  if (IREE_UNLIKELY(header.file_length != 0 &&
                    header.file_length < header.header_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file length is before the first record");
  }
  if (IREE_UNLIKELY(header.file_length > file_contents.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file length extends past file contents");
  }

  *out_header = header;
  *out_record_offset = header.header_length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_replay_file_parse_record(
    iree_const_byte_span_t file_contents, iree_host_size_t record_offset,
    iree_hal_replay_file_record_t* out_record,
    iree_host_size_t* out_next_record_offset) {
  IREE_ASSERT_ARGUMENT(out_record);
  IREE_ASSERT_ARGUMENT(out_next_record_offset);
  memset(out_record, 0, sizeof(*out_record));
  *out_next_record_offset = record_offset;

  if (IREE_UNLIKELY(file_contents.data_length > 0 && !file_contents.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay file contents data is required");
  }
  if (IREE_UNLIKELY(record_offset >= file_contents.data_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay record offset is outside the file");
  }

  iree_host_size_t remaining_file_length =
      file_contents.data_length - record_offset;
  if (IREE_UNLIKELY(remaining_file_length <
                    sizeof(iree_hal_replay_file_record_header_t))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay record is too small for header");
  }

  const uint8_t* record_base = file_contents.data + record_offset;
  iree_hal_replay_file_record_header_t header;
  memcpy(&header, record_base, sizeof(header));
  const iree_hal_replay_file_record_flags_t valid_flags =
      IREE_HAL_REPLAY_FILE_RECORD_FLAG_OPTIONAL;
  if (IREE_UNLIKELY((header.record_flags & ~valid_flags) != 0)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay record reserved flags must be zero");
  }
  if (IREE_UNLIKELY(
          !iree_hal_replay_file_record_type_is_known(header.record_type) &&
          !iree_all_bits_set(header.record_flags,
                             IREE_HAL_REPLAY_FILE_RECORD_FLAG_OPTIONAL))) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unknown required replay record type");
  }
  if (IREE_UNLIKELY(header.reserved0 != 0 || header.reserved1 != 0)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay record reserved fields must be zero");
  }
  if (IREE_UNLIKELY(header.record_length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay record length exceeds host size");
  }
  iree_host_size_t record_length = (iree_host_size_t)header.record_length;
  if (IREE_UNLIKELY(record_length > remaining_file_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay record extends past file end");
  }
  if (IREE_UNLIKELY(header.header_length < sizeof(header) ||
                    header.header_length > record_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay record header length is invalid");
  }
  if (IREE_UNLIKELY(header.payload_length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay record payload length exceeds host size");
  }
  if (IREE_UNLIKELY((iree_host_size_t)header.payload_length !=
                    record_length - header.header_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay record payload length does not match");
  }

  const uint8_t* payload = record_base + header.header_length;
  out_record->header = header;
  out_record->payload = iree_make_const_byte_span(
      payload, (iree_host_size_t)header.payload_length);
  *out_next_record_offset = record_offset + record_length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_replay_file_range_validate(iree_const_byte_span_t file_contents,
                                    const iree_hal_replay_file_range_t* range) {
  IREE_ASSERT_ARGUMENT(range);

  if (IREE_UNLIKELY(file_contents.data_length > 0 && !file_contents.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay file contents data is required");
  }
  if (IREE_UNLIKELY(range->flags != 0 || range->reserved0 != 0)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file range reserved fields must be zero");
  }
  if (IREE_UNLIKELY(range->compression_type !=
                    IREE_HAL_REPLAY_COMPRESSION_TYPE_NONE)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported replay file range compression");
  }
  if (IREE_UNLIKELY(range->uncompressed_length != range->length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "uncompressed replay file range length mismatch");
  }
  if (IREE_UNLIKELY(range->offset > UINT64_MAX - range->length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file range end offset overflow");
  }
  uint64_t end_offset = range->offset + range->length;
  if (IREE_UNLIKELY(end_offset > file_contents.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file range extends past file contents");
  }
  if (IREE_UNLIKELY(range->length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file range length exceeds host size");
  }

  iree_host_size_t range_offset = (iree_host_size_t)range->offset;
  iree_const_byte_span_t contents = iree_make_const_byte_span(
      file_contents.data + range_offset, (iree_host_size_t)range->length);
  switch (range->digest_type) {
    case IREE_HAL_REPLAY_DIGEST_TYPE_NONE:
      if (IREE_UNLIKELY(!iree_hal_replay_digest_is_zero(range->digest))) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay file range digest bytes must be zero");
      }
      return iree_ok_status();
    case IREE_HAL_REPLAY_DIGEST_TYPE_FNV1A_64: {
      const uint64_t expected_digest =
          iree_hal_replay_digest_load_fnv1a64(range->digest);
      uint64_t actual_digest = iree_hal_replay_digest_fnv1a64_update(
          iree_hal_replay_digest_fnv1a64_initialize(), contents);
      if (IREE_UNLIKELY(actual_digest != expected_digest)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay file range digest mismatch");
      }
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported replay file range digest");
  }
}
