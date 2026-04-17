// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/internal.h"

const char* iree_profile_record_type_name(
    iree_hal_profile_file_record_type_t record_type) {
  switch (record_type) {
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN:
      return "session_begin";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK:
      return "chunk";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END:
      return "session_end";
    default:
      return "unknown";
  }
}

void iree_profile_fprint_json_string(FILE* file, iree_string_view_t value) {
  fputc('"', file);
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    uint8_t c = (uint8_t)value.data[i];
    switch (c) {
      case '"':
        fputs("\\\"", file);
        break;
      case '\\':
        fputs("\\\\", file);
        break;
      case '\b':
        fputs("\\b", file);
        break;
      case '\f':
        fputs("\\f", file);
        break;
      case '\n':
        fputs("\\n", file);
        break;
      case '\r':
        fputs("\\r", file);
        break;
      case '\t':
        fputs("\\t", file);
        break;
      default:
        if (c < 0x20) {
          fprintf(file, "\\u%04x", c);
        } else {
          fputc(c, file);
        }
        break;
    }
  }
  fputc('"', file);
}

void iree_profile_fprint_hash_hex(FILE* file, const uint64_t hash[2]) {
  fprintf(file, "%016" PRIx64 "%016" PRIx64, hash[0], hash[1]);
}

static iree_status_t iree_profile_payload_record_length(
    iree_string_view_t content_type, iree_const_byte_span_t payload,
    iree_host_size_t payload_offset, iree_host_size_t minimum_record_length,
    iree_host_size_t* out_record_length) {
  *out_record_length = 0;

  if (payload_offset > payload.data_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile chunk '%.*s' has a typed record offset past the payload",
        (int)content_type.size, content_type.data);
  }

  const iree_host_size_t remaining_length =
      payload.data_length - payload_offset;
  if (remaining_length < minimum_record_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile chunk '%.*s' has a truncated typed record",
                            (int)content_type.size, content_type.data);
  }

  uint32_t record_length = 0;
  memcpy(&record_length, payload.data + payload_offset, sizeof(record_length));
  if (record_length < minimum_record_length ||
      record_length > remaining_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile chunk '%.*s' has invalid typed record length %u",
        (int)content_type.size, content_type.data, record_length);
  }

  *out_record_length = record_length;
  return iree_ok_status();
}

iree_status_t iree_profile_typed_record_parse(
    const iree_hal_profile_file_record_t* chunk,
    iree_host_size_t payload_offset, iree_host_size_t minimum_record_length,
    iree_host_size_t record_index, iree_profile_typed_record_t* out_record) {
  memset(out_record, 0, sizeof(*out_record));

  if (minimum_record_length < sizeof(uint32_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "minimum typed record length must include record_length");
  }

  iree_host_size_t record_length = 0;
  IREE_RETURN_IF_ERROR(iree_profile_payload_record_length(
      chunk->content_type, chunk->payload, payload_offset,
      minimum_record_length, &record_length));

  const uint8_t* record_data = chunk->payload.data + payload_offset;
  out_record->chunk = chunk;
  out_record->record_index = record_index;
  out_record->payload_offset = payload_offset;
  out_record->minimum_record_length = minimum_record_length;
  out_record->record_length = record_length;
  out_record->contents = iree_make_const_byte_span(record_data, record_length);
  out_record->inline_payload =
      iree_make_const_byte_span(record_data + minimum_record_length,
                                record_length - minimum_record_length);
  out_record->following_payload = iree_make_const_byte_span(
      record_data + record_length,
      chunk->payload.data_length - payload_offset - record_length);
  return iree_ok_status();
}

void iree_profile_typed_record_iterator_initialize(
    const iree_hal_profile_file_record_t* chunk,
    iree_host_size_t minimum_record_length,
    iree_profile_typed_record_iterator_t* out_iterator) {
  memset(out_iterator, 0, sizeof(*out_iterator));
  out_iterator->chunk = chunk;
  out_iterator->minimum_record_length = minimum_record_length;
}

iree_status_t iree_profile_typed_record_iterator_next(
    iree_profile_typed_record_iterator_t* iterator,
    iree_profile_typed_record_t* out_record, bool* out_has_record) {
  *out_has_record = false;
  memset(out_record, 0, sizeof(*out_record));

  if (iterator->payload_offset >= iterator->chunk->payload.data_length) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_profile_typed_record_parse(
      iterator->chunk, iterator->payload_offset,
      iterator->minimum_record_length, iterator->record_index, out_record));
  iterator->payload_offset += out_record->record_length;
  ++iterator->record_index;
  *out_has_record = true;
  return iree_ok_status();
}

double iree_profile_sqrt_f64(double value) {
  if (value <= 0.0) return 0.0;
  // Keep this standalone C tool free of libm linkage.
  double estimate = value >= 1.0 ? value : 1.0;
  for (int i = 0; i < 32; ++i) {
    estimate = 0.5 * (estimate + value / estimate);
  }
  return estimate;
}
