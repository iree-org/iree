// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/reader.h"

#include <string.h>

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

iree_status_t iree_profile_executable_trace_record_parse(
    const iree_hal_profile_file_record_t* chunk,
    iree_hal_profile_executable_trace_record_t* out_record,
    iree_const_byte_span_t* out_trace_data) {
  memset(out_record, 0, sizeof(*out_record));
  if (out_trace_data) {
    *out_trace_data = iree_const_byte_span_empty();
  }

  iree_profile_typed_record_t typed_record;
  IREE_RETURN_IF_ERROR(iree_profile_typed_record_parse(
      chunk, 0, sizeof(*out_record), 0, &typed_record));
  if (typed_record.record_length != sizeof(*out_record)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile executable trace chunk record length must exclude raw trace "
        "bytes");
  }

  memcpy(out_record, typed_record.contents.data, sizeof(*out_record));
  if ((iree_host_size_t)out_record->data_length !=
      typed_record.following_payload.data_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile executable trace chunk data length is inconsistent with "
        "payload length");
  }

  if (out_trace_data) {
    *out_trace_data = typed_record.following_payload;
  }
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

iree_status_t iree_profile_file_open(iree_string_view_t path,
                                     iree_allocator_t host_allocator,
                                     iree_profile_file_t* out_profile_file) {
  memset(out_profile_file, 0, sizeof(*out_profile_file));

  iree_status_t status =
      iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ, host_allocator,
                                &out_profile_file->contents);
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(
        out_profile_file->contents->const_buffer, &out_profile_file->header,
        &out_profile_file->first_record_offset);
  }
  if (iree_status_is_ok(status)) {
    uint64_t file_length = out_profile_file->header.file_length;
    if (file_length == 0) {
      file_length = out_profile_file->contents->const_buffer.data_length;
    }
    if (IREE_UNLIKELY(file_length > IREE_HOST_SIZE_MAX)) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "profile file length exceeds host size");
    } else {
      out_profile_file->file_length = (iree_host_size_t)file_length;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_profile_file_close(out_profile_file);
  }
  return status;
}

void iree_profile_file_close(iree_profile_file_t* profile_file) {
  iree_io_file_contents_free(profile_file->contents);
  memset(profile_file, 0, sizeof(*profile_file));
}

iree_status_t iree_profile_file_for_each_record(
    const iree_profile_file_t* profile_file,
    iree_profile_file_record_callback_t callback) {
  if (!callback.fn) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "profile file record callback must have a function");
  }

  iree_host_size_t file_length = profile_file->file_length;
  if (file_length == 0) {
    file_length = profile_file->contents->const_buffer.data_length;
  }
  iree_host_size_t record_offset = profile_file->first_record_offset;
  iree_host_size_t record_index = 0;
  iree_const_byte_span_t file_contents = iree_make_const_byte_span(
      profile_file->contents->const_buffer.data, file_length);
  while (record_offset < file_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_profile_file_parse_record(
        file_contents, record_offset, &record, &next_record_offset));
    IREE_RETURN_IF_ERROR(
        callback.fn(callback.user_data, &record, record_index));
    record_offset = next_record_offset;
    ++record_index;
  }
  return iree_ok_status();
}
