// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/profile_file.h"

#include <stddef.h>
#include <string.h>

#include "iree/io/stream.h"

#if !defined(IREE_ENDIANNESS_LITTLE) || !IREE_ENDIANNESS_LITTLE
#error "IREE HAL profile bundle serialization requires little-endian hosts"
#endif  // !IREE_ENDIANNESS_LITTLE

typedef struct iree_hal_profile_file_sink_t {
  // HAL resource header for the sink.
  iree_hal_resource_t resource;
  // Host allocator used for sink lifetime.
  iree_allocator_t host_allocator;
  // Writable stream receiving the profile bundle.
  iree_io_stream_t* stream;
} iree_hal_profile_file_sink_t;

static const iree_hal_profile_sink_vtable_t iree_hal_profile_file_sink_vtable;

static iree_hal_profile_file_sink_t* iree_hal_profile_file_sink_cast(
    iree_hal_profile_sink_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_profile_file_sink_vtable);
  return (iree_hal_profile_file_sink_t*)base_value;
}

static iree_status_t iree_hal_profile_file_sink_write_file_header(
    iree_io_stream_t* stream) {
  iree_hal_profile_file_header_t header = {
      .magic = IREE_HAL_PROFILE_FILE_MAGIC,
      .version_major = IREE_HAL_PROFILE_FILE_VERSION_MAJOR,
      .version_minor = IREE_HAL_PROFILE_FILE_VERSION_MINOR,
      .header_length = sizeof(header),
      .flags = 0,
  };
  return iree_io_stream_write(stream, sizeof(header), &header);
}

static iree_status_t iree_hal_profile_file_calculate_record_layout(
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs,
    iree_host_size_t* out_record_length, iree_host_size_t* out_payload_length) {
  if (IREE_UNLIKELY(metadata->content_type.size > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile chunk content type is too long");
  }
  if (IREE_UNLIKELY(metadata->name.size > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile chunk name is too long");
  }
  if (IREE_UNLIKELY(iovec_count > 0 && !iovecs)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profile chunk iovec list is required");
  }

  iree_host_size_t payload_length = 0;
  bool payload_length_valid = true;
  for (iree_host_size_t i = 0; i < iovec_count; ++i) {
    payload_length_valid = iree_host_size_checked_add(
        payload_length, iovecs[i].data_length, &payload_length);
    if (!payload_length_valid) break;
  }
  if (IREE_UNLIKELY(!payload_length_valid)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile chunk payload length overflow");
  }

  iree_host_size_t record_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_profile_file_record_header_t), &record_length,
      IREE_STRUCT_FIELD(metadata->content_type.size, uint8_t, NULL),
      IREE_STRUCT_FIELD(metadata->name.size, uint8_t, NULL),
      IREE_STRUCT_FIELD(payload_length, uint8_t, NULL)));

  *out_record_length = record_length;
  *out_payload_length = payload_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_file_sink_write_record(
    iree_hal_profile_file_sink_t* sink,
    iree_hal_profile_file_record_type_t record_type,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs) {
  iree_host_size_t record_length = 0;
  iree_host_size_t payload_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_profile_file_calculate_record_layout(
      metadata, iovec_count, iovecs, &record_length, &payload_length));

  iree_hal_profile_chunk_flags_t chunk_flags = 0;
  uint64_t dropped_record_count = 0;
  if (record_type == IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    chunk_flags = metadata->flags;
    dropped_record_count = metadata->dropped_record_count;
    if (dropped_record_count != 0) {
      chunk_flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
    }
  }

  iree_hal_profile_file_record_header_t header = {
      .record_length = record_length,
      .payload_length = payload_length,
      .session_id = metadata->session_id,
      .stream_id = metadata->stream_id,
      .event_id = metadata->event_id,
      .executable_id = metadata->executable_id,
      .command_buffer_id = metadata->command_buffer_id,
      .header_length = sizeof(header),
      .content_type_length = (uint32_t)metadata->content_type.size,
      .name_length = (uint32_t)metadata->name.size,
      .physical_device_ordinal = metadata->physical_device_ordinal,
      .queue_ordinal = metadata->queue_ordinal,
      .chunk_flags = chunk_flags,
      .dropped_record_count = dropped_record_count,
      .session_status_code = (uint32_t)session_status_code,
      .record_type = record_type,
      .flags = 0,
  };

  IREE_RETURN_IF_ERROR(
      iree_io_stream_write(sink->stream, sizeof(header), &header));
  if (metadata->content_type.size > 0) {
    IREE_RETURN_IF_ERROR(iree_io_stream_write(sink->stream,
                                              metadata->content_type.size,
                                              metadata->content_type.data));
  }
  if (metadata->name.size > 0) {
    IREE_RETURN_IF_ERROR(iree_io_stream_write(sink->stream, metadata->name.size,
                                              metadata->name.data));
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < iovec_count;
       ++i) {
    if (iovecs[i].data_length > 0) {
      status = iree_io_stream_write(sink->stream, iovecs[i].data_length,
                                    iovecs[i].data);
    }
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_profile_file_sink_create(
    iree_io_file_handle_t* file_handle, iree_allocator_t host_allocator,
    iree_hal_profile_sink_t** out_sink) {
  IREE_ASSERT_ARGUMENT(file_handle);
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_open(IREE_IO_STREAM_MODE_WRITABLE, file_handle,
                              /*file_offset=*/0, host_allocator, &stream));

  iree_status_t status = iree_hal_profile_file_sink_write_file_header(stream);

  iree_hal_profile_file_sink_t* sink = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, sizeof(*sink), (void**)&sink);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_profile_file_sink_vtable,
                                 &sink->resource);
    sink->host_allocator = host_allocator;
    sink->stream = stream;
    stream = NULL;
    *out_sink = (iree_hal_profile_sink_t*)sink;
  }

  iree_io_stream_release(stream);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_profile_file_parse_header(iree_const_byte_span_t file_contents,
                                   iree_hal_profile_file_header_t* out_header,
                                   iree_host_size_t* out_record_offset) {
  IREE_ASSERT_ARGUMENT(out_header);
  IREE_ASSERT_ARGUMENT(out_record_offset);
  memset(out_header, 0, sizeof(*out_header));
  *out_record_offset = 0;

  if (IREE_UNLIKELY(file_contents.data_length > 0 && !file_contents.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profile file contents data is required");
  }
  if (IREE_UNLIKELY(file_contents.data_length <
                    sizeof(iree_hal_profile_file_header_t))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile file is too small for header");
  }

  iree_hal_profile_file_header_t header;
  memcpy(&header, file_contents.data, sizeof(header));
  if (IREE_UNLIKELY(header.magic != IREE_HAL_PROFILE_FILE_MAGIC)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid IREE HAL profile file magic");
  }
  if (IREE_UNLIKELY(header.version_major !=
                    IREE_HAL_PROFILE_FILE_VERSION_MAJOR)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported IREE HAL profile file version");
  }
  if (IREE_UNLIKELY(header.header_length < sizeof(header))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile file header length is too small");
  }
  if (IREE_UNLIKELY(header.header_length > file_contents.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile file header extends past file end");
  }

  *out_header = header;
  *out_record_offset = header.header_length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_profile_file_parse_record(
    iree_const_byte_span_t file_contents, iree_host_size_t record_offset,
    iree_hal_profile_file_record_t* out_record,
    iree_host_size_t* out_next_record_offset) {
  IREE_ASSERT_ARGUMENT(out_record);
  IREE_ASSERT_ARGUMENT(out_next_record_offset);
  memset(out_record, 0, sizeof(*out_record));
  *out_next_record_offset = record_offset;

  if (IREE_UNLIKELY(file_contents.data_length > 0 && !file_contents.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profile file contents data is required");
  }
  if (IREE_UNLIKELY(record_offset >= file_contents.data_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile record offset is outside the file");
  }

  iree_host_size_t remaining_file_length =
      file_contents.data_length - record_offset;
  if (IREE_UNLIKELY(remaining_file_length <
                    sizeof(iree_hal_profile_file_record_header_t))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile record is too small for header");
  }

  const uint8_t* record_base = file_contents.data + record_offset;
  iree_hal_profile_file_record_header_t header;
  memcpy(&header, record_base, sizeof(header));
  if (IREE_UNLIKELY(header.record_length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile record length exceeds host size");
  }
  iree_host_size_t record_length = (iree_host_size_t)header.record_length;
  if (IREE_UNLIKELY(record_length > remaining_file_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile record extends past file end");
  }
  if (IREE_UNLIKELY(header.header_length < sizeof(header) ||
                    header.header_length > record_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile record header length is invalid");
  }
  if (IREE_UNLIKELY(header.payload_length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile record payload length exceeds host size");
  }

  iree_host_size_t content_type_length =
      (iree_host_size_t)header.content_type_length;
  iree_host_size_t name_length = (iree_host_size_t)header.name_length;
  iree_host_size_t payload_length = (iree_host_size_t)header.payload_length;
  iree_host_size_t string_length = 0;
  iree_host_size_t expected_data_length = 0;
  bool data_length_valid =
      iree_host_size_checked_add(content_type_length, name_length,
                                 &string_length) &&
      iree_host_size_checked_add(string_length, payload_length,
                                 &expected_data_length);
  if (IREE_UNLIKELY(!data_length_valid ||
                    expected_data_length !=
                        record_length - header.header_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile record data lengths do not match");
  }

  const char* content_type = (const char*)(record_base + header.header_length);
  const char* name = content_type + content_type_length;
  const uint8_t* payload = (const uint8_t*)name + name_length;
  out_record->header = header;
  out_record->content_type =
      iree_make_string_view(content_type, content_type_length);
  out_record->name = iree_make_string_view(name, name_length);
  out_record->payload = iree_make_const_byte_span(payload, payload_length);
  *out_next_record_offset = record_offset + record_length;
  return iree_ok_status();
}

static void iree_hal_profile_file_sink_destroy(
    iree_hal_profile_sink_t* base_sink) {
  iree_hal_profile_file_sink_t* sink =
      iree_hal_profile_file_sink_cast(base_sink);
  iree_allocator_t host_allocator = sink->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stream_release(sink->stream);
  iree_allocator_free(host_allocator, sink);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_profile_file_sink_begin_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  iree_hal_profile_file_sink_t* sink =
      iree_hal_profile_file_sink_cast(base_sink);
  return iree_hal_profile_file_sink_write_record(
      sink, IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN, metadata,
      IREE_STATUS_OK, /*iovec_count=*/0, /*iovecs=*/NULL);
}

static iree_status_t iree_hal_profile_file_sink_write(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  iree_hal_profile_file_sink_t* sink =
      iree_hal_profile_file_sink_cast(base_sink);
  return iree_hal_profile_file_sink_write_record(
      sink, IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK, metadata, IREE_STATUS_OK,
      iovec_count, iovecs);
}

static iree_status_t iree_hal_profile_file_sink_end_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  iree_hal_profile_file_sink_t* sink =
      iree_hal_profile_file_sink_cast(base_sink);
  return iree_hal_profile_file_sink_write_record(
      sink, IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END, metadata,
      session_status_code, /*iovec_count=*/0, /*iovecs=*/NULL);
}

static const iree_hal_profile_sink_vtable_t iree_hal_profile_file_sink_vtable =
    {
        .destroy = iree_hal_profile_file_sink_destroy,
        .begin_session = iree_hal_profile_file_sink_begin_session,
        .write = iree_hal_profile_file_sink_write,
        .end_session = iree_hal_profile_file_sink_end_session,
};
