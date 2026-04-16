// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/profile_file.h"

#include <stddef.h>

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
      .chunk_flags = metadata->flags,
      .session_status_code = (uint32_t)session_status_code,
      .record_type = record_type,
      .flags = 0,
      .reserved0 = 0,
      .reserved1 = 0,
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
