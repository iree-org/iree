// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/stream.h"

// This is arbitrary - we should allow dynamic block sizes and such but keeping
// this small only requires a reasonable fixed size stack alloc.
#define IREE_IO_STREAM_COPY_BLOCK_SIZE (32 * 1024)

//===----------------------------------------------------------------------===//
// Stream utilities
//===----------------------------------------------------------------------===//

static const char* iree_io_stream_seek_mode_string(
    iree_io_stream_seek_mode_t seek_mode) {
  switch (seek_mode) {
    case IREE_IO_STREAM_SEEK_SET:
      return "set";
    case IREE_IO_STREAM_SEEK_FROM_CURRENT:
      return "from-current";
    case IREE_IO_STREAM_SEEK_FROM_END:
      return "from-end";
    default:
      return "?";
  }
}

static iree_string_view_t iree_io_stream_mode_format(
    iree_io_stream_mode_t value, iree_bitfield_string_temp_t* out_temp) {
  static const iree_bitfield_string_mapping_t mappings[] = {
      {IREE_IO_STREAM_MODE_READABLE, IREE_SVL("READABLE")},
      {IREE_IO_STREAM_MODE_WRITABLE, IREE_SVL("WRITABLE")},
      {IREE_IO_STREAM_MODE_RESIZABLE, IREE_SVL("RESIZABLE")},
      {IREE_IO_STREAM_MODE_SEEKABLE, IREE_SVL("SEEKABLE")},
      {IREE_IO_STREAM_MODE_MAPPABLE, IREE_SVL("MAPPABLE")},
  };
  return iree_bitfield_format_inline(value, IREE_ARRAYSIZE(mappings), mappings,
                                     out_temp);
}

// Validates that all bits in |required_mode| are available in |allowed_mode|.
static iree_status_t iree_io_stream_validate_mode(
    iree_io_stream_mode_t allowed_mode, iree_io_stream_mode_t required_mode) {
  if (!iree_all_bits_set(allowed_mode, required_mode)) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t required_mode_str =
        iree_io_stream_mode_format(required_mode, &temp0);
    iree_string_view_t allowed_mode_str =
        iree_io_stream_mode_format(allowed_mode, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "operation requires mode '%.*s' but stream was opened with '%.*s'",
        (int)required_mode_str.size, required_mode_str.data,
        (int)allowed_mode_str.size, allowed_mode_str.data);
#else
    return iree_status_from_code(IREE_STATUS_PERMISSION_DENIED);
#endif  // IREE_STATUS_MODE
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_io_stream_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_io_stream_retain(iree_io_stream_t* stream) {
  if (IREE_LIKELY(stream)) {
    iree_atomic_ref_count_inc(&stream->ref_count);
  }
}

IREE_API_EXPORT void iree_io_stream_release(iree_io_stream_t* stream) {
  if (IREE_LIKELY(stream) &&
      iree_atomic_ref_count_dec(&stream->ref_count) == 1) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_io_stream_destroy");
    stream->vtable->destroy(stream);
    IREE_TRACE_ZONE_END(z0);
  }
}

IREE_API_EXPORT iree_io_stream_mode_t
iree_io_stream_mode(const iree_io_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  return stream->mode;
}

IREE_API_EXPORT iree_io_stream_pos_t
iree_io_stream_offset(iree_io_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  return stream->vtable->offset(stream);
}

IREE_API_EXPORT iree_io_stream_pos_t
iree_io_stream_length(iree_io_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  return stream->vtable->length(stream);
}

IREE_API_EXPORT bool iree_io_stream_is_eos(iree_io_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  return stream->vtable->offset(stream) == stream->vtable->length(stream);
}

IREE_API_EXPORT iree_status_t iree_io_stream_seek(
    iree_io_stream_t* stream, iree_io_stream_seek_mode_t seek_mode,
    iree_io_stream_pos_t offset) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, iree_io_stream_seek_mode_string(seek_mode));
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)offset);
  iree_status_t status = stream->vtable->seek(stream, seek_mode, offset);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_stream_seek_to_alignment(
    iree_io_stream_t* stream, iree_io_stream_pos_t alignment) {
  IREE_ASSERT_ARGUMENT(stream);
  if (alignment == 0) return iree_ok_status();
  if (!iree_is_power_of_two_uint64(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "alignment %" PRIu64 " not a power of two",
                            alignment);
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)alignment);
  iree_io_stream_pos_t current_offset = iree_io_stream_offset(stream);
  iree_io_stream_pos_t aligned_offset =
      iree_align_uint64(current_offset, alignment);
  iree_status_t status = iree_ok_status();
  if (current_offset != aligned_offset) {
    status =
        iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, aligned_offset);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_io_stream_read(iree_io_stream_t* stream, iree_host_size_t buffer_capacity,
                    void* buffer, iree_host_size_t* out_buffer_length) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(!buffer_capacity || buffer);
  if (out_buffer_length) *out_buffer_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_io_stream_validate_mode(iree_io_stream_mode(stream),
                                   IREE_IO_STREAM_MODE_READABLE),
      "reading from the stream");
  if (buffer_capacity == 0) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)buffer_capacity);
  iree_status_t status =
      stream->vtable->read(stream, buffer_capacity, buffer, out_buffer_length);
  if (out_buffer_length) {
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)*out_buffer_length);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_io_stream_write(iree_io_stream_t* stream, iree_host_size_t buffer_length,
                     const void* buffer) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(!buffer_length || buffer);
  IREE_RETURN_IF_ERROR(
      iree_io_stream_validate_mode(iree_io_stream_mode(stream),
                                   IREE_IO_STREAM_MODE_WRITABLE),
      "writing to the stream");
  if (!buffer_length) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)buffer_length);
  iree_status_t status = stream->vtable->write(stream, buffer_length, buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_io_stream_write_char(iree_io_stream_t* stream, char c) {
  return iree_io_stream_write(stream, sizeof(c), &c);
}

IREE_API_EXPORT iree_status_t iree_io_stream_write_string(
    iree_io_stream_t* stream, iree_string_view_t value) {
  if (!value.size) return iree_ok_status();
  return iree_io_stream_write(stream, value.size, value.data);
}

IREE_API_EXPORT iree_status_t
iree_io_stream_fill(iree_io_stream_t* stream, iree_io_stream_pos_t count,
                    const void* pattern, iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(!pattern_length || pattern);
  IREE_RETURN_IF_ERROR(
      iree_io_stream_validate_mode(iree_io_stream_mode(stream),
                                   IREE_IO_STREAM_MODE_WRITABLE),
      "writing to the stream");
  switch (pattern_length) {
    case 1:
    case 2:
    case 4:
    case 8:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported fill pattern length: %" PRIhsz,
                              pattern_length);
  }
  if (!count || !pattern_length) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)count);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)pattern_length);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)(count * pattern_length));
  iree_status_t status =
      stream->vtable->fill(stream, count, pattern, pattern_length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_io_stream_map_read(iree_io_stream_t* stream, iree_host_size_t length,
                        iree_const_byte_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_span);
  *out_span = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_io_stream_validate_mode(
          iree_io_stream_mode(stream),
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_MAPPABLE),
      "mapping the stream for reading");
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);
  iree_status_t status = stream->vtable->map_read(stream, length, out_span);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_io_stream_map_write(iree_io_stream_t* stream, iree_host_size_t length,
                         iree_byte_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_span);
  *out_span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_io_stream_validate_mode(
          iree_io_stream_mode(stream),
          IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_MAPPABLE),
      "mapping the stream for writing");
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);
  iree_status_t status = stream->vtable->map_write(stream, length, out_span);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_stream_copy(
    iree_io_stream_t* source_stream, iree_io_stream_t* target_stream,
    iree_io_stream_pos_t length) {
  IREE_ASSERT_ARGUMENT(source_stream);
  IREE_ASSERT_ARGUMENT(target_stream);
  if (!length) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  iree_status_t status = iree_ok_status();

  uint8_t buffer[IREE_IO_STREAM_COPY_BLOCK_SIZE];
  iree_io_stream_pos_t remaining_length = length;
  while (remaining_length) {
    iree_host_size_t block_length = iree_min(sizeof(buffer), remaining_length);
    iree_host_size_t read_length = 0;
    status =
        iree_io_stream_read(source_stream, block_length, buffer, &read_length);
    if (!iree_status_is_ok(status)) break;
    if (read_length != block_length) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "source stream read underflow, expected to read a block of %" PRIhsz
          " but stream only provided %" PRIhsz " bytes",
          block_length, read_length);
      break;
    }
    status = iree_io_stream_write(target_stream, read_length, buffer);
    if (!iree_status_is_ok(status)) break;
    remaining_length -= read_length;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
