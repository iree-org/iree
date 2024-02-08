// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/memory_stream.h"

//===----------------------------------------------------------------------===//
// Utilities
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

// Validates that at least |access_length| bytes are available at the current
// stream offset. If the optional |out_available_length| is provided the
// available length will be returned matching the requested |access_length| or
// the maximum remaining length to the caller with an OK status even if the full
// |access_length| is not available and otherwise an error is returned.
static iree_status_t iree_io_stream_validate_fixed_range(
    iree_io_stream_pos_t stream_offset, iree_io_stream_pos_t stream_length,
    iree_io_stream_pos_t access_length,
    iree_io_stream_pos_t* out_available_length) {
  if (out_available_length) *out_available_length = 0;

  iree_io_stream_pos_t remaining_length = stream_length - stream_offset;
  if (access_length > remaining_length) {
    // Access exceeds remaining length.
    if (out_available_length) {
      // Let caller know how much is available and return OK so they can use it.
      *out_available_length = remaining_length;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "access to range [%" PRIu64 ", %" PRIu64
                            ") (%" PRIu64
                            " bytes) out of range; stream offset %" PRIu64
                            " and length %" PRIu64 " insufficient",
                            stream_offset, stream_offset + access_length,
                            access_length, stream_offset, stream_length);
  }

  if (out_available_length) *out_available_length = access_length;
  return iree_ok_status();
}

// Applies a seek operation with |seek_mode| and |seek_offset| against a stream.
// |out_stream_offset| will contain the position after the seek or an error will
// be returned if the seek could not be completed.
static iree_status_t iree_io_stream_apply_fixed_seek(
    iree_io_stream_pos_t stream_offset, iree_io_stream_pos_t stream_length,
    iree_io_stream_seek_mode_t seek_mode, iree_io_stream_pos_t seek_offset,
    iree_io_stream_pos_t* out_stream_offset) {
  *out_stream_offset = stream_offset;

  iree_io_stream_pos_t new_offset = stream_offset;
  switch (seek_mode) {
    case IREE_IO_STREAM_SEEK_SET:
      new_offset = seek_offset;
      break;
    case IREE_IO_STREAM_SEEK_FROM_CURRENT:
      new_offset = stream_offset + seek_offset;
      break;
    case IREE_IO_STREAM_SEEK_FROM_END:
      new_offset = stream_length + seek_offset;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unrecognized seek mode %u", (uint32_t)seek_mode);
  }

  if (new_offset < 0 || new_offset > stream_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "seek %s offset %" PRIi64
                            " out of stream bounds; expected 0 <= %" PRIi64
                            " < %" PRIi64,
                            iree_io_stream_seek_mode_string(seek_mode),
                            seek_offset, new_offset, stream_length);
  }

  *out_stream_offset = new_offset;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_io_memory_stream_t
//===----------------------------------------------------------------------===//

typedef struct iree_io_memory_stream_t {
  iree_io_stream_t base;
  iree_allocator_t host_allocator;
  iree_io_memory_stream_release_callback_t release_callback;
  iree_io_stream_pos_t offset;
  iree_io_stream_pos_t length;
  uint8_t* contents;
} iree_io_memory_stream_t;

static const iree_io_stream_vtable_t iree_io_memory_stream_vtable;

static iree_io_memory_stream_t* iree_io_memory_stream_cast(
    iree_io_stream_t* IREE_RESTRICT base_stream) {
  return (iree_io_memory_stream_t*)base_stream;
}

IREE_API_EXPORT iree_status_t iree_io_memory_stream_wrap(
    iree_io_stream_mode_t mode, iree_byte_span_t contents,
    iree_io_memory_stream_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)contents.data_length);

  iree_io_memory_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*stream), (void**)&stream));
  iree_atomic_ref_count_init(&stream->base.ref_count);
  stream->base.vtable = &iree_io_memory_stream_vtable;
  stream->base.mode = mode;
  stream->host_allocator = host_allocator;
  stream->release_callback = release_callback;
  stream->offset = 0;
  stream->length = contents.data_length;
  stream->contents = contents.data;

  *out_stream = &stream->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_io_memory_stream_destroy(
    iree_io_stream_t* IREE_RESTRICT base_stream) {
  iree_io_memory_stream_t* stream = iree_io_memory_stream_cast(base_stream);
  iree_allocator_t host_allocator = stream->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (stream->release_callback.fn) {
    stream->release_callback.fn(stream->release_callback.user_data,
                                base_stream);
  }

  iree_allocator_free(host_allocator, stream);

  IREE_TRACE_ZONE_END(z0);
}

static iree_io_stream_pos_t iree_io_memory_stream_offset(
    iree_io_stream_t* base_stream) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_memory_stream_t* stream = iree_io_memory_stream_cast(base_stream);
  return stream->offset;
}

static iree_io_stream_pos_t iree_io_memory_stream_length(
    iree_io_stream_t* base_stream) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_memory_stream_t* stream = iree_io_memory_stream_cast(base_stream);
  return stream->length;
}

static iree_status_t iree_io_memory_stream_seek(
    iree_io_stream_t* base_stream, iree_io_stream_seek_mode_t seek_mode,
    iree_io_stream_pos_t seek_offset) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_memory_stream_t* stream = iree_io_memory_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_io_stream_apply_fixed_seek(
      stream->offset, stream->length, seek_mode, seek_offset, &stream->offset);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_memory_stream_read(
    iree_io_stream_t* base_stream, iree_host_size_t buffer_capacity,
    void* buffer, iree_host_size_t* out_buffer_length) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(buffer);
  if (out_buffer_length) *out_buffer_length = 0;
  iree_io_memory_stream_t* stream = iree_io_memory_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stream_pos_t read_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_validate_fixed_range(stream->offset, stream->length,
                                              buffer_capacity, &read_length));
  if (!out_buffer_length && read_length != buffer_capacity) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "read of range [%" PRIu64 ", %" PRIu64 ") (%" PRIu64
                         " bytes) out of range; stream offset %" PRIu64
                         " and length %" PRIu64 " insufficient",
                         stream->offset, stream->offset + buffer_capacity,
                         (iree_io_stream_pos_t)buffer_capacity, stream->offset,
                         stream->length));
  }

  memcpy(buffer, stream->contents + stream->offset,
         (iree_host_size_t)read_length);
  stream->offset += read_length;

  if (out_buffer_length) *out_buffer_length = (iree_host_size_t)read_length;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_io_memory_stream_write(iree_io_stream_t* base_stream,
                                                 iree_host_size_t buffer_length,
                                                 const void* buffer) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(buffer);
  iree_io_memory_stream_t* stream = iree_io_memory_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_validate_fixed_range(stream->offset, stream->length,
                                              buffer_length, NULL));

  memcpy(stream->contents + stream->offset, buffer, buffer_length);
  stream->offset += buffer_length;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_io_memory_stream_fill(
    iree_io_stream_t* base_stream, iree_io_stream_pos_t count,
    const void* pattern, iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(pattern);
  iree_io_memory_stream_t* stream = iree_io_memory_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stream_pos_t access_length = count * pattern_length;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_validate_fixed_range(stream->offset, stream->length,
                                              access_length, NULL));

  iree_status_t status = iree_ok_status();
  uint8_t* data_ptr = stream->contents + stream->offset;
  switch (pattern_length) {
    case 1: {
      uint8_t* data = (uint8_t*)data_ptr;
      uint8_t value_bits = *(const uint8_t*)(pattern);
      memset(data, value_bits, count);
      break;
    }
    case 2: {
      uint16_t* data = (uint16_t*)data_ptr;
      uint16_t value_bits = *(const uint16_t*)(pattern);
      for (iree_device_size_t i = 0; i < count; ++i) {
        iree_unaligned_store(&data[i], value_bits);
      }
      break;
    }
    case 4: {
      uint32_t* data = (uint32_t*)data_ptr;
      uint32_t value_bits = *(const uint32_t*)(pattern);
      for (iree_device_size_t i = 0; i < count; ++i) {
        iree_unaligned_store(&data[i], value_bits);
      }
      break;
    }
    case 8: {
      uint64_t* data = (uint64_t*)data_ptr;
      uint64_t value_bits = *(const uint64_t*)(pattern);
      for (iree_device_size_t i = 0; i < count; ++i) {
        iree_unaligned_store(&data[i], value_bits);
      }
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("verified in iree_io_stream_fill");
      break;
  }
  if (iree_status_is_ok(status)) {
    stream->offset += access_length;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_memory_stream_map_read(
    iree_io_stream_t* base_stream, iree_host_size_t length,
    iree_const_byte_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(out_span);
  *out_span = iree_const_byte_span_empty();
  iree_io_memory_stream_t* stream = iree_io_memory_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_validate_fixed_range(stream->offset, stream->length,
                                              length, NULL));

  *out_span =
      iree_make_const_byte_span(stream->contents + stream->offset, length);
  stream->offset += length;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_io_memory_stream_map_write(
    iree_io_stream_t* base_stream, iree_host_size_t length,
    iree_byte_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(out_span);
  *out_span = iree_byte_span_empty();
  iree_io_memory_stream_t* stream = iree_io_memory_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_validate_fixed_range(stream->offset, stream->length,
                                              length, NULL));

  *out_span = iree_make_byte_span(stream->contents + stream->offset, length);
  stream->offset += length;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_io_stream_vtable_t iree_io_memory_stream_vtable = {
    .destroy = iree_io_memory_stream_destroy,
    .offset = iree_io_memory_stream_offset,
    .length = iree_io_memory_stream_length,
    .seek = iree_io_memory_stream_seek,
    .read = iree_io_memory_stream_read,
    .write = iree_io_memory_stream_write,
    .fill = iree_io_memory_stream_fill,
    .map_read = iree_io_memory_stream_map_read,
    .map_write = iree_io_memory_stream_map_write,
};
