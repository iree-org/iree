// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/file_writer.h"

#include <stddef.h>
#include <string.h>

#include "iree/io/stream.h"

#if !defined(IREE_ENDIANNESS_LITTLE) || !IREE_ENDIANNESS_LITTLE
#error "IREE HAL replay file serialization requires little-endian hosts"
#endif  // !IREE_ENDIANNESS_LITTLE

#define IREE_HAL_REPLAY_FNV1A64_OFFSET_BASIS 0xcbf29ce484222325ull
#define IREE_HAL_REPLAY_FNV1A64_PRIME 0x100000001b3ull

struct iree_hal_replay_file_writer_t {
  // Host allocator used for writer lifetime.
  iree_allocator_t host_allocator;
  // Writable stream receiving the replay file.
  iree_io_stream_t* stream;
  // Digest type emitted for payload ranges.
  iree_hal_replay_digest_type_t payload_digest_type;
  // True once the final file length has been written.
  bool closed;
};

static void iree_hal_replay_digest_clear(iree_hal_replay_file_range_t* range) {
  memset(range->digest, 0, sizeof(range->digest));
}

static uint64_t iree_hal_replay_digest_fnv1a64_update(
    uint64_t state, iree_const_byte_span_t bytes) {
  for (iree_host_size_t i = 0; i < bytes.data_length; ++i) {
    state ^= bytes.data[i];
    state *= IREE_HAL_REPLAY_FNV1A64_PRIME;
  }
  return state;
}

static uint64_t iree_hal_replay_digest_fnv1a64_iovecs(
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  uint64_t state = IREE_HAL_REPLAY_FNV1A64_OFFSET_BASIS;
  for (iree_host_size_t i = 0; i < iovec_count; ++i) {
    state = iree_hal_replay_digest_fnv1a64_update(state, iovecs[i]);
  }
  return state;
}

static void iree_hal_replay_digest_store_fnv1a64(
    uint64_t digest, iree_hal_replay_file_range_t* out_range) {
  memcpy(out_range->digest, &digest, sizeof(digest));
}

static iree_status_t iree_hal_replay_file_write_file_header(
    iree_io_stream_t* stream, uint64_t file_length) {
  iree_hal_replay_file_header_t header = {
      .magic = IREE_HAL_REPLAY_FILE_MAGIC,
      .version_major = IREE_HAL_REPLAY_FILE_VERSION_MAJOR,
      .version_minor = IREE_HAL_REPLAY_FILE_VERSION_MINOR,
      .header_length = sizeof(header),
      .flags = 0,
      .file_length = file_length,
  };
  return iree_io_stream_write(stream, sizeof(header), &header);
}

static iree_status_t iree_hal_replay_file_record_metadata_validate(
    const iree_hal_replay_file_record_metadata_t* metadata) {
  if (IREE_UNLIKELY(!metadata)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay record metadata is required");
  }
  const iree_hal_replay_file_record_flags_t valid_flags =
      IREE_HAL_REPLAY_FILE_RECORD_FLAG_OPTIONAL;
  if (IREE_UNLIKELY((metadata->record_flags & ~valid_flags) != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay record reserved flags must be zero");
  }
  if (IREE_UNLIKELY(
          !iree_hal_replay_file_record_type_is_known(metadata->record_type))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot write unknown or reserved replay record "
                            "type");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_file_calculate_payload_length(
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs,
    iree_host_size_t* out_payload_length) {
  if (IREE_UNLIKELY(iovec_count > 0 && !iovecs)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay record iovec list is required");
  }

  iree_host_size_t payload_length = 0;
  bool payload_length_valid = true;
  for (iree_host_size_t i = 0; i < iovec_count; ++i) {
    if (IREE_UNLIKELY(iovecs[i].data_length > 0 && !iovecs[i].data)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "replay record iovec data is required");
    }
    payload_length_valid = iree_host_size_checked_add(
        payload_length, iovecs[i].data_length, &payload_length);
    if (!payload_length_valid) break;
  }
  if (IREE_UNLIKELY(!payload_length_valid)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay record payload length overflow");
  }

  *out_payload_length = payload_length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_replay_file_writer_create(
    iree_io_file_handle_t* file_handle, iree_allocator_t host_allocator,
    iree_hal_replay_file_writer_t** out_writer) {
  IREE_ASSERT_ARGUMENT(file_handle);
  IREE_ASSERT_ARGUMENT(out_writer);
  *out_writer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_open(IREE_IO_STREAM_MODE_WRITABLE, file_handle,
                              /*file_offset=*/0, host_allocator, &stream));

  iree_status_t status =
      iree_hal_replay_file_write_file_header(stream, /*file_length=*/0);

  iree_hal_replay_file_writer_t* writer = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, sizeof(*writer), (void**)&writer);
  }

  if (iree_status_is_ok(status)) {
    writer->host_allocator = host_allocator;
    writer->stream = stream;
    writer->payload_digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_FNV1A_64;
    writer->closed = false;
    stream = NULL;
    *out_writer = writer;
  }

  iree_io_stream_release(stream);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_replay_file_writer_close(iree_hal_replay_file_writer_t* writer) {
  IREE_ASSERT_ARGUMENT(writer);
  if (writer->closed) return iree_ok_status();

  iree_io_stream_pos_t file_length = iree_io_stream_offset(writer->stream);
  if (IREE_UNLIKELY(file_length < 0)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay writer stream offset is invalid");
  }

  IREE_RETURN_IF_ERROR(
      iree_io_stream_seek(writer->stream, IREE_IO_STREAM_SEEK_SET, 0));
  iree_status_t status = iree_hal_replay_file_write_file_header(
      writer->stream, (uint64_t)file_length);
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_seek(writer->stream, IREE_IO_STREAM_SEEK_SET,
                                 file_length);
  }
  if (iree_status_is_ok(status)) writer->closed = true;
  return status;
}

IREE_API_EXPORT void iree_hal_replay_file_writer_free(
    iree_hal_replay_file_writer_t* writer) {
  if (!writer) return;
  iree_allocator_t host_allocator = writer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stream_release(writer->stream);
  iree_allocator_free(host_allocator, writer);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_replay_file_writer_append_record(
    iree_hal_replay_file_writer_t* writer,
    const iree_hal_replay_file_record_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs,
    iree_hal_replay_file_range_t* out_payload_range) {
  IREE_ASSERT_ARGUMENT(writer);
  if (out_payload_range) {
    *out_payload_range = iree_hal_replay_file_range_empty();
  }
  if (IREE_UNLIKELY(writer->closed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot append to a closed replay file writer");
  }
  IREE_RETURN_IF_ERROR(iree_hal_replay_file_record_metadata_validate(metadata));

  iree_host_size_t payload_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_replay_file_calculate_payload_length(
      iovec_count, iovecs, &payload_length));

  iree_host_size_t record_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          sizeof(iree_hal_replay_file_record_header_t), payload_length,
          &record_length))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay record length overflow");
  }

  iree_io_stream_pos_t record_offset = iree_io_stream_offset(writer->stream);
  if (IREE_UNLIKELY(record_offset < 0)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay writer stream offset is invalid");
  }
  const uint64_t payload_offset =
      (uint64_t)record_offset +
      (uint64_t)sizeof(iree_hal_replay_file_record_header_t);

  iree_hal_replay_file_record_header_t header = {
      .record_length = record_length,
      .payload_length = payload_length,
      .sequence_ordinal = metadata->sequence_ordinal,
      .thread_id = metadata->thread_id,
      .device_id = metadata->device_id,
      .object_id = metadata->object_id,
      .related_object_id = metadata->related_object_id,
      .header_length = sizeof(header),
      .record_type = metadata->record_type,
      .record_flags = metadata->record_flags,
      .payload_type = metadata->payload_type,
      .object_type = metadata->object_type,
      .operation_code = metadata->operation_code,
      .status_code = metadata->status_code,
  };

  IREE_RETURN_IF_ERROR(
      iree_io_stream_write(writer->stream, sizeof(header), &header));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < iovec_count;
       ++i) {
    if (iovecs[i].data_length > 0) {
      status = iree_io_stream_write(writer->stream, iovecs[i].data_length,
                                    iovecs[i].data);
    }
  }

  if (iree_status_is_ok(status) && out_payload_range) {
    out_payload_range->offset = payload_offset;
    out_payload_range->length = payload_length;
    out_payload_range->uncompressed_length = payload_length;
    out_payload_range->compression_type = IREE_HAL_REPLAY_COMPRESSION_TYPE_NONE;
    out_payload_range->digest_type = writer->payload_digest_type;
    iree_hal_replay_digest_clear(out_payload_range);
    if (writer->payload_digest_type == IREE_HAL_REPLAY_DIGEST_TYPE_FNV1A_64) {
      iree_hal_replay_digest_store_fnv1a64(
          iree_hal_replay_digest_fnv1a64_iovecs(iovec_count, iovecs),
          out_payload_range);
    }
  }
  return status;
}
