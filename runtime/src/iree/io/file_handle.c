// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/file_handle.h"

#include "iree/base/internal/atomics.h"
#include "iree/io/memory_stream.h"

//===----------------------------------------------------------------------===//
// iree_io_file_handle_t
//===----------------------------------------------------------------------===//

typedef struct iree_io_file_handle_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;
  iree_io_file_access_t access;
  iree_io_file_handle_primitive_t primitive;
  iree_io_file_handle_release_callback_t release_callback;
} iree_io_file_handle_t;

IREE_API_EXPORT iree_status_t iree_io_file_handle_wrap(
    iree_io_file_access_t allowed_access,
    iree_io_file_handle_primitive_t handle_primitive,
    iree_io_file_handle_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle) {
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_file_handle_t* handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*handle), (void**)&handle));
  iree_atomic_ref_count_init(&handle->ref_count);
  handle->host_allocator = host_allocator;
  handle->access = allowed_access;
  handle->primitive = handle_primitive;
  handle->release_callback = release_callback;

  *out_handle = handle;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_file_handle_wrap_host_allocation(
    iree_io_file_access_t allowed_access, iree_byte_span_t host_allocation,
    iree_io_file_handle_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle) {
  iree_io_file_handle_primitive_t handle_primitive = {
      .type = IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION,
      .value =
          {
              .host_allocation = host_allocation,
          },
  };
  return iree_io_file_handle_wrap(allowed_access, handle_primitive,
                                  release_callback, host_allocator, out_handle);
}

static void iree_io_file_handle_destroy(iree_io_file_handle_t* handle) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = handle->host_allocator;

  if (handle->release_callback.fn) {
    handle->release_callback.fn(handle->release_callback.user_data,
                                handle->primitive);
  }

  iree_allocator_free(host_allocator, handle);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_io_file_handle_retain(iree_io_file_handle_t* handle) {
  if (IREE_LIKELY(handle)) {
    iree_atomic_ref_count_inc(&handle->ref_count);
  }
}

IREE_API_EXPORT void iree_io_file_handle_release(
    iree_io_file_handle_t* handle) {
  if (IREE_LIKELY(handle) &&
      iree_atomic_ref_count_dec(&handle->ref_count) == 1) {
    iree_io_file_handle_destroy(handle);
  }
}

IREE_API_EXPORT iree_io_file_access_t
iree_io_file_handle_access(const iree_io_file_handle_t* handle) {
  IREE_ASSERT_ARGUMENT(handle);
  return handle->access;
}

IREE_API_EXPORT iree_io_file_handle_primitive_t
iree_io_file_handle_primitive(const iree_io_file_handle_t* handle) {
  IREE_ASSERT_ARGUMENT(handle);
  return handle->primitive;
}

IREE_API_EXPORT iree_status_t
iree_io_file_handle_flush(iree_io_file_handle_t* handle) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  switch (handle->primitive.type) {
    case IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION: {
      // No-op (though we could flush when known mapped).
      break;
    }
    default: {
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "flush not supported on handle type %d",
                                (int)handle->primitive.type);
      break;
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_io_stream_t utilities
//===----------------------------------------------------------------------===//

static void iree_io_memory_stream_file_release(void* user_data,
                                               iree_io_stream_t* stream) {
  iree_io_file_handle_t* file_handle = (iree_io_file_handle_t*)user_data;
  iree_io_file_handle_release(file_handle);
}

IREE_API_EXPORT iree_status_t iree_io_stream_open(
    iree_io_stream_mode_t mode, iree_io_file_handle_t* file_handle,
    uint64_t file_offset, iree_allocator_t host_allocator,
    iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(file_handle);
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  iree_io_stream_t* stream = NULL;
  iree_io_file_handle_primitive_t file_primitive =
      iree_io_file_handle_primitive(file_handle);
  switch (file_primitive.type) {
    case IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION: {
      if (file_offset > file_primitive.value.host_allocation.data_length) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "file offset %" PRIu64
            " out of range of host allocation with %" PRIhsz " bytes available",
            file_offset, file_primitive.value.host_allocation.data_length);
        break;
      }
      iree_io_memory_stream_release_callback_t release_callback = {
          .fn = iree_io_memory_stream_file_release,
          .user_data = file_handle,
      };
      iree_io_file_handle_retain(file_handle);
      status = iree_io_memory_stream_wrap(
          mode,
          iree_make_byte_span(
              file_primitive.value.host_allocation.data + file_offset,
              file_primitive.value.host_allocation.data_length - file_offset),
          release_callback, host_allocator, &stream);
      if (!iree_status_is_ok(status)) iree_io_file_handle_release(file_handle);
      break;
    }
    default: {
      status =
          iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                           "stream open not yet supported on handle type %d",
                           (int)file_primitive.type);
      break;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_io_stream_release(stream);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_stream_write_file(
    iree_io_stream_t* stream, iree_io_file_handle_t* source_file_handle,
    uint64_t source_file_offset, iree_io_stream_pos_t length,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(source_file_handle);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  iree_io_stream_t* source_stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_io_stream_open(IREE_IO_STREAM_MODE_READABLE, source_file_handle,
                          source_file_offset, host_allocator, &source_stream));

  iree_status_t status = iree_io_stream_copy(source_stream, stream, length);

  iree_io_stream_release(source_stream);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
