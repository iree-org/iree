// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/file_handle.h"

#include "iree/base/internal/atomics.h"

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
