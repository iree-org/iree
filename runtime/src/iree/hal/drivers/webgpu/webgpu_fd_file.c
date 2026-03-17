// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_fd_file.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_fd_file_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_fd_file_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_memory_access_t access;
  iree_io_file_handle_t* handle;  // Retained.
  int fd;
  uint64_t length;
} iree_hal_webgpu_fd_file_t;

static const iree_hal_file_vtable_t iree_hal_webgpu_fd_file_vtable;

static iree_hal_webgpu_fd_file_t* iree_hal_webgpu_fd_file_cast(
    iree_hal_file_t* IREE_RESTRICT base_file) {
  return (iree_hal_webgpu_fd_file_t*)base_file;
}

iree_status_t iree_hal_webgpu_fd_file_from_handle(
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    uint64_t length, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;

  iree_io_file_handle_primitive_t primitive =
      iree_io_file_handle_primitive(handle);
  if (primitive.type != IREE_IO_FILE_HANDLE_TYPE_FD) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected FD file handle type, got %d",
                            (int)primitive.type);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_fd_file_t* file = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*file), (void**)&file));
  iree_hal_resource_initialize(&iree_hal_webgpu_fd_file_vtable,
                               &file->resource);
  file->host_allocator = host_allocator;
  file->access = access;
  file->handle = handle;
  iree_io_file_handle_retain(handle);
  file->fd = primitive.value.fd;
  file->length = length;

  *out_file = (iree_hal_file_t*)file;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

int iree_hal_webgpu_fd_file_fd(iree_hal_file_t* file) {
  return iree_hal_webgpu_fd_file_cast(file)->fd;
}

static void iree_hal_webgpu_fd_file_destroy(
    iree_hal_file_t* IREE_RESTRICT base_file) {
  iree_hal_webgpu_fd_file_t* file = iree_hal_webgpu_fd_file_cast(base_file);
  iree_allocator_t host_allocator = file->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_file_handle_release(file->handle);
  iree_allocator_free(host_allocator, file);

  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_memory_access_t iree_hal_webgpu_fd_file_allowed_access(
    iree_hal_file_t* base_file) {
  return iree_hal_webgpu_fd_file_cast(base_file)->access;
}

static uint64_t iree_hal_webgpu_fd_file_length(iree_hal_file_t* base_file) {
  return iree_hal_webgpu_fd_file_cast(base_file)->length;
}

static iree_hal_buffer_t* iree_hal_webgpu_fd_file_storage_buffer(
    iree_hal_file_t* base_file) {
  // FD files have no host pointer or device buffer. The queue_read/write
  // implementations use the FD-specific bridge imports instead.
  return NULL;
}

static bool iree_hal_webgpu_fd_file_supports_synchronous_io(
    iree_hal_file_t* base_file) {
  // No pread/pwrite on wasm.
  return false;
}

static iree_status_t iree_hal_webgpu_fd_file_read(
    iree_hal_file_t* base_file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "synchronous file read not supported on wasm; use "
                          "queue_read for async file-to-GPU transfer");
}

static iree_status_t iree_hal_webgpu_fd_file_write(
    iree_hal_file_t* base_file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "synchronous file write not supported on wasm; use "
                          "queue_write for async GPU-to-file transfer");
}

static const iree_hal_file_vtable_t iree_hal_webgpu_fd_file_vtable = {
    .destroy = iree_hal_webgpu_fd_file_destroy,
    .allowed_access = iree_hal_webgpu_fd_file_allowed_access,
    .length = iree_hal_webgpu_fd_file_length,
    .storage_buffer = iree_hal_webgpu_fd_file_storage_buffer,
    .supports_synchronous_io = iree_hal_webgpu_fd_file_supports_synchronous_io,
    .read = iree_hal_webgpu_fd_file_read,
    .write = iree_hal_webgpu_fd_file_write,
};
