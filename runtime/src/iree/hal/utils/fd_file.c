// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/fd_file.h"

//===----------------------------------------------------------------------===//
// Platform Support
//===----------------------------------------------------------------------===//

#if IREE_FILE_IO_ENABLE

#define _GNU_SOURCE

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(IREE_PLATFORM_WINDOWS)
#include <io.h>
#else
#include <unistd.h>
#endif  // IREE_PLATFORM_WINDOWS

#if defined(IREE_PLATFORM_WINDOWS)

// Returns the allowed access and length in bytes of the file descriptor.
// Returns 0 if the file descriptor has no length (a /proc stream, etc).
static iree_status_t iree_hal_platform_fd_stat(
    int fd, iree_hal_memory_access_t* out_allowed_access,
    uint64_t* out_length) {
  IREE_ASSERT_ARGUMENT(out_allowed_access);
  IREE_ASSERT_ARGUMENT(out_length);
  *out_allowed_access = IREE_HAL_MEMORY_ACCESS_NONE;
  *out_length = 0;

  struct _stat64 buffer = {0};
  if (_fstat64(fd, &buffer) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "unable to stat file descriptor length");
  }

  *out_allowed_access =
      ((buffer.st_mode & _S_IREAD) ? IREE_HAL_MEMORY_ACCESS_READ : 0) |
      ((buffer.st_mode & _S_IWRITE) ? IREE_HAL_MEMORY_ACCESS_WRITE : 0);
  *out_length = (uint64_t)buffer.st_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_platform_fd_pread(
    int fd, void* buffer, iree_host_size_t count, uint64_t offset,
    iree_host_size_t* out_bytes_read) {
  IREE_ASSERT_ARGUMENT(out_bytes_read);
  *out_bytes_read = 0;

  HANDLE handle = (HANDLE)_get_osfhandle(fd);
  if (handle == INVALID_HANDLE_VALUE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "file descriptor is not backed by a valid Win32 HANDLE");
  }

  DWORD bytes_read = 0;
  OVERLAPPED overlapped = {0};
  overlapped.Offset = (DWORD)(offset & 0xFFFFFFFFu);
  overlapped.OffsetHigh = (DWORD)((offset >> 32) & 0xFFFFFFFFu);
  if (!ReadFile(handle, buffer, (DWORD)count, &bytes_read, &overlapped)) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to read requested buffer length");
  }

  *out_bytes_read = (iree_host_size_t)bytes_read;
  return iree_ok_status();
}

static iree_status_t iree_hal_platform_fd_pwrite(
    int fd, const void* buffer, iree_host_size_t count, uint64_t offset,
    iree_host_size_t* out_bytes_written) {
  IREE_ASSERT_ARGUMENT(out_bytes_written);
  *out_bytes_written = 0;

  HANDLE handle = (HANDLE)_get_osfhandle(fd);
  if (handle == INVALID_HANDLE_VALUE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "file descriptor is not backed by a valid Win32 HANDLE");
  }

  DWORD bytes_written = 0;
  OVERLAPPED overlapped = {0};
  overlapped.Offset = (DWORD)(offset & 0xFFFFFFFFu);
  overlapped.OffsetHigh = (DWORD)((offset >> 32) & 0xFFFFFFFFu);
  if (!WriteFile(handle, buffer, (DWORD)count, &bytes_written, &overlapped)) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to write requested buffer length");
  }

  *out_bytes_written = (iree_host_size_t)bytes_written;
  return iree_ok_status();
}

#else

// Returns the allowed access and length in bytes of the file descriptor.
// Returns 0 if the file descriptor has no length (a /proc stream, etc).
static iree_status_t iree_hal_platform_fd_stat(
    int fd, iree_hal_memory_access_t* out_allowed_access,
    uint64_t* out_length) {
  IREE_ASSERT_ARGUMENT(out_allowed_access);
  IREE_ASSERT_ARGUMENT(out_length);
  *out_allowed_access = IREE_HAL_MEMORY_ACCESS_NONE;
  *out_length = 0;

  struct stat buffer = {0};
  if (fstat(fd, &buffer) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "unable to stat file descriptor length");
  }

  *out_allowed_access =
      ((buffer.st_mode & S_IRUSR) ? IREE_HAL_MEMORY_ACCESS_READ : 0) |
      ((buffer.st_mode & S_IWUSR) ? IREE_HAL_MEMORY_ACCESS_WRITE : 0);
  *out_length = (uint64_t)buffer.st_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_platform_fd_pread(
    int fd, void* buffer, iree_host_size_t count, uint64_t offset,
    iree_host_size_t* out_bytes_read) {
  IREE_ASSERT_ARGUMENT(out_bytes_read);
  *out_bytes_read = 0;
  ssize_t bytes_read = pread(fd, buffer, (size_t)count, (off_t)offset);
  if (bytes_read > 0) {
    *out_bytes_read = (iree_host_size_t)bytes_read;
    return iree_ok_status();
  } else if (bytes_read == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "end of file hit during read");
  } else {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to read requested buffer length");
  }
}

static iree_status_t iree_hal_platform_fd_pwrite(
    int fd, const void* buffer, iree_host_size_t count, uint64_t offset,
    iree_host_size_t* out_bytes_written) {
  IREE_ASSERT_ARGUMENT(out_bytes_written);
  *out_bytes_written = 0;
  ssize_t bytes_written = pwrite(fd, buffer, (size_t)count, (off_t)offset);
  if (bytes_written > 0) {
    *out_bytes_written = (iree_host_size_t)bytes_written;
    return iree_ok_status();
  } else if (bytes_written == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "end of file hit during write");
  } else {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to write requested buffer length");
  }
}

#endif  // IREE_PLATFORM_WINDOWS

#endif  // IREE_FILE_IO_ENABLE

//===----------------------------------------------------------------------===//
// iree_hal_fd_file_t
//===----------------------------------------------------------------------===//

#if IREE_FILE_IO_ENABLE

typedef struct iree_hal_fd_file_t {
  iree_hal_resource_t resource;
  // Used to allocate this structure.
  iree_allocator_t host_allocator;
  // Allowed access bits.
  iree_hal_memory_access_t access;
  // Base file handle, retained.
  iree_io_file_handle_t* handle;
  // File descriptor, unretained (the handle retains it).
  // Note that this descriptor may be shared with multiple threads and all
  // operations we perform against it must be stateless.
  int fd;
  // Total file (stream) length in bytes as queried on creation.
  uint64_t length;
} iree_hal_fd_file_t;

static const iree_hal_file_vtable_t iree_hal_fd_file_vtable;

static iree_hal_fd_file_t* iree_hal_fd_file_cast(
    iree_hal_file_t* IREE_RESTRICT base_value) {
  return (iree_hal_fd_file_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_fd_file_from_handle(
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;

  // For now we only support posix file descriptors but could support other
  // handle types so long as they are compatible with pread/pwrite.
  iree_io_file_handle_primitive_t primitive =
      iree_io_file_handle_primitive(handle);
  if (primitive.type != IREE_IO_FILE_HANDLE_TYPE_FD) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "support for creating non-fd files not supported");
  }
  const int fd = primitive.value.fd;

  IREE_TRACE_ZONE_BEGIN(z0);

  // Query the file length. This also acts as a quick check that the file
  // descriptor is accessible.
  iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_NONE;
  uint64_t length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_platform_fd_stat(fd, &allowed_access, &length));

  // Verify that the requested access can be satisfied.
  if (iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_READ) &&
      !iree_all_bits_set(allowed_access, IREE_HAL_MEMORY_ACCESS_READ)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "read access requested on a file descriptor that is not readable");
  } else if (iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_WRITE) &&
             !iree_all_bits_set(allowed_access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "write access requested on a file descriptor that "
                            "is not writable");
  }

  // Allocate object that retains the underlying file handle and our opened
  // descriptor.
  iree_hal_fd_file_t* file = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*file), (void**)&file));
  iree_hal_resource_initialize(&iree_hal_fd_file_vtable, &file->resource);
  file->host_allocator = host_allocator;
  file->access = access;
  file->handle = handle;
  iree_io_file_handle_retain(file->handle);
  file->fd = fd;
  file->length = length;

  *out_file = (iree_hal_file_t*)file;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_fd_file_destroy(iree_hal_file_t* IREE_RESTRICT base_file) {
  iree_hal_fd_file_t* file = iree_hal_fd_file_cast(base_file);
  iree_allocator_t host_allocator = file->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_file_handle_release(file->handle);

  iree_allocator_free(host_allocator, file);

  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_memory_access_t iree_hal_fd_file_allowed_access(
    iree_hal_file_t* base_file) {
  iree_hal_fd_file_t* file = iree_hal_fd_file_cast(base_file);
  return file->access;
}

static uint64_t iree_hal_fd_file_length(iree_hal_file_t* base_file) {
  iree_hal_fd_file_t* file = iree_hal_fd_file_cast(base_file);
  return file->length;
}

static iree_hal_buffer_t* iree_hal_fd_file_storage_buffer(
    iree_hal_file_t* base_file) {
  // We could map files if we wanted to provide this interface but today leave
  // that up to users (they can pass in HOST_ALLOCATION file handles to import).
  return NULL;
}

static bool iree_hal_fd_file_supports_synchronous_io(
    iree_hal_file_t* base_file) {
  // Host files always support synchronous IO.
  return true;
}

static iree_status_t iree_hal_fd_file_read(iree_hal_file_t* base_file,
                                           uint64_t file_offset,
                                           iree_hal_buffer_t* buffer,
                                           iree_device_size_t buffer_offset,
                                           iree_device_size_t length) {
  if (length == 0) return iree_ok_status();
  iree_hal_fd_file_t* file = iree_hal_fd_file_cast(base_file);

  iree_hal_buffer_mapping_t mapping = {{0}};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, buffer_offset, length, &mapping));

  iree_status_t status = iree_ok_status();
  uint8_t* buffer_ptr = mapping.contents.data;
  iree_host_size_t bytes_remaining = mapping.contents.data_length;
  while (iree_status_is_ok(status) && bytes_remaining > 0) {
    const iree_host_size_t bytes_requested = iree_min(bytes_remaining, INT_MAX);
    iree_host_size_t bytes_read = 0;
    status = iree_hal_platform_fd_pread(file->fd, buffer_ptr, bytes_requested,
                                        file_offset, &bytes_read);
    file_offset += bytes_read;
    buffer_ptr += bytes_read;
    bytes_remaining -= bytes_read;
  }

  if (iree_status_is_ok(status) &&
      !iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_flush_range(&mapping, 0, length);
  }

  return iree_status_join(status, iree_hal_buffer_unmap_range(&mapping));
}

static iree_status_t iree_hal_fd_file_write(iree_hal_file_t* base_file,
                                            uint64_t file_offset,
                                            iree_hal_buffer_t* buffer,
                                            iree_device_size_t buffer_offset,
                                            iree_device_size_t length) {
  if (length == 0) return iree_ok_status();
  iree_hal_fd_file_t* file = iree_hal_fd_file_cast(base_file);

  iree_hal_buffer_mapping_t mapping = {{0}};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      buffer_offset, length, &mapping));

  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_invalidate_range(&mapping, 0, length);
  }

  const uint8_t* buffer_ptr = mapping.contents.data;
  iree_host_size_t bytes_remaining = mapping.contents.data_length;
  while (iree_status_is_ok(status) && bytes_remaining > 0) {
    const iree_host_size_t bytes_requested = iree_min(bytes_remaining, INT_MAX);
    iree_host_size_t bytes_written = 0;
    status = iree_hal_platform_fd_pwrite(file->fd, buffer_ptr, bytes_requested,
                                         file_offset, &bytes_written);
    file_offset += bytes_written;
    buffer_ptr += bytes_written;
    bytes_remaining -= bytes_written;
  }

  return iree_status_join(status, iree_hal_buffer_unmap_range(&mapping));
}

static const iree_hal_file_vtable_t iree_hal_fd_file_vtable = {
    .destroy = iree_hal_fd_file_destroy,
    .allowed_access = iree_hal_fd_file_allowed_access,
    .length = iree_hal_fd_file_length,
    .storage_buffer = iree_hal_fd_file_storage_buffer,
    .supports_synchronous_io = iree_hal_fd_file_supports_synchronous_io,
    .read = iree_hal_fd_file_read,
    .write = iree_hal_fd_file_write,
};

#else

IREE_API_EXPORT iree_status_t iree_hal_fd_file_from_handle(
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

#endif  // IREE_FILE_IO_ENABLE
