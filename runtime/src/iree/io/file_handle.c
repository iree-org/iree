// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/file_handle.h"

#include "iree/base/internal/atomics.h"

#if IREE_FILE_IO_ENABLE
#if defined(IREE_PLATFORM_WINDOWS)

#include <fcntl.h>   // _open_osfhandle constants
#include <io.h>      // _commit
#include <werapi.h>  // WerRegisterExcludedMemoryBlock

#else

#include <fcntl.h>     // open
#include <sys/mman.h>  // mmap
#include <sys/stat.h>  // fstat
#include <unistd.h>    // fsync

#endif  // IREE_PLATFORM_WINDOWS
#endif  // IREE_FILE_IO_ENABLE

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

static iree_status_t iree_io_platform_fd_flush(int fd) {
#if IREE_FILE_IO_ENABLE

#if defined(IREE_PLATFORM_WINDOWS)
  int ret = _commit(fd);
#else
  int ret = fsync(fd);
#endif  // IREE_PLATFORM_WINDOWS
  return ret != -1 ? iree_ok_status()
                   : iree_make_status(iree_status_code_from_errno(errno),
                                      "unable to sync file writes");

#else
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
#endif  // IREE_FILE_IO_ENABLE
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
    case IREE_IO_FILE_HANDLE_TYPE_FD: {
      status = iree_io_platform_fd_flush(handle->primitive.value.fd);
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
// iree_io_file_handle_t utilities
//===----------------------------------------------------------------------===//

#if IREE_FILE_IO_ENABLE

#if defined(IREE_PLATFORM_WINDOWS)

static iree_status_t iree_io_file_handle_platform_open(
    iree_io_file_mode_t mode, iree_string_view_t path, bool open_existing,
    uint64_t initial_size,
    iree_io_file_handle_primitive_t* out_handle_primitive) {
  IREE_ASSERT_ARGUMENT(out_handle_primitive);
  memset(out_handle_primitive, 0, sizeof(*out_handle_primitive));

  // Convert path from a string view to a NUL-terminated C string.
  if (path.size >= IREE_MAX_PATH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "path length %" PRIhsz
                            " exceeds maximum character length of %d",
                            path.size, IREE_MAX_PATH);
  }
  char* path_str = iree_alloca(path.size + 1);
  iree_string_view_to_cstring(path, path_str, path.size + 1);

  DWORD desired_access = 0;
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_READ)) {
    desired_access |= GENERIC_READ;
  }
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_WRITE)) {
    desired_access |= GENERIC_WRITE;
  }

  DWORD share_mode = 0;
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_SHARE_READ)) {
    share_mode |= FILE_SHARE_READ;
  }
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_SHARE_WRITE)) {
    share_mode |= FILE_SHARE_WRITE;
  }

  DWORD creation_disposition = open_existing ? OPEN_EXISTING : CREATE_ALWAYS;

  DWORD flags = FILE_ATTRIBUTE_NORMAL;
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_RANDOM_ACCESS)) {
    flags |= FILE_FLAG_RANDOM_ACCESS;
  } else if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_SEQUENTIAL_SCAN)) {
    flags |= FILE_FLAG_SEQUENTIAL_SCAN;
  }
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_TEMPORARY)) {
    flags |= FILE_FLAG_DELETE_ON_CLOSE;
  }

  // Create or open the file.
  HANDLE handle = CreateFileA(path_str, desired_access, share_mode, NULL,
                              creation_disposition, flags, NULL);
  if (handle == INVALID_HANDLE_VALUE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to open file '%.*s'", (int)path.size,
                            path.data);
  }

  // If we were provided an initialize size and are creating the file then
  // adjust the file length.
  if (!open_existing) {
    // Zeroish-extend the file up to the total file size specified by the
    // caller. This may be larger than the virtual address space can handle but
    // so long as the length requested for mapping is under the size_t limit
    // this will succeed.
    LARGE_INTEGER file_size = {0};
    file_size.QuadPart = initial_size;
    if (!SetFilePointerEx(handle, file_size, NULL, FILE_BEGIN) ||
        !SetEndOfFile(handle)) {
      CloseHandle(handle);
      return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                              "failed to extend file '%.*s' to %" PRIu64
                              " bytes (out of disk space or permission denied)",
                              (int)path.size, path.data, initial_size);
    }
  }

  // Transfer ownership of the handle to a CRT file descriptor.
  // After this succeeds we cannot call CloseHandle as the CRT owns it.
  int open_flags = 0;
  if (!iree_all_bits_set(mode, IREE_IO_FILE_MODE_WRITE)) {
    open_flags |= _O_RDONLY;
  }
  int fd = _open_osfhandle((intptr_t)handle, open_flags);
  if (fd == -1) {
    CloseHandle(handle);  // must close since we didn't transfer
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "unable to transfer Win32 HANDLE to a CRT file descriptor");
  }

  out_handle_primitive->type = IREE_IO_FILE_HANDLE_TYPE_FD;
  out_handle_primitive->value.fd = fd;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_file_handle_platform_open_fd(
    int fd, iree_allocator_t host_allocator,
    iree_io_file_handle_primitive_t* out_handle_primitive) {
  IREE_ASSERT_ARGUMENT(out_handle_primitive);
  memset(out_handle_primitive, 0, sizeof(*out_handle_primitive));

  int new_fd = _dup(fd);

  if (new_fd == -1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid file_descriptor");
  }

  out_handle_primitive->type = IREE_IO_FILE_HANDLE_TYPE_FD;
  out_handle_primitive->value.fd = new_fd;
  return iree_ok_status();
}

static void iree_io_file_handle_platform_close(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  // NOTE: we opened the file using Win32 APIs but it's safe to _close since we
  // transferred ownership to the CRT with _open_osfhandle. If we used
  // IREE_IO_FILE_HANDLE_TYPE_WIN32_HANDLE we'd want to switch on that instead.
  IREE_ASSERT_EQ(handle_primitive.type, IREE_IO_FILE_HANDLE_TYPE_FD);
  _close(handle_primitive.value.fd);
}

#else

static iree_status_t iree_io_file_handle_platform_open(
    iree_io_file_mode_t mode, iree_string_view_t path, bool open_existing,
    uint64_t initial_size,
    iree_io_file_handle_primitive_t* out_handle_primitive) {
  IREE_ASSERT_ARGUMENT(out_handle_primitive);
  memset(out_handle_primitive, 0, sizeof(*out_handle_primitive));

  // Convert path from a string view to a NUL-terminated C string.
  if (path.size >= IREE_MAX_PATH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "path length %" PRIhsz
                            " exceeds maximum character length of %d",
                            path.size, IREE_MAX_PATH);
  }
  char* path_str = iree_alloca(path.size + 1);
  iree_string_view_to_cstring(path, path_str, path.size + 1);

  int flags = 0;
  // TODO(benvanik): add a flag for forking behavior.
  flags |= O_CLOEXEC;
  if (!open_existing) {
    // If the file exists open anyway and truncate as if it had been recreated.
    // This matches Win32 CREATE_ALWAYS behavior.
    flags |= O_CREAT | O_TRUNC;
  }
  if (iree_all_bits_set(mode,
                        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE)) {
    // NOTE: O_RDWR != O_RDONLY | O_WRONLY!
    flags |= O_RDWR;
  } else if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_READ)) {
    flags |= O_RDONLY;
  } else if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_WRITE)) {
    flags |= O_WRONLY;
  }
#if defined(O_DIRECT)
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_DIRECT)) {
    flags |= O_DIRECT;
  }
#endif  // O_DIRECT
#if defined(O_TMPFILE)
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_TEMPORARY)) {
    flags |= O_TMPFILE;
  }
#endif  // O_TMPFILE

  // I don't know, unix file permissions are dumb. User and group seems fine?
  const mode_t open_mode = (S_IRUSR | S_IWUSR) | (S_IRGRP | S_IWGRP);

  int fd = open(path_str, flags, open_mode);
  if (fd == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%.*s'", (int)path.size,
                            path.data);
  }

  // If we were provided an initialize size and are creating the file then
  // adjust the file length.
  if (!open_existing) {
    // Zero-extend the file up to the total file size specified by the
    // caller. Note that `ftruncate` extends too.
    if (ftruncate(fd, (off_t)initial_size) == -1) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "failed to extend file '%.*s' to %" PRIu64
                              " bytes (out of disk space or permission denied)",
                              (int)path.size, path.data, initial_size);
    }
  }

  out_handle_primitive->type = IREE_IO_FILE_HANDLE_TYPE_FD;
  out_handle_primitive->value.fd = fd;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_file_handle_platform_open_fd(
    int fd, iree_allocator_t host_allocator,
    iree_io_file_handle_primitive_t* out_handle_primitive) {
  IREE_ASSERT_ARGUMENT(out_handle_primitive);
  memset(out_handle_primitive, 0, sizeof(*out_handle_primitive));

  int new_fd = dup(fd);

  if (new_fd == -1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid file_descriptor");
  }

  out_handle_primitive->type = IREE_IO_FILE_HANDLE_TYPE_FD;
  out_handle_primitive->value.fd = new_fd;
  return iree_ok_status();
}

static void iree_io_file_handle_platform_close(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  IREE_ASSERT_EQ(handle_primitive.type, IREE_IO_FILE_HANDLE_TYPE_FD);
  close(handle_primitive.value.fd);
}

#endif  // IREE_PLATFORM_WINDOWS

static iree_status_t iree_io_file_handle_create_or_open(
    iree_io_file_mode_t mode, iree_string_view_t path, bool open_existing,
    uint64_t initial_size, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_handle) {
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_RANDOM_ACCESS |
                                  IREE_IO_FILE_MODE_SEQUENTIAL_SCAN)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at most one access pattern hint may be specified");
  }

  iree_io_file_handle_primitive_t handle_primitive = {0};
  IREE_RETURN_IF_ERROR(iree_io_file_handle_platform_open(
      mode, path, open_existing, initial_size, &handle_primitive));

  iree_io_file_access_t allowed_access = 0;
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_READ)) {
    allowed_access |= IREE_IO_FILE_ACCESS_READ;
  }
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_WRITE)) {
    allowed_access |= IREE_IO_FILE_ACCESS_WRITE;
  }
  iree_io_file_handle_release_callback_t release_callback = {
      .fn = iree_io_file_handle_platform_close,
      .user_data = NULL,
  };
  iree_io_file_handle_t* handle = NULL;
  iree_status_t status =
      iree_io_file_handle_wrap(allowed_access, handle_primitive,
                               release_callback, host_allocator, &handle);

  if (iree_status_is_ok(status)) {
    *out_handle = handle;
  } else {
    release_callback.fn(release_callback.user_data, handle_primitive);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_file_handle_create(
    iree_io_file_mode_t mode, iree_string_view_t path, uint64_t initial_size,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle) {
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);
  iree_status_t status = iree_io_file_handle_create_or_open(
      mode, path, /*open_existing=*/false, initial_size, host_allocator,
      out_handle);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_file_handle_open(
    iree_io_file_mode_t mode, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle) {
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);
  iree_status_t status = iree_io_file_handle_create_or_open(
      mode, path, /*open_existing=*/true, 0ull, host_allocator, out_handle);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_file_handle_open_fd(
    iree_io_file_mode_t mode, int fd, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_handle) {
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, fd);

  iree_io_file_access_t allowed_access = 0;
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_READ)) {
    allowed_access |= IREE_IO_FILE_ACCESS_READ;
  }
  if (iree_all_bits_set(mode, IREE_IO_FILE_MODE_WRITE)) {
    allowed_access |= IREE_IO_FILE_ACCESS_WRITE;
  }
  iree_io_file_handle_primitive_t handle_primitive = {0};

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_file_handle_platform_open_fd(fd, host_allocator,
                                               &handle_primitive));

  iree_io_file_handle_release_callback_t release_callback = {
      .fn = iree_io_file_handle_platform_close,
      .user_data = NULL,
  };

  iree_io_file_handle_t* handle = NULL;
  iree_status_t status =
      iree_io_file_handle_wrap(allowed_access, handle_primitive,
                               release_callback, host_allocator, &handle);

  if (iree_status_is_ok(status)) {
    *out_handle = handle;
  } else {
    release_callback.fn(release_callback.user_data, handle_primitive);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#else

IREE_API_EXPORT iree_status_t iree_io_file_handle_create(
    iree_io_file_mode_t mode, iree_string_view_t path, uint64_t initial_size,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle) {
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

IREE_API_EXPORT iree_status_t iree_io_file_handle_open(
    iree_io_file_mode_t mode, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle) {
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

IREE_API_EXPORT iree_status_t iree_io_file_handle_open_fd(
    iree_io_file_mode_t mode, int fd, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_handle) {
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

#endif  // IREE_FILE_IO_ENABLE

//===----------------------------------------------------------------------===//
// iree_io_file_mapping_t support
//===----------------------------------------------------------------------===//

static iree_status_t iree_io_calculate_file_view_range(
    uint64_t file_size, uint64_t offset, iree_host_size_t length,
    iree_host_size_t* out_adjusted_length) {
  *out_adjusted_length = 0;

  // Check if the start of the range runs off the end of the buffer.
  if (IREE_UNLIKELY(offset > file_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "attempted to access an address off the end of the "
                            "file range (offset=%" PRIu64 ", length=%" PRIhsz
                            ", file size=%" PRIu64 ")",
                            offset, length, file_size);
  }

  // Calculate the real length adjusted for our region within the allocation.
  const iree_host_size_t adjusted_length =
      length == IREE_HOST_SIZE_MAX ? file_size - offset : length;
  if (adjusted_length == 0) {
    // Fine (but silly) to have a zero length.
    return iree_ok_status();
  }

  // Check if the end runs over the allocation.
  const uint64_t end = offset + adjusted_length - 1;
  if (IREE_UNLIKELY(end >= file_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "attempted to access an address outside of the "
                            "file range (offset=%" PRIu64
                            ", adjusted_length=%" PRIhsz ", end=%" PRIu64
                            ", file size=%" PRIu64 ")",
                            offset, adjusted_length, end, file_size);
  }

  *out_adjusted_length = adjusted_length;
  return iree_ok_status();
}

static iree_status_t iree_io_file_mapping_from_host_allocation(
    iree_byte_span_t buffer, uint64_t offset, iree_host_size_t length,
    iree_byte_span_t* out_range) {
  *out_range = iree_byte_span_empty();

  iree_host_size_t adjusted_length = 0;
  IREE_RETURN_IF_ERROR(iree_io_calculate_file_view_range(
      (uint64_t)buffer.data_length, offset, length, &adjusted_length));

  *out_range = iree_make_byte_span(buffer.data + offset, adjusted_length);
  return iree_ok_status();
}

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_IOS) || \
    defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_MACOS)

static iree_status_t iree_io_file_handle_to_fd(
    iree_io_file_handle_primitive_t primitive, int* out_fd) {
  *out_fd = -1;
  switch (primitive.type) {
    case IREE_IO_FILE_HANDLE_TYPE_FD:
      *out_fd = primitive.value.fd;
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "no file descriptor available for file handles of type %d",
          (int)primitive.type);
  }
}

static iree_status_t iree_io_platform_map_file_view(
    iree_io_file_handle_primitive_t primitive, iree_io_file_access_t access,
    uint64_t offset, iree_host_size_t length,
    iree_io_file_mapping_flags_t flags, void** out_impl,
    iree_byte_span_t* out_contents) {
  *out_impl = NULL;
  *out_contents = iree_byte_span_empty();

  // Attempt to get a file descriptor from the provided IREE file handle.
  int fd = -1;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_to_fd(primitive, &fd),
                       "mapping file handle to file descriptor");

  // Query file size. We don't support extending/truncating files today and make
  // the user do that - we just allow the length to be IREE_HOST_SIZE_MAX to
  // indicate the remaining file should be mapped.
  struct stat file_stat = {0};
  if (fstat(fd, &file_stat) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "unable to query file size");
  }
  const uint64_t file_size = file_stat.st_size;

  // Validate and adjust view size if needed.
  iree_host_size_t adjusted_length = 0;
  IREE_RETURN_IF_ERROR(iree_io_calculate_file_view_range(
      file_size, offset, length, &adjusted_length));

  int prot = 0;
  if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_READ)) {
    prot |= PROT_READ;
  }
  if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_WRITE)) {
    prot |= PROT_WRITE;
  }

  int map_flags = 0;
  if (iree_all_bits_set(flags, IREE_IO_FILE_MAPPING_FLAG_PRIVATE)) {
    map_flags |= MAP_PRIVATE;
  } else {
    map_flags |= MAP_SHARED;
  }
#if defined(MAP_HUGETLB)
  if (iree_all_bits_set(flags, IREE_IO_FILE_MAPPING_FLAG_LARGE_PAGES)) {
    map_flags |= MAP_HUGETLB;
  }
#endif  // MAP_HUGETLB

  // Map the memory.
  void* ptr = mmap(NULL, adjusted_length, prot, map_flags, fd, offset);
  if (ptr == MAP_FAILED) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to map file handle range %" PRIu64
                            "-%" PRIu64 " (%" PRIhsz
                            " bytes) from file of %" PRIu64 " total bytes",
                            offset, offset + length, length, file_size);
  }

  // Pass hints to the memory manager - informational only.
  int advice = 0;
  if (iree_all_bits_set(flags, IREE_IO_FILE_MAPPING_FLAG_SEQUENTIAL_ACCESS)) {
    advice |= MADV_SEQUENTIAL;
  }
#if defined(MADV_DONTDUMP)
  if (iree_all_bits_set(flags, IREE_IO_FILE_MAPPING_FLAG_EXCLUDE_FROM_DUMPS)) {
    advice |= MADV_DONTDUMP;
  }
#endif  // MADV_DONTDUMP
  if (advice) {
    madvise(ptr, adjusted_length, advice);
  }

  *out_impl = ptr;
  *out_contents = iree_make_byte_span(ptr, adjusted_length);
  return iree_ok_status();
}

static void iree_io_platform_unmap_file_view(iree_io_file_mapping_flags_t flags,
                                             void* impl,
                                             iree_byte_span_t contents) {
  if (impl) {
    munmap(impl, (size_t)contents.data_length);
  }
}

#elif defined(IREE_PLATFORM_WINDOWS)

static iree_status_t iree_io_file_handle_to_win32_handle(
    iree_io_file_handle_primitive_t primitive, HANDLE* out_handle) {
  *out_handle = INVALID_HANDLE_VALUE;
  switch (primitive.type) {
    case IREE_IO_FILE_HANDLE_TYPE_FD:
      *out_handle = (HANDLE)_get_osfhandle(primitive.value.fd);
      if (*out_handle == INVALID_HANDLE_VALUE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "file descriptor is not backed by a valid Win32 HANDLE");
      }
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "no Win32 HANDLE available for file handles of type %d",
          (int)primitive.type);
  }
}

static iree_status_t iree_io_platform_map_file_view(
    iree_io_file_handle_primitive_t primitive, iree_io_file_access_t access,
    uint64_t offset, iree_host_size_t length,
    iree_io_file_mapping_flags_t flags, void** out_impl,
    iree_byte_span_t* out_contents) {
  *out_impl = NULL;
  *out_contents = iree_byte_span_empty();

  // Attempt to get a Win32 HANDLE from the provided IREE file handle.
  HANDLE handle = INVALID_HANDLE_VALUE;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_to_win32_handle(primitive, &handle),
                       "mapping file handle to win32 handle");

  // Query file size. We don't support extending/truncating files today and make
  // the user do that - we just allow the length to be IREE_HOST_SIZE_MAX to
  // indicate the remaining file should be mapped.
  FILE_STANDARD_INFO file_info = {0};
  if (!GetFileInformationByHandleEx(handle, FileStandardInfo, &file_info,
                                    (DWORD)sizeof(file_info))) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to query file handle information");
  }
  const uint64_t file_size = file_info.EndOfFile.QuadPart;

  // Validate and adjust view size if needed.
  iree_host_size_t adjusted_length = 0;
  IREE_RETURN_IF_ERROR(iree_io_calculate_file_view_range(
      file_size, offset, length, &adjusted_length));

  // Create a file mapping object which will retain the file handle for the
  // lifetime of the mapping.
  DWORD protect = 0;
  if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_WRITE)) {
    protect |= PAGE_READWRITE;
  } else if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_READ)) {
    protect |= PAGE_READONLY;
  }
  if (iree_all_bits_set(flags, IREE_IO_FILE_MAPPING_FLAG_LARGE_PAGES)) {
    protect |= SEC_LARGE_PAGES;
  }
  HANDLE mapping =
      CreateFileMappingA(handle, NULL, protect, /*dwMaximumSizeHigh=*/0,
                         /*dwMaximumSizeLow=*/0, /*lpName=*/NULL);
  if (!mapping) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to create file mapping for file handle");
  }

  // Map the requested range into the virtual address space of the process.
  DWORD desired_access = 0;
  if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_READ)) {
    desired_access |= FILE_MAP_READ;
  } else if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_WRITE)) {
    desired_access |= FILE_MAP_WRITE;
  }
  LARGE_INTEGER offset_li = {0};
  offset_li.QuadPart = offset;
  void* ptr = MapViewOfFileEx(mapping, desired_access, offset_li.HighPart,
                              offset_li.LowPart, (SIZE_T)adjusted_length,
                              /*lpBaseAddress=*/NULL);
  if (!ptr) {
    CloseHandle(mapping);
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "failed to map file handle range %" PRIu64 "-%" PRIu64 " (%" PRIhsz
        " bytes) from file of %" PRIu64 " total bytes",
        offset, offset + adjusted_length, adjusted_length, file_size);
  }

#if defined(WER_MAX_REGISTERED_ENTRIES) && \
    WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_SYSTEM)
  // If the user specified that we should exclude the contents from dumps then
  // we need to tell Windows Error Reporting. Unfortunately the API is broken
  // and only accepts a DWORD (it was added in Windows 10 **and uses a DWORD for
  // size** :facepalm:). This is informational so we just try and maybe fail.
  // Note that there's also a very small limit on the number of exclusions
  // (WER_MAX_REGISTERED_ENTRIES = 512) so we can't just loop and try to exclude
  // 4GB blocks in all cases. We try anyway, though. Maybe this isn't even
  // useful - the docs are iffy. Oh well.
  if (iree_all_bits_set(flags, IREE_IO_FILE_MAPPING_FLAG_EXCLUDE_FROM_DUMPS)) {
    iree_host_size_t bytes_excluded = 0;
    iree_host_size_t bytes_remaining = adjusted_length;
    while (bytes_remaining > 0) {
      const DWORD bytes_to_exclude = iree_min(bytes_remaining, UINT32_MAX);
      WerRegisterExcludedMemoryBlock((uint8_t*)ptr + bytes_excluded,
                                     bytes_to_exclude);
      bytes_excluded += bytes_to_exclude;
      bytes_remaining -= bytes_to_exclude;
    }
  }
#endif  // WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP |
        // WINAPI_PARTITION_SYSTEM)

  *out_impl = mapping;  // transferred to caller
  *out_contents = iree_make_byte_span(ptr, adjusted_length);
  return iree_ok_status();
}

static void iree_io_platform_unmap_file_view(iree_io_file_mapping_flags_t flags,
                                             void* impl,
                                             iree_byte_span_t contents) {
  if (contents.data) {
    UnmapViewOfFile(contents.data);
  }

#if defined(WER_MAX_REGISTERED_ENTRIES) && \
    WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_SYSTEM)
  if (contents.data &&
      iree_all_bits_set(flags, IREE_IO_FILE_MAPPING_FLAG_EXCLUDE_FROM_DUMPS)) {
    WerUnregisterExcludedMemoryBlock(contents.data);
    iree_host_size_t bytes_unexcluded = 0;
    iree_host_size_t bytes_remaining = contents.data_length;
    while (bytes_remaining > 0) {
      const DWORD bytes_to_unexclude = iree_min(bytes_remaining, UINT32_MAX);
      WerUnregisterExcludedMemoryBlock(contents.data + bytes_unexcluded);
      bytes_unexcluded += bytes_to_unexclude;
      bytes_remaining -= bytes_to_unexclude;
    }
  }
#endif  // WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP |
        // WINAPI_PARTITION_SYSTEM)

  if (impl) {
    CloseHandle((HANDLE)impl);
  }
}

#else

static iree_status_t iree_io_platform_map_file_view(
    iree_io_file_handle_primitive_t primitive, iree_io_file_access_t access,
    uint64_t offset, iree_host_size_t length,
    iree_io_file_mapping_flags_t flags, void** out_impl,
    iree_byte_span_t* out_contents) {
  *out_impl = NULL;
  *out_contents = iree_byte_span_empty();
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "no support for mapping file views on this platform");
}

static void iree_io_platform_unmap_file_view(iree_io_file_mapping_flags_t flags,
                                             void* impl,
                                             iree_byte_span_t contents) {}

#endif  // IREE_PLATFORM_*

//===----------------------------------------------------------------------===//
// iree_io_file_mapping_t
//===----------------------------------------------------------------------===//

struct iree_io_file_mapping_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;
  // File handle that owns the underlying file. Retained.
  iree_io_file_handle_t* handle;
  // Flags used when creating the mapping.
  iree_io_file_mapping_flags_t flags;
  // Platform-defined implementation handle.
  //  - mmap: base pointer returned from mmap
  //  - Win32: HANDLE returned by CreateFileMappingA
  void* impl;
  // Mapped contents in host memory. Access matches that requested on mapping.
  iree_byte_span_t contents;
};

IREE_API_EXPORT iree_status_t iree_io_file_map_view(
    iree_io_file_handle_t* handle, iree_io_file_access_t access,
    uint64_t offset, iree_host_size_t length,
    iree_io_file_mapping_flags_t flags, iree_allocator_t host_allocator,
    iree_io_file_mapping_t** out_mapping) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_mapping);
  *out_mapping = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, offset);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, length);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, flags);

  iree_io_file_mapping_t* mapping = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*mapping),
                                (void**)&mapping));
  iree_atomic_ref_count_init(&mapping->ref_count);
  mapping->host_allocator = host_allocator;
  mapping->handle = handle;
  iree_io_file_handle_retain(mapping->handle);
  mapping->flags = flags;
  mapping->contents = iree_byte_span_empty();

  iree_status_t status = iree_ok_status();

  // Special case for for host allocations: we can directly use them (with
  // translation). Otherwise we let the platform-specific logic take care of
  // things (if it exists).
  iree_io_file_handle_primitive_t primitive =
      iree_io_file_handle_primitive(handle);
  if (primitive.type == IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    iree_byte_span_t file_buffer = primitive.value.host_allocation;
    status = iree_io_file_mapping_from_host_allocation(
        file_buffer, offset, length, &mapping->contents);
  } else {
    // Use platform APIs to map the file.
    status =
        iree_io_platform_map_file_view(primitive, access, offset, length, flags,
                                       &mapping->impl, &mapping->contents);
  }

  if (iree_status_is_ok(status)) {
    *out_mapping = mapping;
  } else {
    iree_io_file_mapping_release(mapping);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_io_file_mapping_destroy(iree_io_file_mapping_t* mapping) {
  IREE_ASSERT_ARGUMENT(mapping);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = mapping->host_allocator;

  if (mapping->impl) {
    iree_io_platform_unmap_file_view(mapping->flags, mapping->impl,
                                     mapping->contents);
  }

  iree_io_file_handle_release(mapping->handle);

  iree_allocator_free(host_allocator, mapping);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_io_file_mapping_retain(
    iree_io_file_mapping_t* mapping) {
  if (IREE_LIKELY(mapping)) {
    iree_atomic_ref_count_inc(&mapping->ref_count);
  }
}

IREE_API_EXPORT void iree_io_file_mapping_release(
    iree_io_file_mapping_t* mapping) {
  if (IREE_LIKELY(mapping) &&
      iree_atomic_ref_count_dec(&mapping->ref_count) == 1) {
    iree_io_file_mapping_destroy(mapping);
  }
}

IREE_API_EXPORT iree_host_size_t
iree_io_file_mapping_length(const iree_io_file_mapping_t* mapping) {
  IREE_ASSERT_ARGUMENT(mapping);
  return mapping->contents.data_length;
}

IREE_API_EXPORT iree_const_byte_span_t
iree_io_file_mapping_contents_ro(const iree_io_file_mapping_t* mapping) {
  return iree_make_const_byte_span(mapping->contents.data,
                                   mapping->contents.data_length);
}

IREE_API_EXPORT iree_byte_span_t
iree_io_file_mapping_contents_rw(iree_io_file_mapping_t* mapping) {
  IREE_ASSERT_ARGUMENT(mapping);
  return mapping->contents;
}
