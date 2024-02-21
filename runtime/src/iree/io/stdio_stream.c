// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/stdio_stream.h"

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(IREE_PLATFORM_WINDOWS)

#include <fcntl.h>
#include <io.h>

#define IREE_SET_BINARY_MODE(handle) _setmode(_fileno(handle), O_BINARY)

#define iree_fseek _fseeki64
#define iree_ftell _ftelli64

#else

#define IREE_SET_BINARY_MODE(handle) ((void)0)

#if _FILE_OFFSET_BITS == 64 || _POSIX_C_SOURCE >= 200112L
#define iree_fseek fseeko
#define iree_ftell ftello
#else
#define iree_fseek fseek
#define iree_ftell ftell
#endif  // 64-bit file offset support

#endif  // IREE_PLATFORM_WINDOWS

// Makes a new status message ala iree_make_status but includes the error number
// and optional string message on platforms that support it.
#if defined(IREE_PLATFORM_WINDOWS)
#define iree_make_stdio_status(message)                                     \
  iree_make_status(iree_status_code_from_errno(errno), message " (%d: %s)", \
                   errno, strerror(errno))
#define iree_make_stdio_statusf(format, ...)                               \
  iree_make_status(iree_status_code_from_errno(errno), format " (%d: %s)", \
                   __VA_ARGS__, errno, strerror(errno))
#else
#define iree_make_stdio_status(...) \
  iree_make_status(IREE_STATUS_UNKNOWN, __VA_ARGS__)
#define iree_make_stdio_statusf iree_make_stdio_status
#endif  // IREE_PLATFORM_*

//===----------------------------------------------------------------------===//
// iree_io_stdio_stream_t
//===----------------------------------------------------------------------===//

#define IREE_MAX_PATH ((size_t)2048)

typedef struct iree_io_stdio_stream_t {
  iree_io_stream_t base;
  iree_allocator_t host_allocator;
  FILE* handle;
  bool owns_handle;
} iree_io_stdio_stream_t;

static const iree_io_stream_vtable_t iree_io_stdio_stream_vtable;

static iree_io_stdio_stream_t* iree_io_stdio_stream_cast(
    iree_io_stream_t* IREE_RESTRICT base_stream) {
  return (iree_io_stdio_stream_t*)base_stream;
}

IREE_API_EXPORT iree_status_t iree_io_stdio_stream_wrap(
    iree_io_stream_mode_t mode, FILE* handle, bool owns_handle,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stdio_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*stream), (void**)&stream));
  iree_atomic_ref_count_init(&stream->base.ref_count);
  stream->base.vtable = &iree_io_stdio_stream_vtable;
  stream->base.mode = mode;
  stream->host_allocator = host_allocator;
  stream->handle = handle;
  stream->owns_handle = owns_handle;

  *out_stream = &stream->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#if IREE_FILE_IO_ENABLE
IREE_API_EXPORT iree_status_t iree_io_stdio_stream_open(
    iree_io_stdio_stream_mode_t mode, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  iree_io_stream_mode_t stream_mode = IREE_IO_STREAM_MODE_SEEKABLE;
  if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_READ)) {
    stream_mode |= IREE_IO_STREAM_MODE_READABLE;
  }
  if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_WRITE)) {
    stream_mode |= IREE_IO_STREAM_MODE_WRITABLE;
  }

  // NOTE: not all implementations support all mode flags and this may have
  // different behavior. We should paper over it here but don't today given the
  // limited usage of this and our intent to rewrite it all using
  // platform-optimal APIs instead of stdio.
  char fopen_mode[16] = {0};
  if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_READ |
                                  IREE_IO_STDIO_STREAM_MODE_WRITE |
                                  IREE_IO_STDIO_STREAM_MODE_APPEND)) {
    strcat(fopen_mode, "a+");
  } else if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_READ |
                                         IREE_IO_STDIO_STREAM_MODE_WRITE |
                                         IREE_IO_STDIO_STREAM_MODE_DISCARD)) {
    strcat(fopen_mode, "w+");
  } else if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_READ |
                                         IREE_IO_STDIO_STREAM_MODE_WRITE)) {
    strcat(fopen_mode, "r+");
  } else if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_WRITE |
                                         IREE_IO_STDIO_STREAM_MODE_APPEND)) {
    strcat(fopen_mode, "a");
  } else if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_WRITE)) {
    strcat(fopen_mode, "w");
  } else if (iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_READ)) {
    strcat(fopen_mode, "r");
  }
  if (iree_all_bits_set(stream_mode, IREE_IO_STREAM_MODE_WRITABLE) &&
      !iree_all_bits_set(mode, IREE_IO_STDIO_STREAM_MODE_DISCARD)) {
    // If writable and not discard then the file must not exist.
    // TODO(benvanik): actually observe this; the C11 spec says `x` is supported
    // but at least on MSVC's CRT it isn't. We can emulate this with stat and
    // such but today we don't have any uses that require it.
    // strcat(fopen_mode, "x");
  }
  // Force binary mode (avoid Windows CRLF expansion).
  strcat(fopen_mode, "b");

  // Since we stack alloc the path we want to keep it reasonable.
  // We could heap allocate instead but a few thousand chars is quite long and
  // since Windows doesn't support more than ~256 we generally keep them short
  // anyway.
  if (path.size > IREE_MAX_PATH) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "path exceeds reasonable maximum (%" PRIhsz
                            " > %" PRIhsz ")",
                            path.size, IREE_MAX_PATH);
  }
  char* fopen_path = (char*)iree_alloca(path.size + 1);
  memcpy(fopen_path, path.data, path.size);
  fopen_path[path.size] = 0;  // NUL

  iree_status_t status = iree_ok_status();
  FILE* handle = fopen(fopen_path, fopen_mode);
  if (handle == NULL) {
    // NOTE: for some crazy reason errno isn't set by all implementations. We
    // know it is on Windows but currently leave all others to :shrug:. We could
    // check C library implementations and versions to make this better.
    status = iree_make_stdio_statusf("unable to open file `%.*s` with mode %d",
                                     (int)path.size, path.data, mode);
  }

  iree_io_stream_t* stream = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_stdio_stream_wrap(
        stream_mode, handle, /*owns_handle=*/true, host_allocator, &stream);
  }

  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    if (stream) {
      iree_io_stream_release(stream);
    } else {
      fclose(handle);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
#else
IREE_API_EXPORT iree_status_t iree_io_stdio_stream_open(
    iree_io_stdio_stream_mode_t mode, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}
#endif  // IREE_FILE_IO_ENABLE

static void iree_io_stdio_stream_destroy(
    iree_io_stream_t* IREE_RESTRICT base_stream) {
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  iree_allocator_t host_allocator = stream->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  fflush(stream->handle);
  if (stream->owns_handle) {
    fclose(stream->handle);
  }

  iree_allocator_free(host_allocator, stream);

  IREE_TRACE_ZONE_END(z0);
}

static iree_io_stream_pos_t iree_io_stdio_stream_offset(
    iree_io_stream_t* base_stream) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  int64_t pos = iree_ftell(stream->handle);
  if (pos == -1) return 0;
  return (iree_io_stream_pos_t)pos;
}

static iree_io_stream_pos_t iree_io_stdio_stream_length(
    iree_io_stream_t* base_stream) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);

  // Capture original offset so we can return to it.
  int64_t origin = iree_ftell(stream->handle);
  if (origin == -1) return 0;

  // Seek to the end of the file.
  if (iree_fseek(stream->handle, 0, SEEK_END) != 0) return 0;

  // Query the position, telling us the total file length in bytes.
  int64_t length = iree_ftell(stream->handle);
  if (length == -1) return 0;

  // Seek back to the file origin.
  if (iree_fseek(stream->handle, origin, SEEK_SET) != 0) return 0;

  return (iree_io_stream_pos_t)length;
}

static iree_status_t iree_io_stdio_stream_seek(
    iree_io_stream_t* base_stream, iree_io_stream_seek_mode_t seek_mode,
    iree_io_stream_pos_t seek_offset) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  int origin = 0;
  switch (seek_mode) {
    case IREE_IO_STREAM_SEEK_SET:
      origin = SEEK_SET;
      break;
    case IREE_IO_STREAM_SEEK_FROM_CURRENT:
      origin = SEEK_CUR;
      break;
    case IREE_IO_STREAM_SEEK_FROM_END:
      origin = SEEK_END;
      break;
    default:
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid seek mode");
      break;
  }

  if (iree_status_is_ok(status)) {
    if (iree_fseek(stream->handle, seek_offset, origin) != 0) {
      status = iree_make_stdio_status("failed to seek");
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_stdio_stream_read(
    iree_io_stream_t* base_stream, iree_host_size_t buffer_capacity,
    void* buffer, iree_host_size_t* out_buffer_length) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(buffer);
  if (out_buffer_length) *out_buffer_length = 0;
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // Read in ~2GB chunks - even platforms with 64-bit support sometimes don't
  // like read lengths >2GB and there's not really any benefit to doing 12GB
  // reads in one go anyway.
  iree_host_size_t bytes_read = 0;
  while (bytes_read < buffer_capacity) {
    iree_host_size_t chunk_size =
        iree_min(buffer_capacity - bytes_read, INT_MAX);
    iree_host_size_t read_size =
        fread((uint8_t*)buffer + bytes_read, 1, chunk_size, stream->handle);
    if (read_size != chunk_size) {
      // Failed to read chunk - may have reached EOF.
      if (feof(stream->handle)) {
        if (out_buffer_length) {
          // Ok to hit EOF; just return what's valid.
          *out_buffer_length = bytes_read + read_size;
        } else {
          status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                    "end-of-file encountered during read");
        }
      } else {
        status = iree_make_stdio_status("read failed");
      }
      break;
    }
    bytes_read += read_size;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_stdio_stream_write(iree_io_stream_t* base_stream,
                                                iree_host_size_t buffer_length,
                                                const void* buffer) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(buffer);
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // Write in ~2GB chunks - even platforms with 64-bit support sometimes don't
  // like write lengths >2GB and there's not really any benefit to doing 12GB
  // writes in one go anyway.
  iree_host_size_t bytes_written = 0;
  while (bytes_written < buffer_length) {
    iree_host_size_t chunk_size =
        iree_min(buffer_length - bytes_written, INT_MAX);
    iree_host_size_t write_size =
        fwrite((uint8_t*)buffer + bytes_written, 1, chunk_size, stream->handle);
    if (write_size != chunk_size) {
      // Failed to write chunk; likely exhausted disk space.
      status = iree_make_stdio_status(
          "write failed, possibly out of disk space or device lost");
      break;
    }
    bytes_written += write_size;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_stdio_stream_fill(
    iree_io_stream_t* base_stream, iree_io_stream_pos_t count,
    const void* pattern, iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(pattern);
  iree_io_stdio_stream_t* stream = iree_io_stdio_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // There's not an stdio API for filling contents. When using platform APIs we
  // can extend files when the pattern is zeros to quickly do things but here
  // we just bash fwrite. We could buffer up a reasonable size (4096 etc) of the
  // pattern repeating but this shouldn't be performance critical.
  for (iree_io_stream_pos_t i = 0; i < count; ++i) {
    if (fwrite(pattern, pattern_length, 1, stream->handle) != pattern_length) {
      status = iree_make_stdio_status(
          "write failed, possibly out of disk space or device lost");
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_stdio_stream_map_read(
    iree_io_stream_t* stream, iree_host_size_t length,
    iree_const_byte_span_t* out_span) {
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "stdio streams do not support mapping");
}

static iree_status_t iree_io_stdio_stream_map_write(
    iree_io_stream_t* stream, iree_host_size_t length,
    iree_byte_span_t* out_span) {
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "stdio streams do not support mapping");
}

static const iree_io_stream_vtable_t iree_io_stdio_stream_vtable = {
    .destroy = iree_io_stdio_stream_destroy,
    .offset = iree_io_stdio_stream_offset,
    .length = iree_io_stdio_stream_length,
    .seek = iree_io_stdio_stream_seek,
    .read = iree_io_stdio_stream_read,
    .write = iree_io_stdio_stream_write,
    .fill = iree_io_stdio_stream_fill,
    .map_read = iree_io_stdio_stream_map_read,
    .map_write = iree_io_stdio_stream_map_write,
};
