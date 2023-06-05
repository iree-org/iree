// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/file_io.h"

#include "iree/base/config.h"

#if IREE_FILE_IO_ENABLE

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <fcntl.h>
#include <io.h>
#define IREE_SET_BINARY_MODE(handle) _setmode(_fileno(handle), O_BINARY)
#else
#define IREE_SET_BINARY_MODE(handle) ((void)0)
#endif  // IREE_PLATFORM_WINDOWS

// We could take alignment as an arg, but roughly page aligned should be
// acceptable for all uses - if someone cares about memory usage they won't
// be using this method.
#define IREE_FILE_BASE_ALIGNMENT 4096

#if defined(IREE_PLATFORM_WINDOWS)
#define iree_fseek64 _fseeki64
#define iree_ftell64 _ftelli64
#else
#define iree_fseek64 fseek
#define iree_ftell64 ftell
#endif  // IREE_PLATFORM_WINDOWS

iree_status_t iree_file_exists(const char* path) {
  IREE_ASSERT_ARGUMENT(path);
  IREE_TRACE_ZONE_BEGIN(z0);

  struct stat stat_buf;
  iree_status_t status =
      stat(path, &stat_buf) == 0
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_NOT_FOUND, "'%s'", path);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_file_query_length(FILE* file, uint64_t* out_length) {
  IREE_ASSERT_ARGUMENT(out_length);
  *out_length = 0;
  if (!file) return iree_ok_status();

  // Capture original offset so we can return to it.
  uint64_t origin = iree_ftell64(file);

  // Seek to the end of the file.
  if (iree_fseek64(file, 0, SEEK_END) == -1) {
    return iree_make_status(IREE_STATUS_INTERNAL, "seek (end)");
  }

  // Query the position, telling us the total file length in bytes.
  uint64_t file_length = iree_ftell64(file);
  if (file_length == -1L) {
    return iree_make_status(IREE_STATUS_INTERNAL, "size query");
  }

  // Seek back to the file origin.
  if (iree_fseek64(file, origin, SEEK_SET) == -1) {
    return iree_make_status(IREE_STATUS_INTERNAL, "seek (beg)");
  }

  *out_length = file_length;
  return iree_ok_status();
}

bool iree_file_is_at(FILE* file, uint64_t position) {
  return iree_ftell64(file) == position;
}

iree_status_t iree_file_contents_allocator_ctl(void* self,
                                               iree_allocator_command_t command,
                                               const void* params,
                                               void** inout_ptr) {
  if (command != IREE_ALLOCATOR_COMMAND_FREE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "file contents deallocator must only be used to "
                            "deallocate file contents");
  }
  iree_file_contents_t* contents = (iree_file_contents_t*)self;
  if (contents->buffer.data != *inout_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only the file contents buffer is valid");
  }
  iree_allocator_t allocator = contents->allocator;
  iree_allocator_free(allocator, contents);
  return iree_ok_status();
}

iree_allocator_t iree_file_contents_deallocator(
    iree_file_contents_t* contents) {
  iree_allocator_t allocator = {
      .self = contents,
      .ctl = iree_file_contents_allocator_ctl,
  };
  return allocator;
}

void iree_file_contents_free(iree_file_contents_t* contents) {
  if (!contents) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(contents->allocator, contents);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_file_read_contents_impl(
    FILE* file, iree_allocator_t allocator,
    iree_file_contents_t** out_contents) {
  // Query total file length so we can preallocate storage.
  // The file size may be larger than the buffer we can allocate (>2GB file on
  // 32-bit devices) so we check that here early even though it may have false
  // negatives (not enough virtual address space to allocate, etc).
  uint64_t file_length = 0;
  IREE_RETURN_IF_ERROR(iree_file_query_length(file, &file_length));
  if (file_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "file length exceeds host address range");
  }
  iree_host_size_t file_size = (iree_host_size_t)file_length;

  // Compute total size with alignment padding.
  // We allocate +1 to force a trailing \0 in case this is used as a cstring.
  iree_file_contents_t* contents = NULL;
  iree_host_size_t total_size =
      sizeof(*contents) + IREE_FILE_BASE_ALIGNMENT + file_size + /*NUL*/ 1;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&contents));

  contents->allocator = allocator;
  contents->buffer.data = (void*)iree_host_align(
      (uintptr_t)contents + sizeof(*contents), IREE_FILE_BASE_ALIGNMENT);
  contents->buffer.data_length = file_size;

  // Attempt to read the file into memory.
  if (fread(contents->buffer.data, 1, file_size, file) != file_size) {
    iree_allocator_free(allocator, contents);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "unable to read entire %zu file bytes", file_size);
  }

  // Add trailing NUL to make the contents C-string compatible.
  contents->buffer.data[file_size] = 0;  // NUL
  *out_contents = contents;
  return iree_ok_status();
}

iree_status_t iree_file_read_contents(const char* path,
                                      iree_allocator_t allocator,
                                      iree_file_contents_t** out_contents) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;

  FILE* file = fopen(path, "rb");
  if (file == NULL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_NOT_FOUND, "failed to open file '%s'",
                            path);
  }

  // Read the file contents into memory.
  iree_status_t status =
      iree_file_read_contents_impl(file, allocator, out_contents);
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(status, "reading file '%s'", path);
  }

  fclose(file);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_file_write_contents(const char* path,
                                       iree_const_byte_span_t content) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(path);

  FILE* file = fopen(path, "wb");
  if (file == NULL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "failed to open file '%s'", path);
  }

  iree_status_t status = iree_ok_status();
  if (content.data_length > 0) {
    int ret = fwrite((char*)content.data, content.data_length, 1, file);
    if (ret != 1) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "unable to write file contents of %zu bytes to '%s'",
                           content.data_length, path);
    }
  }

  fclose(file);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_stdin_read_contents_impl(
    iree_allocator_t allocator, iree_file_contents_t** out_contents) {
  // HACK: fix stdin mode to binary on Windows to match Unix behavior.
  // Ideally we'd do this in one place for all our tools.
  IREE_SET_BINARY_MODE(stdin);

  iree_host_size_t capacity = 4096;
  iree_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(*contents) + IREE_FILE_BASE_ALIGNMENT + capacity,
      (void**)&contents));
  contents->buffer.data = (void*)iree_host_align(
      (uintptr_t)contents + sizeof(*contents), IREE_FILE_BASE_ALIGNMENT);

  iree_host_size_t size = 0;
  for (int c = getchar(); c != EOF; c = getchar()) {
    if (size >= capacity - /*NUL*/ 1) {
      // NOTE: if we realloc we may end up with a new alignment and need to move
      // the data around.
      uintptr_t old_offset =
          (uintptr_t)contents->buffer.data - (uintptr_t)contents;
      iree_host_size_t new_capacity = capacity * 2;
      iree_file_contents_t* new_contents = contents;
      iree_status_t status = iree_allocator_realloc(
          allocator,
          sizeof(*new_contents) + IREE_FILE_BASE_ALIGNMENT + new_capacity,
          (void**)&new_contents);
      if (!iree_status_is_ok(status)) {
        iree_allocator_free(allocator, contents);
        return status;
      }
      contents = new_contents;
      uint8_t* old_data = (uint8_t*)new_contents + old_offset;
      uint8_t* new_data = (uint8_t*)iree_host_align(
          (uintptr_t)new_contents + sizeof(*new_contents),
          IREE_FILE_BASE_ALIGNMENT);
      if (new_data != old_data) {
        // Alignment changed; move the data with safety for overlapping.
        memmove(new_data, old_data, size);
      }
      contents->buffer.data = new_data;
      capacity = new_capacity;
    }
    contents->buffer.data[size++] = c;
  }

  contents->allocator = allocator;
  contents->buffer.data[size] = 0;  // NUL
  contents->buffer.data_length = size;
  *out_contents = contents;
  return iree_ok_status();
}

iree_status_t iree_stdin_read_contents(iree_allocator_t allocator,
                                       iree_file_contents_t** out_contents) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  iree_status_t status = iree_stdin_read_contents_impl(allocator, out_contents);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#else

iree_status_t iree_file_exists(const char* path) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "File I/O is disabled");
}

iree_allocator_t iree_file_contents_deallocator(
    iree_file_contents_t* contents) {
  return iree_allocator_null();
}

void iree_file_contents_free(iree_file_contents_t* contents) {}

iree_status_t iree_file_read_contents(const char* path,
                                      iree_allocator_t allocator,
                                      iree_file_contents_t** out_contents) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "File I/O is disabled");
}

iree_status_t iree_file_write_contents(const char* path,
                                       iree_const_byte_span_t content) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "File I/O is disabled");
}

iree_status_t iree_stdin_read_contents(iree_allocator_t allocator,
                                       iree_file_contents_t** out_contents) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "File I/O is disabled");
}

#endif  // IREE_FILE_IO_ENABLE
