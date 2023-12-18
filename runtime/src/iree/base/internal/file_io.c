// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/file_io.h"

#if IREE_FILE_IO_ENABLE

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(IREE_PLATFORM_WINDOWS)

#include <fcntl.h>
#include <io.h>

#define IREE_SET_BINARY_MODE(handle) _setmode(_fileno(handle), O_BINARY)

#define iree_fseek64 _fseeki64
#define iree_ftell64 _ftelli64

#else

#define IREE_SET_BINARY_MODE(handle) ((void)0)

#define iree_fseek64 fseeko
#define iree_ftell64 ftello

#endif  // IREE_PLATFORM_WINDOWS

// We could take alignment as an arg, but roughly page aligned should be
// acceptable for all uses - if someone cares about memory usage they won't
// be using this method.
#define IREE_FILE_BASE_ALIGNMENT 4096

static iree_status_t iree_file_map_contents_readonly_platform(
    const char* path, iree_file_contents_t* contents);
static void iree_file_contents_free_platform(iree_file_contents_t* contents);

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
  iree_file_contents_free_platform(contents);
  iree_allocator_free(contents->allocator, contents);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_file_read_contents(const char* path,
                                      iree_file_read_flags_t flags,
                                      iree_allocator_t allocator,
                                      iree_file_contents_t** out_contents) {
  if (iree_all_bits_set(flags, IREE_FILE_READ_FLAG_PRELOAD)) {
    return iree_file_preload_contents(path, allocator, out_contents);
  } else if (iree_all_bits_set(flags, IREE_FILE_READ_FLAG_MMAP)) {
    return iree_file_map_contents_readonly(path, allocator, out_contents);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid read flag mode");
  }
}

static iree_status_t iree_file_preload_contents_impl(
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

  // Attempt to read the file into memory, chunking into ~2GB segments.
  iree_host_size_t bytes_read = 0;
  while (bytes_read < file_size) {
    iree_host_size_t chunk_size = iree_min(file_size - bytes_read, INT_MAX);
    if (fread(contents->buffer.data + bytes_read, 1, chunk_size, file) !=
        chunk_size) {
      iree_allocator_free(allocator, contents);
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "unable to read %" PRIhsz " chunk bytes",
                              chunk_size);
    }
    bytes_read += chunk_size;
  }

  // Add trailing NUL to make the contents C-string compatible.
  contents->buffer.data[file_size] = 0;  // NUL
  *out_contents = contents;
  return iree_ok_status();
}

iree_status_t iree_file_preload_contents(const char* path,
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
      iree_file_preload_contents_impl(file, allocator, out_contents);
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(status, "reading file '%s'", path);
  }

  fclose(file);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_file_map_contents_readonly(
    const char* path, iree_allocator_t allocator,
    iree_file_contents_t** out_contents) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;

  iree_file_contents_t* contents = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*contents), (void**)&contents));
  contents->allocator = allocator;

  iree_status_t status =
      iree_file_map_contents_readonly_platform(path, contents);

  if (iree_status_is_ok(status)) {
    *out_contents = contents;
  } else {
    iree_file_contents_free(contents);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_IOS) || \
    defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_MACOS)

#include <sys/mman.h>
#include <unistd.h>

static iree_status_t iree_file_map_contents_readonly_platform(
    const char* path, iree_file_contents_t* contents) {
  // Open file on disk.
  FILE* file = fopen(path, "rb");
  if (file == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "failed to open file '%s'",
                            path);
  }
  contents->mapping = file;

  // Query total file size and ensure the file will fit in memory.
  uint64_t length = 0;
  IREE_RETURN_IF_ERROR(iree_file_query_length(file, &length));
  if (length > (uint64_t)IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "file size exceeds host pointer size capacity "
                            "(64-bit file loaded into a 32-bit program)");
  }

  // Map the memory.
  void* ptr =
      mmap(NULL, (size_t)length, PROT_READ, MAP_SHARED, fileno(file), 0);
  if (ptr == MAP_FAILED) {
    return iree_make_status(iree_status_code_from_errno(errno), "mmap failed");
  }

  contents->const_buffer =
      iree_make_const_byte_span(ptr, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t iree_file_create_mapped_platform(
    const char* path, uint64_t file_size, uint64_t offset,
    iree_host_size_t length, iree_file_contents_t* contents) {
  // Create file on disk.
  FILE* file = fopen(path, "w+b");
  if (file == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "failed to open file '%s'",
                            path);
  }
  contents->mapping = file;

  // Zero-extend the file ('truncate' can extend, because... unix).
  if (ftruncate(fileno(file), (off_t)file_size) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to extend file '%s' to %" PRIu64
                            " bytes (out of disk space or permission denied)",
                            path, file_size);
  }

  // Map the memory.
  void* ptr = mmap(NULL, (size_t)length, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fileno(file), (off_t)offset);
  if (ptr == MAP_FAILED) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to map '%s' range %" PRIu64 "-%" PRIu64
                            " (%" PRIhsz " bytes) from file of %" PRIu64
                            " total bytes",
                            path, offset, offset + length, length, file_size);
  }

  contents->const_buffer =
      iree_make_const_byte_span(ptr, (iree_host_size_t)length);
  return iree_ok_status();
}

static void iree_file_contents_free_platform(iree_file_contents_t* contents) {
  if (contents->mapping) {
    munmap(contents->buffer.data, (size_t)contents->buffer.data_length);
    fclose((FILE*)contents->mapping);
  }
}

#elif defined(IREE_PLATFORM_WINDOWS)

static iree_status_t iree_file_map_contents_readonly_platform(
    const char* path, iree_file_contents_t* contents) {
  // Open the file.
  HANDLE file =
      CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                  FILE_ATTRIBUTE_READONLY | FILE_FLAG_RANDOM_ACCESS, NULL);
  if (file == INVALID_HANDLE_VALUE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to open file '%s'", path);
  }

  // Query file size and ensure it will fit in the host address space.
  LARGE_INTEGER file_size;
  if (!GetFileSizeEx(file, &file_size) ||
      (ULONGLONG)file_size.QuadPart > (ULONGLONG)IREE_HOST_SIZE_MAX) {
    CloseHandle(file);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "file size exceeds host pointer size capacity "
                            "(64-bit file loaded into a 32-bit program)");
  }

  // Create a mapping object associated with the file.
  HANDLE mapping =
      CreateFileMappingA(file, NULL, PAGE_READONLY, /*dwMaximumSizeHigh=*/0,
                         /*dwMaximumSizeLow=*/0, /*lpName=*/NULL);
  if (!mapping) {
    CloseHandle(file);
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to create file mapping, possibly due to "
                            "unaligned size or resource exhaustion");
  }
  contents->mapping = mapping;

  // Retained by the mapping so safe to release now.
  CloseHandle(file);

  // Map the file into host memory.
  void* ptr = MapViewOfFileEx(
      mapping, FILE_MAP_READ, /*dwFileOffsetHigh=*/0, /*dwFileOffsetLow=*/0,
      /*dwNumberOfBytesToMap=*/0, /*lpBaseAddress=*/NULL);
  if (!ptr) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to map file into host memory");
  }

  contents->const_buffer =
      iree_make_const_byte_span(ptr, (iree_host_size_t)file_size.QuadPart);
  return iree_ok_status();
}

static iree_status_t iree_file_create_mapped_platform(
    const char* path, uint64_t file_size, uint64_t offset,
    iree_host_size_t length, iree_file_contents_t* contents) {
  // TODO(benvanik): investigate FILE_FLAG_SEQUENTIAL_SCAN +
  // FILE_FLAG_WRITE_THROUGH flags once we have some benchmarks.
  HANDLE file = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, NULL,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (file == INVALID_HANDLE_VALUE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to open file '%s'", path);
  }

  // Zero-extend the file up to the total file size specified by the caller.
  // This may be larger than the virtual address space can handle but so long as
  // the length requested for mapping is under the size_t limit this will
  // succeed.
  LARGE_INTEGER file_size_li;
  file_size_li.QuadPart = file_size;
  if (!SetFilePointerEx(file, file_size_li, NULL, FILE_BEGIN) ||
      !SetEndOfFile(file)) {
    CloseHandle(file);
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to extend file '%s' to %" PRIu64
                            " bytes (out of disk space or permission denied)",
                            path, file_size);
  }

  // Create a file mapping object which will retain the file handle for the
  // lifetime of the mapping.
  HANDLE mapping =
      CreateFileMappingA(file, NULL, PAGE_READWRITE, /*dwMaximumSizeHigh=*/0,
                         /*dwMaximumSizeLow=*/0, /*lpName=*/NULL);
  if (!mapping) {
    CloseHandle(file);
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to create file mapping for '%s'", path);
  }

  // Retained by the mapping so safe to release now.
  CloseHandle(file);

  // Map the requested range into the virtual address space of the process.
  LARGE_INTEGER offset_li;
  offset_li.QuadPart = offset;
  void* ptr = MapViewOfFileEx(mapping, FILE_MAP_ALL_ACCESS, offset_li.HighPart,
                              offset_li.LowPart, (SIZE_T)length,
                              /*lpBaseAddress=*/NULL);
  if (!ptr) {
    CloseHandle(mapping);
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to map '%s' range %" PRIu64 "-%" PRIu64
                            " (%" PRIhsz " bytes) from file of %" PRIu64
                            " total bytes",
                            path, offset, offset + length, length, file_size);
  }

  contents->mapping = mapping;
  contents->buffer = iree_make_byte_span(ptr, length);
  return iree_ok_status();
}

static void iree_file_contents_free_platform(iree_file_contents_t* contents) {
  if (contents->mapping) {
    UnmapViewOfFile(contents->buffer.data);
    CloseHandle((HANDLE)contents->mapping);
  }
}

#else

static iree_status_t iree_file_map_contents_readonly_platform(
    const char* path, iree_file_contents_t* contents) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file mapping not supported on this platform");
}

static iree_status_t iree_file_create_mapped_platform(
    const char* path, uint64_t file_size, uint64_t offset,
    iree_host_size_t length, iree_file_contents_t* contents) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file mapping not supported on this platform");
}

static void iree_file_contents_free_platform(iree_file_contents_t* contents) {}

#endif  // IREE_PLATFORM_*

iree_status_t iree_file_create_mapped(const char* path, uint64_t file_size,
                                      uint64_t offset, iree_host_size_t length,
                                      iree_allocator_t allocator,
                                      iree_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_file_contents_t* contents = NULL;
  iree_allocator_malloc(allocator, sizeof(*contents), (void**)&contents);
  contents->allocator = allocator;

  iree_status_t status = iree_file_create_mapped_platform(
      path, file_size, offset, length, contents);

  if (iree_status_is_ok(status)) {
    *out_contents = contents;
  } else {
    iree_file_contents_free(contents);
  }
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
    size_t ret = fwrite((char*)content.data, content.data_length, 1, file);
    if (ret != 1) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "unable to write file contents of %" PRIhsz
                                " bytes to '%s'",
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
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "file I/O is disabled");
}

iree_allocator_t iree_file_contents_deallocator(
    iree_file_contents_t* contents) {
  return iree_allocator_null();
}

void iree_file_contents_free(iree_file_contents_t* contents) {}

iree_status_t iree_file_read_contents(const char* path,
                                      iree_file_read_flags_t flags,
                                      iree_allocator_t allocator,
                                      iree_file_contents_t** out_contents) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "file I/O is disabled");
}

iree_status_t iree_file_preload_contents(const char* path,
                                         iree_allocator_t allocator,
                                         iree_file_contents_t** out_contents) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "file I/O is disabled");
}

iree_status_t iree_file_map_contents_readonly(
    const char* path, iree_allocator_t allocator,
    iree_file_contents_t** out_contents) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "file I/O is disabled");
}

iree_status_t iree_file_write_contents(const char* path,
                                       iree_const_byte_span_t content) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "file I/O is disabled");
}

iree_status_t iree_stdin_read_contents(iree_allocator_t allocator,
                                       iree_file_contents_t** out_contents) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "file I/O is disabled");
}

#endif  // IREE_FILE_IO_ENABLE
