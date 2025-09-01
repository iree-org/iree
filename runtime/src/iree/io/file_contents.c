// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/file_contents.h"

#if IREE_FILE_IO_ENABLE
#include "iree/io/stdio_util.h"
#endif  // IREE_FILE_IO_ENABLE

//===----------------------------------------------------------------------===//
// iree_io_file_contents_t
//===----------------------------------------------------------------------===//

// We could take alignment as an arg, but roughly page aligned should be
// acceptable for all uses - if someone cares about memory usage they won't
// be using preloading.
#define IREE_IO_FILE_CONTENTS_BASE_ALIGNMENT 4096

IREE_API_EXPORT void iree_io_file_contents_free(
    iree_io_file_contents_t* contents) {
  if (!contents) return;
  iree_allocator_t allocator = contents->allocator;
  if (contents->mapping) {
    iree_io_file_mapping_release(contents->mapping);
  }
  iree_allocator_free(allocator, contents);
}

static iree_status_t iree_io_file_contents_allocator_ctl(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_ptr) {
  if (command != IREE_ALLOCATOR_COMMAND_FREE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "file contents deallocator must only be used to "
                            "deallocate file contents");
  }
  iree_io_file_contents_t* contents = (iree_io_file_contents_t*)self;
  if (contents->buffer.data != *inout_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only the file contents buffer is valid");
  }
  iree_io_file_contents_free(contents);
  return iree_ok_status();
}

IREE_API_EXPORT iree_allocator_t
iree_io_file_contents_deallocator(iree_io_file_contents_t* contents) {
  iree_allocator_t allocator = {
      .self = contents,
      .ctl = iree_io_file_contents_allocator_ctl,
  };
  return allocator;
}

#if IREE_FILE_IO_ENABLE

IREE_API_EXPORT iree_status_t iree_io_file_contents_read_stdin(
    iree_allocator_t host_allocator, iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Fix stdin mode to binary on Windows to match Unix behavior.
  IREE_IO_SET_BINARY_MODE(stdin);

  iree_host_size_t capacity = 4096;
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator,
      sizeof(*contents) + IREE_IO_FILE_CONTENTS_BASE_ALIGNMENT + capacity,
      (void**)&contents));
  contents->buffer.data =
      (void*)iree_host_align((uintptr_t)contents + sizeof(*contents),
                             IREE_IO_FILE_CONTENTS_BASE_ALIGNMENT);

  iree_host_size_t size = 0;
  for (int c = getchar(); c != EOF; c = getchar()) {
    if (size >= capacity - /*NUL*/ 1) {
      // NOTE: if we realloc we may end up with a new alignment and need to move
      // the data around.
      uintptr_t old_offset =
          (uintptr_t)contents->buffer.data - (uintptr_t)contents;
      iree_host_size_t new_capacity = capacity * 2;
      iree_io_file_contents_t* new_contents = contents;
      iree_status_t status = iree_allocator_realloc(
          host_allocator,
          sizeof(*new_contents) + IREE_IO_FILE_CONTENTS_BASE_ALIGNMENT +
              new_capacity,
          (void**)&new_contents);
      if (!iree_status_is_ok(status)) {
        iree_allocator_free(host_allocator, contents);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
      contents = new_contents;
      uint8_t* old_data = (uint8_t*)new_contents + old_offset;
      uint8_t* new_data = (uint8_t*)iree_host_align(
          (uintptr_t)new_contents + sizeof(*new_contents),
          IREE_IO_FILE_CONTENTS_BASE_ALIGNMENT);
      if (new_data != old_data) {
        // Alignment changed; move the data with safety for overlapping.
        memmove(new_data, old_data, size);
      }
      contents->buffer.data = new_data;
      capacity = new_capacity;
    }
    contents->buffer.data[size++] = c;
  }

  contents->allocator = host_allocator;
  contents->buffer.data[size] = 0;  // NUL
  contents->buffer.data_length = size;
  *out_contents = contents;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Opens the file |handle| if it is file descriptor-based with the |mode| as
// defined by the fdopen API. The returned FILE* has separate lifetime from the
// |handle| and must be closed by the caller. Operations on the returned file
// may not be coherent with operations made via any other mechanism due to
// buffering in stdio.
static iree_status_t iree_io_file_handle_fdopen(iree_io_file_handle_t* handle,
                                                const char* mode,
                                                FILE** out_file) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(mode);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure the provided handle is fd-based.
  if (iree_io_file_handle_type(handle) != IREE_IO_FILE_HANDLE_TYPE_FD) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "the provided file handle is not file descriptor-based");
  }
  int fd = iree_io_file_handle_primitive(handle).value.fd;

  // Duplicate the file descriptor so that we have our own copy of the seek
  // position. The initial position will be preserved.
  int dup_fd = iree_dup(fd);
  if (dup_fd == -1) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_stdio_status(
        "unable to duplicate file descriptor; possibly out of file descriptors "
        "(see ulimit)");
  }

  // NOTE: after this point the file handle is associated with dup_fd and
  // anything we do to it (like closing) will apply to the dup_fd.
  iree_status_t status = iree_ok_status();
  FILE* file = fdopen(dup_fd, mode);
  if (file == NULL) {
    status = iree_make_stdio_statusf(
        "unable to open file descriptor with mode %s", mode);
  }

  if (iree_status_is_ok(status)) {
    *out_file = file;
  } else {
    if (file) {
      // NOTE: closes the dup_fd.
      fclose(file);
    } else if (dup_fd > 0) {
      iree_close(dup_fd);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Returns the size, in bytes, of the |file|. Must be seekable.
static uint64_t iree_io_stdio_file_length(FILE* file) {
  IREE_ASSERT_ARGUMENT(file);

  // Capture original offset so we can return to it.
  int64_t origin = iree_ftell(file);
  if (origin == -1) return 0;

  // Seek to the end of the file.
  if (iree_fseek(file, 0, SEEK_END) != 0) return 0;

  // Query the position, telling us the total file length in bytes.
  int64_t length = iree_ftell(file);
  if (length == -1) return 0;

  // Seek back to the file origin.
  if (iree_fseek(file, origin, SEEK_SET) != 0) return 0;

  return (uint64_t)length;
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_read(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Open the file for reading.
  iree_io_file_handle_t* handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_file_handle_open(IREE_IO_FILE_MODE_READ, path, host_allocator,
                                   &handle));

  // Get an stdio FILE* for the handle.
  // This will need to be closed as its lifetime is separate from our file
  // handle.
  FILE* file = NULL;
  iree_status_t status = iree_io_file_handle_fdopen(handle, "rb", &file);

  // Query the file size.
  iree_host_size_t file_size = 0;
  if (iree_status_is_ok(status)) {
    uint64_t file_size_u64 = iree_io_stdio_file_length(file);
    if (file_size_u64 < IREE_HOST_SIZE_MAX) {
      file_size = (iree_host_size_t)file_size_u64;
    } else {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "file size %" PRIu64
                                " exceeds host size maximum of %" PRIhsz
                                " bytes and cannot be read into memory",
                                file_size_u64, IREE_HOST_SIZE_MAX);
    }
  }

  // Allocate the file contents struct and its embedded buffer data.
  // We compute total size with alignment padding and allocate +1 bytes to force
  // a trailing \0 in case this is used as a cstring.
  iree_io_file_contents_t* contents = NULL;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t total_size = sizeof(*contents) +
                                        IREE_IO_FILE_CONTENTS_BASE_ALIGNMENT +
                                        file_size + /*NUL*/ 1;
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&contents);
  }
  if (iree_status_is_ok(status)) {
    contents->allocator = host_allocator;
    contents->buffer = iree_make_byte_span(
        (void*)iree_host_align((uintptr_t)contents + sizeof(*contents),
                               IREE_IO_FILE_CONTENTS_BASE_ALIGNMENT),
        file_size);

    // Add trailing NUL to make the contents C-string compatible.
    contents->buffer.data[file_size] = 0;  // NUL
  }

  // Attempt to read the file into memory, chunking into ~2GB segments.
  // Several implementations of read have limits at ~2GB even on 64-bit systems.
  iree_host_size_t bytes_read = 0;
  while (bytes_read < file_size) {
    const iree_host_size_t chunk_size =
        iree_min(file_size - bytes_read, INT_MAX);
    if (fread(contents->buffer.data + bytes_read, 1, chunk_size, file) !=
        chunk_size) {
      status = iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                                "unable to read %" PRIhsz " chunk bytes",
                                chunk_size);
      break;
    }
    bytes_read += chunk_size;
  }

  if (file) {
    fclose(file);
  }
  iree_io_file_handle_release(handle);

  if (iree_status_is_ok(status)) {
    *out_contents = contents;
  } else {
    iree_io_file_contents_free(contents);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_map(
    iree_string_view_t path, iree_io_file_access_t access,
    iree_allocator_t host_allocator, iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_file_mode_t mode = IREE_IO_FILE_MODE_READ;
  if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_WRITE)) {
    mode |= IREE_IO_FILE_MODE_WRITE;
  }

  iree_io_file_handle_t* handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_file_handle_open(mode, path, host_allocator, &handle));

  iree_io_file_contents_t* contents = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*contents), (void**)&contents);

  if (iree_status_is_ok(status)) {
    status =
        iree_io_file_map_view(handle, access, 0, IREE_HOST_SIZE_MAX,
                              IREE_IO_FILE_MAPPING_FLAG_PRIVATE |
                                  IREE_IO_FILE_MAPPING_FLAG_EXCLUDE_FROM_DUMPS,
                              host_allocator, &contents->mapping);
  }

  iree_io_file_handle_release(handle);

  if (iree_status_is_ok(status)) {
    contents->allocator = host_allocator;
    if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_WRITE)) {
      contents->buffer = iree_io_file_mapping_contents_rw(contents->mapping);
    } else {
      contents->const_buffer =
          iree_io_file_mapping_contents_ro(contents->mapping);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_contents = contents;
  } else {
    iree_io_file_contents_free(contents);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_write(
    iree_string_view_t path, iree_const_byte_span_t contents,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Open file for overwriting (if it exists).
  iree_io_file_handle_t* handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_file_handle_open(
              IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_OVERWRITE, path,
              host_allocator, &handle));

  // Get an stdio FILE* handle for the opened file.
  // This will need to be closed as its lifetime is separate from our file
  // handle.
  FILE* file = NULL;
  iree_status_t status = iree_io_file_handle_fdopen(handle, "wb", &file);

  // Write all contents (if any) to the file.
  if (iree_status_is_ok(status) && contents.data_length > 0) {
    size_t ret = fwrite((char*)contents.data, contents.data_length, 1, file);
    if (ret != 1) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "unable to write file contents of %" PRIhsz " bytes to '%.*s'",
          contents.data_length, (int)path.size, path.data);
    }
  }

  // Close the stdio handle as well as our file handle.
  if (file) {
    fclose(file);
  }
  iree_io_file_handle_release(handle);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#else

IREE_API_EXPORT iree_status_t iree_io_file_contents_read_stdin(
    iree_allocator_t host_allocator, iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_read(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_map(
    iree_string_view_t path, iree_io_file_access_t access,
    iree_allocator_t host_allocator, iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

IREE_API_EXPORT iree_status_t iree_io_file_contents_write(
    iree_string_view_t path, iree_const_byte_span_t contents,
    iree_allocator_t host_allocator) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "file support has been compiled out of this binary; "
                          "set IREE_FILE_IO_ENABLE=1 to include it");
}

#endif  // IREE_FILE_IO_ENABLE
