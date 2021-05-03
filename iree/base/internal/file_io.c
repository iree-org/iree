// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/base/internal/file_io.h"

#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

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

static iree_status_t iree_file_read_contents_impl(
    FILE* file, iree_allocator_t allocator, iree_byte_span_t* out_contents) {
  // Seek to the end of the file.
  if (fseek(file, 0, SEEK_END) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno), "seek (end)");
  }

  // Query the position, telling us the total file length in bytes.
  size_t file_size = ftell(file);
  if (file_size == -1L) {
    return iree_make_status(iree_status_code_from_errno(errno), "size query");
  }

  // Seek back to the file start.
  if (fseek(file, 0, SEEK_SET) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno), "seek (beg)");
  }

  // Allocate +1 to force a trailing \0 in case this is a string.
  char* contents = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, file_size + 1, (void**)&contents));

  // Attempt to read the file into memory.
  if (fread(contents, file_size, 1, file) != 1) {
    iree_allocator_free(allocator, contents);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "unable to read entire %zu file bytes", file_size);
  }

  // Add trailing NUL to make the contents C-string compatible.
  contents[file_size] = 0;  // NUL
  *out_contents = iree_make_byte_span(contents, file_size);
  return iree_ok_status();
}

iree_status_t iree_file_read_contents(const char* path,
                                      iree_allocator_t allocator,
                                      iree_byte_span_t* out_contents) {
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(out_contents);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_contents = iree_make_byte_span(NULL, 0);

  FILE* file = fopen(path, "rb");
  if (file == NULL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%s'", path);
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
  IREE_ASSERT_ARGUMENT(path);
  IREE_TRACE_ZONE_BEGIN(z0);

  FILE* file = fopen(path, "wb");
  if (file == NULL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%s'", path);
  }

  int ret = fwrite((char*)content.data, content.data_length, 1, file);
  iree_status_t status = iree_ok_status();
  if (ret != 1) {
    status =
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "unable to write file contents of %zu bytes to '%s'",
                         content.data_length, path);
  }

  fclose(file);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
