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

namespace iree {
namespace file_io {

iree_status_t FileExists(const char* path) {
  IREE_TRACE_ZONE_BEGIN(z0);
  struct stat stat_buf;
  iree_status_t status =
      stat(path, &stat_buf) == 0
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_NOT_FOUND, "'%s'", path);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t GetFileContents(const char* path, std::string* out_contents) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_contents = std::string();
  FILE* file = fopen(path, "rb");
  if (file == NULL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%s'", path);
  }
  iree_status_t status = iree_ok_status();
  if (fseek(file, 0, SEEK_END) == -1) {
    status = iree_make_status(iree_status_code_from_errno(errno), "seek (end)");
  }
  size_t file_size = 0;
  if (iree_status_is_ok(status)) {
    file_size = ftell(file);
    if (file_size == -1L) {
      status =
          iree_make_status(iree_status_code_from_errno(errno), "size query");
    }
  }
  if (iree_status_is_ok(status)) {
    if (fseek(file, 0, SEEK_SET) == -1) {
      status =
          iree_make_status(iree_status_code_from_errno(errno), "seek (beg)");
    }
  }
  std::string contents;
  if (iree_status_is_ok(status)) {
    contents.resize(file_size);
    if (fread((char*)contents.data(), file_size, 1, file) != 1) {
      status =
          iree_make_status(iree_status_code_from_errno(errno),
                           "unable to read entire file contents of '%s'", path);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_contents = std::move(contents);
  }
  fclose(file);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t SetFileContents(const char* path,
                              iree_const_byte_span_t content) {
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
                         "unable to write entire file contents of '%s'", path);
  }
  fclose(file);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

}  // namespace file_io
}  // namespace iree
