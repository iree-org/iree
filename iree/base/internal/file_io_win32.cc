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

#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <io.h>

#include <atomic>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "iree/base/internal/file_handle_win32.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/file_path.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

namespace iree {
namespace file_io {

Status FileExists(const std::string& path) {
  IREE_TRACE_SCOPE0("file_io::FileExists");
  DWORD attrs = ::GetFileAttributesA(path.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "unable to find/access file: %s", path.c_str());
  }
  return OkStatus();
}

Status GetFileContents(const std::string& path, std::string* out_contents) {
  IREE_TRACE_SCOPE0("file_io::GetFileContents");
  *out_contents = std::string();
  std::unique_ptr<FileHandle> file;
  IREE_RETURN_IF_ERROR(
      FileHandle::OpenRead(std::move(path), FILE_FLAG_SEQUENTIAL_SCAN, &file));
  std::string contents;
  contents.resize(file->size());
  DWORD bytes_read = 0;
  if (::ReadFile(file->handle(), const_cast<char*>(contents.data()),
                 static_cast<DWORD>(contents.size()), &bytes_read,
                 nullptr) == FALSE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "unable to read file span of %zu bytes from '%s'",
                            contents.size(), path.c_str());
  } else if (bytes_read != file->size()) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "unable to read all %zu bytes from '%.*s' (got %zu)", file->size(),
        (int)path.size(), path.data(), bytes_read);
  }
  *out_contents = contents;
  return OkStatus();
}

Status SetFileContents(const std::string& path, absl::string_view content) {
  IREE_TRACE_SCOPE0("file_io::SetFileContents");
  std::unique_ptr<FileHandle> file;
  IREE_RETURN_IF_ERROR(FileHandle::OpenWrite(std::move(path), 0, &file));
  if (::WriteFile(file->handle(), content.data(),
                  static_cast<DWORD>(content.size()), NULL, NULL) == FALSE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "unable to write file span of %zu bytes to '%s'",
                            content.size(), path.c_str());
  }
  return OkStatus();
}

Status DeleteFile(const std::string& path) {
  IREE_TRACE_SCOPE0("file_io::DeleteFile");
  if (::DeleteFileA(path.c_str()) == FALSE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "unable to delete/access file: %s", path.c_str());
  }
  return OkStatus();
}

Status MoveFile(const std::string& source_path,
                const std::string& destination_path) {
  IREE_TRACE_SCOPE0("file_io::MoveFile");
  if (::MoveFileA(source_path.c_str(), destination_path.c_str()) == FALSE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "unable to move file '%s' to '%s'",
                            source_path.c_str(), destination_path.c_str());
  }
  return OkStatus();
}

std::string GetTempPath() {
  IREE_TRACE_SCOPE0("file_io::GetTempPath");

  // TEST_TMPDIR will point to a writeable temp path when running bazel tests.
  char* test_tmpdir = getenv("TEST_TMPDIR");
  if (test_tmpdir) {
    return test_tmpdir;
  }

  std::string temp_path(64, '\0');
  for (bool retry_query = true; retry_query;) {
    DWORD required_length =
        ::GetTempPathA(static_cast<DWORD>(temp_path.size()), &temp_path[0]);
    retry_query = required_length > temp_path.size();
    temp_path.resize(required_length);
  }
  return temp_path;
}

// TODO(#3845): remove this when dylibs no longer need temp files.
Status GetTempFile(absl::string_view base_name, std::string* out_path) {
  IREE_TRACE_SCOPE0("file_io::GetTempFile");
  *out_path = std::string();

  std::string temp_path = GetTempPath();
  std::string template_path =
      file_path::JoinPaths(temp_path, base_name) + "XXXXXX";

  if (::_mktemp(&template_path[0]) != nullptr) {
    // Should have been modified by _mktemp.
    static std::atomic<int> next_id{0};
    template_path += std::to_string(next_id++);
    *out_path = std::move(template_path);
    return OkStatus();
  } else {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "unable to create temp file with template '%s'",
                            template_path.c_str());
  }
}

}  // namespace file_io
}  // namespace iree

#endif  // IREE_PLATFORM_WINDOWS
