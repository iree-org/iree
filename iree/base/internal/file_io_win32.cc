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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "iree/base/file_io.h"
#include "iree/base/file_path.h"
#include "iree/base/internal/file_handle_win32.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

namespace iree {
namespace file_io {

Status FileExists(const std::string& path) {
  IREE_TRACE_SCOPE0("file_io::FileExists");
  DWORD attrs = ::GetFileAttributesA(path.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to find/access file: " << path;
  }
  return OkStatus();
}

StatusOr<std::string> GetFileContents(const std::string& path) {
  IREE_TRACE_SCOPE0("file_io::GetFileContents");
  IREE_ASSIGN_OR_RETURN(
      auto file,
      FileHandle::OpenRead(std::move(path), FILE_FLAG_SEQUENTIAL_SCAN));
  std::string result;
  result.resize(file->size());
  DWORD bytes_read = 0;
  if (::ReadFile(file->handle(), const_cast<char*>(result.data()),
                 result.size(), &bytes_read, nullptr) == FALSE) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to read file span of " << result.size() << " bytes from '"
           << path << "'";
  } else if (bytes_read != file->size()) {
    return ResourceExhaustedErrorBuilder(IREE_LOC)
           << "Unable to read all " << file->size() << " bytes from '" << path
           << "' (got " << bytes_read << ")";
  }
  return result;
}

Status SetFileContents(const std::string& path, absl::string_view content) {
  IREE_TRACE_SCOPE0("file_io::SetFileContents");
  IREE_ASSIGN_OR_RETURN(auto file, FileHandle::OpenWrite(std::move(path), 0));
  if (::WriteFile(file->handle(), content.data(), content.size(), NULL, NULL) ==
      FALSE) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to write file span of " << content.size() << " bytes to '"
           << path << "'";
  }
  return OkStatus();
}

Status DeleteFile(const std::string& path) {
  IREE_TRACE_SCOPE0("file_io::DeleteFile");
  if (::DeleteFileA(path.c_str()) == FALSE) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to delete/access file: " << path;
  }
  return OkStatus();
}

Status MoveFile(const std::string& source_path,
                const std::string& destination_path) {
  IREE_TRACE_SCOPE0("file_io::MoveFile");
  if (::MoveFileA(source_path.c_str(), destination_path.c_str()) == FALSE) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to move file " << source_path << " to "
           << destination_path;
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
    DWORD required_length = ::GetTempPathA(temp_path.size(), &temp_path[0]);
    retry_query = required_length > temp_path.size();
    temp_path.resize(required_length);
  }
  return temp_path;
}

StatusOr<std::string> GetTempFile(absl::string_view base_name) {
  IREE_TRACE_SCOPE0("file_io::GetTempFile");

  std::string temp_path = GetTempPath();
  std::string template_path =
      file_path::JoinPaths(temp_path, base_name) + "XXXXXX";

  if (::_mktemp(&template_path[0]) != nullptr) {
    return template_path;  // Should have been modified by _mktemp.
  } else {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to create temp file with template " << template_path;
  }
}

}  // namespace file_io
}  // namespace iree

#endif  // IREE_PLATFORM_WINDOWS
