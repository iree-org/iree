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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "base/file_io.h"
#include "base/internal/file_handle_win32.h"
#include "base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <windows.h>

namespace iree {
namespace file_io {

Status FileExists(const std::string& path) {
  DWORD attrs = ::GetFileAttributesA(path.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to find/access file: " << path;
  }
  return OkStatus();
}

StatusOr<std::string> GetFileContents(const std::string& path) {
  ASSIGN_OR_RETURN(auto file, FileHandle::OpenRead(std::move(path),
                                                   FILE_FLAG_SEQUENTIAL_SCAN));
  std::string result;
  result.resize(file->size());
  DWORD bytes_read = 0;
  if (::ReadFile(file->handle(), const_cast<char*>(result.data()),
                 result.size(), &bytes_read, nullptr) == FALSE) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to read file span of " << result.size() << " bytes";
  } else if (bytes_read != file->size()) {
    return ResourceExhaustedErrorBuilder(IREE_LOC)
           << "Unable to read all " << file->size()
           << " bytes from the file (got " << bytes_read << ")";
  }
  return result;
}

Status DeleteFile(const std::string& path) {
  if (::DeleteFileA(path.c_str()) == FALSE) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to delete/access file: " << path;
  }
  return OkStatus();
}

Status MoveFile(const std::string& source_path,
                const std::string& destination_path) {
  if (::MoveFileA(source_path.c_str(), destination_path.c_str()) == FALSE) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to move file " << source_path << " to "
           << destination_path;
  }
  return OkStatus();
}

}  // namespace file_io
}  // namespace iree

#endif  // IREE_PLATFORM_WINDOWS
