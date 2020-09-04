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

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX)

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "absl/strings/str_cat.h"
#include "iree/base/file_io.h"
#include "iree/base/file_path.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

namespace iree {
namespace file_io {

Status FileExists(const std::string& path) {
  IREE_TRACE_SCOPE0("file_io::FileExists");
  struct stat stat_buf;
  return stat(path.c_str(), &stat_buf) == 0
             ? OkStatus()
             : NotFoundErrorBuilder(IREE_LOC) << "'" << path << "'";
}

StatusOr<std::string> GetFileContents(const std::string& path) {
  IREE_TRACE_SCOPE0("file_io::GetFileContents");
  std::unique_ptr<FILE, void (*)(FILE*)> file = {std::fopen(path.c_str(), "r"),
                                                 +[](FILE* file) {
                                                   if (file) fclose(file);
                                                 }};
  if (file == nullptr) {
    return ErrnoToCanonicalStatusBuilder(errno, IREE_LOC)
           << "Failed to open file '" << path << "'";
  }
  if (std::fseek(file.get(), 0, SEEK_END) == -1) {
    return ErrnoToCanonicalStatusBuilder(errno, IREE_LOC)
           << "Failed to seek file '" << path << "'";
  }
  size_t file_size = std::ftell(file.get());
  if (file_size == -1L) {
    return ErrnoToCanonicalStatusBuilder(errno, IREE_LOC)
           << "Failed to read file length for '" << path << "'";
  }
  if (std::fseek(file.get(), 0, SEEK_SET) == -1) {
    return ErrnoToCanonicalStatusBuilder(errno, IREE_LOC)
           << "Failed to seek back in file '" << path << "'";
  }
  std::string contents;
  contents.resize(file_size);
  if (std::fread(const_cast<char*>(contents.data()), file_size, 1,
                 file.get()) != 1) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Unable to read entire file contents of '" << path << "'";
  }
  return contents;
}

Status SetFileContents(const std::string& path, absl::string_view content) {
  IREE_TRACE_SCOPE0("file_io::SetFileContents");
  std::unique_ptr<FILE, void (*)(FILE*)> file = {std::fopen(path.c_str(), "wb"),
                                                 +[](FILE* file) {
                                                   if (file) fclose(file);
                                                 }};
  if (file == nullptr) {
    return ErrnoToCanonicalStatusBuilder(errno, IREE_LOC)
           << "Failed to open file '" << path << "'";
  }
  if (std::fwrite(const_cast<char*>(content.data()), content.size(), 1,
                  file.get()) != 1) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Unable to write entire file contents of '" << path << "'";
  }
  return OkStatus();
}

Status DeleteFile(const std::string& path) {
  IREE_TRACE_SCOPE0("file_io::DeleteFile");
  if (::remove(path.c_str()) == -1) {
    return ErrnoToCanonicalStatusBuilder(errno, IREE_LOC)
           << "Failed to delete file '" << path << "'";
  }
  return OkStatus();
}

Status MoveFile(const std::string& source_path,
                const std::string& destination_path) {
  IREE_TRACE_SCOPE0("file_io::MoveFile");
  if (::rename(source_path.c_str(), destination_path.c_str()) == -1) {
    return ErrnoToCanonicalStatusBuilder(errno, IREE_LOC)
           << "Failed to rename file '" << source_path << "' to '"
           << destination_path << "'";
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

  char* tmpdir = getenv("TMPDIR");
  if (tmpdir) {
    return tmpdir;
  }

  return "/tmp";
}

StatusOr<std::string> GetTempFile(absl::string_view base_name) {
  IREE_TRACE_SCOPE0("file_io::GetTempFile");

  std::string temp_path = GetTempPath();
  std::string template_path =
      file_path::JoinPaths(temp_path, base_name) + "XXXXXX";

  if (::mkstemp(&template_path[0]) != -1) {
    return template_path;  // Should have been modified by mkstemp.
  } else {
    return ErrnoToCanonicalStatusBuilder(errno, IREE_LOC)
           << "Failed to create temp file with template '" << template_path
           << "'";
  }
}

}  // namespace file_io
}  // namespace iree

#endif  // IREE_PLATFORM_*
