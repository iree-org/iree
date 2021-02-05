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
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/file_path.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

namespace iree {
namespace file_io {

Status FileExists(const std::string& path) {
  IREE_TRACE_SCOPE0("file_io::FileExists");
  struct stat stat_buf;
  return stat(path.c_str(), &stat_buf) == 0
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_NOT_FOUND, "'%s'", path.c_str());
}

Status GetFileContents(const std::string& path, std::string* out_contents) {
  IREE_TRACE_SCOPE0("file_io::GetFileContents");
  *out_contents = std::string();
  std::unique_ptr<FILE, void (*)(FILE*)> file = {std::fopen(path.c_str(), "r"),
                                                 +[](FILE* file) {
                                                   if (file) fclose(file);
                                                 }};
  if (file == nullptr) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%s'", path.c_str());
  }
  if (std::fseek(file.get(), 0, SEEK_END) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno), "seek (end)");
  }
  size_t file_size = std::ftell(file.get());
  if (file_size == -1L) {
    return iree_make_status(iree_status_code_from_errno(errno), "size query");
  }
  if (std::fseek(file.get(), 0, SEEK_SET) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno), "seek (beg)");
  }
  std::string contents;
  contents.resize(file_size);
  if (std::fread(const_cast<char*>(contents.data()), file_size, 1,
                 file.get()) != 1) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "unable to read entire file contents of '%.*s'",
                            (int)path.size(), path.data());
  }
  *out_contents = std::move(contents);
  return OkStatus();
}

Status SetFileContents(const std::string& path, absl::string_view content) {
  IREE_TRACE_SCOPE0("file_io::SetFileContents");
  std::unique_ptr<FILE, void (*)(FILE*)> file = {std::fopen(path.c_str(), "wb"),
                                                 +[](FILE* file) {
                                                   if (file) fclose(file);
                                                 }};
  if (file == nullptr) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%s'", path.c_str());
  }
  if (std::fwrite(const_cast<char*>(content.data()), content.size(), 1,
                  file.get()) != 1) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "unable to write entire file contents of '%.*s'",
                            (int)path.size(), path.data());
  }
  return OkStatus();
}

Status DeleteFile(const std::string& path) {
  IREE_TRACE_SCOPE0("file_io::DeleteFile");
  if (::remove(path.c_str()) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to delete file '%s'", path.c_str());
  }
  return OkStatus();
}

Status MoveFile(const std::string& source_path,
                const std::string& destination_path) {
  IREE_TRACE_SCOPE0("file_io::MoveFile");
  if (::rename(source_path.c_str(), destination_path.c_str()) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to rename file '%s' to '%s'",
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

  char* tmpdir = getenv("TMPDIR");
  if (tmpdir) {
    return tmpdir;
  }

#ifdef __ANDROID__
  // Support running Android command-line programs both as regular shell user
  // and as root. For the latter, TMPDIR is not defined by default.
  return "/data/local/tmp";
#else
  return "/tmp";
#endif
}

// TODO(#3845): remove this when dylibs no longer need temp files.
Status GetTempFile(absl::string_view base_name, std::string* out_path) {
  IREE_TRACE_SCOPE0("file_io::GetTempFile");
  *out_path = std::string();

  std::string temp_path = GetTempPath();
  std::string template_path =
      file_path::JoinPaths(temp_path, base_name) + "XXXXXX";

  if (::mkstemp(&template_path[0]) != -1) {
    // Should have been modified by mkstemp.
    *out_path = std::move(template_path);
    return OkStatus();
  } else {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to create temp file with template '%s'",
                            template_path.c_str());
  }
}

}  // namespace file_io
}  // namespace iree

#endif  // IREE_PLATFORM_*
