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

#include <cstdio>

#include "absl/strings/str_cat.h"
#include "iree/base/file_io.h"
#include "iree/base/status.h"
#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX)

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace iree {
namespace file_io {

Status FileExists(const std::string& path) {
  struct stat stat_buf;
  return stat(path.c_str(), &stat_buf) == 0
             ? OkStatus()
             : NotFoundErrorBuilder(IREE_LOC) << "'" << path << "'";
}

StatusOr<std::string> GetFileContents(const std::string& path) {
  std::unique_ptr<FILE, void (*)(FILE*)> file = {std::fopen(path.c_str(), "r"),
                                                 +[](FILE* file) {
                                                   if (file) fclose(file);
                                                 }};
  if (file == nullptr) {
    return ErrnoToCanonicalStatusBuilder(
        errno, absl::StrCat("Failed to open file '", path, "'"), IREE_LOC);
  }
  if (std::fseek(file.get(), 0, SEEK_END) == -1) {
    return ErrnoToCanonicalStatusBuilder(
        errno, absl::StrCat("Failed to seek file '", path, "'"), IREE_LOC);
  }
  size_t file_size = std::ftell(file.get());
  if (file_size == -1L) {
    return ErrnoToCanonicalStatusBuilder(
        errno, absl::StrCat("Failed to read file length '", path, "'"),
        IREE_LOC);
  }
  if (std::fseek(file.get(), 0, SEEK_SET) == -1) {
    return ErrnoToCanonicalStatusBuilder(
        errno, absl::StrCat("Failed to seek file '", path, "'"), IREE_LOC);
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

Status SetFileContents(const std::string& path, const std::string& content) {
  std::unique_ptr<FILE, void (*)(FILE*)> file = {std::fopen(path.c_str(), "wb"),
                                                 +[](FILE* file) {
                                                   if (file) fclose(file);
                                                 }};
  if (file == nullptr) {
    return ErrnoToCanonicalStatusBuilder(
        errno, absl::StrCat("Failed to open file '", path, "'"), IREE_LOC);
  }
  if (std::fwrite(const_cast<char*>(content.data()), content.size(), 1,
                  file.get()) != 1) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Unable to write entire file contents of '" << path << "'";
  }
  return OkStatus();
}

Status DeleteFile(const std::string& path) {
  if (::remove(path.c_str()) == -1) {
    return ErrnoToCanonicalStatusBuilder(
        errno, absl::StrCat("Failed to delete file '", path, "'"), IREE_LOC);
  }
  return OkStatus();
}

Status MoveFile(const std::string& source_path,
                const std::string& destination_path) {
  if (::rename(source_path.c_str(), destination_path.c_str()) == -1) {
    return ErrnoToCanonicalStatusBuilder(
        errno,
        absl::StrCat("Failed to rename file '", source_path, "' to '",
                     destination_path, "'"),
        IREE_LOC);
  }
  return OkStatus();
}

}  // namespace file_io
}  // namespace iree

#endif  // IREE_PLATFORM_*
