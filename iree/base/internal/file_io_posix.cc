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
  return stat(path.c_str(), &stat_buf) == 0 ? OkStatus()
                                            : NotFoundErrorBuilder(IREE_LOC);
}

StatusOr<std::string> GetFileContents(const std::string& path) {
  std::unique_ptr<FILE, void (*)(FILE*)> file = {std::fopen(path.c_str(), "r"),
                                                 +[](FILE* file) {
                                                   if (file) fclose(file);
                                                 }};
  if (file == nullptr) {
    return ErrnoToCanonicalStatusBuilder(errno, "Failed to open file",
                                         IREE_LOC);
  }
  if (std::fseek(file.get(), 0, SEEK_END) == -1) {
    return ErrnoToCanonicalStatusBuilder(errno, "Failed to seek file",
                                         IREE_LOC);
  }
  size_t file_size = std::ftell(file.get());
  if (file_size == -1L) {
    return ErrnoToCanonicalStatusBuilder(errno, "Failed to read file length",
                                         IREE_LOC);
  }
  if (std::fseek(file.get(), 0, SEEK_SET) == -1) {
    return ErrnoToCanonicalStatusBuilder(errno, "Failed to seek file",
                                         IREE_LOC);
  }
  std::string contents;
  contents.resize(file_size);
  if (std::fread(const_cast<char*>(contents.data()), file_size, 1,
                 file.get()) != file_size) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Unable to read entire file contents";
  }
  return contents;
}

Status DeleteFile(const std::string& path) {
  if (::remove(path.c_str()) == -1) {
    return ErrnoToCanonicalStatusBuilder(errno, "Failed to delete file",
                                         IREE_LOC);
  }
  return OkStatus();
}

Status MoveFile(const std::string& source_path,
                const std::string& destination_path) {
  if (::rename(source_path.c_str(), destination_path.c_str()) == -1) {
    return ErrnoToCanonicalStatusBuilder(errno, "Failed to rename file",
                                         IREE_LOC);
  }
  return OkStatus();
}

}  // namespace file_io
}  // namespace iree

#endif  // IREE_PLATFORM_*
