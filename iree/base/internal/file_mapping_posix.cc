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

#include "iree/base/file_mapping.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <memory>

namespace iree {

namespace {

class FileDescriptor {
 public:
  static StatusOr<std::unique_ptr<FileDescriptor>> OpenRead(std::string path) {
    struct stat buf;
    if (::lstat(path.c_str(), &buf) == -1) {
      return NotFoundErrorBuilder(IREE_LOC)
             << "Unable to stat file " << path << ": " << ::strerror(errno);
    }
    uint64_t file_size = static_cast<size_t>(buf.st_size);

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd == -1) {
      return UnavailableErrorBuilder(IREE_LOC)
             << "Unable to open file " << path << ": " << ::strerror(errno);
    }

    return std::make_unique<FileDescriptor>(std::move(path), fd, file_size);
  }

  FileDescriptor(std::string path, int fd, size_t size)
      : path_(std::move(path)), fd_(fd), size_(size) {}
  ~FileDescriptor() { ::close(fd_); }

  absl::string_view path() const { return path_; }
  int fd() const { return fd_; }
  size_t size() const { return size_; }

 private:
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  std::string path_;
  int fd_;
  size_t size_;
};

class MMapMapping : public FileMapping {
 public:
  MMapMapping(void* data, size_t data_size)
      : FileMapping(
            absl::MakeSpan(reinterpret_cast<uint8_t*>(data), data_size)) {}

  ~MMapMapping() override {
    if (::munmap(const_cast<uint8_t*>(data_.data()), data_.size()) != 0) {
      LOG(WARNING) << "Unable to unmap file: " << strerror(errno);
    }
  }
};

}  // namespace

// static
StatusOr<ref_ptr<FileMapping>> FileMapping::OpenRead(std::string path) {
  IREE_TRACE_SCOPE0("FileMapping::Open");

  // Open the file for reading. Note that we only need to keep it open long
  // enough to map it and we can close the descriptor after that.
  IREE_ASSIGN_OR_RETURN(auto file, FileDescriptor::OpenRead(std::move(path)));

  // Map the file from the file descriptor.
  void* data =
      ::mmap(nullptr, file->size(), PROT_READ, MAP_SHARED, file->fd(), 0);
  if (data == MAP_FAILED) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Mapping failed on file (ensure uncompressed): " << file->path();
  }

  return make_ref<MMapMapping>(data, file->size());
}

}  // namespace iree

#endif  // IREE_PLATFORM_*
