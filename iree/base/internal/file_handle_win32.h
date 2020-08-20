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

#ifndef IREE_BASE_INTERNAL_FILE_HANDLE_WIN32_H_
#define IREE_BASE_INTERNAL_FILE_HANDLE_WIN32_H_

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "iree/base/status.h"
#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)

namespace iree {

class FileHandle {
 public:
  static StatusOr<std::unique_ptr<FileHandle>> OpenRead(std::string path,
                                                        DWORD file_flags);
  static StatusOr<std::unique_ptr<FileHandle>> OpenWrite(std::string path,
                                                         DWORD file_flags);

  FileHandle(HANDLE handle, size_t size) : handle_(handle), size_(size) {}
  ~FileHandle();

  absl::string_view path() const { return path_; }
  HANDLE handle() const { return handle_; }
  size_t size() const { return size_; }

 private:
  FileHandle(const FileHandle&) = delete;
  FileHandle& operator=(const FileHandle&) = delete;

  std::string path_;
  HANDLE handle_;
  size_t size_;
};

}  // namespace iree

#endif  // IREE_PLATFORM_WINDOWS

#endif  // IREE_BASE_INTERNAL_FILE_HANDLE_WIN32_H_
