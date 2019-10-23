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

#ifndef IREE_BASE_FILE_MAPPING_H_
#define IREE_BASE_FILE_MAPPING_H_

#include <cstdint>
#include <string>

#include "absl/types/span.h"
#include "base/ref_ptr.h"
#include "base/status.h"

namespace iree {

// A memory-mapped file handle.
class FileMapping : public RefObject<FileMapping> {
 public:
  // Opens a file and maps it into the calling process memory.
  // The file will be opened for shared read access.
  static StatusOr<ref_ptr<FileMapping>> OpenRead(std::string path);

  virtual ~FileMapping() = default;

  // Read-only contents of the file.
  inline absl::Span<const uint8_t> data() const noexcept { return data_; }

 protected:
  explicit FileMapping(absl::Span<const uint8_t> data) : data_(data) {}

  absl::Span<const uint8_t> data_;

 private:
  FileMapping(const FileMapping&) = delete;
  FileMapping& operator=(const FileMapping&) = delete;
};

}  // namespace iree

#endif  // IREE_BASE_FILE_MAPPING_H_
