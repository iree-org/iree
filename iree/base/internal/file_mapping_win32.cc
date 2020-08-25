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
#include "iree/base/file_mapping.h"
#include "iree/base/internal/file_handle_win32.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

#if defined(IREE_PLATFORM_WINDOWS)

namespace iree {

namespace {

class Win32FileMapping : public FileMapping {
 public:
  Win32FileMapping(HANDLE mapping_handle, void* data, size_t data_size)
      : FileMapping(
            absl::MakeSpan(reinterpret_cast<uint8_t*>(data), data_size)),
        mapping_handle_(mapping_handle) {}

  ~Win32FileMapping() override {
    if (!data_.empty()) {
      if (::UnmapViewOfFile(data_.data()) == FALSE) {
        LOG(WARNING) << "Unable to unmap file: " << GetLastError();
      }
      data_ = {};
    }
    if (mapping_handle_) {
      ::CloseHandle(mapping_handle_);
      mapping_handle_ = nullptr;
    }
  }

 private:
  HANDLE mapping_handle_;
};

}  // namespace

// static
StatusOr<ref_ptr<FileMapping>> FileMapping::OpenRead(std::string path) {
  IREE_TRACE_SCOPE0("FileMapping::Open");

  // Open the file for reading. Note that we only need to keep it open long
  // enough to map it and we can close the descriptor after that.
  IREE_ASSIGN_OR_RETURN(
      auto file,
      FileHandle::OpenRead(std::move(path), FILE_FLAG_RANDOM_ACCESS));

  HANDLE mapping_handle = ::CreateFileMappingA(
      /*hFile=*/file->handle(), /*lpFileMappingAttributes=*/nullptr,
      /*flProtect=*/PAGE_READONLY, /*dwMaximumSizeHigh=*/0,
      /*dwMaximumSizeLow=*/0, /*lpName=*/nullptr);
  if (!mapping_handle) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Failed to create mapping on file (ensure uncompressed): "
           << file->path();
  }

  void* data =
      ::MapViewOfFileEx(/*hFileMappingObject=*/mapping_handle,
                        /*dwDesiredAccess=*/FILE_MAP_READ,
                        /*dwFileOffsetHigh=*/0, /*dwFileOffsetLow=*/0,
                        /*dwNumberOfBytesToMap=*/0, /*lpBaseAddress=*/nullptr);
  if (!data) {
    DWORD map_view_error = GetLastError();
    ::CloseHandle(mapping_handle);
    return Win32ErrorToCanonicalStatusBuilder(map_view_error, IREE_LOC)
           << "Failed to map view of file: " << file->path();
  }

  auto result = make_ref<Win32FileMapping>(mapping_handle, data, file->size());

  // NOTE: file mappings hold references to the file, so we don't need to keep
  // the file around any longer than this function.
  file.reset();

  return result;
}

}  // namespace iree

#endif  // IREE_PLATFORM_WINDOWS
