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

#include "iree/base/internal/file_handle_win32.h"

#include "absl/memory/memory.h"
#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)

namespace iree {

// static
StatusOr<std::unique_ptr<FileHandle>> FileHandle::OpenRead(std::string path,
                                                           DWORD file_flags) {
  HANDLE handle = ::CreateFileA(
      /*lpFileName=*/path.c_str(), /*dwDesiredAccess=*/GENERIC_READ,
      /*dwShareMode=*/FILE_SHARE_READ, /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/OPEN_EXISTING,
      /*dwFlagsAndAttributes=*/FILE_ATTRIBUTE_NORMAL | file_flags,
      /*hTemplateFile=*/nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to open file " << path;
  }

  BY_HANDLE_FILE_INFORMATION file_info;
  if (::GetFileInformationByHandle(handle, &file_info) == FALSE) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to query file info for " << path;
  }

  uint64_t file_size = (static_cast<uint64_t>(file_info.nFileSizeHigh) << 32) |
                       file_info.nFileSizeLow;
  return absl::make_unique<FileHandle>(handle, file_size);
}

// static
StatusOr<std::unique_ptr<FileHandle>> FileHandle::OpenWrite(std::string path,
                                                            DWORD file_flags) {
  HANDLE handle = ::CreateFileA(
      /*lpFileName=*/path.c_str(), /*dwDesiredAccess=*/GENERIC_WRITE,
      /*dwShareMode=*/FILE_SHARE_DELETE, /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/CREATE_ALWAYS,
      /*dwFlagsAndAttributes=*/FILE_ATTRIBUTE_NORMAL | file_flags,
      /*hTemplateFile=*/nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return Win32ErrorToCanonicalStatusBuilder(GetLastError(), IREE_LOC)
           << "Unable to open file " << path;
  }
  return absl::make_unique<FileHandle>(handle, 0);
}

FileHandle::~FileHandle() { ::CloseHandle(handle_); }

}  // namespace iree

#endif  // IREE_PLATFORM_WINDOWS
