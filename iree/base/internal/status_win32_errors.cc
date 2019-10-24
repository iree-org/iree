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

#include "iree/base/internal/status_win32_errors.h"

#include "absl/strings/str_cat.h"
#include "iree/base/platform_headers.h"

#if defined(IREE_PLATFORM_WINDOWS)

namespace iree {

StatusCode Win32ErrorToCanonicalCode(uint32_t error) {
  switch (error) {
    case ERROR_SUCCESS:
      return StatusCode::kOk;
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      return StatusCode::kNotFound;
    case ERROR_TOO_MANY_OPEN_FILES:
    case ERROR_OUTOFMEMORY:
    case ERROR_HANDLE_DISK_FULL:
    case ERROR_HANDLE_EOF:
      return StatusCode::kResourceExhausted;
    case ERROR_ACCESS_DENIED:
      return StatusCode::kPermissionDenied;
    case ERROR_INVALID_HANDLE:
      return StatusCode::kInvalidArgument;
    case ERROR_NOT_READY:
    case ERROR_READ_FAULT:
      return StatusCode::kUnavailable;
    case ERROR_WRITE_FAULT:
      return StatusCode::kDataLoss;
    case ERROR_NOT_SUPPORTED:
      return StatusCode::kUnimplemented;
    default:
      return StatusCode::kUnknown;
  }
}

StatusBuilder Win32ErrorToCanonicalStatusBuilder(uint32_t error,
                                                 SourceLocation location) {
  // TODO(benvanik): use FormatMessage; or defer until required?
  return StatusBuilder(
      Status(Win32ErrorToCanonicalCode(error), absl::StrCat("<TBD>", error)),
      location);
}

}  // namespace iree

#endif  // IREE_PLATFORM_WINDOWS
