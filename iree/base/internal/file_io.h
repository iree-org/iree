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

#ifndef IREE_BASE_INTERNAL_FILE_IO_H_
#define IREE_BASE_INTERNAL_FILE_IO_H_

#include <string>

#include "absl/strings/string_view.h"
#include "iree/base/status.h"

namespace iree {
namespace file_io {

// Checks if a file exists at the provided path.
//
// Returns an OK status if the file definitely exists.
// Errors can include PermissionDeniedError, NotFoundError, etc.
Status FileExists(const std::string& path);

// Synchronously reads a file's contents into a string.
Status GetFileContents(const std::string& path, std::string* out_contents);

// Synchronously writes a string into a file, overwriting its contents.
Status SetFileContents(const std::string& path, absl::string_view content);

// Deletes the file at the provided path.
Status DeleteFile(const std::string& path);

// Moves a file from 'source_path' to 'destination_path'.
//
// This may simply rename the file, but may fall back to a full copy and delete
// of the original if renaming is not possible (for example when moving between
// physical storage locations).
Status MoveFile(const std::string& source_path,
                const std::string& destination_path);

// Gets a platform and environment-dependent path for temporary files.
std::string GetTempPath();

// TODO(#3845): remove this when dylibs no longer need temp files.
// Gets a temporary file name and returns its absolute path.
// The particular path chosen is platform and environment-dependent.
// Unique characters will be automatically inserted after |base_name|.
Status GetTempFile(absl::string_view base_name, std::string* out_path);

}  // namespace file_io
}  // namespace iree

#endif  // IREE_BASE_INTERNAL_FILE_IO_H_
