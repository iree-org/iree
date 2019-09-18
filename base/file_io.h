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

#ifndef THIRD_PARTY_MLIR_EDGE_IREE_BASE_FILE_IO_H_
#define THIRD_PARTY_MLIR_EDGE_IREE_BASE_FILE_IO_H_

#include <cstdint>
#include <string>
#include <vector>

#include "third_party/absl/strings/string_view.h"
#include "third_party/mlir_edge/iree/base/status.h"

namespace iree {
namespace file_io {

// Deletes the file at the provided path.
Status DeleteFile(absl::string_view path);

// Moves a file from 'source_path' to 'destination_path'.
//
// This may simply rename the file, but may fall back to a full copy and delete
// of the original if renaming is not possible (for example when moving between
// physical storage locations).
Status MoveFile(absl::string_view source_path,
                absl::string_view destination_path);

// Checks if a file exists at the provided path.
//
// Returns an OK status if the file definitely exists.
// Errors can include PermissionDeniedError, NotFoundError, etc.
Status FileExists(absl::string_view path);

// Joins two paths together.
//
// For example:
//   JoinFilePaths('foo', 'bar') --> 'foo/bar'
//   JoinFilePaths('/foo/', '/bar') --> '/foo/bar'
std::string JoinFilePaths(absl::string_view path1, absl::string_view path2);

// Gets the directory name component of a file path.
absl::string_view FileDirectoryName(absl::string_view path);

// Returns the part of the path after the final "/".
absl::string_view FileBasename(absl::string_view path);

// Returns the part of the basename of path prior to the final ".".
absl::string_view FileStem(absl::string_view path);

// Synchronously reads a file's contents into a string.
StatusOr<std::string> GetFileContents(absl::string_view path);

}  // namespace file_io
}  // namespace iree

#endif  // THIRD_PARTY_MLIR_EDGE_IREE_BASE_FILE_IO_H_
