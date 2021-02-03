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

#ifndef IREE_BASE_INTERNAL_FILE_PATH_H_
#define IREE_BASE_INTERNAL_FILE_PATH_H_

#include <string>

#include "absl/strings/string_view.h"

namespace iree {
namespace file_path {

// Joins two paths together.
//
// For example:
//   JoinFilePaths('foo', 'bar') --> 'foo/bar'
//   JoinFilePaths('/foo/', '/bar') --> '/foo/bar'
std::string JoinPaths(absl::string_view path1, absl::string_view path2);

// Gets the directory name component of a file path.
absl::string_view DirectoryName(absl::string_view path);

// Returns the part of the path after the final "/".
absl::string_view Basename(absl::string_view path);

// Returns the part of the basename of path prior to the final ".".
absl::string_view Stem(absl::string_view path);

// Returns the part of the basename of path after to the final ".".
absl::string_view Extension(absl::string_view path);

}  // namespace file_path
}  // namespace iree

#endif  // IREE_BASE_INTERNAL_FILE_PATH_H_
