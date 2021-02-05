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

#include "iree/base/internal/file_path.h"

#include "absl/strings/str_cat.h"

namespace iree {
namespace file_path {

namespace {

std::pair<absl::string_view, absl::string_view> SplitPath(
    absl::string_view path) {
  size_t pos = path.find_last_of('/');
  // Handle the case with no '/' in 'path'.
  if (pos == absl::string_view::npos) {
    return std::make_pair(path.substr(0, 0), path);
  }
  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0) {
    return std::make_pair(path.substr(0, 1), absl::ClippedSubstr(path, 1));
  }
  return std::make_pair(path.substr(0, pos),
                        absl::ClippedSubstr(path, pos + 1));
}

// Return the parts of the basename of path, split on the final ".".
// If there is no "." in the basename or "." is the final character in the
// basename, the second value will be empty.
std::pair<absl::string_view, absl::string_view> SplitBasename(
    absl::string_view path) {
  path = Basename(path);
  size_t pos = path.find_last_of('.');
  if (pos == absl::string_view::npos)
    return std::make_pair(path, absl::ClippedSubstr(path, path.size(), 0));
  return std::make_pair(path.substr(0, pos),
                        absl::ClippedSubstr(path, pos + 1));
}

}  // namespace

std::string JoinPaths(absl::string_view path1, absl::string_view path2) {
  if (path1.empty()) return std::string(path2);
  if (path2.empty()) return std::string(path1);
  if (path1.back() == '/') {
    if (path2.front() == '/')
      return absl::StrCat(path1, absl::ClippedSubstr(path2, 1));
  } else {
    if (path2.front() != '/') return absl::StrCat(path1, "/", path2);
  }
  return absl::StrCat(path1, path2);
}

absl::string_view DirectoryName(absl::string_view path) {
  return SplitPath(path).first;
}

absl::string_view Basename(absl::string_view path) {
  return SplitPath(path).second;
}

absl::string_view Stem(absl::string_view path) {
  return SplitBasename(path).first;
}

absl::string_view Extension(absl::string_view path) {
  return SplitBasename(path).second;
}

}  // namespace file_path
}  // namespace iree
