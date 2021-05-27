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

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Canonicalizes the given |path| to the platform convention by replacing `/`
// with the appropriate character (`\` on Windows) and stripping extraneous
// slashes that may have ended up in the filename.
//
// NOTE: this is *not* the same as canonicalizing the path via system utilities
// that may, for example, resolve network paths or symlinks.
//
// |path| (of character length |path_length|) is mutated in-place and will have
// the same or smaller length upon return. Returns the new length of the path. A
// NUL terminator will be ensured at the end.
iree_host_size_t iree_file_path_canonicalize(char* path,
                                             iree_host_size_t path_length);

// Joins two paths together by inserting `/` as needed.
//
// For example:
//   iree_file_path_join('foo', 'bar') --> 'foo/bar'
//   iree_file_path_join('/foo/', '/bar') --> '/foo/bar'
//
// Returns the canonicalized path allocated from |allocator| in |out_path|.
// Callers must free the string when they are done with it.
iree_status_t iree_file_path_join(iree_string_view_t lhs,
                                  iree_string_view_t rhs,
                                  iree_allocator_t allocator, char** out_path);

// Splits |path| into the dirname and basename at the final `/`.
void iree_file_path_split(iree_string_view_t path,
                          iree_string_view_t* out_dirname,
                          iree_string_view_t* out_basename);

// Gets the directory name component of a file |path| (everything before the
// final `/`).
iree_string_view_t iree_file_path_dirname(iree_string_view_t path);

// Returns the part of the |path| after the final `/`.
iree_string_view_t iree_file_path_basename(iree_string_view_t path);

// Returns the parts of the basename of path, split on the final `.`.
// If there is no `.` in the basename or `.` is the final character in the
// basename the second value will be empty.
void iree_file_path_split_basename(iree_string_view_t path,
                                   iree_string_view_t* out_stem,
                                   iree_string_view_t* out_extension);

// Returns the part of the basename of |path| prior to the final `.`.
iree_string_view_t iree_file_path_stem(iree_string_view_t path);

// Returns the part of the basename of |path| after to the final `.`.
iree_string_view_t iree_file_path_extension(iree_string_view_t path);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_FILE_PATH_H_
