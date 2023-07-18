// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_PATH_H_
#define IREE_BASE_INTERNAL_PATH_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// File paths
//===----------------------------------------------------------------------===//

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

// Returns true if |path| _likely_ represents a system dynamic library.
bool iree_file_path_is_dynamic_library(iree_string_view_t path);

//===----------------------------------------------------------------------===//
// URIs
//===----------------------------------------------------------------------===//

// Splits a URI into schema://path?params components.
// Returns empty strings for each component that is not present in the URI.
void iree_uri_split(iree_string_view_t uri, iree_string_view_t* out_schema,
                    iree_string_view_t* out_path,
                    iree_string_view_t* out_params);

// Returns the `schema` from a `schema://path?params` URI.
// Returns an empty string if no schema is present.
iree_string_view_t iree_uri_schema(iree_string_view_t uri);

// Returns the `path` from a `schema://path?params` URI.
// Returns an empty string if no path is present.
iree_string_view_t iree_uri_path(iree_string_view_t uri);

// Returns the `params` from a `schema://path?params` URI.
// Returns an empty string if no params are present.
iree_string_view_t iree_uri_params(iree_string_view_t uri);

// Splits a `key=value&key=value` params string into individual parameters.
// |capacity| defines the number of elements in the |out_params| storage and
// upon return |out_count| will contain the total number of parameters.
// Returns true if |capacity| is sufficient and |out_params| contains the parsed
// parameters and otherwise the caller should increase their storage capacity
// and call again.
bool iree_uri_split_params(iree_string_view_t params, iree_host_size_t capacity,
                           iree_host_size_t* out_count,
                           iree_string_pair_t* out_params);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_PATH_H_
