// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/path.h"

#include <stddef.h>
#include <string.h>

static iree_status_t iree_string_view_dup(iree_string_view_t value,
                                          iree_allocator_t allocator,
                                          char** out_buffer) {
  char* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, value.size + 1, (void**)&buffer));
  memcpy(buffer, value.data, value.size);
  buffer[value.size] = 0;  // NUL
  *out_buffer = buffer;
  return iree_ok_status();
}

static iree_status_t iree_string_view_cat(iree_string_view_t lhs,
                                          iree_string_view_t rhs,
                                          iree_allocator_t allocator,
                                          char** out_buffer) {
  // Allocate storage buffer with NUL character.
  iree_host_size_t total_length = lhs.size + rhs.size;
  char* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_length + 1, (void**)&buffer));

  // Copy both parts.
  memcpy(buffer, lhs.data, lhs.size);
  memcpy(buffer + lhs.size, rhs.data, rhs.size);

  buffer[total_length] = 0;  // NUL
  *out_buffer = buffer;
  return iree_ok_status();
}

static iree_status_t iree_string_view_join(iree_host_size_t part_count,
                                           const iree_string_view_t* parts,
                                           iree_string_view_t separator,
                                           iree_allocator_t allocator,
                                           char** out_buffer) {
  // Compute total output size in characters.
  iree_host_size_t total_length = 0;
  for (iree_host_size_t i = 0; i < part_count; ++i) {
    total_length += parts[i].size;
  }
  total_length += part_count > 0 ? separator.size * (part_count - 1) : 0;

  // Allocate storage buffer with NUL character.
  char* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_length + 1, (void**)&buffer));

  // Append each part and a separator between each.
  char* p = buffer;
  for (iree_host_size_t i = 0; i < part_count; ++i) {
    memcpy(p, parts[i].data, parts[i].size);
    p += parts[i].size;
    if (i != part_count - 1) {
      memcpy(p, separator.data, separator.size);
      p += separator.size;
    }
  }

  buffer[total_length] = 0;  // NUL
  *out_buffer = buffer;
  return iree_ok_status();
}

static iree_host_size_t iree_file_path_canonicalize_unix(
    char* path, iree_host_size_t path_length) {
  char* p = path;
  iree_host_size_t new_length = path_length;

  // Replace `//` with `/`.
  if (new_length > 1) {
    for (iree_host_size_t i = 0; i < new_length - 1; ++i) {
      if (p[i] == '/' && p[i + 1] == '/') {
        memmove(&p[i + 1], &p[i + 2], new_length - i - 2);
        --new_length;
        --i;
      }
    }
  }

  path[new_length] = 0;  // NUL
  return new_length;
}

static iree_host_size_t iree_file_path_canonicalize_win32(
    char* path, iree_host_size_t path_length) {
  char* p = path;
  iree_host_size_t new_length = path_length;

  // Replace `/` with `\`.
  for (iree_host_size_t i = 0; i < new_length; ++i) {
    if (p[i] == '/') p[i] = '\\';
  }

  // Replace `\\` with `\`.
  if (new_length > 1) {
    for (iree_host_size_t i = 0; i < new_length - 1; ++i) {
      if (p[i] == '\\' && p[i + 1] == '\\') {
        memmove(&p[i + 1], &p[i + 2], new_length - i - 2);
        --new_length;
        --i;
      }
    }
  }

  path[new_length] = 0;  // NUL
  return new_length;
}

iree_host_size_t iree_file_path_canonicalize(char* path,
                                             iree_host_size_t path_length) {
#if defined(IREE_PLATFORM_WINDOWS)
  return iree_file_path_canonicalize_win32(path, path_length);
#else
  return iree_file_path_canonicalize_unix(path, path_length);
#endif  // IREE_PLATFORM_WINDOWS
}

iree_status_t iree_file_path_join(iree_string_view_t lhs,
                                  iree_string_view_t rhs,
                                  iree_allocator_t allocator, char** out_path) {
  if (iree_string_view_is_empty(lhs)) {
    return iree_string_view_dup(rhs, allocator, out_path);
  }
  if (iree_string_view_is_empty(rhs)) {
    return iree_string_view_dup(lhs, allocator, out_path);
  }
  if (lhs.data[lhs.size - 1] == '/') {
    if (rhs.data[0] == '/') {
      return iree_string_view_cat(
          lhs, iree_string_view_substr(rhs, 1, IREE_STRING_VIEW_NPOS),
          allocator, out_path);
    }
  } else {
    if (rhs.data[0] != '/') {
      iree_string_view_t parts[2] = {lhs, rhs};
      return iree_string_view_join(IREE_ARRAYSIZE(parts), parts,
                                   iree_make_cstring_view("/"), allocator,
                                   out_path);
    }
  }
  return iree_string_view_cat(lhs, rhs, allocator, out_path);
}

void iree_file_path_split(iree_string_view_t path,
                          iree_string_view_t* out_dirname,
                          iree_string_view_t* out_basename) {
  iree_host_size_t pos = iree_string_view_find_last_of(
      path, iree_make_cstring_view("/"), IREE_STRING_VIEW_NPOS);
  if (pos == IREE_STRING_VIEW_NPOS) {
    // No '/' in path.
    *out_dirname = iree_string_view_empty();
    *out_basename = path;
  } else if (pos == 0) {
    // Single leading '/' in path.
    *out_dirname = iree_string_view_substr(path, 0, 1);
    *out_basename = iree_string_view_substr(path, 1, IREE_STRING_VIEW_NPOS);
  } else {
    *out_dirname = iree_string_view_substr(path, 0, pos);
    *out_basename =
        iree_string_view_substr(path, pos + 1, IREE_STRING_VIEW_NPOS);
  }
}

iree_string_view_t iree_file_path_dirname(iree_string_view_t path) {
  iree_string_view_t dirname = iree_string_view_empty();
  iree_string_view_t basename = iree_string_view_empty();
  iree_file_path_split(path, &dirname, &basename);
  return dirname;
}

iree_string_view_t iree_file_path_basename(iree_string_view_t path) {
  iree_string_view_t dirname = iree_string_view_empty();
  iree_string_view_t basename = iree_string_view_empty();
  iree_file_path_split(path, &dirname, &basename);
  return basename;
}

void iree_file_path_split_basename(iree_string_view_t path,
                                   iree_string_view_t* out_stem,
                                   iree_string_view_t* out_extension) {
  path = iree_file_path_basename(path);
  iree_host_size_t pos = iree_string_view_find_last_of(
      path, iree_make_cstring_view("."), IREE_STRING_VIEW_NPOS);
  if (pos == IREE_STRING_VIEW_NPOS) {
    *out_stem = path;
    *out_extension = iree_string_view_empty();
  } else {
    *out_stem = iree_string_view_substr(path, 0, pos);
    *out_extension =
        iree_string_view_substr(path, pos + 1, IREE_STRING_VIEW_NPOS);
  }
}

iree_string_view_t iree_file_path_stem(iree_string_view_t path) {
  iree_string_view_t stem = iree_string_view_empty();
  iree_string_view_t extension = iree_string_view_empty();
  iree_file_path_split_basename(path, &stem, &extension);
  return stem;
}

iree_string_view_t iree_file_path_extension(iree_string_view_t path) {
  iree_string_view_t stem = iree_string_view_empty();
  iree_string_view_t extension = iree_string_view_empty();
  iree_file_path_split_basename(path, &stem, &extension);
  return extension;
}

// We could limit this to only those libraries supported on the current platform
// or to accept special library suffixes (.so.1, etc). For now most of the
// libraries we produce and follow the cmake defaults which match these.
bool iree_file_path_is_dynamic_library(iree_string_view_t path) {
  iree_string_view_t ext = iree_file_path_extension(path);
  return iree_string_view_equal(ext, IREE_SV("dll")) ||
         iree_string_view_equal(ext, IREE_SV("dylib")) ||
         iree_string_view_equal(ext, IREE_SV("so")) ||
         iree_string_view_equal(ext, IREE_SV("sos"));
}

void iree_uri_split(iree_string_view_t uri, iree_string_view_t* out_schema,
                    iree_string_view_t* out_path,
                    iree_string_view_t* out_params) {
  *out_schema = iree_string_view_empty();
  *out_path = iree_string_view_empty();
  *out_params = iree_string_view_empty();
  if (iree_string_view_is_empty(uri)) return;

  // Split on `schema` `:` (anything).
  iree_string_view_t rhs = iree_string_view_empty();
  iree_string_view_split(uri, ':', out_schema, &rhs);

  // Strip leading // from the remaining string. The // isn't required but does
  // make things more URI-like.
  if (!iree_string_view_consume_prefix(&rhs, IREE_SV("//"))) {
    rhs = iree_string_view_strip_prefix(rhs, IREE_SV("/"));
  }

  // Split on `path` `?` `params.
  iree_string_view_split(rhs, '?', out_path, out_params);
}

iree_string_view_t iree_uri_schema(iree_string_view_t uri) {
  iree_string_view_t schema, path, params;
  iree_uri_split(uri, &schema, &path, &params);
  return schema;
}

iree_string_view_t iree_uri_path(iree_string_view_t uri) {
  iree_string_view_t schema, path, params;
  iree_uri_split(uri, &schema, &path, &params);
  return path;
}

iree_string_view_t iree_uri_params(iree_string_view_t uri) {
  iree_string_view_t schema, path, params;
  iree_uri_split(uri, &schema, &path, &params);
  return params;
}

bool iree_uri_split_params(iree_string_view_t params, iree_host_size_t capacity,
                           iree_host_size_t* out_count,
                           iree_string_pair_t* out_params) {
  // Cleanup string to remove leading/trailing junk.
  params = iree_string_view_strip_prefix(params, IREE_SV("&"));
  params = iree_string_view_strip_suffix(params, IREE_SV("&"));
  params = iree_string_view_trim(params);

  // Scan once to count; URI parsing should not be on a critical path.
  iree_host_size_t required_capacity =
      iree_string_view_is_empty(params) ? 0 : 1;
  for (iree_host_size_t i = 0; i < params.size; ++i) {
    if (params.data[i] == '&') ++required_capacity;
  }
  *out_count = required_capacity;
  if (capacity < required_capacity) return false;
  if (!out_params) return true;

  // Parse each param into a key=value pair.
  iree_string_view_t remaining = params;
  iree_host_size_t count = 0;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t key_value;
    iree_string_view_split(remaining, '&', &key_value, &remaining);
    iree_string_pair_t* pair = &out_params[count++];
    iree_string_view_split(key_value, '=', &pair->key, &pair->value);
  }

  return true;
}
