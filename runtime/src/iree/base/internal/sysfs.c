// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/sysfs.h"

#if defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_EMSCRIPTEN)

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

const char* iree_sysfs_get_root_path(void) {
#if defined(IREE_SYSFS_ROOT)
  return IREE_SYSFS_ROOT;
#else
  return "/sys/devices/system";
#endif  // IREE_SYSFS_ROOT
}

//===----------------------------------------------------------------------===//
// File I/O
//===----------------------------------------------------------------------===//

iree_status_t iree_sysfs_read_small_file(const char* path, char* buffer,
                                         size_t buffer_size,
                                         iree_host_size_t* out_length) {
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(buffer_size > 0);
  IREE_ASSERT_ARGUMENT(out_length);
  *out_length = 0;

  FILE* file = fopen(path, "r");
  if (!file) {
    if (errno == ENOENT) {
      return iree_make_status(IREE_STATUS_NOT_FOUND, "sysfs file not found: %s",
                              path);
    }
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "failed to open sysfs file %s: %s", path,
                            strerror(errno));
  }

  // Read entire file into buffer, reserving one byte for NUL terminator.
  // fread guarantees: bytes_read <= (buffer_size - 1).
  const size_t bytes_read = fread(buffer, 1, buffer_size - 1, file);
  if (ferror(file)) {
    fclose(file);
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to read sysfs file %s: %s", path,
                            strerror(errno));
  }

  // Check if file was truncated.
  if (bytes_read == buffer_size - 1 && !feof(file)) {
    fclose(file);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sysfs file %s exceeds buffer size of %zu bytes",
                            path, buffer_size);
  }

  fclose(file);

  // Safety: bytes_read <= (buffer_size - 1), so buffer[bytes_read] is always
  // a valid index (at most buffer[buffer_size - 1]) and we can add our NUL.
  IREE_ASSERT(bytes_read < buffer_size);

  // NUL-terminate and return length.
  buffer[bytes_read] = '\0';
  *out_length = bytes_read;
  return iree_ok_status();
}

bool iree_sysfs_try_read_small_file(const char* path, char* buffer,
                                    size_t buffer_size,
                                    iree_host_size_t* out_length) {
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(buffer_size > 0);
  IREE_ASSERT_ARGUMENT(out_length);
  *out_length = 0;

  FILE* file = fopen(path, "r");
  if (!file) return false;

  const size_t bytes_read = fread(buffer, 1, buffer_size - 1, file);
  if (ferror(file) || (bytes_read == buffer_size - 1 && !feof(file))) {
    fclose(file);
    return false;
  }
  fclose(file);

  IREE_ASSERT(bytes_read < buffer_size);
  buffer[bytes_read] = '\0';
  *out_length = bytes_read;
  return true;
}

//===----------------------------------------------------------------------===//
// Parsing utilities
//===----------------------------------------------------------------------===//

typedef enum iree_sysfs_cpu_list_parse_result_e {
  IREE_SYSFS_CPU_LIST_PARSE_RESULT_OK = 0,
  IREE_SYSFS_CPU_LIST_PARSE_RESULT_INVALID_NUMBER = 1,
  IREE_SYSFS_CPU_LIST_PARSE_RESULT_INVALID_RANGE = 2,
} iree_sysfs_cpu_list_parse_result_t;

static iree_sysfs_cpu_list_parse_result_t iree_sysfs_parse_cpu_list_impl(
    iree_string_view_t text, iree_sysfs_cpu_list_callback_t callback,
    void* user_data) {
  text = iree_string_view_trim(text);

  // Split on commas and process each element.
  iree_host_size_t offset = 0;
  while (offset < text.size) {
    iree_host_size_t comma_pos = iree_string_view_find_char(text, ',', offset);
    iree_host_size_t segment_end =
        (comma_pos == IREE_STRING_VIEW_NPOS) ? text.size : comma_pos;
    iree_string_view_t segment =
        iree_string_view_substr(text, offset, segment_end - offset);
    segment = iree_string_view_trim(segment);
    if (!iree_string_view_is_empty(segment)) {
      // Check if it's a range "N-M" or single CPU "N".
      iree_host_size_t dash_pos = iree_string_view_find_char(segment, '-', 0);
      uint32_t start_cpu, end_cpu;
      if (dash_pos == IREE_STRING_VIEW_NPOS) {
        // Single CPU: "N"
        if (!iree_string_view_atoi_uint32(segment, &start_cpu)) {
          return IREE_SYSFS_CPU_LIST_PARSE_RESULT_INVALID_NUMBER;
        }
        if (start_cpu == UINT32_MAX) {
          return IREE_SYSFS_CPU_LIST_PARSE_RESULT_INVALID_RANGE;
        }
        end_cpu = start_cpu + 1;
      } else {
        // Range: "N-M"
        iree_string_view_t start_str =
            iree_string_view_substr(segment, 0, dash_pos);
        iree_string_view_t end_str =
            iree_string_view_substr(segment, dash_pos + 1, IREE_HOST_SIZE_MAX);

        if (!iree_string_view_atoi_uint32(start_str, &start_cpu) ||
            !iree_string_view_atoi_uint32(end_str, &end_cpu)) {
          return IREE_SYSFS_CPU_LIST_PARSE_RESULT_INVALID_RANGE;
        }
        if (end_cpu < start_cpu || end_cpu == UINT32_MAX) {
          return IREE_SYSFS_CPU_LIST_PARSE_RESULT_INVALID_RANGE;
        }
        end_cpu += 1;  // Make exclusive.
      }
      if (!callback(start_cpu, end_cpu, user_data)) {
        break;  // Callback requested stop.
      }
    }
    offset = (comma_pos == IREE_STRING_VIEW_NPOS) ? text.size : comma_pos + 1;
  }

  return IREE_SYSFS_CPU_LIST_PARSE_RESULT_OK;
}

iree_status_t iree_sysfs_parse_cpu_list(iree_string_view_t text,
                                        iree_sysfs_cpu_list_callback_t callback,
                                        void* user_data) {
  IREE_ASSERT_ARGUMENT(callback);
  switch (iree_sysfs_parse_cpu_list_impl(text, callback, user_data)) {
    case IREE_SYSFS_CPU_LIST_PARSE_RESULT_OK:
      return iree_ok_status();
    case IREE_SYSFS_CPU_LIST_PARSE_RESULT_INVALID_NUMBER:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid CPU number in list");
    case IREE_SYSFS_CPU_LIST_PARSE_RESULT_INVALID_RANGE:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid CPU range in list");
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid CPU list");
}

bool iree_sysfs_try_parse_cpu_list(iree_string_view_t text,
                                   iree_sysfs_cpu_list_callback_t callback,
                                   void* user_data) {
  IREE_ASSERT_ARGUMENT(callback);
  return iree_sysfs_parse_cpu_list_impl(text, callback, user_data) ==
         IREE_SYSFS_CPU_LIST_PARSE_RESULT_OK;
}

static bool iree_sysfs_try_parse_size_string_impl(iree_string_view_t text,
                                                  uint64_t* out_size) {
  *out_size = 0;

  text = iree_string_view_trim(text);
  if (iree_string_view_is_empty(text)) return false;

  // Check for optional K suffix (case-insensitive).
  uint64_t scale = 1;
  if (iree_string_view_consume_suffix(&text, IREE_SV("K")) ||
      iree_string_view_consume_suffix(&text, IREE_SV("k"))) {
    scale = 1024;
  }

  // Parse the numeric part.
  text = iree_string_view_trim(text);
  uint64_t value = 0;
  if (!iree_string_view_atoi_uint64(text, &value)) return false;
  if (value > UINT64_MAX / scale) return false;

  *out_size = value * scale;
  return true;
}

iree_status_t iree_sysfs_parse_size_string(iree_string_view_t text,
                                           uint64_t* out_size) {
  IREE_ASSERT_ARGUMENT(out_size);
  if (!iree_sysfs_try_parse_size_string_impl(text, out_size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid size string");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Convenience helpers
//===----------------------------------------------------------------------===//

iree_status_t iree_sysfs_read_uint32(const char* path, uint32_t* out_value) {
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  char buffer[64];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      iree_sysfs_read_small_file(path, buffer, sizeof(buffer), &length));
  iree_string_view_t text =
      iree_string_view_trim(iree_make_string_view(buffer, length));
  if (!iree_string_view_atoi_uint32(text, out_value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid uint32 in %s", path);
  }
  return iree_ok_status();
}

bool iree_sysfs_try_read_uint32(const char* path, uint32_t* out_value) {
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  char buffer[64];
  iree_host_size_t length = 0;
  if (!iree_sysfs_try_read_small_file(path, buffer, sizeof(buffer), &length)) {
    return false;
  }
  iree_string_view_t text =
      iree_string_view_trim(iree_make_string_view(buffer, length));
  return iree_string_view_atoi_uint32(text, out_value);
}

iree_status_t iree_sysfs_read_size(const char* path, uint64_t* out_size) {
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;
  char buffer[64];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      iree_sysfs_read_small_file(path, buffer, sizeof(buffer), &length));
  return iree_sysfs_parse_size_string(iree_make_string_view(buffer, length),
                                      out_size);
}

bool iree_sysfs_try_read_size(const char* path, uint64_t* out_size) {
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;
  char buffer[64];
  iree_host_size_t length = 0;
  if (!iree_sysfs_try_read_small_file(path, buffer, sizeof(buffer), &length)) {
    return false;
  }
  return iree_sysfs_try_parse_size_string_impl(
      iree_make_string_view(buffer, length), out_size);
}

#endif  // IREE_PLATFORM_LINUX && !IREE_PLATFORM_EMSCRIPTEN
