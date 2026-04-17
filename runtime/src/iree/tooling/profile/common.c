// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/common.h"

const char* iree_profile_record_type_name(
    iree_hal_profile_file_record_type_t record_type) {
  switch (record_type) {
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN:
      return "session_begin";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK:
      return "chunk";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END:
      return "session_end";
    default:
      return "unknown";
  }
}

void iree_profile_fprint_json_string(FILE* file, iree_string_view_t value) {
  fputc('"', file);
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    uint8_t c = (uint8_t)value.data[i];
    switch (c) {
      case '"':
        fputs("\\\"", file);
        break;
      case '\\':
        fputs("\\\\", file);
        break;
      case '\b':
        fputs("\\b", file);
        break;
      case '\f':
        fputs("\\f", file);
        break;
      case '\n':
        fputs("\\n", file);
        break;
      case '\r':
        fputs("\\r", file);
        break;
      case '\t':
        fputs("\\t", file);
        break;
      default:
        if (c < 0x20) {
          fprintf(file, "\\u%04x", c);
        } else {
          fputc(c, file);
        }
        break;
    }
  }
  fputc('"', file);
}

void iree_profile_fprint_hash_hex(FILE* file, const uint64_t hash[2]) {
  fprintf(file, "%016" PRIx64 "%016" PRIx64, hash[0], hash[1]);
}

bool iree_profile_key_matches(iree_string_view_t key,
                              iree_string_view_t filter) {
  if (iree_string_view_is_empty(filter) ||
      iree_string_view_equal(filter, IREE_SV("*"))) {
    return true;
  }
  return iree_string_view_match_pattern(key, filter);
}

double iree_profile_sqrt_f64(double value) {
  if (value <= 0.0) return 0.0;
  // Keep this standalone C tool free of libm linkage.
  double estimate = value >= 1.0 ? value : 1.0;
  for (int i = 0; i < 32; ++i) {
    estimate = 0.5 * (estimate + value / estimate);
  }
  return estimate;
}
