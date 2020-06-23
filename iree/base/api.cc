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

#include "iree/base/api.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "iree/base/api_util.h"
#include "iree/base/file_mapping.h"
#include "iree/base/init.h"
#include "iree/base/tracing.h"

namespace iree {

//===----------------------------------------------------------------------===//
// iree_status_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT const char* IREE_API_CALL
iree_status_code_string(iree_status_code_t code) {
  switch (code) {
    case IREE_STATUS_OK:
      return "OK";
    case IREE_STATUS_CANCELLED:
      return "CANCELLED";
    case IREE_STATUS_UNKNOWN:
      return "UNKNOWN";
    case IREE_STATUS_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case IREE_STATUS_DEADLINE_EXCEEDED:
      return "DEADLINE_EXCEEDED";
    case IREE_STATUS_NOT_FOUND:
      return "NOT_FOUND";
    case IREE_STATUS_ALREADY_EXISTS:
      return "ALREADY_EXISTS";
    case IREE_STATUS_PERMISSION_DENIED:
      return "PERMISSION_DENIED";
    case IREE_STATUS_UNAUTHENTICATED:
      return "UNAUTHENTICATED";
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
    case IREE_STATUS_FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
    case IREE_STATUS_ABORTED:
      return "ABORTED";
    case IREE_STATUS_OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case IREE_STATUS_UNIMPLEMENTED:
      return "UNIMPLEMENTED";
    case IREE_STATUS_INTERNAL:
      return "INTERNAL";
    case IREE_STATUS_UNAVAILABLE:
      return "UNAVAILABLE";
    case IREE_STATUS_DATA_LOSS:
      return "DATA_LOSS";
    default:
      return "";
  }
}

//===----------------------------------------------------------------------===//
// IREE Core API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_api_version_check(iree_api_version_t expected_version,
                       iree_api_version_t* out_actual_version) {
  iree_api_version_t actual_version = IREE_API_VERSION_0;
  *out_actual_version = actual_version;
  return expected_version == actual_version ? IREE_STATUS_OK
                                            : IREE_STATUS_OUT_OF_RANGE;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_api_init(int* argc,
                                                          char*** argv) {
  InitializeEnvironment(argc, argv);
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree_allocator_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_allocator_malloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr) {
  if (!allocator.alloc) return IREE_STATUS_INVALID_ARGUMENT;
  return allocator.alloc(allocator.self, IREE_ALLOCATION_MODE_ZERO_CONTENTS,
                         byte_length, out_ptr);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_allocator_realloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr) {
  if (!allocator.alloc) return IREE_STATUS_INVALID_ARGUMENT;
  return allocator.alloc(allocator.self,
                         IREE_ALLOCATION_MODE_TRY_REUSE_EXISTING, byte_length,
                         out_ptr);
}

IREE_API_EXPORT void IREE_API_CALL
iree_allocator_free(iree_allocator_t allocator, void* ptr) {
  if (ptr && allocator.free) {
    return allocator.free(allocator.self, ptr);
  }
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_allocator_system_allocate(void* self, iree_allocation_mode_t mode,
                               iree_host_size_t byte_length, void** out_ptr) {
  IREE_TRACE_SCOPE0("iree_allocator_system_allocate");

  if (!out_ptr) {
    return IREE_STATUS_INVALID_ARGUMENT;
  } else if (byte_length <= 0) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  void* existing_ptr = *out_ptr;
  void* ptr = nullptr;
  if (existing_ptr && (mode & IREE_ALLOCATION_MODE_TRY_REUSE_EXISTING)) {
    ptr = std::realloc(existing_ptr, byte_length);
    if (ptr && (mode & IREE_ALLOCATION_MODE_ZERO_CONTENTS)) {
      std::memset(ptr, 0, byte_length);
    }
  } else {
    existing_ptr = NULL;
    if (mode & IREE_ALLOCATION_MODE_ZERO_CONTENTS) {
      ptr = std::calloc(1, byte_length);
    } else {
      ptr = std::malloc(byte_length);
    }
  }
  if (!ptr) {
    return IREE_STATUS_RESOURCE_EXHAUSTED;
  }

  if (existing_ptr) {
    IREE_TRACE_FREE(existing_ptr);
  }
  IREE_TRACE_ALLOC(ptr, byte_length);

  *out_ptr = ptr;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT void IREE_API_CALL iree_allocator_system_free(void* self,
                                                              void* ptr) {
  IREE_TRACE_SCOPE0("iree_allocator_system_free");
  IREE_TRACE_FREE(ptr);
  if (ptr) {
    std::free(ptr);
  }
}

//===----------------------------------------------------------------------===//
// iree_string_view_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_make_cstring_view(const char* str) {
  iree_string_view_t result;
  result.data = str;
  result.size = strlen(str);
  return result;
}

IREE_API_EXPORT int IREE_API_CALL
iree_string_view_compare(iree_string_view_t lhs, iree_string_view_t rhs) {
  size_t min_size = std::min(lhs.size, rhs.size);
  int cmp = strncmp(lhs.data, rhs.data, min_size);
  if (cmp != 0) return cmp;
  return rhs.size - lhs.size;
}

IREE_API_EXPORT bool IREE_API_CALL iree_string_view_starts_with(
    iree_string_view_t value, iree_string_view_t prefix) {
  if (!value.data || !prefix.data) {
    return false;
  } else if (prefix.size > value.size) {
    return false;
  }
  return strncmp(value.data, prefix.data, prefix.size) == 0;
}

IREE_API_EXPORT int IREE_API_CALL iree_string_view_split(
    iree_string_view_t value, char split_char, iree_string_view_t* out_lhs,
    iree_string_view_t* out_rhs) {
  if (!value.data || !value.size) {
    return -1;
  }
  const void* first_ptr = std::memchr(value.data, split_char, value.size);
  if (!first_ptr) {
    return -1;
  }
  int offset =
      static_cast<int>(reinterpret_cast<const char*>(first_ptr) - value.data);
  if (out_lhs) {
    out_lhs->data = value.data;
    out_lhs->size = offset;
  }
  if (out_rhs) {
    out_rhs->data = value.data + offset + 1;
    out_rhs->size = value.size - offset - 1;
  }
  return offset;
}

static bool MatchPattern(absl::string_view value, absl::string_view pattern) {
  size_t next_char_index = pattern.find_first_of("*?");
  if (next_char_index == std::string::npos) {
    return value == pattern;
  } else if (next_char_index > 0) {
    if (value.substr(0, next_char_index) !=
        pattern.substr(0, next_char_index)) {
      return false;
    }
    value = value.substr(next_char_index);
    pattern = pattern.substr(next_char_index);
  }
  if (value.empty() && pattern.empty()) {
    return true;
  }
  char pattern_char = pattern[0];
  if (pattern_char == '*' && pattern.size() > 1 && value.empty()) {
    return false;
  } else if (pattern_char == '*' && pattern.size() == 1) {
    return true;
  } else if (pattern_char == '?' || value[0] == pattern_char) {
    return MatchPattern(value.substr(1), pattern.substr(1));
  } else if (pattern_char == '*') {
    return MatchPattern(value, pattern.substr(1)) ||
           MatchPattern(value.substr(1), pattern);
  }
  return false;
}

IREE_API_EXPORT bool IREE_API_CALL iree_string_view_match_pattern(
    iree_string_view_t value, iree_string_view_t pattern) {
  return MatchPattern(absl::string_view(value.data, value.size),
                      absl::string_view(pattern.data, pattern.size));
}

//===----------------------------------------------------------------------===//
// iree::FileMapping
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_file_mapping_open_read(iree_string_view_t path, iree_allocator_t allocator,
                            iree_file_mapping_t** out_file_mapping) {
  IREE_TRACE_SCOPE0("iree_file_mapping_open_read");

  if (!out_file_mapping) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_file_mapping = nullptr;

  IREE_API_ASSIGN_OR_RETURN(
      auto file_mapping,
      FileMapping::OpenRead(std::string(path.data, path.size)));

  *out_file_mapping =
      reinterpret_cast<iree_file_mapping_t*>(file_mapping.release());

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_file_mapping_retain(iree_file_mapping_t* file_mapping) {
  IREE_TRACE_SCOPE0("iree_file_mapping_retain");
  auto* handle = reinterpret_cast<FileMapping*>(file_mapping);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_file_mapping_release(iree_file_mapping_t* file_mapping) {
  IREE_TRACE_SCOPE0("iree_file_mapping_release");
  auto* handle = reinterpret_cast<FileMapping*>(file_mapping);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_byte_span_t IREE_API_CALL
iree_file_mapping_data(iree_file_mapping_t* file_mapping) {
  IREE_TRACE_SCOPE0("iree_file_mapping_data");
  auto* handle = reinterpret_cast<FileMapping*>(file_mapping);
  CHECK(handle) << "NULL file_mapping handle";
  auto data = handle->data();
  return {const_cast<uint8_t*>(data.data()), data.size()};
}

}  // namespace iree
