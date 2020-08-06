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

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/tracing.h"

static inline size_t iree_min_host_size(size_t a, size_t b) {
  return a < b ? a : b;
}

//===----------------------------------------------------------------------===//
// iree_string_view_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT bool IREE_API_CALL
iree_string_view_equal(iree_string_view_t lhs, iree_string_view_t rhs) {
  if (lhs.size != rhs.size) return false;
  for (iree_host_size_t i = 0; i < lhs.size; ++i) {
    if (lhs.data[i] != rhs.data[i]) return false;
  }
  return true;
}

IREE_API_EXPORT int IREE_API_CALL
iree_string_view_compare(iree_string_view_t lhs, iree_string_view_t rhs) {
  iree_host_size_t min_size = iree_min_host_size(lhs.size, rhs.size);
  int cmp = strncmp(lhs.data, rhs.data, min_size);
  if (cmp != 0) {
    return cmp;
  } else if (lhs.size == rhs.size) {
    return 0;
  }
  return lhs.size < rhs.size ? -1 : 1;
}

IREE_API_EXPORT bool IREE_API_CALL iree_string_view_starts_with(
    iree_string_view_t value, iree_string_view_t prefix) {
  if (!value.data || !prefix.data || prefix.size > value.size) {
    return false;
  }
  return strncmp(value.data, prefix.data, prefix.size) == 0;
}

IREE_API_EXPORT iree_host_size_t IREE_API_CALL iree_string_view_find_char(
    iree_string_view_t value, char c, iree_host_size_t pos) {
  if (iree_string_view_is_empty(value) || pos >= value.size) {
    return IREE_STRING_VIEW_NPOS;
  }
  const char* result =
      (const char*)(memchr(value.data + pos, c, value.size - pos));
  return result != NULL ? result - value.data : IREE_STRING_VIEW_NPOS;
}

IREE_API_EXPORT iree_host_size_t IREE_API_CALL iree_string_view_find_first_of(
    iree_string_view_t value, iree_string_view_t s, iree_host_size_t pos) {
  if (iree_string_view_is_empty(value) || iree_string_view_is_empty(s)) {
    return IREE_STRING_VIEW_NPOS;
  }
  if (s.size == 1) {
    // Avoid the cost of the lookup table for a single-character search.
    return iree_string_view_find_char(value, s.data[0], pos);
  }
  bool lookup_table[UCHAR_MAX + 1] = {0};
  for (iree_host_size_t i = 0; i < s.size; ++i) {
    lookup_table[(uint8_t)s.data[i]] = true;
  }
  for (iree_host_size_t i = pos; i < value.size; ++i) {
    if (lookup_table[(uint8_t)value.data[i]]) {
      return i;
    }
  }
  return IREE_STRING_VIEW_NPOS;
}

IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_string_view_remove_prefix(iree_string_view_t value, iree_host_size_t n) {
  if (n >= value.size) {
    return iree_string_view_empty();
  }
  return iree_make_string_view(value.data + n, value.size - n);
}

IREE_API_EXPORT iree_string_view_t IREE_API_CALL iree_string_view_substr(
    iree_string_view_t value, iree_host_size_t pos, iree_host_size_t n) {
  pos = iree_min_host_size(pos, value.size);
  n = iree_min_host_size(n, value.size - pos);
  return iree_make_string_view(value.data + pos, n);
}

IREE_API_EXPORT intptr_t IREE_API_CALL iree_string_view_split(
    iree_string_view_t value, char split_char, iree_string_view_t* out_lhs,
    iree_string_view_t* out_rhs) {
  if (!value.data || !value.size) {
    return -1;
  }
  const void* first_ptr = memchr(value.data, split_char, value.size);
  if (!first_ptr) {
    return -1;
  }
  intptr_t offset = (intptr_t)((const char*)(first_ptr)-value.data);
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

static bool iree_string_view_match_pattern_impl(iree_string_view_t value,
                                                iree_string_view_t pattern) {
  iree_host_size_t next_char_index = iree_string_view_find_first_of(
      pattern, iree_make_cstring_view("*?"), /*pos=*/0);
  if (next_char_index == IREE_STRING_VIEW_NPOS) {
    return iree_string_view_equal(value, pattern);
  } else if (next_char_index > 0) {
    iree_string_view_t value_prefix =
        iree_string_view_substr(value, 0, next_char_index);
    iree_string_view_t pattern_prefix =
        iree_string_view_substr(pattern, 0, next_char_index);
    if (!iree_string_view_equal(value_prefix, pattern_prefix)) {
      return false;
    }
    value =
        iree_string_view_substr(value, next_char_index, IREE_STRING_VIEW_NPOS);
    pattern = iree_string_view_substr(pattern, next_char_index,
                                      IREE_STRING_VIEW_NPOS);
  }
  if (iree_string_view_is_empty(value) && iree_string_view_is_empty(pattern)) {
    return true;
  }
  char pattern_char = pattern.data[0];
  if (pattern_char == '*' && pattern.size > 1 &&
      iree_string_view_is_empty(value)) {
    return false;
  } else if (pattern_char == '*' && pattern.size == 1) {
    return true;
  } else if (pattern_char == '?' || value.data[0] == pattern_char) {
    return iree_string_view_match_pattern_impl(
        iree_string_view_substr(value, 1, IREE_STRING_VIEW_NPOS),
        iree_string_view_substr(pattern, 1, IREE_STRING_VIEW_NPOS));
  } else if (pattern_char == '*') {
    return iree_string_view_match_pattern_impl(
               value,
               iree_string_view_substr(pattern, 1, IREE_STRING_VIEW_NPOS)) ||
           iree_string_view_match_pattern_impl(
               iree_string_view_substr(value, 1, IREE_STRING_VIEW_NPOS),
               pattern);
  }
  return false;
}

IREE_API_EXPORT bool IREE_API_CALL iree_string_view_match_pattern(
    iree_string_view_t value, iree_string_view_t pattern) {
  return iree_string_view_match_pattern_impl(value, pattern);
}

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

// We cut out all status storage code when not used.
#if IREE_STATUS_FEATURES != 0

// TODO(#265): move payload methods/types to header when API is stabilized.

// Defines the type of an iree_status_payload_t.
typedef enum {
  // Opaque; payload may still be formatted by a formatter but is not possible
  // to retrieve by the programmatic APIs.
  IREE_STATUS_PAYLOAD_TYPE_OPAQUE = 0,
  // A string message annotation of type iree_status_payload_message_t.
  IREE_STATUS_PAYLOAD_TYPE_MESSAGE = 1,
  // Starting type ID for user payloads. IREE reserves all payloads with types
  // less than this.
  IREE_STATUS_PAYLOAD_TYPE_MIN_USER = 0x70000000,
} iree_status_payload_type_t;

typedef struct iree_status_payload_s iree_status_payload_t;

// Function that formats a payload into a human-readable string form for logs.
typedef void(IREE_API_PTR* iree_status_payload_formatter_t)(
    const iree_status_payload_t* payload, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

// Header for optional status payloads.
// Each status may have zero or more payloads associated with it that can later
// be used to produce more detailed logging or programmatically query
// information about an error.
struct iree_status_payload_s {
  // Next payload in the status payload linked list.
  struct iree_status_payload_s* next;
  // Payload type identifier used for programmatic access to payloads. May be
  // IREE_STATUS_PAYLOAD_TYPE_OPAQUE if the payload cannot be accessed directly.
  iree_status_payload_type_t type;
  // Allocator used for the payload and associated resources.
  iree_allocator_t allocator;
  // String formatter callback used to write the payload into a string buffer.
  // If not present then the payload will be mentioned but not dumped when the
  // status is logged.
  iree_status_payload_formatter_t formatter;
};

// A string message (IREE_STATUS_PAYLOAD_TYPE_MESSAGE).
typedef struct {
  iree_status_payload_t header;
  // String data reference. May point to an address immediately following this
  // struct (if copied) or a constant string reference in rodata.
  iree_string_view_t message;
} iree_status_payload_message_t;

// Allocated storage for an iree_status_t.
// Only statuses that have either source information or payloads will have
// storage allocated for them.
typedef struct {
  // __FILE__ of the originating status allocation.
  const char* file;
  // __LINE__ of the originating status allocation.
  uint32_t line;
  // Optional doubly-linked list of payloads associated with the status.
  // Head = first added, tail = last added.
  iree_status_payload_t* payload_head;
  iree_status_payload_t* payload_tail;
} iree_status_storage_t;

#endif  // IREE_STATUS_FEATURES != 0

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_allocate(iree_status_code_t code, const char* file, uint32_t line,
                     iree_string_view_t message) {
  if (code == IREE_STATUS_OK) return iree_ok_status();
#if IREE_STATUS_FEATURES == 0
  return (iree_status_t)code;
#elif (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0
    // TODO(#265): just allocate storage.
#else
  // TODO(#265): status storage + message payload.
  return code;
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_allocate_f(iree_status_code_t code, const char* file, uint32_t line,
                       const char* format, ...) {
#if IREE_STATUS_FEATURES == 0
  return (iree_status_t)code;
#elif (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0
  return iree_status_allocate(code, file, line, iree_string_view_empty());
#else
  // TODO(#265): status storage.
  if (code == IREE_STATUS_OK) return iree_ok_status();
  return code;
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
}

IREE_API_EXPORT void IREE_API_CALL iree_status_free(iree_status_t status) {
#if IREE_STATUS_FEATURES != 0
  iree_status_storage_t* storage =
      (iree_status_storage_t*)(status & ~IREE_STATUS_CODE_MASK);
  if (!storage) return;
  iree_status_payload_t* payload = storage->payload_head;
  while (payload) {
    iree_status_payload_t* next = payload->next;
    iree_allocator_free(payload->allocator, payload);
    payload = next;
  }
  iree_allocator_free(iree_allocator_system(), storage);
#endif  // IREE_STATUS_FEATURES != 0
}

IREE_API_EXPORT iree_status_t iree_status_ignore(iree_status_t status) {
  iree_status_free(status);
  return iree_ok_status();
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_annotate(iree_status_t base_status, iree_string_view_t message) {
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0
  return base_status;
#else
  iree_status_storage_t* storage =
      (iree_status_storage_t*)(base_status & ~IREE_STATUS_CODE_MASK);
  // TODO(benvanik): allocate storage if error but no storage already.
  if (!storage) return base_status;
  // TODO(#265): status storage.
  return base_status;
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
IREE_PRINTF_ATTRIBUTE(2, 3)
    iree_status_annotate_f(iree_status_t base_status, const char* format, ...) {
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0
  return base_status;
#else
  // TODO(#265): status storage.
  return base_status;
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
}

IREE_API_EXPORT bool IREE_API_CALL
iree_status_format(iree_status_t status, iree_host_size_t buffer_capacity,
                   char* buffer, iree_host_size_t* out_buffer_length) {
  // TODO(#265): status storage.
  return false;
}

IREE_API_EXPORT bool IREE_API_CALL
iree_status_to_string(iree_status_t status, char** out_buffer,
                      iree_host_size_t* out_buffer_length) {
  // TODO(#265): status storage.
  return false;
}

//===----------------------------------------------------------------------===//
// IREE Core API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_api_version_check(iree_api_version_t expected_version,
                       iree_api_version_t* out_actual_version) {
  if (!out_actual_version) return IREE_STATUS_INVALID_ARGUMENT;
  iree_api_version_t actual_version = IREE_API_VERSION_0;
  *out_actual_version = actual_version;
  return expected_version == actual_version
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "IREE version mismatch; application expected "
                                "%d but IREE is compiled as %d",
                                expected_version, actual_version);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_api_init(int* argc,
                                                          char*** argv) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_time_t and iree_duration_t
//===----------------------------------------------------------------------===//

// In api_time.cc until we have our own portable C implementation:
// IREE_API_EXPORT iree_time_t iree_time_now();

IREE_API_EXPORT iree_time_t
iree_relative_timeout_to_deadline_ns(iree_duration_t timeout_ns) {
  if (timeout_ns == IREE_DURATION_ZERO) {
    return IREE_TIME_INFINITE_PAST;
  } else if (timeout_ns == IREE_DURATION_INFINITE) {
    return IREE_TIME_INFINITE_FUTURE;
  }
  return iree_time_now() + timeout_ns;
}

//===----------------------------------------------------------------------===//
// iree_allocator_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_allocator_malloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr) {
  if (!allocator.alloc) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocator has no alloc routine");
  }
  return allocator.alloc(allocator.self, IREE_ALLOCATION_MODE_ZERO_CONTENTS,
                         byte_length, out_ptr);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_allocator_realloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr) {
  if (!allocator.alloc) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocator has no alloc routine");
  }
  return allocator.alloc(allocator.self,
                         IREE_ALLOCATION_MODE_TRY_REUSE_EXISTING, byte_length,
                         out_ptr);
}

IREE_API_EXPORT void IREE_API_CALL
iree_allocator_free(iree_allocator_t allocator, void* ptr) {
  if (ptr && allocator.free) {
    allocator.free(allocator.self, ptr);
  }
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_allocator_system_allocate(void* self, iree_allocation_mode_t mode,
                               iree_host_size_t byte_length, void** out_ptr) {
  IREE_ASSERT_ARGUMENT(out_ptr);
  if (byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocations must be >0 bytes");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  void* existing_ptr = *out_ptr;
  void* ptr = NULL;
  if (existing_ptr && (mode & IREE_ALLOCATION_MODE_TRY_REUSE_EXISTING)) {
    ptr = realloc(existing_ptr, byte_length);
    if (ptr && (mode & IREE_ALLOCATION_MODE_ZERO_CONTENTS)) {
      memset(ptr, 0, byte_length);
    }
  } else {
    existing_ptr = NULL;
    if (mode & IREE_ALLOCATION_MODE_ZERO_CONTENTS) {
      ptr = calloc(1, byte_length);
    } else {
      ptr = malloc(byte_length);
    }
  }
  if (!ptr) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "system allocator failed the request");
  }

  if (existing_ptr) {
    IREE_TRACE_FREE(existing_ptr);
  }
  IREE_TRACE_ALLOC(ptr, byte_length);

  *out_ptr = ptr;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void IREE_API_CALL iree_allocator_system_free(void* self,
                                                              void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_FREE(ptr);
  if (ptr) {
    free(ptr);
  }
  IREE_TRACE_ZONE_END(z0);
}
