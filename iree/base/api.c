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
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

static inline size_t iree_min_host_size(size_t a, size_t b) {
  return a < b ? a : b;
}

#if defined(IREE_PLATFORM_WINDOWS)
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/aligned-malloc
#define iree_aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define iree_aligned_free(p) _aligned_free(p)
#elif defined(_ISOC11_SOURCE)
// https://en.cppreference.com/w/c/memory/aligned_alloc
#define iree_aligned_alloc(alignment, size) aligned_alloc(alignment, size)
#define iree_aligned_free(p) free(p)
#elif _POSIX_C_SOURCE >= 200112L
// https://pubs.opengroup.org/onlinepubs/9699919799/functions/posix_memalign.html
static inline void* iree_aligned_alloc(size_t alignment, size_t size) {
  void* ptr = NULL;
  return posix_memalign(&ptr, alignment, size) == 0 ? ptr : NULL;
}
#define iree_aligned_free(p) free(p)
#else
// Emulates alignment with normal malloc. We overallocate by at least the
// alignment + the size of a pointer, store the base pointer at p[-1], and
// return the aligned pointer. This lets us easily get the base pointer in free
// to pass back to the system.
static inline void* iree_aligned_alloc(size_t alignment, size_t size) {
  void* base_ptr = malloc(size + alignment + sizeof(uintptr_t));
  if (!base_ptr) return NULL;
  uintptr_t* aligned_ptr = (uintptr_t*)iree_math_align(
      (uintptr_t)base_ptr + sizeof(uintptr_t), alignment);
  aligned_ptr[-1] = (uintptr_t)base_ptr;
  return aligned_ptr;
}
static inline void iree_aligned_free(void* p) {
  if (IREE_UNLIKELY(!p)) return;
  uintptr_t* aligned_ptr = (uintptr_t*)p;
  void* base_ptr = (void*)aligned_ptr[-1];
  free(base_ptr);
}
#endif  // IREE_PLATFORM_WINDOWS

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

// TODO(#55): move payload methods/types to header when API is stabilized.

// Defines the type of an iree_status_payload_t.
typedef enum {
  // Opaque; payload may still be formatted by a formatter but is not possible
  // to retrieve by the programmatic APIs.
  IREE_STATUS_PAYLOAD_TYPE_OPAQUE = 0u,
  // A string message annotation of type iree_status_payload_message_t.
  IREE_STATUS_PAYLOAD_TYPE_MESSAGE = 1u,
  // Starting type ID for user payloads. IREE reserves all payloads with types
  // less than this.
  IREE_STATUS_PAYLOAD_TYPE_MIN_USER = 0x70000000u,
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
  // Optional doubly-linked list of payloads associated with the status.
  // Head = first added, tail = last added.
  iree_status_payload_t* payload_head;
  iree_status_payload_t* payload_tail;

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_SOURCE_LOCATION) != 0
  // __FILE__ of the originating status allocation.
  const char* file;
  // __LINE__ of the originating status allocation.
  uint32_t line;
#endif  // has IREE_STATUS_FEATURE_SOURCE_LOCATION

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
  // Optional message that is allocated either as a constant string in rodata or
  // present as a suffix on the storage.
  iree_string_view_t message;
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
} iree_status_storage_t;

#define iree_status_storage(status) \
  ((iree_status_storage_t*)(((uintptr_t)(status) & ~IREE_STATUS_CODE_MASK)))

// Appends a payload to the storage doubly-linked list.
static iree_status_t iree_status_append_payload(
    iree_status_t status, iree_status_storage_t* storage,
    iree_status_payload_t* payload) {
  if (!storage->payload_tail) {
    storage->payload_head = payload;
  } else {
    storage->payload_tail->next = payload;
  }
  storage->payload_tail = payload;
  return status;
}

// Formats an iree_status_payload_message_t to the given output |buffer|.
// |out_buffer_length| will be set to the number of characters written excluding
// NUL. If |buffer| is omitted then |out_buffer_length| will be set to the
// total number of characters in |buffer_capacity| required to contain the
// entire message.
static void IREE_API_CALL iree_status_payload_message_formatter(
    const iree_status_payload_t* payload, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length) {
  iree_status_payload_message_t* message_payload =
      (iree_status_payload_message_t*)payload;
  if (!buffer) {
    *out_buffer_length = message_payload->message.size;
    return;
  }
  iree_host_size_t n = buffer_capacity < message_payload->message.size
                           ? buffer_capacity
                           : message_payload->message.size;
  memcpy(buffer, message_payload->message.data, n);
  buffer[n] = '\0';
  *out_buffer_length = n;
}

// Captures the current stack and attaches it to the status storage.
// A count of |skip_frames| will be skipped from the top of the stack.
// Setting |skip_frames|=0 will include the caller in the stack while
// |skip_frames|=1 will exclude it.
static void iree_status_attach_stack_trace(iree_status_storage_t* storage,
                                           int skip_frames) {
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_STACK_TRACE) != 0
  // TODO(#55): backtrace or other magic.
#endif  // has IREE_STATUS_FEATURE_STACK_TRACE
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_allocate(iree_status_code_t code, const char* file, uint32_t line,
                     iree_string_view_t message) {
#if IREE_STATUS_FEATURES == 0
  // More advanced status code features like source location and messages are
  // disabled. All statuses are just the codes.
  return (iree_status_t)(code & IREE_STATUS_CODE_MASK);
#else
  // No-op for OK statuses; we won't get these from the macros but may be called
  // with this from marshaling code.
  if (IREE_UNLIKELY(code == IREE_STATUS_OK)) return iree_ok_status();

  // Allocate storage with the appropriate alignment such that we can pack the
  // code in the lower bits of the pointer. Since failed statuses are rare and
  // likely have much larger costs (like string formatting) the extra bytes for
  // alignment are worth being able to avoid pointer dereferences and other
  // things during the normal code paths that just check codes.
  //
  // Note that we are using the CRT allocation function here, as we can't trust
  // our allocator system to work when we are throwing errors (as we may be
  // allocating this error from a failed allocation!).
  size_t storage_alignment = (IREE_STATUS_CODE_MASK + 1);
  size_t storage_size =
      iree_math_align(sizeof(iree_status_storage_t), storage_alignment);
  iree_status_storage_t* storage = (iree_status_storage_t*)iree_aligned_alloc(
      storage_alignment, storage_size);
  if (IREE_UNLIKELY(!storage)) return iree_status_from_code(code);
  memset(storage, 0, sizeof(*storage));

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_SOURCE_LOCATION) != 0
  storage->file = file;
  storage->line = line;
#endif  // has IREE_STATUS_FEATURE_SOURCE_LOCATION

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
  // NOTE: messages are rodata strings here and not retained.
  storage->message = message;
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS

  iree_status_attach_stack_trace(storage, /*skip_frames=*/1);
  return (iree_status_t)((uintptr_t)storage | (code & IREE_STATUS_CODE_MASK));
#endif  // has any IREE_STATUS_FEATURES
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_allocate_f(iree_status_code_t code, const char* file, uint32_t line,
                       const char* format, ...) {
  va_list varargs_0, varargs_1;
  va_start(varargs_0, format);
  va_start(varargs_1, format);
  iree_status_t ret =
      iree_status_allocate_vf(code, file, line, format, varargs_0, varargs_1);
  va_end(varargs_0);
  va_end(varargs_1);
  return ret;
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_allocate_vf(iree_status_code_t code, const char* file,
                        uint32_t line, const char* format, va_list varargs_0,
                        va_list varargs_1) {
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0
  // Annotations disabled; ignore the format string/args.
  return iree_status_allocate(code, file, line, iree_string_view_empty());
#else
  // No-op for OK statuses; we won't get these from the macros but may be called
  // with this from marshaling code.
  if (IREE_UNLIKELY(code == IREE_STATUS_OK)) return iree_ok_status();

  // Compute the total number of bytes (including NUL) required to store the
  // message.
  size_t message_size =
      vsnprintf(/*buffer=*/NULL, /*buffer_count=*/0, format, varargs_0);
  if (message_size < 0) return iree_status_from_code(code);
  ++message_size;  // NUL byte

  // Allocate storage with the additional room to store the formatted message.
  // This avoids additional allocations for the common case of a message coming
  // only from the original status error site.
  size_t storage_alignment = (IREE_STATUS_CODE_MASK + 1);
  size_t storage_size = iree_math_align(
      sizeof(iree_status_storage_t) + message_size, storage_alignment);
  iree_status_storage_t* storage = (iree_status_storage_t*)iree_aligned_alloc(
      storage_alignment, storage_size);
  if (IREE_UNLIKELY(!storage)) return iree_status_from_code(code);
  memset(storage, 0, sizeof(*storage));

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_SOURCE_LOCATION) != 0
  storage->file = file;
  storage->line = line;
#endif  // has IREE_STATUS_FEATURE_SOURCE_LOCATION

  // vsnprintf directly into message buffer.
  storage->message.size = message_size - 1;
  storage->message.data = (const char*)storage + sizeof(iree_status_storage_t);
  int ret =
      vsnprintf((char*)storage->message.data, message_size, format, varargs_1);
  if (IREE_UNLIKELY(ret < 0)) {
    iree_aligned_free(storage);
    return (iree_status_t)code;
  }

  iree_status_attach_stack_trace(storage, /*skip_frames=*/1);
  return (iree_status_t)((uintptr_t)storage | (code & IREE_STATUS_CODE_MASK));
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_clone(iree_status_t status) {
#if IREE_STATUS_FEATURES == 0
  // Statuses are just codes; nothing to do.
  return status;
#else
  iree_status_storage_t* storage = iree_status_storage(status);
  if (!storage) return status;

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_SOURCE_LOCATION) != 0
  const char* file = storage->file;
  uint32_t line = storage->line;
#else
  const char* file = NULL;
  uint32_t line = 0;
#endif  // has IREE_STATUS_FEATURE_SOURCE_LOCATION

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
  iree_string_view_t message = storage->message;
#else
  iree_string_view_t message = iree_string_view_empty();
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS

  // Always copy the message by performing the formatting as we don't know
  // whether the original status has ownership or not.
  return iree_status_allocate_f(iree_status_code(status), file, line, "%.*s",
                                (int)message.size, message.data);
#endif  // has no IREE_STATUS_FEATURES
}

IREE_API_EXPORT void IREE_API_CALL iree_status_free(iree_status_t status) {
#if IREE_STATUS_FEATURES != 0
  iree_status_storage_t* storage = iree_status_storage(status);
  if (!storage) return;
  iree_status_payload_t* payload = storage->payload_head;
  while (payload) {
    iree_status_payload_t* next = payload->next;
    iree_allocator_free(payload->allocator, payload);
    payload = next;
  }
  iree_aligned_free(storage);
#endif  // has any IREE_STATUS_FEATURES
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_status_ignore(iree_status_t status) {
  // We can set an 'ignored' flag on the status so that we can otherwise assert
  // in iree_status_free when statuses are freed without this being called.
  // Hoping with the C++ Status wrapper we won't hit that often so that
  // complexity is skipped for now.
  iree_status_free(status);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_code_t IREE_API_CALL
iree_status_consume_code(iree_status_t status) {
  iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);
  return code;
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_annotate(iree_status_t base_status, iree_string_view_t message) {
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0
  // Annotations are disabled so we ignore this entirely.
  return base_status;
#else
  if (iree_string_view_is_empty(message)) return base_status;
  // If there's no storage yet we can just reuse normal allocation. Both that
  // and this do not copy |message|.
  iree_status_storage_t* storage = iree_status_storage(base_status);
  if (!storage) {
    return iree_status_allocate(iree_status_code(base_status), NULL, 0,
                                message);
  } else if (iree_string_view_is_empty(storage->message)) {
    storage->message = message;
    return base_status;
  }
  iree_status_payload_message_t* payload =
      (iree_status_payload_message_t*)malloc(
          sizeof(iree_status_payload_message_t));
  if (IREE_UNLIKELY(!payload)) return base_status;
  memset(payload, 0, sizeof(*payload));
  payload->header.type = IREE_STATUS_PAYLOAD_TYPE_MESSAGE;
  payload->header.allocator = iree_allocator_system();
  payload->header.formatter = iree_status_payload_message_formatter;
  payload->message = message;
  return iree_status_append_payload(base_status, storage,
                                    (iree_status_payload_t*)payload);
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
IREE_PRINTF_ATTRIBUTE(2, 3)
    iree_status_annotate_f(iree_status_t base_status, const char* format, ...) {
  va_list varargs_0, varargs_1;
  va_start(varargs_0, format);
  va_start(varargs_1, format);
  iree_status_t ret =
      iree_status_annotate_vf(base_status, format, varargs_0, varargs_1);
  va_end(varargs_0);
  va_end(varargs_1);
  return ret;
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_annotate_vf(iree_status_t base_status, const char* format,
                        va_list varargs_0, va_list varargs_1) {
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0
  return base_status;
#else
  // If there's no storage yet we can just reuse normal allocation. Both that
  // and this do not copy |message|.
  iree_status_storage_t* storage = iree_status_storage(base_status);
  if (!storage) {
    return iree_status_allocate_vf(iree_status_code(base_status), NULL, 0,
                                   format, varargs_0, varargs_1);
  }

  // Compute the total number of bytes (including NUL) required to store the
  // message.
  size_t message_size =
      vsnprintf(/*buffer=*/NULL, /*buffer_count=*/0, format, varargs_0);
  va_end(varargs_0);
  if (message_size < 0) return base_status;
  ++message_size;  // NUL byte

  // Allocate storage with the additional room to store the formatted message.
  // This avoids additional allocations for the common case of a message coming
  // only from the original status error site.
  iree_status_payload_message_t* payload =
      (iree_status_payload_message_t*)malloc(
          sizeof(iree_status_payload_message_t) + message_size);
  if (IREE_UNLIKELY(!payload)) return base_status;
  memset(payload, 0, sizeof(*payload));
  payload->header.type = IREE_STATUS_PAYLOAD_TYPE_MESSAGE;
  payload->header.allocator = iree_allocator_system();
  payload->header.formatter = iree_status_payload_message_formatter;

  // vsnprintf directly into message buffer.
  payload->message.size = message_size - 1;
  payload->message.data =
      (const char*)payload + sizeof(iree_status_payload_message_t);
  int ret = vsnprintf((char*)payload->message.data, payload->message.size + 1,
                      format, varargs_1);
  if (IREE_UNLIKELY(ret < 0)) {
    iree_aligned_free(payload);
    return base_status;
  }
  return iree_status_append_payload(base_status, storage,
                                    (iree_status_payload_t*)payload);
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
}

IREE_API_EXPORT bool IREE_API_CALL
iree_status_format(iree_status_t status, iree_host_size_t buffer_capacity,
                   char* buffer, iree_host_size_t* out_buffer_length) {
  *out_buffer_length = 0;

  // Grab storage which may have a message and zero or more payloads.
  iree_status_storage_t* storage = iree_status_storage(status);

  // Prefix with source location and status code string (may be 'OK').
  iree_host_size_t buffer_length = 0;
  iree_status_code_t status_code = iree_status_code(status);
  int n = 0;
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_SOURCE_LOCATION) != 0
  if (storage && storage->file) {
    n = snprintf(buffer ? buffer + buffer_length : NULL,
                 buffer ? buffer_capacity - buffer_length : 0, "%s:%d: %s",
                 storage->file, storage->line,
                 iree_status_code_string(status_code));
  } else {
    n = snprintf(buffer ? buffer + buffer_length : NULL,
                 buffer ? buffer_capacity - buffer_length : 0, "%s",
                 iree_status_code_string(status_code));
  }
#else
  n = snprintf(buffer ? buffer + buffer_length : NULL,
               buffer ? buffer_capacity - buffer_length : 0, "%s",
               iree_status_code_string(status_code));
#endif  // has IREE_STATUS_FEATURE_SOURCE_LOCATION
  if (IREE_UNLIKELY(n < 0)) {
    return false;
  } else if (buffer && n >= buffer_capacity - buffer_length) {
    buffer = NULL;
  }
  buffer_length += n;

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
  // Append base storage message.
  if (storage && !iree_string_view_is_empty(storage->message)) {
    n = snprintf(buffer ? buffer + buffer_length : NULL,
                 buffer ? buffer_capacity - buffer_length : 0, "; %.*s",
                 (int)storage->message.size, storage->message.data);
    if (IREE_UNLIKELY(n < 0)) {
      return false;
    } else if (buffer && n >= buffer_capacity - buffer_length) {
      buffer = NULL;
    }
    buffer_length += n;
  }
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS

#if IREE_STATUS_FEATURES != 0
  // Append each payload separated by a newline.
  iree_status_payload_t* payload = storage ? storage->payload_head : NULL;
  while (payload != NULL) {
    // Skip payloads that have no textual representation.
    if (!payload->formatter) {
      payload = payload->next;
      continue;
    }

    // Append newline to join with message above and other payloads.
    if (buffer) {
      if (2 >= buffer_capacity - buffer_length) {
        buffer = NULL;
      } else {
        buffer[buffer_length] = ';';
        buffer[buffer_length + 1] = ' ';
        buffer[buffer_length + 2] = '\0';
      }
    }
    buffer_length += 2;  // '; '

    // Append payload via custom formatter callback.
    iree_host_size_t payload_buffer_length = 0;
    payload->formatter(payload, buffer ? buffer_capacity - buffer_length : 0,
                       buffer ? buffer + buffer_length : NULL,
                       &payload_buffer_length);
    if (buffer && payload_buffer_length >= buffer_capacity - buffer_length) {
      buffer = NULL;
    }
    buffer_length += payload_buffer_length;

    payload = payload->next;
  }
#endif  // has IREE_STATUS_FEATURES

  *out_buffer_length = buffer_length;
  return true;
}

IREE_API_EXPORT bool IREE_API_CALL
iree_status_to_string(iree_status_t status, char** out_buffer,
                      iree_host_size_t* out_buffer_length) {
  *out_buffer_length = 0;
  iree_host_size_t buffer_length = 0;
  if (IREE_UNLIKELY(!iree_status_format(status, /*buffer_capacity=*/0,
                                        /*buffer=*/NULL, &buffer_length))) {
    return false;
  }
  char* buffer = (char*)malloc(buffer_length + 1);
  if (IREE_UNLIKELY(!buffer)) return false;
  bool ret =
      iree_status_format(status, buffer_length, buffer, out_buffer_length);
  if (ret) {
    *out_buffer = buffer;
    return true;
  } else {
    free(buffer);
    return false;
  }
}

//===----------------------------------------------------------------------===//
// IREE Core API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_api_version_check(iree_api_version_t expected_version,
                       iree_api_version_t* out_actual_version) {
  if (!out_actual_version) {
    return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
  }
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

IREE_API_EXPORT iree_time_t iree_time_now() {
#if defined(IREE_PLATFORM_WINDOWS)
  // GetSystemTimePreciseAsFileTime requires Windows 8, add a fallback
  // (such as using std::chrono) if older support is needed.
  FILETIME system_time;
  GetSystemTimePreciseAsFileTime(&system_time);

  const int64_t kUnixEpochStartTicks = 116444736000000000i64;
  const int64_t kFtToMicroSec = 10;
  LARGE_INTEGER li;
  li.LowPart = system_time.dwLowDateTime;
  li.HighPart = system_time.dwHighDateTime;
  li.QuadPart -= kUnixEpochStartTicks;
  li.QuadPart /= kFtToMicroSec;
  return li.QuadPart;
#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX)
  struct timespec clock_time;
  clock_gettime(CLOCK_REALTIME, &clock_time);
  return clock_time.tv_nsec;
#else
#error "IREE system clock needs to be set up for your platform"
#endif  // IREE_PLATFORM_*
}

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
