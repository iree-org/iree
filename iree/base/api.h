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

// API Versioning
// -----------------------------------------------------------------------------
//
// The C API is designed to be versioned such that breaking changes either in
// ABI (data types, struct sizes, etc) or signatures (function arguments change)
// will result in a bump of the IREE_API_VERSION_LATEST value.
//
// When linked in statically the runtime should never have a version conflict,
// however dynamic linking where the runtime is a shared object loaded at
// runtime (via dlopen/etc) must always verify the version is as expected.
//
// In the current experimental state of the runtime the API may break frequently
// and the version is pinned at 0.
//
// Example:
//   void* library = dlopen("iree_rt.so", RTLD_LAZY | RTLD_LOCAL);
//   iree_api_version_t actual_version;
//   iree_status_t status = \
//       ((PFN_iree_api_version_check)dlsym(library, "iree_api_version_check"))(
//       IREE_API_VERSION_LATEST, &actual_version);
//   IREE_CHECK_OK(status);
//   dlclose(library);
//
// Object Ownership and Lifetime
// -----------------------------------------------------------------------------
//
// The API follows the CoreFoundation ownership policies:
// https://developer.apple.com/library/archive/documentation/CoreFoundation/Conceptual/CFMemoryMgmt/Concepts/Ownership.html
//
// These boil down to:
// * Objects returned from *_create or *_copy functions are owned by the caller
//   and must be released when the caller no longer needs them.
// * Objects returned from accessors are not owned by the caller and must be
//   retained by the caller if the object lifetime needs to be extended.
// * Objects passed to functions by argument may be retained by the callee if
//   required.
//
// Example:
//   iree_file_mapping_t* file_mapping;
//   s = iree_file_mapping_open_read(..., &file_mapping);
//   // file_mapping is now owned by this function.
//   s = iree_file_mapping_some_call(file_mapping, ...);
//   // Must release ownership when no longer required.
//   s = iree_file_mapping_release(file_mapping);
//
// String Formatting
// -----------------------------------------------------------------------------
//
// Functions that produce variable-length strings follow a standard usage
// pattern with the arguments:
//   `iree_host_size_t buffer_capacity`: total bytes including \0 available.
//   `char* buffer`: optional buffer to write into.
//   `iree_host_size_t* out_buffer_length`: required/actual length excluding \0.
//
// To query the size required for the output and allocate storage:
//   iree_host_size_t required_length = 0;
//   iree_format_xyz(/*buffer_capacity=*/0, /*buffer=*/NULL, &required_length);
//   iree_host_size_t buffer_capacity = required_length + 1;
//   char* buffer = iree_allocator_malloc(buffer_capacity);
//   iree_host_size_t actual_length = 0;
//   iree_format_xyz(buffer_capacity, buffer, &actual_length);
//   ASSERT(required_length == actual_length);
//
// To handle fixed-length maximum strings (common):
//   // Fails if the string is longer than 127 characters (127 + \0 >= 128).
//   char buffer[128];
//   IREE_RETURN_IF_ERROR(iree_format_xyz(sizeof(buffer), buffer, NULL));
//
// Try fixed-length and fallback to a dynamic allocation:
//   char inline_buffer[128];
//   iree_host_size_t required_length = 0;
//   iree_status_t inline_status = iree_format_xyz(sizeof(inline_buffer),
//                                                 inline_buffer,
//                                                 &required_length);
//   if (iree_status_is_out_of_range(inline_status)) {
//     // Spilled inline_buffer, need to allocate required_length bytes and
//     // try again.
//     // ... see above for example ...
//   } else if (iree_status_is_ok(inline_status)) {
//     // Fit inside inline_buffer, required_length contains actual length.
//   } else {
//     return inline_status;
//   }

#ifndef IREE_BASE_API_H_
#define IREE_BASE_API_H_

#include <memory.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
// The safe malloca that may fall back to heap in the case of stack overflows:
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/malloca?view=vs-2019
// Because that gets really annoying to deal with during error handling we just
// go for _alloca which may generate SEH exceptions if we blow the stack.
#include <malloc.h>
#define iree_alloca(sz) _alloca(sz)
#else
#include <alloca.h>
#define iree_alloca(sz) alloca(sz)
#endif  // _WIN32

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#define IREE_API_EXPORT extern "C"
#else
#define IREE_API_EXPORT
#endif  // __cplusplus

#if defined(_WIN32)
#define IREE_API_CALL __stdcall
#define IREE_API_PTR IREE_API_CALL
#else
#define IREE_API_CALL
#define IREE_API_PTR
#endif  // _WIN32

// Queries for [[attribute]] identifiers in modern compilers.
#ifdef __has_attribute
#define IREE_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define IREE_HAVE_ATTRIBUTE(x) 0
#endif  // __has_attribute

// Tells the compiler to perform `printf` format string checking if the
// compiler supports it; see the 'format' attribute in
// <https://gcc.gnu.org/onlinedocs/gcc-4.7.0/gcc/Function-Attributes.html>.
#if IREE_HAVE_ATTRIBUTE(format) || (defined(__GNUC__) && !defined(__clang__))
#define IREE_PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))
#else
// TODO(benvanik): use _Printf_format_string_ in SAL for MSVC.
#define IREE_PRINTF_ATTRIBUTE(string_index, first_to_check)
#endif  // IREE_HAVE_ATTRIBUTE

// Annotation for function return values that ensures that they are used by the
// caller.
#if IREE_HAVE_ATTRIBUTE(nodiscard)
#define IREE_MUST_USE_RESULT [[nodiscard]]
#elif (defined(__clang__) && IREE_HAVE_ATTRIBUTE(warn_unused_result)) || \
    (defined(__GNUC__) && (__GNUC__ >= 4))
#define IREE_MUST_USE_RESULT __attribute__((warn_unused_result))
#elif defined(_MSC_VER) && (_MSC_VER >= 1700)
#define IREE_MUST_USE_RESULT _Check_return_
#else
#define IREE_MUST_USE_RESULT
#endif  // IREE_HAVE_ATTRIBUTE(nodiscard)

// Compiler hint that can be used to indicate conditions that are very very very
// likely or unlikely. This is most useful for ensuring that unlikely cases such
// as error handling are moved off the mainline code path such that the code is
// only paged in when an error occurs.
//
// Example:
//   if (IREE_UNLIKELY(something_failed)) {
//     return do_expensive_error_logging();
//   }
#if IREE_HAVE_ATTRIBUTE(likely) && IREE_HAVE_ATTRIBUTE(unlikely)
#define IREE_LIKELY(x) [[likely]] (x)
#define IREE_UNLIKELY(x) [[unlikely]] (x)
#elif defined(__GNUC__) || defined(__clang__)
#define IREE_LIKELY(x) (__builtin_expect(!!(x), 1))
#define IREE_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define IREE_LIKELY(x) (x)
#define IREE_UNLIKELY(x) (x)
#endif  // IREE_HAVE_ATTRIBUTE(likely)

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define IREE_IS_LITTLE_ENDIAN 1
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define IREE_IS_BIG_ENDIAN 1
#elif defined(_WIN32)
#define IREE_IS_LITTLE_ENDIAN 1
#else
#error "IREE endian detection needs to be set up for your compiler"
#endif  // __BYTE_ORDER__

// Size, in bytes, of a buffer on the host.
typedef size_t iree_host_size_t;

// Size, in bytes, of a buffer on devices.
typedef uint64_t iree_device_size_t;
// Whole length of the underlying buffer.
#define IREE_WHOLE_BUFFER (iree_device_size_t(-1))

// TODO(benvanik): switch to static_cast/reinterpret_cast when in C++.
// TODO(benvanik): see if we can shove in static_asserts somehow?
#define iree_static_cast(type, value) (type)(value)
#define iree_reinterpret_cast(type, value) (type)(value)

// Returns the number of elements in an array as a compile-time constant, which
// can be used in defining new arrays. Fails at compile-time if |arr| is not a
// static array (such as if used on a pointer type).
//
// Example:
//  uint8_t kConstantArray[512];
//  assert(IREE_ARRAYSIZE(kConstantArray) == 512);
#define IREE_ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))

// Aligns |value| up to the given power-of-two |alignment| if required.
// https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
static inline iree_host_size_t iree_math_align(iree_host_size_t value,
                                               iree_host_size_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

//===----------------------------------------------------------------------===//
// Byte buffers and memory utilities
//===----------------------------------------------------------------------===//

// A span of mutable bytes (ala std::span of uint8_t).
typedef struct {
  uint8_t* data;
  iree_host_size_t data_length;
} iree_byte_span_t;

static inline iree_byte_span_t iree_make_byte_span(
    void* data, iree_host_size_t data_length) {
  iree_byte_span_t v = {(uint8_t*)data, data_length};
  return v;
}

// A span of constant bytes (ala std::span of const uint8_t).
typedef struct {
  const uint8_t* data;
  iree_host_size_t data_length;
} iree_const_byte_span_t;

static inline iree_const_byte_span_t iree_make_const_byte_span(
    const void* data, iree_host_size_t data_length) {
  iree_const_byte_span_t v = {(const uint8_t*)data, data_length};
  return v;
}

//===----------------------------------------------------------------------===//
// iree_string_view_t (like std::string_view/absl::string_view)
//===----------------------------------------------------------------------===//

#define IREE_STRING_VIEW_NPOS SIZE_MAX

// A string view (ala std::string_view) into a non-NUL-terminated string.
typedef struct {
  const char* data;
  iree_host_size_t size;
} iree_string_view_t;

// Returns an empty string view ("").
static inline iree_string_view_t iree_string_view_empty() {
  iree_string_view_t v = {0, 0};
  return v;
}

// Returns true if the given string view is the empty string.
#define iree_string_view_is_empty(sv) (((sv).data == NULL) || ((sv).size == 0))

static inline iree_string_view_t iree_make_string_view(
    const char* str, iree_host_size_t str_length) {
  iree_string_view_t v = {str, str_length};
  return v;
}

// Returns a string view initialized with a reference to the given
// NUL-terminated string literal.
static inline iree_string_view_t iree_make_cstring_view(const char* str) {
  iree_string_view_t v = {str, strlen(str)};
  return v;
}

// Returns true if the two strings are equal (compare == 0).
IREE_API_EXPORT bool IREE_API_CALL
iree_string_view_equal(iree_string_view_t lhs, iree_string_view_t rhs);

// Like std::string::compare but with iree_string_view_t values.
IREE_API_EXPORT int IREE_API_CALL
iree_string_view_compare(iree_string_view_t lhs, iree_string_view_t rhs);

// Returns true if the string starts with the given prefix.
IREE_API_EXPORT bool IREE_API_CALL iree_string_view_starts_with(
    iree_string_view_t value, iree_string_view_t prefix);

// Finds the first occurrence of |c| in |value| starting at |pos|.
// Returns the found character position or IREE_STRING_VIEW_NPOS if not found.
IREE_API_EXPORT iree_host_size_t IREE_API_CALL iree_string_view_find_char(
    iree_string_view_t value, char c, iree_host_size_t pos);

// Returns the index of the first occurrence of one of the characters in |s| or
// -1 if none of the characters were found.
IREE_API_EXPORT iree_host_size_t IREE_API_CALL iree_string_view_find_first_of(
    iree_string_view_t value, iree_string_view_t s, iree_host_size_t pos);

// Removes the first |n| characters from the string view (not the data).
IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_string_view_remove_prefix(iree_string_view_t value, iree_host_size_t n);

// Returns a substring of the string view at offset |pos| and length |n|.
// Use |n| == INTPTR_MAX to take the remaineder of the string after |pos|.
// Returns empty string on failure.
IREE_API_EXPORT iree_string_view_t IREE_API_CALL iree_string_view_substr(
    iree_string_view_t value, iree_host_size_t pos, iree_host_size_t n);

// Splits |value| into two parts based on the first occurrence of |split_char|.
// Returns the index of the |split_char| in the original |value| or -1 if not
// found.
IREE_API_EXPORT intptr_t IREE_API_CALL iree_string_view_split(
    iree_string_view_t value, char split_char, iree_string_view_t* out_lhs,
    iree_string_view_t* out_rhs);

// Returns true if the given |value| matches |pattern| (normal * and ? rules).
// This accepts wildcards in the form of '*' and '?' for any delimited value.
// '*' will match zero or more of any character and '?' will match exactly one
// of any character.
//
// For example,
// 'foo-*-bar' matches: 'foo-123-bar', 'foo-456-789-bar'
// 'foo-10?' matches: 'foo-101', 'foo-102'
IREE_API_EXPORT bool IREE_API_CALL iree_string_view_match_pattern(
    iree_string_view_t value, iree_string_view_t pattern);

//===----------------------------------------------------------------------===//
// IREE_STATUS_FEATURE flags and IREE_STATUS_MODE setting
//===----------------------------------------------------------------------===//

// Captures origin source information on a call to iree_make_status.
// Status storage will be allocated and reference the __FILE__ and __LINE__
// of where it is invoked.
#define IREE_STATUS_FEATURE_SOURCE_LOCATION (1 << 0)

// Captures annotation messages provided via iree_make_status or
// iree_status_annotate.
// Status storage will be allocated.
#define IREE_STATUS_FEATURE_ANNOTATIONS (1 << 1)

// Captures the current callstack on a call to iree_make_status.
// Status storage will be allocated.
#define IREE_STATUS_FEATURE_STACK_TRACE (1 << 2)

// If no status mode override is provided we'll change the behavior based on
// build configuration.
#if !defined(IREE_STATUS_MODE)
#ifdef NDEBUG
// Release mode: just source location.
#define IREE_STATUS_MODE 1
#else
// Debug mode: annotations and stack traces.
#define IREE_STATUS_MODE 3
#endif  // NDEBUG
#endif  // !IREE_STATUS_MODE

// Set IREE_STATUS_FEATURES based on IREE_STATUS_MODE if the user hasn't
// overridden it with more specific settings.
//
// IREE_STATUS_MODE = 0: statuses are just integers
// IREE_STATUS_MODE = 1: statuses have source location of error
// IREE_STATUS_MODE = 2: statuses also have custom annotations
// IREE_STATUS_MODE = 3: statuses also have stack traces of the error site
#if !defined(IREE_STATUS_FEATURES)
#if defined(IREE_STATUS_MODE) && IREE_STATUS_MODE == 1
#define IREE_STATUS_FEATURES (IREE_STATUS_FEATURE_SOURCE_LOCATION)
#elif defined(IREE_STATUS_MODE) && IREE_STATUS_MODE == 2
#define IREE_STATUS_FEATURES \
  (IREE_STATUS_FEATURE_SOURCE_LOCATION | IREE_STATUS_FEATURE_ANNOTATIONS)
#elif defined(IREE_STATUS_MODE) && IREE_STATUS_MODE == 3
#define IREE_STATUS_FEATURES                                               \
  (IREE_STATUS_FEATURE_SOURCE_LOCATION | IREE_STATUS_FEATURE_ANNOTATIONS | \
   IREE_STATUS_FEATURE_STACK_TRACE)
#else
#define IREE_STATUS_FEATURES 0
#endif  // IREE_STATUS_MODE
#endif  // !IREE_STATUS_FEATURES

//===----------------------------------------------------------------------===//
// iree_status_t and error reporting
//===----------------------------------------------------------------------===//

// Well-known status codes matching iree::StatusCode.
// Note that any code within IREE_STATUS_CODE_MASK is valid even if not
// enumerated here. Always check for unhandled errors/have default conditions.
typedef enum {
  IREE_STATUS_OK = 0,
  IREE_STATUS_CANCELLED = 1,
  IREE_STATUS_UNKNOWN = 2,
  IREE_STATUS_INVALID_ARGUMENT = 3,
  IREE_STATUS_DEADLINE_EXCEEDED = 4,
  IREE_STATUS_NOT_FOUND = 5,
  IREE_STATUS_ALREADY_EXISTS = 6,
  IREE_STATUS_PERMISSION_DENIED = 7,
  IREE_STATUS_RESOURCE_EXHAUSTED = 8,
  IREE_STATUS_FAILED_PRECONDITION = 9,
  IREE_STATUS_ABORTED = 10,
  IREE_STATUS_OUT_OF_RANGE = 11,
  IREE_STATUS_UNIMPLEMENTED = 12,
  IREE_STATUS_INTERNAL = 13,
  IREE_STATUS_UNAVAILABLE = 14,
  IREE_STATUS_DATA_LOSS = 15,
  IREE_STATUS_UNAUTHENTICATED = 16,

  IREE_STATUS_CODE_MASK = 0x1F,
} iree_status_code_t;

// Opaque status structure containing an iree_status_code_t and optional status
// object with more detailed information and payloads.
//
// The status value uses the lower 5 bits to store the iree_status_code_t and
// the remaining uintptr_t bits to store an optional status payload pointer.
// An OK status will always be bit-equivalent to 0 to make success/failure
// checks as cheap as an integer non-zero comparison. As the payload is optional
// it's legal to construct an iree_status_t from an iree_status_code_t directly
// meaning `return iree_status_from_code(IREE_STATUS_INTERNAL);` (etc) is valid,
// though not as useful as constructing via iree_make_status (which captures
// additional info).
typedef void* iree_status_t;

// Returns an iree_status_t from the an iree_status_code_t.
#define iree_status_from_code(code)                          \
  ((iree_status_t)((uintptr_t)((iree_status_code_t)(code)) & \
                   IREE_STATUS_CODE_MASK))

// Returns the iree_status_code_t from an iree_status_t.
#define iree_status_code(value) \
  ((iree_status_code_t)(((uintptr_t)(value)) & IREE_STATUS_CODE_MASK))

// Macros to check the value of a status code.
#define iree_status_is_ok(value) ((uintptr_t)(value) == IREE_STATUS_OK)
#define iree_status_is_cancelled(value) \
  (iree_status_code(value) == IREE_STATUS_CANCELLED)
#define iree_status_is_unknown(value) \
  (iree_status_code(value) == IREE_STATUS_UNKNOWN)
#define iree_status_is_invalid_argument(value) \
  (iree_status_code(value) == IREE_STATUS_INVALID_ARGUMENT)
#define iree_status_is_deadline_exceeded(value) \
  (iree_status_code(value) == IREE_STATUS_DEADLINE_EXCEEDED)
#define iree_status_is_not_found(value) \
  (iree_status_code(value) == IREE_STATUS_NOT_FOUND)
#define iree_status_is_already_exists(value) \
  (iree_status_code(value) == IREE_STATUS_ALREADY_EXISTS)
#define iree_status_is_permission_denied(value) \
  (iree_status_code(value) == IREE_STATUS_PERMISSION_DENIED)
#define iree_status_is_resource_exhausted(value) \
  (iree_status_code(value) == IREE_STATUS_RESOURCE_EXHAUSTED)
#define iree_status_is_failed_precondition(value) \
  (iree_status_code(value) == IREE_STATUS_FAILED_PRECONDITION)
#define iree_status_is_aborted(value) \
  (iree_status_code(value) == IREE_STATUS_ABORTED)
#define iree_status_is_out_of_range(value) \
  (iree_status_code(value) == IREE_STATUS_OUT_OF_RANGE)
#define iree_status_is_unimplemented(value) \
  (iree_status_code(value) == IREE_STATUS_UNIMPLEMENTED)
#define iree_status_is_internal(value) \
  (iree_status_code(value) == IREE_STATUS_INTERNAL)
#define iree_status_is_unavailable(value) \
  (iree_status_code(value) == IREE_STATUS_UNAVAILABLE)
#define iree_status_is_data_loss(value) \
  (iree_status_code(value) == IREE_STATUS_DATA_LOSS)
#define iree_status_is_unauthenticated(value) \
  (iree_status_code(value) == IREE_STATUS_UNAUTHENTICATED)

#define IREE_STATUS_IMPL_CONCAT_INNER_(x, y) x##y
#define IREE_STATUS_IMPL_CONCAT_(x, y) IREE_STATUS_IMPL_CONCAT_INNER_(x, y)

#define IREE_STATUS_IMPL_IDENTITY_(...) __VA_ARGS__
#define IREE_STATUS_IMPL_GET_EXPR_(expr, ...) expr
#define IREE_STATUS_IMPL_GET_ARGS_(expr, ...) __VA_ARGS__
#define IREE_STATUS_IMPL_GET_MACRO_(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, \
                                    _10, _11, _12, _13, _14, ...)           \
  IREE_STATUS_IMPL_IDENTITY_(                                               \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_GET_EXPR_)(__VA_ARGS__))

#define IREE_STATUS_IMPL_MAKE_EMPTY_(file, line, status_code, ...) \
  iree_status_allocate(status_code, file, line, iree_string_view_empty())
#define IREE_STATUS_IMPL_MAKE_ANNOTATE_(file, line, status_code, message) \
  iree_status_allocate(status_code, file, line, iree_make_cstring_view(message))
#define IREE_STATUS_IMPL_MAKE_ANNOTATE_F_(file, line, status_code, ...) \
  iree_status_allocate_f(status_code, file, line, __VA_ARGS__)
#define IREE_STATUS_IMPL_MAKE_SWITCH_(file, line, ...)                      \
  IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_IDENTITY_(                    \
      IREE_STATUS_IMPL_GET_MACRO_)(                                         \
      __VA_ARGS__, IREE_STATUS_IMPL_MAKE_ANNOTATE_F_,                       \
      IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, \
      IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, \
      IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, \
      IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, \
      IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, \
      IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, \
      IREE_STATUS_IMPL_MAKE_ANNOTATE_, IREE_STATUS_IMPL_MAKE_EMPTY_))       \
  (file, line, IREE_STATUS_IMPL_GET_EXPR_(__VA_ARGS__),                     \
   IREE_STATUS_IMPL_GET_ARGS_(__VA_ARGS__))

#define IREE_STATUS_IMPL_PASS_(var, ...) var
#define IREE_STATUS_IMPL_ANNOTATE_(var, ...)                  \
  IREE_STATUS_IMPL_IDENTITY_(iree_status_annotate(            \
      var, iree_make_cstring_view(IREE_STATUS_IMPL_IDENTITY_( \
               IREE_STATUS_IMPL_GET_ARGS_)(__VA_ARGS__))))
#define IREE_STATUS_IMPL_ANNOTATE_F_(var, ...)       \
  IREE_STATUS_IMPL_IDENTITY_(iree_status_annotate_f( \
      var,                                           \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_GET_ARGS_)(__VA_ARGS__)))
#define IREE_STATUS_IMPL_ANNOTATE_SWITCH_(...)                                 \
  IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_IDENTITY_(                       \
      IREE_STATUS_IMPL_GET_MACRO_)(                                            \
      __VA_ARGS__, IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_, \
      IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_,              \
      IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_,              \
      IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_,              \
      IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_,              \
      IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_,              \
      IREE_STATUS_IMPL_ANNOTATE_, IREE_STATUS_IMPL_PASS_))                     \
  (IREE_STATUS_IMPL_GET_EXPR_(__VA_ARGS__),                                    \
   IREE_STATUS_IMPL_GET_ARGS_(__VA_ARGS__))
#define IREE_STATUS_IMPL_RETURN_IF_API_ERROR_(var, ...)                      \
  iree_status_t var = (IREE_STATUS_IMPL_IDENTITY_(                           \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_GET_EXPR_)(__VA_ARGS__))); \
  if (IREE_UNLIKELY(var)) {                                                  \
    return IREE_STATUS_IMPL_ANNOTATE_SWITCH_(var, __VA_ARGS__);              \
  }
#define IREE_STATUS_IMPL_IGNORE_ERROR_(var, expr) \
  iree_status_t var = (expr);                     \
  if (IREE_UNLIKELY(var)) iree_status_ignore(var);

// We cut out all status storage code when not used.
#if IREE_STATUS_FEATURES == 0
#define IREE_STATUS_IMPL_MAKE_(code, ...) (code)
#undef IREE_STATUS_IMPL_RETURN_IF_API_ERROR_
#define IREE_STATUS_IMPL_RETURN_IF_API_ERROR_(var, expr, ...) \
  iree_status_t var = (expr);                                 \
  if (IREE_UNLIKELY(var)) return var;
#undef IREE_STATUS_IMPL_IGNORE_ERROR_
#define IREE_STATUS_IMPL_IGNORE_ERROR_(var, expr) \
  iree_status_t var = (expr);                     \
  (void)(var);
#else
#define IREE_STATUS_IMPL_MAKE_(...) \
  IREE_STATUS_IMPL_MAKE_SWITCH_(__FILE__, __LINE__, __VA_ARGS__)
#endif  // !IREE_STATUS_FEATURES

// Returns an IREE_STATUS_OK.
#define iree_ok_status() iree_status_from_code(IREE_STATUS_OK)

// Makes an iree_status_t with the given iree_status_code_t code and records
// the current source location.
//
// Optionally either a message string literal or printf-style format string may
// be associated with the status.
//
// Examples:
//  return iree_make_status(IREE_STATUS_CANCELLED);
//  return iree_make_status(IREE_STATUS_CANCELLED, "because reasons");
//  return iree_make_status(IREE_STATUS_CANCELLED, "because %d > %d", a, b);
#define iree_make_status IREE_STATUS_IMPL_MAKE_

// Propagates the error returned by (expr) by returning from the current
// function on non-OK status. Optionally annotates the status with additional
// information (see iree_status_annotate for more information).
//
// Example:
//  iree_status_t OtherFunc(...);
//  iree_status_t MyFunc(...) {
//    IREE_RETURN_IF_ERROR(OtherFunc(...));
//    IREE_RETURN_IF_ERROR(OtherFunc(...), "with a message");
//    IREE_RETURN_IF_ERROR(OtherFunc(...), "with a value: %d", 5);
//    return iree_ok_status();
//  }
#define IREE_RETURN_IF_ERROR(...)                       \
  IREE_STATUS_IMPL_RETURN_IF_API_ERROR_(                \
      IREE_STATUS_IMPL_CONCAT_(__status_, __COUNTER__), \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_IDENTITY_(__VA_ARGS__)))

// Ignores the status result of (expr) regardless of its value.
//
// Example:
//  IREE_IGNORE_ERROR(some_fn_that_may_fail());
#define IREE_IGNORE_ERROR(expr)   \
  IREE_STATUS_IMPL_IGNORE_ERROR_( \
      IREE_STATUS_IMPL_CONCAT_(__status_, __COUNTER__), (expr))

// TODO(#2843): better logging of status checks.
#define IREE_CHECK_OK(expr) \
  CHECK_EQ(IREE_STATUS_OK, iree_status_consume_code(expr))
#define IREE_ASSERT_ARGUMENT(name) assert(name)

// Returns a NUL-terminated string constant for the given status code, such as
// IREE_STATUS_UNAVAILABLE = "UNAVAILABLE". Do not rely on string-matching the
// result as the exact text may change.
IREE_API_EXPORT const char* IREE_API_CALL
iree_status_code_string(iree_status_code_t code);

// Allocates a new status instance for a failing error |code|.
// |file| and |line| should be populated with __FILE__ and __LINE__ at the call
// site and an optional string |message| may be provided.
//
// The status will be allocated using the default system allocator and must be
// freed using either iree_status_free or iree_status_ignore.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_allocate(iree_status_code_t code, const char* file, uint32_t line,
                     iree_string_view_t message);

// Allocates a new status instance for a failing error |code| and annotates it
// with a printf-style format string. Roughly equivalent (though more efficient)
// than iree_status_allocate + iree_status_annotate_f.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
    IREE_PRINTF_ATTRIBUTE(4, 5)
        iree_status_allocate_f(iree_status_code_t code, const char* file,
                               uint32_t line, const char* format, ...);

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_allocate_vf(iree_status_code_t code, const char* file,
                        uint32_t line, const char* format, va_list varargs_0,
                        va_list varargs_1);

// Clones |status| into a new status instance.
// No payloads, if present, will be cloned.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_clone(iree_status_t status);

// Frees |status| if it has any associated storage.
IREE_API_EXPORT void IREE_API_CALL iree_status_free(iree_status_t status);

// Ignores |status| regardless of its value and frees any associated payloads.
// Returns an OK status that can be used when chaining.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_status_ignore(iree_status_t status);

// Consumes the |status| by freeing its storage and returning its code.
IREE_API_EXPORT iree_status_code_t IREE_API_CALL
iree_status_consume_code(iree_status_t status);

// Annotates a status message with the given constant string message.
// Ignored if |base_status| is OK.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_annotate(iree_status_t base_status, iree_string_view_t message);

// Annotates a status message with the given printf-style message.
// Ignored if |base_status| is OK.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
    IREE_PRINTF_ATTRIBUTE(2, 3)
        iree_status_annotate_f(iree_status_t base_status, const char* format,
                               ...);

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_API_CALL
iree_status_annotate_vf(iree_status_t base_status, const char* format,
                        va_list varargs_0, va_list varargs_1);

// Formats the status as a multi-line string containing all associated payloads.
// Note that this may contain PII such as file paths and must only be used for
// presenting errors to users and not sent to a logs aggregation service.
IREE_API_EXPORT bool IREE_API_CALL
iree_status_format(iree_status_t status, iree_host_size_t buffer_capacity,
                   char* buffer, iree_host_size_t* out_buffer_length);

// Converts the status to an allocated string value.
// The caller must free the buffer with the system allocator.
IREE_API_EXPORT bool IREE_API_CALL
iree_status_to_string(iree_status_t status, char** out_buffer,
                      iree_host_size_t* out_buffer_length);

//===----------------------------------------------------------------------===//
// IREE Core API
//===----------------------------------------------------------------------===//

// Known versions of the API that can be referenced in code.
// Out-of-bounds values are possible in forward-versioned changes.
typedef enum {
  IREE_API_VERSION_0 = 0,
  // Always set to the latest version of the library from source.
  IREE_API_VERSION_LATEST = IREE_API_VERSION_0,
} iree_api_version_t;

// Checks whether the |expected_version| of the caller matches the implemented
// version of |out_actual_version|. Forward compatibility of the API is
// supported but backward compatibility is not: newer binaries using older
// shared libraries of the runtime will fail.
//
// Returns IREE_STATUS_OUT_OF_RANGE if the actual version is not compatible with
// the expected version.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_api_version_check(iree_api_version_t expected_version,
                       iree_api_version_t* out_actual_version);

// Initializes IREE for use within a binary.
//
// Specifically, this parses any command line flags and performs module
// initialization (such as for tracing and dynamic driver registration). If
// your application is certain it does not need this functionality, this call
// may be skipped.
//
// |argc| and |argv| should contain any command line flags to parse.
// If there are no flags to parse, nullptr may be passed, but this should still
// be called so other initialization happens.
//
// This should typically be called early in some sort of main() function once,
// before calling most other API functions. Certain core API functions here
// such as iree_api_version_check, iree_allocator_malloc, and
// iree_allocator_free are safe to call before this.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_api_init(int* argc,
                                                          char*** argv);

//===----------------------------------------------------------------------===//
// iree_time_t and iree_duration_t
//===----------------------------------------------------------------------===//

// Like absl::Time, represented as nanoseconds since unix epoch.
// TODO(benvanik): pick something easy to get into/outof time_t/etc.
typedef int64_t iree_time_t;
// Like absl::InfinitePast.
#define IREE_TIME_INFINITE_PAST INT64_MIN
// Like absl::InfiniteFuture.
#define IREE_TIME_INFINITE_FUTURE INT64_MAX

// Like absl::Duration, represented as relative nanoseconds.
typedef int64_t iree_duration_t;
// Like absl::InfiniteDuration.
#define IREE_DURATION_INFINITE INT64_MAX
// Like absl::ZeroDuration.
#define IREE_DURATION_ZERO 0

// Returns the current system time in unix nanoseconds.
// Depending on the system architecture and power mode this time may have a
// very coarse granularity (on the order of microseconds to milliseconds).
//
// The system timer may not be monotonic; users should ensure when comparing
// times they check for negative values in case the time moves backwards.
IREE_API_EXPORT iree_time_t iree_time_now();

// Converts a relative timeout duration to an absolute deadline time.
// This handles the special cases of IREE_DURATION_ZERO and
// IREE_DURATION_INFINITE to avoid extraneous time queries.
IREE_API_EXPORT iree_time_t
iree_relative_timeout_to_deadline_ns(iree_duration_t timeout_ns);

//===----------------------------------------------------------------------===//
// iree_allocator_t (std::allocator-like interface)
//===----------------------------------------------------------------------===//

// Defines how an allocation from an iree_allocator_t should be made.
typedef enum {
  // The contents of the allocation *must* be zeroed by the allocator prior to
  // returning. Allocators may be able to elide the zeroing if they allocate
  // fresh pages from the system. It is always safe to zero contents if the
  // behavior of the allocator is not under our control.
  IREE_ALLOCATION_MODE_ZERO_CONTENTS = 1 << 0,
  // Tries to reuse an existing allocation provided via |out_ptr| if possible.
  // If the existing allocation is not reused then it is freed as if a call to
  // iree_allocator_free had been called on it. If the allocation fails then
  // the provided existing allocation is unmodified.
  //
  // This models the C realloc behavior.
  IREE_ALLOCATION_MODE_TRY_REUSE_EXISTING = 1 << 1,
} iree_allocation_mode_t;

// An allocator for host-memory allocations.
// IREE will attempt to use this in place of the system malloc and free.
// Pass the iree_allocator_system() macro to use the system allocator.
typedef struct {
  // User-defined pointer passed to all functions.
  void* self;
  // Allocates |byte_length| of memory and stores the pointer in |out_ptr|.
  // Systems should align to 16 byte boundaries (or otherwise their natural
  // SIMD alignment). The runtime pools internally and small allocations
  // (usually) won't be made through this interface.
  iree_status_t(IREE_API_PTR* alloc)(void* self, iree_allocation_mode_t mode,
                                     iree_host_size_t byte_length,
                                     void** out_ptr);
  // Frees |ptr| from a previous alloc call.
  void(IREE_API_PTR* free)(void* self, void* ptr);
} iree_allocator_t;

// Allocates a block of |byte_length| bytes from the given allocator.
// The contents of the returned memory is guaranteed to be zeroed.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_allocator_malloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr);

// Reallocates |out_ptr| to |byte_length| bytes with the given allocator.
// If the reallocation fails then the original |out_ptr| is unmodified.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_allocator_realloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr);

// Frees a previously-allocated block of memory to the given allocator.
IREE_API_EXPORT void IREE_API_CALL
iree_allocator_free(iree_allocator_t allocator, void* ptr);

// Allocates a block of |byte_length| bytes from the default system allocator.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_allocator_system_allocate(void* self, iree_allocation_mode_t mode,
                               iree_host_size_t byte_length, void** out_ptr);

// Frees a previously-allocated block of memory to the default system allocator.
IREE_API_EXPORT void IREE_API_CALL iree_allocator_system_free(void* self,
                                                              void* ptr);

// Allocates using the iree_allocator_malloc and iree_allocator_free methods.
// These will usually be backed by malloc and free.
static inline iree_allocator_t iree_allocator_system() {
  iree_allocator_t v = {NULL, iree_allocator_system_allocate,
                        iree_allocator_system_free};
  return v;
}

// Does not perform any allocation or deallocation; used to wrap objects that
// are owned by external code/live in read-only memory/etc.
static inline iree_allocator_t iree_allocator_null() {
  iree_allocator_t v = {NULL, NULL, NULL};
  return v;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_API_H_
