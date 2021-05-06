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

#include <assert.h>
#include <memory.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/attributes.h"
#include "iree/base/config.h"
#include "iree/base/string_view.h"
#include "iree/base/time.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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

  IREE_STATUS_CODE_MASK = 0x1Fu,
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
typedef struct iree_status_handle_t* iree_status_t;

// Returns an iree_status_t from the an iree_status_code_t.
#define iree_status_from_code(code)                          \
  ((iree_status_t)((uintptr_t)((iree_status_code_t)(code)) & \
                   IREE_STATUS_CODE_MASK))

// Returns the iree_status_code_t from an iree_status_t.
#define iree_status_code(value) \
  ((iree_status_code_t)(((uintptr_t)(value)) & IREE_STATUS_CODE_MASK))

// Macros to check the value of a status code.
#define iree_status_is_ok(value) \
  IREE_LIKELY((uintptr_t)(value) == IREE_STATUS_OK)
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
#define IREE_STATUS_IMPL_RETURN_AND_EVAL_IF_API_ERROR_(tail_expr, var, ...)  \
  iree_status_t var = (IREE_STATUS_IMPL_IDENTITY_(                           \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_GET_EXPR_)(__VA_ARGS__))); \
  if (IREE_UNLIKELY(var)) {                                                  \
    (tail_expr);                                                             \
    return IREE_STATUS_IMPL_ANNOTATE_SWITCH_(var, __VA_ARGS__);              \
  }

#define IREE_STATUS_IMPL_IGNORE_ERROR_(var, expr) \
  iree_status_t var = (expr);                     \
  if (IREE_UNLIKELY(var)) iree_status_ignore(var);

#define IREE_STATUS_IMPL_CHECK_OK_(var, expr) \
  iree_status_t var = (expr);                 \
  if (IREE_UNLIKELY(var)) iree_status_abort(var);

// We cut out all status storage code when not used.
#if IREE_STATUS_FEATURES == 0
#define IREE_STATUS_IMPL_MAKE_(code, ...) \
  (iree_status_t)(uintptr_t)((code)&IREE_STATUS_CODE_MASK)
#undef IREE_STATUS_IMPL_RETURN_IF_API_ERROR_
#define IREE_STATUS_IMPL_RETURN_IF_API_ERROR_(var, ...)                      \
  iree_status_t var = (IREE_STATUS_IMPL_IDENTITY_(                           \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_GET_EXPR_)(__VA_ARGS__))); \
  if (IREE_UNLIKELY(var)) return var;
#undef IREE_STATUS_IMPL_RETURN_AND_EVAL_IF_API_ERROR_
#define IREE_STATUS_IMPL_RETURN_AND_EVAL_IF_API_ERROR_(tail_expr, var, ...)  \
  iree_status_t var = (IREE_STATUS_IMPL_IDENTITY_(                           \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_GET_EXPR_)(__VA_ARGS__))); \
  if (IREE_UNLIKELY(var)) {                                                  \
    (tail_expr);                                                             \
    return var;                                                              \
  }
#undef IREE_STATUS_IMPL_IGNORE_ERROR_
#define IREE_STATUS_IMPL_IGNORE_ERROR_(var, expr) \
  iree_status_t var = (expr);                     \
  (void)(var);
#undef IREE_STATUS_IMPL_CHECK_OK_
#define IREE_STATUS_IMPL_CHECK_OK_(var, expr) \
  iree_status_t var = (expr);                 \
  if (IREE_UNLIKELY(!iree_status_is_ok(var))) abort();
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

// IREE_RETURN_IF_ERROR with a custom expression to evaluate before returning.
#define IREE_RETURN_AND_EVAL_IF_ERROR(tail_expr, ...)              \
  IREE_STATUS_IMPL_RETURN_AND_EVAL_IF_API_ERROR_(                  \
      tail_expr, IREE_STATUS_IMPL_CONCAT_(__status_, __COUNTER__), \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_IDENTITY_(__VA_ARGS__)))

// Ignores the status result of (expr) regardless of its value.
//
// Example:
//  IREE_IGNORE_ERROR(some_fn_that_may_fail());
#define IREE_IGNORE_ERROR(expr)   \
  IREE_STATUS_IMPL_IGNORE_ERROR_( \
      IREE_STATUS_IMPL_CONCAT_(__status_, __COUNTER__), (expr))

// Aborts the program if the result of (expr) is not IREE_STATUS_OK.
//
// WARNING: this should only be used when absolutely required and avoided in any
// core IREE code. Aborting is a very user-hostile behavior and on some systems
// can cause major issues. Prefer instead to properly handle errors and route
// them through hosting application infrastructure in a way that preserves more
// context than just an instruction pointer and a SIGABRT.
//
// Example:
//  IREE_CHECK_OK(some_fn_that_may_fail());
#define IREE_CHECK_OK(expr)                                                    \
  IREE_STATUS_IMPL_CHECK_OK_(IREE_STATUS_IMPL_CONCAT_(__status_, __COUNTER__), \
                             (expr))

// Returns the canonical status code for the given errno value.
// https://en.cppreference.com/w/cpp/error/errno_macros
IREE_API_EXPORT iree_status_code_t
iree_status_code_from_errno(int error_number);

#if defined(_WIN32) || defined(_WIN64)
// Returns the canonical status code for the given Win32 GetLastError code.
// https://docs.microsoft.com/en-us/windows/win32/api/errhandlingapi/nf-errhandlingapi-getlasterror
IREE_API_EXPORT iree_status_code_t
iree_status_code_from_win32_error(uint32_t error);
#endif  // _WIN32 || _WIN64

// Returns a NUL-terminated string constant for the given status code, such as
// IREE_STATUS_UNAVAILABLE = "UNAVAILABLE". Do not rely on string-matching the
// result as the exact text may change.
IREE_API_EXPORT const char* iree_status_code_string(iree_status_code_t code);

// Allocates a new status instance for a failing error |code|.
// |file| and |line| should be populated with __FILE__ and __LINE__ at the call
// site and an optional string |message| may be provided.
//
// The status will be allocated using the default system allocator and must be
// freed using either iree_status_free or iree_status_ignore.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_status_allocate(iree_status_code_t code, const char* file, uint32_t line,
                     iree_string_view_t message);

// Allocates a new status instance for a failing error |code| and annotates it
// with a printf-style format string. Roughly equivalent (though more efficient)
// than iree_status_allocate + iree_status_annotate_f.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_PRINTF_ATTRIBUTE(4, 5)
    iree_status_allocate_f(iree_status_code_t code, const char* file,
                           uint32_t line, const char* format, ...);

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t iree_status_allocate_vf(
    iree_status_code_t code, const char* file, uint32_t line,
    const char* format, va_list varargs_0, va_list varargs_1);

// Clones |status| into a new status instance.
// No payloads, if present, will be cloned.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_status_clone(iree_status_t status);

// Frees |status| if it has any associated storage.
IREE_API_EXPORT void iree_status_free(iree_status_t status);

// Ignores |status| regardless of its value and frees any associated payloads.
// Returns an OK status that can be used when chaining.
IREE_API_EXPORT iree_status_t iree_status_ignore(iree_status_t status);

// Aborts the program with a failing |status|.
// This will trigger a SIGABRT. It's best not to use this at all outside of
// demos or tools.
IREE_API_EXPORT IREE_ATTRIBUTE_NORETURN void iree_status_abort(
    iree_status_t status);

// Consumes the |status| by freeing its storage and returning its code.
IREE_API_EXPORT iree_status_code_t
iree_status_consume_code(iree_status_t status);

// NOTE: varargs don't optimize well so we hard-no-op the functions when
// annotations are not enabled.
#if IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS

// Annotates a status message with the given constant string message.
// Ignored if |base_status| is OK.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_status_annotate(iree_status_t base_status, iree_string_view_t message);

// Annotates a status message with the given printf-style message.
// Ignored if |base_status| is OK.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_PRINTF_ATTRIBUTE(2, 3)
    iree_status_annotate_f(iree_status_t base_status, const char* format, ...);

#else
#define iree_status_annotate(base_status, ...) (base_status)
#define iree_status_annotate_f(base_status, ...) (base_status)
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS

// Formats the status as a multi-line string containing all associated payloads.
// Note that this may contain PII such as file paths and must only be used for
// presenting errors to users and not sent to a logs aggregation service.
//
// If |buffer_capacity| is insufficient, then |out_buffer_length| is the
// number of characters that would have been written if |buffer_capacity|
// had been sufficiently large, not counting the terminating null character.
IREE_API_EXPORT bool iree_status_format(iree_status_t status,
                                        iree_host_size_t buffer_capacity,
                                        char* buffer,
                                        iree_host_size_t* out_buffer_length);

// Converts the status to an allocated string value.
// The caller must free the buffer with the system allocator.
IREE_API_EXPORT bool iree_status_to_string(iree_status_t status,
                                           char** out_buffer,
                                           iree_host_size_t* out_buffer_length);

// Prints |status| to the given |file| as a string with all available
// annotations. This will produce multiple lines of output and should be used
// only when dumping a status on failure.
IREE_API_EXPORT void iree_status_fprint(FILE* file, iree_status_t status);

//===----------------------------------------------------------------------===//
// IREE_ASSERT macros
//===----------------------------------------------------------------------===//
// These are no-oped in builds with NDEBUG defined (by default anything but
// `-c dbg`/`-DCMAKE_BUILD_TYPE=Debug`). They differ from assert in that
// they avoid unused variable warnings when NDEBUG is defined. As with normal
// assert() ensure that side-effecting behavior is avoided as the expression
// will not be evaluated when the asserts are removed!

// TODO(benvanik): move to iree/base/assert.h.
#if defined(NDEBUG)  // N(o) DEBUG

// Assertions disabled:

#define IREE_ASSERT(condition, ...) \
  while (false && (condition)) {    \
  }

// TODO(benvanik): replace the status_matchers version with a test macro.
// #define IREE_ASSERT_OK(status) IREE_ASSERT(iree_status_is_ok(status))

// However, we still want the compiler to parse x and y because
// we don't want to lose potentially useful errors and warnings
// (and want to hide unused variable warnings when asserts are disabled).
// _IREE_ASSERT_CMP is a helper and should not be used outside of this file.
#define _IREE_ASSERT_CMP(x, op, y, ...)        \
  while (false && ((void)(x), (void)(y), 0)) { \
  }

#else

// Assertions enabled:

#define IREE_ASSERT(condition, ...) assert(condition)

// TODO(#2843): better logging of status assertions.
// #define IREE_ASSERT_OK(status) IREE_ASSERT(iree_status_is_ok(status))

#define _IREE_ASSERT_CMP(x, op, y, ...) IREE_ASSERT(((x)op(y)), __VA_ARGS__)

#endif  // NDEBUG

#define IREE_ASSERT_ARGUMENT(name) IREE_ASSERT(name)

#define IREE_ASSERT_TRUE(expr, ...) IREE_ASSERT(!!(expr), __VA_ARGS__)
#define IREE_ASSERT_FALSE(expr, ...) IREE_ASSERT(!(expr), __VA_ARGS__)

#define IREE_ASSERT_EQ(x, y, ...) _IREE_ASSERT_CMP(x, ==, y, __VA_ARGS__)
#define IREE_ASSERT_NE(x, y, ...) _IREE_ASSERT_CMP(x, !=, y, __VA_ARGS__)
#define IREE_ASSERT_LE(x, y, ...) _IREE_ASSERT_CMP(x, <=, y, __VA_ARGS__)
#define IREE_ASSERT_LT(x, y, ...) _IREE_ASSERT_CMP(x, <, y, __VA_ARGS__)
#define IREE_ASSERT_GE(x, y, ...) _IREE_ASSERT_CMP(x, >=, y, __VA_ARGS__)
#define IREE_ASSERT_GT(x, y, ...) _IREE_ASSERT_CMP(x, >, y, __VA_ARGS__)

//===----------------------------------------------------------------------===//
// IREE Core API
//===----------------------------------------------------------------------===//

// Sprinkle this wherever to make it easier to find structs/functions that are
// not yet stable.
#define IREE_API_UNSTABLE

// Known versions of the API that can be referenced in code.
// Out-of-bounds values are possible in forward-versioned changes.
enum iree_api_version_e {
  IREE_API_VERSION_0 = 0u,
  // Always set to the latest version of the library from source.
  IREE_API_VERSION_LATEST = IREE_API_VERSION_0,
};
typedef uint32_t iree_api_version_t;

// Checks whether the |expected_version| of the caller matches the implemented
// version of |out_actual_version|. Forward compatibility of the API is
// supported but backward compatibility is not: newer binaries using older
// shared libraries of the runtime will fail.
//
// Returns IREE_STATUS_OUT_OF_RANGE if the actual version is not compatible with
// the expected version.
IREE_API_EXPORT iree_status_t
iree_api_version_check(iree_api_version_t expected_version,
                       iree_api_version_t* out_actual_version);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#include "iree/base/allocator.h"

#endif  // IREE_BASE_API_H_
