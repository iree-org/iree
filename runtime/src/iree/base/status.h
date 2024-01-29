// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_STATUS_H_
#define IREE_BASE_STATUS_H_

#include <errno.h>
#include <memory.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"
#include "iree/base/string_view.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_allocator_t iree_allocator_t;

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
typedef enum iree_status_code_e {
  // Successful operation.
  IREE_STATUS_OK = 0,

  // Operation was cancelled by the caller.
  IREE_STATUS_CANCELLED = 1,

  // Unknown error, or error that could not be mapped to this enum.
  IREE_STATUS_UNKNOWN = 2,

  // The caller provided an invalid argument and that future calls with the same
  // arguments will fail. If the failure is predicated on system state that may
  // change prefer IREE_STATUS_OUT_OF_RANGE.
  IREE_STATUS_INVALID_ARGUMENT = 3,

  // A deadline was exceeded before the call could complete.
  // This can be returned even if the operation would have completed
  // successfully had the deadline not been met.
  IREE_STATUS_DEADLINE_EXCEEDED = 4,

  // A referenced resource could not be found or was unavailable to all
  // requesters. IREE_STATUS_PERMISSION_DENIED should be used if only an
  // individual requester is denied access.
  IREE_STATUS_NOT_FOUND = 5,

  // The resource the caller attempted to create already exists.
  IREE_STATUS_ALREADY_EXISTS = 6,

  // The caller does not have permission to execute the operation or have access
  // to the requested resources.
  IREE_STATUS_PERMISSION_DENIED = 7,

  // Some resource type has been exhausted and the operation is unable to
  // reserve what it requires, either by quota or underlying system exhaustion.
  IREE_STATUS_RESOURCE_EXHAUSTED = 8,

  // The operation was rejected because the system is not in a state required
  // for the operation's execution.
  //
  // Use IREE_STATUS_UNAVAILABLE if the caller can retry the operation.
  // Use IREE_STATUS_ABORTED if the caller should restart their transaction
  // (the entire sequence of operations is invalid).
  // Use IREE_STATUS_FAILED_PRECONDITION if the caller should not retry until
  // the system state has been explicitly fixed.
  IREE_STATUS_FAILED_PRECONDITION = 9,

  // The operation was aborted by the system.
  // If responding to a caller-requested cancellation use IREE_STATUS_CANCELLED.
  IREE_STATUS_ABORTED = 10,

  // The operation was attempted past the valid range (of a resource, etc).
  // Indicates the operation can be retried if the system state is fixed.
  IREE_STATUS_OUT_OF_RANGE = 11,

  // Operation has not been implemented or is not supported.
  IREE_STATUS_UNIMPLEMENTED = 12,

  // An internal error has occurred and some invariants expected by an
  // underlying system have been violated. This error code is reserved for
  // serious errors.
  IREE_STATUS_INTERNAL = 13,

  // The system used to perform the operation is currently (and transiently)
  // unavailable. Callers can retry with backoff.
  IREE_STATUS_UNAVAILABLE = 14,

  // An serious unrecoverable data loss or corruption has occurred.
  // Indicates that an underlying system or resource has failed in such a way
  // that all related operations may produce incorrect results.
  IREE_STATUS_DATA_LOSS = 15,

  // The requested operation does not have proper authentication.
  // Callers can correct this and retry.
  IREE_STATUS_UNAUTHENTICATED = 16,

  // The operation has been deferred and must be resumed at a future point.
  // Used by resumable operations as part of scheduling and execution systems.
  // Callers that do not handle deferred execution can treat this as a failure.
  IREE_STATUS_DEFERRED = 17,

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
#define iree_status_is_deferred(value) \
  (iree_status_code(value) == IREE_STATUS_DEFERRED)

#define IREE_STATUS_IMPL_CONCAT_INNER_(x, y) x##y
#define IREE_STATUS_IMPL_CONCAT_(x, y) IREE_STATUS_IMPL_CONCAT_INNER_(x, y)

#define IREE_STATUS_IMPL_IDENTITY_(...) __VA_ARGS__
#define IREE_STATUS_IMPL_GET_EXPR_(expr, ...) expr
#define IREE_STATUS_IMPL_GET_ARGS_(expr, ...) __VA_ARGS__
#define IREE_STATUS_IMPL_GET_MACRO_(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, \
                                    _10, _11, _12, _13, _14, _15, _16, _17, \
                                    ...)                                    \
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
      IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, \
      IREE_STATUS_IMPL_MAKE_ANNOTATE_F_, IREE_STATUS_IMPL_MAKE_ANNOTATE_,   \
      IREE_STATUS_IMPL_MAKE_EMPTY_))                                        \
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
#define IREE_STATUS_IMPL_ANNOTATE_SWITCH_(...)                        \
  IREE_STATUS_IMPL_IDENTITY_(                                         \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_GET_MACRO_)(        \
          __VA_ARGS__, IREE_STATUS_IMPL_ANNOTATE_F_,                  \
          IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_, \
          IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_, \
          IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_, \
          IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_, \
          IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_, \
          IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_, \
          IREE_STATUS_IMPL_ANNOTATE_F_, IREE_STATUS_IMPL_ANNOTATE_F_, \
          IREE_STATUS_IMPL_ANNOTATE_, IREE_STATUS_IMPL_PASS_))        \
  (IREE_STATUS_IMPL_GET_EXPR_(__VA_ARGS__),                           \
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
#define IREE_STATUS_IMPL_MAKE_LOC_(file, line, code, ...) \
  IREE_STATUS_IMPL_MAKE_(code)
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
  if (IREE_UNLIKELY(!iree_status_is_ok(var))) iree_abort();
#else
#define IREE_STATUS_IMPL_MAKE_(...) \
  IREE_STATUS_IMPL_MAKE_SWITCH_(__FILE__, __LINE__, __VA_ARGS__)
#define IREE_STATUS_IMPL_MAKE_LOC_(file, line, ...) \
  IREE_STATUS_IMPL_MAKE_SWITCH_(file, line, __VA_ARGS__)
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

// Makes an iree_status_t with the given iree_status_code_t code using the given
// source location. Besides taking the file and line of the source location this
// is the same as iree_make_status.
//
// Examples:
//  return iree_make_status_with_location(
//      "file.c", 40, IREE_STATUS_CANCELLED, "because %d > %d", a, b);
#define iree_make_status_with_location IREE_STATUS_IMPL_MAKE_LOC_

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

// Freezes |status| into a new status instance. Formats all attached
// annotations, payloads included, into a single message string allocated to
// the new status. This always frees the original status.
IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_status_freeze(iree_status_t status);

// Frees |status| if it has any associated storage.
IREE_API_EXPORT void iree_status_free(iree_status_t status);

// Ignores |status| regardless of its value and frees any associated payloads.
// Returns an OK status that can be used when chaining.
IREE_API_EXPORT iree_status_t iree_status_ignore(iree_status_t status);

// Returns a new status that is |base_status| if not OK and otherwise returns
// |new_status|. This allows for chaining failure handling code that may also
// return statuses.
//
// Example:
//   iree_status_t status = do_something();
//   return iree_status_join(status, do_cleanup());
IREE_API_EXPORT iree_status_t iree_status_join(iree_status_t base_status,
                                               iree_status_t new_status);

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

// Converts the status to an allocated string value using the given allocator.
// |out_buffer| will contain |out_buffer_length| characters as well as a NUL
// terminator. The caller must free the buffer with |allocator|.
//
// NOTE: |allocator| is passed as a pointer to avoid a circular dependency with
// iree/base/allocator.h (which uses this file a lot more than this file uses
// it) and must be non-NULL.
//
// Example:
//  iree_allocator_t allocator = iree_allocator_system();
//  char* buffer = NULL;
//  iree_host_size_t length = 0;
//  if (iree_status_to_string(status, &allocator, &buffer, &length)) {
//    // |buffer| is NUL terminated but if possible use the length.
//    LOG_MESSAGE("%.*s", (int)length, buffer);
//    iree_allocator_free(allocator, buffer);
//  } else {
//    // Could still use iree_status_code_string to get the status code string.
//    LOG_MESSAGE("failed to convert status to string");
//  }
IREE_API_EXPORT bool iree_status_to_string(iree_status_t status,
                                           const iree_allocator_t* allocator,
                                           char** out_buffer,
                                           iree_host_size_t* out_buffer_length);

// Prints |status| to the given |file| as a string with all available
// annotations. This will produce multiple lines of output and should be used
// only when dumping a status on failure.
IREE_API_EXPORT void iree_status_fprint(FILE* file, iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

// Optional C++ iree::Status wrapper.
// This makes it easier to safely use iree_status_t in C++ code and not leak.
#ifdef __cplusplus
#include "iree/base/status_cc.h"
#endif  // __cplusplus

#endif  // IREE_BASE_STATUS_H_
