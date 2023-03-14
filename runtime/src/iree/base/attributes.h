// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_ATTRIBUTES_H_
#define IREE_BASE_ATTRIBUTES_H_

#include "iree/base/target_platform.h"

//===----------------------------------------------------------------------===//
// API/ABI interop
//===----------------------------------------------------------------------===//

// Denotes a method exported by the IREE API.
// Any call annotated with this will be relatively stable.
// Calls without this are considered private to the IREE implementation and
// should not be relied upon.
#if defined(__cplusplus)
#define IREE_API_EXPORT extern "C"
#else
#define IREE_API_EXPORT
#endif  // __cplusplus

#if defined(__cplusplus)
#define IREE_API_EXPORT_VARIABLE extern "C"
#else
#define IREE_API_EXPORT_VARIABLE extern
#endif  // __cplusplus

// Denotes a function pointer that is exposed as part of the IREE API.
// Example:
//   iree_status_t(IREE_API_PTR* some_callback)(int value);
#define IREE_API_PTR

//===----------------------------------------------------------------------===//
// IREE_HAVE_ATTRIBUTE
//===----------------------------------------------------------------------===//

// Queries for [[attribute]] identifiers in modern compilers.
#if defined(__has_attribute)
#define IREE_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define IREE_HAVE_ATTRIBUTE(x) 0
#endif  // __has_attribute

//===----------------------------------------------------------------------===//
// IREE_HAVE_BUILTIN
//===----------------------------------------------------------------------===//

#if defined(__has_builtin)
#define IREE_HAVE_BUILTIN(x) __has_builtin(x)
#else
#define IREE_HAVE_BUILTIN(x) 0
#endif  // __has_builtin

//===----------------------------------------------------------------------===//
// IREE_PRINTF_ATTRIBUTE
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// IREE_ATTRIBUTE_NORETURN
//===----------------------------------------------------------------------===//

// Tells the compiler that a given function never returns.
#if IREE_HAVE_ATTRIBUTE(noreturn) || (defined(__GNUC__) && !defined(__clang__))
#define IREE_ATTRIBUTE_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define IREE_ATTRIBUTE_NORETURN __declspec(noreturn)
#else
#define IREE_ATTRIBUTE_NORETURN
#endif  // IREE_HAVE_ATTRIBUTE(noreturn)

//===----------------------------------------------------------------------===//
// IREE_MUST_USE_RESULT
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// IREE_RESTRICT
//===----------------------------------------------------------------------===//

// `restrict` keyword, not supported by some older compilers.
// We define our own macro in case dependencies use `restrict` differently.
#if defined(_MSC_VER) && _MSC_VER >= 1900
#define IREE_RESTRICT __restrict
#elif defined(_MSC_VER)
#define IREE_RESTRICT
#elif defined(__cplusplus)
#define IREE_RESTRICT __restrict__
#else
#define IREE_RESTRICT restrict
#endif  // _MSC_VER

//===----------------------------------------------------------------------===//
// IREE_ATTRIBUTE_ALWAYS_INLINE / IREE_ATTRIBUTE_NOINLINE
//===----------------------------------------------------------------------===//

// Forces functions to either inline or not inline. Introduced in gcc 3.1.
#if IREE_HAVE_ATTRIBUTE(always_inline) || \
    (defined(__GNUC__) && !defined(__clang__))
#define IREE_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#else
#define IREE_ATTRIBUTE_ALWAYS_INLINE
#endif  // IREE_HAVE_ATTRIBUTE(always_inline)

#if IREE_HAVE_ATTRIBUTE(noinline) || (defined(__GNUC__) && !defined(__clang__))
#define IREE_ATTRIBUTE_NOINLINE __attribute__((noinline))
#else
#define IREE_ATTRIBUTE_NOINLINE
#endif  // IREE_HAVE_ATTRIBUTE(noinline)

//===----------------------------------------------------------------------===//
// IREE_ATTRIBUTE_HOT / IREE_ATTRIBUTE_COLD
//===----------------------------------------------------------------------===//

// Tells GCC that a function is hot or cold. GCC can use this information to
// improve static analysis, i.e. a conditional branch to a cold function
// is likely to be not-taken.
// This annotation is used for function declarations.
//
// Example:
//   int foo() IREE_ATTRIBUTE_HOT;
#if IREE_HAVE_ATTRIBUTE(hot) || (defined(__GNUC__) && !defined(__clang__))
#define IREE_ATTRIBUTE_HOT __attribute__((hot))
#else
#define IREE_ATTRIBUTE_HOT
#endif  // IREE_HAVE_ATTRIBUTE(hot)

#if IREE_HAVE_ATTRIBUTE(cold) || (defined(__GNUC__) && !defined(__clang__))
#define IREE_ATTRIBUTE_COLD __attribute__((cold))
#else
#define IREE_ATTRIBUTE_COLD
#endif  // IREE_HAVE_ATTRIBUTE(cold)

//===----------------------------------------------------------------------===//
// IREE_LIKELY / IREE_UNLIKELY
//===----------------------------------------------------------------------===//

// Compiler hint that can be used to indicate conditions that are very very very
// likely or unlikely. This is most useful for ensuring that unlikely cases such
// as error handling are moved off the mainline code path such that the code is
// only paged in when an error occurs.
//
// Example:
//   if (IREE_UNLIKELY(something_failed)) {
//     return do_expensive_error_logging();
//   }
#if defined(__GNUC__) || defined(__clang__)
#define IREE_LIKELY(x) (__builtin_expect(!!(x), 1))
#define IREE_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define IREE_LIKELY(x) (x)
#define IREE_UNLIKELY(x) (x)
#endif  // IREE_HAVE_ATTRIBUTE(likely)

//===----------------------------------------------------------------------===//
// IREE_ATTRIBUTE_PACKED
//===----------------------------------------------------------------------===//

#if IREE_HAVE_ATTRIBUTE(packed) || (defined(__GNUC__) && !defined(__clang__))
#define IREE_ATTRIBUTE_PACKED __attribute__((__packed__))
#else
#define IREE_ATTRIBUTE_PACKED
#endif  // IREE_HAVE_ATTRIBUTE(packed)

//===----------------------------------------------------------------------===//
// IREE_ATTRIBUTE_UNUSED
//===----------------------------------------------------------------------===//

// Hints that a variable is _maybe_ unused. This is primarily to quiet
// diagnostic messages about unused variables that crop up around variables
// passed to assert/logging/etc that gets stripped in certain configurations.
//
// Example:
//   int some_info IREE_ATTRIBUTE_UNUSED = compute_debug_info();
//   assert(some_info > 0);  // stripped in NDEBUG
#if IREE_HAVE_ATTRIBUTE(maybe_unused) && defined(__clang__)
#define IREE_ATTRIBUTE_UNUSED __attribute__((maybe_unused))
#elif IREE_HAVE_ATTRIBUTE(unused) || (defined(__GNUC__) && !defined(__clang__))
#define IREE_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define IREE_ATTRIBUTE_UNUSED
#endif  // IREE_HAVE_ATTRIBUTE(maybe_unused / unused)

#endif  // IREE_BASE_ATTRIBUTES_H_
