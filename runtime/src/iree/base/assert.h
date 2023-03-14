// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_ASSERT_H_
#define IREE_BASE_ASSERT_H_

#include <assert.h>
#include <stdlib.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// abort() wrapper
//===----------------------------------------------------------------------===//
// We shim this so that it's easier to set breakpoints on CRTs where `abort` is
// a define to some internal symbol rather than an actual function.

IREE_ATTRIBUTE_NORETURN static inline void iree_abort(void) { abort(); }

//===----------------------------------------------------------------------===//
// IREE_ASSERT macros
//===----------------------------------------------------------------------===//
// These are no-oped in builds with NDEBUG defined (by default anything but
// `-c dbg`/`-DCMAKE_BUILD_TYPE=Debug`). They differ from assert in that
// they avoid unused variable warnings when NDEBUG is defined. As with normal
// assert() ensure that side-effecting behavior is avoided as the expression
// will not be evaluated when the asserts are removed!

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

#define IREE_ASSERT_UNREACHABLE(...) IREE_ASSERT(false, __VA_ARGS__)

#define IREE_ASSERT_EQ(x, y, ...) _IREE_ASSERT_CMP(x, ==, y, __VA_ARGS__)
#define IREE_ASSERT_NE(x, y, ...) _IREE_ASSERT_CMP(x, !=, y, __VA_ARGS__)
#define IREE_ASSERT_LE(x, y, ...) _IREE_ASSERT_CMP(x, <=, y, __VA_ARGS__)
#define IREE_ASSERT_LT(x, y, ...) _IREE_ASSERT_CMP(x, <, y, __VA_ARGS__)
#define IREE_ASSERT_GE(x, y, ...) _IREE_ASSERT_CMP(x, >=, y, __VA_ARGS__)
#define IREE_ASSERT_GT(x, y, ...) _IREE_ASSERT_CMP(x, >, y, __VA_ARGS__)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_ASSERT_H_
