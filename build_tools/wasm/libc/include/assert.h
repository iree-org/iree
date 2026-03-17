// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <assert.h> for wasm32.

// assert.h is intentionally re-includable (NDEBUG can change between
// includes), so no include guard.

#include <stddef.h>

#undef assert
#undef static_assert

#ifdef NDEBUG
#define assert(expression) ((void)0)
#else
// iree_wasm_assert_fail is provided by the libc implementation and calls
// through to the JS host for reporting.
extern _Noreturn void iree_wasm_assert_fail(const char* expression,
                                            const char* file, int line,
                                            const char* function);
#define assert(expression) \
  ((expression)            \
       ? ((void)0)         \
       : iree_wasm_assert_fail(#expression, __FILE__, __LINE__, __func__))
#endif

// C11 static_assert.
#define static_assert _Static_assert
