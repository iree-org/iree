// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Freestanding <stddef.h> for wasm32.

#ifndef IREE_WASM_LIBC_STDDEF_H_
#define IREE_WASM_LIBC_STDDEF_H_

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

// wchar_t: 32-bit with -fno-short-wchar (matching the toolchain setting).
// In C++, wchar_t is a built-in type and must not be redefined.
#ifndef __cplusplus
typedef __WCHAR_TYPE__ wchar_t;
#endif

#ifdef __cplusplus
#define NULL __null
#else
#define NULL ((void*)0)
#endif
#define offsetof(type, member) __builtin_offsetof(type, member)

// C23 nullptr_t.
#if __STDC_VERSION__ >= 202311L
typedef typeof(nullptr) nullptr_t;
#endif

// max_align_t: the most-aligned fundamental type on wasm32.
// long double is 128-bit (quad) on wasm but alignment is 16 bytes.
typedef struct {
  long long __max_align_ll __attribute__((__aligned__(__alignof__(long long))));
  long double __max_align_ld
      __attribute__((__aligned__(__alignof__(long double))));
} max_align_t;

#endif  // IREE_WASM_LIBC_STDDEF_H_
