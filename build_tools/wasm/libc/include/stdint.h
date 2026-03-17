// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Freestanding <stdint.h> for wasm32 (ILP32 data model).

#ifndef IREE_WASM_LIBC_STDINT_H_
#define IREE_WASM_LIBC_STDINT_H_

// Exact-width integer types.
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;

// Minimum-width integer types.
typedef int8_t int_least8_t;
typedef uint8_t uint_least8_t;
typedef int16_t int_least16_t;
typedef uint16_t uint_least16_t;
typedef int32_t int_least32_t;
typedef uint32_t uint_least32_t;
typedef int64_t int_least64_t;
typedef uint64_t uint_least64_t;

// Fastest minimum-width integer types.
// On wasm32, 32-bit operations are native.
typedef int32_t int_fast8_t;
typedef uint32_t uint_fast8_t;
typedef int32_t int_fast16_t;
typedef uint32_t uint_fast16_t;
typedef int32_t int_fast32_t;
typedef uint32_t uint_fast32_t;
typedef int64_t int_fast64_t;
typedef uint64_t uint_fast64_t;

// Pointer-width integer types (wasm32 = 32-bit pointers).
typedef int32_t intptr_t;
typedef uint32_t uintptr_t;

// Greatest-width integer types.
typedef int64_t intmax_t;
typedef uint64_t uintmax_t;

// Limits of exact-width integer types.
#define INT8_MIN (-128)
#define INT8_MAX 127
#define UINT8_MAX 255
#define INT16_MIN (-32768)
#define INT16_MAX 32767
#define UINT16_MAX 65535
#define INT32_MIN (-2147483647 - 1)
#define INT32_MAX 2147483647
#define UINT32_MAX 4294967295U
#define INT64_MIN (-9223372036854775807LL - 1)
#define INT64_MAX 9223372036854775807LL
#define UINT64_MAX 18446744073709551615ULL

// Limits of minimum-width integer types.
#define INT_LEAST8_MIN INT8_MIN
#define INT_LEAST8_MAX INT8_MAX
#define UINT_LEAST8_MAX UINT8_MAX
#define INT_LEAST16_MIN INT16_MIN
#define INT_LEAST16_MAX INT16_MAX
#define UINT_LEAST16_MAX UINT16_MAX
#define INT_LEAST32_MIN INT32_MIN
#define INT_LEAST32_MAX INT32_MAX
#define UINT_LEAST32_MAX UINT32_MAX
#define INT_LEAST64_MIN INT64_MIN
#define INT_LEAST64_MAX INT64_MAX
#define UINT_LEAST64_MAX UINT64_MAX

// Limits of fastest minimum-width integer types.
#define INT_FAST8_MIN INT32_MIN
#define INT_FAST8_MAX INT32_MAX
#define UINT_FAST8_MAX UINT32_MAX
#define INT_FAST16_MIN INT32_MIN
#define INT_FAST16_MAX INT32_MAX
#define UINT_FAST16_MAX UINT32_MAX
#define INT_FAST32_MIN INT32_MIN
#define INT_FAST32_MAX INT32_MAX
#define UINT_FAST32_MAX UINT32_MAX
#define INT_FAST64_MIN INT64_MIN
#define INT_FAST64_MAX INT64_MAX
#define UINT_FAST64_MAX UINT64_MAX

// Limits of pointer-width integer types.
#define INTPTR_MIN INT32_MIN
#define INTPTR_MAX INT32_MAX
#define UINTPTR_MAX UINT32_MAX

// Limits of greatest-width integer types.
#define INTMAX_MIN INT64_MIN
#define INTMAX_MAX INT64_MAX
#define UINTMAX_MAX UINT64_MAX

// Limits of other integer types.
#define PTRDIFF_MIN INT32_MIN
#define PTRDIFF_MAX INT32_MAX
#define SIZE_MAX UINT32_MAX
#define SIG_ATOMIC_MIN INT32_MIN
#define SIG_ATOMIC_MAX INT32_MAX
#define WINT_MIN 0U
#define WINT_MAX UINT32_MAX

// Macros for integer constant expressions.
#define INT8_C(x) (x)
#define UINT8_C(x) (x)
#define INT16_C(x) (x)
#define UINT16_C(x) (x)
#define INT32_C(x) (x)
#define UINT32_C(x) (x##U)
#define INT64_C(x) (x##LL)
#define UINT64_C(x) (x##ULL)
#define INTMAX_C(x) INT64_C(x)
#define UINTMAX_C(x) UINT64_C(x)

#endif  // IREE_WASM_LIBC_STDINT_H_
