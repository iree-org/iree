// Copyright 2020 Google LLC
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

// An implementation of the C11 stdatomics.h utilities we use (which is limited
// to a subset of types for now). We need this for non-C11-compliant platforms
// (MSVC), but it has the added benefit of not conflicting with <atomic>
// (stdatomic.h and atomic cannot be included in the same compilation unit...
// great design). There shouldn't be any difference between what we do here and
// what any implementation would do with the platform atomic functions so it's
// used everywhere.
//
// https://en.cppreference.com/w/c/atomic

#ifndef IREE_BASE_ATOMICS_H_
#define IREE_BASE_ATOMICS_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// C11 atomics using Clang builtins
//==============================================================================
#if defined(IREE_COMPILER_CLANG)

#define IREE_ATOMIC_VAR_INIT(value) (value)

typedef _Atomic int32_t iree_atomic_int32_t;
typedef _Atomic int64_t iree_atomic_int64_t;
typedef _Atomic __int128 iree_atomic_int128_t;

#define iree_atomic_load_auto(object) \
  __c11_atomic_load(object, __ATOMIC_SEQ_CST)
#define iree_atomic_store_auto(object, desired) \
  __c11_atomic_store(object, desired, __ATOMIC_SEQ_CST)
#define iree_atomic_fetch_add_auto(object, operand) \
  __c11_atomic_fetch_add(object, operand, __ATOMIC_SEQ_CST)
#define iree_atomic_fetch_sub_auto(object, operand) \
  __c11_atomic_fetch_sub(object, operand, __ATOMIC_SEQ_CST)
#define iree_atomic_compare_exchange_strong_auto(object, expected, desired) \
  __c11_atomic_compare_exchange_strong(object, &(expected), desired,        \
                                       __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)

//==============================================================================
// Atomics using the Win32 Interlocked* APIs
//==============================================================================
#elif defined(IREE_COMPILER_MSVC)

#define IREE_ATOMIC_VAR_INIT(value) \
  { (value) }

typedef struct {
  int32_t __val;
} iree_atomic_int32_t;
typedef struct {
  int64_t __val;
} iree_atomic_int64_t;
typedef __declspec(align(16)) struct {
  uint64_t __val[2];
} iree_atomic_int128_t;

#define iree_atomic_load_int32(object) \
  InterlockedExchangeAdd((volatile LONG*)object, 0)
#define iree_atomic_store_int32(object, desired) \
  InterlockedExchange((volatile LONG*)object, desired)
#define iree_atomic_fetch_add_int32(object, operand) \
  InterlockedExchangeAdd((volatile LONG*)object, operand)
#define iree_atomic_fetch_sub_int32(object, operand) \
  InterlockedExchangeAdd((volatile LONG*)object, -(operand))
#define iree_atomic_compare_exchange_strong_int32(object, expected, desired) \
  (InterlockedCompareExchange((volatile LONG*)object, desired, expected) ==  \
   expected)

#define iree_atomic_load_int64(object) \
  InterlockedExchangeAdd64((volatile LONG64*)object, 0)
#define iree_atomic_store_int64(object, desired) \
  InterlockedExchange64((volatile LONG64*)object, desired)
#define iree_atomic_fetch_add_int64(object, operand) \
  InterlockedExchangeAdd64((volatile LONG64*)object, operand)
#define iree_atomic_fetch_sub_int64(object, operand) \
  InterlockedExchangeAdd64((volatile LONG64*)object, -(operand))
#define iree_atomic_compare_exchange_strong_int64(object, expected, desired) \
  (InterlockedCompareExchange64((volatile LONG64*)object, desired,           \
                                expected) == expected)

#define iree_atomic_compare_exchange_strong_int128(object, expected, desired) \
  !!InterlockedCompareExchange128(                                            \
      ((volatile iree_atomic_int128_t*)(object))->__val,                      \
      ((iree_atomic_int128_t*)(desired))->__val[0],                           \
      ((iree_atomic_int128_t*)(desired))->__val[1],                           \
      ((iree_atomic_int128_t*)(expected))->__val)

//==============================================================================
// Atomics for GCC (compatible with both C and
// C++)
//==============================================================================
#elif defined(IREE_COMPILER_GCC)

#define IREE_ATOMIC_VAR_INIT(value) (value)

typedef int32_t iree_atomic_int32_t;
typedef int64_t iree_atomic_int64_t;
typedef __int128 iree_atomic_int128_t;

#ifdef __cplusplus
// Equiv to C++ auto keyword in C++ mode.
#define __iree_auto_type auto
#else
// Only defined in C mode.
#define __iree_auto_type __auto_type
#endif

#define iree_atomic_load_auto(object)                                         \
  __atomic_load_ptr(object, __ATOMIC_SEQ_CST) __extension__({                 \
    __iree_auto_type __atomic_load_ptr = (object);                            \
    __typeof__(*__atomic_load_ptr) __atomic_load_tmp;                         \
    __atomic_load(__atomic_load_ptr, &__atomic_load_tmp, (__ATOMIC_SEQ_CST)); \
    __atomic_load_tmp;                                                        \
  })
#define iree_atomic_store_auto(object, desired)                     \
  __extension__({                                                   \
    __iree_auto_type __atomic_store_ptr = (object);                 \
    __typeof__(*__atomic_store_ptr) __atomic_store_tmp = (desired); \
    __atomic_store(__atomic_store_ptr, &__atomic_store_tmp,         \
                   (__ATOMIC_SEQ_CST));                             \
  })
#define iree_atomic_fetch_add_auto(object, operand) \
  __atomic_fetch_add((object), (operand), __ATOMIC_SEQ_CST)
#define iree_atomic_fetch_sub_auto(object, operand) \
  __atomic_fetch_sub((object), (operand), __ATOMIC_SEQ_CST)
#define iree_atomic_compare_exchange_strong_auto(object, expected, desired) \
  __atomic_compare_exchange_n(object, &(expected), desired, /*weak=*/false, \
                              (__ATOMIC_SEQ_CST), (__ATOMIC_SEQ_CST))

//==============================================================================
// Unsupported architecture
//==============================================================================
#else

#error Compiler does not have supported C11-style atomics

#endif  // IREE_COMPILER_*

// If the compiler can automatically determine the types:
#ifdef iree_atomic_load_auto

#define iree_atomic_load_int32 iree_atomic_load_auto
#define iree_atomic_store_int32 iree_atomic_store_auto
#define iree_atomic_fetch_add_int32 iree_atomic_fetch_add_auto
#define iree_atomic_fetch_sub_int32 iree_atomic_fetch_sub_auto
#define iree_atomic_compare_exchange_strong_int32 \
  iree_atomic_compare_exchange_strong_auto

#define iree_atomic_load_int64 iree_atomic_load_auto
#define iree_atomic_store_int64 iree_atomic_store_auto
#define iree_atomic_fetch_add_int64 iree_atomic_fetch_add_auto
#define iree_atomic_fetch_sub_int64 iree_atomic_fetch_sub_auto
#define iree_atomic_compare_exchange_strong_int64 \
  iree_atomic_compare_exchange_strong_auto

#define iree_atomic_compare_exchange_strong_int128 \
  iree_atomic_compare_exchange_strong_auto

#endif  // iree_atomic_load_auto

//==============================================================================
// Pointer-sized atomics
//==============================================================================

#if UINTPTR_MAX == UINT32_MAX
typedef iree_atomic_int32_t iree_atomic_intptr_t;
#define iree_atomic_load_intptr iree_atomic_load_int32
#define iree_atomic_store_intptr iree_atomic_store_int32
#define iree_atomic_fetch_add_intptr iree_atomic_fetch_add_int32
#define iree_atomic_fetch_sub_intptr iree_atomic_fetch_sub_int32
#define iree_atomic_compare_exchange_strong_intptr \
  iree_atomic_compare_exchange_strong_int32
#else
typedef iree_atomic_int64_t iree_atomic_intptr_t;
#define iree_atomic_load_intptr iree_atomic_load_int64
#define iree_atomic_store_intptr iree_atomic_store_int64
#define iree_atomic_fetch_add_intptr iree_atomic_fetch_add_int64
#define iree_atomic_fetch_sub_intptr iree_atomic_fetch_sub_int64
#define iree_atomic_compare_exchange_strong_intptr \
  iree_atomic_compare_exchange_strong_int64
#endif  // 32/64-bit intptr_t

static_assert(sizeof(iree_atomic_intptr_t) == sizeof(intptr_t),
              "atomic intptr_t must be an intptr_t");

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_ATOMICS_H_
