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
// to intptr_t for now). We need this for non-C11-compliant platforms (MSVC),
// but it has the added benefit of not conflicting with <atomic> (these two
// files cannot be included in the same compilation unit... great design). There
// shouldn't be any difference between what we do here and what any
// implementation would do with the platform atomic functions so it's used
// everywhere.
//
// https://en.cppreference.com/w/c/atomic
//
// TODO(benvanik): configuration for single-threaded mode to disable atomics.

#ifndef IREE_BASE_ATOMICS_H_
#define IREE_BASE_ATOMICS_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(IREE_COMPILER_CLANG)
// Emulate C11 atomics with builtins.
typedef _Atomic intptr_t iree_atomic_intptr_t;
#define IREE_ATOMIC_VAR_INIT(value) (value)
#define iree_atomic_load(object) __c11_atomic_load(object, __ATOMIC_SEQ_CST)
#define iree_atomic_store(object, desired) \
  __c11_atomic_store(object, desired, __ATOMIC_SEQ_CST)
#define iree_atomic_fetch_add(object, operand) \
  __c11_atomic_fetch_add(object, operand, __ATOMIC_SEQ_CST)
#define iree_atomic_fetch_sub(object, operand) \
  __c11_atomic_fetch_sub(object, operand, __ATOMIC_SEQ_CST)

#elif defined(IREE_COMPILER_MSVC)
// Emulate C11 atomics with Interlocked win32 APIs.
// NOTE: currently assumes sizeof(intptr_t) == 8.
typedef struct {
  intptr_t __val;
} iree_atomic_intptr_t;
#define IREE_ATOMIC_VAR_INIT(value) \
  { (value) }
#define iree_atomic_load(object) \
  InterlockedExchangeAdd64((volatile LONGLONG*)object, 0)
#define iree_atomic_store(object, desired) \
  InterlockedExchange64((volatile LONGLONG*)object, desired)
#define iree_atomic_fetch_add(object, operand) \
  InterlockedExchangeAdd64((volatile LONGLONG*)object, operand)
#define iree_atomic_fetch_sub(object, operand) \
  InterlockedExchangeAdd64((volatile LONGLONG*)object, -(operand))

#elif defined(IREE_COMPILER_GCC)
// Emulate atomics for GCC in a way that is compatible for inclusion in
// both C and C++ modes.
#ifdef __cplusplus
// Equiv to C++ auto keyword in C++ mode.
#define __iree_auto_type auto
#else
// Only defined in C mode.
#define __iree_auto_type __auto_type
#endif
typedef __INTPTR_TYPE__ iree_atomic_intptr_t;
#define IREE_ATOMIC_VAR_INIT(value) (value)
#define iree_atomic_load(object)                                              \
  __atomic_load_ptr(object, __ATOMIC_SEQ_CST) __extension__({                 \
    __iree_auto_type __atomic_load_ptr = (object);                            \
    __typeof__(*__atomic_load_ptr) __atomic_load_tmp;                         \
    __atomic_load(__atomic_load_ptr, &__atomic_load_tmp, (__ATOMIC_SEQ_CST)); \
    __atomic_load_tmp;                                                        \
  })
#define iree_atomic_store(object, desired)                          \
  __extension__({                                                   \
    __iree_auto_type __atomic_store_ptr = (object);                 \
    __typeof__(*__atomic_store_ptr) __atomic_store_tmp = (desired); \
    __atomic_store(__atomic_store_ptr, &__atomic_store_tmp,         \
                   (__ATOMIC_SEQ_CST));                             \
  })
#define iree_atomic_fetch_add(object, operand) \
  __atomic_fetch_add((object), (operand), __ATOMIC_SEQ_CST)
#define iree_atomic_fetch_sub(object, operand) \
  __atomic_fetch_sub((object), (operand), __ATOMIC_SEQ_CST)

#else
#error "compiler does not have supported C11-style atomics"
#endif  // IREE_COMPILER_*

static_assert(sizeof(iree_atomic_intptr_t) == sizeof(intptr_t),
              "atomic intptr_t must be an intptr_t");

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_ATOMICS_H_
