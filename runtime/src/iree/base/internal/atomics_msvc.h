// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_ATOMICS_MSVC_H_
#define IREE_BASE_INTERNAL_ATOMICS_MSVC_H_

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/target_platform.h"

#if defined(IREE_COMPILER_MSVC)

// TODO(benvanik): make MSVC's C11 atomic support work.
// It's difficult to detect and has some weird configuration assertions around
// mixed C and C++ code. Support is only present when the
// `/experimental:c11atomics` but that is ignored on /TP (C++) compilation.
// __STDC_NO_ATOMICS__ is not unset when included/enabled so we can't use the
// standard check. Hopefully that'd be fixed if it ever leaves experimental.
#define IREE_ATOMIC_USE_MSVC_C11 0
#if IREE_ATOMIC_USE_MSVC_C11
#include <stdatomic.h>
#endif  // IREE_ATOMIC_USE_MSVC_C11

#if IREE_ATOMIC_USE_MSVC_C11 && defined(atomic_init)

typedef enum iree_memory_order_e {
  iree_memory_order_relaxed = _Atomic_memory_order_relaxed,
  iree_memory_order_consume = _Atomic_memory_order_consume,
  iree_memory_order_acquire = _Atomic_memory_order_acquire,
  iree_memory_order_release = _Atomic_memory_order_release,
  iree_memory_order_acq_rel = _Atomic_memory_order_acq_rel,
  iree_memory_order_seq_cst = _Atomic_memory_order_seq_cst,
} iree_memory_order_t;

#define IREE_ATOMIC_VAR_INIT(value) (value)

typedef _Atomic int32_t iree_atomic_int32_t;
typedef _Atomic int64_t iree_atomic_int64_t;
typedef _Atomic uint32_t iree_atomic_uint32_t;
typedef _Atomic uint64_t iree_atomic_uint64_t;
// TODO(#3453): check for __int128 support before using
// typedef _Atomic __int128 iree_atomic_int128_t;
typedef _Atomic intptr_t iree_atomic_intptr_t;

#define iree_atomic_thread_fence(order) atomic_thread_fence(order)

#define iree_atomic_load(object, order) __c11_atomic_load((object), (order))
#define iree_atomic_store(object, desired, order) \
  __c11_atomic_store((object), (desired), (order))
#define iree_atomic_fetch_add(object, operand, order) \
  __c11_atomic_fetch_add((object), (operand), (order))
#define iree_atomic_fetch_sub(object, operand, order) \
  __c11_atomic_fetch_sub((object), (operand), (order))
#define iree_atomic_fetch_and(object, operand, order) \
  __c11_atomic_fetch_and((object), (operand), (order))
#define iree_atomic_fetch_or(object, operand, order) \
  __c11_atomic_fetch_or((object), (operand), (order))
#define iree_atomic_fetch_xor(object, operand, order) \
  __c11_atomic_fetch_xor((object), (operand), (order))
#define iree_atomic_exchange(object, operand, order) \
  __c11_atomic_exchange((object), (operand), (order))
#define iree_atomic_compare_exchange_strong(object, expected, desired,  \
                                            order_succ, order_fail)     \
  __c11_atomic_compare_exchange_strong((object), (expected), (desired), \
                                       (order_succ), (order_fail))
#define iree_atomic_compare_exchange_weak(object, expected, desired,  \
                                          order_succ, order_fail)     \
  __c11_atomic_compare_exchange_weak((object), (expected), (desired), \
                                     (order_succ), (order_fail))

#elif __cplusplus

// When compiling for C++ we reinterpret atomics as std::atomic<T>. This relies
// on std::atomic on primitive types being lock-free such that the memory for
// each atomic is just the atomic value. We need this special path because MSVC
// doesn't support C features like _Generic in C++.

extern "C++" {
#include <atomic>
}  // extern "C++"

extern "C" {

typedef enum iree_memory_order_e {
  iree_memory_order_relaxed = std::memory_order::memory_order_relaxed,
  iree_memory_order_consume = std::memory_order::memory_order_consume,
  iree_memory_order_acquire = std::memory_order::memory_order_acquire,
  iree_memory_order_release = std::memory_order::memory_order_release,
  iree_memory_order_acq_rel = std::memory_order::memory_order_acq_rel,
  iree_memory_order_seq_cst = std::memory_order::memory_order_seq_cst,
} iree_memory_order_t;

#define IREE_ATOMIC_VAR_INIT(value) (value)

typedef std::atomic<int32_t> iree_atomic_int32_t;
typedef std::atomic<int64_t> iree_atomic_int64_t;
typedef std::atomic<uint32_t> iree_atomic_uint32_t;
typedef std::atomic<uint64_t> iree_atomic_uint64_t;
typedef std::atomic<intptr_t> iree_atomic_intptr_t;

#define iree_atomic_thread_fence(order) std::atomic_thread_fence(order)

#define iree_atomic_load(object, order) \
  std::atomic_load_explicit((object), (std::memory_order)(order))
#define iree_atomic_store(object, desired, order) \
  std::atomic_store_explicit((object), (desired), (std::memory_order)(order))
#define iree_atomic_fetch_add(object, operand, order) \
  std::atomic_fetch_add_explicit((object), (operand), \
                                 (std::memory_order)(order))
#define iree_atomic_fetch_sub(object, operand, order) \
  std::atomic_fetch_sub_explicit((object), (operand), \
                                 (std::memory_order)(order))
#define iree_atomic_fetch_and(object, operand, order) \
  std::atomic_fetch_and_explicit((object), (operand), \
                                 (std::memory_order)(order))
#define iree_atomic_fetch_or(object, operand, order) \
  std::atomic_fetch_or_explicit((object), (operand), (std::memory_order)(order))
#define iree_atomic_fetch_xor(object, operand, order) \
  std::atomic_fetch_xor_explicit((object), (operand), \
                                 (std::memory_order)(order))
#define iree_atomic_exchange(object, operand, order) \
  std::atomic_exchange_explicit((object), (operand), (std::memory_order)(order))
#define iree_atomic_compare_exchange_strong(object, expected, desired,  \
                                            order_succ, order_fail)     \
  std::atomic_compare_exchange_strong_explicit(                         \
      (object), (expected), (desired), (std::memory_order)(order_succ), \
      (std::memory_order)(order_fail))
#define iree_atomic_compare_exchange_weak(object, expected, desired,          \
                                          order_succ, order_fail)             \
  std::atomic_compare_exchange_weak_explicit((object), (expected), (desired), \
                                             (std::memory_order)(order_succ), \
                                             (std::memory_order)(order_fail))

}  // extern "C"

#else

// When compiling in C we can use _Generic to automatically route to the
// builtins that change their name based on the atomic type. This implementation
// is not good: it ignores memory order entirely and uses the full barrier
// implied by any of the _Interlocked* builtins. There are some variants of the
// builtins that we could use based on the order but their support across
// targets differs. Hopefully ~soon we can use C11 atomics directly and drop
// this code path.

typedef enum iree_memory_order_e {
  iree_memory_order_relaxed = 0u,
  iree_memory_order_consume,
  iree_memory_order_acquire,
  iree_memory_order_release,
  iree_memory_order_acq_rel,
  iree_memory_order_seq_cst,
} iree_memory_order_t;

#define IREE_ATOMIC_VAR_INIT(value) (value)

typedef int32_t iree_atomic_int32_t;
typedef int64_t iree_atomic_int64_t;
typedef uint32_t iree_atomic_uint32_t;
typedef uint64_t iree_atomic_uint64_t;
typedef intptr_t iree_atomic_intptr_t;

#define iree_atomic_thread_fence(order) MemoryBarrier()

#define iree_atomic_load(object, order)                          \
  _Generic((object),                                             \
      iree_atomic_int32_t *: _InterlockedExchangeAdd(            \
                               (volatile int32_t*)(object), 0),  \
      iree_atomic_int64_t *: _InterlockedExchangeAdd64(          \
                               (volatile int64_t*)(object), 0),  \
      iree_atomic_uint32_t *: _InterlockedExchangeAdd(           \
                                (volatile int32_t*)(object), 0), \
      iree_atomic_uint64_t *: _InterlockedExchangeAdd64(         \
                                (volatile int64_t*)(object), 0))
#define iree_atomic_store(object, desired, order)                              \
  _Generic((object),                                                           \
      iree_atomic_int32_t *: _InterlockedExchange((volatile int32_t*)(object), \
                                                  (int32_t)(desired)),         \
      iree_atomic_int64_t *: _InterlockedExchange64(                           \
                               (volatile int64_t*)(object),                    \
                               (int64_t)(desired)),                            \
      iree_atomic_uint32_t *: _InterlockedExchange(                            \
                                (volatile int32_t*)(object),                   \
                                (int32_t)(desired)),                           \
      iree_atomic_uint64_t *: _InterlockedExchange64(                          \
                                (volatile int64_t*)(object),                   \
                                (int64_t)(desired)))
#define iree_atomic_fetch_add(object, operand, order)        \
  _Generic((object),                                         \
      iree_atomic_int32_t *: _InterlockedExchangeAdd(        \
                               (volatile int32_t*)(object),  \
                               (int32_t)(operand)),          \
      iree_atomic_int64_t *: _InterlockedExchangeAdd64(      \
                               (volatile int64_t*)(object),  \
                               (int64_t)(operand)),          \
      iree_atomic_uint32_t *: _InterlockedExchangeAdd(       \
                                (volatile int32_t*)(object), \
                                (int32_t)(operand)),         \
      iree_atomic_uint64_t *: _InterlockedExchangeAdd64(     \
                                (volatile int64_t*)(object), \
                                (int64_t)(operand)))
#define iree_atomic_fetch_sub(object, operand, order)        \
  _Generic((object),                                         \
      iree_atomic_int32_t *: _InterlockedExchangeAdd(        \
                               (volatile int32_t*)(object),  \
                               -((int32_t)(operand))),       \
      iree_atomic_int64_t *: _InterlockedExchangeAdd64(      \
                               (volatile int64_t*)(object),  \
                               -((int64_t)(operand))),       \
      iree_atomic_uint32_t *: _InterlockedExchangeAdd(       \
                                (volatile int32_t*)(object), \
                                -((int32_t)(operand))),      \
      iree_atomic_uint64_t *: _InterlockedExchangeAdd64(     \
                                (volatile int64_t*)(object), \
                                -((int64_t)(operand))))
#define iree_atomic_fetch_and(object, operand, order)                        \
  _Generic((object),                                                         \
      iree_atomic_int32_t *: _InterlockedAnd((volatile int32_t*)(object),    \
                                             (int32_t)(operand)),            \
      iree_atomic_int64_t *: _InterlockedAnd64((volatile int64_t*)(object),  \
                                               (int64_t)(operand)),          \
      iree_atomic_uint32_t *: _InterlockedAnd((volatile int32_t*)(object),   \
                                              (int32_t)(operand)),           \
      iree_atomic_uint64_t *: _InterlockedAnd64((volatile int64_t*)(object), \
                                                (int64_t)(operand)))
#define iree_atomic_fetch_or(object, operand, order)                        \
  _Generic((object),                                                        \
      iree_atomic_int32_t *: _InterlockedOr((volatile int32_t*)(object),    \
                                            (int32_t)(operand)),            \
      iree_atomic_int64_t *: _InterlockedOr64((volatile int64_t*)(object),  \
                                              (int64_t)(operand)),          \
      iree_atomic_uint32_t *: _InterlockedOr((volatile int32_t*)(object),   \
                                             (int32_t)(operand)),           \
      iree_atomic_uint64_t *: _InterlockedOr64((volatile int64_t*)(object), \
                                               (int64_t)(operand)))
#define iree_atomic_fetch_xor(object, operand, order)                        \
  _Generic((object),                                                         \
      iree_atomic_int32_t *: _InterlockedXor((volatile int32_t*)(object),    \
                                             (int32_t)(operand)),            \
      iree_atomic_int64_t *: _InterlockedXor64((volatile int64_t*)(object),  \
                                               (int64_t)(operand)),          \
      iree_atomic_uint32_t *: _InterlockedXor((volatile int32_t*)(object),   \
                                              (int32_t)(operand)),           \
      iree_atomic_uint64_t *: _InterlockedXor64((volatile int64_t*)(object), \
                                                (int64_t)(operand)))
#define iree_atomic_exchange(object, desired, order)                           \
  _Generic((object),                                                           \
      iree_atomic_int32_t *: _InterlockedExchange((volatile int32_t*)(object), \
                                                  (int32_t)(desired)),         \
      iree_atomic_int64_t *: _InterlockedExchange64(                           \
                               (volatile int64_t*)(object),                    \
                               (int64_t)(desired)),                            \
      iree_atomic_uint32_t *: _InterlockedExchange(                            \
                                (volatile int32_t*)(object),                   \
                                (int32_t)(desired)),                           \
      iree_atomic_uint64_t *: _InterlockedExchange64(                          \
                                (volatile int64_t*)(object),                   \
                                (int64_t)(desired)))
#define iree_atomic_compare_exchange_strong(object, expected, desired,        \
                                            order_succ, order_fail)           \
  _Generic((object),                                                          \
      iree_atomic_int32_t *: iree_atomic_compare_exchange_strong_int32_impl(  \
                               (volatile iree_atomic_int32_t*)(object),       \
                               (int32_t*)(expected), (int32_t)(desired),      \
                               (order_succ), (order_fail)),                   \
      iree_atomic_int64_t *: iree_atomic_compare_exchange_strong_int64_impl(  \
                               (volatile iree_atomic_int64_t*)(object),       \
                               (int64_t*)(expected), (int64_t)(desired),      \
                               (order_succ), (order_fail)),                   \
      iree_atomic_uint32_t *: iree_atomic_compare_exchange_strong_int32_impl( \
                                (volatile iree_atomic_int32_t*)(object),      \
                                (int32_t*)(expected), (int32_t)(desired),     \
                                (order_succ), (order_fail)),                  \
      iree_atomic_uint64_t *: iree_atomic_compare_exchange_strong_int64_impl( \
                                (volatile iree_atomic_int64_t*)(object),      \
                                (int64_t*)(expected), (int64_t)(desired),     \
                                (order_succ), (order_fail)))
#define iree_atomic_compare_exchange_weak iree_atomic_compare_exchange_strong

static inline bool iree_atomic_compare_exchange_strong_int32_impl(
    volatile iree_atomic_int32_t* object, int32_t* expected, int32_t desired,
    iree_memory_order_t order_succ, iree_memory_order_t order_fail) {
  int32_t expected_value = *expected;
  int32_t old_value = InterlockedCompareExchange((volatile LONG*)object,
                                                 desired, expected_value);
  if (old_value == expected_value) {
    return true;
  } else {
    *expected = old_value;
    return false;
  }
}

static inline bool iree_atomic_compare_exchange_strong_int64_impl(
    volatile iree_atomic_int64_t* object, int64_t* expected, int64_t desired,
    iree_memory_order_t order_succ, iree_memory_order_t order_fail) {
  int64_t expected_value = *expected;
  int64_t old_value = InterlockedCompareExchange64((volatile LONG64*)object,
                                                   desired, expected_value);
  if (old_value == expected_value) {
    return true;
  } else {
    *expected = old_value;
    return false;
  }
}

#endif  // IREE_ATOMIC_USE_MSVC_C11

#endif  // IREE_COMPILER_MSVC

#endif  // IREE_BASE_INTERNAL_ATOMICS_MSVC_H_
