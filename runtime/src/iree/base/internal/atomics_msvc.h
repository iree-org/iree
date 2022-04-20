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

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iree_memory_order_e {
  iree_memory_order_relaxed,
  iree_memory_order_consume,
  iree_memory_order_acquire,
  iree_memory_order_release,
  iree_memory_order_acq_rel,
  iree_memory_order_seq_cst,
} iree_memory_order_t;

#define IREE_ATOMIC_VAR_INIT(value) \
  { (value) }

typedef struct {
  int32_t __val;
} iree_atomic_int32_t;
typedef struct {
  int64_t __val;
} iree_atomic_int64_t;
// typedef __declspec(align(16)) struct {
//   uint64_t __val[2];
// } iree_atomic_int128_t;
typedef struct {
  intptr_t __val;
} iree_atomic_intptr_t;

#define iree_atomic_load_int32(object, order) \
  InterlockedExchangeAdd((volatile LONG*)object, 0)
#define iree_atomic_store_int32(object, desired, order) \
  InterlockedExchange((volatile LONG*)object, desired)
#define iree_atomic_fetch_add_int32(object, operand, order) \
  InterlockedExchangeAdd((volatile LONG*)object, operand)
#define iree_atomic_fetch_sub_int32(object, operand, order) \
  InterlockedExchangeAdd((volatile LONG*)object, -((int32_t)(operand)))
#define iree_atomic_fetch_and_int32(object, operand, order) \
  InterlockedAnd((volatile LONG*)object, operand)
#define iree_atomic_fetch_or_int32(object, operand, order) \
  InterlockedOr((volatile LONG*)object, operand)
#define iree_atomic_fetch_xor_int32(object, operand, order) \
  InterlockedXor((volatile LONG*)object, operand)
#define iree_atomic_exchange_int32(object, desired, order) \
  InterlockedExchange((volatile LONG*)object, desired)
#define iree_atomic_compare_exchange_strong_int32(object, expected, desired, \
                                                  order_succ, order_fail)    \
  iree_atomic_compare_exchange_strong_int32_impl(                            \
      (volatile iree_atomic_int32_t*)(object), (int32_t*)(expected),         \
      (int32_t)(desired), (order_succ), (order_fail))
#define iree_atomic_compare_exchange_weak_int32 \
  iree_atomic_compare_exchange_strong_int32

#define iree_atomic_load_int64(object, order) \
  InterlockedExchangeAdd64((volatile LONG64*)object, 0)
#define iree_atomic_store_int64(object, desired, order) \
  InterlockedExchange64((volatile LONG64*)object, (LONG64)desired)
#define iree_atomic_fetch_add_int64(object, operand, order) \
  InterlockedExchangeAdd64((volatile LONG64*)object, (LONG64)operand)
#define iree_atomic_fetch_sub_int64(object, operand, order) \
  InterlockedExchangeAdd64((volatile LONG64*)object, -(operand))
#define iree_atomic_fetch_and_int64(object, operand, order) \
  InterlockedAnd64((volatile LONG64*)object, operand)
#define iree_atomic_fetch_or_int64(object, operand, order) \
  InterlockedOr64((volatile LONG64*)object, operand)
#define iree_atomic_fetch_xor_int64(object, operand, order) \
  InterlockedXor64((volatile LONG64*)object, operand)
#define iree_atomic_exchange_int64(object, desired, order) \
  InterlockedExchange64((volatile LONG64*)object, desired)
#define iree_atomic_compare_exchange_strong_int64(object, expected, desired, \
                                                  order_succ, order_fail)    \
  iree_atomic_compare_exchange_strong_int64_impl(                            \
      (volatile iree_atomic_int64_t*)(object), (int64_t*)(expected),         \
      (int64_t)(desired), (order_succ), (order_fail))
#define iree_atomic_compare_exchange_weak_int64 \
  iree_atomic_compare_exchange_strong_int64

#define iree_atomic_thread_fence(order) MemoryBarrier()

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

#define iree_atomic_thread_fence(order) MemoryBarrier()

// There are no pointer-width atomic ops in MSVC so we need to specialize based
// on the pointer size.
#if defined(IREE_PTR_SIZE_32)
#define iree_atomic_load_intptr(object, order) \
  (intptr_t) iree_atomic_load_int32((iree_atomic_int32_t*)(object), (order))
#define iree_atomic_store_intptr(object, desired, order)             \
  (intptr_t) iree_atomic_store_int32((iree_atomic_int32_t*)(object), \
                                     (int32_t)(desired), (order))
#define iree_atomic_fetch_add_intptr(object, operand, order)             \
  (intptr_t) iree_atomic_fetch_add_int32((iree_atomic_int32_t*)(object), \
                                         (int32_t)(operand), (order))
#define iree_atomic_fetch_sub_intptr(object, operand, order)             \
  (intptr_t) iree_atomic_fetch_sub_int32((iree_atomic_int32_t*)(object), \
                                         (int32_t)(operand), (order))
#define iree_atomic_exchange_intptr(object, desired, order)             \
  (intptr_t) iree_atomic_exchange_int32((iree_atomic_int32_t*)(object), \
                                        (int32_t)(desired), (order))
#define iree_atomic_compare_exchange_strong_intptr(object, expected, desired, \
                                                   order_succ, order_fail)    \
  iree_atomic_compare_exchange_strong_int32(                                  \
      (iree_atomic_int32_t*)(object), (int32_t*)(expected),                   \
      (int32_t)(desired), (order_succ), (order_fail))
#define iree_atomic_compare_exchange_weak_intptr \
  iree_atomic_compare_exchange_strong_intptr
#else
#define iree_atomic_load_intptr(object, order) \
  (intptr_t) iree_atomic_load_int64((iree_atomic_int64_t*)(object), (order))
#define iree_atomic_store_intptr(object, desired, order)             \
  (intptr_t) iree_atomic_store_int64((iree_atomic_int64_t*)(object), \
                                     (int64_t)(desired), (order))
#define iree_atomic_fetch_add_intptr(object, operand, order)             \
  (intptr_t) iree_atomic_fetch_add_int64((iree_atomic_int64_t*)(object), \
                                         (int64_t)(operand), (order))
#define iree_atomic_fetch_sub_intptr(object, operand, order)             \
  (intptr_t) iree_atomic_fetch_sub_int64((iree_atomic_int64_t*)(object), \
                                         (int64_t)(operand), (order))
#define iree_atomic_exchange_intptr(object, desired, order)             \
  (intptr_t) iree_atomic_exchange_int64((iree_atomic_int64_t*)(object), \
                                        (int64_t)(desired), (order))
#define iree_atomic_compare_exchange_strong_intptr(object, expected, desired, \
                                                   order_succ, order_fail)    \
  iree_atomic_compare_exchange_strong_int64(                                  \
      (iree_atomic_int64_t*)(object), (int64_t*)(expected),                   \
      (int64_t)(desired), (order_succ), (order_fail))
#define iree_atomic_compare_exchange_weak_intptr \
  iree_atomic_compare_exchange_strong_intptr
#endif  // IREE_PTR_SIZE_32

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_COMPILER_MSVC

#endif  // IREE_BASE_INTERNAL_ATOMICS_MSVC_H_
