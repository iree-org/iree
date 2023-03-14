// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_ATOMICS_DISABLED_H_
#define IREE_BASE_INTERNAL_ATOMICS_DISABLED_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/config.h"
#include "iree/base/target_platform.h"

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE

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

#define IREE_ATOMIC_VAR_INIT(value) (value)

typedef int32_t iree_atomic_int32_t;
typedef int64_t iree_atomic_int64_t;
// TODO(#3453): check for __int128 support before using
// typedef __int128 iree_atomic_int128_t;
typedef intptr_t iree_atomic_intptr_t;

#define iree_atomic_load_int32(object, order) (*(object))
#define iree_atomic_store_int32(object, desired, order) (*(object) = (desired))
#define iree_atomic_fetch_add_int32(object, operand, order)                 \
  iree_atomic_fetch_add_int32_impl((volatile iree_atomic_int32_t*)(object), \
                                   (int32_t)(operand))
#define iree_atomic_fetch_sub_int32(object, operand, order)                 \
  iree_atomic_fetch_add_int32_impl((volatile iree_atomic_int32_t*)(object), \
                                   -(int32_t)(operand))
#define iree_atomic_fetch_and_int32(object, operand, order)                 \
  iree_atomic_fetch_and_int32_impl((volatile iree_atomic_int32_t*)(object), \
                                   (int32_t)(operand))
#define iree_atomic_fetch_or_int32(object, operand, order)                 \
  iree_atomic_fetch_or_int32_impl((volatile iree_atomic_int32_t*)(object), \
                                  (int32_t)(operand))
#define iree_atomic_fetch_xor_int32(object, operand, order)                 \
  iree_atomic_fetch_xor_int32_impl((volatile iree_atomic_int32_t*)(object), \
                                   (int32_t)(operand))
#define iree_atomic_exchange_int32(object, desired, order) \
  iree_atomic_fetch_exchange_int32_impl(                   \
      (volatile iree_atomic_int32_t*)(object), (int32_t)(desired))
#define iree_atomic_compare_exchange_strong_int32(object, expected, desired, \
                                                  order_succ, order_fail)    \
  iree_atomic_compare_exchange_int32_impl(                                   \
      (volatile iree_atomic_int32_t*)(object), (int32_t*)(expected),         \
      (int32_t)(desired))
#define iree_atomic_compare_exchange_weak_int32 \
  iree_atomic_compare_exchange_strong_int32

#define iree_atomic_load_int64(object, order) (*(object))
#define iree_atomic_store_int64(object, desired, order) (*(object) = (desired))
#define iree_atomic_fetch_add_int64(object, operand, order)                 \
  iree_atomic_fetch_add_int64_impl((volatile iree_atomic_int64_t*)(object), \
                                   (int64_t)(operand))
#define iree_atomic_fetch_sub_int64(object, operand, order)                 \
  iree_atomic_fetch_add_int64_impl((volatile iree_atomic_int64_t*)(object), \
                                   -(int64_t)(operand))
#define iree_atomic_fetch_and_int64(object, operand, order)                 \
  iree_atomic_fetch_and_int64_impl((volatile iree_atomic_int64_t*)(object), \
                                   (int64_t)(operand))
#define iree_atomic_fetch_or_int64(object, operand, order)                 \
  iree_atomic_fetch_or_int64_impl((volatile iree_atomic_int64_t*)(object), \
                                  (int64_t)(operand))
#define iree_atomic_fetch_xor_int64(object, operand, order)                 \
  iree_atomic_fetch_xor_int64_impl((volatile iree_atomic_int64_t*)(object), \
                                   (int64_t)(operand))
#define iree_atomic_exchange_int64(object, desired, order) \
  iree_atomic_fetch_exchange_int64_impl(                   \
      (volatile iree_atomic_int64_t*)(object), (int64_t)(desired))
#define iree_atomic_compare_exchange_strong_int64(object, expected, desired, \
                                                  order_succ, order_fail)    \
  iree_atomic_compare_exchange_int64_impl(                                   \
      (volatile iree_atomic_int64_t*)(object), (int64_t*)(expected),         \
      (int64_t)(desired))
#define iree_atomic_compare_exchange_weak_int64 \
  iree_atomic_compare_exchange_strong_int64

static inline int32_t iree_atomic_fetch_add_int32_impl(
    volatile iree_atomic_int32_t* object, int32_t operand) {
  int32_t original = *object;
  *object += operand;
  return original;
}

static inline int32_t iree_atomic_fetch_and_int32_impl(
    volatile iree_atomic_int32_t* object, int32_t operand) {
  int32_t original = *object;
  *object &= operand;
  return original;
}

static inline int32_t iree_atomic_fetch_or_int32_impl(
    volatile iree_atomic_int32_t* object, int32_t operand) {
  int32_t original = *object;
  *object |= operand;
  return original;
}

static inline int32_t iree_atomic_fetch_xor_int32_impl(
    volatile iree_atomic_int32_t* object, int32_t operand) {
  int32_t original = *object;
  *object ^= operand;
  return original;
}

static inline int32_t iree_atomic_fetch_exchange_int32_impl(
    volatile iree_atomic_int32_t* object, int32_t desired) {
  int32_t original = *object;
  *object = desired;
  return original;
}

static inline bool iree_atomic_compare_exchange_int32_impl(
    volatile iree_atomic_int32_t* object, int32_t* expected, int32_t desired) {
  if (*object == *expected) {
    *object = desired;
    return true;
  } else {
    *expected = *object;
    return false;
  }
}

static inline int64_t iree_atomic_fetch_add_int64_impl(
    volatile iree_atomic_int64_t* object, int64_t operand) {
  int64_t original = *object;
  *object += operand;
  return original;
}

static inline int64_t iree_atomic_fetch_and_int64_impl(
    volatile iree_atomic_int64_t* object, int64_t operand) {
  int64_t original = *object;
  *object &= operand;
  return original;
}

static inline int64_t iree_atomic_fetch_or_int64_impl(
    volatile iree_atomic_int64_t* object, int64_t operand) {
  int64_t original = *object;
  *object |= operand;
  return original;
}

static inline int64_t iree_atomic_fetch_xor_int64_impl(
    volatile iree_atomic_int64_t* object, int64_t operand) {
  int64_t original = *object;
  *object ^= operand;
  return original;
}

static inline int64_t iree_atomic_fetch_exchange_int64_impl(
    volatile iree_atomic_int64_t* object, int64_t desired) {
  int64_t original = *object;
  *object = desired;
  return original;
}

static inline bool iree_atomic_compare_exchange_int64_impl(
    volatile iree_atomic_int64_t* object, int64_t* expected, int64_t desired) {
  if (*object == *expected) {
    *object = desired;
    return true;
  } else {
    *expected = *object;
    return false;
  }
}

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

#define iree_atomic_thread_fence(order)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#endif  // IREE_BASE_INTERNAL_ATOMICS_DISABLED_H_
