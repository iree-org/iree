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
// TODO(#3453): check for __int128 support before using
// typedef __int128 iree_atomic_int128_t;
typedef intptr_t iree_atomic_intptr_t;

#define iree_atomic_thread_fence(order)

#ifdef __cplusplus

extern "C++" {

#define iree_atomic_load(object, order) (*(object))
#define iree_atomic_store(object, desired, order) (*(object) = (desired))
#define iree_atomic_fetch_add(object, operand, order) \
  iree_atomic_fetch_add_impl((object), (operand))
#define iree_atomic_fetch_sub(object, operand, order) \
  iree_atomic_fetch_sub_impl((object), (operand))
#define iree_atomic_fetch_and(object, operand, order) \
  iree_atomic_fetch_and_impl((object), (operand))
#define iree_atomic_fetch_or(object, operand, order) \
  iree_atomic_fetch_or_impl((object), (operand))
#define iree_atomic_fetch_xor(object, operand, order) \
  iree_atomic_fetch_xor_impl((object), (operand))
#define iree_atomic_exchange(object, desired, order) \
  iree_atomic_fetch_exchange_impl((object), (desired))
#define iree_atomic_compare_exchange_strong(object, expected, desired, \
                                            order_succ, order_fail)    \
  iree_atomic_compare_exchange_impl((object), (expected), (desired))
#define iree_atomic_compare_exchange_weak iree_atomic_compare_exchange_strong

template <typename T, typename V>
static inline T iree_atomic_fetch_add_impl(volatile T* object, V operand) {
  T original = *object;
  *object += operand;
  return original;
}

template <typename T, typename V>
static inline T iree_atomic_fetch_sub_impl(volatile T* object, V operand) {
  T original = *object;
  *object -= operand;
  return original;
}

template <typename T, typename V>
static inline T iree_atomic_fetch_and_impl(volatile T* object, V operand) {
  T original = *object;
  *object &= operand;
  return original;
}

template <typename T, typename V>
static inline T iree_atomic_fetch_or_impl(volatile T* object, V operand) {
  T original = *object;
  *object |= operand;
  return original;
}

template <typename T, typename V>
static inline T iree_atomic_fetch_xor_impl(volatile T* object, V operand) {
  T original = *object;
  *object ^= operand;
  return original;
}

template <typename T, typename V>
static inline T iree_atomic_fetch_exchange_impl(volatile T* object, V desired) {
  T original = *object;
  *object = desired;
  return original;
}

template <typename T, typename V>
static inline bool iree_atomic_compare_exchange_impl(volatile T* object,
                                                     V* expected, V desired) {
  if (*object == *expected) {
    *object = desired;
    return true;
  } else {
    *expected = *object;
    return false;
  }
}

}  // extern "C"

#else

#define iree_atomic_load(object, order) (*(object))
#define iree_atomic_store(object, desired, order) (*(object) = (desired))
#define iree_atomic_fetch_add(object, operand, order)                     \
  _Generic((object),                                                      \
      iree_atomic_int32_t *: iree_atomic_fetch_add_int32_impl(            \
                               (volatile iree_atomic_int32_t*)(object),   \
                               (int32_t)(operand)),                       \
      iree_atomic_int64_t *: iree_atomic_fetch_add_int64_impl(            \
                               (volatile iree_atomic_int64_t*)(object),   \
                               (int64_t)(operand)),                       \
      iree_atomic_uint32_t *: iree_atomic_fetch_add_uint32_impl(          \
                                (volatile iree_atomic_uint32_t*)(object), \
                                (uint32_t)(operand)),                     \
      iree_atomic_uint64_t *: iree_atomic_fetch_add_uint64_impl(          \
                                (volatile iree_atomic_uint64_t*)(object), \
                                (uint64_t)(operand)))
#define iree_atomic_fetch_sub(object, operand, order)                     \
  _Generic((object),                                                      \
      iree_atomic_int32_t *: iree_atomic_fetch_sub_int32_impl(            \
                               (volatile iree_atomic_int32_t*)(object),   \
                               (int32_t)(operand)),                       \
      iree_atomic_int64_t *: iree_atomic_fetch_sub_int64_impl(            \
                               (volatile iree_atomic_int64_t*)(object),   \
                               (int64_t)(operand)),                       \
      iree_atomic_uint32_t *: iree_atomic_fetch_sub_uint32_impl(          \
                                (volatile iree_atomic_uint32_t*)(object), \
                                (uint32_t)(operand)),                     \
      iree_atomic_uint64_t *: iree_atomic_fetch_sub_uint64_impl(          \
                                (volatile iree_atomic_uint64_t*)(object), \
                                (uint64_t)(operand)))
#define iree_atomic_fetch_and(object, operand, order)                    \
  _Generic((object),                                                     \
      iree_atomic_int32_t *: iree_atomic_fetch_and_int32_impl(           \
                               (volatile iree_atomic_int32_t*)(object),  \
                               (int32_t)(operand)),                      \
      iree_atomic_int64_t *: iree_atomic_fetch_and_int64_impl(           \
                               (volatile iree_atomic_int64_t*)(object),  \
                               (int64_t)(operand)),                      \
      iree_atomic_uint32_t *: iree_atomic_fetch_and_int32_impl(          \
                                (volatile iree_atomic_int32_t*)(object), \
                                (int32_t)(operand)),                     \
      iree_atomic_uint64_t *: iree_atomic_fetch_and_int64_impl(          \
                                (volatile iree_atomic_int64_t*)(object), \
                                (int64_t)(operand)))
#define iree_atomic_fetch_or(object, operand, order)                     \
  _Generic((object),                                                     \
      iree_atomic_int32_t *: iree_atomic_fetch_or_int32_impl(            \
                               (volatile iree_atomic_int32_t*)(object),  \
                               (int32_t)(operand)),                      \
      iree_atomic_int64_t *: iree_atomic_fetch_or_int64_impl(            \
                               (volatile iree_atomic_int64_t*)(object),  \
                               (int64_t)(operand)),                      \
      iree_atomic_uint32_t *: iree_atomic_fetch_or_int32_impl(           \
                                (volatile iree_atomic_int32_t*)(object), \
                                (int32_t)(operand)),                     \
      iree_atomic_uint64_t *: iree_atomic_fetch_or_int64_impl(           \
                                (volatile iree_atomic_int64_t*)(object), \
                                (int64_t)(operand)))
#define iree_atomic_fetch_xor(object, operand, order)                    \
  _Generic((object),                                                     \
      iree_atomic_int32_t *: iree_atomic_fetch_xor_int32_impl(           \
                               (volatile iree_atomic_int32_t*)(object),  \
                               (int32_t)(operand)),                      \
      iree_atomic_int64_t *: iree_atomic_fetch_xor_int64_impl(           \
                               (volatile iree_atomic_int64_t*)(object),  \
                               (int64_t)(operand)),                      \
      iree_atomic_uint32_t *: iree_atomic_fetch_xor_int32_impl(          \
                                (volatile iree_atomic_int32_t*)(object), \
                                (int32_t)(operand)),                     \
      iree_atomic_uint64_t *: iree_atomic_fetch_xor_int64_impl(          \
                                (volatile iree_atomic_int64_t*)(object), \
                                (int64_t)(operand)))
#define iree_atomic_exchange(object, desired, order)                     \
  _Generic((object),                                                     \
      iree_atomic_int32_t *: iree_atomic_fetch_exchange_int32_impl(      \
                               (volatile iree_atomic_int32_t*)(object),  \
                               (int32_t)(desired)),                      \
      iree_atomic_int64_t *: iree_atomic_fetch_exchange_int64_impl(      \
                               (volatile iree_atomic_int64_t*)(object),  \
                               (int64_t)(desired)),                      \
      iree_atomic_uint32_t *: iree_atomic_fetch_exchange_int32_impl(     \
                                (volatile iree_atomic_int32_t*)(object), \
                                (int32_t)(desired)),                     \
      iree_atomic_uint64_t *: iree_atomic_fetch_exchange_int64_impl(     \
                                (volatile iree_atomic_int64_t*)(object), \
                                (int64_t)(desired)))
#define iree_atomic_compare_exchange_strong(object, expected, desired,     \
                                            order_succ, order_fail)        \
  _Generic((object),                                                       \
      iree_atomic_int32_t *: iree_atomic_compare_exchange_int32_impl(      \
                               (volatile iree_atomic_int32_t*)(object),    \
                               (int32_t*)(expected), (int32_t)(desired)),  \
      iree_atomic_int64_t *: iree_atomic_compare_exchange_int64_impl(      \
                               (volatile iree_atomic_int64_t*)(object),    \
                               (int64_t*)(expected), (int64_t)(desired)),  \
      iree_atomic_uint32_t *: iree_atomic_compare_exchange_int32_impl(     \
                                (volatile iree_atomic_int32_t*)(object),   \
                                (int32_t*)(expected), (int32_t)(desired)), \
      iree_atomic_uint64_t *: iree_atomic_compare_exchange_int64_impl(     \
                                (volatile iree_atomic_int64_t*)(object),   \
                                (int64_t*)(expected), (int64_t)(desired)))
#define iree_atomic_compare_exchange_weak iree_atomic_compare_exchange_strong

static inline int32_t iree_atomic_fetch_add_int32_impl(
    volatile iree_atomic_int32_t* object, int32_t operand) {
  int32_t original = *object;
  *object += operand;
  return original;
}

static inline int32_t iree_atomic_fetch_sub_int32_impl(
    volatile iree_atomic_int32_t* object, int32_t operand) {
  int32_t original = *object;
  *object -= operand;
  return original;
}

static inline int32_t iree_atomic_fetch_add_uint32_impl(
    volatile iree_atomic_int32_t* object, uint32_t operand) {
  uint32_t original = *object;
  *object += operand;
  return original;
}

static inline int32_t iree_atomic_fetch_sub_uint32_impl(
    volatile iree_atomic_uint32_t* object, uint32_t operand) {
  uint32_t original = *object;
  *object -= operand;
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

static inline int64_t iree_atomic_fetch_sub_int64_impl(
    volatile iree_atomic_int64_t* object, int64_t operand) {
  int64_t original = *object;
  *object -= operand;
  return original;
}

static inline int64_t iree_atomic_fetch_add_uint64_impl(
    volatile iree_atomic_uint64_t* object, uint64_t operand) {
  uint64_t original = *object;
  *object += operand;
  return original;
}

static inline int64_t iree_atomic_fetch_sub_uint64_impl(
    volatile iree_atomic_uint64_t* object, uint64_t operand) {
  uint64_t original = *object;
  *object -= operand;
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

#endif  // __cplusplus

#endif  // IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#endif  // IREE_BASE_INTERNAL_ATOMICS_DISABLED_H_
