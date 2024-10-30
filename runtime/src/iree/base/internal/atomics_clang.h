// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_ATOMICS_CLANG_H_
#define IREE_BASE_INTERNAL_ATOMICS_CLANG_H_

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/target_platform.h"

#if defined(IREE_COMPILER_CLANG)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iree_memory_order_e {
  iree_memory_order_relaxed = __ATOMIC_RELAXED,
  iree_memory_order_consume = __ATOMIC_CONSUME,
  iree_memory_order_acquire = __ATOMIC_ACQUIRE,
  iree_memory_order_release = __ATOMIC_RELEASE,
  iree_memory_order_acq_rel = __ATOMIC_ACQ_REL,
  iree_memory_order_seq_cst = __ATOMIC_SEQ_CST,
} iree_memory_order_t;

#define IREE_ATOMIC_VAR_INIT(value) (value)

typedef _Atomic int32_t iree_atomic_int32_t;
typedef _Atomic int64_t iree_atomic_int64_t;
typedef _Atomic uint32_t iree_atomic_uint32_t;
typedef _Atomic uint64_t iree_atomic_uint64_t;
// TODO(#3453): check for __int128 support before using
// typedef _Atomic __int128 iree_atomic_int128_t;
typedef _Atomic intptr_t iree_atomic_intptr_t;

#define iree_atomic_thread_fence(order) __c11_atomic_thread_fence(order)

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_COMPILER_CLANG

#endif  // IREE_BASE_INTERNAL_ATOMICS_CLANG_H_
