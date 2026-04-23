// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_AFFINITY_SET_H_
#define IREE_TASK_AFFINITY_SET_H_

#include <limits.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/math.h"
#include "iree/base/target_platform.h"
#include "iree/task/tuning.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TODO(benvanik): if IREE_TASK_EXECUTOR_MAX_WORKER_COUNT <= 32 then switch
// these to using the 32-bit primitives. No real effect on larger 64-bit systems
// but if we were on a smaller 32-bit system with 2 cores it's kind of silly to
// be doing expensive 64-bit atomics on a 32-bit bus all for just 2 bits of
// data :)

//===----------------------------------------------------------------------===//
// iree_task_affinity_set_t
//===----------------------------------------------------------------------===//

#if IREE_TASK_EXECUTOR_MAX_WORKER_COUNT > 64

typedef unsigned __int128 iree_task_affinity_set_t;

static_assert(sizeof(iree_task_affinity_set_t) == 16,
              "task affinity masks must be 128-bit when MAX_WORKER_COUNT > 64");
static_assert(IREE_TASK_EXECUTOR_MAX_WORKER_COUNT <=
                  sizeof(iree_task_affinity_set_t) * CHAR_BIT,
              "worker count cannot exceed affinity mask width");

// Allows for only a specific worker to be selected.
static inline iree_task_affinity_set_t iree_task_affinity_for_worker(
    uint8_t worker_index) {
  return (iree_task_affinity_set_t)1 << worker_index;
}

// Allows for a range of workers to be selected.
static inline iree_task_affinity_set_t iree_task_affinity_for_worker_range(
    uint8_t worker_start, uint8_t worker_end) {
  return (((iree_task_affinity_set_t)1 << (worker_start - 1)) - 1) ^
         (((iree_task_affinity_set_t)1 << worker_end) - 1);
}

static inline iree_task_affinity_set_t iree_task_private_affinity_set_ones(
    uint32_t count) {
  if (count == 0) return 0;
  if (count >= sizeof(iree_task_affinity_set_t) * CHAR_BIT) {
    return ~(iree_task_affinity_set_t)0;
  }
  return ((iree_task_affinity_set_t)1 << count) - 1;
}

// Allows for any worker to be selected (all bits up to MAX_WORKER_COUNT).
static inline iree_task_affinity_set_t iree_task_affinity_for_any_worker(void) {
  return iree_task_private_affinity_set_ones(
      IREE_TASK_EXECUTOR_MAX_WORKER_COUNT);
}

#define iree_task_affinity_set_ones(count) \
  iree_task_private_affinity_set_ones(count)

static inline int iree_task_private_affinity_set_count_leading_zeros_u128(
    iree_task_affinity_set_t set) {
  if (set == 0) return 128;
  uint64_t lo = (uint64_t)set;
  uint64_t hi = (uint64_t)(set >> 64);
  if (hi) {
    return iree_math_count_leading_zeros_u64(hi);
  }
  return 64 + iree_math_count_leading_zeros_u64(lo);
}

static inline int iree_task_private_affinity_set_count_trailing_zeros_u128(
    iree_task_affinity_set_t set) {
  if (set == 0) return 128;
  uint64_t lo = (uint64_t)set;
  if (lo) {
    return iree_math_count_trailing_zeros_u64(lo);
  }
  return 64 + iree_math_count_trailing_zeros_u64((uint64_t)(set >> 64));
}

static inline int iree_task_private_affinity_set_count_ones_u128(
    iree_task_affinity_set_t set) {
  uint64_t lo = (uint64_t)set;
  uint64_t hi = (uint64_t)(set >> 64);
  return iree_math_count_ones_u64(lo) + iree_math_count_ones_u64(hi);
}

static inline iree_task_affinity_set_t iree_task_private_affinity_set_rotr_u128(
    iree_task_affinity_set_t n, uint32_t c) {
  const uint32_t mask = (uint32_t)(sizeof(n) * CHAR_BIT - 1);
  c &= mask;
  if (!c) return n;
  return (n >> c) | (n << ((-c) & mask));
}

#define iree_task_affinity_set_count_leading_zeros(set) \
  iree_task_private_affinity_set_count_leading_zeros_u128(set)
#define iree_task_affinity_set_count_trailing_zeros(set) \
  iree_task_private_affinity_set_count_trailing_zeros_u128(set)
#define iree_task_affinity_set_count_ones(set) \
  iree_task_private_affinity_set_count_ones_u128(set)
#define iree_task_affinity_set_rotr(set, count) \
  iree_task_private_affinity_set_rotr_u128((set), (count))

//===----------------------------------------------------------------------===//
// iree_atomic_task_affinity_set_t
//===----------------------------------------------------------------------===//

typedef _Atomic unsigned __int128 iree_atomic_task_affinity_set_t;

// Clang warns about 16-byte atomics when __atomic_always_lock_free is false for
// the translation unit's assumptions, but on x86-64 these lower to lock-free
// instructions when the object is 16-byte aligned (which the type guarantees).
#if defined(IREE_COMPILER_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Watomic-alignment"
#endif  // IREE_COMPILER_CLANG

static inline iree_task_affinity_set_t iree_atomic_task_affinity_set_load(
    iree_atomic_task_affinity_set_t* set, iree_memory_order_t order) {
  return iree_atomic_load(set, order);
}

static inline void iree_atomic_task_affinity_set_store(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_set_t value,
    iree_memory_order_t order) {
  iree_atomic_store(set, value, order);
}

static inline iree_task_affinity_set_t iree_atomic_task_affinity_set_fetch_and(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_set_t value,
    iree_memory_order_t order) {
  return iree_atomic_fetch_and(set, value, order);
}

static inline iree_task_affinity_set_t iree_atomic_task_affinity_set_fetch_or(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_set_t value,
    iree_memory_order_t order) {
  return iree_atomic_fetch_or(set, value, order);
}

#if defined(IREE_COMPILER_CLANG)
#pragma clang diagnostic pop
#endif  // IREE_COMPILER_CLANG

#else  // IREE_TASK_EXECUTOR_MAX_WORKER_COUNT <= 64

typedef uint64_t iree_task_affinity_set_t;

static_assert(IREE_TASK_EXECUTOR_MAX_WORKER_COUNT <=
                  sizeof(iree_task_affinity_set_t) * CHAR_BIT,
              "worker count cannot exceed affinity mask width");

// Allows for only a specific worker to be selected.
static inline iree_task_affinity_set_t iree_task_affinity_for_worker(
    uint8_t worker_index) {
  return 1ull << worker_index;
}

// Allows for a range of workers to be selected.
static inline iree_task_affinity_set_t iree_task_affinity_for_worker_range(
    uint8_t worker_start, uint8_t worker_end) {
  return ((1ull << (worker_start - 1)) - 1) ^ ((1ull << worker_end) - 1);
}

// Allows for any worker to be selected.
static inline iree_task_affinity_set_t iree_task_affinity_for_any_worker(void) {
  return UINT64_MAX;
}

#define iree_task_affinity_set_ones(count) \
  (0xFFFFFFFFFFFFFFFFull >> (64 - (count)))
#define iree_task_affinity_set_count_leading_zeros(set) \
  iree_math_count_leading_zeros_u64(set)
#define iree_task_affinity_set_count_trailing_zeros(set) \
  iree_math_count_trailing_zeros_u64(set)
#define iree_task_affinity_set_count_ones(set) iree_math_count_ones_u64(set)
#define iree_task_affinity_set_rotr(set, count) iree_math_rotr_u64(set, count)

//===----------------------------------------------------------------------===//
// iree_atomic_task_affinity_set_t
//===----------------------------------------------------------------------===//

typedef iree_atomic_uint64_t iree_atomic_task_affinity_set_t;

static inline iree_task_affinity_set_t iree_atomic_task_affinity_set_load(
    iree_atomic_task_affinity_set_t* set, iree_memory_order_t order) {
  return iree_atomic_load(set, order);
}

static inline void iree_atomic_task_affinity_set_store(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_set_t value,
    iree_memory_order_t order) {
  iree_atomic_store(set, value, order);
}

static inline iree_task_affinity_set_t iree_atomic_task_affinity_set_fetch_and(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_set_t value,
    iree_memory_order_t order) {
  return iree_atomic_fetch_and(set, value, order);
}

static inline iree_task_affinity_set_t iree_atomic_task_affinity_set_fetch_or(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_set_t value,
    iree_memory_order_t order) {
  return iree_atomic_fetch_or(set, value, order);
}

#endif  // IREE_TASK_EXECUTOR_MAX_WORKER_COUNT > 64

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_AFFINITY_SET_H_
