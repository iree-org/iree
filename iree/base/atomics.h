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
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Hardware concurrency information
//==============================================================================

// https://en.cppreference.com/w/cpp/thread/hardware_destructive_interference_size
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0154r1.html
// https://norrischiu.github.io/2018/09/08/Cpp-jargon-1.html

// TODO(benvanik): test 128 on x64 (to thwart hardware prefetcher).

// Minimum offset between two objects to avoid false sharing.
// If two members are aligned to this value they will (likely) not share the
// same L1 cache line.
#define iree_hardware_destructive_interference_size 64

// Maximum size of contiguous memory to promote true sharing.
// If two members are within a span of this value they will (likely) share the
// same L1 cache line.
#define iree_hardware_constructive_interference_size 64

//==============================================================================
// Atomics using the Win32 Interlocked* APIs
//==============================================================================
#if defined(IREE_COMPILER_MSVC)

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
  InterlockedExchange64((volatile LONG64*)object, desired)
#define iree_atomic_fetch_add_int64(object, operand, order) \
  InterlockedExchangeAdd64((volatile LONG64*)object, operand)
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

//==============================================================================
// C11 atomics using Clang builtins
//==============================================================================
#elif defined(IREE_COMPILER_CLANG)

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
// TODO(#3453): check for __int128 support before using
// typedef _Atomic __int128 iree_atomic_int128_t;

#define iree_atomic_load_auto(object, order) \
  __c11_atomic_load((object), (order))
#define iree_atomic_store_auto(object, desired, order) \
  __c11_atomic_store((object), (desired), (order))
#define iree_atomic_fetch_add_auto(object, operand, order) \
  __c11_atomic_fetch_add((object), (operand), (order))
#define iree_atomic_fetch_sub_auto(object, operand, order) \
  __c11_atomic_fetch_sub((object), (operand), (order))
#define iree_atomic_fetch_and_auto(object, operand, order) \
  __c11_atomic_fetch_and((object), (operand), (order))
#define iree_atomic_fetch_or_auto(object, operand, order) \
  __c11_atomic_fetch_or((object), (operand), (order))
#define iree_atomic_fetch_xor_auto(object, operand, order) \
  __c11_atomic_fetch_xor((object), (operand), (order))
#define iree_atomic_exchange_auto(object, operand, order) \
  __c11_atomic_exchange((object), (operand), (order))
#define iree_atomic_compare_exchange_strong_auto(object, expected, desired, \
                                                 order_succ, order_fail)    \
  __c11_atomic_compare_exchange_strong((object), (expected), (desired),     \
                                       (order_succ), (order_fail))
#define iree_atomic_compare_exchange_weak_auto(object, expected, desired, \
                                               order_succ, order_fail)    \
  __c11_atomic_compare_exchange_weak((object), (expected), (desired),     \
                                     (order_succ), (order_fail))

#define iree_atomic_thread_fence(order) __c11_atomic_thread_fence(order)

//==============================================================================
// Atomics for GCC (compatible with both C and C++)
//==============================================================================
#elif defined(IREE_COMPILER_GCC)

typedef enum iree_memory_order_e {
  iree_memory_order_relaxed = __ATOMIC_RELAXED,
  iree_memory_order_consume = __ATOMIC_CONSUME,
  iree_memory_order_acquire = __ATOMIC_ACQUIRE,
  iree_memory_order_release = __ATOMIC_RELEASE,
  iree_memory_order_acq_rel = __ATOMIC_ACQ_REL,
  iree_memory_order_seq_cst = __ATOMIC_SEQ_CST,
} iree_memory_order_t;

#define IREE_ATOMIC_VAR_INIT(value) (value)

typedef int32_t iree_atomic_int32_t;
typedef int64_t iree_atomic_int64_t;
// typedef __int128 iree_atomic_int128_t;

#ifdef __cplusplus
// Equiv to C++ auto keyword in C++ mode.
#define __iree_auto_type auto
#else
// Only defined in C mode.
#define __iree_auto_type __auto_type
#endif

#define iree_atomic_load_auto(object, order)                       \
  __atomic_load_ptr(object, order) __extension__({                 \
    __iree_auto_type __atomic_load_ptr = (object);                 \
    __typeof__(*__atomic_load_ptr) __atomic_load_tmp;              \
    __atomic_load(__atomic_load_ptr, &__atomic_load_tmp, (order)); \
    __atomic_load_tmp;                                             \
  })
#define iree_atomic_store_auto(object, desired, order)                \
  __extension__({                                                     \
    __iree_auto_type __atomic_store_ptr = (object);                   \
    __typeof__(*__atomic_store_ptr) __atomic_store_tmp = (desired);   \
    __atomic_store(__atomic_store_ptr, &__atomic_store_tmp, (order)); \
  })
#define iree_atomic_fetch_add_auto(object, operand, order) \
  __atomic_fetch_add((object), (operand), (order))
#define iree_atomic_fetch_sub_auto(object, operand, order) \
  __atomic_fetch_sub((object), (operand), (order))
#define iree_atomic_fetch_and_auto(object, operand, order) \
  __atomic_fetch_and((object), (operand), (order))
#define iree_atomic_fetch_or_auto(object, operand, order) \
  __atomic_fetch_or((object), (operand), (order))
#define iree_atomic_fetch_xor_auto(object, operand, order) \
  __atomic_fetch_xor((object), (operand), (order))
#define iree_atomic_exchange_auto(object, operand, order) \
  __atomic_exchange_n((object), (operand), (order))
#define iree_atomic_compare_exchange_strong_auto(object, expected, desired, \
                                                 order_succ, order_fail)    \
  __atomic_compare_exchange_n(object, expected, desired, /*weak=*/false,    \
                              (order_succ), (order_fail))
#define iree_atomic_compare_exchange_weak_auto(object, expected, desired, \
                                               order_succ, order_fail)    \
  __atomic_compare_exchange_n(object, expected, desired, /*weak=*/true,   \
                              (order_succ), (order_fail))

#define iree_atomic_thread_fence(order) __atomic_thread_fence(order)

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
#define iree_atomic_fetch_and_int32 iree_atomic_fetch_and_auto
#define iree_atomic_fetch_or_int32 iree_atomic_fetch_or_auto
#define iree_atomic_fetch_xor_int32 iree_atomic_fetch_xor_auto
#define iree_atomic_exchange_int32 iree_atomic_exchange_auto
#define iree_atomic_compare_exchange_strong_int32 \
  iree_atomic_compare_exchange_strong_auto
#define iree_atomic_compare_exchange_weak_int32 \
  iree_atomic_compare_exchange_weak_auto

#define iree_atomic_load_int64 iree_atomic_load_auto
#define iree_atomic_store_int64 iree_atomic_store_auto
#define iree_atomic_fetch_add_int64 iree_atomic_fetch_add_auto
#define iree_atomic_fetch_sub_int64 iree_atomic_fetch_sub_auto
#define iree_atomic_fetch_and_int64 iree_atomic_fetch_and_auto
#define iree_atomic_fetch_or_int64 iree_atomic_fetch_or_auto
#define iree_atomic_fetch_xor_int64 iree_atomic_fetch_xor_auto
#define iree_atomic_exchange_int64 iree_atomic_exchange_auto
#define iree_atomic_compare_exchange_strong_int64 \
  iree_atomic_compare_exchange_strong_auto
#define iree_atomic_compare_exchange_weak_int64 \
  iree_atomic_compare_exchange_weak_auto

#endif  // iree_atomic_load_auto

//==============================================================================
// Pointer-width atomics
//==============================================================================

#if defined(IREE_PTR_SIZE_32)
typedef iree_atomic_int32_t iree_atomic_ptr_t;
#define iree_atomic_load_ptr iree_atomic_load_int32
#define iree_atomic_store_ptr iree_atomic_store_int32
#define iree_atomic_fetch_add_ptr iree_atomic_fetch_add_int32
#define iree_atomic_fetch_sub_ptr iree_atomic_fetch_sub_int32
#define iree_atomic_exchange_ptr iree_atomic_exchange_int32
#define iree_atomic_compare_exchange_strong_ptr \
  iree_atomic_compare_exchange_strong_int32
#define iree_atomic_compare_exchange_weak_ptr \
  iree_atomic_compare_exchange_weak_int32
#else
typedef iree_atomic_int64_t iree_atomic_ptr_t;
#define iree_atomic_load_ptr iree_atomic_load_int64
#define iree_atomic_store_ptr iree_atomic_store_int64
#define iree_atomic_fetch_add_ptr iree_atomic_fetch_add_int64
#define iree_atomic_fetch_sub_ptr iree_atomic_fetch_sub_int64
#define iree_atomic_exchange_ptr iree_atomic_exchange_int64
#define iree_atomic_compare_exchange_strong_ptr \
  iree_atomic_compare_exchange_strong_int64
#define iree_atomic_compare_exchange_weak_ptr \
  iree_atomic_compare_exchange_weak_int64
#endif  // IREE_PTR_SIZE_32

//==============================================================================
// Reference count atomics
//==============================================================================
// These are just aliases that allow use to have nicely readable ref counting
// operands without caring about the exact bit sizes at each site.

typedef iree_atomic_int32_t iree_atomic_ref_count_t;
#define iree_atomic_ref_count_init(count_ptr) \
  iree_atomic_store_int32(count_ptr, 1, iree_memory_order_relaxed)
#define iree_atomic_ref_count_inc(count_ptr) \
  iree_atomic_fetch_add_int32(count_ptr, 1, iree_memory_order_relaxed)
#define iree_atomic_ref_count_dec(count_ptr) \
  iree_atomic_fetch_sub_int32(count_ptr, 1, iree_memory_order_release)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_ATOMICS_H_
