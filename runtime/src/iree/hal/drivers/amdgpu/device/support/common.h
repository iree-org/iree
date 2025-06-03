// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// NOTE: builtins are defined in the LLVM AMDGPU device library that is linked
// into the device runtime. We need to redefine them as externs here as they are
// not defined in any accessible headers.
//
// Sources:
// https://github.com/ROCm/rocMLIR/blob/develop/external/llvm-project/amd/device-libs/README.md

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_COMMON_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_COMMON_H_

//===----------------------------------------------------------------------===//
// Compiler Configuration
//===----------------------------------------------------------------------===//

#if defined(__AMDGPU__)
#define IREE_AMDGPU_TARGET_DEVICE 1
#else
#define IREE_AMDGPU_TARGET_HOST 1
#endif  // __AMDGPU__

#if defined(IREE_AMDGPU_TARGET_DEVICE)

typedef char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long int64_t;
typedef unsigned long uint64_t;

typedef int64_t ssize_t;
typedef uint64_t size_t;
typedef int64_t intptr_t;
typedef uint64_t uintptr_t;

#define UINT64_MAX 0xFFFFFFFFFFFFFFFFull

#define NULL ((void*)0)

#else

// NOTE: minimal support for including headers in host code is provided to make
// sharing enums/structures possible; no code is expected to compile.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/threading.h"
#include "third_party/hsa-runtime-headers/include/hsa/hsa.h"  // IWYU pragma: export

#endif  // IREE_AMDGPU_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

#if defined(IREE_AMDGPU_TARGET_DEVICE)

#define IREE_AMDGPU_RESTRICT __restrict__
#define IREE_AMDGPU_ALIGNAS(x) __attribute__((aligned(x)))

#define IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define IREE_AMDGPU_ATTRIBUTE_SINGLE_WORK_ITEM
#define IREE_AMDGPU_ATTRIBUTE_PACKED __attribute__((__packed__))

#define IREE_AMDGPU_ATTRIBUTE_KERNEL \
  [[clang::amdgpu_kernel, gnu::visibility("protected")]]

#define IREE_AMDGPU_LIKELY(x) (__builtin_expect(!!(x), 1))
#define IREE_AMDGPU_UNLIKELY(x) (__builtin_expect(!!(x), 0))

#define IREE_AMDGPU_GUARDED_BY(mutex)

#else

#define IREE_AMDGPU_RESTRICT IREE_RESTRICT
#define IREE_AMDGPU_ALIGNAS(x) iree_alignas(x)

#define IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE IREE_ATTRIBUTE_ALWAYS_INLINE
#define IREE_AMDGPU_ATTRIBUTE_SINGLE_WORK_ITEM
#define IREE_AMDGPU_ATTRIBUTE_PACKED IREE_ATTRIBUTE_PACKED

#define IREE_AMDGPU_LIKELY(x) IREE_LIKELY(x)
#define IREE_AMDGPU_UNLIKELY(x) IREE_UNLIKELY(x)

#define IREE_AMDGPU_GUARDED_BY(mutex)

#endif  // IREE_AMDGPU_TARGET_DEVICE

// Indicates a pointer is on the device. Used as annotations in host code.
#if !defined(IREE_AMDGPU_DEVICE_PTR)
#define IREE_AMDGPU_DEVICE_PTR
#endif  // IREE_AMDGPU_DEVICE_PTR

//===----------------------------------------------------------------------===//
// Alignment / Math
//===----------------------------------------------------------------------===//

#define IREE_AMDGPU_ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define IREE_AMDGPU_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define IREE_AMDGPU_MAX(a, b) (((a) > (b)) ? (a) : (b))

#define IREE_AMDGPU_CEIL_DIV(lhs, rhs) (((lhs) + (rhs) - 1) / (rhs))

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_amdgpu_align(size_t value, size_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

// Returns true if any bit from |rhs| is set in |lhs|.
#define IREE_AMDGPU_ANY_BIT_SET(lhs, rhs) (((lhs) & (rhs)) != 0)
// Returns true iff all bits from |rhs| are set in |lhs|.
#define IREE_AMDGPU_ALL_BITS_SET(lhs, rhs) (((lhs) & (rhs)) == (rhs))

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Returns the number of leading zeros in a 64-bit bitfield.
// Returns -1 if no bits are set.
// Commonly used in HIP as `__lastbit_u32_u64`.
//
// Examples:
//  0x0000000000000000 = -1
//  0x0000000000000001 =  0
//  0x0000000000000010 =  4
//  0xFFFFFFFFFFFFFFFF = -1
#define IREE_AMDGPU_LASTBIT_U64(v) ((v) == 0 ? -1 : __builtin_ctzl(v))

#else

#define IREE_AMDGPU_LASTBIT_U64(v) \
  ((v) == 0 ? -1 : iree_math_count_trailing_zeros_u64(v))

#endif  // IREE_AMDGPU_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// OpenCL-like Scoped Atomics
//===----------------------------------------------------------------------===//

#define iree_amdgpu_destructive_interference_size 64
#define iree_amdgpu_constructive_interference_size 64

#if defined(IREE_AMDGPU_TARGET_DEVICE)

typedef uint32_t iree_amdgpu_memory_order_t;
#define iree_amdgpu_memory_order_relaxed __ATOMIC_RELAXED
#define iree_amdgpu_memory_order_acquire __ATOMIC_ACQUIRE
#define iree_amdgpu_memory_order_release __ATOMIC_RELEASE
#define iree_amdgpu_memory_order_acq_rel __ATOMIC_ACQ_REL
#define iree_amdgpu_memory_order_seq_cst __ATOMIC_SEQ_CST

#define iree_amdgpu_memory_scope_work_item __MEMORY_SCOPE_SINGLE
#define iree_amdgpu_memory_scope_sub_group __MEMORY_SCOPE_WVFRNT
#define iree_amdgpu_memory_scope_work_group __MEMORY_SCOPE_WRKGRP
#if defined(__MEMORY_SCOPE_DEVICE) && defined(__MEMORY_SCOPE_SYSTEM)
#define iree_amdgpu_memory_scope_device __MEMORY_SCOPE_DEVICE
#define iree_amdgpu_memory_scope_system __MEMORY_SCOPE_SYSTEM
#else
#define iree_amdgpu_memory_scope_device 0
#define iree_amdgpu_memory_scope_system 0
#endif  // __MEMORY_SCOPE_DEVICE / __MEMORY_SCOPE_SYSTEM

#define IREE_AMDGPU_SCOPED_ATOMIC_INIT(object, value) *(object) = (value)

typedef /*_Atomic*/ int32_t iree_amdgpu_scoped_atomic_int32_t;
typedef /*_Atomic*/ int64_t iree_amdgpu_scoped_atomic_int64_t;
typedef /*_Atomic*/ uint32_t iree_amdgpu_scoped_atomic_uint32_t;
typedef /*_Atomic*/ uint64_t iree_amdgpu_scoped_atomic_uint64_t;

#define iree_amdgpu_scoped_atomic_load(object, memory_order, memory_scope) \
  __scoped_atomic_load_n((object), (memory_order), (memory_scope))
#define iree_amdgpu_scoped_atomic_store(object, desired, memory_order, \
                                        memory_scope)                  \
  __scoped_atomic_store_n((object), (desired), (memory_order), (memory_scope))

#define iree_amdgpu_scoped_atomic_fetch_add(object, operand, memory_order, \
                                            memory_scope)                  \
  __scoped_atomic_fetch_add((object), (operand), (memory_order), (memory_scope))
#define iree_amdgpu_scoped_atomic_fetch_sub(object, operand, memory_order, \
                                            memory_scope)                  \
  __scoped_atomic_fetch_sub((object), (operand), (memory_order), (memory_scope))
#define iree_amdgpu_scoped_atomic_fetch_and(object, operand, memory_order, \
                                            memory_scope)                  \
  __scoped_atomic_fetch_and((object), (operand), (memory_order), (memory_scope))
#define iree_amdgpu_scoped_atomic_fetch_or(object, operand, memory_order, \
                                           memory_scope)                  \
  __scoped_atomic_fetch_or((object), (operand), (memory_order), (memory_scope))
#define iree_amdgpu_scoped_atomic_fetch_xor(object, operand, memory_order, \
                                            memory_scope)                  \
  __scoped_atomic_fetch_xor((object), (operand), (memory_order), (memory_scope))

#define iree_amdgpu_scoped_atomic_exchange(object, desired, memory_order, \
                                           memory_scope)                  \
  __scoped_atomic_exchange_n((object), (desired), (memory_order),         \
                             (memory_scope))

#define iree_amdgpu_scoped_atomic_compare_exchange_weak(                    \
    object, expected, desired, memory_order_success, memory_order_fail,     \
    memory_scope)                                                           \
  __scoped_atomic_compare_exchange_n((object), (expected), (desired),       \
                                     /*weak=*/true, (memory_order_success), \
                                     (memory_order_fail), (memory_scope))
#define iree_amdgpu_scoped_atomic_compare_exchange_strong(                   \
    object, expected, desired, memory_order_success, memory_order_fail,      \
    memory_scope)                                                            \
  __scoped_atomic_compare_exchange_n((object), (expected), (desired),        \
                                     /*weak=*/false, (memory_order_success), \
                                     (memory_order_fail), (memory_scope))

#else

typedef uint32_t iree_amdgpu_memory_order_t;
#define iree_amdgpu_memory_order_relaxed iree_memory_order_relaxed
#define iree_amdgpu_memory_order_acquire iree_memory_order_acquire
#define iree_amdgpu_memory_order_release iree_memory_order_release
#define iree_amdgpu_memory_order_acq_rel iree_memory_order_acq_rel
#define iree_amdgpu_memory_order_seq_cst iree_memory_order_seq_cst

#define iree_amdgpu_memory_scope_work_item 0
#define iree_amdgpu_memory_scope_sub_group 0
#define iree_amdgpu_memory_scope_work_group 0
#define iree_amdgpu_memory_scope_device 0
#define iree_amdgpu_memory_scope_system 0

#define IREE_AMDGPU_SCOPED_ATOMIC_INIT(object, value) \
  *(object) = IREE_ATOMIC_VAR_INIT(value)

typedef iree_atomic_int32_t iree_amdgpu_scoped_atomic_int32_t;
typedef iree_atomic_int64_t iree_amdgpu_scoped_atomic_int64_t;
typedef iree_atomic_uint32_t iree_amdgpu_scoped_atomic_uint32_t;
typedef iree_atomic_uint64_t iree_amdgpu_scoped_atomic_uint64_t;

#define iree_amdgpu_scoped_atomic_load(object, memory_order, memory_scope) \
  iree_atomic_load((object), (memory_order))
#define iree_amdgpu_scoped_atomic_store(object, desired, memory_order, \
                                        memory_scope)                  \
  iree_atomic_store((object), (desired), (memory_order))

#define iree_amdgpu_scoped_atomic_fetch_add(object, operand, memory_order, \
                                            memory_scope)                  \
  iree_atomic_fetch_add((object), (operand), (memory_order))
#define iree_amdgpu_scoped_atomic_fetch_sub(object, operand, memory_order, \
                                            memory_scope)                  \
  iree_atomic_fetch_sub((object), (operand), (memory_order))
#define iree_amdgpu_scoped_atomic_fetch_and(object, operand, memory_order, \
                                            memory_scope)                  \
  iree_atomic_fetch_and((object), (operand), (memory_order))
#define iree_amdgpu_scoped_atomic_fetch_or(object, operand, memory_order, \
                                           memory_scope)                  \
  iree_atomic_fetch_or((object), (operand), (memory_order))
#define iree_amdgpu_scoped_atomic_fetch_xor(object, operand, memory_order, \
                                            memory_scope)                  \
  iree_atomic_fetch_xor((object), (operand), (memory_order))

#define iree_amdgpu_scoped_atomic_exchange(object, desired, memory_order, \
                                           memory_scope)                  \
  iree_atomic_exchange((object), (desired), (memory_order))

#define iree_amdgpu_scoped_atomic_compare_exchange_weak(                \
    object, expected, desired, memory_order_success, memory_order_fail, \
    memory_scope)                                                       \
  iree_atomic_compare_exchange_weak((object), (expected), (desired),    \
                                    (memory_order_success),             \
                                    (memory_order_fail))
#define iree_amdgpu_scoped_atomic_compare_exchange_strong(              \
    object, expected, desired, memory_order_success, memory_order_fail, \
    memory_scope)                                                       \
  iree_atomic_compare_exchange_strong((object), (expected), (desired),  \
                                      (memory_order_success),           \
                                      (memory_order_fail))

#endif  // IREE_AMDGPU_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// Timing
//===----------------------------------------------------------------------===//

// Tick in the agent domain.
// This can be converted to the system domain for correlation across agents and
// the host with hsa_amd_profiling_convert_tick_to_system_domain.
typedef uint64_t iree_amdgpu_device_tick_t;

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Returns a tick in the agent domain.
// This can be converted to the system domain for correlation across agents and
// the host with hsa_amd_profiling_convert_tick_to_system_domain. The value is
// the same as that placed into signal start_ts/end_ts by the command processor.
#define iree_amdgpu_device_timestamp __builtin_readsteadycounter

// Sleeps the current thread for some "short" amount of time.
// This maps to the S_SLEEP instruction that varies on different architectures
// in how long it can delay execution. The behavior cannot be mapped to wall
// time as it suspends for 64*arg + 1-64 clocks but archs have different limits,
// clock speed can vary over the course of execution, etc. This is mostly only
// useful as a "yield for a few instructions to stop hammering a memory
// location" primitive.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_yield(void) {
  __builtin_amdgcn_s_sleep(1);
}

#else

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_yield(void) {
  iree_thread_yield();
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// Memory Utilities
//===----------------------------------------------------------------------===//

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// TODO(benvanik): use memcpy builtin - these should all be small.
//
// NOTE: doing a memcpy in a single thread is totally not how one should use a
// GPU, but meh. Nearly all tracing usage is with literals we pass as pointers
// and this is really only used by log messages that may be snprintf'ed.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_memcpy(
    void* IREE_AMDGPU_RESTRICT dst, const void* IREE_AMDGPU_RESTRICT src,
    size_t length) {
  for (size_t i = 0; i < length; ++i) {
    ((char*)dst)[i] = ((const char*)src)[i];
  }
}

// TODO(benvanik): use memset builtin - these should all be small.
//
// NOTE: doing a memset in a single thread is totally not how one should use a
// GPU - this should only be used when debugging.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_memset(
    void* IREE_AMDGPU_RESTRICT dst, char value, size_t length) {
  for (size_t i = 0; i < length; ++i) {
    ((char*)dst)[i] = value;
  }
}

#else

#define iree_amdgpu_memcpy memcpy
#define iree_amdgpu_memset memset

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_COMMON_H_
