// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An implementation of the C11 stdatomics.h utilities we use (which is limited
// to a subset of types for now). We need this for non-C11-compliant platforms
// (MSVC), but it has the added benefit of not conflicting with <atomic>
// (stdatomic.h and atomic cannot be included in the same compilation unit...
// great design). There shouldn't be any difference between what we do here and
// what any implementation would do with the platform atomic functions so it's
// used everywhere.
//
// https://en.cppreference.com/w/c/atomic

#ifndef IREE_BASE_INTERNAL_ATOMICS_H_
#define IREE_BASE_INTERNAL_ATOMICS_H_

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/assert.h"
#include "iree/base/config.h"
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
// C11-compatible atomic operations
//==============================================================================
// We expose support for int32_t, int64_t, and intptr_t (which aliases one of
// int32_t or int64_t). This limits what we need to port and it's really all
// that's needed anyway.

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE

// Atomics are disabled as we've forced ourselves into a fully thread-hostile
// configuration. Used on bare-metal systems with single cores.
#include "iree/base/internal/atomics_disabled.h"  // IWYU pragma: export

#elif defined(IREE_COMPILER_MSVC)

// Atomics using the Win32 Interlocked* APIs.
#include "iree/base/internal/atomics_msvc.h"  // IWYU pragma: export

#elif defined(IREE_COMPILER_CLANG)

// C11 atomics using Clang builtins.
#include "iree/base/internal/atomics_clang.h"  // IWYU pragma: export

#elif defined(IREE_COMPILER_GCC)

// Atomics for GCC (compatible with both C and C++).
#include "iree/base/internal/atomics_gcc.h"  // IWYU pragma: export

#else

// Unsupported architecture.
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

#define iree_atomic_load_intptr iree_atomic_load_auto
#define iree_atomic_store_intptr iree_atomic_store_auto
#define iree_atomic_fetch_add_intptr iree_atomic_fetch_add_auto
#define iree_atomic_fetch_sub_intptr iree_atomic_fetch_sub_auto
#define iree_atomic_exchange_intptr iree_atomic_exchange_auto
#define iree_atomic_compare_exchange_strong_intptr \
  iree_atomic_compare_exchange_strong_auto
#define iree_atomic_compare_exchange_weak_intptr \
  iree_atomic_compare_exchange_weak_auto

#endif  // iree_atomic_load_auto

//==============================================================================
// Reference count atomics
//==============================================================================
// These are just aliases that allow use to have nicely readable ref counting
// operands without caring about the exact bit sizes at each site.

typedef iree_atomic_int32_t iree_atomic_ref_count_t;

// TODO: This should be a non-atomic operation (a relaxed atomic only serves to
// sweep TSan reports under the rug in case of a race on initialization) and
// should use IREE_ATOMIC_VAR_INIT, but apparently this has to be fixed
// at call sites (where the variables are initialized in the first place).
#define iree_atomic_ref_count_init_value(count_ptr, value) \
  iree_atomic_store_int32(count_ptr, value, iree_memory_order_relaxed)

#define iree_atomic_ref_count_init(count_ptr) \
  iree_atomic_ref_count_init_value(count_ptr, 1)

// Why relaxed order:
// https://www.boost.org/doc/libs/1_57_0/doc/html/atomic/usage_examples.html#boost_atomic.usage_examples.example_reference_counters.discussion
// > New references to an object can only be formed from an existing
// > reference, and passing an existing reference from one thread to another
// > must already provide any required synchronization.
//
// Callers of iree_atomic_ref_count_inc typically don't need it to return a
// value (unlike iree_atomic_ref_count_dec), so we make sure that it does not,
// which allows the implementation to use faster atomic instructions where
// available, e.g. STADD on ARMv8.1-a.
#define iree_atomic_ref_count_inc(count_ptr)                              \
  do {                                                                    \
    iree_atomic_fetch_add_int32(count_ptr, 1, iree_memory_order_relaxed); \
  } while (false)

// For now we stick to acq_rel order. TODO: should we follow Boost's advice?
// https://www.boost.org/doc/libs/1_57_0/doc/html/atomic/usage_examples.html#boost_atomic.usage_examples.example_reference_counters.discussion
// > It would be possible to use memory_order_acq_rel for the fetch_sub
// > operation, but this results in unneeded "acquire" operations when the
// > reference counter does not yet reach zero [...].
// It doesn't mention that replacing an acquire-load by an acquire-thread-fence
// may be a pessimization... I would like to hear a second opinion on this,
// particularly regarding how x86-centric this might be.
#define iree_atomic_ref_count_dec(count_ptr) \
  iree_atomic_fetch_sub_int32(count_ptr, 1, iree_memory_order_acq_rel)

// memory_order_acquire order ensures that this sees decrements from
// iree_atomic_ref_count_dec. On the other hand, there is no ordering with
// iree_atomic_ref_count_inc.
#define iree_atomic_ref_count_load(count_ptr) \
  iree_atomic_load_int32(count_ptr, iree_memory_order_acquire)

// Aborts the program if the given reference count value is not 1.
// This should be avoided in all situations but those where continuing execution
// would be invalid. If a reference object is allocated on the stack and the
// parent function is about to return it *must* have a ref count of 1: anything
// else that may be retaining the object will hold a pointer to (effectively)
// uninitialized stack memory.
#define iree_atomic_ref_count_abort_if_uses(count_ptr)             \
  if (IREE_UNLIKELY(iree_atomic_ref_count_load(count_ptr) != 1)) { \
    iree_abort();                                                  \
  }

// Asserts that the given reference count value is zero.
#define IREE_ASSERT_REF_COUNT_ZERO(count_ptr)              \
  IREE_ASSERT_EQ(iree_atomic_ref_count_load(count_ptr), 0, \
                 "ref counted object still has uses")

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_ATOMICS_H_
