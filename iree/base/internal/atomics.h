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

#ifndef IREE_BASE_INTERNAL_ATOMICS_H_
#define IREE_BASE_INTERNAL_ATOMICS_H_

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
// C11-compatible atomic operations
//==============================================================================
// We expose support for int32_t, int64_t, and intptr_t (which aliases one of
// int32_t or int64_t). This limits what we need to port and it's really all
// that's needed anyway.

#if defined(IREE_COMPILER_MSVC)

// Atomics using the Win32 Interlocked* APIs.
#include "iree/base/internal/atomics_msvc.h"

#elif defined(IREE_COMPILER_CLANG)

// C11 atomics using Clang builtins.
#include "iree/base/internal/atomics_clang.h"

#elif defined(IREE_COMPILER_GCC)

// Atomics for GCC (compatible with both C and C++).
#include "iree/base/internal/atomics_gcc.h"

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
#define iree_atomic_ref_count_init(count_ptr) \
  iree_atomic_store_int32(count_ptr, 1, iree_memory_order_relaxed)
#define iree_atomic_ref_count_inc(count_ptr) \
  iree_atomic_fetch_add_int32(count_ptr, 1, iree_memory_order_relaxed)
#define iree_atomic_ref_count_dec(count_ptr) \
  iree_atomic_fetch_sub_int32(count_ptr, 1, iree_memory_order_release)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_ATOMICS_H_
