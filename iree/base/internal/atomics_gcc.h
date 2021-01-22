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

#ifndef IREE_BASE_INTERNAL_ATOMICS_GCC_H_
#define IREE_BASE_INTERNAL_ATOMICS_GCC_H_

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/target_platform.h"

#if defined(IREE_COMPILER_GCC)

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

typedef int32_t iree_atomic_int32_t;
typedef int64_t iree_atomic_int64_t;
// typedef __int128 iree_atomic_int128_t;
typedef intptr_t iree_atomic_intptr_t;

#ifdef __cplusplus
// Equiv to C++ auto keyword in C++ mode.
#define __iree_auto_type auto
#else
// Only defined in C mode.
#define __iree_auto_type __auto_type
#endif

#define iree_atomic_load_auto(object, order)                       \
  __extension__({                                                  \
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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_COMPILER_GCC

#endif  // IREE_BASE_INTERNAL_ATOMICS_GCC_H_
