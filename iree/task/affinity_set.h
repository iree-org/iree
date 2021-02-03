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

#ifndef IREE_TASK_AFFINITY_SET_H_
#define IREE_TASK_AFFINITY_SET_H_

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/math.h"
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

typedef uint64_t iree_task_affinity_set_t;

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
static inline iree_task_affinity_set_t iree_task_affinity_for_any_worker() {
  return UINT64_MAX;
}

#define iree_task_affinity_set_count_trailing_zeros \
  iree_math_count_trailing_zeros_u64
#define iree_task_affinity_set_count_ones iree_math_count_ones_u64
#define iree_task_affinity_set_rotr iree_math_rotr_u64

//===----------------------------------------------------------------------===//
// iree_atomic_task_affinity_set_t
//===----------------------------------------------------------------------===//

typedef iree_atomic_int64_t iree_atomic_task_affinity_set_t;

static inline iree_task_affinity_set_t iree_atomic_task_affinity_set_load(
    iree_atomic_task_affinity_set_t* set, iree_memory_order_t order) {
  return iree_atomic_load_int64(set, order);
}

static inline void iree_atomic_task_affinity_set_store(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_set_t value,
    iree_memory_order_t order) {
  iree_atomic_store_int64(set, value, order);
}

static inline iree_task_affinity_set_t iree_atomic_task_affinity_set_fetch_and(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_set_t value,
    iree_memory_order_t order) {
  return iree_atomic_fetch_and_int64(set, value, order);
}

static inline iree_task_affinity_set_t iree_atomic_task_affinity_set_fetch_or(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_set_t value,
    iree_memory_order_t order) {
  return iree_atomic_fetch_or_int64(set, value, order);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_AFFINITY_SET_H_
