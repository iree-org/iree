// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_AFFINITY_SET_H_
#define IREE_TASK_AFFINITY_SET_H_

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/math.h"
#include "iree/task/tuning.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_task_affinity_bit_t
//===----------------------------------------------------------------------===//

// A precomputed single-bit position within an affinity set. Stored per-worker
// so that atomic set/clear operations can address the correct word directly
// without recomputing the decomposition each time.
typedef struct iree_task_affinity_bit_t {
  uint8_t word_index;  // Which 64-bit word in the set (index / 64).
  uint64_t
      bit_mask;  // Single-bit mask within that word (1ull << (index % 64)).
} iree_task_affinity_bit_t;

// Returns a precomputed bit position for the given worker index.
static inline iree_task_affinity_bit_t iree_task_affinity_bit_for_worker(
    iree_host_size_t worker_index) {
  iree_task_affinity_bit_t bit;
  bit.word_index = (uint8_t)(worker_index / 64);
  bit.bit_mask = 1ull << (worker_index % 64);
  return bit;
}

//===----------------------------------------------------------------------===//
// iree_task_affinity_set_t
//===----------------------------------------------------------------------===//

// A multi-word bitmask. Used for both topology group masks (constructive
// sharing) and executor worker masks (idle/live). Each bit corresponds to one
// index (0 to IREE_TASK_AFFINITY_SET_WORD_COUNT*64-1). Words are ordered
// little-endian: word 0 contains bits 0-63, word 1 contains 64-127, etc.
typedef struct iree_task_affinity_set_t {
  uint64_t words[IREE_TASK_AFFINITY_SET_WORD_COUNT];
} iree_task_affinity_set_t;

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

// Returns an empty affinity set (no workers selected).
static inline iree_task_affinity_set_t iree_task_affinity_set_empty(void) {
  iree_task_affinity_set_t set;
  memset(&set, 0, sizeof(set));
  return set;
}

// Returns an affinity set with only the given worker selected.
static inline iree_task_affinity_set_t iree_task_affinity_for_worker(
    iree_host_size_t worker_index) {
  iree_task_affinity_set_t set;
  memset(&set, 0, sizeof(set));
  set.words[worker_index / 64] = 1ull << (worker_index % 64);
  return set;
}

// Returns an affinity set with the lowest |count| bits set.
static inline iree_task_affinity_set_t iree_task_affinity_set_ones(
    iree_host_size_t count) {
  iree_task_affinity_set_t set;
  memset(&set, 0, sizeof(set));
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    if (count >= 64) {
      set.words[i] = UINT64_MAX;
      count -= 64;
    } else if (count > 0) {
      set.words[i] = (1ull << count) - 1;
      count = 0;
    }
  }
  return set;
}

// Returns an affinity set with all bits set (any worker may be selected).
static inline iree_task_affinity_set_t iree_task_affinity_for_any_worker(void) {
  return iree_task_affinity_set_ones(IREE_TASK_EXECUTOR_MAX_WORKER_COUNT);
}

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

// Returns true if two sets are identical (all words match).
static inline bool iree_task_affinity_set_equal(iree_task_affinity_set_t a,
                                                iree_task_affinity_set_t b) {
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    if (a.words[i] != b.words[i]) return false;
  }
  return true;
}

// Returns true if the set has no bits set.
static inline bool iree_task_affinity_set_is_empty(
    iree_task_affinity_set_t set) {
  uint64_t combined = 0;
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    combined |= set.words[i];
  }
  return combined == 0;
}

// Returns the total number of bits set (popcount across all words).
static inline iree_host_size_t iree_task_affinity_set_count_ones(
    iree_task_affinity_set_t set) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    count += iree_math_count_ones_u64(set.words[i]);
  }
  return count;
}

// Returns the index of the lowest set bit, or -1 if the set is empty.
static inline int iree_task_affinity_set_find_first(
    iree_task_affinity_set_t set) {
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    if (set.words[i]) {
      return (int)(i * 64 + iree_math_count_trailing_zeros_u64(set.words[i]));
    }
  }
  return -1;
}

// Tests whether a specific bit is set.
static inline bool iree_task_affinity_set_test(iree_task_affinity_set_t set,
                                               iree_task_affinity_bit_t bit) {
  return (set.words[bit.word_index] & bit.bit_mask) != 0;
}

//===----------------------------------------------------------------------===//
// Mutation (for local/snapshot sets during iteration)
//===----------------------------------------------------------------------===//

// Sets a bit identified by an affinity_bit_t.
static inline void iree_task_affinity_set_set(iree_task_affinity_set_t* set,
                                              iree_task_affinity_bit_t bit) {
  set->words[bit.word_index] |= bit.bit_mask;
}

// Clears a bit identified by an affinity_bit_t.
static inline void iree_task_affinity_set_clear(iree_task_affinity_set_t* set,
                                                iree_task_affinity_bit_t bit) {
  set->words[bit.word_index] &= ~bit.bit_mask;
}

// Sets a bit by raw index (convenience — computes word decomposition
// internally).
static inline void iree_task_affinity_set_set_index(
    iree_task_affinity_set_t* set, iree_host_size_t index) {
  set->words[index / 64] |= 1ull << (index % 64);
}

// Clears a bit by raw index.
static inline void iree_task_affinity_set_clear_index(
    iree_task_affinity_set_t* set, iree_host_size_t index) {
  set->words[index / 64] &= ~(1ull << (index % 64));
}

//===----------------------------------------------------------------------===//
// Bulk operations
//===----------------------------------------------------------------------===//

// Returns the bitwise AND of two sets.
static inline iree_task_affinity_set_t iree_task_affinity_set_and(
    iree_task_affinity_set_t a, iree_task_affinity_set_t b) {
  iree_task_affinity_set_t result;
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    result.words[i] = a.words[i] & b.words[i];
  }
  return result;
}

// Returns the bitwise OR of two sets.
static inline iree_task_affinity_set_t iree_task_affinity_set_or(
    iree_task_affinity_set_t a, iree_task_affinity_set_t b) {
  iree_task_affinity_set_t result;
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    result.words[i] = a.words[i] | b.words[i];
  }
  return result;
}

// Returns the bitwise complement of a set.
static inline iree_task_affinity_set_t iree_task_affinity_set_not(
    iree_task_affinity_set_t set) {
  iree_task_affinity_set_t result;
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    result.words[i] = ~set.words[i];
  }
  return result;
}

//===----------------------------------------------------------------------===//
// iree_atomic_task_affinity_set_t
//===----------------------------------------------------------------------===//
//
// An array of atomic 64-bit words matching the layout of
// iree_task_affinity_set_t. Whole-set loads and stores are NOT atomically
// consistent across words — this is intentional. The idle and live masks are
// hints (explicitly documented with relaxed ordering), and no protocol depends
// on cross-word consistency.

typedef struct iree_atomic_task_affinity_set_t {
  iree_atomic_int64_t words[IREE_TASK_AFFINITY_SET_WORD_COUNT];
} iree_atomic_task_affinity_set_t;

//===----------------------------------------------------------------------===//
// Atomic per-bit operations (hot path: one word, lock-free)
//===----------------------------------------------------------------------===//

// Atomically sets a single bit. Each worker only ever sets its own bit, so
// this touches exactly one word regardless of set width.
static inline void iree_atomic_task_affinity_set_set(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_bit_t bit,
    iree_memory_order_t order) {
  iree_atomic_fetch_or(&set->words[bit.word_index], (int64_t)bit.bit_mask,
                       order);
}

// Atomically clears a single bit.
static inline void iree_atomic_task_affinity_set_clear(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_bit_t bit,
    iree_memory_order_t order) {
  iree_atomic_fetch_and(&set->words[bit.word_index], (int64_t)~bit.bit_mask,
                        order);
}

//===----------------------------------------------------------------------===//
// Atomic whole-set operations (snapshot / initialize)
//===----------------------------------------------------------------------===//

// Loads a snapshot of the atomic set into a non-atomic set for local
// iteration. Not atomically consistent across words — see type comment.
static inline iree_task_affinity_set_t iree_atomic_task_affinity_set_load(
    const iree_atomic_task_affinity_set_t* set, iree_memory_order_t order) {
  iree_task_affinity_set_t result;
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    result.words[i] = (uint64_t)iree_atomic_load(
        &((iree_atomic_task_affinity_set_t*)set)->words[i], order);
  }
  return result;
}

// Stores a non-atomic set into an atomic set word-by-word.
// Intended for initialization (single-threaded context).
static inline void iree_atomic_task_affinity_set_store(
    iree_atomic_task_affinity_set_t* set, iree_task_affinity_set_t value,
    iree_memory_order_t order) {
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    iree_atomic_store(&set->words[i], (int64_t)value.words[i], order);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_AFFINITY_SET_H_
