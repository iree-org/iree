// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lock-free index freelist using 64-bit compare-and-swap.
//
// This provides a thread-safe LIFO stack for managing pool indices (up to
// 65534 entries). It uses a generation counter to prevent ABA problems,
// allowing concurrent acquire/release from multiple threads without locks.
//
// Usage pattern:
//   - Create a pool with N entries (max 65534)
//   - Initialize the freelist with all indices available
//   - Threads call try_pop() to acquire an index
//   - Threads call push() to release an index back to the pool
//
// Unlike iree_atomic_slist_t (which uses intrusive linked lists with pointers),
// this freelist stores "next" links in a separate array of indices. This allows
// the entire state to fit in 64 bits, avoiding the need for 128-bit CAS.
//
// Memory ordering: All operations use acquire-release semantics to ensure
// proper synchronization of the data associated with each index.

#ifndef IREE_BASE_INTERNAL_ATOMIC_FREELIST_H_
#define IREE_BASE_INTERNAL_ATOMIC_FREELIST_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/alignment.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel value indicating an empty freelist or end of chain.
#define IREE_ATOMIC_FREELIST_EMPTY UINT16_MAX

// Maximum number of entries supported (UINT16_MAX - 1 to reserve sentinel).
#define IREE_ATOMIC_FREELIST_MAX_COUNT ((iree_host_size_t)65534)

// Slot type for freelist next-pointers.
// Uses 32-bit atomics for portability (RISC-V and other platforms don't
// guarantee 16-bit atomics). The extra 2 bytes per slot is acceptable overhead
// for lock-free correctness.
typedef iree_atomic_uint32_t iree_atomic_freelist_slot_t;

// Packed freelist state: [32-bit generation | 16-bit count | 16-bit head]
typedef struct iree_atomic_freelist_state_t {
  // Incremented on every push/pop to prevent ABA problems. With 32 bits, ~4
  // billion operations before wrap. Even at 1B ops/sec, a thread would need to
  // sleep 4+ seconds between read and CAS to hit a collision, which is
  // acceptable for this use case.
  uint32_t generation;
  // Number of available entries. Allows O(1) availability queries without
  // walking the list.
  uint16_t count;
  // Index of the first available entry, or IREE_ATOMIC_FREELIST_EMPTY if the
  // freelist is exhausted.
  uint16_t head;
} iree_atomic_freelist_state_t;
static_assert(sizeof(iree_atomic_freelist_state_t) == 8,
              "freelist size must be 8 to fit in an atomic uint64_t");

// Thread-safe LIFO freelist for fixed-size index pools.
// Aligned to cache line to avoid false sharing when the freelist is embedded
// in a larger structure that may have fields written by other threads.
typedef struct iree_atomic_freelist_t {
  // Packed state accessed atomically.
  // Access via iree_atomic_freelist_pack/unpack helpers.
  iree_atomic_uint64_t packed_state;
} iree_atomic_freelist_t;

// Packs state components into a 64-bit value.
static inline uint64_t iree_atomic_freelist_pack(
    iree_atomic_freelist_state_t state) {
  return ((uint64_t)state.generation << 32) | ((uint64_t)state.count << 16) |
         (uint64_t)state.head;
}

// Unpacks a 64-bit value into state components.
static inline iree_atomic_freelist_state_t iree_atomic_freelist_unpack(
    uint64_t packed) {
  iree_atomic_freelist_state_t state;
  state.generation = (uint32_t)(packed >> 32);
  state.count = (uint16_t)(packed >> 16);
  state.head = (uint16_t)packed;
  return state;
}

// Initializes a freelist with all indices [0, count) available.
// |slots| must have at least |count| entries and remain valid for the lifetime
// of the freelist. The slots array stores the "next" index for each entry.
//
// After initialization:
//   - slots[0] = 1, slots[1] = 2, ..., slots[count-2] = count-1
//   - slots[count-1] = IREE_ATOMIC_FREELIST_EMPTY
//   - head = 0, available count = count
//
// Returns IREE_STATUS_INVALID_ARGUMENT if count exceeds
// IREE_ATOMIC_FREELIST_MAX_COUNT (65534).
//
// Thread-safety: NOT thread-safe. Must be called before any concurrent access.
static inline iree_status_t iree_atomic_freelist_initialize(
    iree_atomic_freelist_slot_t* slots, iree_host_size_t count,
    iree_atomic_freelist_t* out_freelist) {
  // Empty freelist fast-path.
  if (count == 0) {
    iree_atomic_freelist_state_t state = {0, 0, IREE_ATOMIC_FREELIST_EMPTY};
    iree_atomic_store(&out_freelist->packed_state,
                      iree_atomic_freelist_pack(state),
                      iree_memory_order_relaxed);
    return iree_ok_status();
  }

  // Validate count fits in 16-bit index space. Without this check, counts
  // > 65534 would silently truncate, causing corruption (loop writes past
  // allocated slots, state.count wraps to wrong value).
  if (count > IREE_ATOMIC_FREELIST_MAX_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "freelist count %" PRIhsz " exceeds maximum %" PRIhsz, count,
        (iree_host_size_t)IREE_ATOMIC_FREELIST_MAX_COUNT);
  }

  // Chain all indices together: 0 -> 1 -> 2 -> ... -> (count-1) -> EMPTY.
  // Uses relaxed ordering since this is single-threaded initialization.
  for (iree_host_size_t i = 0; i < count - 1; ++i) {
    iree_atomic_store(&slots[i], (uint32_t)(i + 1), iree_memory_order_relaxed);
  }
  iree_atomic_store(&slots[count - 1], IREE_ATOMIC_FREELIST_EMPTY,
                    iree_memory_order_relaxed);

  // Initialize head to 0 with full count.
  iree_atomic_freelist_state_t state = {
      0,                // generation
      (uint16_t)count,  // count
      0                 // head = first index
  };
  iree_atomic_store(&out_freelist->packed_state,
                    iree_atomic_freelist_pack(state),
                    iree_memory_order_release);
  return iree_ok_status();
}

// Deinitializes a freelist.
// The freelist should ideally be empty (all indices returned), but this is not
// enforced to allow cleanup during error handling.
//
// Thread-safety: NOT thread-safe. Must be called after all concurrent access
// has ceased.
static inline void iree_atomic_freelist_deinitialize(
    iree_atomic_freelist_t* freelist) {
  // Nothing to do - slots array is managed externally.
  (void)freelist;
}

// Attempts to pop an index from the freelist.
// Returns true if an index was acquired, false if the freelist is empty.
// On success, |*out_index| contains the acquired index.
//
// Thread-safety: Safe to call concurrently from multiple threads.
static inline bool iree_atomic_freelist_try_pop(
    iree_atomic_freelist_t* freelist, const iree_atomic_freelist_slot_t* slots,
    uint16_t* out_index) {
  uint64_t current_packed =
      iree_atomic_load(&freelist->packed_state, iree_memory_order_acquire);

  for (;;) {
    iree_atomic_freelist_state_t current =
        iree_atomic_freelist_unpack(current_packed);

    // Check if empty.
    if (current.head == IREE_ATOMIC_FREELIST_EMPTY) {
      return false;
    }

    // Read the next index from slots array.
    // This read is outside the CAS, but the generation counter protects us:
    // if another thread modifies the list between our read and CAS, the
    // generation will have changed and our CAS will fail.
    // Uses relaxed ordering since correctness is ensured by the CAS.
    uint16_t next = (uint16_t)iree_atomic_load(
        (iree_atomic_uint32_t*)&slots[current.head], iree_memory_order_relaxed);

    // Prepare new state: pop the head, decrement count, increment generation.
    iree_atomic_freelist_state_t new_state = {
        current.generation + 1,  // Increment generation (ABA prevention)
        (uint16_t)(current.count - 1), next};

    // Attempt CAS. If it fails, another thread modified the list; retry.
    uint64_t new_packed = iree_atomic_freelist_pack(new_state);
    if (iree_atomic_compare_exchange_weak(
            &freelist->packed_state, &current_packed, new_packed,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      *out_index = current.head;
      return true;
    }
    // CAS failed, current_packed now contains the actual value. Loop and retry.
  }
}

// Pushes an index back to the freelist.
// The index must have been previously acquired via try_pop() and not yet
// returned. Pushing an invalid or already-free index causes undefined behavior.
//
// Thread-safety: Safe to call concurrently from multiple threads.
static inline void iree_atomic_freelist_push(iree_atomic_freelist_t* freelist,
                                             iree_atomic_freelist_slot_t* slots,
                                             uint16_t index) {
  uint64_t current_packed =
      iree_atomic_load(&freelist->packed_state, iree_memory_order_acquire);

  for (;;) {
    iree_atomic_freelist_state_t current =
        iree_atomic_freelist_unpack(current_packed);

    // Set our entry to point to the current head.
    // This write is outside the CAS, but if the CAS fails (because head
    // changed), we'll retry and update slots[index] with the new head.
    // Uses relaxed ordering since correctness is ensured by the CAS.
    iree_atomic_store(&slots[index], current.head, iree_memory_order_relaxed);

    // Prepare new state: push our index as new head, increment count and gen.
    iree_atomic_freelist_state_t new_state = {
        current.generation + 1,  // Increment generation (ABA prevention)
        (uint16_t)(current.count + 1), index};

    // Attempt CAS. If it fails, another thread modified the list; retry.
    uint64_t new_packed = iree_atomic_freelist_pack(new_state);
    if (iree_atomic_compare_exchange_weak(
            &freelist->packed_state, &current_packed, new_packed,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      return;
    }
    // CAS failed, current_packed now contains the actual value. Loop and retry.
  }
}

// Returns the current number of available indices in the freelist.
// This is an instantaneous snapshot; the actual count may change immediately
// after this call returns.
//
// Thread-safety: Safe to call concurrently.
static inline iree_host_size_t iree_atomic_freelist_count(
    const iree_atomic_freelist_t* freelist) {
  uint64_t packed =
      iree_atomic_load((iree_atomic_uint64_t*)&freelist->packed_state,
                       iree_memory_order_acquire);
  iree_atomic_freelist_state_t state = iree_atomic_freelist_unpack(packed);
  return (iree_host_size_t)state.count;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_ATOMIC_FREELIST_H_
