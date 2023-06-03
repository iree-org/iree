// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_RESOURCE_SET_H_
#define IREE_HAL_UTILS_RESOURCE_SET_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Capacity is limited by how many bits we reserve for the count.
#define IREE_HAL_RESOURCE_SET_CHUNK_MAX_CAPACITY 0xFFFFu

// A chunk of resources within a resource set.
// Chunks contain a fixed number of resources based on the block size of the
// pool the set was allocated from.
typedef struct iree_hal_resource_set_chunk_t {
  // Next chunk in the chunk linked list.
  struct iree_hal_resource_set_chunk_t* next_chunk;

  // Retained resources - may be less than the capacity derived from the block
  // pool block size. We keep the counts small here to reduce chunk overhead. We
  // could recompute the capacity each time but at the point that we use even 1
  // byte we've already consumed 4 (or 8) thanks to padding and should make use
  // of the rest.
  uint16_t capacity;
  uint16_t count;
  iree_hal_resource_t* resources[];
} iree_hal_resource_set_chunk_t;

// Returns true if the chunk is stored inline in the parent resource set.
#define iree_hal_resource_set_chunk_is_stored_inline(set, chunk) \
  ((const void*)(chunk) == (const uint8_t*)set + sizeof(*set))

// Number of elements in the most-recently-used resource list of a set.
// The larger the number the greater the chance of having a hit but the more
// expensive every miss will be.
//
// To try to keep the MRU in cache we size this based on how many pointers will
// fit in a single cache line. This also makes it easier to author SIMD lookups
// as we'll (in-theory) be able to load the entries into SIMD registers.
//
// Values for the platforms we specify for:
//   32-bit: 64 / 4 = 16x4b ptrs (4 x uint32x4_t)
//   64-bit: 64 / 8 = 8x8b ptrs (4 x uint64x2_t)
// We could scale this up if we wanted but being able to unroll is nice.
#define IREE_HAL_RESOURCE_SET_MRU_SIZE \
  (iree_hardware_constructive_interference_size / sizeof(uintptr_t))

// "Efficient" append-only set for retaining a set of resources.
// This is a non-deterministic data structure that tries to reduce the amount of
// overhead involved in tracking a reasonably-sized set of resources (~dozens to
// hundreds). Set insertion may have false negatives and retain resources more
// than strictly required by trading off the expense of precisely detecting
// redundant insertions with the expense of an additional atomic operation.
//
// This tries to elide insertions by maintaining a most-recently-used list.
// This optimizes for temporal locality of resources used (the same executables,
// same buffers, etc) and is implemented to have a fixed cost regardless of
// whether the values are found and should hopefully trigger enough to avoid the
// subsequent full insertion that can introduce allocations and ref counting.
// The idea is that if we can keep the MRU in cache and spend a dozen cycles to
// manage it we only need to avoid a single cache miss that would occur doing
// the full insertion. We care here because this is on the critical path of
// command encoding and the parasitic cost of maintaining the set scales with
// the number of commands issued. This never needs to be free, only as fast as
// whatever user code may need to do to maintain proper lifetime - or as small
// in terms of code-size.
//
// **WARNING**: thread-unsafe insertion: it's assumed that sets are constructed
// by a single thread, sealed, and then released at once at a future time point.
// Multiple threads needing to insert into a set should have their own sets and
// then join them afterward.
typedef struct iree_hal_resource_set_t {
  // A small MRUish list of resources for quickly deduplicating insertions.
  // We use this to perform an O(k) comparison traded off with the cost of a
  // miss that results in an atomic inc/dec. We shouldn't make this
  // more expensive than the additional cost of the retain/release.
  //
  // This lives at the head of the struct as it's used in 100% of insertions and
  // if we can get lucky with it staying in cache we reduce a lot of memory
  // traffic. Once we spill the MRU and go to main memory to add the resource
  // we're going to have a cache miss and this way we avoid two (one for the
  // set and one for the chunk).
  //
  // TODO(benvanik): ensure alignment on the set - should be at
  // iree_hardware_constructive_interference_size.
  iree_hal_resource_t* mru[IREE_HAL_RESOURCE_SET_MRU_SIZE];

  // Block pool used for allocating additional set storage slabs.
  iree_arena_block_pool_t* block_pool;

  // Linked list of storage chunks.
  iree_hal_resource_set_chunk_t* chunk_head;
} iree_hal_resource_set_t;

// TODO(benvanik): add an allocation method that allows for placement; in many
// command buffer implementations we could allocate the command buffer from the
// same resource set block to avoid a double block pool allocation.

// Allocates a new resource from the given |block_pool|.
// Resources can be inserted and are retained until the set is freed.
IREE_API_EXPORT iree_status_t iree_hal_resource_set_allocate(
    iree_arena_block_pool_t* block_pool, iree_hal_resource_set_t** out_set);

// Frees a resource set and releases all inserted resources.
// The |set| itself will be returned back to the block pool it was allocated
// from.
IREE_API_EXPORT void iree_hal_resource_set_free(iree_hal_resource_set_t* set);

// Freezes the resource set to indicate that it is not expected to change until
// it is freed. This only impacts debugging/ASAN and doesn't otherwise prevent
// insertion.
IREE_API_EXPORT void iree_hal_resource_set_freeze(iree_hal_resource_set_t* set);

// Inserts zero or more resources into the set.
// Each resource will be retained for at least the lifetime of the set.
IREE_API_EXPORT iree_status_t
iree_hal_resource_set_insert(iree_hal_resource_set_t* set,
                             iree_host_size_t count, const void* resources);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_RESOURCE_SET_H_
