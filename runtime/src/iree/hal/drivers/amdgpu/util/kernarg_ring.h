// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_KERNARG_RING_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_KERNARG_RING_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_kernarg_block_t
//===----------------------------------------------------------------------===//

// 64-byte aligned kernarg slot. Each dispatch's kernarg region starts on a
// cache-line boundary, preventing false sharing between the GPU command
// processor reading one dispatch's args and the CPU writing the next. CLR uses
// the same alignment (L3 cache line, 64 bytes on CDNA/RDNA).
//
// The ring is indexed in units of this block type: sizes and offsets are in
// block counts, not bytes. A dispatch needing 128 bytes of kernarg space
// consumes 2 blocks.
typedef struct iree_alignas(64) iree_hal_amdgpu_kernarg_block_t {
  uint8_t data[64];
} iree_hal_amdgpu_kernarg_block_t;
static_assert(sizeof(iree_hal_amdgpu_kernarg_block_t) == 64,
              "kernarg block must be exactly one cache line");

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_kernarg_ring_t
//===----------------------------------------------------------------------===//

// Per-queue bump allocator for dispatch kernarg memory backed by a shared host
// memory pool with HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT. The physical
// memory is allocated from a CPU pool and then
// explicitly made visible to the queue's GPU agent with
// hsa_amd_agents_allow_access.
//
// Thread safety:
//   allocate() is multi-producer safe (CAS on write_position).
//   reclaim() must be called from a single thread (proactor drain).
//   Multiple threads may call allocate() concurrently.
//
// Backpressure contract:
//   The caller must prove capacity for both the AQL ring and kernarg ring
//   before publishing work. These resources are intentionally independent:
//   crossing a 64-byte kernarg block boundary must not require extra AQL
//   packets. Callers use can_allocate() as a non-mutating admission check
//   before reserving AQL packets, then allocate() the same block count before
//   committing packet headers.
//
//   A non-VMEM ring may need to skip one tail fragment at wrap to preserve
//   contiguous multi-block allocations. The skipped fragment is counted as
//   in-flight space until the allocation that caused the wrap retires. Both
//   can_allocate() and allocate() model that skip identically so admission
//   cannot succeed and then fail in normal execution.
//
// Memory ordering:
//   write_position uses relaxed CAS. It only claims space; the actual
//   kernarg data writes are made visible to the GPU by the AQL packet header
//   commit (release store). The GPU cannot read the kernargs until it sees a
//   valid header, and the release/acquire pair on the header orders all prior
//   stores (including the kernarg writes).
//
//   read_position uses release (drain) / acquire (allocators). The drain
//   publishes reclamation; allocators observe it for the defensive fullness
//   check. On x86 (the only host architecture for this driver) release and
//   relaxed stores are identical instructions, but we use release to be
//   correct by construction.
typedef struct iree_hal_amdgpu_kernarg_ring_t {
  // Base pointer to the HSA memory-pool allocation, cast to block type for
  // natural indexing (base[i] gives block i without byte arithmetic).
  iree_hal_amdgpu_kernarg_block_t* base;

  // Power-of-two capacity in blocks.
  uint32_t capacity;
  // capacity - 1, for masking logical positions to physical ring indices.
  uint32_t mask;

  // Monotonically increasing write position in blocks. Each allocate()
  // atomically advances this via a CAS loop. The position is a logical
  // (unwrapped) index; (write_position & mask) gives the physical ring offset.
  //
  // Relaxed ordering: see the memory ordering discussion above.
  iree_atomic_int64_t write_position;

  // Read position in blocks. Advanced by the proactor drain when the GPU
  // completes work that referenced the kernarg space. The number of blocks
  // currently in use is (write_position - read_position).
  //
  // Single writer (proactor drain), multiple readers (allocating threads).
  // Release store from drain, acquire load from allocators.
  iree_atomic_int64_t read_position;
} iree_hal_amdgpu_kernarg_ring_t;

// Initializes the kernarg ring by allocating at least |min_capacity_in_blocks|
// blocks from |memory_pool|.
//
// |min_capacity_in_blocks| must be a power of two. The actual capacity may be
// larger if the HSA allocation granule requires rounding; the ring preserves a
// power-of-two block count so physical indices can be masked.
//
// |memory_pool| must be a CPU pool with
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT and
// HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL=true so |device_agent| can be
// granted direct access. HSA VMEM handles are not supported on at least current
// ROCm stacks for the host pools used here, so this ring uses a plain pool
// allocation and skips a tail fragment when needed to preserve contiguous
// multi-block spans.
iree_status_t iree_hal_amdgpu_kernarg_ring_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    hsa_amd_memory_pool_t memory_pool, uint32_t min_capacity_in_blocks,
    iree_hal_amdgpu_kernarg_ring_t* out_ring);

// Deinitializes the kernarg ring and frees the backing HSA memory-pool
// allocation.
// All in-flight work must have completed and been reclaimed before calling.
void iree_hal_amdgpu_kernarg_ring_deinitialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_kernarg_ring_t* ring);

// Returns true if |block_count| contiguous blocks can currently be allocated
// without exceeding the ring capacity. The check accounts for the same
// tail-fragment skip used by allocate() and does not mutate the ring.
//
// This is a snapshot. It is a reservation proof only when the caller serializes
// this check with a following allocate() call, such as under the host queue
// submission mutex.
static inline bool iree_hal_amdgpu_kernarg_ring_can_allocate(
    iree_hal_amdgpu_kernarg_ring_t* ring, uint32_t block_count) {
  if (IREE_UNLIKELY(block_count == 0 || block_count > ring->capacity)) {
    return false;
  }
  uint64_t first_block = (uint64_t)iree_atomic_load(&ring->write_position,
                                                    iree_memory_order_relaxed);
  const uint64_t tail_block_count =
      (uint64_t)ring->capacity - (first_block & ring->mask);
  if (block_count > tail_block_count) {
    first_block += tail_block_count;
  }
  const uint64_t next_write_position = first_block + block_count;
  const uint64_t read_position = (uint64_t)iree_atomic_load(
      &ring->read_position, iree_memory_order_acquire);
  return next_write_position - read_position <= ring->capacity;
}

// Allocates |block_count| contiguous blocks from the ring.
//
// Returns a pointer to the first block, 64-byte aligned and suitable for use
// as a dispatch packet's kernarg_address. Multi-block allocations never wrap in
// physical memory: if the tail fragment is too small, the allocator skips to
// the first block and leaves the tail fragment to be reclaimed with the
// allocation's end position.
//
// |out_end_position| receives the logical write position after this allocation
// (first_block + block_count). The caller records this value for epoch-driven
// reclamation: when the GPU completes the associated submission, the drain
// calls reclaim() with this position.
//
// REQUIRES: The caller must have already proved capacity with can_allocate().
// Under the host queue submission mutex, no other allocator can consume the
// proved space before this call. A NULL return after a successful admission
// check indicates an internal synchronization or sizing invariant failure.
//
// Returns NULL if block_count is 0 or exceeds ring capacity (checked before
// touching any atomics), or if the ring is full.
static inline iree_hal_amdgpu_kernarg_block_t*
iree_hal_amdgpu_kernarg_ring_allocate(iree_hal_amdgpu_kernarg_ring_t* ring,
                                      uint32_t block_count,
                                      uint64_t* out_end_position) {
  IREE_ASSERT_ARGUMENT(out_end_position);
  if (IREE_UNLIKELY(!block_count || block_count > ring->capacity)) {
    *out_end_position = 0;
    return NULL;
  }

  // Claim one contiguous physical span with a CAS loop. If the current tail
  // fragment is too small, skip it and wrap to block 0. Relaxed ordering is
  // sufficient: we are only claiming indices. The subsequent kernarg writes are
  // ordered by the AQL header commit's release store.
  int64_t observed_write_position =
      iree_atomic_load(&ring->write_position, iree_memory_order_relaxed);
  uint64_t first_block = 0;
  uint64_t next_write_position = 0;
  for (;;) {
    first_block = (uint64_t)observed_write_position;
    const uint64_t tail_block_count =
        (uint64_t)ring->capacity - (first_block & ring->mask);
    if (block_count > tail_block_count) {
      first_block += tail_block_count;
    }
    next_write_position = first_block + block_count;
    const uint64_t read_position = (uint64_t)iree_atomic_load(
        &ring->read_position, iree_memory_order_acquire);
    if (IREE_UNLIKELY(next_write_position - read_position > ring->capacity)) {
      *out_end_position = 0;
      return NULL;
    }
    if (iree_atomic_compare_exchange_weak(
            &ring->write_position, &observed_write_position,
            (int64_t)next_write_position, iree_memory_order_relaxed,
            iree_memory_order_relaxed)) {
      break;
    }
  }

  // Record the end position for the caller's reclamation tracking.
  *out_end_position = next_write_position;
  return &ring->base[first_block & ring->mask];
}

// Reclaims all kernarg blocks up to |new_read_position|. Called by the
// proactor drain after confirming the GPU has completed work that referenced
// the kernarg space.
//
// |new_read_position| is the end_position returned by allocate() at
// submission time. The drain processes completions in epoch order, so
// read_position advances monotonically.
//
// Must only be called from the proactor drain thread (single writer).
static inline void iree_hal_amdgpu_kernarg_ring_reclaim(
    iree_hal_amdgpu_kernarg_ring_t* ring, uint64_t new_read_position) {
  // The drain processes completions in epoch order, so read_position must
  // advance monotonically and never exceed write_position.
  IREE_ASSERT(new_read_position >=
              (uint64_t)iree_atomic_load(&ring->read_position,
                                         iree_memory_order_relaxed));
  IREE_ASSERT(new_read_position <=
              (uint64_t)iree_atomic_load(&ring->write_position,
                                         iree_memory_order_relaxed));
  // Release store publishes the reclamation. Allocating threads loading
  // read_position with acquire will see the updated available space.
  iree_atomic_store(&ring->read_position, (int64_t)new_read_position,
                    iree_memory_order_release);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_KERNARG_RING_H_
