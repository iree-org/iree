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
#include "iree/hal/drivers/amdgpu/util/vmem.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

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

// Per-queue bump allocator for dispatch kernarg memory backed by a vmem magic
// ringbuffer. The physical memory is allocated from the HSA coarse-grain
// kernarg pool (host-writable, device-readable, no coherency traffic) and
// triple-mapped via virtual memory so that multi-block allocations are always
// contiguous in virtual address space regardless of ring wrap position.
//
// Thread safety:
//   allocate() is multi-producer safe (atomic_fetch_add on write_position).
//   reclaim() must be called from a single thread (proactor drain).
//   Multiple threads may call allocate() concurrently.
//
// Backpressure contract:
//   The caller MUST reserve AQL ring slots BEFORE allocating kernarg blocks.
//   The AQL ring spin-wait is the sole backpressure gate. The kernarg ring is
//   sized so that:
//
//     kernarg_capacity >= aql_ring_capacity * max_kernarg_blocks_per_packet
//
//   This guarantees that a successful AQL reservation implies sufficient
//   kernarg space. If AQL slots are reserved first, at most aql_ring_capacity
//   dispatches can be in flight, consuming at most kernarg_capacity blocks
//   total. Violating the AQL-first ordering breaks this invariant: N threads
//   could allocate kernarg without AQL slots, and N is unbounded.
//
// Memory ordering:
//   write_position uses relaxed fetch_add. It only claims space; the actual
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
  // Underlying vmem ringbuffer. Physical memory from the HSA kernarg pool,
  // triple-mapped (prev/base/next) so that contiguous virtual access across
  // the wrap boundary resolves to the correct physical memory. This eliminates
  // all wrap-handling logic from the allocation path.
  iree_hal_amdgpu_vmem_ringbuffer_t ringbuffer;

  // Base pointer into the central virtual mapping, cast to block type for
  // natural indexing (base[i] gives block i without byte arithmetic).
  // Points to ringbuffer.ring_base_ptr.
  iree_hal_amdgpu_kernarg_block_t* base;

  // Power-of-two capacity in blocks.
  uint32_t capacity;
  // capacity - 1, for masking logical positions to physical ring indices.
  uint32_t mask;

  // Monotonically increasing write position in blocks. Each allocate()
  // atomically advances this via fetch_add. The position is a logical
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

// Initializes the kernarg ring by allocating a vmem ringbuffer of at least
// |min_capacity_in_blocks| blocks from |memory_pool|.
//
// |min_capacity_in_blocks| must be a power of two. The actual capacity may
// be larger if the HSA allocation granule requires rounding (the result is
// always a power of two in blocks, since the granule is a power of two and
// blocks are 64 bytes).
//
// |memory_pool| should be a coarse-grain kernarg-capable pool on the CPU
// agent. Coarse-grain means no coherency traffic: CPU writes go directly to
// DRAM, GPU reads via PCIe. This matches HIP/CLR's default kernarg path.
//
// |topology| is used to configure vmem access permissions (all agents get
// read/write access to support CPU writes and GPU reads).
iree_status_t iree_hal_amdgpu_kernarg_ring_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t local_agent,
    hsa_amd_memory_pool_t memory_pool, uint32_t min_capacity_in_blocks,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_kernarg_ring_t* out_ring);

// Deinitializes the kernarg ring and frees all physical and virtual memory.
// All in-flight work must have completed and been reclaimed before calling.
void iree_hal_amdgpu_kernarg_ring_deinitialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_kernarg_ring_t* ring);

// Allocates |block_count| contiguous blocks from the ring.
//
// Returns a pointer to the first block, 64-byte aligned and suitable for use
// as a dispatch packet's kernarg_address. Multi-block allocations that span
// the physical ring boundary are contiguous in virtual address space (the vmem
// triple-mapping handles the wrap transparently).
//
// |out_end_position| receives the logical write position after this allocation
// (first_block + block_count). The caller records this value for epoch-driven
// reclamation: when the GPU completes the associated submission, the drain
// calls reclaim() with this position.
//
// REQUIRES: The caller must have already reserved AQL ring slots for this
// submission. The AQL reservation is the backpressure gate; once it succeeds,
// the sizing invariant guarantees kernarg space is available.
//
// Returns NULL if block_count is 0 or exceeds ring capacity (checked before
// touching any atomics), or if the ring is full after advancing
// write_position. The latter indicates a violated sizing invariant (a
// configuration error, not a recoverable runtime condition); the fetch_add
// has already advanced write_position, which is acceptable because the
// system is already in a bad state.
static inline iree_hal_amdgpu_kernarg_block_t*
iree_hal_amdgpu_kernarg_ring_allocate(iree_hal_amdgpu_kernarg_ring_t* ring,
                                      uint32_t block_count,
                                      uint64_t* out_end_position) {
  IREE_ASSERT_ARGUMENT(out_end_position);
  // Over-capacity would extend past the vmem triple-mapping into unmapped VA
  // (hard page fault). This check must be a runtime guard, not just a debug
  // assertion, because the consequence is memory access outside our mappings.
  if (IREE_UNLIKELY(!block_count || block_count > ring->capacity)) {
    *out_end_position = 0;
    return NULL;
  }

  // Claim space with a single atomic fetch-add. Each concurrent thread gets
  // a unique, non-overlapping range of blocks. Relaxed ordering is sufficient:
  // we are only claiming indices. The subsequent kernarg data writes are
  // ordered by the AQL header commit's release store, which the GPU's command
  // processor acquires when it observes a valid packet header.
  const uint64_t first_block = (uint64_t)iree_atomic_fetch_add(
      &ring->write_position, (int64_t)block_count, iree_memory_order_relaxed);

  // Record the end position for the caller's reclamation tracking.
  *out_end_position = first_block + block_count;

  // Defensive fullness check. Under the AQL-first backpressure invariant this
  // should never fire: the AQL ring bounds in-flight packets, and the kernarg
  // ring is sized to hold the worst-case kernarg consumption for the full AQL
  // ring. Acquire pairs with the drain's release store on read_position.
  const uint64_t read = (uint64_t)iree_atomic_load(&ring->read_position,
                                                   iree_memory_order_acquire);
  if (IREE_UNLIKELY(*out_end_position - read > ring->capacity)) {
    return NULL;
  }

  // The vmem triple-mapping guarantees that base[physical] through
  // base[physical + block_count - 1] are contiguous virtual addresses even
  // when the allocation wraps past the physical ring end. The "next" virtual
  // mapping aliases back to the start of the physical buffer.
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
