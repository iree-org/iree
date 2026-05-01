// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_QUEUE_UPLOAD_RING_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_QUEUE_UPLOAD_RING_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Memory backing policy for a queue-owned upload ring.
//
// The descriptor is consumed during ring initialization and may reference
// caller-owned stack storage for |access_agents|. Upload rings use the same
// host-write publication policy as queue-owned kernargs because both carry
// host-produced records consumed by device packets after AQL publication.
typedef struct iree_hal_amdgpu_queue_upload_ring_memory_t {
  // HSA memory pool used for the ring allocation.
  hsa_amd_memory_pool_t memory_pool;
  // Agents granted explicit access to the ring allocation.
  const hsa_agent_t* access_agents;
  // Number of entries in |access_agents|.
  iree_host_size_t access_agent_count;
  // Host-write publication mechanism for this memory pool.
  iree_hal_amdgpu_kernarg_ring_publication_t publication;
} iree_hal_amdgpu_queue_upload_ring_memory_t;

// Byte span reserved from a queue upload ring.
typedef struct iree_hal_amdgpu_queue_upload_span_t {
  // Host pointer used by the CPU submission path to populate the record.
  uint8_t* host_ptr;
  // Device-visible pointer value corresponding to |host_ptr|.
  uint64_t device_ptr;
  // Number of bytes reserved in this span.
  iree_host_size_t length;
  // Logical write position after this allocation, used for epoch reclaim.
  uint64_t end_position;
} iree_hal_amdgpu_queue_upload_span_t;

// Per-queue byte-granular upload allocator for small device-visible control
// records such as binding pointer tables and device-side fixup arguments.
//
// Thread safety:
//   allocate() is multi-producer safe (CAS on write_position).
//   reclaim() must be called from a single thread (notification drain).
//   Multiple threads may call allocate() concurrently.
//
// Backpressure contract:
//   Callers use can_allocate() as a non-mutating admission check under the
//   queue submission mutex, then allocate() the same request before publishing
//   packets. Alignment padding and wrap skips are in-flight bytes until the
//   submission that caused them retires.
//
// Memory ordering:
//   write_position only claims space and uses relaxed ordering. Device
//   consumers cannot observe records until the caller publishes host writes and
//   commits the dependent AQL packet headers. read_position uses release
//   (drain) / acquire (allocators) so submitters observe reclaimed capacity.
typedef struct iree_hal_amdgpu_queue_upload_ring_t {
  // Base pointer to the HSA memory-pool allocation.
  uint8_t* base;

  // Device-visible pointer value corresponding to |base|.
  uint64_t device_base;

  // Power-of-two capacity in bytes.
  uint32_t capacity;
  // capacity - 1, for masking logical positions to physical ring offsets.
  uint32_t mask;

  // Host-write publication mechanism for this ring.
  iree_hal_amdgpu_kernarg_ring_publication_t publication;

  // Monotonically increasing write position in bytes.
  iree_atomic_int64_t write_position;

  // Read position in bytes. Advanced by notification drain when the GPU
  // completes work that referenced upload records.
  iree_atomic_int64_t read_position;
} iree_hal_amdgpu_queue_upload_ring_t;

static inline uint64_t iree_hal_amdgpu_queue_upload_ring_align_position(
    uint64_t position, iree_host_size_t alignment) {
  return (position + alignment - 1) & ~((uint64_t)alignment - 1);
}

// Initializes the upload ring by allocating at least |min_capacity| bytes from
// |memory->memory_pool|. |min_capacity| must be a power of two.
iree_status_t iree_hal_amdgpu_queue_upload_ring_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_queue_upload_ring_memory_t* memory,
    uint32_t min_capacity, iree_hal_amdgpu_queue_upload_ring_t* out_ring);

// Deinitializes the upload ring and frees the backing HSA allocation. All
// in-flight work must have completed and been reclaimed before calling.
void iree_hal_amdgpu_queue_upload_ring_deinitialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_queue_upload_ring_t* ring);

// Publishes host writes to upload records before packet headers referencing
// them become visible to the command processor.
static inline void iree_hal_amdgpu_queue_upload_ring_publish_host_writes(
    const iree_hal_amdgpu_queue_upload_ring_t* ring) {
  if (ring->publication.mode ==
      IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_NONE) {
    return;
  }
  IREE_ASSERT(ring->publication.mode ==
              IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_HDP_FLUSH);
  IREE_ASSERT(ring->publication.hdp_mem_flush_control);
  iree_hal_amdgpu_kernarg_ring_host_write_fence();
  *ring->publication.hdp_mem_flush_control = 1u;
  (void)*ring->publication.hdp_mem_flush_control;
}

// Returns true if |length| bytes with |alignment| can currently be allocated.
// The check accounts for the same alignment padding and wrap skip used by
// allocate() and does not mutate the ring.
static inline bool iree_hal_amdgpu_queue_upload_ring_can_allocate(
    iree_hal_amdgpu_queue_upload_ring_t* ring, iree_host_size_t length,
    iree_host_size_t alignment) {
  if (IREE_UNLIKELY(length == 0 || length > ring->capacity || alignment == 0 ||
                    alignment > ring->capacity ||
                    !iree_host_size_is_power_of_two(alignment))) {
    return false;
  }
  uint64_t first_byte = (uint64_t)iree_atomic_load(&ring->write_position,
                                                   iree_memory_order_relaxed);
  first_byte =
      iree_hal_amdgpu_queue_upload_ring_align_position(first_byte, alignment);
  const uint64_t tail_length =
      (uint64_t)ring->capacity - (first_byte & ring->mask);
  if (length > tail_length) {
    first_byte += tail_length;
    first_byte =
        iree_hal_amdgpu_queue_upload_ring_align_position(first_byte, alignment);
  }
  const uint64_t next_write_position = first_byte + length;
  const uint64_t read_position = (uint64_t)iree_atomic_load(
      &ring->read_position, iree_memory_order_acquire);
  return next_write_position - read_position <= ring->capacity;
}

// Allocates |length| contiguous bytes aligned to |alignment| from the ring.
//
// REQUIRES: The caller must have already proved capacity with can_allocate()
// under the same queue submission admission critical section.
static inline iree_hal_amdgpu_queue_upload_span_t
iree_hal_amdgpu_queue_upload_ring_allocate(
    iree_hal_amdgpu_queue_upload_ring_t* ring, iree_host_size_t length,
    iree_host_size_t alignment) {
  iree_hal_amdgpu_queue_upload_span_t span = {0};
  if (IREE_UNLIKELY(length == 0 || length > ring->capacity || alignment == 0 ||
                    alignment > ring->capacity ||
                    !iree_host_size_is_power_of_two(alignment))) {
    return span;
  }

  int64_t observed_write_position =
      iree_atomic_load(&ring->write_position, iree_memory_order_relaxed);
  uint64_t first_byte = 0;
  uint64_t next_write_position = 0;
  for (;;) {
    first_byte = (uint64_t)observed_write_position;
    first_byte =
        iree_hal_amdgpu_queue_upload_ring_align_position(first_byte, alignment);
    const uint64_t tail_length =
        (uint64_t)ring->capacity - (first_byte & ring->mask);
    if (length > tail_length) {
      first_byte += tail_length;
      first_byte = iree_hal_amdgpu_queue_upload_ring_align_position(first_byte,
                                                                    alignment);
    }
    next_write_position = first_byte + length;
    const uint64_t read_position = (uint64_t)iree_atomic_load(
        &ring->read_position, iree_memory_order_acquire);
    if (IREE_UNLIKELY(next_write_position - read_position > ring->capacity)) {
      return span;
    }
    if (iree_atomic_compare_exchange_weak(
            &ring->write_position, &observed_write_position,
            (int64_t)next_write_position, iree_memory_order_relaxed,
            iree_memory_order_relaxed)) {
      break;
    }
  }

  const uint32_t physical_offset = (uint32_t)(first_byte & ring->mask);
  span.host_ptr = ring->base + physical_offset;
  span.device_ptr = ring->device_base + physical_offset;
  span.length = length;
  span.end_position = next_write_position;
  return span;
}

// Reclaims all upload bytes up to |new_read_position|. Called by notification
// drain after confirming the GPU has completed work that referenced the span.
static inline void iree_hal_amdgpu_queue_upload_ring_reclaim(
    iree_hal_amdgpu_queue_upload_ring_t* ring, uint64_t new_read_position) {
  IREE_ASSERT(new_read_position >=
              (uint64_t)iree_atomic_load(&ring->read_position,
                                         iree_memory_order_relaxed));
  IREE_ASSERT(new_read_position <=
              (uint64_t)iree_atomic_load(&ring->write_position,
                                         iree_memory_order_relaxed));
  iree_atomic_store(&ring->read_position, (int64_t)new_read_position,
                    iree_memory_order_release);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_QUEUE_UPLOAD_RING_H_
