// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Low-level interface to the hardware AQL ring buffer. Handles packet slot
// reservation, header commit, and doorbell signaling.
//
// The ring caches hot pointers from iree_amd_queue_t at initialization:
// ring base, mask, doorbell MMIO pointer, and the atomic write/read dispatch
// IDs. All hot-path operations are inline with zero libhsa indirection.
//
// Thread safety:
//   reserve() is multi-producer safe (atomic_fetch_add on write_dispatch_id).
//   commit() is per-slot (no cross-slot contention).
//   doorbell() is idempotent (multiple concurrent writes are harmless).
//
// Memory ordering:
//   The CP starts processing a packet when it observes a valid header. The
//   header commit (atomic store-release) is therefore the publication barrier:
//   all prior writes to the packet's fields and to kernarg memory are ordered
//   before the CP can read them. The doorbell is a wakeup hint with release
//   semantics to ensure header writes are visible before the CP re-scans.

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_AQL_RING_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_AQL_RING_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/processor.h"
#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/abi/signal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_aql_packet_t
//===----------------------------------------------------------------------===//

// A single AQL packet slot. 64 bytes, cache-line aligned. Using a union gives
// natural ring[i] indexing and allows type-punning to the specific packet type
// without casts. The alignment attribute ensures the compiler knows every
// packet is cache-line aligned at every access point.
typedef union iree_alignas(64) iree_hal_amdgpu_aql_packet_t {
  iree_hsa_kernel_dispatch_packet_t dispatch;
  iree_hsa_barrier_and_packet_t barrier_and;
  iree_hsa_barrier_or_packet_t barrier_or;
  iree_hsa_amd_aql_pm4_ib_packet_t pm4_ib;
  iree_hsa_amd_barrier_value_packet_t barrier_value;
  uint8_t raw[64];
} iree_hal_amdgpu_aql_packet_t;
static_assert(sizeof(iree_hal_amdgpu_aql_packet_t) == 64,
              "AQL packet must be exactly one cache line");

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_aql_ring_t
//===----------------------------------------------------------------------===//

// Cached hardware AQL ring buffer state. Initialized once from iree_amd_queue_t
// and used for all subsequent packet operations. The cached pointers avoid
// repeated indirection through the queue descriptor on the hot path.
typedef struct iree_hal_amdgpu_aql_ring_t {
  // Packet ring buffer base (from hsa_queue.base_address), cast for natural
  // indexing: ring.base[id & ring.mask] gives the packet slot.
  iree_hal_amdgpu_aql_packet_t* base;

  // Power-of-two ring mask (hsa_queue.size - 1). Slot = packet_id & mask.
  uint32_t mask;

  // Cached hardware doorbell MMIO pointer. Resolved at init from the doorbell
  // signal's iree_amd_signal_t.hardware_doorbell_ptr. Writing a packet ID here
  // wakes the CP to process new packets. Inlined: no libhsa function pointer
  // indirection, just an atomic store to MMIO.
  volatile int64_t* doorbell;

  // Atomic write dispatch ID. Points into the hardware queue descriptor
  // (iree_amd_queue_t.write_dispatch_id). Multi-producer safe: each thread
  // atomically increments to reserve a unique range of packet IDs.
  iree_atomic_int64_t* write_dispatch_id;

  // Read dispatch ID. Points into the hardware queue descriptor
  // (iree_amd_queue_t.read_dispatch_id). Advanced by the packet processor as
  // queue slots become reusable. This is a packet-slot lifetime signal, not a
  // proof that unrelated sidecar storage associated with a dispatch has been
  // consumed by later queue work.
  const volatile int64_t* read_dispatch_id;
} iree_hal_amdgpu_aql_ring_t;

// Initializes the AQL ring from a hardware queue descriptor.
// Resolves the doorbell pointer from the signal's iree_amd_signal_t and
// caches all hot pointers for zero-indirection access.
static inline void iree_hal_amdgpu_aql_ring_initialize(
    iree_amd_queue_t* hardware_queue, iree_hal_amdgpu_aql_ring_t* out_ring) {
  out_ring->base =
      (iree_hal_amdgpu_aql_packet_t*)hardware_queue->hsa_queue.base_address;
  out_ring->mask = hardware_queue->hsa_queue.size - 1;

  // Resolve the doorbell MMIO pointer from the signal handle. The signal is
  // DOORBELL kind: its hardware_doorbell_ptr points to the memory-mapped
  // doorbell register. Writing a packet ID there wakes the CP.
  iree_amd_signal_t* doorbell_signal =
      (iree_amd_signal_t*)hardware_queue->hsa_queue.doorbell_signal.handle;
  out_ring->doorbell =
      (volatile int64_t*)doorbell_signal->hardware_doorbell_ptr;

  out_ring->write_dispatch_id =
      (iree_atomic_int64_t*)&hardware_queue->write_dispatch_id;
  out_ring->read_dispatch_id =
      (const volatile int64_t*)&hardware_queue->read_dispatch_id;
}

// Reserves |count| contiguous packet slots. Returns the first packet ID.
// Multi-producer safe (atomic_fetch_add on write_dispatch_id). Spins if the
// ring is full (GPU hasn't consumed enough packets).
//
// The caller populates reserved slots via iree_hal_amdgpu_aql_ring_packet()
// and commits each via iree_hal_amdgpu_aql_ring_commit(). The doorbell
// should be rung once after all packets in a batch are committed.
//
// IMPORTANT: The returned packet slots have INVALID headers from the CP's
// perspective. The CP will not process them until a valid header is published.
// Normal host submissions commit every reserved slot before ringing the
// doorbell; device-side patching may intentionally leave a later slot invalid
// only when an earlier packet is guaranteed to publish it before the CP reaches
// that slot.
static inline uint64_t iree_hal_amdgpu_aql_ring_reserve(
    iree_hal_amdgpu_aql_ring_t* ring, uint32_t count) {
  // Atomically claim |count| slots. Each concurrent thread gets a unique,
  // non-overlapping range. Relaxed ordering: we're only claiming indices,
  // not publishing data. Data visibility comes from the header commit.
  const uint64_t first_id = (uint64_t)iree_atomic_fetch_add(
      ring->write_dispatch_id, (int64_t)count, iree_memory_order_relaxed);

  // Backpressure: spin until the ring has space for all reserved slots.
  // The ring can hold (mask + 1) packets. The CP advances read_dispatch_id
  // as packet slots become reusable.
  const uint64_t ring_capacity = (uint64_t)(ring->mask + 1);
  while (IREE_UNLIKELY(first_id + count -
                           (uint64_t)iree_atomic_load(
                               (iree_atomic_int64_t*)ring->read_dispatch_id,
                               iree_memory_order_acquire) >
                       ring_capacity)) {
    iree_processor_yield();
  }

  return first_id;
}

// Attempts to reserve |count| contiguous packet slots without waiting.
//
// Returns false if the ring does not currently have room for the entire range.
// Unlike reserve(), this never advances write_dispatch_id unless the range is
// immediately available, so callers can park and retry after completion drain.
static inline bool iree_hal_amdgpu_aql_ring_try_reserve(
    iree_hal_amdgpu_aql_ring_t* ring, uint32_t count, uint64_t* out_first_id) {
  const uint64_t ring_capacity = (uint64_t)(ring->mask + 1);
  int64_t current_write =
      iree_atomic_load(ring->write_dispatch_id, iree_memory_order_relaxed);
  for (;;) {
    const uint64_t first_id = (uint64_t)current_write;
    const uint64_t current_read =
        (uint64_t)iree_atomic_load((iree_atomic_int64_t*)ring->read_dispatch_id,
                                   iree_memory_order_acquire);
    if (first_id + count - current_read > ring_capacity) {
      *out_first_id = 0;
      return false;
    }
    int64_t desired_write = current_write + (int64_t)count;
    if (iree_atomic_compare_exchange_weak(
            ring->write_dispatch_id, &current_write, desired_write,
            iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
      *out_first_id = first_id;
      return true;
    }
  }
}

// Returns a pointer to the packet slot for |packet_id|. The caller
// populates the packet fields (all except the header) and then commits
// the header via iree_hal_amdgpu_aql_ring_commit().
static inline iree_hal_amdgpu_aql_packet_t* iree_hal_amdgpu_aql_ring_packet(
    iree_hal_amdgpu_aql_ring_t* ring, uint64_t packet_id) {
  return &ring->base[packet_id & ring->mask];
}

// Commits a packet by writing its header + setup as a single atomic
// store-release. This makes the packet visible to the CP: the release
// semantics ensure all prior writes to the packet's fields (and to
// kernarg memory) are ordered before the CP can read them.
//
// |header| is the 16-bit AQL packet header (type, barrier, fence scopes).
// |setup| is the upper 16 bits: grid dimensions for dispatch packets, zero for
// standard barriers, or vendor-specific format bits for extension packets.
//
// Use iree_hsa_make_packet_header() from abi/queue.h to construct |header|.
static inline void iree_hal_amdgpu_aql_ring_commit(
    iree_hal_amdgpu_aql_packet_t* packet, uint16_t header, uint16_t setup) {
  const uint32_t header_setup = (uint32_t)header | ((uint32_t)setup << 16);
  // The first 32 bits of every AQL packet are the header (16 bits) +
  // setup/reserved (16 bits). An atomic store-release on this word is the
  // publication barrier for the entire packet.
  iree_atomic_store((iree_atomic_int32_t*)packet, (int32_t)header_setup,
                    iree_memory_order_release);
}

// Rings the hardware doorbell to wake the CP. The |packet_id| should be the
// highest committed packet ID (the CP processes from its current position up
// to the doorbell value). Call once after committing all packets in a batch
// to amortize the PCIe BAR write cost (~100-300ns).
//
// The doorbell is purely a wakeup hint — the CP will process any packet
// whose header is valid, regardless of whether the doorbell has been rung.
// Multiple concurrent doorbell writes are harmless (the CP re-scans).
//
// Release semantics ensure all committed packet headers are visible before
// the CP wakes and starts scanning.
static inline void iree_hal_amdgpu_aql_ring_doorbell(
    iree_hal_amdgpu_aql_ring_t* ring, uint64_t packet_id) {
  iree_atomic_store((iree_atomic_int64_t*)ring->doorbell, (int64_t)packet_id,
                    iree_memory_order_release);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_AQL_RING_H_
