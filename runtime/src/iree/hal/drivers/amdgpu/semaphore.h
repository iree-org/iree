// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_

#include <string.h>

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_last_signal_t
//===----------------------------------------------------------------------===//

typedef uint8_t iree_hal_amdgpu_last_signal_flags_t;
enum iree_hal_amdgpu_last_signal_flag_bits_e {
  IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE = 0u,
  // The cache contains a producer axis/epoch/value snapshot from at least one
  // signal submission.
  IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_VALID = 1u << 0,
  // The semaphore's post-publish frontier is exactly the producer queue's
  // frontier at |epoch|. A single barrier on |producer_axis|@|epoch| therefore
  // implies all transitive dependencies carried by this semaphore signal, even
  // when the producer frontier contains multiple peer axes.
  IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT = 1u << 1,
};

// Seqlock-protected cache of the most recent queue signal on a semaphore.
// Written by the submission path when queue_execute signals the semaphore,
// read by the submission path when processing waits (for same-queue FIFO
// elision and direct producer-epoch cross-queue barriers) and by the
// host-wait fast path.
//
// The seqlock ensures torn reads across the payload fields are detected and
// retried. Writers increment the sequence counter to an odd value before the
// update and to an even value after. Readers retry if the sequence is odd
// (write in progress) or changed between the start and end of the read.
typedef struct iree_hal_amdgpu_last_signal_t {
  // Seqlock sequence counter; odd means a writer is updating payload fields.
  iree_atomic_int32_t sequence;
  // Cached signal validity and producer-frontier precision flags.
  iree_hal_amdgpu_last_signal_flags_t flags;
  // Reserved bytes kept zero so the payload stays naturally aligned.
  uint8_t reserved[3];
  // Producer queue axis that submitted the last cached signal.
  iree_async_axis_t producer_axis;
  // Producer queue epoch associated with the last cached signal.
  uint64_t epoch;
  // Semaphore payload value signaled at |producer_axis|/|epoch|.
  uint64_t value;
} iree_hal_amdgpu_last_signal_t;

// Stores a new last-signal snapshot. Thread-safe (seqlock writer).
static inline void iree_hal_amdgpu_last_signal_store(
    iree_hal_amdgpu_last_signal_t* cache,
    iree_hal_amdgpu_last_signal_flags_t flags, iree_async_axis_t producer_axis,
    uint64_t epoch, uint64_t value) {
  // Increment to odd: signals write in progress.
  iree_atomic_fetch_add(&cache->sequence, 1, iree_memory_order_acquire);
  cache->flags = flags;
  memset(cache->reserved, 0, sizeof(cache->reserved));
  cache->producer_axis = producer_axis;
  cache->epoch = epoch;
  cache->value = value;
  // Increment to even: signals write complete.
  iree_atomic_fetch_add(&cache->sequence, 1, iree_memory_order_release);
}

// Loads the last-signal snapshot. Thread-safe (seqlock reader).
// Returns true if the cache has been written at least once and remains valid.
static inline bool iree_hal_amdgpu_last_signal_load(
    const iree_hal_amdgpu_last_signal_t* cache,
    iree_hal_amdgpu_last_signal_flags_t* out_flags,
    iree_async_axis_t* out_producer_axis, uint64_t* out_epoch,
    uint64_t* out_value) {
  int32_t sequence;
  do {
    sequence = iree_atomic_load(&cache->sequence, iree_memory_order_acquire);
    if (IREE_UNLIKELY(sequence & 1)) continue;  // writer in progress
    *out_flags = cache->flags;
    *out_producer_axis = cache->producer_axis;
    *out_epoch = cache->epoch;
    *out_value = cache->value;
  } while (
      IREE_UNLIKELY(iree_atomic_load(&cache->sequence,
                                     iree_memory_order_acquire) != sequence));
  return (*out_flags & IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_VALID) != 0;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_semaphore_t
//===----------------------------------------------------------------------===//

// Creates an AMDGPU HAL semaphore backed by an embedded async semaphore.
//
// Signal, query, and wait all delegate to the async semaphore infrastructure.
// The semaphore embeds iree_async_semaphore_t at offset 0 for toll-free
// bridging between HAL and async layers.
//
// |device| is stored as a back-pointer for type discrimination (checking
// whether a semaphore belongs to a specific logical device). Not retained.
//
// |queue_affinity| hints which queues will signal/wait on the semaphore. If
// IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL is set, the semaphore is only used on
// those queues and the implementation may optimize accordingly.
//
// |flags| controls semaphore behavior:
//   DEVICE_LOCAL: only signaled/waited by queues within this device. Enables
//     epoch-based hardware synchronization (barrier-value packets).
//   HOST_INTERRUPT: host may call iree_hal_semaphore_wait. Enables
//     interrupt-driven host blocking via HSA signal waits.
//   SINGLE_PRODUCER: signals come from one producer timeline, allowing the
//     implementation to treat the latest producer queue epoch as the complete
//     causal frontier for the latest payload value.
iree_status_t iree_hal_amdgpu_semaphore_create(
    iree_hal_amdgpu_logical_device_t* device, iree_async_proactor_t* proactor,
    iree_hal_queue_affinity_t queue_affinity, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is an AMDGPU semaphore.
bool iree_hal_amdgpu_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Returns true if |semaphore| is an AMDGPU semaphore belonging to |device|.
// Used by the submission path to gate the epoch-based synchronization fast
// path: only semaphores local to the submitting device can use barrier-value
// packets on the device's queue epoch signals. Non-local semaphores (from
// other HAL devices, remoting, etc.) always use the software timepoint path.
bool iree_hal_amdgpu_semaphore_is_local(
    iree_hal_semaphore_t* semaphore,
    const iree_hal_amdgpu_logical_device_t* device);

// Returns the AMDGPU semaphore creation flags. Caller must verify
// iree_hal_amdgpu_semaphore_isa() first.
iree_hal_semaphore_flags_t iree_hal_amdgpu_semaphore_flags(
    iree_hal_semaphore_t* semaphore);

// Returns the AMDGPU semaphore creation queue affinity. Caller must verify
// iree_hal_amdgpu_semaphore_isa() first.
iree_hal_queue_affinity_t iree_hal_amdgpu_semaphore_queue_affinity(
    iree_hal_semaphore_t* semaphore);

// Returns true if |semaphore| has the strict private-stream contract used by
// HIP-on-HAL stream timelines:
//   - owned by |device|;
//   - device-local;
//   - single-producer; and
//   - not host-interrupt/export/timepoint-export capable.
//
// Such semaphores are still normal HAL timeline semaphores, but AMDGPU may use
// the single-producer proof to publish only the producer queue epoch on the
// signal hot path. Completion drain still advances the timeline value, but
// does not need to accumulate a multi-producer async frontier for the private
// stream handoff.
bool iree_hal_amdgpu_semaphore_has_private_stream_semantics(
    iree_hal_semaphore_t* semaphore,
    const iree_hal_amdgpu_logical_device_t* device);

// Returns a pointer to the last_signal cache on an AMDGPU semaphore.
// Caller must verify iree_hal_amdgpu_semaphore_isa() first.
iree_hal_amdgpu_last_signal_t* iree_hal_amdgpu_semaphore_last_signal(
    iree_hal_semaphore_t* semaphore);

// Publishes the submission-time frontier and last-signal cache for a signal
// from |producer_axis| at (|producer_epoch|, |producer_value|).
//
// Merges |producer_frontier| into the semaphore's accumulated frontier under
// the semaphore mutex, then updates the last-signal cache while still holding
// that mutex so PRODUCER_FRONTIER_EXACT reflects the post-merge frontier
// precisely. Returns false if the frontier merge overflowed capacity; in that
// case the cache is cleared and callers must fall back to software waits for
// not-yet-complete values.
//
// Caller must verify iree_hal_amdgpu_semaphore_isa() first.
bool iree_hal_amdgpu_semaphore_publish_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t producer_axis,
    const iree_async_frontier_t* producer_frontier, uint64_t producer_epoch,
    uint64_t producer_value);

// Publishes a single-producer private-stream signal without accumulating the
// full semaphore frontier under the async semaphore mutex.
//
// Caller must prove iree_hal_amdgpu_semaphore_has_private_stream_semantics()
// and serialize all signals through |producer_axis|. The last-signal cache is
// updated as PRODUCER_FRONTIER_EXACT because waiting on the producer queue
// epoch is sufficient to observe the signaled payload's transitive
// dependencies.
void iree_hal_amdgpu_semaphore_publish_private_stream_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t producer_axis,
    uint64_t producer_epoch, uint64_t producer_value);

// Clears the semaphore's last-signal cache.
//
// Caller must verify iree_hal_amdgpu_semaphore_isa() first.
void iree_hal_amdgpu_semaphore_clear_last_signal(
    iree_hal_semaphore_t* semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
