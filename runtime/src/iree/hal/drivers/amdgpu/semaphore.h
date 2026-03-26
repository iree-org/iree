// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;
typedef struct iree_hal_amdgpu_virtual_queue_t iree_hal_amdgpu_virtual_queue_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_last_signal_t
//===----------------------------------------------------------------------===//

// Seqlock-protected cache of the most recent queue signal on a semaphore.
// Written by the submission path when queue_execute signals the semaphore,
// read by the submission path when processing waits (for same-queue FIFO
// elision and cross-queue epoch lookups) and by the host-wait fast path.
//
// The seqlock ensures torn reads across the three payload fields are detected
// and retried. Writers increment the sequence counter to an odd value before
// the update and to an even value after. Readers retry if the sequence is odd
// (write in progress) or changed between the start and end of the read.
typedef struct iree_hal_amdgpu_last_signal_t {
  iree_atomic_int32_t sequence;
  iree_hal_amdgpu_virtual_queue_t* queue;
  uint64_t epoch;
  uint64_t value;
} iree_hal_amdgpu_last_signal_t;

// Stores a new last-signal snapshot. Thread-safe (seqlock writer).
static inline void iree_hal_amdgpu_last_signal_store(
    iree_hal_amdgpu_last_signal_t* cache,
    iree_hal_amdgpu_virtual_queue_t* queue, uint64_t epoch, uint64_t value) {
  // Increment to odd: signals write in progress.
  iree_atomic_fetch_add(&cache->sequence, 1, iree_memory_order_acquire);
  cache->queue = queue;
  cache->epoch = epoch;
  cache->value = value;
  // Increment to even: signals write complete.
  iree_atomic_fetch_add(&cache->sequence, 1, iree_memory_order_release);
}

// Loads the last-signal snapshot. Thread-safe (seqlock reader).
// Returns true if the cache has been written at least once (queue != NULL).
static inline bool iree_hal_amdgpu_last_signal_load(
    const iree_hal_amdgpu_last_signal_t* cache,
    iree_hal_amdgpu_virtual_queue_t** out_queue, uint64_t* out_epoch,
    uint64_t* out_value) {
  int32_t sequence;
  do {
    sequence = iree_atomic_load(&cache->sequence, iree_memory_order_acquire);
    if (IREE_UNLIKELY(sequence & 1)) continue;  // writer in progress
    *out_queue = cache->queue;
    *out_epoch = cache->epoch;
    *out_value = cache->value;
  } while (
      IREE_UNLIKELY(iree_atomic_load(&cache->sequence,
                                     iree_memory_order_acquire) != sequence));
  return *out_queue != NULL;
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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
