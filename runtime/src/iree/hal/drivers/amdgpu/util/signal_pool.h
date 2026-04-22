// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_SIGNAL_POOL_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_SIGNAL_POOL_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_signal_pool_t
//===----------------------------------------------------------------------===//

// Default number of signals to pre-create in a batch when the pool is empty.
// Each signal costs ~10-50us to create via hsa_amd_signal_create, so batching
// amortizes the overhead across multiple acquisitions.
#define IREE_HAL_AMDGPU_HOST_SIGNAL_POOL_BATCH_SIZE_DEFAULT 32

// Pool of HSA signals created via hsa_amd_signal_create.
// These are full-featured signals with interrupt capability (mailbox event,
// eventfd bridge) suitable for host waits, cross-device synchronization, and
// proactor integration.
//
// Signals are expensive to create (~10-50us each) so the pool pre-creates them
// in batches and maintains a free list for O(1) acquire/release. Released
// signals are returned to the free list for reuse; they are only destroyed when
// the pool is deinitialized.
//
// Thread-safe.
typedef struct iree_hal_amdgpu_host_signal_pool_t {
  // HSA API handle. Unowned.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Host allocator for the free list array.
  iree_allocator_t host_allocator;
  // Number of signals to create per batch when the pool is empty.
  iree_host_size_t batch_size;

  // Guards access to the free list and growth.
  iree_slim_mutex_t mutex;
  // LIFO stack of available signals. Capacity grows to steady state and stays.
  hsa_signal_t* free_signals IREE_GUARDED_BY(mutex);
  iree_host_size_t free_count IREE_GUARDED_BY(mutex);
  iree_host_size_t free_capacity IREE_GUARDED_BY(mutex);
  // Total signals created by this pool. Used to assert all signals are returned
  // before deinitialization.
  iree_host_size_t allocated_count IREE_GUARDED_BY(mutex);
} iree_hal_amdgpu_host_signal_pool_t;

// Initializes the host signal pool.
// |initial_capacity| signals will be pre-created. If 0, signals are created
// lazily on first acquire.
iree_status_t iree_hal_amdgpu_host_signal_pool_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_host_size_t initial_capacity,
    iree_host_size_t batch_size, iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_signal_pool_t* out_pool);

// Deinitializes the pool and destroys all signals.
// All acquired signals must have been released before calling.
void iree_hal_amdgpu_host_signal_pool_deinitialize(
    iree_hal_amdgpu_host_signal_pool_t* pool);

// Acquires a signal from the pool, resetting its value to |initial_value|.
// The signal will have interrupt capability for host waits.
// Must be released back to the pool when no longer needed.
iree_status_t iree_hal_amdgpu_host_signal_pool_acquire(
    iree_hal_amdgpu_host_signal_pool_t* pool, hsa_signal_value_t initial_value,
    hsa_signal_t* out_signal);

// Releases a signal back to the pool for reuse.
// The signal must not be in use by any pending operations.
void iree_hal_amdgpu_host_signal_pool_release(
    iree_hal_amdgpu_host_signal_pool_t* pool, hsa_signal_t signal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_SIGNAL_POOL_H_
