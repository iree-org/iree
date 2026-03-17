// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_EVENT_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_CUDA_EVENT_SEMAPHORE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/timepoint_pool.h"
#include "iree/hal/utils/deferred_work_queue.h"

typedef struct iree_async_proactor_t iree_async_proactor_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an IREE HAL semaphore with the given |initial_value|.
//
// The HAL semaphore uses CUevent objects for device timepoints along the
// timeline. Those timepoints will be allocated from the |timepoint_pool|.
// Host waits delegate to iree_async_semaphore_multi_wait (futex-based).
//
// |proactor| is borrowed from the device's proactor pool and must outlive the
// semaphore.
//
// This semaphore is meant to be used together with a pending queue actions; it
// may advance the given |work_queue| if new values are signaled.
//
// Thread-safe; multiple threads may signal/wait values on the same semaphore.
iree_status_t iree_hal_cuda_event_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    const iree_hal_cuda_dynamic_symbols_t* symbols,
    iree_hal_cuda_timepoint_pool_t* timepoint_pool,
    iree_hal_deferred_work_queue_t* work_queue, iree_allocator_t host_allocator,
    iree_hal_semaphore_t** out_semaphore);

// Acquires a timepoint to signal the timeline to the given |to_value| from the
// device. The underlying CUDA event is written into |out_event| for interacting
// with CUDA APIs.
iree_status_t iree_hal_cuda_event_semaphore_acquire_timepoint_device_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t to_value,
    CUevent* out_event);

// Acquires an iree_hal_cuda_event_t object to wait on the host for the
// timeline to reach at least the given |min_value| on the device.
// Returns true and writes to |out_event| if we can find such an event;
// returns false otherwise.
// The caller should release the |out_event| once done.
bool iree_hal_cuda_semaphore_acquire_event_host_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t min_value,
    iree_hal_cuda_event_t** out_event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_EVENT_SEMAPHORE_H_
