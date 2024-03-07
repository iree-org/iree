// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_EVENT_POOL_H_
#define IREE_HAL_DRIVERS_HIP_EVENT_POOL_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_hip_event_t
//===----------------------------------------------------------------------===//

// An struct that wraps a hipEvent_t object with a reference count for lifetime
// management.
//
// iree_hal_hip_event_t objects cannot be directly created; they should be
// acquired from the event pool and released back to the pool once done.
//
// Thread-safe; multiple threads may retain and release the same event.
typedef struct iree_hal_hip_event_t iree_hal_hip_event_t;

// Returns the underlying hipEvent_t handle behind |event|.
hipEvent_t iree_hal_hip_event_handle(const iree_hal_hip_event_t* event);

// Retains the given |event| by increasing its reference count.
void iree_hal_hip_event_retain(iree_hal_hip_event_t* event);

// Releases the given |event| by decreasing its reference count.
//
// |event| will be returned to its owning pool when the reference count is 0.
void iree_hal_hip_event_release(iree_hal_hip_event_t* event);

//===----------------------------------------------------------------------===//
// iree_hal_hip_event_pool_t
//===----------------------------------------------------------------------===//

// A simple pool of iree_hal_event_t objects to recycle.
//
// Thread-safe; multiple threads may acquire and release events from the pool.
typedef struct iree_hal_hip_event_pool_t iree_hal_hip_event_pool_t;

// Allocates a new event pool with up to |available_capacity| events.
//
// Extra events requested beyond the capability are directly created and
// destroyed without pooling.
iree_status_t iree_hal_hip_event_pool_allocate(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_host_size_t available_capacity, iree_allocator_t host_allocator,
    iree_hal_hip_event_pool_t** out_event_pool);

// Retains the given |event_pool| by increasing its reference count.
void iree_hal_hip_event_pool_retain(iree_hal_hip_event_pool_t* event_pool);

// Releases the given |event_pool| by decreasing its reference count.
//
// Once the |event_pool|'s reference count becomes zero, it will be freed.
void iree_hal_hip_event_pool_release(iree_hal_hip_event_pool_t* event_pool);

// Acquires one or more events from the event pool.
//
// Each returned event have an initial reference count of 1. The returned
// hipEvent_t objects may retain captured states of some queues from previous
// uses; callers should record again to overwrite.
iree_status_t iree_hal_hip_event_pool_acquire(
    iree_hal_hip_event_pool_t* event_pool, iree_host_size_t event_count,
    iree_hal_hip_event_t** out_events);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_EVENT_POOL_H_
