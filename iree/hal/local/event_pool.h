// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_LOCAL_EVENT_POOL_H_
#define IREE_HAL_LOCAL_EVENT_POOL_H_

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A simple pool of iree_event_ts to recycle.
//
// Thread-safe; multiple threads may acquire and release events from the pool.
typedef struct iree_hal_local_event_pool_s iree_hal_local_event_pool_t;

// Allocates a new event pool with up to |available_capacity| events.
iree_status_t iree_hal_local_event_pool_allocate(
    iree_host_size_t available_capacity, iree_allocator_t host_allocator,
    iree_hal_local_event_pool_t** out_event_pool);

// Deallocates an event pool and destroys all events.
// All events that were acquired from the pool must have already been released
// back to it prior to deallocation.
void iree_hal_local_event_pool_free(iree_hal_local_event_pool_t* event_pool);

// Acquires one or more events from the event pool.
// The returned events will be unsignaled and ready for use. Callers may set and
// reset the events as much as they want prior to releasing them back to the
// pool with iree_hal_local_event_pool_release.
iree_status_t iree_hal_local_event_pool_acquire(
    iree_hal_local_event_pool_t* event_pool, iree_host_size_t event_count,
    iree_event_t* out_events);

// Releases one or more events back to the block pool.
void iree_hal_local_event_pool_release(iree_hal_local_event_pool_t* event_pool,
                                       iree_host_size_t event_count,
                                       iree_event_t* events);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_EVENT_POOL_H_
