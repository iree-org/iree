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

#ifndef IREE_HAL_LOCAL_SYNC_SEMAPHORE_H_
#define IREE_HAL_LOCAL_SYNC_SEMAPHORE_H_

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_sync_semaphore_state_t
//===----------------------------------------------------------------------===//

// State shared between all sync semaphores.
// Owned by the device and guaranteed to remain valid for the lifetime of any
// semaphore created from it.
typedef struct {
  // In-process notification signaled when any semaphore value changes.
  iree_notification_t notification;
} iree_hal_sync_semaphore_state_t;

// Initializes state used to perform semaphore synchronization.
void iree_hal_sync_semaphore_state_initialize(
    iree_hal_sync_semaphore_state_t* out_shared_state);

// Deinitializes state used to perform semaphore synchronization; no semaphores
// must be live with references.
void iree_hal_sync_semaphore_state_deinitialize(
    iree_hal_sync_semaphore_state_t* shared_state);

//===----------------------------------------------------------------------===//
// iree_hal_sync_semaphore_t
//===----------------------------------------------------------------------===//

// Creates a semaphore that allows for ordering of operations on the local host.
// Backed by a shared iree_notification_t in |shared_state|. Not efficient under
// high contention or many simultaneous users but that's not what the
// synchronous backend is intended for - if you want something efficient in the
// face of hundreds or thousands of active asynchronous operations then use the
// task system.
iree_status_t iree_hal_sync_semaphore_create(
    iree_hal_sync_semaphore_state_t* shared_state, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

// Performs a signal of a list of semaphores.
// The semaphores will transition to their new values (nearly) atomically and
// batching up signals will reduce synchronization overhead.
iree_status_t iree_hal_sync_semaphore_multi_signal(
    iree_hal_sync_semaphore_state_t* shared_state,
    const iree_hal_semaphore_list_t* semaphore_list);

// Performs a multi-wait on one or more semaphores.
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the wait does not complete before
// |timeout| elapses.
iree_status_t iree_hal_sync_semaphore_multi_wait(
    iree_hal_sync_semaphore_state_t* shared_state,
    iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_SYNC_SEMAPHORE_H_
