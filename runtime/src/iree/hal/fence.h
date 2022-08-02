// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_FENCE_H_
#define IREE_HAL_FENCE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_fence_t
//===----------------------------------------------------------------------===//

// A list of semaphores and their corresponding payloads.
// When signaling each semaphore will be set to the new payload value provided.
// When waiting each semaphore must reach or exceed the payload value.
// This points at external storage and does not retain the semaphores itself.
typedef struct iree_hal_semaphore_list_t {
  iree_host_size_t count;
  iree_hal_semaphore_t** semaphores;
  uint64_t* payload_values;
} iree_hal_semaphore_list_t;

// A set of semaphores and their corresponding payloads.
// When signaling each semaphore will be set to the new payload value provided.
// When waiting each semaphore must reach or exceed the payload value.
//
// Fences can also store additional internal information and are more efficient
// when used for both signaling and waiting; users should try to build as few
// fences as possible. Semaphores are retained for the lifetime of the fence.
//
// Fences must not be modified once consumed by an API call; mutation behavior
// is undefined if any queue operations using the fence are in-flight.
//
// APIs that accept fences allow NULL to indicate that no fencing is required.
// Waiting on a NULL fence completes immediately and signaling a NULL fence is
// a no-op.
typedef struct iree_hal_fence_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;
  uint16_t capacity;
  uint16_t count;
  // Following arrays aligned to their natural element alignment:
  // iree_hal_semaphore_t* semaphores[capacity];
  // uint64_t payload_values[capacity];
} iree_hal_fence_t;

// Creates a new fence with the given |capacity| and returns it in |out_fence|.
// The capacity defines the maximum number of unique semaphores that can be
// inserted into the fence.
IREE_API_EXPORT iree_status_t iree_hal_fence_create(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_hal_fence_t** out_fence);

// Creates a new fence joining all |fences| as a wait-all operation.
IREE_API_EXPORT iree_status_t iree_hal_fence_join(
    iree_host_size_t fence_count, iree_hal_fence_t** fences,
    iree_allocator_t host_allocator, iree_hal_fence_t** out_fence);

// Retains the |fence| for the caller.
IREE_API_EXPORT void iree_hal_fence_retain(iree_hal_fence_t* fence);

// Releases |fence| and destroys it if the caller is the last owner.
IREE_API_EXPORT void iree_hal_fence_release(iree_hal_fence_t* fence);

// Returns a list of unique semaphores and their maximum payload values.
IREE_API_EXPORT iree_hal_semaphore_list_t
iree_hal_fence_semaphore_list(iree_hal_fence_t* fence);

// Returns the number of unique timepoints the fence represents.
IREE_API_EXPORT iree_host_size_t
iree_hal_fence_timepoint_count(const iree_hal_fence_t* fence);

// Inserts a |semaphore| with the given payload |value| into |fence|.
// If the semaphore is already present the maximum value between this and the
// existing insertion will be used.
IREE_API_EXPORT iree_status_t iree_hal_fence_insert(
    iree_hal_fence_t* fence, iree_hal_semaphore_t* semaphore, uint64_t value);

// Queries the status of the fence.
// Returns OK if the fence has been signaled, IREE_STATUS_DEFERRED if it has
// not yet been signaled, or a failure if one or more timepoint semaphores have
// failed. The same failure status will be returned regardless of when in the
// timeline the error occurred.
IREE_API_EXPORT iree_status_t iree_hal_fence_query(iree_hal_fence_t* fence);

// Signals a |fence| to indicate the joined timepoint it represents has been
// reached.
IREE_API_EXPORT iree_status_t iree_hal_fence_signal(iree_hal_fence_t* fence);

// Signals a |fence| to indicate it has failed and all semaphores will fail with
// |signal_status|.
IREE_API_EXPORT void iree_hal_fence_fail(iree_hal_fence_t* fence,
                                         iree_status_t signal_status);

// Blocks the caller until the fence is reached or the |timeout| elapses.
//
// Returns success if the wait is successful and the fence reached its
// timepoints successfully.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the |timeout| elapses without the
// fence being reached. If an asynchronous failure occurred on any timeline
// tracked by the fence this will return the failure status that was set
// immediately.
//
// Returns IREE_STATUS_ABORTED if one or more semaphores has failed. Callers can
// use iree_hal_fence_query to get the status.
//
// NOTE: this is not the most optimal way to wait on fences; if at all possible
// use a single wait on a single semaphore to avoid additional overheads in
// multiplexing fences across device implementations. This fence wait should be
// used to perform a join that will propagate failures from any semaphore used
// in timepoints.
IREE_API_EXPORT iree_status_t iree_hal_fence_wait(iree_hal_fence_t* fence,
                                                  iree_timeout_t timeout);

// Returns a wait source reference to |fence| after it reaches or exceeds
// all defined timepoints.
IREE_API_EXPORT iree_wait_source_t
iree_hal_fence_await(iree_hal_fence_t* fence);

//===----------------------------------------------------------------------===//
// iree_hal_fence_t implementation details
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_fence_destroy(iree_hal_fence_t* fence);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_FENCE_H_
