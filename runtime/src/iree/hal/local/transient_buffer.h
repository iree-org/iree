// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transient buffer: a reservation handle for queue-ordered local allocations.
//
// Queue allocators can return a transient buffer to the caller before the
// physical backing is ready, then commit a real backing buffer once queue
// ordering allows it. queue_dealloca can later decommit the wrapper while the
// transient buffer object itself remains live so stale host references fail
// cleanly with IREE_STATUS_FAILED_PRECONDITION instead of accessing freed
// storage.

#ifndef IREE_HAL_LOCAL_TRANSIENT_BUFFER_H_
#define IREE_HAL_LOCAL_TRANSIENT_BUFFER_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_local_transient_buffer_t
    iree_hal_local_transient_buffer_t;

// Creates a transient buffer with the given metadata but no backing memory.
// The buffer starts in the uncommitted state.
//
// |allocation_size| is the physical reservation size to report through
// iree_hal_buffer_allocation_size() and |byte_length| is the logical byte range
// exposed by the wrapper. |byte_length| must be <= |allocation_size|.
//
// |placement| must include IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS so the
// HAL dealloca path routes through the owning driver's queue_dealloca vtable.
iree_status_t iree_hal_local_transient_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns true if |buffer| is a local transient buffer wrapper.
bool iree_hal_local_transient_buffer_isa(const iree_hal_buffer_t* buffer);

// Returns true if |buffer| has a committed backing buffer visible to callers.
//
// This is a cheap state query for queue implementations that need to decide
// whether a transient can be mapped immediately or must be resolved after its
// queue allocation dependency has completed.
bool iree_hal_local_transient_buffer_is_committed(iree_hal_buffer_t* buffer);

// Returns the stable profiling id assigned to this transient buffer wrapper.
//
// The id is process-global and nonzero. Producers use it as a session-local
// allocation id whenever the wrapper participates in an active profile capture.
uint64_t iree_hal_local_transient_buffer_profile_id(iree_hal_buffer_t* buffer);

// Attaches a pool reservation to the transient buffer. The wrapper keeps only
// a borrowed pointer to |pool| and takes ownership of |reservation| until
// iree_hal_local_transient_buffer_release_reservation() or wrapper destroy.
void iree_hal_local_transient_buffer_attach_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation);

// Stages a materialized backing buffer view for a future commit. The backing
// buffer is retained, but remains invisible to map/flush/invalidate calls until
// iree_hal_local_transient_buffer_commit() publishes it.
void iree_hal_local_transient_buffer_stage_backing(iree_hal_buffer_t* buffer,
                                                   iree_hal_buffer_t* backing);

// Publishes the staged backing buffer. Must be called exactly once while the
// wrapper is uncommitted and has a staged backing view.
void iree_hal_local_transient_buffer_commit(iree_hal_buffer_t* buffer);

// Decommits the backing buffer and returns the wrapper to the uncommitted
// state. Any staged-but-uncommitted backing view is also released. Safe to
// call on an already-uncommitted wrapper.
void iree_hal_local_transient_buffer_decommit(iree_hal_buffer_t* buffer);

// Returns the attached pool reservation without transferring ownership.
//
// This is a cold diagnostic/profiling helper. Returns false when the wrapper
// has no live reservation or the reservation has already been released.
bool iree_hal_local_transient_buffer_query_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation);

// Releases the attached reservation exactly once. If the wrapper has no
// reservation or the reservation was already released, this is a no-op.
//
// |death_frontier| is forwarded to iree_hal_pool_release_reservation() when
// the reservation is still owned. Pass NULL for an immediately reusable
// reservation in synchronous/drain-ordered paths.
void iree_hal_local_transient_buffer_release_reservation(
    iree_hal_buffer_t* buffer, const iree_async_frontier_t* death_frontier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_TRANSIENT_BUFFER_H_
