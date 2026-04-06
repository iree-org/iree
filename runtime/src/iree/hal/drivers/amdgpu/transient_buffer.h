// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_TRANSIENT_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_TRANSIENT_BUFFER_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// AMDGPU queue-ordered transient buffer wrapper.
//
// The wrapper is returned by queue_alloca before its backing allocation becomes
// user-visible. The queue stages a borrowed provider buffer view immediately so
// later queue submissions can resolve device pointers, then commits that staged
// backing in the notification ring's pre-signal phase. queue_dealloca releases
// the pool reservation at submit time with a death frontier and decommits the
// wrapper in the pre-signal phase before publishing dealloca completion.
iree_status_t iree_hal_amdgpu_transient_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns true if |buffer| is an AMDGPU transient wrapper.
bool iree_hal_amdgpu_transient_buffer_isa(const iree_hal_buffer_t* buffer);

// Attaches a queue-owned pool reservation to |buffer|.
//
// |pool| is borrowed and must outlive the transient buffer. |reservation|
// ownership transfers to the wrapper until
// iree_hal_amdgpu_transient_buffer_release_reservation() or destroy.
void iree_hal_amdgpu_transient_buffer_attach_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation);

// Retains and stages a backing view for future queue packet emission and
// commit.
void iree_hal_amdgpu_transient_buffer_stage_backing(
    iree_hal_buffer_t* buffer, iree_hal_buffer_t* backing_buffer);

// Publishes the staged backing buffer to host-visible map/flush/invalidate
// APIs.
void iree_hal_amdgpu_transient_buffer_commit(iree_hal_buffer_t* buffer);

// Decommits the wrapper and releases the staged backing view.
void iree_hal_amdgpu_transient_buffer_decommit(iree_hal_buffer_t* buffer);

// Marks the wrapper as queued for deallocation. Returns false if a dealloca has
// already been queued for this wrapper.
bool iree_hal_amdgpu_transient_buffer_begin_dealloca(iree_hal_buffer_t* buffer);

// Clears a queued-dealloca marker after a submission/capture failure. Must only
// be used when no dealloca completion action was published.
void iree_hal_amdgpu_transient_buffer_abort_dealloca(iree_hal_buffer_t* buffer);

// Releases the attached reservation exactly once. No-op if none is attached or
// if the reservation has already been released.
void iree_hal_amdgpu_transient_buffer_release_reservation(
    iree_hal_buffer_t* buffer, const iree_async_frontier_t* death_frontier);

// Returns a backing buffer suitable for queue packet emission, or NULL if the
// wrapper has no staged backing or has already been queued for deallocation.
//
// This is intentionally more permissive than map/flush/invalidate: queue packet
// builders need the staged backing before user-visible alloca commit so
// same-queue and cross-queue submissions can be chained by semaphore waits.
iree_hal_buffer_t* iree_hal_amdgpu_transient_buffer_backing_buffer(
    iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_TRANSIENT_BUFFER_H_
