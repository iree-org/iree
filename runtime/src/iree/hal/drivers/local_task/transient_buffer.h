// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transient buffer: a reservation handle for queue-ordered allocations.
//
// Created immediately by queue_alloca and returned to the caller before the
// backing memory exists. The drain handler allocates real memory and "commits"
// it into the transient buffer. Downstream work that waits on the alloca's
// signal semaphores sees the committed backing via the vtable forwarding.
//
// queue_dealloca "decommits" the backing: the transient buffer handle remains
// valid (the caller may still hold references) but accessing the buffer after
// decommit fails with IREE_STATUS_FAILED_PRECONDITION.
//
// This is the first step toward a pooled allocator where alloca acquires a
// reservation from a block pool (instant, no syscall) and commit/decommit
// wire/unwire physical blocks. The transient buffer IS the reservation handle.

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_TRANSIENT_BUFFER_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_TRANSIENT_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_task_transient_buffer_t
    iree_hal_task_transient_buffer_t;

// Creates a transient buffer with the given metadata but no backing memory.
// The buffer starts in the uncommitted state. The caller receives a valid
// buffer handle that can be passed to command buffers and queue operations,
// but any attempt to access the buffer's data before commit will fail.
//
// |placement| must include IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS so
// that the HAL dealloca path routes through the driver vtable.
iree_status_t iree_hal_task_transient_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

// Returns true if |buffer| is a transient buffer created by this driver.
bool iree_hal_task_transient_buffer_isa(const iree_hal_buffer_t* buffer);

// Commits a backing buffer into the transient buffer. The backing buffer is
// retained. Must be called exactly once; the buffer must be uncommitted.
// Uses release semantics so that subsequent acquires (via semaphore ordering)
// observe the committed state.
void iree_hal_task_transient_buffer_commit(iree_hal_buffer_t* buffer,
                                           iree_hal_buffer_t* backing);

// Decommits the backing buffer, releasing it and returning the transient
// buffer to the uncommitted state. Safe to call on an already-uncommitted
// buffer (idempotent for destroy safety).
void iree_hal_task_transient_buffer_decommit(iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_TRANSIENT_BUFFER_H_
