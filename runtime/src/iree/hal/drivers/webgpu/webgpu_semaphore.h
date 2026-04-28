// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_SEMAPHORE_H_

#include "iree/async/frontier.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_semaphore_t
//===----------------------------------------------------------------------===//

// Creates a WebGPU timeline semaphore.
//
// WebGPU has no hardware semaphore/fence primitives. Synchronization is
// implicit within the single queue (commands execute in submission order) and
// explicit between CPU and GPU via onSubmittedWorkDone() and mapAsync().
//
// This semaphore tracks queue progress as a monotonically increasing timeline
// value. The device signals the semaphore when onSubmittedWorkDone()
// completions arrive through the proactor. The async semaphore base provides
// timeline tracking, frontier merge, and timepoint dispatch — the WebGPU
// semaphore adds only the wait implementation.
iree_status_t iree_hal_webgpu_semaphore_create(
    iree_async_proactor_t* proactor, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

// Returns true if the semaphore has a pending submitted signal from |axis|
// that will reach at least |minimum_value|. Used by queue ops to determine
// if GPU FIFO ordering guarantees a wait will be satisfied without needing
// an async proactor wait.
bool iree_hal_webgpu_semaphore_has_submitted_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t axis,
    uint64_t minimum_value);

// Records that a signal to |value| has been submitted by |axis|. Called
// after GPU work is submitted in fast-path queue operations so that
// subsequent same-queue operations can use FIFO wait elision.
void iree_hal_webgpu_semaphore_mark_submitted_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t axis, uint64_t value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_SEMAPHORE_H_
