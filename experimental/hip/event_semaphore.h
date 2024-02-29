// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HIP_EVENT_SEMAPHORE_H_
#define IREE_EXPERIMENTAL_HIP_EVENT_SEMAPHORE_H_

#include <stdint.h>

#include "experimental/hip/dynamic_symbols.h"
#include "experimental/hip/pending_queue_actions.h"
#include "experimental/hip/timepoint_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an IREE HAL semaphore with the given |initial_value|.
//
// The HAL semaphore are backed by iree_event_t or hipEvent_t objects for
// different timepoints along the timeline under the hood. Those timepoints will
// be allocated from the |timepoint_pool|.
//
// This semaphore is meant to be used together with a pending queue actions; it
// may advance the given |pending_queue_actions| if new values are signaled.
//
// Thread-safe; multiple threads may signal/wait values on the same semaphore.
iree_status_t iree_hal_hip_event_semaphore_create(
    uint64_t initial_value, const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_hal_hip_timepoint_pool_t* timepoint_pool,
    iree_hal_hip_pending_queue_actions_t* pending_queue_actions,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

// Acquires a timepoint to signal the timeline to the given |to_value| from the
// device. The underlying HIP event is written into |out_event| for interacting
// with HIP APIs.
iree_status_t iree_hal_hip_event_semaphore_acquire_timepoint_device_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t to_value,
    hipEvent_t* out_event);

// Acquires a timepoint to wait the timeline to reach at least the given
// |min_value| on the device. The underlying HIP event is written into
// |out_event| for interacting with HIP APIs.
iree_status_t iree_hal_hip_event_semaphore_acquire_timepoint_device_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t min_value,
    hipEvent_t* out_event);

// Performs a multi-wait on one or more semaphores.
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the wait does not complete before
// |deadline_ns| elapses.
iree_status_t iree_hal_hip_semaphore_multi_wait(
    const iree_hal_semaphore_list_t semaphore_list,
    iree_hal_wait_mode_t wait_mode, iree_timeout_t timeout,
    iree_arena_block_pool_t* block_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HIP_EVENT_SEMAPHORE_H_
