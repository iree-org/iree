// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_EVENT_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_HIP_EVENT_SEMAPHORE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/per_device_information.h"

typedef struct iree_hal_hip_event_t iree_hal_hip_event_t;
typedef struct iree_hal_hip_event_pool_t iree_hal_hip_event_pool_t;
typedef iree_status_t (*iree_hal_hip_event_semaphore_scheduled_callback_t)(
    void* user_data, iree_hal_semaphore_t* semaphore,
    iree_status_t semaphore_status);

// Creates an IREE HAL semaphore with the given |initial_value|.
//
// The HAL semaphore are backed by iree_event_t or hipEvent_t objects for
// different timepoints along the timeline under the hood. Those timepoints will
// be allocated from the |timepoint_pool|.
//
// Thread-safe; multiple threads may signal/wait values on the same semaphore.
iree_status_t iree_hal_hip_event_semaphore_create(
    uint64_t initial_value, const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_allocator_t host_allocator, iree_hal_hip_device_topology_t topology,
    iree_hal_semaphore_t** out_semaphore);

// Performs a multi-wait on one or more semaphores. Returns
// IREE_STATUS_DEADLINE_EXCEEDED if the wait does not complete before |timeout|.
iree_status_t iree_hal_hip_semaphore_multi_wait(
    const iree_hal_semaphore_list_t semaphore_list,
    iree_hal_wait_mode_t wait_mode, iree_timeout_t timeout,
    iree_allocator_t host_allocator);

// Adds a work item to be executed once we have a forward progress
// guarantee on this semaphore to reach a particular value.
// The event pool must be an event pool specifically
// for the queue that will be doing the work.
iree_status_t iree_hal_hip_semaphore_notify_work(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_hip_event_pool_t* event_pool,
    iree_hal_hip_event_semaphore_scheduled_callback_t callback,
    void* user_data);

// Notifies this semaphore that we have guaranteed
// forward progress until the particular value is reached.
iree_status_t iree_hal_hip_semaphore_notify_forward_progress_to(
    iree_hal_semaphore_t* base_semaphore, uint64_t value);

// Returns the hip event that needs to be signaled in order
// for the semaphore to reach a given value.
// This event *must* have been previously notified for
// forward progress by iree_hal_hip_semaphore_notify_forward_progress_to.
// If the return status is iree_ok_status(), and the out_hip_event is NULL,
// it is because the event has already been signaled, and the result
// is visible on the host.
// The refcount for the event is incremented here, and the caller
// must decrement when no longer needed.
iree_status_t iree_hal_hip_semaphore_wait_hip_events(
    iree_hal_semaphore_t* base_semaphore, uint64_t value, hipStream_t stream);

// Waits until all exported timepoints (up to value) have been
// submitted to the dispatch thread.
iree_status_t iree_hal_hip_semaphore_for_exported_timepoints(
    iree_hal_semaphore_t* base_semaphore, uint64_t value);

iree_status_t iree_hal_hip_semaphore_create_event_and_record_if_necessary(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_hip_per_device_info_t* device, hipStream_t dispatch_stream,
    iree_hal_hip_event_pool_t* event_pool);

iree_status_t iree_hal_hip_event_semaphore_advance(
    iree_hal_semaphore_t* semaphore);

#endif  // IREE_HAL_DRIVERS_HIP_EVENT_SEMAPHORE_H_
