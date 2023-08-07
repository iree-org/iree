// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_METAL_METAL_SHARED_EVENT_H_
#define IREE_HAL_DRIVERS_METAL_METAL_SHARED_EVENT_H_

#import <Metal/Metal.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a Metal shared event with the given |initial_value| to implement an
// IREE semaphore.
//
// |listener| is used for dispatching notifications for async execution.
//
// |out_semaphore| must be released by the caller (see
// iree_hal_semaphore_release).
iree_status_t iree_hal_metal_shared_event_create(
    id<MTLDevice> device, uint64_t initial_value,
    MTLSharedEventListener* listener, iree_allocator_t host_allocator,
    iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is a Metal shared event.
bool iree_hal_metal_shared_event_isa(iree_hal_semaphore_t* semaphore);

// Returns the underlying Metal shared event handle for the given |semaphore|.
id<MTLSharedEvent> iree_hal_metal_shared_event_handle(
    const iree_hal_semaphore_t* semaphore);

// Waits on the shared events in the given |semaphore_list| according to the
// |wait_mode| before |timeout|.
iree_status_t iree_hal_metal_shared_event_multi_wait(
    iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_METAL_METAL_SHARED_EVENT_H_
