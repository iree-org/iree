// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_DISPATCH_THREAD_H_
#define IREE_HAL_DRIVERS_HIP_DISPATCH_THREAD_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"

// iree_hal_hip_dispatch_thread is used to get work off of the main thread.
// This is important to do for a single reason. There are 2 types of
// command buffer that we use in hip. One is a pre-recorded command buffer
// iree_hal_deferred_command_buffer_t, which when executed
// calls all of the associated hipStream based commands.
// The other is iree_hal_hip_graph_command_buffer_t which when executed
// executes hipGraphLaunch. In practice what hipGraphLaunch does
// under the hood is call the associated stream API for each node in the
// graph. Either way these block the main thread for
// quite a lot of time. If a host program wants to execute
// command buffers on multiple GPUs from the same thread
// blocking that thread will cause stalls.
// So instead this thread exists to simply move that
// work off of the main thread. There are a couple of
// caveats, as now we have to move async allocations and deallocations
// to that thread as well, as they need to remain in-order.
typedef struct iree_hal_hip_dispatch_thread_t iree_hal_hip_dispatch_thread_t;

typedef struct iree_hal_hip_event_t iree_hal_hip_event_t;

typedef iree_status_t (*iree_hal_hip_dispatch_callback_t)(void* user_data,
                                                          iree_status_t status);

// Initializes the dispatch thread for HIP driver.
iree_status_t iree_hal_hip_dispatch_thread_initialize(
    iree_allocator_t host_allocator,
    iree_hal_hip_dispatch_thread_t** out_thread);

// Deinitializes the dispatch thread for HIP driver.
void iree_hal_hip_dispatch_thread_deinitialize(
    iree_hal_hip_dispatch_thread_t* thread);

// Adds a dispatch to the thread, which will be executed
// in order.
//
// |user_data| must remain valid until the callback is called,
// and it is up to the callee to clean up user_data if required.
// The callback will always be called regardless of whether
// or not this function returns an error. An error indicates there
// was an asynchronous failure on the thread, or a semaphore.
iree_status_t iree_hal_hip_dispatch_thread_add_dispatch(
    iree_hal_hip_dispatch_thread_t* thread,
    iree_hal_hip_dispatch_callback_t callback, void* user_data);

#endif  // IREE_HAL_DRIVERS_HIP_DISPATCH_THREAD_H_
