// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_CLEANUP_THREAD_H_
#define IREE_HAL_DRIVERS_HIP_CLEANUP_THREAD_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"

typedef struct iree_hal_hip_cleanup_thread_t iree_hal_hip_cleanup_thread_t;
typedef struct iree_hal_hip_event_t iree_hal_hip_event_t;

typedef iree_status_t (*iree_hal_hip_cleanup_callback_t)(
    void* user_data, iree_hal_hip_event_t* event, iree_status_t status);

// Initializes the cleanup thread for HIP driver.
iree_status_t iree_hal_hip_cleanup_thread_initialize(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_allocator_t host_allocator,
    iree_hal_hip_cleanup_thread_t** out_thread);

// Deinitializes the cleanup thread for HIP driver.
void iree_hal_hip_cleanup_thread_deinitialize(
    iree_hal_hip_cleanup_thread_t* thread);

// Adds a pending cleanup to the thread.
//
// The thread will wait on the event and fire the callback,
// once the event has completed.
// |user_data| must remain valid until the callback is called,
// and it is up to the callee to clean up user_data if required.
iree_status_t iree_hal_hip_cleanup_thread_add_cleanup(
    iree_hal_hip_cleanup_thread_t* thread, iree_hal_hip_event_t* event,
    iree_hal_hip_cleanup_callback_t callback, void* user_data);

#endif  // IREE_HAL_DRIVERS_HIP_CLEANUP_THREAD_H_
