// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_INLINE_COMMAND_BUFFER_H_
#define IREE_HAL_LOCAL_INLINE_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an inline synchronous one-shot single-threaded command "buffer".
// This is designed for ultra-low latency situations where we know the command
// buffer is going to be submitted with no wait semaphores indicating that it
// can begin execution immediately. No inter-command-buffer scheduling will be
// performed and all barriers and events are ignored.
//
// Executes all work on the calling thread synchronously (today).
//
// Must have IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION set.
iree_status_t iree_hal_inline_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_INLINE_COMMAND_BUFFER_H_
