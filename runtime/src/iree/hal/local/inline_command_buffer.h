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

// Returns the size, in bytes, of an inline command buffer.
// This can be used for arena/stack allocations along with
// iree_hal_inline_command_buffer_initialize/iree_hal_inline_command_buffer_deinitialize.
iree_host_size_t iree_hal_inline_command_buffer_size(void);

// Initializes an inline synchronous one-shot single-threaded command "buffer".
// This is equivalent to iree_hal_inline_command_buffer_create but uses
// caller-allocated |storage| (must be at least the capacity specified by
// iree_hal_inline_command_buffer_size).
//
// NOTE: this must only be used when the command buffer handle cannot escape
// the caller: attempting to use the resulting command buffer as a ref object
// is invalid.
iree_status_t iree_hal_inline_command_buffer_initialize(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator, iree_byte_span_t storage,
    iree_hal_command_buffer_t** out_command_buffer);

// Deinitializes an inline command buffer previously initialized with
// iree_hal_inline_command_buffer_initialize.
void iree_hal_inline_command_buffer_deinitialize(
    iree_hal_command_buffer_t* command_buffer);

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
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is an inline command buffer.
bool iree_hal_inline_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_INLINE_COMMAND_BUFFER_H_
