// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_DEFERRED_COMMAND_BUFFER_H_
#define IREE_HAL_UTILS_DEFERRED_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t deferred record/replay wrapper
//===----------------------------------------------------------------------===//

// Records an in-memory command buffer that can be replayed against a target
// command buffer at a later time.
//
// Argument arrays (like push constants) and host buffers (like the source
// buffer in iree_hal_command_buffer_update_buffer) that usually live on the
// stack will be cloned. All resources used will be referenced until the command
// buffer is reset or released.
//
// |block_pool| will be used to allocate the underlying storage and the blocks
// will be retained until the command buffer is reset or released, or if
// IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT is set after the first time the command
// buffer is replayed. The block size of the pool can be whatever the caller
// wants with the caveat being that smaller sizes may result in more oversized
// allocations from the system. 16KB, 32KB, and 64KB are reasonable starting
// points based on system availability.
// NOTE: the |block_pool| must remain live for the lifetime of the command
// buffers that use it.
//
// After recording iree_hal_deferred_command_buffer_apply can be used to replay
// the sequence of commands against a target command buffer implementation.
// The command buffer can be replayed multiple times.
IREE_API_EXPORT iree_status_t iree_hal_deferred_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a deferred command buffer.
IREE_API_EXPORT bool iree_hal_deferred_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Replays a recorded |command_buffer| against a |target_command_buffer|.
// If the command buffer was recorded in one-shot mode it will be reset upon
// return. The provided |binding_table| will be used for indirect bindings
// referenced in the command buffer.
IREE_API_EXPORT iree_status_t iree_hal_deferred_command_buffer_apply(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_DEFERRED_COMMAND_BUFFER_H_
