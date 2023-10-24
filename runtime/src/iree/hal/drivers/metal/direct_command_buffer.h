// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_METAL_METAL_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_METAL_METAL_COMMAND_BUFFER_H_

#import <Metal/Metal.h>

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/api.h"
#include "iree/hal/drivers/metal/builtin_executables.h"
#include "iree/hal/drivers/metal/refcount_block_pool.h"
#include "iree/hal/drivers/metal/staging_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a Metal command buffer that directly records into a MTLCommandBuffer.
// Such command buffers are one shot--they can only be submitted once.
//
// The command buffer would have the given |mode| and be recorded and submitted
// to the given |queue|.
//
// |block_pool| will be used for internal allocations and retaining copies of
// input data until reset.
//
// |staging_buffer| is used for recording argument buffers and uploading source
// buffer data for buffer updates.
//
// |builtin_executables| are used for polyfilling fill/copy/update buffers that
// are not directly supported by Metal APIs.
//
// |out_command_buffer| must be released by the caller (see
// iree_hal_command_buffer_release).
iree_status_t iree_hal_metal_direct_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity,
    iree_hal_metal_command_buffer_resource_reference_mode_t
        resource_reference_mode,
    id<MTLCommandQueue> queue, iree_hal_metal_arena_block_pool_t* block_pool,
    iree_hal_metal_staging_buffer_t* staging_buffer,
    iree_hal_metal_builtin_executable_t* builtin_executable,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a direct Metal command buffer.
bool iree_hal_metal_direct_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns the underlying Metal command buffer handle for the given
// |command_buffer|.
id<MTLCommandBuffer> iree_hal_metal_direct_command_buffer_handle(
    const iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_METAL_METAL_COMMAND_BUFFER_H_
