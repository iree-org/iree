// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/handle_table.h"
#include "iree/hal/drivers/webgpu/webgpu_builder.h"
#include "iree/hal/drivers/webgpu/webgpu_builtins.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_command_buffer_t
//===----------------------------------------------------------------------===//

// Creates a WebGPU command buffer that records commands into a compact uint32
// instruction stream via the builder. The instruction stream is executed by a
// JS-side processor at queue_execute time.
//
// For ONE_SHOT command buffers: the instruction stream is passed directly to
// the JS processor via execute_instructions (no caching).
//
// For reusable command buffers: end() creates a JS-side Recording object that
// caches the instruction stream and static bindings. Subsequent queue_execute
// calls only resolve dynamic bindings.
//
// |device_handle| is the bridge handle for the GPUDevice.
// |queue_handle| is the bridge handle for the GPUQueue.
// |builtins| provides pre-created compute pipelines for fill/copy operations.
// Must remain valid for the lifetime of the command buffer.
// |block_pool| is the shared block pool for instruction stream blocks. Must
// remain valid for the lifetime of the command buffer.
iree_status_t iree_hal_webgpu_command_buffer_create(
    iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_handle_t queue_handle,
    const iree_hal_webgpu_builtins_t* builtins,
    iree_arena_block_pool_t* block_pool, iree_hal_allocator_t* device_allocator,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a WebGPU command buffer.
bool iree_hal_webgpu_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns the builder for direct access (used by queue operations that build
// scratch instruction streams).
iree_hal_webgpu_builder_t* iree_hal_webgpu_command_buffer_builder(
    iree_hal_command_buffer_t* command_buffer);

// Returns the recording handle for reusable command buffers. Returns 0 for
// ONE_SHOT command buffers (the instruction stream is executed directly).
iree_hal_webgpu_handle_t iree_hal_webgpu_command_buffer_recording_handle(
    iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_COMMAND_BUFFER_H_
