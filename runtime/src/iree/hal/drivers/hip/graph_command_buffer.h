// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_GRAPH_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_HIP_GRAPH_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/hip_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// NOTE: hipGraph API used in this module is marked as beta in the HIP
// documentation, meaning, while this is feature complete it is still open to
// changes and may have outstanding issues.

typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;

// Creates a command buffer that records into a HIP graph.
//
// NOTE: the |block_pool| must remain live for the lifetime of the command
// buffers that use it.
iree_status_t iree_hal_hip_graph_command_buffer_create(
    iree_hal_device_t* device,
    const iree_hal_hip_dynamic_symbols_t* hip_symbols, hipCtx_t context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a HIP graph-based command buffer.
bool iree_hal_hip_graph_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns the native HIP graph associated to the command buffer.
hipGraphExec_t iree_hal_hip_graph_command_buffer_handle(
    iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_GRAPH_COMMAND_BUFFER_H_
