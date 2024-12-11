// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_MULTI_QUEUE_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_HIP_MULTI_QUEUE_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/per_device_information.h"

typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t deferred record/replay wrapper
//===----------------------------------------------------------------------===//

// Creates a command buffer that records into multiple command buffers
// at a time based on the given queue affinity.
//
// After recording the underlying command buffers can be retrieved with
// iree_hal_hip_multi_queue_command_buffer_get for submission.
IREE_API_EXPORT iree_status_t iree_hal_hip_multi_queue_command_buffer_create(
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t** in_command_buffers,
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_hip_dynamic_symbols_t* hip_symbols,
    iree_hal_hip_device_topology_t topology, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a multi command buffer.
IREE_API_EXPORT bool iree_hal_hip_multi_queue_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns a recorded command_buffer with the given |queue_affinity|.
// It is expected that only a single bit is set for the queue affinity here.
IREE_API_EXPORT iree_status_t iree_hal_hip_multi_queue_command_buffer_get(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer);

#endif  // IREE_HAL_DRIVERS_HIP_MULTI_QUEUE_COMMAND_BUFFER_H_
