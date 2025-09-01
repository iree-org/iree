// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_

#include "iree/hal/drivers/amdgpu/util/error_callback.h"
#include "iree/hal/drivers/amdgpu/virtual_queue.h"

typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;
typedef struct iree_hal_amdgpu_block_allocators_t
    iree_hal_amdgpu_block_allocators_t;
typedef struct iree_hal_amdgpu_buffer_pool_t iree_hal_amdgpu_buffer_pool_t;
typedef struct iree_hal_amdgpu_host_service_t iree_hal_amdgpu_host_service_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_queue_t
//===----------------------------------------------------------------------===//

// Calculates the size in bytes of the storage required for a queue
// implementation based on the provided |options|.
iree_host_size_t iree_hal_amdgpu_host_queue_calculate_size(
    const iree_hal_amdgpu_queue_options_t* options);

// Initializes |out_queue| in-place based on |options|.
iree_status_t iree_hal_amdgpu_host_queue_initialize(
    iree_hal_amdgpu_system_t* system, iree_hal_amdgpu_queue_options_t options,
    hsa_agent_t device_agent, iree_host_size_t device_ordinal,
    iree_hal_amdgpu_host_service_t* host_service,
    iree_arena_block_pool_t* host_block_pool,
    iree_hal_amdgpu_block_allocators_t* block_allocators,
    iree_hal_amdgpu_buffer_pool_t* buffer_pool,
    iree_hal_amdgpu_error_callback_t error_callback,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_virtual_queue_t* out_queue);

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_
