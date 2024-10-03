// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/device/command_buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_command_buffer_t
//===----------------------------------------------------------------------===//

// Creates AMDGPU command buffer.
iree_status_t iree_hal_amdgpu_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a AMDGPU command buffer.
bool iree_hal_amdgpu_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Queries the device-side command buffer representation for the GPU device
// agent with |device_ordinal| in the system topology.
// |out_max_kernarg_capacity| will be set to the minimum required kernarg
// reservation used by any block in the command buffer.
iree_status_t iree_hal_amdgpu_command_buffer_query_execution_state(
    iree_hal_command_buffer_t* command_buffer, iree_host_size_t device_ordinal,
    iree_hal_amdgpu_device_command_buffer_t** out_device_command_buffer,
    iree_host_size_t* out_max_kernarg_capacity);

#endif  // IREE_HAL_DRIVERS_AMDGPU_COMMAND_BUFFER_H_
