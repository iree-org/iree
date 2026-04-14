// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_H_

#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validates queue_execute flags supported by the AMDGPU host queue.
iree_status_t iree_hal_amdgpu_host_queue_validate_execute_flags(
    iree_hal_execute_flags_t flags);

// Creates a resource set retaining the binding table prefix required by
// |command_buffer| unless |execute_flags| explicitly borrows buffer lifetimes.
iree_status_t iree_hal_amdgpu_host_queue_create_binding_table_resource_set(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t execute_flags,
    iree_hal_resource_set_t** out_resource_set);

// Replays an AMDGPU AQL command buffer program onto the host queue.
// Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_command_buffer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t execute_flags,
    iree_hal_resource_set_t** inout_binding_resource_set,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_H_
