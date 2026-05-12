// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_REPLAY_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_REPLAY_H_

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Starts multi-block command-buffer replay. Caller must hold
// |queue->locks.submission_mutex|.
iree_status_t iree_hal_amdgpu_command_buffer_replay_start_under_lock(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t execute_flags,
    iree_hal_resource_set_t** inout_binding_resource_set);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_REPLAY_H_
