// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_BLOCK_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_BLOCK_H_

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Submits one finalized command-buffer block to the queue. Caller must hold
// |queue->locks.submission_mutex|.
iree_status_t iree_hal_amdgpu_host_queue_submit_command_buffer_block(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table, const uint64_t* binding_ptrs,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_resource_set_t** inout_binding_resource_set,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

// Resolves queue_execute binding table entries into raw device base pointers
// indexed by their original binding table slot.
iree_status_t iree_hal_amdgpu_host_queue_resolve_command_buffer_binding_ptrs(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table, uint64_t* out_binding_ptrs);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_BLOCK_H_
