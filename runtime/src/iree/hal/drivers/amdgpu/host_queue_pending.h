// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PENDING_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PENDING_H_

#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Starts wait registration or capacity retry for a captured operation.
iree_status_t iree_hal_amdgpu_pending_op_start(iree_hal_amdgpu_pending_op_t* op,
                                               bool wait_for_capacity);

// Enqueues the cold alloca memory-readiness wait prepared on |op|.
void iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op);

// Cancels all queue pending operations during shutdown or fatal queue failure.
void iree_hal_amdgpu_host_queue_cancel_pending(
    iree_hal_amdgpu_host_queue_t* queue, iree_status_code_t status_code,
    const char* status_message);

// Captures a queue_alloca operation for later issue. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_defer_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_pending_op_t** out_op);

// Submits an alloca operation after wait resolution. Caller must hold
// queue->locks.submission_mutex. If memory readiness must wait,
// |out_memory_wait_op| receives the operation that owns the prepared wait
// sidecar.
iree_status_t iree_hal_amdgpu_host_queue_submit_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op, bool* out_ready);

// Captures a queue_dealloca operation for later issue. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_defer_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_amdgpu_pending_op_t** out_op);

// Captures a queue_fill operation for later issue. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_defer_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op);

// Captures a queue_copy/read/write operation for later issue. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_defer_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_profile_queue_event_type_t profile_event_type,
    iree_hal_amdgpu_pending_op_t** out_op);

// Captures a queue_update operation for later issue. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_defer_update(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op);

// Captures a queue_execute operation for later issue. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_defer_execute(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags, iree_hal_amdgpu_pending_op_t** out_op);

// Captures a queue_dispatch operation for later issue. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_defer_dispatch(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op);

// Captures a driver host action for later issue. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_defer_host_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_pending_op_t** out_op);

// Captures a queue_host_call operation for later issue. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_defer_host_call(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags, iree_hal_amdgpu_pending_op_t** out_op);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PENDING_H_
