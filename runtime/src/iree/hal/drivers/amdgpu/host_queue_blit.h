// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_BLIT_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_BLIT_H_

#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validates a queue_update request and resolves the source host span and target
// device pointer. The source host pointer is captured by the caller either into
// the pending-op arena or into the queue-owned kernarg ring.
iree_status_t iree_hal_amdgpu_host_queue_prepare_update_copy(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    const uint8_t** out_source_bytes, iree_host_size_t* out_source_length,
    uint8_t** out_target_device_ptr);

// Emits a fill blit kernel submission. Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

// Emits a copy blit kernel submission. Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_profile_queue_event_type_t profile_event_type,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

// Emits a copy blit kernel submission with additional completion behavior.
// Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_copy_with_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hsa_fence_scope_t minimum_acquire_scope,
    iree_hsa_fence_scope_t minimum_release_scope,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* extra_operation_resources,
    iree_host_size_t extra_operation_resource_count,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

// Emits an update blit kernel submission. Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_update(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_BLIT_H_
