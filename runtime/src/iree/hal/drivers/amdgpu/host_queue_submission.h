// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_SUBMISSION_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_SUBMISSION_H_

#include "iree/hal/drivers/amdgpu/host_queue_waits.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef void(IREE_API_PTR* iree_hal_amdgpu_host_queue_post_commit_fn_t)(
    void* user_data, const iree_async_frontier_t* queue_frontier);

// Flags controlling submission helper ownership transfers.
typedef uint32_t iree_hal_amdgpu_host_queue_submission_flags_t;
enum iree_hal_amdgpu_host_queue_submission_flag_bits_t {
  IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE = 0u,
  // Retains signal semaphores and operation resources into the reclaim entry.
  // When omitted, the helper transfers one existing retain for each resource
  // from the caller into the reclaim entry on success.
  IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES = 1u << 0,
};

// One in-flight kernel-dispatch submission assembled under submission_mutex.
// All queue-private resource transfer and reclaim bookkeeping flows through
// this struct so fill/copy/update/dispatch do not accidentally grow divergent
// kernel-ring retirement mechanisms.
typedef struct iree_hal_amdgpu_host_queue_dispatch_submission_t {
  // Reclaim entry reserved from the notification ring for this submission.
  iree_hal_amdgpu_reclaim_entry_t* reclaim_entry;
  // Reclaim resource slots owned by |reclaim_entry|.
  iree_hal_resource_t** reclaim_resources;
  // Final uncommitted dispatch AQL slot.
  iree_hal_amdgpu_aql_packet_t* dispatch_slot;
  // Queue-owned kernarg blocks reserved for this submission.
  iree_hal_amdgpu_kernarg_block_t* kernarg_blocks;
  // First AQL packet id reserved for this submission.
  uint64_t first_packet_id;
  // Kernarg ring write position to reclaim after this submission completes.
  uint64_t kernarg_write_position;
  // Number of AQL packets reserved starting at |first_packet_id|.
  uint32_t packet_count;
  // Number of valid entries in |reclaim_resources|.
  uint16_t reclaim_resource_count;
  // Setup bits published with |dispatch_slot|'s final header.
  uint16_t dispatch_setup;
  // Minimum acquire fence scope required by operation-local data visibility.
  iree_hsa_fence_scope_t minimum_acquire_scope;
  // Minimum release fence scope required by operation-local data visibility.
  iree_hsa_fence_scope_t minimum_release_scope;
  // Optional action executed before user signals are published when this
  // submission completes.
  iree_hal_amdgpu_reclaim_action_t pre_signal_action;
} iree_hal_amdgpu_host_queue_dispatch_submission_t;

// Returns the number of retained resources required for a submission with
// |signal_semaphore_count| user-visible signal semaphores and
// |operation_resource_count| additional operation-owned resources.
iree_status_t iree_hal_amdgpu_host_queue_count_reclaim_resources(
    iree_host_size_t signal_semaphore_count,
    iree_host_size_t operation_resource_count,
    uint16_t* out_reclaim_resource_count);

// Begins one kernel-dispatch submission by reserving notification/reclaim
// state, AQL slots, and |kernarg_block_count| queue-owned kernarg blocks.
// Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_begin_dispatch_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_resource_count, uint32_t kernarg_block_count,
    iree_hal_amdgpu_host_queue_dispatch_submission_t* out_submission);

// Writes one final dispatch packet body into an AQL slot in forward field
// order and returns the setup bits that must be published with the header.
uint16_t iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
    iree_hsa_kernel_dispatch_packet_t* IREE_RESTRICT dispatch_packet,
    const iree_hsa_kernel_dispatch_packet_t* IREE_RESTRICT
        dispatch_packet_template,
    void* kernarg_address, iree_hsa_signal_t completion_signal);

// Finishes a submission by transferring retained resources to the reclaim
// entry, publishing queue/semaphore frontier state, committing the final
// dispatch header, and ringing the doorbell. Caller must hold submission_mutex.
void iree_hal_amdgpu_host_queue_finish_dispatch_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_dispatch_submission_t* submission);

// Emits one kernel-dispatch submission using an already-prepared packet shape
// and kernargs blob. Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_dispatch_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hsa_kernel_dispatch_packet_t* dispatch_packet_template,
    const void* kernargs, iree_host_size_t kernarg_length,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags);

// Submits a barrier-only operation using the notification/reclaim path.
// Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_barrier(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_post_commit_fn_t post_commit_fn,
    void* post_commit_user_data,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags);

// Replays an AMDGPU AQL command buffer program onto the host queue.
// Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_command_buffer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_SUBMISSION_H_
