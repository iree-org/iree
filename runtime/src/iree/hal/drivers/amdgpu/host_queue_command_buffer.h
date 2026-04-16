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

#if !defined(NDEBUG)
typedef struct iree_hal_amdgpu_host_queue_command_buffer_packet_summary_t {
  // Total payload packets represented by the summarized block.
  uint32_t packet_count;
  // Payload packets with the AQL BARRIER bit set.
  uint32_t barrier_packet_count;
  // Payload packets with SYSTEM acquire scope.
  uint32_t system_acquire_packet_count;
  // Payload packets with SYSTEM release scope.
  uint32_t system_release_packet_count;
  // Header word for the first payload packet, or 0 for empty blocks.
  uint16_t first_packet_header;
  // Header word for the last payload packet, or 0 for empty blocks.
  uint16_t last_packet_header;
} iree_hal_amdgpu_host_queue_command_buffer_packet_summary_t;

// Summarizes the AQL payload packet headers a command-buffer block would emit.
// This is a debug/test helper and does not replay packet bodies or touch rings.
iree_status_t iree_hal_amdgpu_host_queue_summarize_command_buffer_block_packets(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_amdgpu_host_queue_command_buffer_packet_summary_t* out_summary);
#endif  // !defined(NDEBUG)

// Replays an AMDGPU AQL command buffer program onto the host queue.
// Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_command_buffer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t execute_flags,
    iree_hal_resource_set_t** inout_binding_resource_set, bool* out_ready);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_H_
