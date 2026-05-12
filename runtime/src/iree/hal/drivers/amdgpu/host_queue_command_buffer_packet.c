// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_packet.h"

#include "iree/hal/drivers/amdgpu/host_queue_policy.h"

static bool iree_hal_amdgpu_host_queue_command_buffer_packet_has_barrier(
    const iree_hal_amdgpu_wait_resolution_t* resolution, uint32_t packet_index,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags) {
  return iree_any_bit_set(
             packet_flags,
             IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_EXECUTION_BARRIER) ||
         iree_any_bit_set(
             packet_flags,
             IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL) ||
         (packet_index == 0 && resolution->barrier_count > 0) ||
         (packet_index == 0 &&
          resolution->inline_acquire_scope != IREE_HSA_FENCE_SCOPE_NONE);
}

iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_host_queue_command_buffer_packet_control(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    uint32_t packet_index, iree_hsa_fence_scope_t minimum_acquire_scope,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags) {
  const bool has_barrier =
      iree_hal_amdgpu_host_queue_command_buffer_packet_has_barrier(
          resolution, packet_index, packet_flags);
  const iree_hsa_fence_scope_t execution_acquire_scope =
      iree_hal_amdgpu_host_queue_command_buffer_packet_flags_acquire_scope(
          packet_flags);
  const iree_hsa_fence_scope_t execution_release_scope =
      iree_hal_amdgpu_host_queue_command_buffer_packet_flags_release_scope(
          packet_flags);
  const iree_hsa_fence_scope_t acquire_scope =
      packet_index == 0
          ? iree_hal_amdgpu_host_queue_max_fence_scope(
                execution_acquire_scope, resolution->inline_acquire_scope)
          : execution_acquire_scope;
  const iree_hsa_fence_scope_t effective_acquire_scope =
      iree_hal_amdgpu_host_queue_max_fence_scope(acquire_scope,
                                                 minimum_acquire_scope);
  iree_hsa_fence_scope_t release_scope = execution_release_scope;
  if (iree_any_bit_set(
          packet_flags,
          IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL)) {
    release_scope = iree_hal_amdgpu_host_queue_max_fence_scope(
        release_scope, iree_hal_amdgpu_host_queue_signal_list_release_scope(
                           queue, signal_semaphore_list));
  }
  return iree_hal_amdgpu_aql_packet_control(
      has_barrier, effective_acquire_scope, release_scope);
}
