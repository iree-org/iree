// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_policy.h"

#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/system.h"

// Returns true when |queue_affinity| names only queues on |queue|'s physical
// HSA agent. HSA AGENT scope is not a logical-device-wide guarantee: a
// multi-GPU logical device still needs SYSTEM scope for cross-physical-agent
// synchronization.
static bool iree_hal_amdgpu_host_queue_affinity_is_same_agent(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_queue_affinity_t queue_affinity) {
  const iree_hal_amdgpu_logical_device_t* logical_device =
      (const iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  const iree_hal_amdgpu_queue_affinity_domain_t domain = {
      .supported_affinity = logical_device->queue_affinity_mask,
      .physical_device_count = logical_device->physical_device_count,
      .queue_count_per_physical_device =
          logical_device->system->topology.gpu_agent_queue_count,
  };
  return iree_hal_amdgpu_queue_affinity_is_physical_device_local(
      domain, queue_affinity, queue->device_ordinal);
}

// Returns true when a semaphore edge can be represented with HSA AGENT scope
// on |queue|. Public/exportable/host-visible semaphores and cross-agent
// affinities use SYSTEM scope.
static bool iree_hal_amdgpu_host_queue_can_use_agent_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_semaphore_t* semaphore) {
  const iree_hal_amdgpu_logical_device_t* logical_device =
      (const iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  if (!iree_hal_amdgpu_semaphore_is_local(semaphore, logical_device)) {
    return false;
  }

  const iree_hal_semaphore_flags_t flags =
      iree_hal_amdgpu_semaphore_flags(semaphore);
  if (!iree_all_bits_set(flags, IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL)) {
    return false;
  }
  const iree_hal_semaphore_flags_t public_flags =
      IREE_HAL_SEMAPHORE_FLAG_HOST_INTERRUPT |
      IREE_HAL_SEMAPHORE_FLAG_EXPORTABLE |
      IREE_HAL_SEMAPHORE_FLAG_EXPORTABLE_TIMEPOINTS;
  if (iree_any_bit_set(flags, public_flags)) return false;

  return iree_hal_amdgpu_host_queue_affinity_is_same_agent(
      queue, iree_hal_amdgpu_semaphore_queue_affinity(semaphore));
}

iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_wait_acquire_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_semaphore_t* semaphore) {
  return iree_hal_amdgpu_host_queue_can_use_agent_scope(queue, semaphore)
             ? IREE_HSA_FENCE_SCOPE_AGENT
             : IREE_HSA_FENCE_SCOPE_SYSTEM;
}

iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_axis_acquire_scope(
    const iree_hal_amdgpu_host_queue_t* queue, iree_async_axis_t axis) {
  return iree_async_axis_device_index(axis) == queue->device_ordinal
             ? IREE_HSA_FENCE_SCOPE_AGENT
             : IREE_HSA_FENCE_SCOPE_SYSTEM;
}

iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_signal_release_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_semaphore_t* semaphore) {
  return iree_hal_amdgpu_host_queue_can_use_agent_scope(queue, semaphore)
             ? IREE_HSA_FENCE_SCOPE_AGENT
             : IREE_HSA_FENCE_SCOPE_SYSTEM;
}

iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_signal_list_release_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_semaphore_list_t semaphores) {
  iree_hsa_fence_scope_t release_scope = IREE_HSA_FENCE_SCOPE_AGENT;
  for (iree_host_size_t i = 0; i < semaphores.count; ++i) {
    release_scope = iree_hal_amdgpu_host_queue_max_fence_scope(
        release_scope, iree_hal_amdgpu_host_queue_signal_release_scope(
                           queue, semaphores.semaphores[i]));
  }
  return release_scope;
}
