// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_profile.h"

#include "iree/hal/drivers/amdgpu/logical_device.h"

uint32_t iree_hal_amdgpu_host_queue_profile_device_ordinal(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return queue->device_ordinal <= UINT32_MAX ? (uint32_t)queue->device_ordinal
                                             : UINT32_MAX;
}

uint32_t iree_hal_amdgpu_host_queue_profile_queue_ordinal(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return iree_async_axis_queue_index(queue->axis);
}

uint64_t iree_hal_amdgpu_host_queue_profile_stream_id(
    const iree_hal_amdgpu_host_queue_t* queue) {
  const uint32_t physical_device_ordinal =
      iree_hal_amdgpu_host_queue_profile_device_ordinal(queue);
  const uint32_t queue_ordinal =
      iree_hal_amdgpu_host_queue_profile_queue_ordinal(queue);
  return ((uint64_t)physical_device_ordinal << 32) | (uint64_t)queue_ordinal;
}

uint32_t iree_hal_amdgpu_host_queue_profile_semaphore_count(
    const iree_hal_semaphore_list_t semaphore_list) {
  return semaphore_list.count > UINT32_MAX ? UINT32_MAX
                                           : (uint32_t)semaphore_list.count;
}

iree_hal_amdgpu_profile_queue_device_event_t*
iree_hal_amdgpu_host_queue_initialize_profile_queue_device_event(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_queue_device_event_reservation_t reservation,
    const iree_hal_amdgpu_host_queue_profile_event_info_t* info) {
  if (reservation.event_count == 0) return NULL;
  IREE_ASSERT(info != NULL,
              "queue device event reservation requires profile event info");
  iree_hal_amdgpu_profile_queue_device_event_t* event =
      iree_hal_amdgpu_host_queue_profile_queue_device_event_at(
          queue, reservation.first_event_position);
  event->type = info->type;
  event->flags = info->flags;
  event->command_buffer_id = info->command_buffer_id;
  event->allocation_id = info->allocation_id;
  event->payload_length = info->payload_length;
  event->operation_count = info->operation_count;
  event->start_tick = 0;
  event->end_tick = 0;
  return event;
}

static iree_hal_profile_queue_dependency_strategy_t
iree_hal_amdgpu_host_queue_profile_dependency_strategy(
    const iree_hal_amdgpu_wait_resolution_t* resolution) {
  if (resolution->needs_deferral) {
    return IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER;
  }
  if (iree_any_bit_set(resolution->profile_event_flags,
                       IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED)) {
    return IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER;
  }
  if (resolution->wait_count == 0) {
    return IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE;
  }
  if (resolution->barrier_count != 0) {
    return IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER;
  }
  return IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE;
}

void iree_hal_amdgpu_host_queue_record_profile_queue_event(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_profile_event_info_t* info) {
  if (!queue->profiling.queue_events_enabled) return;

  iree_hal_profile_queue_event_t event = iree_hal_profile_queue_event_default();
  event.type = info->type;
  event.flags = info->flags | resolution->profile_event_flags;
  event.dependency_strategy =
      iree_hal_amdgpu_host_queue_profile_dependency_strategy(resolution);
  event.submission_id = info->submission_id;
  event.command_buffer_id = info->command_buffer_id;
  event.allocation_id = info->allocation_id;
  event.stream_id = iree_hal_amdgpu_host_queue_profile_stream_id(queue);
  event.physical_device_ordinal =
      iree_hal_amdgpu_host_queue_profile_device_ordinal(queue);
  event.queue_ordinal = iree_hal_amdgpu_host_queue_profile_queue_ordinal(queue);
  event.wait_count = resolution->wait_count;
  event.signal_count =
      iree_hal_amdgpu_host_queue_profile_semaphore_count(signal_semaphore_list);
  event.barrier_count = resolution->barrier_count;
  event.operation_count = info->operation_count;
  event.payload_length = info->payload_length;
  iree_hal_amdgpu_logical_device_record_profile_queue_event(
      queue->logical_device, &event);
}
