// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PROFILE_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PROFILE_H_

#include "iree/hal/drivers/amdgpu/host_queue_waits.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef uint32_t iree_hal_amdgpu_host_queue_profile_flags_t;
enum iree_hal_amdgpu_host_queue_profile_flag_bits_t {
  IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_NONE = 0u,
  // Host-timestamped queue operation events should be recorded.
  IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_QUEUE_EVENTS = 1u << 0,
  // Device-timestamped queue operation events should be recorded.
  IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_QUEUE_DEVICE_EVENTS = 1u << 1,
  // Per-dispatch profiling augmentation may be applied to selected dispatches.
  IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_FLAG_DISPATCHES = 1u << 2,
};

// Additional details for one queue operation profile event.
typedef struct iree_hal_amdgpu_host_queue_profile_event_info_t {
  // Type of queue operation represented by the event.
  iree_hal_profile_queue_event_type_t type;
  // Flags describing queue operation properties.
  iree_hal_profile_queue_event_flags_t flags;
  // Queue submission epoch assigned by the operation.
  uint64_t submission_id;
  // Session-local command-buffer identifier, or 0 when not applicable.
  uint64_t command_buffer_id;
  // Producer-defined allocation identifier, or 0 when not applicable.
  uint64_t allocation_id;
  // Type-specific payload byte length, or 0 when not applicable.
  uint64_t payload_length;
  // Number of encoded payload operations represented by this event.
  uint32_t operation_count;
} iree_hal_amdgpu_host_queue_profile_event_info_t;

// Returns the session-local profiling ordinal for |queue|'s physical device.
uint32_t iree_hal_amdgpu_host_queue_profile_device_ordinal(
    const iree_hal_amdgpu_host_queue_t* queue);

// Returns the session-local profiling ordinal for |queue| within its device.
uint32_t iree_hal_amdgpu_host_queue_profile_queue_ordinal(
    const iree_hal_amdgpu_host_queue_t* queue);

// Returns the stream id used by queue metadata and queue/dispatch events.
uint64_t iree_hal_amdgpu_host_queue_profile_stream_id(
    const iree_hal_amdgpu_host_queue_t* queue);

// Returns |semaphore_list.count| saturated to the queue-event field width.
uint32_t iree_hal_amdgpu_host_queue_profile_semaphore_count(
    const iree_hal_semaphore_list_t semaphore_list);

// Sets queue-local profile recording flags for an active session.
void iree_hal_amdgpu_host_queue_set_profile_flags(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_profile_flags_t flags);

// Initializes one reserved device-timestamped queue operation event.
iree_hal_amdgpu_profile_queue_device_event_t*
iree_hal_amdgpu_host_queue_initialize_profile_queue_device_event(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_queue_device_event_reservation_t reservation,
    const iree_hal_amdgpu_host_queue_profile_event_info_t* info);

// Records one queue operation event when queue profiling is enabled.
//
// This performs a cheap queue-local enabled check before preparing the full
// event. The sink is never called here; the logical device buffers events for
// profiling_flush/end.
void iree_hal_amdgpu_host_queue_record_profile_queue_event(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_profile_event_info_t* info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PROFILE_H_
