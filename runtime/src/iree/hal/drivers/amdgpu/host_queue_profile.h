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

// Additional details for one queue operation profile event.
typedef struct iree_hal_amdgpu_host_queue_profile_event_info_t {
  // Type of queue operation represented by the event.
  iree_hal_profile_queue_event_type_t type;
  // Flags describing queue operation properties.
  iree_hal_profile_queue_event_flags_t flags;
  // Queue submission epoch assigned by the operation.
  uint64_t submission_id;
  // Process-local command-buffer identifier, or 0 when not applicable.
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
