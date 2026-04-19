// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/command_buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Emits a PM4 timestamp packet that writes the start tick for a profiled
// command-buffer queue operation.
void iree_hal_amdgpu_host_queue_commit_command_buffer_profile_start(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event);

// Emits a PM4 timestamp packet that writes the end tick for a profiled
// command-buffer queue operation.
void iree_hal_amdgpu_host_queue_commit_command_buffer_profile_end(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal,
    iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event);

// Emits a single PM4 packet that writes both start and end ticks for an empty
// profiled command-buffer queue operation.
void iree_hal_amdgpu_host_queue_commit_command_buffer_profile_timestamp_range(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal,
    iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event);

// Counts dispatch commands in |block| selected by the active capture filter.
uint32_t
iree_hal_amdgpu_host_queue_count_command_buffer_profile_dispatch_events(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_amdgpu_command_buffer_block_header_t* block);

// Returns true when |dispatch_command| should receive exact dispatch timestamp
// profiling in this queue and profile session.
bool iree_hal_amdgpu_host_queue_should_profile_command_buffer_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command);

// Records the queue-owned dispatch event and harvest source paired with one
// command-buffer dispatch packet.
void iree_hal_amdgpu_host_queue_record_command_buffer_profile_dispatch_source(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events,
    iree_hal_amdgpu_profile_dispatch_harvest_source_t* profile_harvest_sources,
    bool profile_dispatch_packet, uint32_t* inout_profile_event_index);

// Prepares code-object side data needed by trace captures for the selected
// dispatch events in |block|.
iree_status_t
iree_hal_amdgpu_host_queue_prepare_command_buffer_profile_trace_code_objects(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_H_
