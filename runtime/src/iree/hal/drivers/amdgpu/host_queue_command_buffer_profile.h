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
#include "iree/hal/drivers/amdgpu/aql_block_processor_profile.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Selects dispatch commands in |block| matched by the active capture filter.
iree_status_t
iree_hal_amdgpu_host_queue_select_command_buffer_profile_dispatches(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_arena_allocator_t* scratch_arena,
    iree_hal_amdgpu_aql_block_processor_profile_dispatch_list_t*
        out_dispatches);

// Records the queue-owned dispatch event and harvest source paired with one
// command-buffer dispatch packet.
void iree_hal_amdgpu_host_queue_record_command_buffer_profile_dispatch_source(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_amdgpu_aql_block_processor_profile_dispatch_t* dispatch,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events,
    iree_hal_amdgpu_profile_dispatch_harvest_source_t* profile_harvest_sources,
    uint32_t* inout_profile_event_index);

// Prepares code-object side data needed by trace captures for the selected
// dispatch events in |block|.
iree_status_t
iree_hal_amdgpu_host_queue_prepare_command_buffer_profile_trace_code_objects(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_aql_block_processor_profile_dispatch_list_t dispatches,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_H_
