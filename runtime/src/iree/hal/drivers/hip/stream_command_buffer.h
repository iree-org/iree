// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_STREAM_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_HIP_STREAM_COMMAND_BUFFER_H_

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/hip_headers.h"
#include "iree/hal/drivers/hip/rccl_dynamic_symbols.h"
#include "iree/hal/utils/stream_tracing.h"

// Creates command buffer that immediately issues commands against the given
// HIP |stream|. Access to |stream| must be synchronized by the user.
//
// If |block_pool| is non-NULL then the stream command buffer will retain copies
// of input data until reset. If NULL then the caller must ensure the lifetime
// of input data outlives the command buffer.
//
// This command buffer is used to replay deferred command buffers. When
// replaying the scratch data required for things like buffer updates is
// retained by the source deferred command buffer and as such the |block_pool|
// and can be NULL to avoid a double copy.
iree_status_t iree_hal_hip_stream_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_hip_dynamic_symbols_t* hip_symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    iree_hal_stream_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    hipStream_t stream, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a HIP stream-based command buffer.
bool iree_hal_hip_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// This is to be called after a command buffer has been submitted
// in order to notify the tracing system that there are events
// to collect.
void iree_hal_hip_stream_notify_submitted_commands(
    iree_hal_command_buffer_t* base_command_buffer);

// Returns the set of tracing events that are associated with
// this command buffer.
iree_hal_stream_tracing_context_event_list_t
iree_hal_hip_stream_command_buffer_tracing_events(
    iree_hal_command_buffer_t* base_command_buffer);

#endif  // IREE_HAL_DRIVERS_HIP_STREAM_COMMAND_BUFFER_H_
