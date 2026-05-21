// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Block-ISA command buffer: compiles HAL commands into a block recording
// during recording. The recording is an immutable .text stream (block_isa.h
// types) consumed by the block processor at execution time.
//
// The queue manages execution: it allocates a processor context from the
// recording, installs it in a compute recording item, and drains it
// cooperatively across workers via the shared compute process.

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/block_isa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_block_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a block-ISA command buffer.
bool iree_hal_block_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns the recording from a completed (end() called) block command buffer.
// The returned pointer is valid until the command buffer is destroyed.
const iree_hal_cmd_block_recording_t* iree_hal_block_command_buffer_recording(
    iree_hal_command_buffer_t* command_buffer);

// Returns direct transient bindings that must be mapped after queue waits have
// resolved. The returned storage is owned by |command_buffer| and remains valid
// for its lifetime.
const iree_hal_buffer_binding_t* iree_hal_block_command_buffer_late_bindings(
    iree_hal_command_buffer_t* command_buffer,
    iree_host_size_t* out_late_binding_count);

// Returns the distinct executable list recorded for profiling metadata.
//
// Returned executable pointers are retained by the command buffer resource set
// and remain valid for the command buffer lifetime. The list is populated while
// recording so a profiling session can later replay an existing command buffer
// without scanning its encoded command stream.
iree_hal_executable_t* const* iree_hal_block_command_buffer_profile_executables(
    iree_hal_command_buffer_t* command_buffer,
    iree_host_size_t* out_executable_count);

// Returns the number of command-buffer operations recorded for profiling.
iree_host_size_t iree_hal_block_command_buffer_profile_operation_count(
    iree_hal_command_buffer_t* command_buffer);

// Returns immutable command-buffer metadata and operation records for
// profiling.
//
// The operation record storage is owned by |command_buffer| and remains valid
// for its lifetime. Records are populated while recording so a profiling
// session can later replay an existing command buffer without scanning its
// encoded command stream.
void iree_hal_block_command_buffer_profile_metadata(
    iree_hal_command_buffer_t* command_buffer, uint32_t physical_device_ordinal,
    iree_hal_profile_command_buffer_record_t* out_command_buffer,
    const iree_hal_profile_command_operation_record_t** out_operations,
    iree_host_size_t* out_operation_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_BUFFER_H_
