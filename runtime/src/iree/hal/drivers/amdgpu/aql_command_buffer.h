// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_AQL_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_AQL_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/aql_program_builder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a host-recorded AQL command buffer backed by |block_pool|.
//
// The command buffer borrows |block_pool| and the pool must outlive all command
// buffers created from it.
iree_status_t iree_hal_amdgpu_aql_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is an AMDGPU AQL command buffer.
bool iree_hal_amdgpu_aql_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns the immutable program produced by end().
const iree_hal_amdgpu_aql_program_t* iree_hal_amdgpu_aql_command_buffer_program(
    iree_hal_command_buffer_t* command_buffer);

// Returns a direct buffer recorded in the command-buffer static binding table.
iree_hal_buffer_t* iree_hal_amdgpu_aql_command_buffer_static_buffer(
    iree_hal_command_buffer_t* command_buffer, uint32_t ordinal);

// Returns command-buffer-owned rodata referenced by |command_buffer|.
const uint8_t* iree_hal_amdgpu_aql_command_buffer_rodata(
    iree_hal_command_buffer_t* command_buffer, uint64_t ordinal,
    uint32_t length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_AQL_COMMAND_BUFFER_H_
