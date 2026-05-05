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
#include "iree/hal/drivers/amdgpu/aql_prepublished_kernarg_storage.h"
#include "iree/hal/drivers/amdgpu/aql_program_builder.h"
#include "iree/hal/drivers/amdgpu/profile_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a host-recorded AQL command buffer.
//
// The command buffer borrows |program_block_pool| for durable AQL program
// storage and recording scratch memory. It borrows |resource_set_block_pool|
// for retained HAL resource sets. Both pools must outlive all command buffers
// created from them.
iree_status_t iree_hal_amdgpu_aql_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_host_size_t device_ordinal,
    iree_hal_amdgpu_aql_prepublished_kernarg_storage_t
        prepublished_kernarg_storage,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_arena_block_pool_t* program_block_pool,
    iree_arena_block_pool_t* resource_set_block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is an AMDGPU AQL command buffer.
bool iree_hal_amdgpu_aql_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Retained command-buffer metadata for one dispatch operation.
typedef struct iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t {
  // Next retained dispatch summary in the same command-buffer block.
  const struct iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* next;
  // Payload packet ordinals produced by command-buffer replay.
  struct {
    // First payload packet ordinal emitted for this command.
    uint32_t first_ordinal;
    // Payload packet ordinal of the dispatch whose completion signal represents
    // this command.
    uint32_t dispatch_ordinal;
  } packets;
  // Correlation metadata used by profiling, timestamps, and diagnostics.
  struct {
    // Session-local profile executable id used for event attribution.
    uint64_t executable_id;
    // Program-global command index used for profiling/source attribution.
    uint32_t command_index;
    // Executable export ordinal used for profiling and diagnostics.
    uint32_t export_ordinal;
    // Dispatch flags from iree_hal_amdgpu_command_buffer_dispatch_flag_bits_t.
    uint8_t dispatch_flags;
    // Reserved bytes that must be zero.
    uint8_t reserved0[3];
  } metadata;
  // Dispatch launch dimensions used for event metadata.
  struct {
    // Static workgroup counts. Zero for indirect dispatches whose counts are
    // read at device execution time.
    uint32_t count[3];
    // AQL dispatch packet workgroup size fields.
    uint16_t size[3];
  } workgroup;
} iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t;

// Returns the immutable program produced by end().
const iree_hal_amdgpu_aql_program_t* iree_hal_amdgpu_aql_command_buffer_program(
    iree_hal_command_buffer_t* command_buffer);

// Returns the physical device ordinal this command buffer was recorded for.
iree_host_size_t iree_hal_amdgpu_aql_command_buffer_device_ordinal(
    iree_hal_command_buffer_t* command_buffer);

// Returns the producer-local profile command-buffer id, or 0 when recording
// did not retain command-buffer profile metadata.
uint64_t iree_hal_amdgpu_aql_command_buffer_profile_id(
    iree_hal_command_buffer_t* command_buffer);

// Returns retained dispatch summaries for |block|, or NULL when no
// summaries were retained. |out_count| receives the number of linked summaries.
const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t*
iree_hal_amdgpu_aql_command_buffer_dispatch_summaries(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint32_t* out_count);

// Returns a direct buffer recorded in the command-buffer static binding table.
iree_hal_buffer_t* iree_hal_amdgpu_aql_command_buffer_static_buffer(
    iree_hal_command_buffer_t* command_buffer, uint32_t ordinal);

// Returns command-buffer-owned rodata referenced by |command_buffer|.
const uint8_t* iree_hal_amdgpu_aql_command_buffer_rodata(
    iree_hal_command_buffer_t* command_buffer, uint64_t ordinal,
    uint32_t length);

// Returns command-buffer-owned device-visible prepublished kernargs at
// |byte_offset| within the finalized prepublished kernarg storage.
void* iree_hal_amdgpu_aql_command_buffer_prepublished_kernarg(
    iree_hal_command_buffer_t* command_buffer, uint32_t byte_offset,
    uint32_t length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_AQL_COMMAND_BUFFER_H_
