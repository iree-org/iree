// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Builder for the WebGPU block command buffer ISA.
//
// HAL command buffer methods delegate to the builder, which emits a compact
// uint32 instruction stream into fixed-size blocks from a shared block pool.
// The stream is later executed by a JS-side processor in a single wasm↔JS
// bridge call.
//
// The builder manages:
//   - A block list backed by an arena block pool (no realloc of data).
//   - A slot map for deduplicating static buffer references.
//   - Automatic ENCODER_BEGIN/END insertion around encoder command sequences.
//
// Slots [0, dynamic_count) are reserved for indirect binding table entries
// resolved per issue. Static slots [dynamic_count, total) are assigned during
// recording and resolved once at recording creation.

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_BUILDER_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_BUILDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/handle_table.h"
#include "iree/hal/drivers/webgpu/webgpu_isa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Slot map
//===----------------------------------------------------------------------===//

// Maps an iree_hal_buffer_t* to a static slot index for deduplication.
// The buffer pointer refers to the allocated buffer (via
// iree_hal_buffer_allocated_buffer), not the subspan.
typedef struct iree_hal_webgpu_builder_slot_entry_t {
  iree_hal_buffer_t* buffer;
  iree_hal_webgpu_handle_t gpu_buffer_handle;
  uint32_t slot;
} iree_hal_webgpu_builder_slot_entry_t;

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_builder_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_builder_t {
  // Block pool for acquiring fixed-size instruction stream blocks.
  iree_arena_block_pool_t* block_pool;

  // Block pointer table: block_ptrs[i] points to the first usable byte of
  // block i. The JS side reads this table via wasm memory to walk blocks
  // zero-copy. The table itself is a small growable array (initial capacity 16,
  // realloc of 64-128 bytes is fine — the instruction data never reallocs).
  uint32_t** block_ptrs;
  uint32_t block_count;
  uint32_t block_ptrs_capacity;

  // Write position within the current (last) block.
  uint32_t cursor;

  // Words per block: total_block_size / sizeof(uint32_t). We use the full block
  // including the arena footer area (we maintain our own block list and
  // reconstruct the footer chain only at release time).
  uint32_t block_word_capacity;

  // Encoder auto-tracking: true when an ENCODER_BEGIN has been emitted without
  // a matching ENCODER_END.
  bool in_encoder;

  // Dynamic slots [0, dynamic_count) are reserved for indirect binding table
  // entries. Static slots start at dynamic_count.
  uint32_t dynamic_count;

  // Slot map for static buffer deduplication.
  iree_hal_webgpu_builder_slot_entry_t* slot_entries;
  uint32_t static_slot_count;
  uint32_t slot_map_capacity;

  iree_allocator_t host_allocator;
} iree_hal_webgpu_builder_t;

// Initializes a builder that acquires instruction stream blocks from
// |block_pool|. |dynamic_count| dynamic binding table slots are reserved.
iree_status_t iree_hal_webgpu_builder_initialize(
    iree_arena_block_pool_t* block_pool, uint32_t dynamic_count,
    iree_allocator_t host_allocator, iree_hal_webgpu_builder_t* out_builder);

// Releases all blocks back to the pool and frees the block pointer table and
// slot map.
void iree_hal_webgpu_builder_deinitialize(iree_hal_webgpu_builder_t* builder);

// Resets the builder for reuse: releases all blocks back to the pool and
// acquires a fresh first block. Retains the block pointer table allocation,
// dynamic_count, and slot map capacity.
iree_status_t iree_hal_webgpu_builder_reset(iree_hal_webgpu_builder_t* builder);

// Closes any open encoder and emits a RETURN instruction.
iree_status_t iree_hal_webgpu_builder_finalize(
    iree_hal_webgpu_builder_t* builder);

// Returns the block pointer table. Each entry is a uint32_t* pointing to the
// first word of that block's instruction data.
static inline uint32_t* const* iree_hal_webgpu_builder_block_table(
    const iree_hal_webgpu_builder_t* builder) {
  return builder->block_ptrs;
}

// Returns the number of blocks in the instruction stream.
static inline uint32_t iree_hal_webgpu_builder_block_count(
    const iree_hal_webgpu_builder_t* builder) {
  return builder->block_count;
}

// Returns the word capacity of each block (uniform across all blocks).
static inline uint32_t iree_hal_webgpu_builder_block_word_capacity(
    const iree_hal_webgpu_builder_t* builder) {
  return builder->block_word_capacity;
}

// Returns the number of valid words in the last block.
static inline uint32_t iree_hal_webgpu_builder_last_block_word_count(
    const iree_hal_webgpu_builder_t* builder) {
  return builder->cursor;
}

// Returns dynamic_count + static_slot_count.
static inline uint32_t iree_hal_webgpu_builder_total_slot_count(
    const iree_hal_webgpu_builder_t* builder) {
  return builder->dynamic_count + builder->static_slot_count;
}

// Returns the static slot entries for populating the binding table at recording
// creation. Entries are in slot assignment order (slot indices are contiguous
// starting at dynamic_count).
static inline const iree_hal_webgpu_builder_slot_entry_t*
iree_hal_webgpu_builder_static_slot_entries(
    const iree_hal_webgpu_builder_t* builder) {
  return builder->slot_entries;
}

// Returns the number of static slot entries.
static inline uint32_t iree_hal_webgpu_builder_static_slot_count(
    const iree_hal_webgpu_builder_t* builder) {
  return builder->static_slot_count;
}

//===----------------------------------------------------------------------===//
// Command methods
//===----------------------------------------------------------------------===//

// Emits a FILL_BUFFER instruction.
// |target_ref| is the buffer to fill. |pattern| is the fill value (1, 2, or 4
// bytes). The pattern is replicated to uint32 at recording time.
iree_status_t iree_hal_webgpu_builder_fill_buffer(
    iree_hal_webgpu_builder_t* builder, iree_hal_buffer_ref_t target_ref,
    const void* pattern, iree_host_size_t pattern_length);

// Emits one or more UPDATE_BUFFER instructions with inline host data.
// |source_buffer| + |source_offset| points to the host data to copy.
// |target_ref| is the destination GPU buffer. Large updates may be split across
// multiple instructions to fit within block boundaries.
iree_status_t iree_hal_webgpu_builder_update_buffer(
    iree_hal_webgpu_builder_t* builder, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref);

// Emits a COPY_BUFFER instruction.
iree_status_t iree_hal_webgpu_builder_copy_buffer(
    iree_hal_webgpu_builder_t* builder, iree_hal_buffer_ref_t source_ref,
    iree_hal_buffer_ref_t target_ref);

// Emits a DISPATCH instruction.
// |pipeline_handle| and |bind_group_layout_handle| are baked from the
// executable at recording time. |config| provides the workgroup count.
// |bindings| lists all buffer bindings for the dispatch.
iree_status_t iree_hal_webgpu_builder_dispatch(
    iree_hal_webgpu_builder_t* builder,
    iree_hal_webgpu_handle_t pipeline_handle,
    iree_hal_webgpu_handle_t bind_group_layout_handle,
    const uint32_t workgroup_count[3], iree_hal_buffer_ref_list_t bindings);

// Emits a BARRIER instruction (no-op, reserved for future use).
iree_status_t iree_hal_webgpu_builder_execution_barrier(
    iree_hal_webgpu_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_BUILDER_H_
