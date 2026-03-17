// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_builder.h"

#include <string.h>

#include "iree/hal/drivers/webgpu/webgpu_buffer.h"

// Default initial capacity for the block pointer table. 16 entries = 64 bytes
// on wasm32 / 128 bytes on wasm64. Realloc of this tiny array is fine.
#define IREE_HAL_WEBGPU_BUILDER_DEFAULT_BLOCK_PTRS_CAPACITY 16

// Default initial capacity for the slot map.
#define IREE_HAL_WEBGPU_BUILDER_DEFAULT_SLOT_MAP_CAPACITY 16

//===----------------------------------------------------------------------===//
// Block management
//===----------------------------------------------------------------------===//

// Releases all blocks back to the pool. Does not free the block_ptrs array.
static void iree_hal_webgpu_builder_release_blocks(
    iree_hal_webgpu_builder_t* builder) {
  if (builder->block_count == 0) return;

  // Reconstruct the arena footer chain from our block pointer table.
  // The footer is at the end of each block's full allocation. Since we use the
  // full block (including footer area) for instruction data, the footer bytes
  // may contain instruction data — that's fine, we overwrite them here and the
  // pool's release overwrites them again.
  iree_arena_block_t* head =
      iree_arena_block_trailer(builder->block_pool, builder->block_ptrs[0]);
  iree_arena_block_t* prev = head;
  for (uint32_t i = 1; i < builder->block_count; ++i) {
    iree_arena_block_t* current =
        iree_arena_block_trailer(builder->block_pool, builder->block_ptrs[i]);
    prev->next = current;
    prev = current;
  }
  prev->next = NULL;

  iree_arena_block_pool_release(builder->block_pool, head, prev);
  builder->block_count = 0;
  builder->cursor = 0;
}

// Acquires a new block from the pool and appends it to the block table.
static iree_status_t iree_hal_webgpu_builder_acquire_block(
    iree_hal_webgpu_builder_t* builder) {
  iree_arena_block_t* block = NULL;
  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_block_pool_acquire(builder->block_pool, &block, &ptr));

  // Grow the block pointer table if needed.
  if (builder->block_count >= builder->block_ptrs_capacity) {
    IREE_TRACE_ZONE_BEGIN(z0);
    iree_host_size_t capacity = builder->block_ptrs_capacity;
    iree_status_t status = iree_allocator_grow_array(
        builder->host_allocator, builder->block_count + 1, sizeof(uint32_t*),
        &capacity, (void**)&builder->block_ptrs);
    if (!iree_status_is_ok(status)) {
      // Return the block we just acquired.
      block->next = NULL;
      iree_arena_block_pool_release(builder->block_pool, block, block);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    builder->block_ptrs_capacity = (uint32_t)capacity;
    IREE_TRACE_ZONE_END(z0);
  }

  builder->block_ptrs[builder->block_count++] = (uint32_t*)ptr;
  builder->cursor = 0;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Word reservation
//===----------------------------------------------------------------------===//

// Reserves |word_count| contiguous words in the current block, returning a
// pointer to the first word. The cursor is advanced by |word_count|. If the
// current block can't fit them, acquires a new block (wasting any remaining
// words in the current block).
static iree_status_t iree_hal_webgpu_builder_reserve(
    iree_hal_webgpu_builder_t* builder, uint32_t word_count,
    uint32_t** out_words) {
  if (IREE_UNLIKELY(builder->cursor + word_count >
                    builder->block_word_capacity)) {
    IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_acquire_block(builder));
  }
  *out_words = &builder->block_ptrs[builder->block_count - 1][builder->cursor];
  builder->cursor += word_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Slot map management
//===----------------------------------------------------------------------===//

// Ensures the slot map can hold at least one more entry.
static iree_status_t iree_hal_webgpu_builder_ensure_slot_capacity(
    iree_hal_webgpu_builder_t* builder) {
  if (IREE_LIKELY(builder->static_slot_count < builder->slot_map_capacity)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t capacity = builder->slot_map_capacity;
  iree_status_t status = iree_allocator_grow_array(
      builder->host_allocator, builder->static_slot_count + 1,
      sizeof(iree_hal_webgpu_builder_slot_entry_t), &capacity,
      (void**)&builder->slot_entries);
  if (iree_status_is_ok(status)) {
    builder->slot_map_capacity = (uint32_t)capacity;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Resolves a buffer_ref to a (slot, instruction_offset) pair.
//
// For indirect references (buffer == NULL): slot = buffer_slot (within the
// dynamic range), instruction_offset = caller's offset only.
//
// For direct references (buffer != NULL): looks up the allocated buffer in the
// slot map. If found, reuses the existing static slot. If new, assigns the next
// static slot. instruction_offset = subspan_offset + caller_offset.
static iree_status_t iree_hal_webgpu_builder_resolve_ref(
    iree_hal_webgpu_builder_t* builder, iree_hal_buffer_ref_t ref,
    uint32_t* out_slot, uint32_t* out_offset) {
  if (!ref.buffer) {
    // Indirect reference: slot is the binding table ordinal.
    if (ref.buffer_slot >= builder->dynamic_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "binding table slot %u exceeds dynamic slot count %u",
          ref.buffer_slot, builder->dynamic_count);
    }
    *out_slot = ref.buffer_slot;
    *out_offset = (uint32_t)ref.offset;
    return iree_ok_status();
  }

  // Direct reference: resolve to allocated buffer.
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(ref.buffer);
  iree_hal_webgpu_handle_t gpu_buffer_handle =
      iree_hal_webgpu_buffer_handle(allocated_buffer);
  uint32_t instruction_offset =
      (uint32_t)(iree_hal_buffer_byte_offset(ref.buffer) + ref.offset);

  // Search the slot map for an existing entry.
  for (uint32_t i = 0; i < builder->static_slot_count; ++i) {
    if (builder->slot_entries[i].buffer == allocated_buffer) {
      *out_slot = builder->slot_entries[i].slot;
      *out_offset = instruction_offset;
      return iree_ok_status();
    }
  }

  // New buffer: assign the next static slot.
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_ensure_slot_capacity(builder));
  uint32_t slot = builder->dynamic_count + builder->static_slot_count;
  iree_hal_webgpu_builder_slot_entry_t* entry =
      &builder->slot_entries[builder->static_slot_count++];
  entry->buffer = allocated_buffer;
  entry->gpu_buffer_handle = gpu_buffer_handle;
  entry->slot = slot;

  *out_slot = slot;
  *out_offset = instruction_offset;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Encoder auto-tracking
//===----------------------------------------------------------------------===//

// Emits ENCODER_BEGIN if not already in an encoder context.
static iree_status_t iree_hal_webgpu_builder_ensure_encoder_open(
    iree_hal_webgpu_builder_t* builder) {
  if (builder->in_encoder) return iree_ok_status();
  uint32_t* words;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_reserve(builder, 1, &words));
  words[0] = iree_hal_webgpu_isa_header_encode(
      IREE_HAL_WEBGPU_ISA_OP_ENCODER_BEGIN, /*flags=*/0, /*size_words=*/1);
  builder->in_encoder = true;
  return iree_ok_status();
}

// Emits ENCODER_END if currently in an encoder context.
static iree_status_t iree_hal_webgpu_builder_ensure_encoder_closed(
    iree_hal_webgpu_builder_t* builder) {
  if (!builder->in_encoder) return iree_ok_status();
  uint32_t* words;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_reserve(builder, 1, &words));
  words[0] = iree_hal_webgpu_isa_header_encode(
      IREE_HAL_WEBGPU_ISA_OP_ENCODER_END, /*flags=*/0, /*size_words=*/1);
  builder->in_encoder = false;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_builder_initialize(
    iree_arena_block_pool_t* block_pool, uint32_t dynamic_count,
    iree_allocator_t host_allocator, iree_hal_webgpu_builder_t* out_builder) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_builder);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->block_pool = block_pool;
  out_builder->host_allocator = host_allocator;
  out_builder->dynamic_count = dynamic_count;
  out_builder->block_word_capacity =
      (uint32_t)(block_pool->total_block_size / sizeof(uint32_t));

  // Allocate the block pointer table.
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, IREE_HAL_WEBGPU_BUILDER_DEFAULT_BLOCK_PTRS_CAPACITY,
      sizeof(uint32_t*), (void**)&out_builder->block_ptrs);
  if (iree_status_is_ok(status)) {
    out_builder->block_ptrs_capacity =
        IREE_HAL_WEBGPU_BUILDER_DEFAULT_BLOCK_PTRS_CAPACITY;
  }

  // Allocate the slot map.
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, IREE_HAL_WEBGPU_BUILDER_DEFAULT_SLOT_MAP_CAPACITY,
        sizeof(iree_hal_webgpu_builder_slot_entry_t),
        (void**)&out_builder->slot_entries);
  }
  if (iree_status_is_ok(status)) {
    out_builder->slot_map_capacity =
        IREE_HAL_WEBGPU_BUILDER_DEFAULT_SLOT_MAP_CAPACITY;
  }

  // Acquire the first block.
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_builder_acquire_block(out_builder);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_webgpu_builder_deinitialize(out_builder);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_webgpu_builder_deinitialize(iree_hal_webgpu_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_webgpu_builder_release_blocks(builder);
  iree_allocator_free(builder->host_allocator, builder->block_ptrs);
  iree_allocator_free(builder->host_allocator, builder->slot_entries);
  memset(builder, 0, sizeof(*builder));
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_webgpu_builder_reset(
    iree_hal_webgpu_builder_t* builder) {
  iree_hal_webgpu_builder_release_blocks(builder);
  builder->static_slot_count = 0;
  builder->in_encoder = false;
  // Acquire a fresh first block.
  return iree_hal_webgpu_builder_acquire_block(builder);
}

iree_status_t iree_hal_webgpu_builder_finalize(
    iree_hal_webgpu_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_ensure_encoder_closed(builder));
  uint32_t* words;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_reserve(builder, 1, &words));
  words[0] = iree_hal_webgpu_isa_header_encode(IREE_HAL_WEBGPU_ISA_OP_RETURN,
                                               /*flags=*/0, /*size_words=*/1);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Command methods
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_builder_fill_buffer(
    iree_hal_webgpu_builder_t* builder, iree_hal_buffer_ref_t target_ref,
    const void* pattern, iree_host_size_t pattern_length) {
  uint32_t dst_slot = 0;
  uint32_t dst_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_resolve_ref(
      builder, target_ref, &dst_slot, &dst_offset));

  // Replicate the 1/2/4-byte pattern to fill a uint32.
  uint32_t pattern_u32 = 0;
  memcpy(&pattern_u32, pattern, pattern_length);
  if (pattern_length == 1) {
    pattern_u32 *= 0x01010101u;
  } else if (pattern_length == 2) {
    pattern_u32 |= pattern_u32 << 16;
  }

  // FILL_BUFFER is an encoder command.
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_ensure_encoder_open(builder));

  uint32_t* words;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_reserve(builder, 6, &words));
  words[0] = iree_hal_webgpu_isa_header_encode(
      IREE_HAL_WEBGPU_ISA_OP_FILL_BUFFER, /*flags=*/0, /*size_words=*/6);
  words[1] = dst_slot;
  words[2] = dst_offset;
  words[3] = (uint32_t)target_ref.length;
  words[4] = pattern_u32;
  words[5] = (uint32_t)pattern_length;

  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_builder_update_buffer(
    iree_hal_webgpu_builder_t* builder, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref) {
  uint32_t dst_slot = 0;
  uint32_t dst_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_resolve_ref(
      builder, target_ref, &dst_slot, &dst_offset));

  // UPDATE_BUFFER is a queue command — close any open encoder first.
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_ensure_encoder_closed(builder));

  // Max inline data per instruction: block capacity minus the 4-word header.
  const uint32_t max_data_words = builder->block_word_capacity - 4;
  const uint32_t max_data_bytes = max_data_words * 4;

  const uint8_t* source_data = (const uint8_t*)source_buffer + source_offset;
  iree_host_size_t remaining = target_ref.length;
  uint32_t chunk_dst_offset = dst_offset;

  // Split large updates across multiple instructions to fit within blocks.
  // With 64KB blocks this only fires for nearly-max-size updates (>65520
  // bytes).
  while (remaining > 0) {
    iree_host_size_t chunk_length =
        remaining > max_data_bytes ? max_data_bytes : remaining;
    iree_host_size_t padded_length = iree_host_align(chunk_length, 4);
    uint32_t data_words = (uint32_t)(padded_length / 4);
    uint32_t size_words = 4 + data_words;

    // Determine alignment flag for this chunk.
    uint32_t flags = 0;
    if ((chunk_dst_offset % 4 == 0) && (chunk_length % 4 == 0)) {
      flags = IREE_HAL_WEBGPU_ISA_UPDATE_FLAG_ALIGNED;
    }

    uint32_t* words;
    IREE_RETURN_IF_ERROR(
        iree_hal_webgpu_builder_reserve(builder, size_words, &words));
    words[0] = iree_hal_webgpu_isa_header_encode(
        IREE_HAL_WEBGPU_ISA_OP_UPDATE_BUFFER, flags, size_words);
    words[1] = dst_slot;
    words[2] = chunk_dst_offset;
    words[3] = (uint32_t)chunk_length;

    // Copy inline data directly into the reserved instruction words.
    // Zero-pad the trailing bytes of the last word if not 4-byte aligned.
    uint8_t* data_dest = (uint8_t*)&words[4];
    memcpy(data_dest, source_data, chunk_length);
    if (padded_length > chunk_length) {
      memset(data_dest + chunk_length, 0, padded_length - chunk_length);
    }

    source_data += chunk_length;
    chunk_dst_offset += (uint32_t)chunk_length;
    remaining -= chunk_length;
  }

  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_builder_copy_buffer(
    iree_hal_webgpu_builder_t* builder, iree_hal_buffer_ref_t source_ref,
    iree_hal_buffer_ref_t target_ref) {
  uint32_t src_slot = 0;
  uint32_t src_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_resolve_ref(
      builder, source_ref, &src_slot, &src_offset));

  uint32_t dst_slot = 0;
  uint32_t dst_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_resolve_ref(
      builder, target_ref, &dst_slot, &dst_offset));

  // COPY_BUFFER is an encoder command.
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_ensure_encoder_open(builder));

  uint32_t* words;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_reserve(builder, 6, &words));
  words[0] = iree_hal_webgpu_isa_header_encode(
      IREE_HAL_WEBGPU_ISA_OP_COPY_BUFFER, /*flags=*/0, /*size_words=*/6);
  words[1] = src_slot;
  words[2] = src_offset;
  words[3] = dst_slot;
  words[4] = dst_offset;
  words[5] = (uint32_t)source_ref.length;

  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_builder_dispatch(
    iree_hal_webgpu_builder_t* builder,
    iree_hal_webgpu_handle_t pipeline_handle,
    iree_hal_webgpu_handle_t bind_group_layout_handle,
    const uint32_t workgroup_count[3], iree_hal_buffer_ref_list_t bindings) {
  // DISPATCH is an encoder command.
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_ensure_encoder_open(builder));

  uint32_t binding_count = (uint32_t)bindings.count;
  uint32_t size_words = 6 + binding_count * 3;

  // Reserve the header. Bindings are resolved individually below and may grow
  // the slot map, so we can't reserve the full instruction upfront (the slot
  // map growth path allocates, and we need reserve to have completed first).
  uint32_t* header_words;
  IREE_RETURN_IF_ERROR(
      iree_hal_webgpu_builder_reserve(builder, 6, &header_words));
  header_words[0] = iree_hal_webgpu_isa_header_encode(
      IREE_HAL_WEBGPU_ISA_OP_DISPATCH, /*flags=*/0, size_words);
  header_words[1] = pipeline_handle;
  header_words[2] = bind_group_layout_handle;
  header_words[3] = workgroup_count[0];
  header_words[4] = workgroup_count[1];
  header_words[5] = workgroup_count[2];

  // Emit per-binding entries.
  for (uint32_t i = 0; i < binding_count; ++i) {
    iree_hal_buffer_ref_t binding = bindings.values[i];
    uint32_t slot = 0;
    uint32_t offset = 0;
    if (binding.buffer) {
      IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_resolve_ref(builder, binding,
                                                               &slot, &offset));
    } else {
      // Indirect binding: slot is the binding table ordinal.
      if (binding.buffer_slot >= builder->dynamic_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dispatch binding %u references binding table slot %u which "
            "exceeds dynamic slot count %u",
            i, binding.buffer_slot, builder->dynamic_count);
      }
      slot = binding.buffer_slot;
      offset = (uint32_t)binding.offset;
    }
    uint32_t* binding_words;
    IREE_RETURN_IF_ERROR(
        iree_hal_webgpu_builder_reserve(builder, 3, &binding_words));
    binding_words[0] = slot;
    binding_words[1] = offset;
    binding_words[2] = (uint32_t)binding.length;
  }

  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_builder_execution_barrier(
    iree_hal_webgpu_builder_t* builder) {
  uint32_t* words;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_builder_reserve(builder, 1, &words));
  words[0] = iree_hal_webgpu_isa_header_encode(IREE_HAL_WEBGPU_ISA_OP_BARRIER,
                                               /*flags=*/0, /*size_words=*/1);
  return iree_ok_status();
}
