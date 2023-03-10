// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/utils/block_list.h"

#include "iree/base/tracing.h"

iree_status_t iree_vm_bytecode_block_list_initialize(
    uint32_t capacity, iree_allocator_t allocator,
    iree_vm_bytecode_block_list_t* out_block_list) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_block_list);

  // In case we fail present an empty list.
  out_block_list->capacity = 0;
  out_block_list->count = 0;
  out_block_list->values = NULL;

  // Configure storage either inline if it fits or as a heap allocation.
  if (capacity > IREE_ARRAYSIZE(out_block_list->inline_storage)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(allocator,
                                  sizeof(out_block_list->values[0]) * capacity,
                                  (void**)&out_block_list->values));
  } else {
    out_block_list->values = out_block_list->inline_storage;
  }

  // Reset state and clear only the blocks we are using.
  out_block_list->capacity = capacity;
  out_block_list->count = 0;
  memset(out_block_list->values, 0,
         sizeof(out_block_list->values[0]) * capacity);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_vm_bytecode_block_list_deinitialize(
    iree_vm_bytecode_block_list_t* block_list, iree_allocator_t allocator) {
  if (!block_list) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (block_list->values != block_list->inline_storage) {
    iree_allocator_free(allocator, block_list->values);
  }
  block_list->capacity = 0;
  block_list->count = 0;
  block_list->values = NULL;

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_vm_bytecode_block_list_insert(
    iree_vm_bytecode_block_list_t* block_list, uint32_t pc,
    iree_vm_bytecode_block_t** out_block) {
  IREE_ASSERT_ARGUMENT(block_list);
  *out_block = NULL;

  if (IREE_UNLIKELY(pc >= IREE_VM_PC_BLOCK_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "block pc %08X greater than max %08X", pc,
                            IREE_VM_PC_BLOCK_MAX);
  }

  // Try to find the block or the next block greater than it in the list in case
  // we need to insert.
  iree_host_size_t ordinal = 0;
  iree_status_t status =
      iree_vm_bytecode_block_list_find(block_list, pc, &ordinal);
  if (iree_status_is_ok(status)) {
    // Block found.
    *out_block = &block_list->values[ordinal];
    return iree_ok_status();
  }
  status = iree_status_ignore(status);

  // Not found, need to insert at ordinal point at the next greatest pc.
  if (IREE_UNLIKELY(block_list->count + 1 > block_list->capacity)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid descriptor block count %u; found at least %u blocks",
        block_list->capacity, block_list->count + 1);
  }

  // Shift list up and declare new block.
  if (ordinal != block_list->count) {
    memmove(&block_list->values[ordinal + 1], &block_list->values[ordinal],
            (block_list->count - ordinal) * sizeof(block_list->values[0]));
  }
  iree_vm_bytecode_block_t* block = &block_list->values[ordinal];
  block->defined = 0;
  block->reserved = 0;
  block->pc = pc;

  ++block_list->count;
  *out_block = block;
  return iree_ok_status();
}

// Finds the ordinal of the block with the given |pc| within the block list.
// Note that these ordinals will change with each insertion and this is
// generally only safe to use after the list has been completed.
// If NOT_FOUND then |out_ordinal| will contain the index into the list of where
// the block would be inserted.
iree_status_t iree_vm_bytecode_block_list_find(
    const iree_vm_bytecode_block_list_t* block_list, uint32_t pc,
    iree_host_size_t* out_ordinal) {
  IREE_ASSERT_ARGUMENT(block_list);
  *out_ordinal = 0;
  int low = 0;
  int high = (int)block_list->count - 1;
  while (low <= high) {
    const int mid = low + (high - low) / 2;
    const uint32_t mid_pc = block_list->values[mid].pc;
    if (mid_pc < pc) {
      low = mid + 1;
    } else if (mid_pc > pc) {
      high = mid - 1;
    } else {
      // Found; early exit.
      *out_ordinal = mid;
      return iree_ok_status();
    }
  }
  // Not found; return the next highest slot. Note that this may be off the
  // end of the list if the search pc is greater than all current values.
  *out_ordinal = low;
  return iree_status_from_code(IREE_STATUS_NOT_FOUND);
}

iree_status_t iree_vm_bytecode_block_list_verify(
    const iree_vm_bytecode_block_list_t* block_list,
    iree_const_byte_span_t bytecode_data) {
  IREE_ASSERT_ARGUMENT(block_list);

  // Ensure we have as many blocks as expected.
  if (block_list->count != block_list->capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid descriptor block count %u; found %u blocks",
        block_list->capacity, block_list->count);
  }

  for (uint32_t i = 0; i < block_list->count; ++i) {
    const iree_vm_bytecode_block_t* block = &block_list->values[i];

    // Ensure all blocks are defined.
    if (!block->defined) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "block at pc %08X not defined in bytecode",
                              block->pc);
    }

    // Ensure each block pc is in bounds - we need at least 1 byte for the
    // marker.
    if (block->pc + 1 >= bytecode_data.data_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "block at pc %08X (%u) out of bytecode data range %" PRIhsz,
          block->pc, block->pc, bytecode_data.data_length);
    }

    // Ensure each block has a block opcode at its target.
    uint8_t opc = bytecode_data.data[block->pc];
    if (opc != IREE_VM_OP_CORE_Block) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "block at pc %08X does not start with a block marker opcode",
          block->pc);
    }
  }

  return iree_ok_status();
}
