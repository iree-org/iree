// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_UTILS_BLOCK_LIST_H_
#define IREE_VM_BYTECODE_UTILS_BLOCK_LIST_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/utils/isa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reserved inline storage capacity for blocks in the block list.
// If more blocks than this are requested we'll heap allocate instead.
// Most programs today end up with a small number of blocks (sometimes 2-3 for
// the entire program).
#define IREE_VM_BYTECODE_INLINE_BLOCK_LIST_CAPACITY (32)

// A tracked block within the block list.
// Blocks may be either declared (observed as a branch target) or defined
// (observed within the bytecode stream).
typedef struct iree_vm_bytecode_block_t {
  // Set only if the block definition has been seen.
  uint32_t defined : 1;
  uint32_t reserved : 7;
  // Program counter of the block within the function.
  uint32_t pc : 24;
} iree_vm_bytecode_block_t;

// A sorted list of blocks.
// Allows for single-pass verification
typedef struct iree_vm_bytecode_block_list_t {
  // Available capacity in |values| and the total expected block count.
  uint32_t capacity;
  // Current count of blocks in the |values| list.
  uint32_t count;
  // List of blocks sorted by program counter.
  // Will either point to inline_storage or be a heap allocation.
  iree_vm_bytecode_block_t* values;
  // Inlined storage for reasonable block counts to avoid the heap alloc.
  iree_vm_bytecode_block_t
      inline_storage[IREE_VM_BYTECODE_INLINE_BLOCK_LIST_CAPACITY];
} iree_vm_bytecode_block_list_t;

// Initializes a block list with the expected count of |capacity|.
// The same |allocator| must be passed to
// iree_vm_bytecode_block_list_deinitialize; the expectation is that the hosting
// data structure already has a reference to the allocator.
iree_status_t iree_vm_bytecode_block_list_initialize(
    uint32_t capacity, iree_allocator_t allocator,
    iree_vm_bytecode_block_list_t* out_block_list);

// Deinitializes a block list using |allocator| if any heap allocations were
// required (must be the same as passed to
// iree_vm_bytecode_block_list_initialize).
void iree_vm_bytecode_block_list_deinitialize(
    iree_vm_bytecode_block_list_t* block_list, iree_allocator_t allocator);

// Looks up a block at |pc| in the |block_list|. If not found the block is
// inserted as declared. Returns the block. Fails if capacity is exceeded.
// The returned |out_block| pointer is only valid until the next insertion.
iree_status_t iree_vm_bytecode_block_list_insert(
    iree_vm_bytecode_block_list_t* block_list, uint32_t pc,
    iree_vm_bytecode_block_t** out_block);

// Finds the ordinal of the block with the given |pc| within the block list.
// Note that these ordinals will change with each insertion and this is
// generally only safe to use after the list has been completed.
// If NOT_FOUND then |out_ordinal| will contain the index into the list of where
// the block would be inserted.
iree_status_t iree_vm_bytecode_block_list_find(
    const iree_vm_bytecode_block_list_t* block_list, uint32_t pc,
    iree_host_size_t* out_ordinal);

// Verifies that all blocks in the block list were defined and have proper
// tracking in |bytecode_data|.
iree_status_t iree_vm_bytecode_block_list_verify(
    const iree_vm_bytecode_block_list_t* block_list,
    iree_const_byte_span_t bytecode_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_UTILS_BLOCK_LIST_H_
