// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/debugging.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_arena_block_pool_t
//===----------------------------------------------------------------------===//

void iree_arena_block_pool_initialize(iree_host_size_t total_block_size,
                                      iree_allocator_t block_allocator,
                                      iree_arena_block_pool_t* out_block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_block_pool, 0, sizeof(*out_block_pool));
  out_block_pool->total_block_size = total_block_size;
  out_block_pool->usable_block_size =
      total_block_size - sizeof(iree_arena_block_t);
  out_block_pool->block_allocator = block_allocator;
  iree_atomic_arena_block_slist_initialize(&out_block_pool->available_slist);

  IREE_TRACE_ZONE_END(z0);
}

void iree_arena_block_pool_deinitialize(iree_arena_block_pool_t* block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Since all blocks must have been released we can just reuse trim (today) as
  // it doesn't retain any blocks.
  iree_arena_block_pool_trim(block_pool);
  iree_atomic_arena_block_slist_deinitialize(&block_pool->available_slist);

  IREE_TRACE_ZONE_END(z0);
}

void iree_arena_block_pool_trim(iree_arena_block_pool_t* block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_arena_block_t* head = NULL;
  iree_atomic_arena_block_slist_flush(
      &block_pool->available_slist,
      IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO, &head, NULL);
  while (head) {
    void* ptr = (uint8_t*)head - block_pool->usable_block_size;
    head = head->next;
    iree_allocator_free(block_pool->block_allocator, ptr);
  }

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_arena_block_pool_acquire(iree_arena_block_pool_t* block_pool,
                                            iree_arena_block_t** out_block,
                                            void** out_ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_arena_block_t* block =
      iree_atomic_arena_block_slist_pop(&block_pool->available_slist);

  if (!block) {
    // No blocks available; allocate one now.
    // Note that it's possible for there to be a race here where one thread
    // releases a block to the pool while we are trying to acquire one - in that
    // case we may end up allocating a block when perhaps we didn't need to but
    // that's fine - it's just one block and the contention means there's likely
    // to be a need for more anyway.
    uint8_t* block_base = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc_uninitialized(block_pool->block_allocator,
                                                block_pool->total_block_size,
                                                (void**)&block_base));
    block = iree_arena_block_trailer(block_pool, block_base);
    *out_ptr = block_base;
  } else {
    *out_ptr = iree_arena_block_ptr(block_pool, block);
  }

  block->next = NULL;
  *out_block = block;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_arena_block_pool_release(iree_arena_block_pool_t* block_pool,
                                   iree_arena_block_t* block_head,
                                   iree_arena_block_t* block_tail) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_atomic_arena_block_slist_concat(&block_pool->available_slist, block_head,
                                       block_tail);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_arena_allocator_t
//===----------------------------------------------------------------------===//

void iree_arena_initialize(iree_arena_block_pool_t* block_pool,
                           iree_arena_allocator_t* out_arena) {
  memset(out_arena, 0, sizeof(*out_arena));
  out_arena->block_pool = block_pool;
}

void iree_arena_deinitialize(iree_arena_allocator_t* arena) {
  iree_arena_reset(arena);
}

void iree_arena_reset(iree_arena_allocator_t* arena) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (arena->allocation_head != NULL) {
    iree_arena_oversized_allocation_t* head = arena->allocation_head;
    do {
      void* ptr = (void*)head;
      head = head->next;
      iree_allocator_free(arena->block_pool->block_allocator, ptr);
    } while (head);
    arena->allocation_head = NULL;
  }
  if (arena->block_head != NULL) {
#if defined(IREE_SANITIZER_ADDRESS)
    iree_arena_block_t* block = arena->block_head;
    while (block) {
      IREE_ASAN_UNPOISON_MEMORY_REGION(
          iree_arena_block_ptr(arena->block_pool, block),
          arena->block_pool->usable_block_size);
      block = block->next;
    }
#endif  // IREE_SANITIZER_ADDRESS
    iree_arena_block_pool_release(arena->block_pool, arena->block_head,
                                  arena->block_tail);
    arena->block_head = NULL;
    arena->block_tail = NULL;
  }

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_arena_allocate(iree_arena_allocator_t* arena,
                                  iree_host_size_t byte_length,
                                  void** out_ptr) {
  *out_ptr = NULL;

  iree_arena_block_pool_t* block_pool = arena->block_pool;

  if (byte_length > block_pool->usable_block_size) {
    // Oversized allocation that can't be handled by the block pool. We'll
    // allocate directly from the system allocator and track it ourselves for
    // freeing during reset.
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_arena_allocate_oversize");
    iree_host_size_t allocation_size =
        sizeof(iree_arena_oversized_allocation_t) + byte_length;
    iree_arena_oversized_allocation_t* allocation = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_allocator_malloc_uninitialized(
            block_pool->block_allocator, allocation_size, (void**)&allocation));
    allocation->next = arena->allocation_head;
    arena->allocation_head = allocation;
    arena->total_allocation_size += allocation_size;
    arena->used_allocation_size += byte_length;
    *out_ptr = (uint8_t*)allocation + sizeof(iree_arena_oversized_allocation_t);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Pad length allocated so that each pointer bump is always ending at an
  // aligned address and the next allocation will start aligned.
  iree_host_size_t aligned_length =
      iree_host_align(byte_length, iree_max_align_t);

  // Check to see if the current block (if any) has space - if not, get another.
  if (arena->block_head == NULL ||
      arena->block_bytes_remaining < aligned_length) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_arena_allocate_grow");
    iree_arena_block_t* block = NULL;
    void* ptr = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_block_pool_acquire(arena->block_pool, &block, &ptr));
    block->next = arena->block_head;
    arena->block_head = block;
    if (!arena->block_tail) arena->block_tail = block;
    arena->total_allocation_size += block_pool->total_block_size;
    arena->block_bytes_remaining = block_pool->usable_block_size;
    IREE_ASAN_POISON_MEMORY_REGION(
        iree_arena_block_ptr(block_pool, arena->block_head),
        block_pool->usable_block_size);
    IREE_TRACE_ZONE_END(z0);
  }

  // Slice out the allocation from the current block.
  void* ptr = (uint8_t*)arena->block_head - arena->block_bytes_remaining;
  IREE_ASAN_UNPOISON_MEMORY_REGION(ptr, aligned_length);
  arena->block_bytes_remaining -= aligned_length;
  arena->used_allocation_size += aligned_length;
  *out_ptr = ptr;
  return iree_ok_status();
}

static iree_status_t iree_arena_allocator_ctl(void* self,
                                              iree_allocator_command_t command,
                                              const void* params,
                                              void** inout_ptr) {
  iree_arena_allocator_t* arena = (iree_arena_allocator_t*)self;
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC: {
      const iree_allocator_alloc_params_t* alloc_params =
          (const iree_allocator_alloc_params_t*)params;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, alloc_params->byte_length, inout_ptr));
      if (command == IREE_ALLOCATOR_COMMAND_CALLOC) {
        memset(*inout_ptr, 0, alloc_params->byte_length);
      }
      return iree_ok_status();
    }
    case IREE_ALLOCATOR_COMMAND_FREE: {
      // Do nothing: can't free from an arena.
      return iree_ok_status();
    }
    default:
      // NOTE: we could try to support IREE_ALLOCATOR_COMMAND_REALLOC, but
      // it requires the original size to be able to do properly (without
      // copying memory we shouldn't have access to). For this and other reasons
      // we very rarely realloc in IREE so having this limitation isn't too bad.
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported iree_arena_t allocator command");
  }
}

iree_allocator_t iree_arena_allocator(iree_arena_allocator_t* arena) {
  iree_allocator_t v = {
      .self = arena,
      .ctl = iree_arena_allocator_ctl,
  };
  return v;
}
