// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/local/arena.h"

#include "iree/base/alignment.h"

//===----------------------------------------------------------------------===//
// iree_arena_block_pool_t
//===----------------------------------------------------------------------===//

void iree_arena_block_pool_initialize(iree_host_size_t total_block_size,
                                      iree_allocator_t block_allocator,
                                      iree_arena_block_pool_t* out_block_pool) {
  memset(out_block_pool, 0, sizeof(*out_block_pool));
  out_block_pool->total_block_size = total_block_size;
  out_block_pool->usable_block_size =
      total_block_size - sizeof(iree_arena_block_t);
  out_block_pool->block_allocator = block_allocator;
  iree_atomic_arena_block_slist_initialize(&out_block_pool->available_slist);
}

void iree_arena_block_pool_deinitialize(iree_arena_block_pool_t* block_pool) {
  // Since all blocks must have been released we can just reuse trim (today) as
  // it doesn't retain any blocks.
  iree_arena_block_pool_trim(block_pool);
  iree_atomic_arena_block_slist_deinitialize(&block_pool->available_slist);
}

void iree_arena_block_pool_trim(iree_arena_block_pool_t* block_pool) {
  iree_arena_block_t* head = NULL;
  iree_atomic_arena_block_slist_flush(
      &block_pool->available_slist,
      IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO, &head, NULL);
  while (head) {
    void* ptr = (uint8_t*)head - block_pool->usable_block_size;
    head = head->next;
    iree_allocator_free(block_pool->block_allocator, ptr);
  }
}

iree_status_t iree_arena_block_pool_acquire(iree_arena_block_pool_t* block_pool,
                                            iree_arena_block_t** out_block) {
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
        z0, block_pool->block_allocator.alloc(block_pool->block_allocator.self,
                                              0, block_pool->total_block_size,
                                              (void**)&block_base));
    block = (iree_arena_block_t*)(block_base + (block_pool->total_block_size -
                                                sizeof(iree_arena_block_t)));
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
    iree_arena_block_pool_release(arena->block_pool, arena->block_head,
                                  arena->block_tail);
    arena->block_head = NULL;
    arena->block_tail = NULL;
  }
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
    iree_host_size_t allocation_size =
        sizeof(iree_arena_oversized_allocation_t) + byte_length;
    iree_arena_oversized_allocation_t* allocation = NULL;
    IREE_RETURN_IF_ERROR(block_pool->block_allocator.alloc(
        block_pool->block_allocator.self, 0, allocation_size,
        (void**)&allocation));
    allocation->next = arena->allocation_head;
    arena->allocation_head = allocation;
    arena->total_allocation_size += allocation_size;
    arena->used_allocation_size += byte_length;
    *out_ptr = (uint8_t*)allocation + sizeof(iree_arena_oversized_allocation_t);
    return iree_ok_status();
  }

  // Pad length allocated so that each pointer bump is always ending at an
  // aligned address and the next allocation will start aligned.
  iree_host_size_t aligned_length = iree_align(byte_length, iree_max_align_t);

  // Check to see if the current block (if any) has space - if not, get another.
  if (arena->block_head == NULL ||
      arena->block_bytes_remaining < aligned_length) {
    iree_arena_block_t* block = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_block_pool_acquire(arena->block_pool, &block));
    block->next = arena->block_head;
    arena->block_head = block;
    if (!arena->block_tail) arena->block_tail = block;
    arena->total_allocation_size += block_pool->total_block_size;
    arena->block_bytes_remaining = block_pool->usable_block_size;
  }

  // Slice out the allocation from the current block.
  void* ptr = (uint8_t*)arena->block_head - arena->block_bytes_remaining;
  arena->block_bytes_remaining -= aligned_length;
  arena->used_allocation_size += aligned_length;
  *out_ptr = ptr;
  return iree_ok_status();
}

static iree_status_t iree_arena_allocate_thunk(void* self,
                                               iree_allocation_mode_t mode,
                                               iree_host_size_t byte_length,
                                               void** out_ptr) {
  iree_arena_allocator_t* arena = (iree_arena_allocator_t*)self;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, byte_length, out_ptr));
  if (mode & IREE_ALLOCATION_MODE_ZERO_CONTENTS) {
    memset(*out_ptr, 0, byte_length);
  }
  return iree_ok_status();
}

iree_allocator_t iree_arena_allocator(iree_arena_allocator_t* arena) {
  iree_allocator_t v = {
      .self = arena,
      .alloc = (iree_allocator_alloc_fn_t)iree_arena_allocate_thunk,
      .free = NULL,
  };
  return v;
}
