// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_ARENA_H_
#define IREE_BASE_INTERNAL_ARENA_H_

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomic_slist.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_arena_block_pool_t
//===----------------------------------------------------------------------===//

struct iree_arena_block_t;

// NOTE: this struct is at the *end* of allocated blocks such that we don't mess
// with alignment - byte 0 of a block is always byte 0 of the allocation from
// the system. We can do this as all blocks have the same size so computing the
// footer offset from a pointer is easy.
typedef struct iree_arena_block_t {
  struct iree_arena_block_t* next;
} iree_arena_block_t;

// An atomic approximately LIFO singly-linked list.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_atomic_arena_block, iree_arena_block_t,
                                offsetof(iree_arena_block_t, next));

// Returns the first usable byte of the given |block| following
// iree_max_align_t. Up to usable_block_size is available.
#define iree_arena_block_ptr(block_pool, block) \
  (void*)((uint8_t*)(block) - (block_pool)->usable_block_size)

// Returns the iree_arena_block_t from the given usable block |ptr| at offset 0.
#define iree_arena_block_trailer(block_pool, ptr) \
  (iree_arena_block_t*)((const uint8_t*)(ptr) + (block_pool)->usable_block_size)

// A simple atomic fixed-size block pool.
// Blocks are allocated from the system as required and kept in the pool to
// satisfy future requests. Blocks are all of a uniform size specified when the
// pool is created. It's recommended that power-of-two sizes are used for the
// blocks so that the underlying allocator is more likely to bucket them
// appropriately.
//
// Thread-safe; multiple threads may acquire and release blocks from the pool.
// The underlying allocator must also be thread-safe.
typedef struct iree_arena_block_pool_t {
  // Block size, in bytes. All blocks in the available_slist will have this
  // byte size which includes the iree_arena_block_t footer.
  iree_host_size_t total_block_size;
  // Block size, in bytes, of the usable bytes within a block.
  iree_host_size_t usable_block_size;
  // Allocator used for allocating/freeing each allocation block.
  iree_allocator_t block_allocator;
  // Linked list of free blocks (LIFO).
  iree_atomic_arena_block_slist_t available_slist;
} iree_arena_block_pool_t;

// Initializes a new block pool in |out_block_pool|.
// |block_allocator| will be used to allocate and free blocks for the pool.
// Each block allocated will be |total_block_size| but have a slightly smaller
// usable size due to the tracking overhead. Prefer powers of two.
void iree_arena_block_pool_initialize(iree_host_size_t total_block_size,
                                      iree_allocator_t block_allocator,
                                      iree_arena_block_pool_t* out_block_pool);

// Deinitializes a block pool and frees all allocations.
// All blocks that were acquired from the pool must have already been released
// back to it.
void iree_arena_block_pool_deinitialize(iree_arena_block_pool_t* block_pool);

// Trims the pool by freeing unused blocks back to the allocator.
// Acquired blocks are not freed and remain valid.
void iree_arena_block_pool_trim(iree_arena_block_pool_t* block_pool);

// Acquires a single block from the pool and returns it in |out_block|.
// The first usable byte of the block is returned in |out_ptr|.
// The block may be either a new allocation with undefined contents or a reused
// prior allocation with undefined contents.
iree_status_t iree_arena_block_pool_acquire(iree_arena_block_pool_t* block_pool,
                                            iree_arena_block_t** out_block,
                                            void** out_ptr);

// Releases one or more blocks back to the block pool.
// Any blocks chained in |block_head| will also be released allowing for
// low-overhead resets when the blocks are already tracked in linked lists.
void iree_arena_block_pool_release(iree_arena_block_pool_t* block_pool,
                                   iree_arena_block_t* block_head,
                                   iree_arena_block_t* block_tail);

//===----------------------------------------------------------------------===//
// iree_arena_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_arena_oversized_allocation_t {
  struct iree_arena_oversized_allocation_t* next;
} iree_arena_oversized_allocation_t;

// A lightweight bump-pointer arena allocator using a shared block pool.
// As allocations are made from the arena and block capacity is exhausted new
// blocks will be acquired from the pool. Upon being reset all blocks will be
// released back to the pool for reuse by either the same arena in the future or
// other arenas sharing the same pool.
//
// The size of each allocated block used by the arena is inherited from the
// block pool. Allocations from the arena may exceed the block size but will
// incur additional allocation overhead as the block pool is bypassed and the
// system allocator is directly used to service the request.
//
// Thread-compatible; the shared block pool is thread-safe and may be used by
// arenas on multiple threads but each arena must only be used by a single
// thread.
typedef struct iree_arena_allocator_t {
  // Fixed-size block pool used to acquire new blocks for the arena.
  iree_arena_block_pool_t* block_pool;
  // Total bytes allocated to the arena from the block pool or system allocator.
  iree_host_size_t total_allocation_size;
  // Total bytes allocated from the arena; the utilization of the arena can be
  // checked with `used_allocation_size / total_allocation_size`.
  iree_host_size_t used_allocation_size;
  // Linked list of oversized allocations made directly from the system
  // allocator used by the block pool.
  iree_arena_oversized_allocation_t* allocation_head;
  // Linked list of allocated blocks maintained so that reset can release them.
  iree_arena_block_t* block_head;
  iree_arena_block_t* block_tail;
  // The number of bytes remaining in the block pointed to by block_head.
  iree_host_size_t block_bytes_remaining;
} iree_arena_allocator_t;

// Initializes an arena that will use |block_pool| for allocating blocks as
// needed.
void iree_arena_initialize(iree_arena_block_pool_t* block_pool,
                           iree_arena_allocator_t* out_arena);

// Deinitializes the arena and returns allocated blocks to the parent pool.
void iree_arena_deinitialize(iree_arena_allocator_t* arena);

// Resets the entire arena and returns allocated blocks to the parent pool.
void iree_arena_reset(iree_arena_allocator_t* arena);

// Allocates |byte_length| contiguous bytes from the arena.
// The returned bytes will have undefined contents and must be initialized by
// the caller.
iree_status_t iree_arena_allocate(iree_arena_allocator_t* arena,
                                  iree_host_size_t byte_length, void** out_ptr);

// Returns an iree_allocator_t that allocates from the given |arena|.
// Frees are ignored as arenas can only be reset as a whole.
iree_allocator_t iree_arena_allocator(iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_ARENA_H_
