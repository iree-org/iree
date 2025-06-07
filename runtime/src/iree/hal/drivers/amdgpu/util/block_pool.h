// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_BLOCK_POOL_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_BLOCK_POOL_H_

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Allocation and Transfer Utilities
//===----------------------------------------------------------------------===//

// TODO(benvanik): verify that 16 is enough - there are some rules for kernarg
// alignment we may need to respect. Things seem to work but that may be by
// chance and out of spec.
#define iree_hal_amdgpu_max_align_t 16

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_block_pool_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_block_t iree_hal_amdgpu_block_t;
typedef struct iree_hal_amdgpu_block_allocation_t
    iree_hal_amdgpu_block_allocation_t;
typedef struct iree_hal_amdgpu_block_pool_t iree_hal_amdgpu_block_pool_t;

// Options for configuring a block pool.
typedef struct iree_hal_amdgpu_block_pool_options_t {
  // Size in bytes of the device block. Must be a power of two.
  iree_device_size_t block_size;
  // Blocks per device allocation made.
  // This trades off potential underutilized allocations with the number of
  // allocations made. May be rounded up to meet device requirements.
  // If 0 then as many blocks as fit within a single recommended device
  // allocation will be used.
  iree_host_size_t min_blocks_per_allocation;
  // Initial capacity of the pool in blocks.
  // At least this number of blocks will be allocated during pool
  // initialization, possibly split into multiple block pool allocations.
  iree_host_size_t initial_capacity;
} iree_hal_amdgpu_block_pool_options_t;

// A block in the block pool.
// This is a suballocation of an iree_hal_amdgpu_block_allocation_t device
// allocation.
typedef struct iree_hal_amdgpu_block_t {
  // Device pointer to the allocated block.
  IREE_AMDGPU_DEVICE_PTR void* ptr;
  // Parent allocation of the block.
  // This could be derived from the block pointer if we placed the parent
  // iree_hal_amdgpu_block_allocation_t at a fixed address (if we ever need
  // another user_data field).
  iree_hal_amdgpu_block_allocation_t* allocation;
  // Next block in a user-defined block list. May be used for any purpose.
  // Initially NULL on blocks acquired from the pool. Note that this may
  // reference blocks in another allocation or even pool.
  iree_hal_amdgpu_block_t* next;
  // Previous block in a user-defined block list. May be used for any purpose.
  // Initially NULL on blocks acquired from the pool. Note that this may
  // reference blocks in another allocation or even pool.
  iree_hal_amdgpu_block_t* prev;
  // Arbitrary user data valid while the block is held by the user.
  // This can be used to sequester small amounts of metadata for tracking.
  // Initially 0 on blocks acquired from the pool. No cleanup is performed upon
  // release.
  uint64_t user_data[4];
} iree_hal_amdgpu_block_t;
static_assert(sizeof(iree_hal_amdgpu_block_t) == 64,
              "keep blocks cache line sized");

// Size of the user data field in a block in bytes.
#define IREE_HAL_AMDGPU_BLOCK_USER_DATA_SIZE \
  sizeof(((iree_hal_amdgpu_block_t*)NULL)->user_data)

// A single device allocation containing one or more blocks.
typedef struct iree_hal_amdgpu_block_allocation_t {
  // Next in the linked list of allocations managed by the block pool.
  iree_hal_amdgpu_block_allocation_t* next;
  // Base pointer of the device allocation.
  IREE_AMDGPU_DEVICE_PTR void* base_ptr;
  // Number of used blocks in the allocation outstanding.
  iree_host_size_t used_count;
  // Contiguously allocated blocks within the allocation.
  iree_alignas(64) iree_hal_amdgpu_block_t blocks[/*blocks_per_allocation*/];
} iree_hal_amdgpu_block_allocation_t;

// A shared pool of equal-sized blocks in device agent memory.
// Tries to make as few device allocations as possible and of the granularity
// requested by the driver (or larger). Device memory is not touched by the host
// as part of management and the memory may not even be host accessible.
//
// This uses a linked data structure to allow growth without reallocation (user
// workloads are unpredictable). The savings from pooling blocks by not having
// to call into HSA is many orders of magnitude greater than the cost of some
// linked-list pointer walks. Since users of the block pool almost always need a
// linked list to store the block in their own lists (iovecs, etc) we expose
// those as part of the internal tracking structure we need for managing free
// lists.
//
// Thread-safe; may be used by multiple queues on the same physical device with
// independent host threads.
typedef struct iree_hal_amdgpu_block_pool_t {
  // HSA API handle. Unowned and must be kept live by parent.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Host allocator used for block lists.
  iree_allocator_t host_allocator;
  // Agent the block pool is managing blocks on.
  hsa_agent_t agent;
  // Memory pool blocks are allocated from.
  hsa_amd_memory_pool_t memory_pool;
  // Size in bytes of a block on device.
  iree_device_size_t block_size;
  // Number of blocks in a single device allocation.
  iree_device_size_t blocks_per_allocation;
  // Mutex managing the block pool resources.
  iree_slim_mutex_t mutex;
  // Linked list of allocations managed by the pool.
  // Newly allocated blocks are inserted at the head of the list.
  iree_hal_amdgpu_block_allocation_t* allocations_head IREE_GUARDED_BY(mutex);
  // Linked list of free blocks in the pool.
  // The first block in the list is usually the last block released.
  iree_hal_amdgpu_block_t* free_blocks_head IREE_GUARDED_BY(mutex);
} iree_hal_amdgpu_block_pool_t;

// Initializes the block pool to allocate from the given |agent| |memory_pool|.
// Each block will have the same block size and multiple blocks may be allocated
// at the same time to hit the pool allocation granularity.
//
// If an initial block capacity is provided the block pool will make its initial
// growth allocation to have the given number of blocks available for use prior
// to returning.
//
// A reference to |libhsa| will be kept by the block pool in order to allocate
// and free memory and must remain live.
iree_status_t iree_hal_amdgpu_block_pool_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_block_pool_options_t options, hsa_agent_t agent,
    hsa_amd_memory_pool_t memory_pool, iree_allocator_t host_allocator,
    iree_hal_amdgpu_block_pool_t* out_block_pool);

// Deinitializes the block pool and releases all resources back to the device.
// Any outstanding block pointers will become invalid.
void iree_hal_amdgpu_block_pool_deinitialize(
    iree_hal_amdgpu_block_pool_t* block_pool);

// Trims the block pool by releasing all allocations that have no outstanding
// blocks allocated. Does not compact allocations to reduce fragmentation.
void iree_hal_amdgpu_block_pool_trim(iree_hal_amdgpu_block_pool_t* block_pool);

// Acquires a block from the |block_pool|, growing the pool if needed.
iree_status_t iree_hal_amdgpu_block_pool_acquire(
    iree_hal_amdgpu_block_pool_t* block_pool,
    iree_hal_amdgpu_block_t** out_block);

// Releases a block back to the pool.
// The block must have been acquired from |block_pool|.
void iree_hal_amdgpu_block_pool_release(
    iree_hal_amdgpu_block_pool_t* block_pool, iree_hal_amdgpu_block_t* block);

// Releases a linked list of blocks back to the pool.
// All blocks must have been acquired from |block_pool|.
void iree_hal_amdgpu_block_pool_release_list(
    iree_hal_amdgpu_block_pool_t* block_pool,
    iree_hal_amdgpu_block_t* block_head);

// Block pools for device memory blocks of various sizes.
// The pools may be configured differently for their usage based on who owns
// them but generally follow the same bucketing strategy.
typedef struct iree_hal_amdgpu_block_pools_t {
  // Used for small allocations of around ~4-32KB.
  iree_hal_amdgpu_block_pool_t small;
  // Used for large page-sized allocations of around ~64kB-512KB.
  iree_hal_amdgpu_block_pool_t large;
  // Any larger should (probably) be dedicated allocations.
} iree_hal_amdgpu_block_pools_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_block_arena_t
//===----------------------------------------------------------------------===//

// A lightweight bump-pointer arena allocator using a shared block pool.
// As allocations are made from the arena and block capacity is exhausted new
// blocks will be acquired from the pool. Upon being reset all blocks will be
// released back to the pool for reuse by either the same arena in the future or
// other arenas sharing the same pool.
//
// The size of each allocated block used by the arena is inherited from the
// block pool. Allocations from the arena can not exceed the block size.
//
// Thread-compatible; the shared block pool is thread-safe and may be used by
// arenas on multiple threads but each arena must only be used by a single
// thread at a time.
typedef struct iree_hal_amdgpu_block_arena_t {
  // Fixed-size block pool used to acquire new blocks for the arena.
  iree_hal_amdgpu_block_pool_t* block_pool;
  // Total bytes allocated to the arena from the block pool.
  iree_device_size_t total_allocation_size;
  // Total bytes allocated from the arena; the utilization of the arena can be
  // checked with `used_allocation_size / total_allocation_size`.
  iree_device_size_t used_allocation_size;
  // Linked list of allocated blocks maintained so that reset can release them.
  // Newly allocated blocks are appended to the list such that block_tail is
  // always the most recently allocated block.
  iree_hal_amdgpu_block_t* block_head;
  iree_hal_amdgpu_block_t* block_tail;
  // The number of bytes remaining in the block pointed to by block_head.
  iree_device_size_t block_bytes_remaining;
} iree_hal_amdgpu_block_arena_t;

// Initializes an arena that will use |block_pool| for allocating blocks as
// needed from device memory.
void iree_hal_amdgpu_block_arena_initialize(
    iree_hal_amdgpu_block_pool_t* block_pool,
    iree_hal_amdgpu_block_arena_t* out_arena);

// Deinitializes the arena and returns allocated blocks to the parent pool.
void iree_hal_amdgpu_block_arena_deinitialize(
    iree_hal_amdgpu_block_arena_t* arena);

// Resets the entire arena and returns allocated blocks to the parent pool.
void iree_hal_amdgpu_block_arena_reset(iree_hal_amdgpu_block_arena_t* arena);

// Releases ownership of the allocated blocks and returns them as a FIFO linked
// list. The arena will be reset and ready to allocate new blocks.
iree_hal_amdgpu_block_t* iree_hal_amdgpu_block_arena_release_blocks(
    iree_hal_amdgpu_block_arena_t* arena);

// Allocates |byte_length| contiguous bytes from the arena.
// The returned bytes will have undefined contents and must be initialized by
// the caller.
iree_status_t iree_hal_amdgpu_block_arena_allocate(
    iree_hal_amdgpu_block_arena_t* arena, iree_device_size_t byte_length,
    IREE_AMDGPU_DEVICE_PTR void** out_ptr);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_block_allocator_t
//===----------------------------------------------------------------------===//

// Opaque token handed out with allocations used to quickly free the allocation.
typedef union iree_hal_amdgpu_block_token_t {
  uint64_t bits;
  struct {
    // Number of pages the allocation occupies in the block.
    uint64_t page_count : 8;
    // iree_hal_amdgpu_block_t pointer; we only need the lower 7 bytes to
    // represent blocks. The block is aligned and we could steal some lower bits
    // if we wanted to allow a larger page_count.
    uint64_t block : 56;
  };
} iree_hal_amdgpu_block_token_t;
static_assert(sizeof(iree_hal_amdgpu_block_token_t) == sizeof(uint64_t),
              "must match reserved space in the device library");

// A block suballocator intended for relatively small allocations (~256 bytes to
// ~4096 bytes) made at average frequency (~once per queue submission).
//
// Each block acquired from the block pool is divided into 256 fixed size pages.
// This allows for a bitmap of used pages to be stored inline in the metadata of
// the block the pages are present in. It also allows for O(1) deallocation at
// the cost of O(block count) allocation. This is achieved by providing a token
// with each allocation that contains the block pointer packed with the page
// count of the allocation (knowing it is always <= 256) avoiding the need to
// either touch the device-side memory or allocate additional host-side
// metadata. The memory overhead of each allocation rounds to zero as the token
// is stored in the data structures of the client code requesting the allocation
// and each block already has 256 bits of host-local user data storage available
// for use.
//
// The downside of this implementation is that it can suffer from internal
// fragmentation. If exclusively 129 page allocations are made nearly half of
// all acquired block storage will be wasted. That's (hopefully) rare for the
// intended usage which is either ~64-128 byte allocations (most scheduler queue
// entries) or ~1024-4096 byte allocations (execution entries with binding
// tables). If fragmentation becomes an issue we can bucket the free list by
// number of contiguous pages free and reduce the scan cost.
//
// The allocator uses the user_data[] field in iree_hal_amdgpu_block_t to store
// the page occupancy bitmap. Though fixed today we could extend it to allow
// for larger bitmaps when block sizes are much greater than page size. It's
// expected that users will route requests to a block pool corresponding to
// their size class so as to avoid overallocation/under-utilization: allocating
// 1 byte from the large block pool would acquire an entire large block that
// would be nearly entirely unused if we didn't do that first-level filtering.
//
// Thread-safe; allocate/free are guarded within the allocator and the
// underlying block pool is also thread-safe.
typedef struct iree_hal_amdgpu_block_allocator_t {
  // Power-of-two calculated allocation granularity in bytes. All allocations
  // are padded to this size.
  uint32_t page_size;
  // Power-of-two number of pages within each block.
  uint32_t page_count;
  // Block pool with fixed-size blocks that the allocator uses for storage.
  iree_hal_amdgpu_block_pool_t* block_pool;
  // Guards access to the block lists and block bitmaps.
  iree_slim_mutex_t mutex;
  // Doubly linked list of blocks. Roughly sorted with blocks that have free
  // space near the front.
  iree_hal_amdgpu_block_t* block_head IREE_GUARDED_BY(mutex);
  iree_hal_amdgpu_block_t* block_tail IREE_GUARDED_BY(mutex);
} iree_hal_amdgpu_block_allocator_t;

// Initializes a new allocator that acquires its memory from the given
// |block_pool| and with a fixed power-of-two allocation |min_page_size|.
// Allocations will be padded to the granularity.
iree_status_t iree_hal_amdgpu_block_allocator_initialize(
    iree_hal_amdgpu_block_pool_t* block_pool, iree_host_size_t min_page_size,
    iree_hal_amdgpu_block_allocator_t* out_allocator);

// Deinitializes the allocator and frees all blocks back to the block pool.
// Requires that all outstanding allocations have been freed.
void iree_hal_amdgpu_block_allocator_deinitialize(
    iree_hal_amdgpu_block_allocator_t* allocator);

// Allocates a range of memory with iree_hal_amdgpu_max_align_t alignment.
// |out_ptr| will point to the device address of the allocation (which may not
// be host accessible) and |out_token| is opaque allocation-specific metadata
// that must be passed to iree_hal_amdgpu_block_allocator_free.
iree_status_t iree_hal_amdgpu_block_allocator_allocate(
    iree_hal_amdgpu_block_allocator_t* allocator, iree_host_size_t size,
    IREE_AMDGPU_DEVICE_PTR void** out_ptr,
    iree_hal_amdgpu_block_token_t* out_token);

// Frees an allocated |ptr| with associated metadata |token|.
// The corresponding pages will be marked as free within the parent block and
// if the block has no more used pages it will be returned to the pool.
void iree_hal_amdgpu_block_allocator_free(
    iree_hal_amdgpu_block_allocator_t* allocator,
    IREE_AMDGPU_DEVICE_PTR void* ptr, iree_hal_amdgpu_block_token_t token);

// Block allocators for device memory blocks of various sizes.
// The allocators are configured to use the block pools in
// iree_hal_amdgpu_block_pools_t.
typedef struct iree_hal_amdgpu_block_allocators_t {
  // Used for small allocations of around ~64B-256B.
  iree_hal_amdgpu_block_allocator_t small;
  // Used for large allocations of around ~4096B-256KB.
  iree_hal_amdgpu_block_allocator_t large;
} iree_hal_amdgpu_block_allocators_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_BLOCK_POOL_H_
