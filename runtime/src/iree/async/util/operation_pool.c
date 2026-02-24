// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/operation_pool.h"

#include "iree/base/internal/atomic_slist.h"

//===----------------------------------------------------------------------===//
// Size class configuration
//===----------------------------------------------------------------------===//

// Power-of-two size classes from 64 bytes to 16KB.
// Each size class includes the 16-byte slot header, so the usable space is
// (size_class - 16) bytes.
static const iree_host_size_t iree_async_op_pool_size_classes[] = {
    64,     // 48 bytes usable
    128,    // 112 bytes usable
    256,    // 240 bytes usable
    512,    // 496 bytes usable
    1024,   // 1008 bytes usable
    2048,   // 2032 bytes usable
    4096,   // 4080 bytes usable
    8192,   // 8176 bytes usable
    16384,  // 16368 bytes usable
};

#define IREE_ASYNC_OP_POOL_SIZE_CLASS_COUNT \
  (sizeof(iree_async_op_pool_size_classes) / sizeof(iree_host_size_t))

// Minimum size class (64 bytes).
#define IREE_ASYNC_OP_POOL_MIN_SIZE_CLASS 64

// Maximum pooled size (16KB). Requests larger than this go direct to allocator.
#define IREE_ASYNC_OP_POOL_DEFAULT_MAX_POOLED_SIZE 16384

// Default block size for new allocations (64KB).
#define IREE_ASYNC_OP_POOL_DEFAULT_BLOCK_SIZE (64 * 1024)

// Minimum block size (4KB).
#define IREE_ASYNC_OP_POOL_MIN_BLOCK_SIZE (4 * 1024)

//===----------------------------------------------------------------------===//
// Slot header
//===----------------------------------------------------------------------===//

// Hidden header placed before each pooled slot. This allows release() to
// determine which freelist to return the slot to without external tracking.
//
// Layout: [slot_header][operation...]
//         ^            ^
//         |            +-- returned to caller (16-byte aligned)
//         +-- internal tracking
//
// The header is 16 bytes to ensure operations start at IREE_MAX_ALIGNMENT
// boundaries. Operations may embed types requiring 16-byte alignment (atomics,
// SIMD data, etc.), and misalignment would cause undefined behavior.
typedef struct iree_async_op_pool_slot_header_t {
  // Index into iree_async_op_pool_size_classes[].
  // For direct allocations, set to IREE_ASYNC_OP_POOL_SIZE_CLASS_COUNT.
  uint8_t size_class;

  // Slot flags.
  uint8_t flags;

  // Reserved for future use (alignment padding to 16 bytes).
  uint8_t reserved[14];
} iree_async_op_pool_slot_header_t;

static_assert(sizeof(iree_async_op_pool_slot_header_t) == 16,
              "slot header must be 16 bytes for alignment");

// Slot was allocated directly from the allocator (oversized request).
#define IREE_ASYNC_OP_POOL_FLAG_DIRECT_ALLOC (1u << 0)

//===----------------------------------------------------------------------===//
// Block header
//===----------------------------------------------------------------------===//

// Block of slots for a single size class. Blocks are allocated from the
// allocator and carved into fixed-size slots. All slots in a block share the
// same size class.
//
// Layout: [block_header][slot 0][slot 1]...[slot N]
//
// Each slot is: [slot_header][operation space]
typedef struct iree_async_op_pool_block_t {
  // Intrusive pointer for the atomic block list.
  iree_atomic_slist_intrusive_ptr_t slist_next;

  // Size class index for all slots in this block.
  uint8_t size_class;

  // Number of slots in this block.
  uint16_t slot_count;

  // Total allocated size of this block (for freeing).
  iree_host_size_t allocated_size;
} iree_async_op_pool_block_t;

// Generate typed atomic slist wrappers for blocks.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_async_op_pool_block,
                                iree_async_op_pool_block_t,
                                offsetof(iree_async_op_pool_block_t,
                                         slist_next));

// Generate typed atomic slist wrappers for operations (slots in freelists).
// Uses the operation's existing `next` field.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_async_operation, iree_async_operation_t,
                                offsetof(iree_async_operation_t, next));

//===----------------------------------------------------------------------===//
// Pool structure
//===----------------------------------------------------------------------===//

struct iree_async_operation_pool_t {
  // Allocator for blocks and oversized operations.
  iree_allocator_t allocator;

  // Block size for new allocations.
  iree_host_size_t block_size;

  // Maximum operation size to pool. Larger requests go direct to allocator.
  iree_host_size_t max_pooled_size;

  // All allocated blocks (for cleanup). Blocks are never removed during normal
  // operation; they're only freed when the pool is destroyed.
  iree_async_op_pool_block_slist_t blocks;

  // Per-size-class freelists. Each list contains available slots of that class.
  iree_async_operation_slist_t freelists[IREE_ASYNC_OP_POOL_SIZE_CLASS_COUNT];
};

//===----------------------------------------------------------------------===//
// Size class utilities
//===----------------------------------------------------------------------===//

// Returns the size class index for a given requested size (including header).
// Returns SIZE_CLASS_COUNT if the size exceeds the maximum pooled size.
static iree_host_size_t iree_async_op_pool_size_class_for_size(
    iree_async_operation_pool_t* pool, iree_host_size_t size) {
  // Include header in size calculation.
  iree_host_size_t total_size = size + sizeof(iree_async_op_pool_slot_header_t);

  // Check if oversized.
  if (total_size > pool->max_pooled_size) {
    return IREE_ASYNC_OP_POOL_SIZE_CLASS_COUNT;
  }

  // Find smallest size class that fits.
  for (iree_host_size_t i = 0; i < IREE_ASYNC_OP_POOL_SIZE_CLASS_COUNT; ++i) {
    if (iree_async_op_pool_size_classes[i] >= total_size) {
      return i;
    }
  }

  // Should not reach here if max_pooled_size is configured correctly.
  return IREE_ASYNC_OP_POOL_SIZE_CLASS_COUNT;
}

//===----------------------------------------------------------------------===//
// Block allocation
//===----------------------------------------------------------------------===//

// Allocates a new block and adds all its slots to the freelist for the given
// size class. Returns the first slot from the new block (already removed from
// freelist).
static iree_status_t iree_async_op_pool_grow(
    iree_async_operation_pool_t* pool, iree_host_size_t size_class,
    iree_async_operation_t** out_operation) {
  *out_operation = NULL;

  iree_host_size_t slot_size = iree_async_op_pool_size_classes[size_class];
  iree_host_size_t header_size =
      iree_host_align(sizeof(iree_async_op_pool_block_t), 16);

  // Calculate how many slots fit in a block.
  iree_host_size_t usable_size = pool->block_size - header_size;
  iree_host_size_t slot_count = usable_size / slot_size;
  if (slot_count == 0) {
    // Block too small for even one slot of this size class.
    // This shouldn't happen with reasonable configuration.
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "block size %" PRIhsz
                            " too small for slot size %" PRIhsz,
                            pool->block_size, slot_size);
  }

  // Allocate the block.
  iree_async_op_pool_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(pool->allocator, pool->block_size, (void**)&block));

  // Initialize block header.
  block->slist_next = NULL;
  block->size_class = (uint8_t)size_class;
  block->slot_count = (uint16_t)slot_count;
  block->allocated_size = pool->block_size;

  // Add block to the pool's block list (for cleanup).
  iree_async_op_pool_block_slist_push(&pool->blocks, block);

  // Carve block into slots. First slot goes to caller, rest to freelist.
  uint8_t* slot_base = (uint8_t*)block + header_size;

  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    uint8_t* slot_ptr = slot_base + (i * slot_size);

    // Initialize slot header.
    iree_async_op_pool_slot_header_t* slot_header =
        (iree_async_op_pool_slot_header_t*)slot_ptr;
    slot_header->size_class = (uint8_t)size_class;
    slot_header->flags = 0;
    memset(slot_header->reserved, 0, sizeof(slot_header->reserved));

    // Operation pointer is after the header.
    iree_async_operation_t* operation =
        (iree_async_operation_t*)(slot_ptr +
                                  sizeof(iree_async_op_pool_slot_header_t));

    // Zero the operation memory.
    memset(operation, 0, slot_size - sizeof(iree_async_op_pool_slot_header_t));

    if (i == 0) {
      // First slot goes directly to caller.
      *out_operation = operation;
    } else {
      // Remaining slots go to freelist.
      iree_async_operation_slist_push(&pool->freelists[size_class], operation);
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Pool API
//===----------------------------------------------------------------------===//

iree_status_t iree_async_operation_pool_allocate(
    iree_async_operation_pool_options_t options, iree_allocator_t allocator,
    iree_async_operation_pool_t** out_pool) {
  *out_pool = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate and apply defaults.
  iree_host_size_t block_size = options.block_size;
  if (block_size == 0) {
    block_size = IREE_ASYNC_OP_POOL_DEFAULT_BLOCK_SIZE;
  } else if (block_size < IREE_ASYNC_OP_POOL_MIN_BLOCK_SIZE) {
    block_size = IREE_ASYNC_OP_POOL_MIN_BLOCK_SIZE;
  }

  iree_host_size_t max_pooled_size = options.max_pooled_size;
  if (max_pooled_size == 0) {
    max_pooled_size = IREE_ASYNC_OP_POOL_DEFAULT_MAX_POOLED_SIZE;
  }

  // Allocate pool structure.
  iree_async_operation_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*pool), (void**)&pool));

  pool->allocator = allocator;
  pool->block_size = block_size;
  pool->max_pooled_size = max_pooled_size;

  // Initialize atomic lists.
  iree_async_op_pool_block_slist_initialize(&pool->blocks);
  for (iree_host_size_t i = 0; i < IREE_ASYNC_OP_POOL_SIZE_CLASS_COUNT; ++i) {
    iree_async_operation_slist_initialize(&pool->freelists[i]);
  }

  *out_pool = pool;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_async_operation_pool_free(iree_async_operation_pool_t* pool) {
  if (!pool) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Free all blocks. Flush the block list and walk it.
  iree_async_op_pool_block_t* block = NULL;
  iree_async_op_pool_block_t* tail = NULL;
  if (iree_async_op_pool_block_slist_flush(
          &pool->blocks, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO, &block,
          &tail)) {
    while (block) {
      iree_async_op_pool_block_t* next =
          iree_async_op_pool_block_slist_get_next(block);
      iree_allocator_free(pool->allocator, block);
      block = next;
    }
  }

  // Deinitialize atomic lists.
  iree_async_op_pool_block_slist_deinitialize(&pool->blocks);
  for (iree_host_size_t i = 0; i < IREE_ASYNC_OP_POOL_SIZE_CLASS_COUNT; ++i) {
    iree_async_operation_slist_deinitialize(&pool->freelists[i]);
  }

  // Free pool structure.
  iree_allocator_t allocator = pool->allocator;
  iree_allocator_free(allocator, pool);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_async_operation_pool_acquire(
    iree_async_operation_pool_t* pool, iree_host_size_t size,
    iree_async_operation_t** out_operation) {
  *out_operation = NULL;

  // Determine which size class to use.
  iree_host_size_t size_class =
      iree_async_op_pool_size_class_for_size(pool, size);

  if (size_class >= IREE_ASYNC_OP_POOL_SIZE_CLASS_COUNT) {
    // Oversized: allocate directly.
    // Check for overflow before adding header size.
    if (size > IREE_HOST_SIZE_MAX - sizeof(iree_async_op_pool_slot_header_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "operation size %" PRIhsz " too large", size);
    }
    iree_host_size_t alloc_size =
        sizeof(iree_async_op_pool_slot_header_t) + size;
    uint8_t* memory = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(pool->allocator, alloc_size, (void**)&memory));

    // Initialize header with direct-alloc flag.
    // Use invalid size_class as defense-in-depth against flag corruption.
    iree_async_op_pool_slot_header_t* header =
        (iree_async_op_pool_slot_header_t*)memory;
    header->size_class = (uint8_t)IREE_ASYNC_OP_POOL_SIZE_CLASS_COUNT;
    header->flags = IREE_ASYNC_OP_POOL_FLAG_DIRECT_ALLOC;
    memset(header->reserved, 0, sizeof(header->reserved));

    // Zero the operation memory.
    iree_async_operation_t* operation =
        (iree_async_operation_t*)(memory +
                                  sizeof(iree_async_op_pool_slot_header_t));
    memset(operation, 0, size);

    *out_operation = operation;
    return iree_ok_status();
  }

  // Try to pop from freelist (fast path).
  iree_async_operation_t* operation =
      iree_async_operation_slist_pop(&pool->freelists[size_class]);
  if (operation) {
    // Zero the operation memory before returning.
    iree_host_size_t slot_size = iree_async_op_pool_size_classes[size_class];
    memset(operation, 0, slot_size - sizeof(iree_async_op_pool_slot_header_t));
    *out_operation = operation;
    return iree_ok_status();
  }

  // Freelist empty: grow the pool.
  return iree_async_op_pool_grow(pool, size_class, out_operation);
}

void iree_async_operation_pool_release(iree_async_operation_pool_t* pool,
                                       iree_async_operation_t* operation) {
  if (!pool || !operation) return;

  // Get the slot header (immediately before the operation).
  iree_async_op_pool_slot_header_t* header =
      (iree_async_op_pool_slot_header_t*)((uint8_t*)operation -
                                          sizeof(*header));

  if (header->flags & IREE_ASYNC_OP_POOL_FLAG_DIRECT_ALLOC) {
    // Direct allocation: free to allocator.
    iree_allocator_free(pool->allocator, header);
    return;
  }

  // Return to freelist for this size class.
  iree_host_size_t size_class = header->size_class;
  if (size_class < IREE_ASYNC_OP_POOL_SIZE_CLASS_COUNT) {
    iree_async_operation_slist_push(&pool->freelists[size_class], operation);
  }
  // Invalid size_class is silently ignored (corruption guard).
}

iree_host_size_t iree_async_operation_pool_trim(
    iree_async_operation_pool_t* pool, iree_host_size_t max_to_trim) {
  // Deferred: trimming would require tracking free slot counts per block and
  // returning entire blocks to the allocator when all slots are free. For now,
  // blocks are held until pool destruction.
  (void)pool;
  (void)max_to_trim;
  return 0;
}
