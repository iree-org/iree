// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/resource_set.h"

#include "iree/base/internal/debugging.h"
#include "iree/base/tracing.h"

// Computes the total capacity in resources of a chunk allocated with a total
// |storage_size| (including the header).
static uint16_t iree_hal_resource_set_chunk_capacity(
    iree_host_size_t storage_size) {
  return iree_min((storage_size - sizeof(iree_hal_resource_set_chunk_t)) /
                      sizeof(iree_hal_resource_t*),
                  IREE_HAL_RESOURCE_SET_CHUNK_MAX_CAPACITY);
}

IREE_API_EXPORT iree_status_t iree_hal_resource_set_allocate(
    iree_arena_block_pool_t* block_pool, iree_hal_resource_set_t** out_set) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // We could allow larger sizes (would require widening the capacity/count
  // fields in the chunk) but in real usage having even 64k is a bit too much.
  IREE_ASSERT_LE(block_pool->usable_block_size, 64 * 1024,
                 "keep block sizes small for resource sets");

  // Acquire block and place the set struct at the head.
  iree_arena_block_t* block = NULL;
  iree_hal_resource_set_t* set = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_block_pool_acquire(block_pool, &block, (void**)&set));
  memset(set, 0, sizeof(*set));
  set->block_pool = block_pool;

  // Inline the first chunk into the block using all of the remaining space.
  // This is a special case chunk that is released back to the pool with the
  // resource set and lets us avoid an additional allocation.
  // The total capacity in resources will be less than those of chunks allocated
  // from the block pool as there's reserved space at the front for the
  // iree_hal_resource_set_t.
  iree_hal_resource_set_chunk_t* inlined_chunk =
      (iree_hal_resource_set_chunk_t*)((uint8_t*)set + sizeof(*set));
  inlined_chunk->next_chunk = NULL;
  inlined_chunk->capacity = iree_hal_resource_set_chunk_capacity(
      block_pool->usable_block_size -
      iree_host_align(sizeof(iree_hal_resource_set_t), iree_max_align_t));
  inlined_chunk->count = 0;
  set->chunk_head = inlined_chunk;

  *out_set = set;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_resource_set_release_blocks(iree_hal_resource_set_t* set) {
  // Release all resources in all chunks and stitch together the blocks in a
  // linked list. We do this first so that we can release all of the chunks back
  // to the block pool in one operation. Ideally we'd maintain the linked list
  // in our chunks but there's some weirdness with prefix/suffix header/footers
  // that isn't worth the complexity.
  iree_arena_block_t* block_head = NULL;
  iree_arena_block_t* block_tail = NULL;
  iree_hal_resource_set_chunk_t* chunk = set->chunk_head;
  while (chunk) {
    iree_hal_resource_set_chunk_t* next_chunk = chunk->next_chunk;

    // Release all resources in the chunk.
    for (iree_host_size_t i = 0; i < chunk->count; ++i) {
      iree_hal_resource_release(chunk->resources[i]);
    }

    // Consume the chunk and add it to the block pool release linked list.
    // Note that the block metadata is located in different places depending on
    // whether this is the inline chunk or not.
    iree_arena_block_t* block = iree_arena_block_trailer(
        set->block_pool,
        iree_hal_resource_set_chunk_is_stored_inline(set, chunk)
            ? (void*)set
            : (void*)chunk);
    block->next = block_head;
    block_head = block;
    if (!block_tail) block_tail = block;

    chunk = next_chunk;
  }

  // Release all blocks back to the block pool in one operation.
  // NOTE: this invalidates the |set| memory.
  iree_arena_block_pool_t* block_pool = set->block_pool;
  iree_arena_block_pool_release(block_pool, block_head, block_tail);
}

IREE_API_EXPORT void iree_hal_resource_set_free(iree_hal_resource_set_t* set) {
  IREE_TRACE_ZONE_BEGIN(z0);

#if defined(IREE_SANITIZER_ADDRESS)
  // Unpoison the set so we can access the chunk list.
  IREE_ASAN_UNPOISON_MEMORY_REGION(set, sizeof(iree_hal_resource_set_t));
  // Unpoison all chunks so that we can free them.
  iree_hal_resource_set_chunk_t* chunk = set->chunk_head;
  while (chunk) {
    // NOTE: as the chunk header is stored in the poisoned memory we need to
    // unpoison based only on the base pointer and the shared set information.
    IREE_ASAN_UNPOISON_MEMORY_REGION(
        iree_hal_resource_set_chunk_is_stored_inline(set, chunk) ? (void*)set
                                                                 : (void*)chunk,
        set->block_pool->usable_block_size);
    // Once unpoisoned we can read the memory to get the next chunk.
    chunk = chunk->next_chunk;
  }
#endif  // IREE_SANITIZER_ADDRESS

  // Release all resources and the arena block used by the set.
  // The set pointer is invalid after this call returns.
  iree_hal_resource_set_release_blocks(set);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_resource_set_freeze(
    iree_hal_resource_set_t* set) {
#if defined(IREE_SANITIZER_ADDRESS)
  // Poison all chunks until the resource set is freed.
  iree_hal_resource_set_chunk_t* chunk = set->chunk_head;
  while (chunk) {
    iree_hal_resource_set_chunk_t* next_chunk = chunk->next_chunk;
    IREE_ASAN_POISON_MEMORY_REGION(
        chunk, set->block_pool->usable_block_size -
                   (iree_hal_resource_set_chunk_is_stored_inline(set, chunk)
                        ? iree_host_align(sizeof(iree_hal_resource_set_t),
                                          iree_max_align_t)
                        : 0));
    chunk = next_chunk;
  }
  // Poison the set.
  IREE_ASAN_POISON_MEMORY_REGION(set, sizeof(iree_hal_resource_set_t));
#endif  // IREE_SANITIZER_ADDRESS
}

// Retains |resource| and adds it to the main |set| list.
static iree_status_t iree_hal_resource_set_insert_retain(
    iree_hal_resource_set_t* set, iree_hal_resource_t* resource) {
  iree_hal_resource_set_chunk_t* chunk = set->chunk_head;
  if (IREE_UNLIKELY(chunk->count + 1 > chunk->capacity)) {
    // Ran out of room in the current chunk - acquire a new one and link it into
    // the list of chunks.
    iree_arena_block_t* block = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_block_pool_acquire(set->block_pool, &block, (void**)&chunk));
    chunk->next_chunk = set->chunk_head;
    set->chunk_head = chunk;
    chunk->capacity = iree_hal_resource_set_chunk_capacity(
        set->block_pool->usable_block_size);
    chunk->count = 0;
  }

  // Retain and insert into the chunk.
  chunk->resources[chunk->count++] = resource;
  iree_hal_resource_retain(resource);
  return iree_ok_status();
}

// Scans the lookaside for the resource pointer and updates the order if found.
// If the resource was not found then it will be inserted into the main list as
// well as the MRU.
//
// This performs a full scan over the MRU and if the resource is found will
// move the resource to the front of the list before returning. Otherwise the
// resource will be retained in the main source-of-truth list.
//
// Example (hit):
//   +----+----+----+----+
//   | AA | BB | CC | DD |  resource: CC
//   +----+----+----+----+
//   scan mru to find CC:
//     found at mru[2]
//     shift prefix down 1:
//       +----+----+----+----+
//       | AA | AA | BB | DD |
//       +----+----+----+----+
//     insert resource at front:
//       +----+----+----+----+
//       | CC | AA | BB | DD |
//       +----+----+----+----+
//
// Example (miss):
//   +----+----+----+----+
//   | AA | BB | CC | DD |  resource: EE
//   +----+----+----+----+
//   scan mru to find EE: not found
//   shift set down 1:
//     +----+----+----+----+
//     | AA | AA | BB | CC |
//     +----+----+----+----+
//   insert resource at front:
//     +----+----+----+----+
//     | EE | AA | BB | CC |
//     +----+----+----+----+
//   insert resource into main list
//
// The intent here is that we can model this behavior with SIMD ops to perform
// both the scan and update using comparison, extraction, and permutation. The
// best and worst case flows will load the entire MRU into registers from a
// single cache line, do all the scanning and shifting in registers, and then
// store back to the single cache line.
//
// Today, though, we leave this as an exercise to whoever comes across this :)
// Notes:
//   As the MRU is a fixed size we can unroll it entirely and avoid any looping.
//   On a 32-bit system with uint32x4_t we only need 4 registers.
//   On a 64-bit system with uint64x2_t we also only need 4 registers - though
//   the MRU has half as many entries and we may want to go >1 cache line.
//
//   If we wanted to process more than one resource at a time we can specialize
//   the code paths to handle 1/2/4/etc resources and process in batches with
//   an optional remainder. This would increase the ratio of work performed on
//   the loaded MRU registers before we do the shift/store.
//
//   The tree sequence we likely want is something like:
//    https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupq_n_u32
//    https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqq_u32
//    https://developer.arm.com/architectures/instruction-sets/intrinsics/vorrq_u32
//    https://developer.arm.com/architectures/instruction-sets/intrinsics/vmaxvq_u32
//    or
//    https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupq_n_u64
//    https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqq_u64
//    https://developer.arm.com/architectures/instruction-sets/intrinsics/vorrq_u64
//    https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_u64_u32
//    https://developer.arm.com/architectures/instruction-sets/intrinsics/vmaxvq_u32
//   This would yield whether the pointer was found, but instead of maxing at
//   the end we can use the produced mask to extract out a single register with
//   which positions are hits and use that to then permute the registers into
//   the proper order. At the end we could use a table instruction to remap and
//   extract out a byte/bitmap of the indices that we need to insert into the
//   main set.
//
//   The shifting can be performed with
//    https://developer.arm.com/architectures/instruction-sets/intrinsics/vextq_u32
//    https://developer.arm.com/architectures/instruction-sets/intrinsics/vextq_u64
//   This takes n low elements of LHS and rest from RHS and we can cascade them
//   to shift down the whole MRU.
//
//   We can use SIMDE as a rosetta stone for getting neon/avx/wasm/etc:
//   https://github.com/simd-everywhere/simde/blob/master/simde/arm/neon/ceq.h#L591
static iree_status_t iree_hal_resource_set_insert_1(
    iree_hal_resource_set_t* set, iree_hal_resource_t* resource) {
  // Scan and hope for a hit.
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(set->mru); ++i) {
    if (set->mru[i] != resource) continue;
    // Hit - keep the list sorted by most->least recently used.
    // We shift the MRU down to make room at index 0 and store the
    // resource there.
    if (i > 0) {
      memmove(&set->mru[1], &set->mru[0], sizeof(set->mru[0]) * i);
      set->mru[0] = resource;
    }
    return iree_ok_status();
  }

  // Miss - insert into the main list (slow path).
  // Note that we do this before updating the MRU in case allocation fails - we
  // don't want to keep the pointer around unless we've really retained it.
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_retain(set, resource));

  // Shift the MRU down and insert the new item at the head.
  memmove(&set->mru[1], &set->mru[0],
          sizeof(set->mru[0]) * (IREE_ARRAYSIZE(set->mru) - 1));
  set->mru[0] = resource;

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_resource_set_insert(iree_hal_resource_set_t* set,
                             iree_host_size_t count, const void* resources) {
  // For now we process one at a time. We should have a stride that lets us
  // amortize the cost of doing the MRU update and insertion allocation by
  // say slicing off 4/8/16/32 resources at a time etc. Today each miss that
  // requires a full insertion goes down the whole path of checking chunk
  // capacity and such.
  iree_hal_resource_t* const* typed_resources =
      (iree_hal_resource_t* const*)resources;
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_resource_set_insert_1(set, typed_resources[i]));
  }
  return iree_ok_status();
}
