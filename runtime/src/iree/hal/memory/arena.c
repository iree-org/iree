// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/arena.h"

#include "iree/base/internal/atomics.h"

iree_status_t iree_hal_memory_arena_allocate(
    iree_hal_memory_arena_options_t options, iree_allocator_t host_allocator,
    iree_hal_memory_arena_t** out_arena) {
  IREE_ASSERT_ARGUMENT(out_arena);
  *out_arena = NULL;

  if (options.capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capacity must be > 0");
  }
  if (options.frontier_capacity == 0) {
    options.frontier_capacity = IREE_HAL_MEMORY_ARENA_DEFAULT_FRONTIER_CAPACITY;
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Compute total allocation size: the arena struct followed by two inline
  // frontier slots (previous_frontier and accumulator), each with capacity
  // for frontier_capacity entries.
  iree_host_size_t total_size = 0;
  iree_host_size_t previous_offset = 0;
  iree_host_size_t accumulator_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          sizeof(iree_hal_memory_arena_t), &total_size,
          IREE_STRUCT_FIELD_ALIGNED(1, iree_async_frontier_t,
                                    iree_alignof(iree_async_frontier_entry_t),
                                    &previous_offset),
          IREE_STRUCT_FIELD(options.frontier_capacity,
                            iree_async_frontier_entry_t, NULL),
          IREE_STRUCT_FIELD_ALIGNED(1, iree_async_frontier_t,
                                    iree_alignof(iree_async_frontier_entry_t),
                                    &accumulator_offset),
          IREE_STRUCT_FIELD(options.frontier_capacity,
                            iree_async_frontier_entry_t, NULL)));

  iree_hal_memory_arena_t* arena = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc_aligned(host_allocator, total_size,
                                    iree_hardware_destructive_interference_size,
                                    /*offset=*/0, (void**)&arena));

  arena->capacity = options.capacity;
  arena->frontier_capacity = options.frontier_capacity;
  arena->host_allocator = host_allocator;
  arena->previous_frontier =
      (iree_async_frontier_t*)((uint8_t*)arena + previous_offset);
  arena->accumulator =
      (iree_async_frontier_t*)((uint8_t*)arena + accumulator_offset);
  arena->used = 0;
  arena->allocation_count = 0;
  arena->accumulator_tainted = false;
  arena->tainted = false;

  iree_async_frontier_initialize(arena->previous_frontier, 0);
  iree_async_frontier_initialize(arena->accumulator, 0);

  *out_arena = arena;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_memory_arena_free(iree_hal_memory_arena_t* arena) {
  if (!arena) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (arena->allocation_count > 0) {
    IREE_ASSERT(false, "arena free with %" PRIu32 " outstanding acquisitions",
                arena->allocation_count);
  }
  iree_allocator_t host_allocator = arena->host_allocator;
  iree_allocator_free_aligned(host_allocator, arena);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_memory_arena_acquire(
    iree_hal_memory_arena_t* arena, iree_device_size_t length,
    iree_device_size_t alignment,
    iree_hal_memory_arena_allocation_t* out_allocation) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_allocation);
  memset(out_allocation, 0, sizeof(*out_allocation));

  if (length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "length must be > 0");
  }
  if (alignment == 0 || !iree_device_size_is_power_of_two(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "alignment must be a power of two and > 0");
  }

  iree_device_size_t aligned_offset = iree_device_align(arena->used, alignment);
  iree_device_size_t new_used = 0;
  if (!iree_device_size_checked_add(aligned_offset, length, &new_used) ||
      new_used > arena->capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "arena exhausted: %" PRIu64 " bytes requested (aligned to %" PRIu64
        ") but only %" PRIu64 " of %" PRIu64 " bytes remain",
        (uint64_t)length, (uint64_t)alignment,
        (uint64_t)(arena->capacity - arena->used), (uint64_t)arena->capacity);
  }

  arena->used = new_used;
  arena->allocation_count++;

  out_allocation->offset = aligned_offset;
  out_allocation->death_frontier = (arena->previous_frontier->entry_count > 0)
                                       ? arena->previous_frontier
                                       : NULL;
  out_allocation->flags = arena->tainted ? IREE_HAL_MEMORY_ARENA_FLAG_TAINTED
                                         : IREE_HAL_MEMORY_ARENA_FLAG_NONE;
  return iree_ok_status();
}

void iree_hal_memory_arena_release(
    iree_hal_memory_arena_t* arena,
    const iree_async_frontier_t* death_frontier) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT(arena->allocation_count > 0,
              "arena release with no outstanding acquisitions");

  // JOIN the death frontier into the batch accumulator.
  if (death_frontier && death_frontier->entry_count > 0 &&
      !arena->accumulator_tainted) {
    if (!iree_async_frontier_merge(arena->accumulator, arena->frontier_capacity,
                                   death_frontier)) {
      // Overflow: mark accumulator tainted, zero the frontier.
      arena->accumulator_tainted = true;
      arena->accumulator->entry_count = 0;
    }
  }

  arena->allocation_count--;

  // When all acquisitions have been released, reset the arena.
  if (arena->allocation_count == 0) {
    // Copy the accumulated frontier (header + populated entries) into the
    // previous frontier slot. entry_count is uint8_t so this cannot overflow.
    iree_host_size_t frontier_copy_size =
        sizeof(iree_async_frontier_t) +
        (iree_host_size_t)arena->accumulator->entry_count *
            sizeof(iree_async_frontier_entry_t);
    memcpy(arena->previous_frontier, arena->accumulator, frontier_copy_size);

    arena->tainted = arena->accumulator_tainted;
    arena->accumulator_tainted = false;
    iree_async_frontier_initialize(arena->accumulator, 0);
    arena->used = 0;
  }
}

void iree_hal_memory_arena_query_stats(
    const iree_hal_memory_arena_t* arena,
    iree_hal_memory_arena_stats_t* out_stats) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_stats);
  out_stats->capacity = arena->capacity;
  out_stats->bytes_used = arena->used;
  out_stats->allocation_count = arena->allocation_count;
}
