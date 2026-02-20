// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/buffer_pool.h"

#include "iree/base/internal/atomic_freelist.h"

//===----------------------------------------------------------------------===//
// Pool structure
//===----------------------------------------------------------------------===//

struct iree_async_buffer_pool_t {
  // Allocator used for this pool's memory.
  iree_allocator_t allocator;

  // Registered region providing buffer memory and backend handles.
  // Retained reference; released in pool_free.
  iree_async_region_t* region;

  // Lock-free freelist for buffer index management.
  // Thread-safe for concurrent acquire/release from multiple threads.
  iree_atomic_freelist_t freelist;

  // Freelist slots: maps index -> next index in the freelist chain.
  // This is the "next pointer" array used by the atomic freelist.
  // Flexible array member - must be last.
  iree_atomic_freelist_slot_t freelist_slots[];
};

//===----------------------------------------------------------------------===//
// Pool release callback
//===----------------------------------------------------------------------===//

// Release callback for pool leases. Pushes the buffer index back to the
// freelist, making it available for future acquire calls.
static void iree_async_buffer_pool_lease_release(void* context,
                                                 uint32_t index) {
  iree_async_buffer_pool_t* pool = (iree_async_buffer_pool_t*)context;
  iree_atomic_freelist_push(&pool->freelist, pool->freelist_slots,
                            (uint16_t)index);
}

//===----------------------------------------------------------------------===//
// Pool lifecycle
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_buffer_pool_allocate(
    iree_async_region_t* region, iree_allocator_t allocator,
    iree_async_buffer_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(region);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_pool = NULL;

  // Get buffer configuration from region's portable buffer fields.
  // These are set by register_slab for indexed buffer regions.
  iree_host_size_t buffer_count = region->buffer_count;

  if (buffer_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region has zero buffer count");
  }
  if (buffer_count > IREE_ATOMIC_FREELIST_MAX_COUNT) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_count %" PRIhsz " exceeds maximum %" PRIhsz,
                            buffer_count,
                            (iree_host_size_t)IREE_ATOMIC_FREELIST_MAX_COUNT);
  }

  // Calculate single allocation size for pool struct + freelist slots FAM.
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(sizeof(iree_async_buffer_pool_t), &total_size,
                             IREE_STRUCT_FIELD_FAM(
                                 buffer_count, iree_atomic_freelist_slot_t)));

  // Allocate pool structure (includes freelist slots FAM).
  iree_async_buffer_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&pool));
  memset(pool, 0, total_size);
  pool->allocator = allocator;
  pool->region = region;
  iree_async_region_retain(region);

  // Initialize lock-free freelist with all buffers available.
  iree_status_t status = iree_atomic_freelist_initialize(
      pool->freelist_slots, buffer_count, &pool->freelist);

  if (iree_status_is_ok(status)) {
    *out_pool = pool;
  } else {
    iree_async_region_release(region);
    iree_allocator_free(allocator, pool);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_async_buffer_pool_free(
    iree_async_buffer_pool_t* pool) {
  if (!pool) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ATTRIBUTE_UNUSED iree_host_size_t buffer_count =
      pool->region->buffer_count;

  // Debug check: all buffers should be returned.
  iree_host_size_t available = iree_atomic_freelist_count(&pool->freelist);
  IREE_ASSERT(available == buffer_count,
              "freeing pool with %" PRIhsz " outstanding leases",
              buffer_count - available);

  // Deinitialize freelist.
  iree_atomic_freelist_deinitialize(&pool->freelist);

  // Release region reference.
  iree_async_region_release(pool->region);

  // Free pool struct.
  iree_allocator_t allocator = pool->allocator;
  iree_allocator_free(allocator, pool);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Acquire
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_buffer_pool_acquire(
    iree_async_buffer_pool_t* pool, iree_async_buffer_lease_t* out_lease) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_lease);

  iree_host_size_t buffer_size = pool->region->buffer_size;
  IREE_ATTRIBUTE_UNUSED iree_host_size_t buffer_count =
      pool->region->buffer_count;

  // Try to pop from lock-free freelist.
  uint16_t index;
  if (!iree_atomic_freelist_try_pop(&pool->freelist, pool->freelist_slots,
                                    &index)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "buffer pool exhausted (0 of %" PRIhsz " available)", buffer_count);
  }

  // Build lease with polymorphic release callback.
  out_lease->span = iree_async_span_make(
      pool->region, (iree_host_size_t)index * buffer_size, buffer_size);
  out_lease->release = (iree_async_buffer_recycle_callback_t){
      .fn = iree_async_buffer_pool_lease_release,
      .user_data = pool,
  };
  out_lease->buffer_index = (iree_async_buffer_index_t)index;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Query
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_host_size_t
iree_async_buffer_pool_available(const iree_async_buffer_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return iree_atomic_freelist_count(&pool->freelist);
}

IREE_API_EXPORT iree_host_size_t
iree_async_buffer_pool_capacity(const iree_async_buffer_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return pool->region->buffer_count;
}

IREE_API_EXPORT iree_host_size_t
iree_async_buffer_pool_buffer_size(const iree_async_buffer_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return pool->region->buffer_size;
}

IREE_API_EXPORT iree_async_region_t* iree_async_buffer_pool_region(
    const iree_async_buffer_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return pool->region;
}
