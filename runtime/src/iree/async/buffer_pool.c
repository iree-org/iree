// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/buffer_pool.h"

#include "iree/base/internal/atomic_freelist.h"

//===----------------------------------------------------------------------===//
// Shared pool memory layout
//===----------------------------------------------------------------------===//

// Magic value: "IRBP" (IREE Buffer Pool) in little-endian.
#define IREE_ASYNC_SHARED_BUFFER_POOL_MAGIC ((uint32_t)0x50425249)

// Format version. Must match exactly between creator and opener.
#define IREE_ASYNC_SHARED_BUFFER_POOL_VERSION ((uint32_t)1)

// Immutable header at offset 0 of the shared memory region.
// Written once during create_shared, validated during open_shared.
// Occupies one cache line to avoid false sharing with the freelist state.
typedef struct iree_alignas(iree_hardware_destructive_interference_size)
    iree_async_shared_buffer_pool_header_t {
  uint32_t magic;
  uint32_t version;
  // Fixed-width types for cross-process safety (both processes are same
  // platform when sharing memory, but explicit widths are clearer).
  uint64_t buffer_size;
  uint32_t buffer_count;
  uint8_t reserved[44];
} iree_async_shared_buffer_pool_header_t;
static_assert(sizeof(iree_async_shared_buffer_pool_header_t) ==
                  iree_hardware_destructive_interference_size,
              "header must be exactly one cache line");

// Computes the shared memory layout for a given buffer count.
//
// The layout has three cache-line-isolated sections:
//   [header]   Immutable after creation (magic, version, geometry).
//   [freelist]  Hot contended atomic packed_state (generation|count|head).
//   [slots]     Per-buffer next-index array (contended per-slot, not per-line).
//
// Each section starts on a cache line boundary to prevent false sharing
// between the immutable header and the hot freelist, and between the freelist
// packed_state and the first slot entries.
static iree_status_t iree_async_shared_buffer_pool_compute_layout(
    iree_host_size_t buffer_count, iree_host_size_t* out_total,
    iree_host_size_t* out_freelist_offset, iree_host_size_t* out_slots_offset) {
  return IREE_STRUCT_LAYOUT(
      0, out_total,
      IREE_STRUCT_FIELD_ALIGNED(1, iree_async_shared_buffer_pool_header_t,
                                iree_hardware_destructive_interference_size,
                                NULL),
      IREE_STRUCT_FIELD_ALIGNED(1, iree_atomic_freelist_t,
                                iree_hardware_destructive_interference_size,
                                out_freelist_offset),
      IREE_STRUCT_FIELD_ALIGNED(buffer_count, iree_atomic_freelist_slot_t,
                                iree_hardware_destructive_interference_size,
                                out_slots_offset));
}

//===----------------------------------------------------------------------===//
// Pool structure
//===----------------------------------------------------------------------===//

struct iree_async_buffer_pool_t {
  // Allocator used for this pool struct's memory.
  iree_allocator_t allocator;

  // Registered region providing buffer memory and backend handles.
  // Retained reference; released in pool_free.
  iree_async_region_t* region;

  // Active freelist pointers used by acquire/release.
  // Local pools: point to embedded_freelist/embedded_slots below.
  // Shared pools: point into caller-provided shared memory.
  iree_atomic_freelist_t* freelist;
  iree_atomic_freelist_slot_t* freelist_slots;

  // Whether the freelist is in shared memory (affects free() behavior).
  // Shared pools skip the "all returned" assertion on free because other
  // processes may still hold leases, and do not deinitialize the freelist
  // because it lives in shared memory managed by the caller.
  bool is_shared;

  // Embedded freelist storage for local (non-shared) pools.
  // Shared pools allocate only sizeof(pool) without the FAM elements.
  iree_atomic_freelist_t embedded_freelist;
  iree_atomic_freelist_slot_t embedded_slots[];  // FAM, must be last.
};

//===----------------------------------------------------------------------===//
// Pool release callback
//===----------------------------------------------------------------------===//

// Release callback for pool leases. Pushes the buffer index back to the
// freelist, making it available for future acquire calls.
static void iree_async_buffer_pool_lease_release(void* context,
                                                 uint32_t index) {
  iree_async_buffer_pool_t* pool = (iree_async_buffer_pool_t*)context;
  iree_atomic_freelist_push(pool->freelist, pool->freelist_slots,
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

  // Point freelist accessors at the embedded storage.
  pool->freelist = &pool->embedded_freelist;
  pool->freelist_slots = pool->embedded_slots;
  pool->is_shared = false;

  // Initialize lock-free freelist with all buffers available.
  iree_status_t status = iree_atomic_freelist_initialize(
      pool->freelist_slots, buffer_count, pool->freelist);

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

  if (!pool->is_shared) {
    // Local pool: verify all buffers returned and deinitialize freelist.
    IREE_ATTRIBUTE_UNUSED iree_host_size_t buffer_count =
        pool->region->buffer_count;
    iree_host_size_t available = iree_atomic_freelist_count(pool->freelist);
    IREE_ASSERT(available == buffer_count,
                "freeing pool with %" PRIhsz " outstanding leases",
                buffer_count - available);
    iree_atomic_freelist_deinitialize(pool->freelist);
  }
  // Shared pool: do not assert (other processes may hold leases) and do not
  // deinitialize (freelist lives in caller-managed shared memory).

  // Release region reference.
  iree_async_region_release(pool->region);

  // Free pool struct.
  iree_allocator_t allocator = pool->allocator;
  iree_allocator_free(allocator, pool);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Shared (cross-process) pool lifecycle
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_buffer_pool_shared_storage_size(
    iree_host_size_t buffer_count, iree_host_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;
  return iree_async_shared_buffer_pool_compute_layout(
      buffer_count, out_size, /*out_freelist_offset=*/NULL,
      /*out_slots_offset=*/NULL);
}

// Allocates a pool struct for shared mode (no FAM, small fixed-size
// allocation) and binds its freelist pointers into the shared memory region.
static iree_status_t iree_async_buffer_pool_bind_shared(
    void* shared_memory, iree_async_region_t* region,
    iree_allocator_t allocator, iree_async_buffer_pool_t** out_pool) {
  // Compute field offsets within the shared memory region.
  iree_host_size_t freelist_offset = 0;
  iree_host_size_t slots_offset = 0;
  iree_host_size_t total = 0;
  IREE_RETURN_IF_ERROR(iree_async_shared_buffer_pool_compute_layout(
      region->buffer_count, &total, &freelist_offset, &slots_offset));

  iree_async_buffer_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  pool->allocator = allocator;
  pool->region = region;
  iree_async_region_retain(region);

  uint8_t* base = (uint8_t*)shared_memory;
  pool->freelist = (iree_atomic_freelist_t*)(base + freelist_offset);
  pool->freelist_slots = (iree_atomic_freelist_slot_t*)(base + slots_offset);
  pool->is_shared = true;

  *out_pool = pool;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_buffer_pool_create_shared(
    void* shared_memory, iree_host_size_t shared_memory_size,
    iree_async_region_t* region, iree_allocator_t allocator,
    iree_async_buffer_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(shared_memory);
  IREE_ASSERT_ARGUMENT(region);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_pool = NULL;

  iree_host_size_t buffer_count = region->buffer_count;
  iree_host_size_t buffer_size = region->buffer_size;

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

  iree_host_size_t required_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_async_buffer_pool_shared_storage_size(buffer_count, &required_size));
  if (shared_memory_size < required_size) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared_memory_size %" PRIhsz
                            " < required %" PRIhsz,
                            shared_memory_size, required_size);
  }

  // Validate shared_memory alignment. The header and freelist require
  // cache-line alignment for correct atomic operations (especially on ARM
  // where unaligned atomics may fault).
  if (IREE_UNLIKELY((uintptr_t)shared_memory %
                        iree_hardware_destructive_interference_size !=
                    0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "shared_memory pointer %p is not aligned to %" PRIhsz " bytes",
        shared_memory,
        (iree_host_size_t)iree_hardware_destructive_interference_size);
  }

  // Zero the entire shared region. Header fields (except magic) are written
  // before freelist initialization; magic is written last as a commit step
  // so that openers never see a valid-looking header with uninitialized
  // freelist state.
  memset(shared_memory, 0, required_size);
  iree_async_shared_buffer_pool_header_t* header =
      (iree_async_shared_buffer_pool_header_t*)shared_memory;
  header->version = IREE_ASYNC_SHARED_BUFFER_POOL_VERSION;
  header->buffer_size = (uint64_t)buffer_size;
  header->buffer_count = (uint32_t)buffer_count;

  // Allocate the process-local pool handle and bind into shared memory.
  iree_async_buffer_pool_t* pool = NULL;
  iree_status_t status = iree_async_buffer_pool_bind_shared(
      shared_memory, region, allocator, &pool);

  // Initialize the freelist in shared memory with all indices available.
  if (iree_status_is_ok(status)) {
    status = iree_atomic_freelist_initialize(pool->freelist_slots, buffer_count,
                                             pool->freelist);
  }

  // Write magic last: this is the commit step. Only after the freelist is
  // fully initialized does the shared region become valid for openers.
  if (iree_status_is_ok(status)) {
    header->magic = IREE_ASYNC_SHARED_BUFFER_POOL_MAGIC;
    *out_pool = pool;
  } else {
    if (pool) {
      iree_async_region_release(region);
      iree_allocator_free(allocator, pool);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_async_buffer_pool_open_shared(
    void* shared_memory, iree_host_size_t shared_memory_size,
    iree_async_region_t* region, iree_allocator_t allocator,
    iree_async_buffer_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(shared_memory);
  IREE_ASSERT_ARGUMENT(region);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_pool = NULL;

  // Validate shared_memory alignment (same requirement as create_shared).
  if (IREE_UNLIKELY((uintptr_t)shared_memory %
                        iree_hardware_destructive_interference_size !=
                    0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "shared_memory pointer %p is not aligned to %" PRIhsz " bytes",
        shared_memory,
        (iree_host_size_t)iree_hardware_destructive_interference_size);
  }

  // Validate that we can at least read the header.
  if (shared_memory_size < sizeof(iree_async_shared_buffer_pool_header_t)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "shared_memory_size %" PRIhsz " too small for header (%" PRIhsz ")",
        shared_memory_size,
        (iree_host_size_t)sizeof(iree_async_shared_buffer_pool_header_t));
  }

  // Validate header fields.
  const iree_async_shared_buffer_pool_header_t* header =
      (const iree_async_shared_buffer_pool_header_t*)shared_memory;
  if (header->magic != IREE_ASYNC_SHARED_BUFFER_POOL_MAGIC) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "shared pool magic mismatch: expected 0x%08X, got 0x%08X",
        IREE_ASYNC_SHARED_BUFFER_POOL_MAGIC, header->magic);
  }
  if (header->version != IREE_ASYNC_SHARED_BUFFER_POOL_VERSION) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "shared pool version mismatch: expected %" PRIu32 ", got %" PRIu32,
        IREE_ASYNC_SHARED_BUFFER_POOL_VERSION, header->version);
  }

  // Validate buffer geometry matches the region.
  if (header->buffer_size != (uint64_t)region->buffer_size) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "shared pool buffer_size mismatch: header has %" PRIu64
        ", region has %" PRIhsz,
        header->buffer_size, region->buffer_size);
  }
  if (header->buffer_count != (uint32_t)region->buffer_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "shared pool buffer_count mismatch: header has %" PRIu32
        ", region has %" PRIu32,
        header->buffer_count, (uint32_t)region->buffer_count);
  }

  // Validate freelist constraints.
  if (header->buffer_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared pool has zero buffer count");
  }
  if (header->buffer_count > IREE_ATOMIC_FREELIST_MAX_COUNT) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "shared pool buffer_count %" PRIu32 " exceeds maximum %" PRIhsz,
        header->buffer_count, (iree_host_size_t)IREE_ATOMIC_FREELIST_MAX_COUNT);
  }

  // Validate shared memory is large enough for the header's buffer count.
  iree_host_size_t required_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_buffer_pool_shared_storage_size(header->buffer_count,
                                                     &required_size));
  if (shared_memory_size < required_size) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared_memory_size %" PRIhsz " < required %" PRIhsz
                            " for %" PRIu32 " buffers",
                            shared_memory_size, required_size,
                            header->buffer_count);
  }

  // Allocate the process-local pool handle and bind into shared memory.
  // Does NOT reinitialize the freelist — binds to existing state.
  iree_status_t status = iree_async_buffer_pool_bind_shared(
      shared_memory, region, allocator, out_pool);
  IREE_TRACE_ZONE_END(z0);
  return status;
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
  if (!iree_atomic_freelist_try_pop(pool->freelist, pool->freelist_slots,
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
  return iree_atomic_freelist_count(pool->freelist);
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
