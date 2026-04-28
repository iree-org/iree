// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tlsf_pool.h"

#include "iree/async/frontier.h"
#include "iree/async/notification.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/memory/tracing.h"

enum {
  IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY = 4,
};

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tlsf_pool_release_node_t {
  // Intrusive next pointer in pool->pending_release_head or
  // pool->release_node_free_head.
  struct iree_hal_tlsf_pool_release_node_t* next;

  // Stable slab object holding |block_index|.
  struct iree_hal_tlsf_pool_slab_t* slab;

  // Index of |slab| in pool->slabs.
  uint16_t slab_index;

  // TLSF block handle owned by this reservation.
  iree_hal_memory_tlsf_block_index_t block_index;
} iree_hal_tlsf_pool_release_node_t;

typedef struct iree_hal_tlsf_pool_slab_t {
  // Offset allocator for the address range backed by |slab|.
  iree_hal_memory_tlsf_t tlsf;

  // Physical memory backing |tlsf|'s offset range.
  iree_hal_slab_t slab;
} iree_hal_tlsf_pool_slab_t;

typedef struct iree_hal_tlsf_pool_allocation_t {
  // Index into pool->slabs identifying the owning TLSF instance.
  uint16_t slab_index;

  // Allocation returned by the owning TLSF instance.
  iree_hal_memory_tlsf_allocation_t allocation;
} iree_hal_tlsf_pool_allocation_t;

typedef struct iree_hal_tlsf_pool_t {
  // Base pool resource for vtable dispatch and ref counting.
  iree_hal_resource_t resource;

  // Provider used to acquire additional slabs as the pool grows.
  iree_hal_slab_provider_t* slab_provider;

  // Notification signaled when a reservation release may unblock waiters.
  iree_async_notification_t* notification;

  // Guards TLSF mutation and the slab array.
  iree_slim_mutex_t mutex;

  // Template options used when initializing each new TLSF slab.
  iree_hal_memory_tlsf_options_t slab_options;

  // Fixed byte length requested for newly grown slabs.
  iree_device_size_t slab_length;

  // Dynamic array of committed slab pointers. Protected by |mutex|.
  iree_hal_tlsf_pool_slab_t** slabs;

  // Number of initialized entries in |slabs|. Protected by |mutex|.
  uint32_t slab_count;

  // Allocated entry capacity of |slabs|. Protected by |mutex|.
  iree_host_size_t slab_capacity;

  // Preferred slab to try first for the next allocation. Protected by |mutex|.
  uint16_t preferred_slab_index;

  // Bounded set of slabs that recently received releases. Protected by |mutex|.
  uint16_t
      reuse_candidate_slab_indices[IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY];

  // Valid entries in |reuse_candidate_slab_indices|. Protected by |mutex|.
  uint8_t reuse_candidate_slab_count;

  // Next candidate slot to overwrite when the bounded set is full.
  uint8_t reuse_candidate_slab_cursor;

  // Approximate committed bytes across all slabs for lock-free stats queries.
  iree_atomic_int64_t bytes_committed;

  // Approximate committed slab count for lock-free stats queries.
  iree_atomic_int32_t committed_slab_count;

  // Pending release nodes pushed by queue-retirement paths.
  iree_atomic_intptr_t pending_release_head;

  // Free list of release nodes ready for reuse. Protected by |mutex|.
  iree_hal_tlsf_pool_release_node_t* release_node_free_head;

  // Optional completion predicate for try-before-fence reuse.
  iree_hal_pool_epoch_query_t epoch_query;

  // Host allocator used for pool metadata.
  iree_allocator_t host_allocator;

  // Stable named-memory stream for logical reservations from this pool.
  iree_hal_memory_trace_t trace;

  // Memory type properties provided by |slab_provider|.
  iree_hal_memory_type_t memory_type;

  // Buffer usages supported by |slab_provider|.
  iree_hal_buffer_usage_t supported_usage;

  // Logical byte budget for live reservations. 0 means unlimited.
  iree_device_size_t budget_limit;

  // Byte size of each release node including inline frontier storage.
  iree_host_size_t release_node_size;

  // Byte offset of inline frontier storage within a release node.
  iree_host_size_t release_frontier_offset;

  // Scratch storage used under |mutex| to hold rejected block indices while a
  // reserve call keeps searching for a satisfiable block.
  iree_hal_memory_tlsf_block_index_t* rejected_block_indices;
  iree_host_size_t rejected_block_capacity;

  // Approximate live reservation bytes for lock-free stats queries.
  iree_atomic_int64_t bytes_reserved;

  // Approximate live reservation count for lock-free stats queries.
  iree_atomic_int32_t reservation_count;

  // Total successful acquire_reservation() calls.
  iree_atomic_int64_t reserve_count;

  // Total release_reservation() calls.
  iree_atomic_int64_t release_count;

  // Reserves that hit frontier-dominated reuse.
  iree_atomic_int64_t reuse_count;

  // Reserves where dominance check failed.
  iree_atomic_int64_t reuse_miss_count;

  // Reserves from fresh (never-used) blocks.
  iree_atomic_int64_t fresh_count;

  // Reserves that returned EXHAUSTED.
  iree_atomic_int64_t exhausted_count;

  // Reserves that returned OVER_BUDGET.
  iree_atomic_int64_t over_budget_count;

  // Reserves that returned NEEDS_WAIT.
  iree_atomic_int64_t wait_count;
} iree_hal_tlsf_pool_t;

typedef struct iree_hal_tlsf_pool_buffer_state_t {
  // Borrowed from the wrapped buffer's creator. Pool owners must keep the pool
  // alive until all buffers sourced from it are destroyed.
  iree_hal_pool_t* pool;

  // Reservation returned to |pool| when the buffer is destroyed.
  iree_hal_pool_reservation_t reservation;

  // Host allocator used for this state object.
  iree_allocator_t host_allocator;
} iree_hal_tlsf_pool_buffer_state_t;

static const iree_hal_pool_vtable_t iree_hal_tlsf_pool_vtable;
static void iree_hal_tlsf_pool_destroy(iree_hal_pool_t* base_pool);

static const char* IREE_HAL_TLSF_POOL_TRACE_ID = "iree-hal-tlsf-pool";

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

static inline iree_async_frontier_t* iree_hal_tlsf_pool_release_node_frontier(
    const iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node) {
  return (iree_async_frontier_t*)((uint8_t*)node +
                                  pool->release_frontier_offset);
}

static void iree_hal_tlsf_pool_push_pending_release(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node) {
  intptr_t expected =
      iree_atomic_load(&pool->pending_release_head, iree_memory_order_relaxed);
  do {
    node->next = (iree_hal_tlsf_pool_release_node_t*)expected;
  } while (!iree_atomic_compare_exchange_weak(
      &pool->pending_release_head, &expected, (intptr_t)node,
      iree_memory_order_release, iree_memory_order_relaxed));
}

static iree_hal_tlsf_pool_release_node_t*
iree_hal_tlsf_pool_take_pending_releases(iree_hal_tlsf_pool_t* pool) {
  return (iree_hal_tlsf_pool_release_node_t*)iree_atomic_exchange(
      &pool->pending_release_head, 0, iree_memory_order_acquire);
}

static bool iree_hal_tlsf_pool_try_charge_reservation(
    iree_hal_tlsf_pool_t* pool, iree_device_size_t length) {
  if (pool->budget_limit == 0) {
    iree_atomic_fetch_add(&pool->bytes_reserved, (int64_t)length,
                          iree_memory_order_relaxed);
    return true;
  }
  int64_t expected =
      iree_atomic_load(&pool->bytes_reserved, iree_memory_order_relaxed);
  for (;;) {
    const iree_device_size_t current = (iree_device_size_t)expected;
    if (current > pool->budget_limit || length > pool->budget_limit - current) {
      return false;
    }
    const int64_t desired = (int64_t)(current + length);
    if (iree_atomic_compare_exchange_weak(&pool->bytes_reserved, &expected,
                                          desired, iree_memory_order_relaxed,
                                          iree_memory_order_relaxed)) {
      return true;
    }
  }
}

static void iree_hal_tlsf_pool_uncharge_reservation(iree_hal_tlsf_pool_t* pool,
                                                    iree_device_size_t length) {
  iree_atomic_fetch_add(&pool->bytes_reserved, -(int64_t)length,
                        iree_memory_order_relaxed);
}

static bool iree_hal_tlsf_pool_adjust_charged_reservation(
    iree_hal_tlsf_pool_t* pool, iree_device_size_t* charged_length,
    iree_device_size_t actual_length) {
  if (actual_length <= *charged_length) {
    iree_hal_tlsf_pool_uncharge_reservation(pool,
                                            *charged_length - actual_length);
    *charged_length = actual_length;
    return true;
  }
  const iree_device_size_t additional_length = actual_length - *charged_length;
  if (!iree_hal_tlsf_pool_try_charge_reservation(pool, additional_length)) {
    return false;
  }
  *charged_length = actual_length;
  return true;
}

static void iree_hal_tlsf_pool_note_reuse_candidate(iree_hal_tlsf_pool_t* pool,
                                                    uint16_t slab_index)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  for (uint8_t i = 0; i < pool->reuse_candidate_slab_count; ++i) {
    if (pool->reuse_candidate_slab_indices[i] == slab_index) return;
  }
  if (pool->reuse_candidate_slab_count <
      IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY) {
    pool->reuse_candidate_slab_indices[pool->reuse_candidate_slab_count++] =
        slab_index;
    return;
  }
  pool->reuse_candidate_slab_indices[pool->reuse_candidate_slab_cursor] =
      slab_index;
  pool->reuse_candidate_slab_cursor =
      (uint8_t)((pool->reuse_candidate_slab_cursor + 1) %
                IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY);
}

static iree_status_t iree_hal_tlsf_pool_acquire_release_node(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t** out_node)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* node = pool->release_node_free_head;
  if (node) {
    pool->release_node_free_head = node->next;
  } else {
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_tlsf_pool_grow_release_nodes");
    iree_status_t status = iree_allocator_malloc(
        pool->host_allocator, pool->release_node_size, (void**)&node);
    IREE_TRACE_ZONE_END(z0);
    if (!iree_status_is_ok(status)) return status;
  }
  node->next = NULL;
  node->slab = NULL;
  node->slab_index = 0;
  node->block_index = 0;
  iree_async_frontier_initialize(
      iree_hal_tlsf_pool_release_node_frontier(pool, node), 0);
  *out_node = node;
  return iree_ok_status();
}

static void iree_hal_tlsf_pool_recycle_release_node(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  if (!node) return;
  node->next = pool->release_node_free_head;
  pool->release_node_free_head = node;
}

static void iree_hal_tlsf_pool_free_release_nodes(iree_hal_tlsf_pool_t* pool)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* node = pool->release_node_free_head;
  pool->release_node_free_head = NULL;
  while (node) {
    iree_hal_tlsf_pool_release_node_t* next = node->next;
    iree_allocator_free(pool->host_allocator, node);
    node = next;
  }
}

static void iree_hal_tlsf_pool_drain_pending_releases(
    iree_hal_tlsf_pool_t* pool)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* node =
      iree_hal_tlsf_pool_take_pending_releases(pool);
  while (node) {
    iree_hal_tlsf_pool_release_node_t* next = node->next;
    iree_async_frontier_t* death_frontier =
        iree_hal_tlsf_pool_release_node_frontier(pool, node);
    iree_hal_tlsf_pool_slab_t* slab = node->slab;
    const uint16_t slab_index = node->slab_index;
    iree_hal_memory_tlsf_free(
        &slab->tlsf, node->block_index,
        death_frontier->entry_count > 0 ? death_frontier : NULL);
    if (death_frontier->entry_count == 0) {
      pool->preferred_slab_index = slab_index;
    } else {
      iree_hal_tlsf_pool_note_reuse_candidate(pool, slab_index);
    }
    iree_hal_tlsf_pool_recycle_release_node(pool, node);
    node = next;
  }
}

static bool iree_hal_tlsf_pool_frontier_is_satisfied(
    const iree_hal_tlsf_pool_t* pool,
    const iree_async_frontier_t* requester_frontier,
    const iree_async_frontier_t* death_frontier,
    iree_hal_memory_tlsf_block_flags_t block_flags) {
  if (!death_frontier) {
    return (block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED) == 0;
  }
  if (block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED) {
    return false;
  }
  if (requester_frontier) {
    const iree_async_frontier_comparison_t comparison =
        iree_async_frontier_compare(requester_frontier, death_frontier);
    if (comparison == IREE_ASYNC_FRONTIER_AFTER ||
        comparison == IREE_ASYNC_FRONTIER_EQUAL) {
      return true;
    }
  }
  if (!pool->epoch_query.fn) return false;

  iree_host_size_t requester_index = 0;
  for (uint8_t i = 0; i < death_frontier->entry_count; ++i) {
    const iree_async_axis_t axis = death_frontier->entries[i].axis;
    const uint64_t epoch = death_frontier->entries[i].epoch;
    while (requester_frontier &&
           requester_index < requester_frontier->entry_count &&
           requester_frontier->entries[requester_index].axis < axis) {
      ++requester_index;
    }
    if (requester_frontier &&
        requester_index < requester_frontier->entry_count &&
        requester_frontier->entries[requester_index].axis == axis &&
        requester_frontier->entries[requester_index].epoch >= epoch) {
      continue;
    }
    if (!pool->epoch_query.fn(pool->epoch_query.user_data, axis, epoch)) {
      return false;
    }
  }
  return true;
}

static void iree_hal_tlsf_pool_restore_rejected_blocks(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_slab_t* slab,
    uint32_t rejected_block_count)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  for (uint32_t i = 0; i < rejected_block_count; ++i) {
    iree_hal_memory_tlsf_restore(&slab->tlsf, pool->rejected_block_indices[i]);
  }
}

static iree_status_t iree_hal_tlsf_pool_ensure_rejected_capacity(
    iree_hal_tlsf_pool_t* pool, iree_host_size_t capacity)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  if (pool->rejected_block_capacity >= capacity) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
      pool->host_allocator, capacity, sizeof(*pool->rejected_block_indices),
      &pool->rejected_block_capacity, (void**)&pool->rejected_block_indices));
  return iree_ok_status();
}

static iree_status_t iree_hal_tlsf_pool_aligned_slab_length(
    const iree_hal_tlsf_pool_t* pool, iree_device_size_t* out_slab_length) {
  const iree_device_size_t alignment =
      pool->slab_options.alignment
          ? pool->slab_options.alignment
          : (iree_device_size_t)IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  if (pool->slab_length > IREE_DEVICE_SIZE_MAX - (alignment - 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "slab length %" PRIdsz
                            " overflows when aligned to %" PRIdsz,
                            pool->slab_length, alignment);
  }
  const iree_device_size_t required_length =
      iree_device_align(pool->slab_length, alignment);
  *out_slab_length = required_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_tlsf_pool_append_slab(iree_hal_tlsf_pool_t* pool,
                                                    uint16_t* out_slab_index)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  *out_slab_index = 0;
  if (pool->slab_count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "TLSF pool reached maximum slab count (%" PRIu32
                            ")",
                            (uint32_t)UINT16_MAX);
  }
  if (pool->slab_count >= pool->slab_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        pool->host_allocator, pool->slab_count + 1, sizeof(*pool->slabs),
        &pool->slab_capacity, (void**)&pool->slabs));
  }

  iree_device_size_t slab_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_tlsf_pool_aligned_slab_length(pool, &slab_length));

  iree_hal_tlsf_pool_slab_t* slab_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      pool->host_allocator, sizeof(*slab_entry), (void**)&slab_entry));
  memset(slab_entry, 0, sizeof(*slab_entry));

  iree_hal_slab_t slab;
  iree_status_t status = iree_hal_slab_provider_acquire_slab(
      pool->slab_provider, slab_length, &slab);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(pool->host_allocator, slab_entry);
    return status;
  }

  iree_hal_memory_tlsf_options_t tlsf_options = pool->slab_options;
  tlsf_options.range_length = slab.length;
  status = iree_hal_memory_tlsf_initialize(tlsf_options, pool->host_allocator,
                                           &slab_entry->tlsf);
  if (!iree_status_is_ok(status)) {
    iree_hal_slab_provider_release_slab(pool->slab_provider, &slab);
    iree_allocator_free(pool->host_allocator, slab_entry);
    return status;
  }
  slab_entry->slab = slab;

  const uint16_t slab_index = (uint16_t)pool->slab_count;
  pool->slabs[slab_index] = slab_entry;
  ++pool->slab_count;
  pool->preferred_slab_index = slab_index;
  iree_atomic_store(&pool->committed_slab_count, (int32_t)pool->slab_count,
                    iree_memory_order_release);
  iree_atomic_fetch_add(&pool->bytes_committed, (int64_t)slab.length,
                        iree_memory_order_relaxed);
  *out_slab_index = slab_index;
  return iree_ok_status();
}

static void iree_hal_tlsf_pool_deinitialize_slabs(iree_hal_tlsf_pool_t* pool) {
  for (uint32_t i = 0; i < pool->slab_count; ++i) {
    iree_hal_tlsf_pool_slab_t* slab = pool->slabs[i];
    if (!slab) continue;
    if (slab->tlsf.block_storage) {
      iree_hal_memory_tlsf_deinitialize(&slab->tlsf);
    }
    if (slab->slab.length > 0) {
      iree_hal_slab_provider_release_slab(pool->slab_provider, &slab->slab);
    }
    iree_allocator_free(pool->host_allocator, slab);
  }
  iree_allocator_free(pool->host_allocator, pool->slabs);
  pool->slabs = NULL;
  pool->slab_count = 0;
  pool->slab_capacity = 0;
  pool->preferred_slab_index = 0;
  pool->reuse_candidate_slab_count = 0;
  pool->reuse_candidate_slab_cursor = 0;
  iree_atomic_store(&pool->committed_slab_count, 0, iree_memory_order_release);
  iree_atomic_store(&pool->bytes_committed, 0, iree_memory_order_release);
}

static iree_status_t iree_hal_tlsf_pool_return_allocation(
    iree_hal_tlsf_pool_t* pool,
    const iree_hal_tlsf_pool_allocation_t* pool_allocation,
    iree_hal_pool_acquire_result_t result,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_slab_t* slab = pool->slabs[pool_allocation->slab_index];
  const iree_hal_memory_tlsf_allocation_t* allocation =
      &pool_allocation->allocation;
  iree_hal_tlsf_pool_release_node_t* release_node = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_tlsf_pool_acquire_release_node(pool, &release_node));
  release_node->slab = slab;
  release_node->slab_index = pool_allocation->slab_index;
  release_node->block_index = allocation->block_index;
  pool->preferred_slab_index = pool_allocation->slab_index;

  memset(out_reservation, 0, sizeof(*out_reservation));
  out_reservation->offset = allocation->offset;
  out_reservation->length = allocation->length;
  out_reservation->block_handle = (uint64_t)(uintptr_t)release_node;
  out_reservation->slab_index = pool_allocation->slab_index;

  memset(out_info, 0, sizeof(*out_info));

  iree_atomic_fetch_add(&pool->reservation_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, 1, iree_memory_order_relaxed);
  switch (result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
      iree_atomic_fetch_add(&pool->reuse_count, 1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      iree_atomic_fetch_add(&pool->fresh_count, 1, iree_memory_order_relaxed);
      break;
    default:
      IREE_ASSERT(false, "invalid successful TLSF pool result: %u", result);
      break;
  }
  iree_hal_memory_trace_alloc(
      &pool->trace, (uint8_t*)slab->slab.base_ptr + allocation->offset,
      allocation->length);
  *out_result = result;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Create / Destroy
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_tlsf_pool_create(
    iree_hal_tlsf_pool_options_t options,
    iree_hal_slab_provider_t* slab_provider,
    iree_async_notification_t* notification,
    iree_hal_pool_epoch_query_t epoch_query, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(slab_provider);
  IREE_ASSERT_ARGUMENT(notification);
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (options.tlsf_options.range_length == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "range_length must be > 0");
  }

  iree_hal_tlsf_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  iree_hal_resource_initialize(&iree_hal_tlsf_pool_vtable, &pool->resource);
  iree_slim_mutex_initialize(&pool->mutex);
  iree_atomic_store(&pool->pending_release_head, 0, iree_memory_order_relaxed);
  pool->host_allocator = host_allocator;
  pool->epoch_query = epoch_query;
  pool->budget_limit = options.budget_limit;
  pool->slab_options = options.tlsf_options;
  pool->slab_length = options.tlsf_options.range_length;
  iree_atomic_store(&pool->bytes_committed, 0, iree_memory_order_relaxed);
  iree_atomic_store(&pool->committed_slab_count, 0, iree_memory_order_relaxed);

  iree_hal_slab_provider_retain(slab_provider);
  pool->slab_provider = slab_provider;
  iree_async_notification_retain(notification);
  pool->notification = notification;
  iree_hal_slab_provider_query_properties(slab_provider, &pool->memory_type,
                                          &pool->supported_usage);

  iree_status_t status = iree_hal_memory_trace_initialize_pool(
      options.trace_name, IREE_HAL_TLSF_POOL_TRACE_ID, host_allocator,
      &pool->trace);
  if (iree_status_is_ok(status)) {
    uint16_t initial_slab_index = 0;
    iree_slim_mutex_lock(&pool->mutex);
    status = iree_hal_tlsf_pool_append_slab(pool, &initial_slab_index);
    iree_slim_mutex_unlock(&pool->mutex);
  }
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_tlsf_pool_release_node_t), &pool->release_node_size,
        IREE_STRUCT_FIELD_ALIGNED(1, iree_async_frontier_t,
                                  iree_alignof(iree_async_frontier_entry_t),
                                  &pool->release_frontier_offset),
        IREE_STRUCT_FIELD(pool->slabs[0]->tlsf.frontier_capacity,
                          iree_async_frontier_entry_t, NULL));
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_tlsf_pool_destroy((iree_hal_pool_t*)pool);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_pool = (iree_hal_pool_t*)pool;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_tlsf_pool_destroy(iree_hal_pool_t* base_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;

  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_tlsf_pool_drain_pending_releases(pool);
  iree_hal_tlsf_pool_free_release_nodes(pool);
  iree_slim_mutex_unlock(&pool->mutex);

  iree_hal_tlsf_pool_deinitialize_slabs(pool);
  iree_slim_mutex_deinitialize(&pool->mutex);
  iree_allocator_free(pool->host_allocator, pool->rejected_block_indices);
  iree_hal_memory_trace_deinitialize(&pool->trace);
  iree_async_notification_release(pool->notification);
  iree_hal_slab_provider_release(pool->slab_provider);
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_allocator_free(host_allocator, pool);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Reserve / Release
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_tlsf_pool_try_acquire_from_slab(
    iree_hal_tlsf_pool_t* pool, uint16_t slab_index, iree_device_size_t size,
    const iree_async_frontier_t* requester_frontier, bool record_reuse_miss,
    iree_hal_tlsf_pool_allocation_t* out_allocation,
    iree_hal_pool_acquire_result_t* out_result)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_status_t status = iree_ok_status();
  iree_hal_tlsf_pool_slab_t* slab = pool->slabs[slab_index];
  uint32_t rejected_block_count = 0;
  *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;

  while (iree_status_is_ok(status)) {
    iree_hal_memory_tlsf_allocation_t allocation;
    iree_hal_memory_tlsf_allocate_result_t allocation_result =
        IREE_HAL_MEMORY_TLSF_ALLOCATE_EXHAUSTED;
    status = iree_hal_memory_tlsf_try_allocate(&slab->tlsf, size, &allocation,
                                               &allocation_result);
    if (!iree_status_is_ok(status) ||
        allocation_result == IREE_HAL_MEMORY_TLSF_ALLOCATE_EXHAUSTED) {
      break;
    }

    if (!iree_hal_tlsf_pool_frontier_is_satisfied(pool, requester_frontier,
                                                  allocation.death_frontier,
                                                  allocation.block_flags)) {
      if (record_reuse_miss) {
        iree_atomic_fetch_add(&pool->reuse_miss_count, 1,
                              iree_memory_order_relaxed);
      }
      status = iree_hal_tlsf_pool_ensure_rejected_capacity(
          pool, (iree_host_size_t)rejected_block_count + 1);
      if (!iree_status_is_ok(status)) {
        iree_hal_memory_tlsf_restore(&slab->tlsf, allocation.block_index);
        break;
      }
      pool->rejected_block_indices[rejected_block_count++] =
          allocation.block_index;
      continue;
    }

    out_allocation->slab_index = slab_index;
    out_allocation->allocation = allocation;
    *out_result = allocation.death_frontier ? IREE_HAL_POOL_ACQUIRE_OK
                                            : IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    break;
  }

  iree_hal_tlsf_pool_restore_rejected_blocks(pool, slab, rejected_block_count);
  return status;
}

static iree_status_t iree_hal_tlsf_pool_acquire_reservation(
    iree_hal_pool_t* base_pool, iree_device_size_t size,
    iree_device_size_t alignment,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  (void)flags;

  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation size must be > 0");
  }
  if (alignment == 0 || !iree_device_size_is_power_of_two(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment (%" PRIdsz
                            ") must be a power of two > 0",
                            alignment);
  }
  const iree_device_size_t pool_alignment =
      pool->slab_options.alignment
          ? pool->slab_options.alignment
          : (iree_device_size_t)IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  if (alignment > pool_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment %" PRIdsz
                            " exceeds TLSF pool alignment %" PRIdsz,
                            alignment, pool_alignment);
  }
  if (size > pool->slab_length) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  iree_device_size_t charged_length = size;
  if (!iree_hal_tlsf_pool_try_charge_reservation(pool, charged_length)) {
    iree_atomic_fetch_add(&pool->over_budget_count, 1,
                          iree_memory_order_relaxed);
    memset(out_reservation, 0, sizeof(*out_reservation));
    memset(out_info, 0, sizeof(*out_info));
    *out_result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
    return iree_ok_status();
  }

  iree_hal_tlsf_pool_allocation_t selected_allocation;
  iree_hal_pool_acquire_result_t selected_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  bool has_selected_allocation = false;

  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_tlsf_pool_drain_pending_releases(pool);

  iree_status_t status = iree_ok_status();
  const uint16_t preferred_slab_index = pool->preferred_slab_index;
  if (preferred_slab_index < pool->slab_count) {
    status = iree_hal_tlsf_pool_try_acquire_from_slab(
        pool, preferred_slab_index, size, requester_frontier,
        /*record_reuse_miss=*/true, &selected_allocation, &selected_result);
    has_selected_allocation = selected_result == IREE_HAL_POOL_ACQUIRE_OK ||
                              selected_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  }

  for (uint8_t i = 0; i < pool->reuse_candidate_slab_count &&
                      iree_status_is_ok(status) && !has_selected_allocation;
       ++i) {
    const uint16_t slab_index = pool->reuse_candidate_slab_indices[i];
    if (slab_index == preferred_slab_index || slab_index >= pool->slab_count) {
      continue;
    }
    status = iree_hal_tlsf_pool_try_acquire_from_slab(
        pool, slab_index, size, requester_frontier,
        /*record_reuse_miss=*/true, &selected_allocation, &selected_result);
    has_selected_allocation = selected_result == IREE_HAL_POOL_ACQUIRE_OK ||
                              selected_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  }

  if (iree_status_is_ok(status) && !has_selected_allocation) {
    uint16_t slab_index = 0;
    status = iree_hal_tlsf_pool_append_slab(pool, &slab_index);
    if (iree_status_is_ok(status)) {
      status = iree_hal_tlsf_pool_try_acquire_from_slab(
          pool, slab_index, size, requester_frontier,
          /*record_reuse_miss=*/true, &selected_allocation, &selected_result);
      has_selected_allocation =
          selected_result == IREE_HAL_POOL_ACQUIRE_OK ||
          selected_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    }
  }

  if (iree_status_is_ok(status)) {
    if (has_selected_allocation) {
      iree_hal_tlsf_pool_slab_t* slab =
          pool->slabs[selected_allocation.slab_index];
      if (!iree_hal_tlsf_pool_adjust_charged_reservation(
              pool, &charged_length, selected_allocation.allocation.length)) {
        iree_atomic_fetch_add(&pool->over_budget_count, 1,
                              iree_memory_order_relaxed);
        iree_hal_memory_tlsf_restore(
            &slab->tlsf, selected_allocation.allocation.block_index);
        memset(out_reservation, 0, sizeof(*out_reservation));
        memset(out_info, 0, sizeof(*out_info));
        *out_result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
        iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
        charged_length = 0;
      } else {
        status = iree_hal_tlsf_pool_return_allocation(
            pool, &selected_allocation, selected_result, out_reservation,
            out_info, out_result);
        if (!iree_status_is_ok(status)) {
          iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
          charged_length = 0;
          iree_hal_memory_tlsf_restore(
              &slab->tlsf, selected_allocation.allocation.block_index);
        }
      }
    } else {
      iree_atomic_fetch_add(&pool->exhausted_count, 1,
                            iree_memory_order_relaxed);
      memset(out_reservation, 0, sizeof(*out_reservation));
      memset(out_info, 0, sizeof(*out_info));
      *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
      iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
      charged_length = 0;
    }
  } else {
    iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
    charged_length = 0;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

static void iree_hal_tlsf_pool_release_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  iree_hal_tlsf_pool_release_node_t* release_node =
      (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)reservation->block_handle;
  iree_async_frontier_t* release_frontier =
      iree_hal_tlsf_pool_release_node_frontier(pool, release_node);
  iree_hal_tlsf_pool_slab_t* slab = release_node->slab;

  if (death_frontier && death_frontier->entry_count > 0) {
    if (death_frontier->entry_count <= slab->tlsf.frontier_capacity) {
      if (release_frontier != death_frontier) {
        memcpy(release_frontier, death_frontier,
               sizeof(iree_async_frontier_t) +
                   (iree_host_size_t)death_frontier->entry_count *
                       sizeof(iree_async_frontier_entry_t));
      }
    } else {
      // Sentinel count larger than capacity. The drain path forwards this
      // header to TLSF free(), which marks the block tainted without reading
      // entries.
      iree_async_frontier_initialize(
          release_frontier, (uint8_t)(slab->tlsf.frontier_capacity + 1u));
    }
  } else {
    iree_async_frontier_initialize(release_frontier, 0);
  }

  iree_hal_memory_trace_free(
      &pool->trace, (uint8_t*)slab->slab.base_ptr + reservation->offset);

  iree_hal_tlsf_pool_push_pending_release(pool, release_node);

  iree_hal_tlsf_pool_uncharge_reservation(pool, reservation->length);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->release_count, 1, iree_memory_order_relaxed);
  iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
}

//===----------------------------------------------------------------------===//
// Wrap / Query / Trim / Notification
//===----------------------------------------------------------------------===//

static void iree_hal_tlsf_pool_buffer_release(void* user_data,
                                              iree_hal_buffer_t* buffer) {
  iree_hal_tlsf_pool_buffer_state_t* state =
      (iree_hal_tlsf_pool_buffer_state_t*)user_data;
  iree_hal_pool_release_reservation(state->pool, &state->reservation, NULL);
  iree_allocator_t host_allocator = state->host_allocator;
  iree_allocator_free(host_allocator, state);
}

static iree_status_t iree_hal_tlsf_pool_materialize_reservation(
    iree_hal_pool_t* base_pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffer) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  iree_hal_slab_t slab;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&pool->mutex);
  if (reservation->slab_index < pool->slab_count) {
    slab = pool->slabs[reservation->slab_index]->slab;
  } else {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "reservation slab_index %" PRIu16
                              " exceeds TLSF pool slab_count %" PRIu32,
                              reservation->slab_index, pool->slab_count);
  }
  iree_slim_mutex_unlock(&pool->mutex);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  iree_hal_tlsf_pool_buffer_state_t* state = NULL;
  iree_hal_buffer_release_callback_t release_callback =
      iree_hal_buffer_release_callback_null();
  if (iree_all_bits_set(
          flags,
          IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator,
                                               sizeof(*state), (void**)&state));
    state->pool = base_pool;
    state->reservation = *reservation;
    state->host_allocator = pool->host_allocator;
    release_callback.fn = iree_hal_tlsf_pool_buffer_release;
    release_callback.user_data = state;
  }
  status = iree_hal_slab_provider_wrap_buffer(
      pool->slab_provider, &slab, reservation->offset, reservation->length,
      params, release_callback, out_buffer);
  if (!iree_status_is_ok(status) && state) {
    iree_allocator_free(pool->host_allocator, state);
  }
  return status;
}

static void iree_hal_tlsf_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  const iree_hal_tlsf_pool_t* pool = (const iree_hal_tlsf_pool_t*)base_pool;
  out_capabilities->memory_type = pool->memory_type;
  out_capabilities->supported_usage = pool->supported_usage;
  out_capabilities->min_allocation_size = 1;
  out_capabilities->max_allocation_size = pool->slab_length;
}

static void iree_hal_tlsf_pool_query_stats(const iree_hal_pool_t* base_pool,
                                           iree_hal_pool_stats_t* out_stats) {
  const iree_hal_tlsf_pool_t* pool = (const iree_hal_tlsf_pool_t*)base_pool;
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_reserved, iree_memory_order_relaxed);
  out_stats->bytes_committed = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_committed, iree_memory_order_relaxed);
  out_stats->bytes_free =
      out_stats->bytes_committed > out_stats->bytes_reserved
          ? out_stats->bytes_committed - out_stats->bytes_reserved
          : 0;
  out_stats->budget_limit = pool->budget_limit;
  out_stats->reservation_count = (uint32_t)iree_atomic_load(
      &pool->reservation_count, iree_memory_order_relaxed);
  out_stats->slab_count = (uint32_t)iree_atomic_load(
      &pool->committed_slab_count, iree_memory_order_relaxed);
  out_stats->reserve_count = (uint64_t)iree_atomic_load(
      &pool->reserve_count, iree_memory_order_relaxed);
  out_stats->release_count = (uint64_t)iree_atomic_load(
      &pool->release_count, iree_memory_order_relaxed);
  out_stats->reuse_count =
      (uint64_t)iree_atomic_load(&pool->reuse_count, iree_memory_order_relaxed);
  out_stats->reuse_miss_count = (uint64_t)iree_atomic_load(
      &pool->reuse_miss_count, iree_memory_order_relaxed);
  out_stats->fresh_count =
      (uint64_t)iree_atomic_load(&pool->fresh_count, iree_memory_order_relaxed);
  out_stats->exhausted_count = (uint64_t)iree_atomic_load(
      &pool->exhausted_count, iree_memory_order_relaxed);
  out_stats->over_budget_count = (uint64_t)iree_atomic_load(
      &pool->over_budget_count, iree_memory_order_relaxed);
  out_stats->wait_count =
      (uint64_t)iree_atomic_load(&pool->wait_count, iree_memory_order_relaxed);
}

static iree_status_t iree_hal_tlsf_pool_trim(iree_hal_pool_t* base_pool) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_tlsf_pool_drain_pending_releases(pool);
  iree_hal_tlsf_pool_free_release_nodes(pool);
  iree_slim_mutex_unlock(&pool->mutex);
  iree_hal_slab_provider_trim(pool->slab_provider,
                              IREE_HAL_SLAB_PROVIDER_TRIM_FLAG_EXCESS);
  return iree_ok_status();
}

static iree_async_notification_t* iree_hal_tlsf_pool_notification(
    iree_hal_pool_t* base_pool) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  return pool->notification;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_pool_vtable_t iree_hal_tlsf_pool_vtable = {
    .destroy = iree_hal_tlsf_pool_destroy,
    .acquire_reservation = iree_hal_tlsf_pool_acquire_reservation,
    .release_reservation = iree_hal_tlsf_pool_release_reservation,
    .materialize_reservation = iree_hal_tlsf_pool_materialize_reservation,
    .query_capabilities = iree_hal_tlsf_pool_query_capabilities,
    .query_stats = iree_hal_tlsf_pool_query_stats,
    .trim = iree_hal_tlsf_pool_trim,
    .notification = iree_hal_tlsf_pool_notification,
};
