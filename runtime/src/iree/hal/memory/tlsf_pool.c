// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tlsf_pool.h"

#include "iree/async/frontier.h"
#include "iree/async/notification.h"
#include "iree/base/threading/mutex.h"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tlsf_pool_release_node_t {
  // Intrusive next pointer in pool->pending_release_head.
  struct iree_hal_tlsf_pool_release_node_t* next;

  // TLSF block handle owned by this reservation.
  iree_hal_memory_tlsf_block_index_t block_index;
} iree_hal_tlsf_pool_release_node_t;

typedef struct iree_hal_tlsf_pool_t {
  iree_hal_resource_t resource;
  iree_hal_slab_provider_t* slab_provider;
  iree_async_notification_t* notification;
  iree_slim_mutex_t mutex;
  iree_hal_memory_tlsf_t tlsf;
  iree_hal_slab_t slab;
  iree_atomic_intptr_t pending_release_head;
  iree_hal_pool_epoch_query_t epoch_query;
  iree_allocator_t host_allocator;

  // Cached from slab_provider and TLSF options at creation time.
  iree_hal_memory_type_t memory_type;
  iree_hal_buffer_usage_t supported_usage;
  iree_device_size_t budget_limit;
  iree_host_size_t release_node_size;
  iree_host_size_t release_frontier_offset;

  // Scratch storage used under |mutex| to hold rejected block indices while a
  // reserve call keeps searching for a satisfiable block.
  iree_hal_memory_tlsf_block_index_t* rejected_block_indices;
  iree_host_size_t rejected_block_capacity;

  // Statistics (relaxed atomics, incremented on reserve/release).
  iree_atomic_int64_t bytes_reserved;
  iree_atomic_int32_t reservation_count;
  iree_atomic_int64_t reserve_count;
  iree_atomic_int64_t release_count;
  iree_atomic_int64_t reuse_count;
  iree_atomic_int64_t reuse_miss_count;
  iree_atomic_int64_t fresh_count;
  iree_atomic_int64_t exhausted_count;
  iree_atomic_int64_t over_budget_count;
} iree_hal_tlsf_pool_t;

typedef struct iree_hal_tlsf_pool_buffer_state_t {
  // Borrowed from the wrapped buffer's creator. Pool owners must keep the pool
  // alive until all buffers sourced from it are destroyed.
  iree_hal_pool_t* pool;
  iree_hal_pool_reservation_t reservation;
  iree_allocator_t host_allocator;
} iree_hal_tlsf_pool_buffer_state_t;

static const iree_hal_pool_vtable_t iree_hal_tlsf_pool_vtable;
static void iree_hal_tlsf_pool_destroy(iree_hal_pool_t* base_pool);

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

static void iree_hal_tlsf_pool_drain_pending_releases(
    iree_hal_tlsf_pool_t* pool)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* node =
      iree_hal_tlsf_pool_take_pending_releases(pool);
  while (node) {
    iree_hal_tlsf_pool_release_node_t* next = node->next;
    iree_async_frontier_t* death_frontier =
        iree_hal_tlsf_pool_release_node_frontier(pool, node);
    iree_hal_memory_tlsf_free(
        &pool->tlsf, node->block_index,
        death_frontier->entry_count > 0 ? death_frontier : NULL);
    iree_allocator_free(pool->host_allocator, node);
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
    iree_hal_tlsf_pool_t* pool, uint32_t rejected_block_count)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  for (uint32_t i = 0; i < rejected_block_count; ++i) {
    iree_hal_memory_tlsf_restore(&pool->tlsf, pool->rejected_block_indices[i]);
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

  iree_hal_slab_provider_retain(slab_provider);
  pool->slab_provider = slab_provider;
  iree_async_notification_retain(notification);
  pool->notification = notification;
  iree_hal_slab_provider_query_properties(slab_provider, &pool->memory_type,
                                          &pool->supported_usage);

  iree_status_t status = iree_hal_memory_tlsf_initialize(
      options.tlsf_options, pool->host_allocator, &pool->tlsf);
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_tlsf_pool_release_node_t), &pool->release_node_size,
        IREE_STRUCT_FIELD_ALIGNED(1, iree_async_frontier_t,
                                  iree_alignof(iree_async_frontier_entry_t),
                                  &pool->release_frontier_offset),
        IREE_STRUCT_FIELD(pool->tlsf.frontier_capacity,
                          iree_async_frontier_entry_t, NULL));
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(pool->host_allocator,
                                         pool->tlsf.block_capacity,
                                         sizeof(*pool->rejected_block_indices),
                                         (void**)&pool->rejected_block_indices);
  }
  if (iree_status_is_ok(status)) {
    pool->rejected_block_capacity = pool->tlsf.block_capacity;
    status = iree_hal_slab_provider_acquire_slab(
        pool->slab_provider, pool->tlsf.range_length, &pool->slab);
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
  iree_slim_mutex_unlock(&pool->mutex);

  if (pool->tlsf.block_storage) {
    iree_hal_memory_tlsf_deinitialize(&pool->tlsf);
  }
  if (pool->slab.length > 0) {
    iree_hal_slab_provider_release_slab(pool->slab_provider, &pool->slab);
  }
  iree_slim_mutex_deinitialize(&pool->mutex);
  iree_allocator_free(pool->host_allocator, pool->rejected_block_indices);
  iree_async_notification_release(pool->notification);
  iree_hal_slab_provider_release(pool->slab_provider);
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_allocator_free(host_allocator, pool);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Reserve / Release
//===----------------------------------------------------------------------===//

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
  if (alignment > pool->tlsf.alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment %" PRIdsz
                            " exceeds TLSF pool alignment %" PRIdsz,
                            alignment, pool->tlsf.alignment);
  }

  if (pool->budget_limit > 0) {
    const iree_device_size_t bytes_reserved =
        (iree_device_size_t)iree_atomic_load(&pool->bytes_reserved,
                                             iree_memory_order_relaxed);
    if (bytes_reserved > pool->budget_limit ||
        size > pool->budget_limit - bytes_reserved) {
      iree_atomic_fetch_add(&pool->over_budget_count, 1,
                            iree_memory_order_relaxed);
      *out_result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
      return iree_ok_status();
    }
  }

  iree_status_t status = iree_ok_status();
  iree_hal_memory_tlsf_allocation_t allocation;
  uint32_t rejected_block_count = 0;

  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_tlsf_pool_drain_pending_releases(pool);

  while (true) {
    status = iree_hal_memory_tlsf_allocate(&pool->tlsf, size, &allocation);
    if (!iree_status_is_ok(status)) {
      iree_hal_tlsf_pool_restore_rejected_blocks(pool, rejected_block_count);
      iree_slim_mutex_unlock(&pool->mutex);
      if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
        iree_status_ignore(status);
        iree_atomic_fetch_add(&pool->exhausted_count, 1,
                              iree_memory_order_relaxed);
        *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
        return iree_ok_status();
      }
      return status;
    }

    if (!iree_hal_tlsf_pool_frontier_is_satisfied(pool, requester_frontier,
                                                  allocation.death_frontier,
                                                  allocation.block_flags)) {
      iree_atomic_fetch_add(&pool->reuse_miss_count, 1,
                            iree_memory_order_relaxed);
      status = iree_hal_tlsf_pool_ensure_rejected_capacity(
          pool, (iree_host_size_t)rejected_block_count + 1);
      if (!iree_status_is_ok(status)) {
        iree_hal_memory_tlsf_restore(&pool->tlsf, allocation.block_index);
        iree_hal_tlsf_pool_restore_rejected_blocks(pool, rejected_block_count);
        iree_slim_mutex_unlock(&pool->mutex);
        return status;
      }
      pool->rejected_block_indices[rejected_block_count++] =
          allocation.block_index;
      continue;
    }

    iree_hal_tlsf_pool_release_node_t* release_node = NULL;
    status = iree_allocator_malloc(
        pool->host_allocator, pool->release_node_size, (void**)&release_node);
    if (!iree_status_is_ok(status)) {
      iree_hal_memory_tlsf_restore(&pool->tlsf, allocation.block_index);
      iree_hal_tlsf_pool_restore_rejected_blocks(pool, rejected_block_count);
      iree_slim_mutex_unlock(&pool->mutex);
      return status;
    }
    release_node->next = NULL;
    release_node->block_index = allocation.block_index;

    iree_hal_tlsf_pool_restore_rejected_blocks(pool, rejected_block_count);
    iree_slim_mutex_unlock(&pool->mutex);

    memset(out_reservation, 0, sizeof(*out_reservation));
    out_reservation->offset = allocation.offset;
    out_reservation->length = allocation.length;
    out_reservation->block_handle = (uint64_t)(uintptr_t)release_node;
    out_reservation->slab_index = 0;

    memset(out_info, 0, sizeof(*out_info));
    iree_atomic_fetch_add(&pool->bytes_reserved, (int64_t)allocation.length,
                          iree_memory_order_relaxed);
    iree_atomic_fetch_add(&pool->reservation_count, 1,
                          iree_memory_order_relaxed);
    iree_atomic_fetch_add(&pool->reserve_count, 1, iree_memory_order_relaxed);
    if (allocation.death_frontier) {
      iree_atomic_fetch_add(&pool->reuse_count, 1, iree_memory_order_relaxed);
      *out_result = IREE_HAL_POOL_ACQUIRE_OK;
    } else {
      iree_atomic_fetch_add(&pool->fresh_count, 1, iree_memory_order_relaxed);
      *out_result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    }
    return iree_ok_status();
  }
}

static void iree_hal_tlsf_pool_release_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  iree_hal_tlsf_pool_release_node_t* release_node =
      (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)reservation->block_handle;
  iree_async_frontier_t* release_frontier =
      iree_hal_tlsf_pool_release_node_frontier(pool, release_node);

  if (death_frontier && death_frontier->entry_count > 0) {
    if (death_frontier->entry_count <= pool->tlsf.frontier_capacity) {
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
          release_frontier, (uint8_t)(pool->tlsf.frontier_capacity + 1u));
    }
  } else {
    iree_async_frontier_initialize(release_frontier, 0);
  }

  iree_hal_tlsf_pool_push_pending_release(pool, release_node);

  iree_atomic_fetch_add(&pool->bytes_reserved, -(int64_t)reservation->length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->release_count, 1, iree_memory_order_relaxed);
  iree_async_notification_signal(pool->notification, INT32_MAX);
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
  iree_status_t status = iree_hal_slab_provider_wrap_buffer(
      pool->slab_provider, &pool->slab, reservation->offset,
      reservation->length, params, release_callback, out_buffer);
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
  out_capabilities->min_allocation_size = pool->tlsf.alignment;
  out_capabilities->max_allocation_size = pool->tlsf.range_length;
}

static void iree_hal_tlsf_pool_query_stats(const iree_hal_pool_t* base_pool,
                                           iree_hal_pool_stats_t* out_stats) {
  const iree_hal_tlsf_pool_t* pool = (const iree_hal_tlsf_pool_t*)base_pool;
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_reserved, iree_memory_order_relaxed);
  out_stats->bytes_free = pool->tlsf.range_length - out_stats->bytes_reserved;
  out_stats->bytes_committed = pool->slab.length;
  out_stats->budget_limit = pool->budget_limit;
  out_stats->reservation_count = (uint32_t)iree_atomic_load(
      &pool->reservation_count, iree_memory_order_relaxed);
  out_stats->slab_count = 1;
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
  out_stats->wait_count = 0;
}

static iree_status_t iree_hal_tlsf_pool_trim(iree_hal_pool_t* base_pool) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
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
