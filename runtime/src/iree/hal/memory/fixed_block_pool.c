// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/fixed_block_pool.h"

#include "iree/async/frontier.h"
#include "iree/async/notification.h"
#include "iree/hal/memory/tracing.h"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

typedef struct iree_hal_fixed_block_pool_t {
  // Base pool resource for vtable dispatch and ref counting.
  iree_hal_resource_t resource;

  // Provider backing the single fixed-block slab.
  iree_hal_slab_provider_t* slab_provider;

  // Notification signaled when a reservation release may unblock waiters.
  iree_async_notification_t* notification;

  // Lock-free offset allocator for fixed-size blocks within |slab|.
  iree_hal_memory_fixed_block_allocator_t* block_allocator;

  // Physical memory backing all fixed blocks.
  iree_hal_slab_t slab;

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

  // Byte size of every block in |block_allocator|.
  iree_device_size_t block_size;

  // Number of blocks managed by |block_allocator|.
  uint32_t block_count;

  // Logical byte budget for live reservations. 0 means unlimited.
  iree_device_size_t budget_limit;

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
} iree_hal_fixed_block_pool_t;

// Per-buffer release state. Allocated when reservation ownership is transferred
// to a materialized buffer, and freed in the buffer release callback.
typedef struct iree_hal_fixed_block_pool_buffer_state_t {
  // Borrowed from the wrapped buffer's creator. Pool owners must keep the pool
  // alive until all buffers sourced from it are destroyed.
  iree_hal_pool_t* pool;

  // Reservation returned to |pool| when the buffer is destroyed.
  iree_hal_pool_reservation_t reservation;

  // Host allocator used for this state object.
  iree_allocator_t host_allocator;
} iree_hal_fixed_block_pool_buffer_state_t;

static const iree_hal_pool_vtable_t iree_hal_fixed_block_pool_vtable;
static void iree_hal_fixed_block_pool_destroy(iree_hal_pool_t* base_pool);

static const char* IREE_HAL_FIXED_BLOCK_POOL_TRACE_ID =
    "iree-hal-fixed-block-pool";

//===----------------------------------------------------------------------===//
// Frontier helpers
//===----------------------------------------------------------------------===//

static bool iree_hal_fixed_block_pool_frontier_is_satisfied(
    const iree_hal_fixed_block_pool_t* pool,
    const iree_async_frontier_t* requester_frontier,
    const iree_async_frontier_t* death_frontier,
    iree_hal_memory_fixed_block_allocator_block_flags_t block_flags) {
  if (!death_frontier) {
    return block_flags == IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_NONE;
  }
  if (block_flags & IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED) {
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

static void iree_hal_fixed_block_pool_restore_rejected_blocks(
    iree_hal_fixed_block_pool_t* pool, uint32_t rejected_block_count,
    const uint32_t* rejected_block_indices) {
  for (uint32_t i = 0; i < rejected_block_count; ++i) {
    iree_hal_memory_fixed_block_allocator_restore(pool->block_allocator,
                                                  rejected_block_indices[i]);
  }
}

static bool iree_hal_fixed_block_pool_try_charge_reservation(
    iree_hal_fixed_block_pool_t* pool, iree_device_size_t length) {
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

static void iree_hal_fixed_block_pool_uncharge_reservation(
    iree_hal_fixed_block_pool_t* pool, iree_device_size_t length) {
  iree_atomic_fetch_add(&pool->bytes_reserved, -(int64_t)length,
                        iree_memory_order_relaxed);
}

static bool iree_hal_fixed_block_pool_can_wait_for_allocation(
    iree_hal_pool_reserve_flags_t flags,
    const iree_hal_memory_fixed_block_allocator_allocation_t* allocation) {
  return iree_all_bits_set(flags,
                           IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER) &&
         allocation->death_frontier &&
         !iree_all_bits_set(
             allocation->block_flags,
             IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED);
}

static iree_status_t iree_hal_fixed_block_pool_return_allocation(
    iree_hal_fixed_block_pool_t* pool,
    const iree_hal_memory_fixed_block_allocator_allocation_t* allocation,
    iree_hal_pool_acquire_result_t result,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  memset(out_reservation, 0, sizeof(*out_reservation));
  out_reservation->offset = allocation->offset;
  out_reservation->length = pool->block_size;
  out_reservation->block_handle = allocation->block_index;
  out_reservation->slab_index = 0;

  // Tainted blocks never reach this helper: frontier_is_satisfied rejects
  // them (so they never become OK/OK_FRESH) and can_wait_for_allocation
  // excludes them (so they never become OK_NEEDS_WAIT). Only NEEDS_WAIT
  // returns a wait_frontier here.
  memset(out_info, 0, sizeof(*out_info));
  if (result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
    out_info->wait_frontier = allocation->death_frontier;
  }

  iree_atomic_fetch_add(&pool->reservation_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, 1, iree_memory_order_relaxed);
  switch (result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
      iree_atomic_fetch_add(&pool->reuse_count, 1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      iree_atomic_fetch_add(&pool->fresh_count, 1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      iree_atomic_fetch_add(&pool->wait_count, 1, iree_memory_order_relaxed);
      break;
    default:
      IREE_ASSERT(false, "invalid successful fixed-block pool result: %u",
                  result);
      break;
  }
  iree_hal_memory_trace_alloc(
      &pool->trace, (uint8_t*)pool->slab.base_ptr + allocation->offset,
      pool->block_size);
  *out_result = result;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Create / Destroy
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_fixed_block_pool_create(
    iree_hal_fixed_block_pool_options_t options,
    iree_hal_slab_provider_t* slab_provider,
    iree_async_notification_t* notification,
    iree_hal_pool_epoch_query_t epoch_query, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(slab_provider);
  IREE_ASSERT_ARGUMENT(notification);
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_device_size_t slab_length = 0;
  if (!iree_device_size_checked_mul(options.block_allocator_options.block_count,
                                    options.block_allocator_options.block_size,
                                    &slab_length)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixed-block pool slab length overflows: block_count=%u "
        "block_size=%" PRIdsz,
        (unsigned)options.block_allocator_options.block_count,
        options.block_allocator_options.block_size);
  }

  iree_hal_fixed_block_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  iree_hal_resource_initialize(&iree_hal_fixed_block_pool_vtable,
                               &pool->resource);
  pool->host_allocator = host_allocator;
  pool->epoch_query = epoch_query;
  pool->block_size = options.block_allocator_options.block_size;
  pool->block_count = options.block_allocator_options.block_count;
  pool->budget_limit = options.budget_limit;

  iree_hal_slab_provider_retain(slab_provider);
  pool->slab_provider = slab_provider;
  iree_async_notification_retain(notification);
  pool->notification = notification;
  iree_hal_slab_provider_query_properties(slab_provider, &pool->memory_type,
                                          &pool->supported_usage);

  iree_status_t status = iree_hal_memory_trace_initialize_pool(
      options.trace_name, IREE_HAL_FIXED_BLOCK_POOL_TRACE_ID, host_allocator,
      &pool->trace);
  if (iree_status_is_ok(status)) {
    status = iree_hal_memory_fixed_block_allocator_allocate(
        options.block_allocator_options, pool->host_allocator,
        &pool->block_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_slab_provider_acquire_slab(pool->slab_provider,
                                                 slab_length, &pool->slab);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_fixed_block_pool_destroy((iree_hal_pool_t*)pool);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_pool = (iree_hal_pool_t*)pool;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_fixed_block_pool_destroy(iree_hal_pool_t* base_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_hal_memory_fixed_block_allocator_free(pool->block_allocator);
  if (pool->slab.length > 0) {
    iree_hal_slab_provider_release_slab(pool->slab_provider, &pool->slab);
  }
  iree_hal_memory_trace_deinitialize(&pool->trace);
  iree_async_notification_release(pool->notification);
  iree_hal_slab_provider_release(pool->slab_provider);
  iree_allocator_free(host_allocator, pool);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Reserve / Release
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_fixed_block_pool_acquire_reservation(
    iree_hal_pool_t* base_pool, iree_device_size_t size,
    iree_device_size_t alignment,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;

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
  if (size > pool->block_size) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  if (alignment > pool->block_size || (pool->block_size % alignment) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment %" PRIdsz
                            " is incompatible with fixed block size %" PRIdsz,
                            alignment, pool->block_size);
  }

  if (!iree_hal_fixed_block_pool_try_charge_reservation(pool,
                                                        pool->block_size)) {
    iree_atomic_fetch_add(&pool->over_budget_count, 1,
                          iree_memory_order_relaxed);
    memset(out_reservation, 0, sizeof(*out_reservation));
    memset(out_info, 0, sizeof(*out_info));
    *out_result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
    return iree_ok_status();
  }

  uint32_t
      rejected_block_indices[IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS];
  uint32_t rejected_block_count = 0;
  iree_hal_memory_fixed_block_allocator_allocation_t selected_allocation;
  iree_hal_pool_acquire_result_t selected_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  bool has_selected_allocation = false;
  iree_hal_memory_fixed_block_allocator_allocation_t wait_allocation;
  bool has_wait_allocation = false;
  iree_status_t status = iree_ok_status();

  while (iree_status_is_ok(status) && !has_selected_allocation) {
    iree_hal_memory_fixed_block_allocator_allocation_t allocation;
    iree_hal_memory_fixed_block_allocator_acquire_result_t allocation_result =
        IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED;
    status = iree_hal_memory_fixed_block_allocator_try_acquire(
        pool->block_allocator, &allocation, &allocation_result);
    if (iree_status_is_ok(status) &&
        allocation_result ==
            IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED) {
      if (has_wait_allocation) {
        selected_allocation = wait_allocation;
        selected_result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
        has_selected_allocation = true;
        has_wait_allocation = false;
      } else {
        selected_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
      }
      break;
    }
    if (!iree_status_is_ok(status)) break;

    const bool frontier_is_satisfied =
        iree_hal_fixed_block_pool_frontier_is_satisfied(
            pool, requester_frontier, allocation.death_frontier,
            allocation.block_flags);
    if (!frontier_is_satisfied) {
      iree_atomic_fetch_add(&pool->reuse_miss_count, 1,
                            iree_memory_order_relaxed);
      if (!has_wait_allocation &&
          iree_hal_fixed_block_pool_can_wait_for_allocation(flags,
                                                            &allocation)) {
        wait_allocation = allocation;
        has_wait_allocation = true;
        continue;
      }
      rejected_block_indices[rejected_block_count++] = allocation.block_index;
      continue;
    }

    selected_allocation = allocation;
    selected_result = allocation.death_frontier
                          ? IREE_HAL_POOL_ACQUIRE_OK
                          : IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    has_selected_allocation = true;
  }

  if (iree_status_is_ok(status)) {
    iree_hal_fixed_block_pool_restore_rejected_blocks(
        pool, rejected_block_count, rejected_block_indices);
    rejected_block_count = 0;
    if (has_wait_allocation) {
      iree_hal_memory_fixed_block_allocator_restore(
          pool->block_allocator, wait_allocation.block_index);
      has_wait_allocation = false;
    }
    if (has_selected_allocation) {
      status = iree_hal_fixed_block_pool_return_allocation(
          pool, &selected_allocation, selected_result, out_reservation,
          out_info, out_result);
      if (!iree_status_is_ok(status)) {
        iree_hal_fixed_block_pool_uncharge_reservation(pool, pool->block_size);
        iree_hal_memory_fixed_block_allocator_restore(
            pool->block_allocator, selected_allocation.block_index);
      }
    } else {
      iree_atomic_fetch_add(&pool->exhausted_count, 1,
                            iree_memory_order_relaxed);
      memset(out_reservation, 0, sizeof(*out_reservation));
      memset(out_info, 0, sizeof(*out_info));
      *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
      iree_hal_fixed_block_pool_uncharge_reservation(pool, pool->block_size);
    }
  } else {
    iree_hal_fixed_block_pool_restore_rejected_blocks(
        pool, rejected_block_count, rejected_block_indices);
    if (has_wait_allocation) {
      iree_hal_memory_fixed_block_allocator_restore(
          pool->block_allocator, wait_allocation.block_index);
    }
    iree_hal_fixed_block_pool_uncharge_reservation(pool, pool->block_size);
  }
  return status;
}

static void iree_hal_fixed_block_pool_release_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;

  iree_hal_memory_trace_free(
      &pool->trace, (uint8_t*)pool->slab.base_ptr + reservation->offset);

  iree_hal_memory_fixed_block_allocator_release(
      pool->block_allocator, (uint32_t)reservation->block_handle,
      death_frontier);

  iree_hal_fixed_block_pool_uncharge_reservation(pool, reservation->length);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->release_count, 1, iree_memory_order_relaxed);
  iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
}

//===----------------------------------------------------------------------===//
// Wrap / Query / Trim / Notification
//===----------------------------------------------------------------------===//

static void iree_hal_fixed_block_pool_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  iree_hal_fixed_block_pool_buffer_state_t* state =
      (iree_hal_fixed_block_pool_buffer_state_t*)user_data;
  iree_hal_pool_release_reservation(state->pool, &state->reservation, NULL);
  iree_allocator_t host_allocator = state->host_allocator;
  iree_allocator_free(host_allocator, state);
}

static iree_status_t iree_hal_fixed_block_pool_materialize_reservation(
    iree_hal_pool_t* base_pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffer) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;

  iree_hal_fixed_block_pool_buffer_state_t* state = NULL;
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
    release_callback.fn = iree_hal_fixed_block_pool_buffer_release;
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

static void iree_hal_fixed_block_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  const iree_hal_fixed_block_pool_t* pool =
      (const iree_hal_fixed_block_pool_t*)base_pool;
  out_capabilities->memory_type = pool->memory_type;
  out_capabilities->supported_usage = pool->supported_usage;
  out_capabilities->min_allocation_size = pool->block_size;
  out_capabilities->max_allocation_size = pool->block_size;
}

static void iree_hal_fixed_block_pool_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_stats) {
  const iree_hal_fixed_block_pool_t* pool =
      (const iree_hal_fixed_block_pool_t*)base_pool;
  iree_hal_memory_fixed_block_allocator_stats_t block_stats;
  iree_hal_memory_fixed_block_allocator_query_stats(pool->block_allocator,
                                                    &block_stats);
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_reserved, iree_memory_order_relaxed);
  const iree_device_size_t managed_bytes =
      (iree_device_size_t)block_stats.block_count * pool->block_size;
  out_stats->bytes_free = managed_bytes - out_stats->bytes_reserved;
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
  out_stats->wait_count =
      (uint64_t)iree_atomic_load(&pool->wait_count, iree_memory_order_relaxed);
}

static iree_status_t iree_hal_fixed_block_pool_trim(
    iree_hal_pool_t* base_pool) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  iree_hal_slab_provider_trim(pool->slab_provider,
                              IREE_HAL_SLAB_PROVIDER_TRIM_FLAG_EXCESS);
  return iree_ok_status();
}

static iree_async_notification_t* iree_hal_fixed_block_pool_notification(
    iree_hal_pool_t* base_pool) {
  iree_hal_fixed_block_pool_t* pool = (iree_hal_fixed_block_pool_t*)base_pool;
  return pool->notification;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_pool_vtable_t iree_hal_fixed_block_pool_vtable = {
    .destroy = iree_hal_fixed_block_pool_destroy,
    .acquire_reservation = iree_hal_fixed_block_pool_acquire_reservation,
    .release_reservation = iree_hal_fixed_block_pool_release_reservation,
    .materialize_reservation =
        iree_hal_fixed_block_pool_materialize_reservation,
    .query_capabilities = iree_hal_fixed_block_pool_query_capabilities,
    .query_stats = iree_hal_fixed_block_pool_query_stats,
    .trim = iree_hal_fixed_block_pool_trim,
    .notification = iree_hal_fixed_block_pool_notification,
};
