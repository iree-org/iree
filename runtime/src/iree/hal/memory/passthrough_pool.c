// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/passthrough_pool.h"

#include "iree/async/notification.h"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

typedef struct iree_hal_passthrough_pool_t {
  iree_hal_resource_t resource;
  iree_hal_slab_provider_t* slab_provider;
  iree_async_notification_t* notification;
  iree_allocator_t host_allocator;

  // Cached from slab_provider at creation time.
  iree_hal_memory_type_t memory_type;
  iree_hal_buffer_usage_t supported_usage;

  // Statistics (relaxed atomics, incremented on reserve/release).
  iree_atomic_int64_t bytes_reserved;
  iree_atomic_int32_t reservation_count;
  iree_atomic_int32_t slab_count;
  iree_atomic_int64_t reserve_count;
  iree_atomic_int64_t release_count;
} iree_hal_passthrough_pool_t;

// Per-buffer release state. Allocated in wrap_reservation, freed in the
// buffer release callback.
typedef struct iree_hal_passthrough_pool_buffer_state_t {
  iree_hal_pool_t* pool;  // retained
  iree_hal_pool_reservation_t reservation;
  iree_allocator_t host_allocator;
} iree_hal_passthrough_pool_buffer_state_t;

static const iree_hal_pool_vtable_t iree_hal_passthrough_pool_vtable;

//===----------------------------------------------------------------------===//
// Create / Destroy
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_passthrough_pool_create(
    iree_hal_slab_provider_t* slab_provider,
    iree_async_notification_t* notification, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(slab_provider);
  IREE_ASSERT_ARGUMENT(notification);
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;

  iree_hal_passthrough_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  iree_hal_resource_initialize(&iree_hal_passthrough_pool_vtable,
                               &pool->resource);

  iree_hal_slab_provider_retain(slab_provider);
  pool->slab_provider = slab_provider;
  iree_async_notification_retain(notification);
  pool->notification = notification;
  pool->host_allocator = host_allocator;

  iree_hal_slab_provider_query_properties(slab_provider, &pool->memory_type,
                                          &pool->supported_usage);

  iree_atomic_store(&pool->bytes_reserved, 0, iree_memory_order_relaxed);
  iree_atomic_store(&pool->reservation_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&pool->slab_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&pool->reserve_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&pool->release_count, 0, iree_memory_order_relaxed);

  *out_pool = (iree_hal_pool_t*)pool;
  return iree_ok_status();
}

static void iree_hal_passthrough_pool_destroy(iree_hal_pool_t* base_pool) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_async_notification_release(pool->notification);
  iree_hal_slab_provider_release(pool->slab_provider);
  iree_allocator_free(host_allocator, pool);
}

//===----------------------------------------------------------------------===//
// Reserve / Release
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_passthrough_pool_reserve(
    iree_hal_pool_t* base_pool, iree_device_size_t size,
    iree_device_size_t alignment,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_reserve_result_t* out_result) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;

  iree_hal_slab_t slab;
  IREE_RETURN_IF_ERROR(
      iree_hal_slab_provider_acquire_slab(pool->slab_provider, size, &slab));

  // The passthrough pool stashes base_ptr in block_handle and discards
  // provider_handle. This only works for providers where provider_handle
  // is unused (CPU malloc, etc.). GPU providers that need the handle for
  // VMM unmapping or driver resource tracking require a different pool type.
  if (slab.provider_handle != 0) {
    iree_hal_slab_provider_release_slab(pool->slab_provider, &slab);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "passthrough pool requires provider_handle == 0; provider returned "
        "%" PRIu64 " (use a suballocating pool for providers with handles)",
        slab.provider_handle);
  }

  memset(out_reservation, 0, sizeof(*out_reservation));
  out_reservation->offset = 0;
  out_reservation->length = slab.length;
  out_reservation->block_handle = (uint64_t)(uintptr_t)slab.base_ptr;

  iree_atomic_fetch_add(&pool->bytes_reserved, (int64_t)slab.length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reservation_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->slab_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, 1, iree_memory_order_relaxed);

  *out_result = IREE_HAL_POOL_RESERVE_OK_FRESH;
  return iree_ok_status();
}

static void iree_hal_passthrough_pool_release_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;

  // Reconstruct the slab from the reservation. The base_ptr was stashed in
  // block_handle during reserve().
  iree_hal_slab_t slab = {
      .base_ptr = (uint8_t*)(uintptr_t)reservation->block_handle,
      .length = reservation->length,
      .provider_handle = 0,
  };
  iree_hal_slab_provider_release_slab(pool->slab_provider, &slab);

  iree_atomic_fetch_add(&pool->bytes_reserved, -(int64_t)reservation->length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->slab_count, -1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->release_count, 1, iree_memory_order_relaxed);

  iree_async_notification_signal(pool->notification, INT32_MAX);
}

//===----------------------------------------------------------------------===//
// Wrap / Query / Trim / Notification
//===----------------------------------------------------------------------===//

// Buffer release callback: returns the slab to the pool when the buffer's
// ref count reaches zero.
static void iree_hal_passthrough_pool_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  iree_hal_passthrough_pool_buffer_state_t* state =
      (iree_hal_passthrough_pool_buffer_state_t*)user_data;
  iree_hal_pool_release_reservation(state->pool, &state->reservation, NULL);
  iree_hal_pool_release(state->pool);
  iree_allocator_t host_allocator = state->host_allocator;
  iree_allocator_free(host_allocator, state);
}

static iree_status_t iree_hal_passthrough_pool_wrap_reservation(
    iree_hal_pool_t* base_pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;

  // Allocate the release state that the buffer callback will use to return
  // the slab to the pool.
  iree_hal_passthrough_pool_buffer_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator,
                                             sizeof(*state), (void**)&state));
  iree_hal_pool_retain(base_pool);
  state->pool = base_pool;
  state->reservation = *reservation;
  state->host_allocator = pool->host_allocator;

  iree_hal_buffer_params_canonicalize(&params);
  iree_hal_buffer_release_callback_t release_callback = {
      .fn = iree_hal_passthrough_pool_buffer_release,
      .user_data = state,
  };
  iree_byte_span_t data = {
      .data = (uint8_t*)(uintptr_t)reservation->block_handle,
      .data_length = (iree_host_size_t)reservation->length,
  };

  iree_status_t status = iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(), params.type, params.access,
      params.usage, reservation->length, data, release_callback,
      pool->host_allocator, out_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_pool_release(base_pool);
    iree_allocator_free(pool->host_allocator, state);
  }
  return status;
}

static void iree_hal_passthrough_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  const iree_hal_passthrough_pool_t* pool =
      (const iree_hal_passthrough_pool_t*)base_pool;
  out_capabilities->memory_type = pool->memory_type;
  out_capabilities->supported_usage = pool->supported_usage;
  out_capabilities->min_allocation_size = 0;
  out_capabilities->max_allocation_size = 0;
}

static void iree_hal_passthrough_pool_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_stats) {
  const iree_hal_passthrough_pool_t* pool =
      (const iree_hal_passthrough_pool_t*)base_pool;
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_reserved, iree_memory_order_relaxed);
  out_stats->bytes_free = 0;
  out_stats->bytes_committed = out_stats->bytes_reserved;
  out_stats->budget_limit = 0;
  out_stats->reservation_count = (uint32_t)iree_atomic_load(
      &pool->reservation_count, iree_memory_order_relaxed);
  out_stats->slab_count =
      (uint32_t)iree_atomic_load(&pool->slab_count, iree_memory_order_relaxed);
  out_stats->reserve_count = (uint64_t)iree_atomic_load(
      &pool->reserve_count, iree_memory_order_relaxed);
  out_stats->release_count = (uint64_t)iree_atomic_load(
      &pool->release_count, iree_memory_order_relaxed);
  out_stats->reuse_count = 0;
  out_stats->reuse_miss_count = 0;
  out_stats->fresh_count = out_stats->reserve_count;
  out_stats->exhausted_count = 0;
  out_stats->over_budget_count = 0;
  out_stats->wait_count = 0;
}

static iree_status_t iree_hal_passthrough_pool_trim(
    iree_hal_pool_t* base_pool) {
  return iree_ok_status();
}

static iree_async_notification_t* iree_hal_passthrough_pool_notification(
    iree_hal_pool_t* base_pool) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;
  return pool->notification;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_pool_vtable_t iree_hal_passthrough_pool_vtable = {
    .destroy = iree_hal_passthrough_pool_destroy,
    .reserve = iree_hal_passthrough_pool_reserve,
    .release_reservation = iree_hal_passthrough_pool_release_reservation,
    .wrap_reservation = iree_hal_passthrough_pool_wrap_reservation,
    .query_capabilities = iree_hal_passthrough_pool_query_capabilities,
    .query_stats = iree_hal_passthrough_pool_query_stats,
    .trim = iree_hal_passthrough_pool_trim,
    .notification = iree_hal_passthrough_pool_notification,
};
