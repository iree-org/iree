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

  // Statistics (relaxed atomics, incremented on acquire/release).
  iree_atomic_int64_t bytes_reserved;
  iree_atomic_int32_t reservation_count;
  iree_atomic_int32_t slab_count;
  iree_atomic_int64_t reserve_count;
  iree_atomic_int64_t release_count;
} iree_hal_passthrough_pool_t;

// Per-reservation slab state owned by the reservation until release.
typedef struct iree_hal_passthrough_pool_reservation_state_t {
  // Borrowed from the source pool. Pool owners must keep the pool alive until
  // all reservations and buffers sourced from it are destroyed.
  iree_hal_pool_t* pool;

  iree_hal_slab_t slab;
  iree_device_size_t reservation_length;

  // One reference for the live reservation token plus one reference for each
  // materialized buffer view. The slab is released when the last reference
  // drops, which allows explicit release_reservation() to run before borrowed
  // backing views are decommitted.
  iree_atomic_int32_t reference_count;

  // Set exactly once when the reservation token is released. Owning
  // materialized buffers consume that reservation release in their destroy
  // callback; borrowed views only drop their own reference.
  iree_atomic_int32_t reservation_released;
} iree_hal_passthrough_pool_reservation_state_t;

static const iree_hal_pool_vtable_t iree_hal_passthrough_pool_vtable;
static void iree_hal_passthrough_pool_destroy(iree_hal_pool_t* base_pool);

//===----------------------------------------------------------------------===//
// Reservation state helpers
//===----------------------------------------------------------------------===//

static void iree_hal_passthrough_pool_reservation_state_release_reference(
    iree_hal_passthrough_pool_reservation_state_t* reservation_state) {
  iree_hal_passthrough_pool_t* pool =
      (iree_hal_passthrough_pool_t*)reservation_state->pool;
  const int32_t previous_count = iree_atomic_fetch_sub(
      &reservation_state->reference_count, 1, iree_memory_order_acq_rel);
  IREE_ASSERT(previous_count > 0);
  if (previous_count != 1) return;

  iree_hal_slab_provider_release_slab(pool->slab_provider,
                                      &reservation_state->slab);
  iree_atomic_fetch_add(&pool->slab_count, -1, iree_memory_order_relaxed);
  iree_allocator_free(pool->host_allocator, reservation_state);
}

static void iree_hal_passthrough_pool_reservation_state_release_reservation(
    iree_hal_passthrough_pool_reservation_state_t* reservation_state) {
  iree_hal_passthrough_pool_t* pool =
      (iree_hal_passthrough_pool_t*)reservation_state->pool;
  const int32_t already_released = iree_atomic_exchange(
      &reservation_state->reservation_released, 1, iree_memory_order_acq_rel);
  IREE_ASSERT_EQ(already_released, 0);

  iree_atomic_fetch_add(&pool->bytes_reserved,
                        -(int64_t)reservation_state->reservation_length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->release_count, 1, iree_memory_order_relaxed);
  iree_async_notification_signal(pool->notification, INT32_MAX);

  iree_hal_passthrough_pool_reservation_state_release_reference(
      reservation_state);
}

static void iree_hal_passthrough_pool_borrowed_view_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_passthrough_pool_reservation_state_t* reservation_state =
      (iree_hal_passthrough_pool_reservation_state_t*)user_data;
  iree_hal_passthrough_pool_reservation_state_release_reference(
      reservation_state);
}

static void iree_hal_passthrough_pool_owned_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_passthrough_pool_reservation_state_t* reservation_state =
      (iree_hal_passthrough_pool_reservation_state_t*)user_data;
  iree_hal_passthrough_pool_reservation_state_release_reservation(
      reservation_state);
  iree_hal_passthrough_pool_reservation_state_release_reference(
      reservation_state);
}

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
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_passthrough_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  iree_hal_resource_initialize(&iree_hal_passthrough_pool_vtable,
                               &pool->resource);
  pool->host_allocator = host_allocator;

  iree_hal_slab_provider_retain(slab_provider);
  pool->slab_provider = slab_provider;
  iree_async_notification_retain(notification);
  pool->notification = notification;

  iree_hal_slab_provider_query_properties(slab_provider, &pool->memory_type,
                                          &pool->supported_usage);

  *out_pool = (iree_hal_pool_t*)pool;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_passthrough_pool_destroy(iree_hal_pool_t* base_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_async_notification_release(pool->notification);
  iree_hal_slab_provider_release(pool->slab_provider);
  iree_allocator_free(host_allocator, pool);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Reserve / Release
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_passthrough_pool_acquire_reservation(
    iree_hal_pool_t* base_pool, iree_device_size_t size,
    iree_device_size_t alignment,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;
  (void)alignment;
  (void)requester_frontier;

  iree_hal_slab_t slab;
  IREE_RETURN_IF_ERROR(
      iree_hal_slab_provider_acquire_slab(pool->slab_provider, size, &slab));

  iree_hal_passthrough_pool_reservation_state_t* reservation_state = NULL;
  iree_status_t status =
      iree_allocator_malloc(pool->host_allocator, sizeof(*reservation_state),
                            (void**)&reservation_state);
  if (!iree_status_is_ok(status)) {
    iree_hal_slab_provider_release_slab(pool->slab_provider, &slab);
    return status;
  }
  reservation_state->pool = base_pool;
  reservation_state->slab = slab;
  reservation_state->reservation_length = slab.length;
  iree_atomic_store(&reservation_state->reference_count, 1,
                    iree_memory_order_relaxed);
  iree_atomic_store(&reservation_state->reservation_released, 0,
                    iree_memory_order_relaxed);

  memset(out_reservation, 0, sizeof(*out_reservation));
  out_reservation->offset = 0;
  out_reservation->length = slab.length;
  out_reservation->block_handle = (uint64_t)(uintptr_t)reservation_state;

  iree_atomic_fetch_add(&pool->bytes_reserved, (int64_t)slab.length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reservation_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->slab_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, 1, iree_memory_order_relaxed);

  memset(out_info, 0, sizeof(*out_info));
  *out_result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  return iree_ok_status();
}

static void iree_hal_passthrough_pool_release_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  (void)base_pool;
  (void)death_frontier;

  iree_hal_passthrough_pool_reservation_state_t* reservation_state =
      (iree_hal_passthrough_pool_reservation_state_t*)(uintptr_t)
          reservation->block_handle;
  iree_hal_passthrough_pool_reservation_state_release_reservation(
      reservation_state);
}

//===----------------------------------------------------------------------===//
// Wrap / Query / Trim / Notification
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_passthrough_pool_materialize_reservation(
    iree_hal_pool_t* base_pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffer) {
  iree_hal_passthrough_pool_t* pool = (iree_hal_passthrough_pool_t*)base_pool;
  iree_hal_passthrough_pool_reservation_state_t* reservation_state =
      (iree_hal_passthrough_pool_reservation_state_t*)(uintptr_t)
          reservation->block_handle;

  iree_atomic_fetch_add(&reservation_state->reference_count, 1,
                        iree_memory_order_acq_rel);
  iree_hal_buffer_release_callback_t release_callback = {
      .fn = iree_hal_passthrough_pool_borrowed_view_release,
      .user_data = reservation_state,
  };
  if (iree_all_bits_set(
          flags,
          IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP)) {
    release_callback.fn = iree_hal_passthrough_pool_owned_buffer_release;
  }

  iree_status_t status = iree_hal_slab_provider_wrap_buffer(
      pool->slab_provider, &reservation_state->slab, reservation->offset,
      reservation->length, params, release_callback, out_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_passthrough_pool_reservation_state_release_reference(
        reservation_state);
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
  (void)base_pool;
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
    .acquire_reservation = iree_hal_passthrough_pool_acquire_reservation,
    .release_reservation = iree_hal_passthrough_pool_release_reservation,
    .materialize_reservation =
        iree_hal_passthrough_pool_materialize_reservation,
    .query_capabilities = iree_hal_passthrough_pool_query_capabilities,
    .query_stats = iree_hal_passthrough_pool_query_stats,
    .trim = iree_hal_passthrough_pool_trim,
    .notification = iree_hal_passthrough_pool_notification,
};
