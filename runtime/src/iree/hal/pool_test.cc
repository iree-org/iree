// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

typedef struct iree_hal_needs_wait_test_pool_t {
  iree_hal_resource_t resource;
  iree_async_single_frontier_t wait_frontier;
  const iree_async_frontier_t* released_frontier = nullptr;
  iree_hal_pool_reservation_t released_reservation = {0};
  bool wrap_called = false;
} iree_hal_needs_wait_test_pool_t;

static void iree_hal_needs_wait_test_pool_destroy(iree_hal_pool_t* base_pool) {
  delete (iree_hal_needs_wait_test_pool_t*)base_pool;
}

static iree_status_t iree_hal_needs_wait_test_pool_acquire_reservation(
    iree_hal_pool_t* base_pool, iree_device_size_t size,
    iree_device_size_t alignment,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  auto* pool = (iree_hal_needs_wait_test_pool_t*)base_pool;
  (void)alignment;
  (void)requester_frontier;
  (void)flags;
  iree_async_single_frontier_initialize(
      &pool->wait_frontier, iree_async_axis_make_queue(1, 2, 3, 4), 42);
  memset(out_reservation, 0, sizeof(*out_reservation));
  out_reservation->offset = 128;
  out_reservation->length = size;
  out_reservation->block_handle = 0xB10Cu;
  out_reservation->slab_index = 7;
  memset(out_info, 0, sizeof(*out_info));
  out_info->wait_frontier =
      iree_async_single_frontier_as_const_frontier(&pool->wait_frontier);
  *out_result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
  return iree_ok_status();
}

static void iree_hal_needs_wait_test_pool_release_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  auto* pool = (iree_hal_needs_wait_test_pool_t*)base_pool;
  pool->released_reservation = *reservation;
  pool->released_frontier = death_frontier;
}

static iree_status_t iree_hal_needs_wait_test_pool_materialize_reservation(
    iree_hal_pool_t* base_pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffer) {
  auto* pool = (iree_hal_needs_wait_test_pool_t*)base_pool;
  (void)params;
  (void)reservation;
  (void)flags;
  (void)out_buffer;
  pool->wrap_called = true;
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "materialize_reservation must not be called");
}

static void iree_hal_needs_wait_test_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  (void)base_pool;
  (void)out_capabilities;
}

static void iree_hal_needs_wait_test_pool_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_stats) {
  (void)base_pool;
  (void)out_stats;
}

static iree_status_t iree_hal_needs_wait_test_pool_trim(
    iree_hal_pool_t* base_pool) {
  (void)base_pool;
  return iree_ok_status();
}

static iree_async_notification_t* iree_hal_needs_wait_test_pool_notification(
    iree_hal_pool_t* base_pool) {
  (void)base_pool;
  return NULL;
}

static const iree_hal_pool_vtable_t iree_hal_needs_wait_test_pool_vtable = {
    .destroy = iree_hal_needs_wait_test_pool_destroy,
    .acquire_reservation = iree_hal_needs_wait_test_pool_acquire_reservation,
    .release_reservation = iree_hal_needs_wait_test_pool_release_reservation,
    .materialize_reservation =
        iree_hal_needs_wait_test_pool_materialize_reservation,
    .query_capabilities = iree_hal_needs_wait_test_pool_query_capabilities,
    .query_stats = iree_hal_needs_wait_test_pool_query_stats,
    .trim = iree_hal_needs_wait_test_pool_trim,
    .notification = iree_hal_needs_wait_test_pool_notification,
};

static iree_hal_needs_wait_test_pool_t* CreateNeedsWaitTestPool() {
  auto* pool = new iree_hal_needs_wait_test_pool_t;
  iree_hal_resource_initialize(&iree_hal_needs_wait_test_pool_vtable,
                               &pool->resource);
  iree_async_single_frontier_initialize(
      &pool->wait_frontier, iree_async_axis_make_queue(1, 2, 3, 4), 42);
  pool->released_frontier = nullptr;
  memset(&pool->released_reservation, 0, sizeof(pool->released_reservation));
  pool->wrap_called = false;
  return pool;
}

TEST(PoolAllocateBufferTest,
     DetectsUnexpectedNeedsWaitReservationAndPreservesFrontier) {
  iree_hal_needs_wait_test_pool_t* concrete_pool = CreateNeedsWaitTestPool();
  iree_hal_pool_t* pool = (iree_hal_pool_t*)concrete_pool;

  iree_hal_buffer_params_t params = {0};
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_pool_allocate_buffer(
                            pool, params, 4096,
                            iree_async_single_frontier_as_const_frontier(
                                &concrete_pool->wait_frontier),
                            iree_make_timeout_ms(0), &buffer));
  EXPECT_EQ(buffer, nullptr);
  EXPECT_FALSE(concrete_pool->wrap_called);
  EXPECT_EQ(concrete_pool->released_reservation.offset, 128u);
  EXPECT_EQ(concrete_pool->released_reservation.length, 4096u);
  EXPECT_EQ(concrete_pool->released_reservation.block_handle, 0xB10Cu);
  EXPECT_EQ(concrete_pool->released_reservation.slab_index, 7u);
  EXPECT_EQ(concrete_pool->released_frontier,
            iree_async_single_frontier_as_const_frontier(
                &concrete_pool->wait_frontier));

  iree_hal_pool_release(pool);
}

}  // namespace
