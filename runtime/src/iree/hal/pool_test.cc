// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_async_proactor_t* test_proactor() {
  static iree_async_proactor_t* proactor = nullptr;
  if (!proactor) {
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
    atexit([] {
      iree_async_proactor_release(proactor);
      proactor = nullptr;
    });
  }
  return proactor;
}

typedef struct iree_hal_needs_wait_test_pool_t {
  iree_hal_resource_t resource;

  // Notification used by the generic synchronous allocation helper.
  iree_async_notification_t* notification = nullptr;

  iree_async_single_frontier_t wait_frontier;
  const iree_async_frontier_t* released_frontier = nullptr;
  iree_hal_pool_reservation_t released_reservation = {0};
  bool wrap_called = false;
} iree_hal_needs_wait_test_pool_t;

static void iree_hal_needs_wait_test_pool_destroy(iree_hal_pool_t* base_pool) {
  auto* pool = (iree_hal_needs_wait_test_pool_t*)base_pool;
  iree_async_notification_release(pool->notification);
  delete pool;
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
  auto* pool = (iree_hal_needs_wait_test_pool_t*)base_pool;
  return pool->notification;
}

static const iree_hal_pool_vtable_t iree_hal_needs_wait_test_pool_vtable = {
    iree_hal_needs_wait_test_pool_destroy,
    iree_hal_needs_wait_test_pool_acquire_reservation,
    iree_hal_needs_wait_test_pool_release_reservation,
    iree_hal_needs_wait_test_pool_materialize_reservation,
    iree_hal_needs_wait_test_pool_query_capabilities,
    iree_hal_needs_wait_test_pool_query_stats,
    iree_hal_needs_wait_test_pool_trim,
    iree_hal_needs_wait_test_pool_notification,
};

static iree_hal_needs_wait_test_pool_t* CreateNeedsWaitTestPool() {
  auto* pool = new iree_hal_needs_wait_test_pool_t;
  iree_hal_resource_initialize(&iree_hal_needs_wait_test_pool_vtable,
                               &pool->resource);
  IREE_CHECK_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &pool->notification));
  iree_async_single_frontier_initialize(
      &pool->wait_frontier, iree_async_axis_make_queue(1, 2, 3, 4), 42);
  pool->released_frontier = nullptr;
  memset(&pool->released_reservation, 0, sizeof(pool->released_reservation));
  pool->wrap_called = false;
  return pool;
}

typedef struct iree_hal_routing_test_pool_t {
  // Base resource header for vtable dispatch and ref counting.
  iree_hal_resource_t resource;

  // Capabilities returned when registering this pool in a pool set.
  iree_hal_pool_capabilities_t capabilities = {0};
} iree_hal_routing_test_pool_t;

static void iree_hal_routing_test_pool_destroy(iree_hal_pool_t* base_pool) {
  delete (iree_hal_routing_test_pool_t*)base_pool;
}

static iree_status_t iree_hal_routing_test_pool_acquire_reservation(
    iree_hal_pool_t* base_pool, iree_device_size_t size,
    iree_device_size_t alignment,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  (void)base_pool;
  (void)size;
  (void)alignment;
  (void)requester_frontier;
  (void)flags;
  (void)out_reservation;
  (void)out_info;
  (void)out_result;
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "acquire_reservation must not be called");
}

static void iree_hal_routing_test_pool_release_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  (void)base_pool;
  (void)reservation;
  (void)death_frontier;
}

static iree_status_t iree_hal_routing_test_pool_materialize_reservation(
    iree_hal_pool_t* base_pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffer) {
  (void)base_pool;
  (void)params;
  (void)reservation;
  (void)flags;
  (void)out_buffer;
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "materialize_reservation must not be called");
}

static void iree_hal_routing_test_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  auto* pool = (iree_hal_routing_test_pool_t*)base_pool;
  *out_capabilities = pool->capabilities;
}

static void iree_hal_routing_test_pool_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_stats) {
  (void)base_pool;
  memset(out_stats, 0, sizeof(*out_stats));
}

static iree_status_t iree_hal_routing_test_pool_trim(
    iree_hal_pool_t* base_pool) {
  (void)base_pool;
  return iree_ok_status();
}

static iree_async_notification_t* iree_hal_routing_test_pool_notification(
    iree_hal_pool_t* base_pool) {
  (void)base_pool;
  return NULL;
}

static const iree_hal_pool_vtable_t iree_hal_routing_test_pool_vtable = {
    iree_hal_routing_test_pool_destroy,
    iree_hal_routing_test_pool_acquire_reservation,
    iree_hal_routing_test_pool_release_reservation,
    iree_hal_routing_test_pool_materialize_reservation,
    iree_hal_routing_test_pool_query_capabilities,
    iree_hal_routing_test_pool_query_stats,
    iree_hal_routing_test_pool_trim,
    iree_hal_routing_test_pool_notification,
};

static iree_hal_routing_test_pool_t* CreateRoutingTestPool(
    iree_device_size_t max_allocation_size) {
  auto* pool = new iree_hal_routing_test_pool_t;
  iree_hal_resource_initialize(&iree_hal_routing_test_pool_vtable,
                               &pool->resource);
  pool->capabilities.memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  pool->capabilities.supported_usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  pool->capabilities.min_allocation_size = 0;
  pool->capabilities.max_allocation_size = max_allocation_size;
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

TEST(PoolSetTest, SelectsHighestPriorityCompatiblePoolBySize) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_pool_set_t pool_set;
  IREE_ASSERT_OK(iree_hal_pool_set_initialize(/*initial_capacity=*/2,
                                              host_allocator, &pool_set));

  iree_hal_routing_test_pool_t* direct_pool = CreateRoutingTestPool(0);
  iree_hal_routing_test_pool_t* tlsf_pool = CreateRoutingTestPool(1024);
  IREE_ASSERT_OK(
      iree_hal_pool_set_register(&pool_set, 0, (iree_hal_pool_t*)direct_pool));
  IREE_ASSERT_OK(
      iree_hal_pool_set_register(&pool_set, 10, (iree_hal_pool_t*)tlsf_pool));

  iree_hal_buffer_params_t params = {0};
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  EXPECT_EQ((iree_hal_pool_t*)tlsf_pool,
            iree_hal_pool_set_select(&pool_set, params, 512));
  EXPECT_EQ((iree_hal_pool_t*)direct_pool,
            iree_hal_pool_set_select(&pool_set, params, 2048));

  params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  EXPECT_EQ(nullptr, iree_hal_pool_set_select(&pool_set, params, 512));

  iree_hal_pool_set_deinitialize(&pool_set);
  iree_hal_pool_release((iree_hal_pool_t*)tlsf_pool);
  iree_hal_pool_release((iree_hal_pool_t*)direct_pool);
}

}  // namespace
