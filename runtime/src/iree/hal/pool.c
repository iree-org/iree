// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/pool.h"

#include <stddef.h>

#include "iree/async/notification.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

#define _VTABLE_DISPATCH(pool, method_name) \
  IREE_HAL_VTABLE_DISPATCH(pool, iree_hal_pool, method_name)

IREE_HAL_API_RETAIN_RELEASE(pool);

IREE_API_EXPORT iree_status_t
iree_hal_pool_reserve(iree_hal_pool_t* pool, iree_device_size_t size,
                      iree_device_size_t alignment,
                      const iree_async_frontier_t* requester_frontier,
                      iree_hal_pool_reservation_t* out_reservation,
                      iree_hal_pool_reserve_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_reservation);
  IREE_ASSERT_ARGUMENT(out_result);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(pool, reserve)(
      pool, size, alignment, requester_frontier, out_reservation, out_result);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_pool_release_reservation(
    iree_hal_pool_t* pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation);
  IREE_TRACE_ZONE_BEGIN(z0);
  _VTABLE_DISPATCH(pool, release_reservation)(pool, reservation,
                                              death_frontier);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_pool_wrap_reservation(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(pool, wrap_reservation)(
      pool, params, reservation, out_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_pool_query_capabilities(
    const iree_hal_pool_t* pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_capabilities);
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  _VTABLE_DISPATCH(pool, query_capabilities)(pool, out_capabilities);
}

IREE_API_EXPORT void iree_hal_pool_query_stats(
    const iree_hal_pool_t* pool, iree_hal_pool_stats_t* out_stats) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_stats);
  memset(out_stats, 0, sizeof(*out_stats));
  _VTABLE_DISPATCH(pool, query_stats)(pool, out_stats);
}

IREE_API_EXPORT iree_status_t iree_hal_pool_trim(iree_hal_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(pool, trim)(pool);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_async_notification_t* iree_hal_pool_notification(
    iree_hal_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return _VTABLE_DISPATCH(pool, notification)(pool);
}

IREE_API_EXPORT iree_status_t iree_hal_pool_allocate_buffer(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    const iree_async_frontier_t* requester_frontier, iree_timeout_t timeout,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Convert to absolute so retries after spurious wakes use a consistent
  // cutoff.
  iree_convert_timeout_to_absolute(&timeout);
  iree_async_notification_t* notification = iree_hal_pool_notification(pool);
  iree_status_t status = iree_ok_status();
  bool retry = true;
  while (retry) {
    iree_hal_pool_reservation_t reservation;
    iree_hal_pool_reserve_result_t result;
    status = iree_hal_pool_reserve(
        pool, allocation_size, params.min_alignment ? params.min_alignment : 1,
        requester_frontier, &reservation, &result);
    if (!iree_status_is_ok(status)) break;

    switch (result) {
      case IREE_HAL_POOL_RESERVE_OK:
      case IREE_HAL_POOL_RESERVE_OK_FRESH:
      case IREE_HAL_POOL_RESERVE_OK_NEEDS_WAIT:
        // Reservation succeeded — wrap it in a buffer.
        status = iree_hal_pool_wrap_reservation(pool, params, &reservation,
                                                out_buffer);
        if (!iree_status_is_ok(status)) {
          // Wrapping failed — release the reservation to avoid leaking the
          // offset back to the pool.
          iree_hal_pool_release_reservation(pool, &reservation, NULL);
        }
        retry = false;
        break;
      case IREE_HAL_POOL_RESERVE_EXHAUSTED:
      case IREE_HAL_POOL_RESERVE_OVER_BUDGET:
        // Wait for a release to signal the notification, then retry.
        if (!iree_async_notification_wait(notification, timeout)) {
          status = iree_make_status(
              IREE_STATUS_DEADLINE_EXCEEDED,
              "pool allocate_buffer timed out waiting for a free block (%s)",
              result == IREE_HAL_POOL_RESERVE_EXHAUSTED ? "exhausted"
                                                        : "over budget");
          retry = false;
        }
        break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
