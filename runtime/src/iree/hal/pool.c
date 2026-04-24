// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/pool.h"

#include <stddef.h>
#include <string.h>

#include "iree/async/notification.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

#define _VTABLE_DISPATCH(pool, method_name) \
  IREE_HAL_VTABLE_DISPATCH(pool, iree_hal_pool, method_name)

IREE_HAL_API_RETAIN_RELEASE(pool);

IREE_API_EXPORT iree_status_t iree_hal_pool_acquire_reservation(
    iree_hal_pool_t* pool, iree_device_size_t size,
    iree_device_size_t alignment,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_reservation);
  IREE_ASSERT_ARGUMENT(out_info);
  IREE_ASSERT_ARGUMENT(out_result);
  memset(out_info, 0, sizeof(*out_info));
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(pool, acquire_reservation)(
      pool, size, alignment, requester_frontier, flags, out_reservation,
      out_info, out_result);
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

IREE_API_EXPORT iree_status_t iree_hal_pool_materialize_reservation(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(pool, materialize_reservation)(
      pool, params, reservation, flags, out_buffer);
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
    const uint32_t wait_token =
        iree_async_notification_begin_observe(notification);

    iree_hal_pool_reservation_t reservation;
    iree_hal_pool_acquire_info_t acquire_info;
    iree_hal_pool_acquire_result_t result;
    status = iree_hal_pool_acquire_reservation(
        pool, allocation_size, params.min_alignment ? params.min_alignment : 1,
        requester_frontier, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
        &acquire_info, &result);
    if (iree_status_is_ok(status)) {
      switch (result) {
        case IREE_HAL_POOL_ACQUIRE_OK:
        case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
          // Reservation succeeded; transfer ownership to the returned buffer.
          status = iree_hal_pool_materialize_reservation(
              pool, params, &reservation,
              IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP,
              out_buffer);
          if (!iree_status_is_ok(status)) {
            // Wrapping failed; release the reservation to avoid leaking the
            // offset back to the pool.
            iree_hal_pool_release_reservation(pool, &reservation, NULL);
          }
          retry = false;
          break;
        case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
          // Synchronous allocation cannot model a hidden queue wait edge. A
          // pool used through this helper must skip non-dominated blocks and
          // return EXHAUSTED/OVER_BUDGET until an immediately-usable
          // reservation exists. Preserve the block's original frontier and
          // report a pool implementation bug, not a caller precondition
          // failure.
          iree_hal_pool_release_reservation(pool, &reservation,
                                            acquire_info.wait_frontier);
          status = iree_make_status(
              IREE_STATUS_INTERNAL,
              "iree_hal_pool_allocate_buffer received an "
              "IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT reservation from a pool "
              "that must only return immediately-usable reservations in the "
              "synchronous helper path");
          retry = false;
          break;
        case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
        case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
          // Wait for a release to advance the notification, then retry.
          if (!iree_async_notification_wait_for_token(notification, wait_token,
                                                      timeout)) {
            status = iree_make_status(
                IREE_STATUS_DEADLINE_EXCEEDED,
                "pool allocate_buffer timed out waiting for a free block (%s)",
                result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ? "exhausted"
                                                          : "over budget");
            retry = false;
          }
          break;
      }
    }
    iree_async_notification_end_observe(notification);
    if (!iree_status_is_ok(status)) {
      retry = false;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
