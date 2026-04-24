// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/notification.h"

#include "iree/async/proactor.h"

IREE_API_EXPORT iree_status_t iree_async_notification_create(
    iree_async_proactor_t* proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_notification);
  *out_notification = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      proactor->vtable->create_notification(proactor, flags, out_notification);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_async_notification_create_shared(
    iree_async_proactor_t* proactor,
    const iree_async_notification_shared_options_t* options,
    iree_async_notification_t** out_notification) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(options->epoch_address);
  IREE_ASSERT_ARGUMENT(out_notification);
  *out_notification = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = proactor->vtable->create_notification_shared(
      proactor, options, out_notification);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_async_notification_retain(
    iree_async_notification_t* notification) {
  iree_atomic_ref_count_inc(&notification->ref_count);
}

IREE_API_EXPORT void iree_async_notification_release(
    iree_async_notification_t* notification) {
  if (notification &&
      iree_atomic_ref_count_dec(&notification->ref_count) == 1) {
    notification->proactor->vtable->destroy_notification(notification->proactor,
                                                         notification);
  }
}

IREE_API_EXPORT uint32_t
iree_async_notification_query_epoch(iree_async_notification_t* notification) {
  return iree_atomic_load(notification->epoch_ptr, iree_memory_order_acquire);
}

IREE_API_EXPORT uint32_t
iree_async_notification_begin_observe(iree_async_notification_t* notification) {
  iree_atomic_fetch_add(&notification->observer_count, 1,
                        iree_memory_order_acq_rel);
  return iree_async_notification_query_epoch(notification);
}

IREE_API_EXPORT void iree_async_notification_end_observe(
    iree_async_notification_t* notification) {
  iree_atomic_fetch_sub(&notification->observer_count, 1,
                        iree_memory_order_acq_rel);
}

IREE_API_EXPORT void iree_async_notification_signal(
    iree_async_notification_t* notification, int32_t wake_count) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Increment epoch first — waiters check this after waking.
  iree_atomic_fetch_add(notification->epoch_ptr, 1, iree_memory_order_release);

  // Platform-specific wakeup (futex, eventfd, pipe, etc.).
  notification->proactor->vtable->notification_signal(notification->proactor,
                                                      notification, wake_count);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT bool iree_async_notification_signal_if_observed(
    iree_async_notification_t* notification, int32_t wake_count) {
  // Always advance the epoch so a waiter that has observed the token and is
  // between condition re-check and platform wait cannot miss the release. The
  // expensive platform wake is conditional on a known observer.
  iree_atomic_fetch_add(notification->epoch_ptr, 1, iree_memory_order_release);
  if (iree_atomic_load(&notification->observer_count,
                       iree_memory_order_acquire) <= 0) {
    return false;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  notification->proactor->vtable->notification_signal(notification->proactor,
                                                      notification, wake_count);
  IREE_TRACE_ZONE_END(z0);
  return true;
}

IREE_API_EXPORT bool iree_async_notification_wait(
    iree_async_notification_t* notification, iree_timeout_t timeout) {
  const uint32_t wait_token =
      iree_async_notification_begin_observe(notification);
  const bool signaled =
      iree_async_notification_wait_for_token(notification, wait_token, timeout);
  iree_async_notification_end_observe(notification);
  return signaled;
}

IREE_API_EXPORT bool iree_async_notification_wait_for_token(
    iree_async_notification_t* notification, uint32_t wait_token,
    iree_timeout_t timeout) {
  IREE_TRACE_ZONE_BEGIN(z0);
  bool signaled = notification->proactor->vtable->notification_wait(
      notification->proactor, notification, wait_token, timeout);
  IREE_TRACE_ZONE_END(z0);
  return signaled;
}
