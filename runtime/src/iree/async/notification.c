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
  return iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
}

IREE_API_EXPORT void iree_async_notification_signal(
    iree_async_notification_t* notification, int32_t wake_count) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Increment epoch first — waiters check this after waking.
  iree_atomic_fetch_add(&notification->epoch, 1, iree_memory_order_release);

  // Platform-specific wakeup (futex, eventfd, pipe, etc.).
  notification->proactor->vtable->notification_signal(notification->proactor,
                                                      notification, wake_count);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT bool iree_async_notification_wait(
    iree_async_notification_t* notification, iree_timeout_t timeout) {
  IREE_TRACE_ZONE_BEGIN(z0);
  bool signaled = notification->proactor->vtable->notification_wait(
      notification->proactor, notification, timeout);
  IREE_TRACE_ZONE_END(z0);
  return signaled;
}
