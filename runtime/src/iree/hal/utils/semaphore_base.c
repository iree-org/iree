// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/semaphore_base.h"

#include <stddef.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_semaphore_initialize(
    const iree_hal_semaphore_vtable_t* vtable, uint64_t initial_value,
    iree_host_size_t frontier_offset, uint8_t frontier_capacity,
    iree_hal_semaphore_t* out_semaphore) {
  IREE_ASSERT_ARGUMENT(vtable);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  // The HAL vtable has .async at offset 0 — toll-free cast to async vtable.
  iree_async_semaphore_initialize((const iree_async_semaphore_vtable_t*)vtable,
                                  initial_value, frontier_offset,
                                  frontier_capacity, &out_semaphore->async);
}

IREE_API_EXPORT void iree_hal_semaphore_deinitialize(
    iree_hal_semaphore_t* semaphore) {
  IREE_ASSERT_ARGUMENT(semaphore);
  iree_async_semaphore_deinitialize(&semaphore->async);
}

//===----------------------------------------------------------------------===//
// Bridge: old HAL timepoint API over async semaphore internals
//===----------------------------------------------------------------------===//

// Bridge adapter callback: translates the new async timepoint dispatch
// signature to the old HAL semaphore callback signature.
//
// The bridge timepoint (iree_hal_semaphore_timepoint_t) embeds the async
// timepoint (iree_async_semaphore_timepoint_t) at offset 0, so we cast
// directly from the async timepoint pointer to the bridge timepoint pointer.
//
// Fires under the async semaphore's lock for deferred timepoints
// (dispatch-under-lock), or outside the lock for immediately satisfied
// timepoints in acquire_timepoint.
static void iree_hal_semaphore_bridge_timepoint_callback(
    void* user_data, iree_async_semaphore_timepoint_t* async_timepoint,
    iree_status_t status) {
  (void)user_data;
  iree_hal_semaphore_timepoint_t* timepoint =
      (iree_hal_semaphore_timepoint_t*)async_timepoint;

  // Extract fields before modifications (old callback may reuse storage).
  iree_hal_semaphore_callback_t callback = timepoint->callback;
  iree_hal_semaphore_t* semaphore = timepoint->retained_semaphore;

  // Query the current timeline value for the old callback signature.
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &semaphore->async.timeline_value, iree_memory_order_acquire);
  iree_status_code_t status_code = iree_status_consume_code(status);

  // Mark the bridge timepoint as fired. This tells cancel_timepoint not to
  // double-release the semaphore, and allows the old callback to reuse the
  // timepoint storage.
  timepoint->retained_semaphore = NULL;
  memset(&timepoint->callback, 0, sizeof(timepoint->callback));

  // Issue the old-style callback (return value is ignored per old contract).
  iree_status_ignore(
      callback.fn(callback.user_data, semaphore, current_value, status_code));

  // Release the semaphore reference held by this bridge timepoint.
  // The caller of notify/dispatch always holds their own reference, so this
  // release will not drop the refcount to zero while the lock is held.
  iree_hal_semaphore_release(semaphore);
}

IREE_API_EXPORT void iree_hal_semaphore_acquire_timepoint(
    iree_hal_semaphore_t* semaphore, uint64_t minimum_value,
    iree_timeout_t timeout, iree_hal_semaphore_callback_t callback,
    iree_hal_semaphore_timepoint_t* out_timepoint) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_timepoint);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Set up bridge timepoint fields.
  out_timepoint->retained_semaphore = semaphore;
  iree_hal_semaphore_retain(semaphore);
  out_timepoint->deadline_ns = iree_timeout_as_deadline_ns(timeout);
  out_timepoint->callback = callback;

  // Set up the embedded async timepoint for registration with the async
  // semaphore. The bridge adapter translates the callback signature.
  out_timepoint->async.callback = iree_hal_semaphore_bridge_timepoint_callback;
  out_timepoint->async.user_data = NULL;

  // Register with the semaphore's timepoint list.
  // The callback may fire before this returns if the value is already reached
  // or the semaphore has already failed.
  iree_status_ignore(iree_async_semaphore_insert_timepoint(
      &semaphore->async, minimum_value, &out_timepoint->async));

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_semaphore_cancel_timepoint(
    iree_hal_semaphore_t* semaphore,
    iree_hal_semaphore_timepoint_t* timepoint) {
  if (!semaphore || !timepoint) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Remove from the semaphore's timepoint list (if still present).
  // With dispatch-under-lock, after this returns the callback has either not
  // started or has already completed — no ambiguity.
  iree_async_semaphore_remove_timepoint(&semaphore->async, &timepoint->async);

  // If the bridge callback hasn't fired (retained_semaphore still set),
  // release the semaphore ourselves. If the callback already fired, it
  // handled the release and set retained_semaphore to NULL.
  if (timepoint->retained_semaphore != NULL) {
    timepoint->retained_semaphore = NULL;
    iree_hal_semaphore_release(semaphore);
  }

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_semaphore_notify(
    iree_hal_semaphore_t* semaphore, uint64_t new_value,
    iree_status_code_t new_status_code) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Retain so the semaphore outlives timepoint dispatch. Bridge timepoint
  // callbacks release their retained reference, which could otherwise drop
  // the refcount to zero while dispatch_timepoints still holds the lock.
  iree_hal_semaphore_retain(semaphore);

  if (new_status_code == IREE_STATUS_OK) {
    // Sync the async timeline value for bridge-period drivers that
    // maintain their own current_value field. This is idempotent if the
    // driver already synced via the timeline sync one-liner in its signal
    // method.
    iree_atomic_store(&semaphore->async.timeline_value, (int64_t)new_value,
                      iree_memory_order_release);
    // Dispatch all timepoints whose minimum_value <= new_value.
    iree_async_semaphore_dispatch_timepoints(&semaphore->async, new_value);
  } else {
    // Sync failure status to the async semaphore (first failure wins via
    // CAS). The status is a bare code (no payload), so no ordering concern
    // for payload visibility.
    intptr_t expected = 0;
    iree_atomic_compare_exchange_strong(
        &semaphore->async.failure_status, &expected,
        (intptr_t)iree_status_from_code(new_status_code),
        iree_memory_order_release, iree_memory_order_relaxed);
    // Dispatch all pending timepoints with the failure status.
    iree_async_semaphore_dispatch_timepoints_failed(
        &semaphore->async, iree_status_from_code(new_status_code));
  }

  // May destroy the semaphore if this was the last reference.
  iree_hal_semaphore_release(semaphore);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_semaphore_poll(iree_hal_semaphore_t* semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);
  uint64_t value = 0;
  iree_status_t status = iree_hal_semaphore_query(semaphore, &value);
  iree_hal_semaphore_notify(semaphore, value, iree_status_consume_code(status));
  IREE_TRACE_ZONE_END(z0);
}
