// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/shared_wake.h"

// Carrier accessors defined in carrier.c. We use extern declarations rather
// than including carrier.h to avoid a circular build dependency (carrier.c
// includes shared_wake.h, and shared_wake.c must not include carrier.h).

// Returns the next carrier in the sleeping list.
extern iree_net_shm_carrier_t* iree_net_shm_carrier_sleeping_next(
    iree_net_shm_carrier_t* carrier);

// Sets the next carrier in the sleeping list.
extern void iree_net_shm_carrier_set_sleeping_next(
    iree_net_shm_carrier_t* carrier, iree_net_shm_carrier_t* next);

// Performs per-carrier drain logic when woken by the shared wake scan.
// Returns true if the carrier should remain in the sleeping list, false if
// it was removed (transitioned to poll mode or stopped).
extern bool iree_net_shm_carrier_drain_from_wake(
    iree_net_shm_carrier_t* carrier);

static void iree_net_shm_shared_wake_scan(void* user_data,
                                          iree_async_operation_t* operation,
                                          iree_status_t status,
                                          iree_async_completion_flags_t flags);

// Posts the NOTIFICATION_WAIT if not already posted and there are sleeping
// carriers.
static iree_status_t iree_net_shm_shared_wake_ensure_wait_posted(
    iree_net_shm_shared_wake_t* shared_wake) {
  if (shared_wake->wait_posted) return iree_ok_status();
  if (!shared_wake->sleeping_head) return iree_ok_status();

  memset(&shared_wake->wait_operation, 0, sizeof(shared_wake->wait_operation));
  iree_async_operation_initialize(&shared_wake->wait_operation.base,
                                  IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_net_shm_shared_wake_scan, shared_wake);
  shared_wake->wait_operation.notification = shared_wake->notification;
  iree_status_t status = iree_async_proactor_submit_one(
      shared_wake->proactor, &shared_wake->wait_operation.base);
  if (iree_status_is_ok(status)) {
    shared_wake->wait_posted = true;
  }
  return status;
}

// Scan callback: fires when the shared notification's NOTIFICATION_WAIT
// completes. Iterates all sleeping carriers, dispatches per-carrier drain,
// and re-posts the wait if carriers remain in the list.
static void iree_net_shm_shared_wake_scan(void* user_data,
                                          iree_async_operation_t* operation,
                                          iree_status_t status,
                                          iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_shm_shared_wake_t* shared_wake =
      (iree_net_shm_shared_wake_t*)user_data;
  iree_status_ignore(status);
  shared_wake->wait_posted = false;

  // Iterate sleeping list with save-next-before-processing: drain_from_wake
  // may cause the carrier to transition out of the sleeping list.
  iree_net_shm_carrier_t* carrier = shared_wake->sleeping_head;
  iree_net_shm_carrier_t* previous = NULL;
  while (carrier) {
    iree_net_shm_carrier_t* next = iree_net_shm_carrier_sleeping_next(carrier);
    bool keep_sleeping = iree_net_shm_carrier_drain_from_wake(carrier);
    if (keep_sleeping) {
      previous = carrier;
    } else {
      // Remove from sleeping list. Do NOT touch the carrier after this —
      // drain_from_wake may have completed deactivation inline, which fires
      // the user's deactivation callback. That callback may release/free the
      // carrier. (Same pattern as progress callback's deferred on_remove.)
      if (previous) {
        iree_net_shm_carrier_set_sleeping_next(previous, next);
      } else {
        shared_wake->sleeping_head = next;
      }
    }
    carrier = next;
  }

  // Re-post if carriers remain sleeping.
  if (shared_wake->sleeping_head) {
    iree_status_t wait_status =
        iree_net_shm_shared_wake_ensure_wait_posted(shared_wake);
    if (IREE_UNLIKELY(!iree_status_is_ok(wait_status))) {
      iree_status_ignore(wait_status);
      // Cannot re-post — all sleeping carriers are orphaned. This is a
      // catastrophic failure (fd/proactor exhaustion). Each carrier's
      // deactivation will eventually time out or be forced externally.
    }
  }
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_shm_shared_wake_create(
    iree_async_proactor_t* proactor, iree_allocator_t allocator,
    iree_net_shm_shared_wake_t** out_shared_wake) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_shared_wake);
  *out_shared_wake = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_shm_shared_wake_t* shared_wake = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*shared_wake),
                                (void**)&shared_wake));
  memset(shared_wake, 0, sizeof(*shared_wake));
  iree_atomic_ref_count_init(&shared_wake->ref_count);
  shared_wake->proactor = proactor;
  iree_async_proactor_retain(proactor);
  shared_wake->allocator = allocator;

  // Create a local notification on the proactor. This uses whatever the
  // proactor's optimal mode is (futex, eventfd, pipe, etc.).
  iree_status_t notification_status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &shared_wake->notification);
  if (!iree_status_is_ok(notification_status)) {
    iree_async_proactor_release(proactor);
    iree_allocator_free(allocator, shared_wake);
    IREE_TRACE_ZONE_END(z0);
    return notification_status;
  }

  *out_shared_wake = shared_wake;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_net_shm_shared_wake_retain(
    iree_net_shm_shared_wake_t* shared_wake) {
  if (shared_wake) {
    iree_atomic_ref_count_inc(&shared_wake->ref_count);
  }
}

IREE_API_EXPORT void iree_net_shm_shared_wake_release(
    iree_net_shm_shared_wake_t* shared_wake) {
  if (!shared_wake) return;
  if (iree_atomic_ref_count_dec(&shared_wake->ref_count) == 1) {
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_ASSERT(!shared_wake->sleeping_head,
                "shared_wake destroyed with carriers still in the sleeping "
                "list; unregister all carriers before releasing");
    IREE_ASSERT(!shared_wake->wait_posted,
                "shared_wake destroyed with NOTIFICATION_WAIT in flight");
    iree_async_notification_release(shared_wake->notification);
    iree_async_proactor_release(shared_wake->proactor);
    iree_allocator_t allocator = shared_wake->allocator;
    iree_allocator_free(allocator, shared_wake);
    IREE_TRACE_ZONE_END(z0);
  }
}

IREE_API_EXPORT iree_status_t iree_net_shm_shared_wake_register(
    iree_net_shm_shared_wake_t* shared_wake, iree_net_shm_carrier_t* carrier) {
  IREE_ASSERT_ARGUMENT(shared_wake);
  IREE_ASSERT_ARGUMENT(carrier);
  // Prepend to sleeping list.
  iree_net_shm_carrier_set_sleeping_next(carrier, shared_wake->sleeping_head);
  shared_wake->sleeping_head = carrier;
  // Ensure the NOTIFICATION_WAIT is posted to catch signals.
  iree_status_t status =
      iree_net_shm_shared_wake_ensure_wait_posted(shared_wake);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    // Remove the carrier we just prepended — it can never be woken without a
    // functioning NOTIFICATION_WAIT.
    shared_wake->sleeping_head = iree_net_shm_carrier_sleeping_next(carrier);
    iree_net_shm_carrier_set_sleeping_next(carrier, NULL);
  }
  return status;
}

IREE_API_EXPORT void iree_net_shm_shared_wake_unregister(
    iree_net_shm_shared_wake_t* shared_wake, iree_net_shm_carrier_t* carrier) {
  IREE_ASSERT_ARGUMENT(shared_wake);
  IREE_ASSERT_ARGUMENT(carrier);
  // Linear scan to find and remove. O(N) but N is small (typically 1-4
  // carriers per proactor) and this runs on the poll thread only.
  iree_net_shm_carrier_t* current = shared_wake->sleeping_head;
  iree_net_shm_carrier_t* previous = NULL;
  while (current) {
    if (current == carrier) {
      iree_net_shm_carrier_t* next =
          iree_net_shm_carrier_sleeping_next(carrier);
      if (previous) {
        iree_net_shm_carrier_set_sleeping_next(previous, next);
      } else {
        shared_wake->sleeping_head = next;
      }
      iree_net_shm_carrier_set_sleeping_next(carrier, NULL);
      return;
    }
    previous = current;
    current = iree_net_shm_carrier_sleeping_next(current);
  }
  // Not found — no-op (carrier may have already been removed by scan).
}
