// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_sync/sync_semaphore.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/tracing.h"
#include "iree/hal/utils/semaphore_base.h"

// Sentinel used the semaphore has failed and an error status is set.
#define IREE_HAL_SYNC_SEMAPHORE_FAILURE_VALUE UINT64_MAX

//===----------------------------------------------------------------------===//
// iree_hal_sync_semaphore_state_t
//===----------------------------------------------------------------------===//

void iree_hal_sync_semaphore_state_initialize(
    iree_hal_sync_semaphore_state_t* out_shared_state) {
  memset(out_shared_state, 0, sizeof(*out_shared_state));
  iree_notification_initialize(&out_shared_state->notification);
}

void iree_hal_sync_semaphore_state_deinitialize(
    iree_hal_sync_semaphore_state_t* shared_state) {
  iree_notification_deinitialize(&shared_state->notification);
  memset(shared_state, 0, sizeof(*shared_state));
}

//===----------------------------------------------------------------------===//
// iree_hal_sync_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_sync_semaphore_t {
  iree_hal_semaphore_t base;
  iree_allocator_t host_allocator;

  // Shared across all semaphores.
  iree_hal_sync_semaphore_state_t* shared_state;

  // Guards all mutable fields. We expect low contention on semaphores and since
  // iree_slim_mutex_t is (effectively) just a CAS this keeps things simpler
  // than trying to make the entire structure lock-free.
  iree_slim_mutex_t mutex;

  // Current signaled value. May be IREE_HAL_SYNC_SEMAPHORE_FAILURE_VALUE to
  // indicate that the semaphore has been signaled for failure and
  // |failure_status| contains the error.
  uint64_t current_value;

  // OK or the status passed to iree_hal_semaphore_fail. Owned by the semaphore.
  iree_status_t failure_status;
} iree_hal_sync_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_sync_semaphore_vtable;

static iree_hal_sync_semaphore_t* iree_hal_sync_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_sync_semaphore_vtable);
  return (iree_hal_sync_semaphore_t*)base_value;
}

iree_status_t iree_hal_sync_semaphore_create(
    iree_hal_sync_semaphore_state_t* shared_state, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(shared_state);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_sync_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_initialize(&iree_hal_sync_semaphore_vtable,
                                  &semaphore->base);
    semaphore->host_allocator = host_allocator;
    semaphore->shared_state = shared_state;

    iree_slim_mutex_initialize(&semaphore->mutex);
    semaphore->current_value = initial_value;
    semaphore->failure_status = iree_ok_status();

    *out_semaphore = &semaphore->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_sync_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_sync_semaphore_t* semaphore =
      iree_hal_sync_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_deinitialize(&semaphore->mutex);
  iree_status_ignore(semaphore->failure_status);

  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_sync_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_sync_semaphore_t* semaphore =
      iree_hal_sync_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);

  *out_value = semaphore->current_value;

  iree_status_t status = iree_ok_status();
  if (*out_value >= IREE_HAL_SYNC_SEMAPHORE_FAILURE_VALUE) {
    status = iree_status_clone(semaphore->failure_status);
  }

  iree_slim_mutex_unlock(&semaphore->mutex);

  return status;
}

// Signals |semaphore| to |new_value| or returns an error if doing so would be
// invalid. The semaphore mutex must be held.
static iree_status_t iree_hal_sync_semaphore_signal_unsafe(
    iree_hal_sync_semaphore_t* semaphore, uint64_t new_value) {
  if (new_value <= semaphore->current_value) {
    uint64_t current_value IREE_ATTRIBUTE_UNUSED = semaphore->current_value;
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "semaphore values must be monotonically "
                            "increasing; current_value=%" PRIu64
                            ", new_value=%" PRIu64,
                            current_value, new_value);
  }

  // Update to the new value.
  semaphore->current_value = new_value;

  return iree_ok_status();
}

static iree_status_t iree_hal_sync_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_sync_semaphore_t* semaphore =
      iree_hal_sync_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);

  iree_status_t status =
      iree_hal_sync_semaphore_signal_unsafe(semaphore, new_value);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&semaphore->mutex);
    return status;
  }

  iree_slim_mutex_unlock(&semaphore->mutex);

  // Notify timepoints of the new value.
  iree_hal_semaphore_notify(&semaphore->base, new_value, IREE_STATUS_OK);

  // Post a global notification so that any waiter will wake.
  // TODO(#4680): make notifications per-semaphore; would make multi-wait
  // impossible with iree_notification_t and we'd have to use wait handles.
  iree_notification_post(&semaphore->shared_state->notification,
                         IREE_ALL_WAITERS);

  return iree_ok_status();
}

static void iree_hal_sync_semaphore_fail(iree_hal_semaphore_t* base_semaphore,
                                         iree_status_t status) {
  iree_hal_sync_semaphore_t* semaphore =
      iree_hal_sync_semaphore_cast(base_semaphore);
  const iree_status_code_t status_code = iree_status_code(status);

  iree_slim_mutex_lock(&semaphore->mutex);

  // Try to set our local status - we only preserve the first failure so only
  // do this if we are going from a valid semaphore to a failed one.
  if (!iree_status_is_ok(semaphore->failure_status)) {
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(status);
    iree_slim_mutex_unlock(&semaphore->mutex);
    return;
  }

  // Signal to our failure sentinel value.
  semaphore->current_value = IREE_HAL_SYNC_SEMAPHORE_FAILURE_VALUE;
  semaphore->failure_status = status;

  iree_slim_mutex_unlock(&semaphore->mutex);

  // Notify timepoints of the failure.
  iree_hal_semaphore_notify(&semaphore->base,
                            IREE_HAL_SYNC_SEMAPHORE_FAILURE_VALUE, status_code);

  iree_notification_post(&semaphore->shared_state->notification,
                         IREE_ALL_WAITERS);
}

iree_status_t iree_hal_sync_semaphore_multi_signal(
    iree_hal_sync_semaphore_state_t* shared_state,
    const iree_hal_semaphore_list_t semaphore_list) {
  IREE_ASSERT_ARGUMENT(shared_state);
  if (semaphore_list.count == 0) {
    return iree_ok_status();
  } else if (semaphore_list.count == 1) {
    // Fast-path for a single semaphore.
    return iree_hal_semaphore_signal(semaphore_list.semaphores[0],
                                     semaphore_list.payload_values[0]);
  }

  // Try to signal all semaphores, stopping if we encounter any issues.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_sync_semaphore_t* semaphore =
        iree_hal_sync_semaphore_cast(semaphore_list.semaphores[i]);

    iree_slim_mutex_lock(&semaphore->mutex);
    status = iree_hal_sync_semaphore_signal_unsafe(
        semaphore, semaphore_list.payload_values[i]);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&semaphore->mutex);
      break;
    }

    iree_slim_mutex_unlock(&semaphore->mutex);

    // Notify timepoints that the new value has been reached.
    iree_hal_semaphore_notify(semaphore_list.semaphores[i],
                              semaphore_list.payload_values[i], IREE_STATUS_OK);
  }

  // Notify all waiters that we've updated semaphores. They'll wake and check
  // to see if they are satisfied.
  // NOTE: we do this even if there was a failure as we may have signaled some
  // of the list.
  iree_notification_post(&shared_state->notification, IREE_ALL_WAITERS);

  return status;
}

typedef struct iree_hal_sync_semaphore_notify_state_t {
  iree_hal_sync_semaphore_t* semaphore;
  uint64_t value;
} iree_hal_sync_semaphore_notify_state_t;

static bool iree_hal_sync_semaphore_is_signaled(
    iree_hal_sync_semaphore_notify_state_t* state) {
  iree_hal_sync_semaphore_t* semaphore = state->semaphore;
  iree_slim_mutex_lock(&semaphore->mutex);
  bool is_signaled = semaphore->current_value >= state->value ||
                     !iree_status_is_ok(semaphore->failure_status);
  iree_slim_mutex_unlock(&semaphore->mutex);
  return is_signaled;
}

static iree_status_t iree_hal_sync_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_sync_semaphore_t* semaphore =
      iree_hal_sync_semaphore_cast(base_semaphore);

  // Try to see if we can return immediately.
  iree_slim_mutex_lock(&semaphore->mutex);
  if (!iree_status_is_ok(semaphore->failure_status)) {
    // Fastest path: failed; return an error to tell callers to query for it.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_status_from_code(IREE_STATUS_ABORTED);
  } else if (semaphore->current_value >= value) {
    // Fast path: already satisfied.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_ok_status();
  } else if (iree_timeout_is_immediate(timeout)) {
    // Not satisfied but a poll, so can avoid the expensive wait handle work.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }
  iree_slim_mutex_unlock(&semaphore->mutex);

  // TODO(#4680): we should be checking for DEADLINE_EXCEEDED here. This is
  // easy when it's iree_timeout_is_infinite (we can just use the notification
  // as below) but if it's an actual deadline we'll need to probably switch to
  // iree_wait_handle_t.

  // Perform wait on the global notification. Will wait forever.
  iree_hal_sync_semaphore_state_t* shared_state = semaphore->shared_state;
  iree_hal_sync_semaphore_notify_state_t notify_state = {
      .semaphore = semaphore,
      .value = value,
  };
  iree_notification_await(
      &shared_state->notification,
      (iree_condition_fn_t)iree_hal_sync_semaphore_is_signaled,
      (void*)&notify_state, timeout);

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&semaphore->mutex);
  if (!iree_status_is_ok(semaphore->failure_status)) {
    // Semaphore has failed.
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  } else if (semaphore->current_value < value) {
    // Deadline expired before the semaphore was signaled.
    status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }
  iree_slim_mutex_unlock(&semaphore->mutex);
  return status;
}

// Returns true if any semaphore in the list has signaled (or failed).
// Used with with iree_condition_fn_t and must match that signature.
static bool iree_hal_sync_semaphore_any_signaled(
    const iree_hal_semaphore_list_t* semaphore_list) {
  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    iree_hal_sync_semaphore_t* semaphore =
        iree_hal_sync_semaphore_cast(semaphore_list->semaphores[i]);
    iree_slim_mutex_lock(&semaphore->mutex);
    bool is_signaled =
        semaphore->current_value >= semaphore_list->payload_values[i] ||
        !iree_status_is_ok(semaphore->failure_status);
    iree_slim_mutex_unlock(&semaphore->mutex);
    if (is_signaled) return true;
  }
  return false;
}

// Returns true if all semaphores in the list has signaled (or any failed).
// Used with with iree_condition_fn_t and must match that signature.
static bool iree_hal_sync_semaphore_all_signaled(
    const iree_hal_semaphore_list_t* semaphore_list) {
  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    iree_hal_sync_semaphore_t* semaphore =
        iree_hal_sync_semaphore_cast(semaphore_list->semaphores[i]);
    iree_slim_mutex_lock(&semaphore->mutex);
    bool is_signaled =
        semaphore->current_value >= semaphore_list->payload_values[i] ||
        !iree_status_is_ok(semaphore->failure_status);
    iree_slim_mutex_unlock(&semaphore->mutex);
    if (!is_signaled) return false;
  }
  return true;
}

// Returns a status derived from the |semaphore_list| at the current time:
// - IREE_STATUS_OK: any or all semaphores signaled (based on |wait_mode|).
// - IREE_STATUS_ABORTED: one or more semaphores failed.
// - IREE_STATUS_DEADLINE_EXCEEDED: any or all semaphores unsignaled.
static iree_status_t iree_hal_sync_semaphore_result_from_state(
    iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list) {
  bool any_signaled = false;
  bool all_signaled = true;
  bool any_failed = false;
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_sync_semaphore_t* semaphore =
        iree_hal_sync_semaphore_cast(semaphore_list.semaphores[i]);
    iree_slim_mutex_lock(&semaphore->mutex);
    const uint64_t current_value = semaphore->current_value;
    const iree_status_code_t current_status_code =
        iree_status_code(semaphore->failure_status);
    if (current_status_code != IREE_STATUS_OK) {
      // Semaphore has failed.
      any_failed = true;
    } else if (current_value < semaphore_list.payload_values[i]) {
      // Deadline expired before the semaphore was signaled.
      all_signaled = false;
    } else {
      // Signaled!
      any_signaled = true;
    }
    iree_slim_mutex_unlock(&semaphore->mutex);
  }
  if (any_failed) {
    // Always prioritize failure state.
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }
  switch (wait_mode) {
    default:
    case IREE_HAL_WAIT_MODE_ALL:
      return all_signaled
                 ? iree_ok_status()
                 : iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    case IREE_HAL_WAIT_MODE_ANY:
      return any_signaled
                 ? iree_ok_status()
                 : iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }
}

iree_status_t iree_hal_sync_semaphore_multi_wait(
    iree_hal_sync_semaphore_state_t* shared_state,
    iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  if (semaphore_list.count == 0) {
    return iree_ok_status();
  } else if (semaphore_list.count == 1) {
    // Fast-path for a single semaphore.
    return iree_hal_semaphore_wait(semaphore_list.semaphores[0],
                                   semaphore_list.payload_values[0], timeout);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Fast-path for polling; we'll never wait and can just do a quick query.
  if (iree_timeout_is_immediate(timeout)) {
    iree_status_t status =
        iree_hal_sync_semaphore_result_from_state(wait_mode, semaphore_list);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Perform wait on the global notification.
  iree_notification_await(
      &shared_state->notification,
      wait_mode == IREE_HAL_WAIT_MODE_ALL
          ? (iree_condition_fn_t)iree_hal_sync_semaphore_all_signaled
          : (iree_condition_fn_t)iree_hal_sync_semaphore_any_signaled,
      (void*)&semaphore_list, iree_infinite_timeout());

  // We may have been successful - or may have a partial failure.
  iree_status_t status =
      iree_hal_sync_semaphore_result_from_state(wait_mode, semaphore_list);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_semaphore_vtable_t iree_hal_sync_semaphore_vtable = {
    .destroy = iree_hal_sync_semaphore_destroy,
    .query = iree_hal_sync_semaphore_query,
    .signal = iree_hal_sync_semaphore_signal,
    .fail = iree_hal_sync_semaphore_fail,
    .wait = iree_hal_sync_semaphore_wait,
};
