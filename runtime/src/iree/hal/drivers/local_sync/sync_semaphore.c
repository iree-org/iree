// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_sync/sync_semaphore.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/hal/utils/semaphore_base.h"

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
  iree_host_size_t frontier_offset = 0, total_size = 0;
  iree_status_t status = iree_async_semaphore_layout(
      sizeof(*semaphore), 0, &frontier_offset, &total_size);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_initialize(&iree_hal_sync_semaphore_vtable,
                                  initial_value, frontier_offset, 0,
                                  &semaphore->base);
    semaphore->host_allocator = host_allocator;
    semaphore->shared_state = shared_state;

    *out_semaphore = &semaphore->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_sync_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_sync_semaphore_t* semaphore =
      iree_hal_sync_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static uint64_t iree_hal_sync_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  // Both fields are atomic — fully lock-free query.
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_hal_sync_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  iree_hal_sync_semaphore_t* semaphore =
      iree_hal_sync_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));

  // Advance the timeline (CAS) and merge frontier.
  iree_status_t status = iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier);
  if (!iree_status_is_ok(status)) return status;

  // Dispatch satisfied timepoints.
  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);

  // Post a global notification so that any waiter will wake.
  // TODO(#4680): make notifications per-semaphore; would make multi-wait
  // impossible with iree_notification_t and we'd have to use wait handles.
  iree_notification_post(&semaphore->shared_state->notification,
                         IREE_ALL_WAITERS);

  return iree_ok_status();
}

static void iree_hal_sync_semaphore_fail(iree_async_semaphore_t* base_semaphore,
                                         iree_status_t status) {
  iree_hal_sync_semaphore_t* semaphore =
      iree_hal_sync_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));

  // First failure wins via CAS. Clone for storage, pass original to dispatch.
  iree_status_t stored = iree_status_clone(status);
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &semaphore->base.async.failure_status, &expected, (intptr_t)stored,
          iree_memory_order_release, iree_memory_order_acquire)) {
    // Already failed — drop both the clone and the incoming status.
    iree_status_free(stored);
    iree_status_free(status);
    return;
  }

  // Dispatch all pending timepoints with the failure status.
  // Takes ownership of |status| (clones per-timepoint, frees original).
  iree_async_semaphore_dispatch_timepoints_failed(base_semaphore, status);

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
    // Advance the timeline (CAS) and merge frontier.
    status = iree_async_semaphore_advance_timeline(
        (iree_async_semaphore_t*)semaphore_list.semaphores[i],
        semaphore_list.payload_values[i], /*frontier=*/NULL);
    if (!iree_status_is_ok(status)) break;

    // Dispatch satisfied timepoints.
    iree_async_semaphore_dispatch_timepoints(
        (iree_async_semaphore_t*)semaphore_list.semaphores[i],
        semaphore_list.payload_values[i]);
  }

  // Notify all waiters that we've updated semaphores. They'll wake and check
  // to see if they are satisfied.
  // NOTE: we do this even if there was a failure as we may have signaled some
  // of the list.
  iree_notification_post(&shared_state->notification, IREE_ALL_WAITERS);

  return status;
}

typedef struct iree_hal_sync_semaphore_notify_state_t {
  iree_async_semaphore_t* async;
  uint64_t value;
} iree_hal_sync_semaphore_notify_state_t;

static bool iree_hal_sync_semaphore_is_signaled(
    iree_hal_sync_semaphore_notify_state_t* state) {
  iree_async_semaphore_t* async_sem = state->async;
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &async_sem->timeline_value, iree_memory_order_acquire);
  if (current_value >= state->value) return true;
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  return !iree_status_is_ok(failure);
}

static bool iree_hal_sync_semaphore_is_signaled_thunk(void* arg) {
  return iree_hal_sync_semaphore_is_signaled(
      (iree_hal_sync_semaphore_notify_state_t*)arg);
}

static iree_status_t iree_hal_sync_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_hal_wait_flags_t flags) {
  iree_hal_sync_semaphore_t* semaphore =
      iree_hal_sync_semaphore_cast(base_semaphore);
  iree_async_semaphore_t* async_sem = &semaphore->base.async;

  // Try to see if we can return immediately. Both fields are atomic.
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &async_sem->timeline_value, iree_memory_order_acquire);
  if (current_value >= value) return iree_ok_status();

  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }

  if (iree_timeout_is_immediate(timeout)) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  // TODO(#4680): we should be checking for DEADLINE_EXCEEDED here. This is
  // easy when it's iree_timeout_is_infinite (we can just use the notification
  // as below) but if it's an actual deadline we'll need to probably switch to
  // iree_wait_handle_t.

  // Perform wait on the global notification. Will wait forever.
  iree_hal_sync_semaphore_notify_state_t notify_state = {
      .async = async_sem,
      .value = value,
  };
  iree_notification_await(&semaphore->shared_state->notification,
                          iree_hal_sync_semaphore_is_signaled_thunk,
                          (void*)&notify_state, timeout);

  // Re-check after waking. Both fields are atomic.
  current_value = (uint64_t)iree_atomic_load(&async_sem->timeline_value,
                                             iree_memory_order_acquire);
  if (current_value >= value) return iree_ok_status();

  failure = (iree_status_t)iree_atomic_load(&async_sem->failure_status,
                                            iree_memory_order_acquire);
  return iree_status_from_code(!iree_status_is_ok(failure)
                                   ? IREE_STATUS_ABORTED
                                   : IREE_STATUS_DEADLINE_EXCEEDED);
}

// Returns true if any semaphore in the list has signaled (or failed).
static bool iree_hal_sync_semaphore_any_signaled(
    const iree_hal_semaphore_list_t* semaphore_list) {
  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    iree_async_semaphore_t* async_sem = &semaphore_list->semaphores[i]->async;
    uint64_t current_value = (uint64_t)iree_atomic_load(
        &async_sem->timeline_value, iree_memory_order_acquire);
    if (current_value >= semaphore_list->payload_values[i]) return true;
    iree_status_t failure = (iree_status_t)iree_atomic_load(
        &async_sem->failure_status, iree_memory_order_acquire);
    if (!iree_status_is_ok(failure)) return true;
  }
  return false;
}

static bool iree_hal_sync_semaphore_any_signaled_thunk(void* arg) {
  return iree_hal_sync_semaphore_any_signaled(
      (const iree_hal_semaphore_list_t*)arg);
}

// Returns true if all semaphores in the list have signaled (or any failed).
static bool iree_hal_sync_semaphore_all_signaled(
    const iree_hal_semaphore_list_t* semaphore_list) {
  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    iree_async_semaphore_t* async_sem = &semaphore_list->semaphores[i]->async;
    uint64_t current_value = (uint64_t)iree_atomic_load(
        &async_sem->timeline_value, iree_memory_order_acquire);
    if (current_value >= semaphore_list->payload_values[i]) continue;
    iree_status_t failure = (iree_status_t)iree_atomic_load(
        &async_sem->failure_status, iree_memory_order_acquire);
    if (iree_status_is_ok(failure)) return false;
  }
  return true;
}

static bool iree_hal_sync_semaphore_all_signaled_thunk(void* arg) {
  return iree_hal_sync_semaphore_all_signaled(
      (const iree_hal_semaphore_list_t*)arg);
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
    iree_async_semaphore_t* async_sem = &semaphore_list.semaphores[i]->async;
    uint64_t current_value = (uint64_t)iree_atomic_load(
        &async_sem->timeline_value, iree_memory_order_acquire);
    iree_status_t failure = (iree_status_t)iree_atomic_load(
        &async_sem->failure_status, iree_memory_order_acquire);
    iree_status_code_t status_code = iree_status_code(failure);
    if (status_code != IREE_STATUS_OK) {
      any_failed = true;
    } else if (current_value < semaphore_list.payload_values[i]) {
      all_signaled = false;
    } else {
      any_signaled = true;
    }
  }
  if (any_failed) {
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
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_hal_wait_flags_t flags) {
  if (semaphore_list.count == 0) {
    return iree_ok_status();
  } else if (semaphore_list.count == 1) {
    // Fast-path for a single semaphore.
    return iree_hal_semaphore_wait(semaphore_list.semaphores[0],
                                   semaphore_list.payload_values[0], timeout,
                                   flags);
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
  iree_notification_await(&shared_state->notification,
                          wait_mode == IREE_HAL_WAIT_MODE_ALL
                              ? iree_hal_sync_semaphore_all_signaled_thunk
                              : iree_hal_sync_semaphore_any_signaled_thunk,
                          (void*)&semaphore_list, iree_infinite_timeout());

  // We may have been successful - or may have a partial failure.
  iree_status_t status =
      iree_hal_sync_semaphore_result_from_state(wait_mode, semaphore_list);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_sync_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_sync_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint export is not yet implemented");
}

static const iree_hal_semaphore_vtable_t iree_hal_sync_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_sync_semaphore_destroy,
            .query = iree_hal_sync_semaphore_query,
            .signal = iree_hal_sync_semaphore_signal,
            .query_frontier = iree_hal_semaphore_default_query_frontier,
            .fail = iree_hal_sync_semaphore_fail,
            .acquire_timepoint = iree_hal_semaphore_default_acquire_timepoint,
            .cancel_timepoint = iree_hal_semaphore_default_cancel_timepoint,
            .export_primitive = iree_hal_semaphore_default_export_primitive,
        },
    .wait = iree_hal_sync_semaphore_wait,
    .import_timepoint = iree_hal_sync_semaphore_import_timepoint,
    .export_timepoint = iree_hal_sync_semaphore_export_timepoint,
};
