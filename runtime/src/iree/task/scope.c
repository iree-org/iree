// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/scope.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/threading.h"

void iree_task_scope_initialize(iree_string_view_t name,
                                iree_task_scope_flags_t flags,
                                iree_task_scope_t* out_scope) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_scope, 0, sizeof(*out_scope));
  iree_atomic_ref_count_init_value(&out_scope->pending_submissions, 0);

  iree_host_size_t name_length =
      iree_min(name.size, IREE_ARRAYSIZE(out_scope->name) - 1);
  memcpy(out_scope->name, name.data, name_length);
  out_scope->name[name_length] = 0;

  out_scope->flags = flags;

  // TODO(benvanik): pick trace colors based on name hash.
  IREE_TRACE(out_scope->task_trace_color = 0xFFFF0000u);

  iree_notification_initialize(&out_scope->idle_notification);

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_scope_deinitialize(iree_task_scope_t* scope) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT(
      iree_task_scope_is_idle(scope),
      "pending submissions must be aborted prior to deinitializing their "
      "scope");

  // Makes it easier to see if we were incorrectly using the name even after the
  // scope is deinitialized. Since scopes may be stack allocated we don't want
  // to have anyone trying to access them (like tracy).
  memset(scope->name, 0xCD, sizeof(scope->name));

  // In most cases the status will have been consumed by the scope owner.
  iree_status_t status = (iree_status_t)iree_atomic_exchange_intptr(
      &scope->permanent_status, (intptr_t)NULL, iree_memory_order_acquire);
  IREE_IGNORE_ERROR(status);

  while (iree_atomic_load_int32(&scope->pending_idle_notification_posts,
                                iree_memory_order_acquire)) {
    iree_thread_yield();
  }
  iree_notification_deinitialize(&scope->idle_notification);

  IREE_TRACE_ZONE_END(z0);
}

iree_string_view_t iree_task_scope_name(iree_task_scope_t* scope) {
  return iree_make_cstring_view(scope->name);
}

iree_task_dispatch_statistics_t iree_task_scope_consume_statistics(
    iree_task_scope_t* scope) {
  iree_task_dispatch_statistics_t result = scope->dispatch_statistics;
  memset(&scope->dispatch_statistics, 0, sizeof(scope->dispatch_statistics));
  return result;
}

bool iree_task_scope_has_failed(iree_task_scope_t* scope) {
  return iree_atomic_load_intptr(&scope->permanent_status,
                                 iree_memory_order_acquire) != 0;
}

iree_status_t iree_task_scope_consume_status(iree_task_scope_t* scope) {
  iree_status_t old_status = iree_ok_status();
  iree_status_t new_status = iree_ok_status();
  while (!iree_atomic_compare_exchange_strong_intptr(
      &scope->permanent_status, (intptr_t*)&old_status, (intptr_t)new_status,
      iree_memory_order_acq_rel,
      iree_memory_order_acquire /* old_status is actually used */)) {
    // Previous status was not OK; we have it now though and can try again.
    new_status = iree_status_from_code(iree_status_code(old_status));
  }
  // If old_status is not iree_ok_status then it was obtained through the
  // comparison-failed mode of the above compare_exchange, which loaded it with
  // iree_memory_order_acquire. This guarantees that if we are returning a
  // failure status to the caller then all past memory operations are already
  // visible, such as any information attached to that failure status.
  return old_status;
}

static void iree_task_scope_try_set_status(iree_task_scope_t* scope,
                                           iree_status_t new_status) {
  if (IREE_UNLIKELY(iree_status_is_ok(new_status))) return;

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "failed: ");
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(new_status)));

  // Pretty-print and abort() the program to make it easier to find the stack
  // of an asynchronous queue failure. Hosting applications should properly
  // handle the errors by retrieving the failure status from the appropriate
  // query or wait primitive.
  if (iree_all_bits_set(scope->flags, IREE_TASK_SCOPE_FLAG_ABORT_ON_FAILURE)) {
    iree_status_abort(new_status);
  }

  iree_status_t old_status = iree_ok_status();
  if (!iree_atomic_compare_exchange_strong_intptr(
          &scope->permanent_status, (intptr_t*)&old_status,
          (intptr_t)new_status, iree_memory_order_acq_rel,
          iree_memory_order_relaxed /* old_status is unused */)) {
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(new_status);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_scope_abort(iree_task_scope_t* scope) {
  iree_status_t status =
      iree_make_status(IREE_STATUS_ABORTED, "entire scope aborted by user");
  iree_task_scope_try_set_status(scope, status);
}

void iree_task_scope_fail(iree_task_scope_t* scope, iree_status_t status) {
  iree_task_scope_try_set_status(scope, status);
}

void iree_task_scope_begin(iree_task_scope_t* scope) {
  iree_atomic_ref_count_inc(&scope->pending_submissions);
  // relaxed because this 'begin' call will be paired with a 'end' call that
  // will perform the release-store, and this value is only read by
  // 'deinitialize'.
  iree_atomic_store_int32(&scope->pending_idle_notification_posts, 1,
                          iree_memory_order_relaxed);
}

void iree_task_scope_end(iree_task_scope_t* scope) {
  if (iree_atomic_ref_count_dec(&scope->pending_submissions) == 1) {
    // All submissions have completed in this scope - notify any waiters.
    iree_notification_post(&scope->idle_notification, IREE_ALL_WAITERS);
    iree_atomic_store_int32(&scope->pending_idle_notification_posts, 0,
                            iree_memory_order_release);
  }
}

bool iree_task_scope_is_idle(iree_task_scope_t* scope) {
  return (iree_atomic_ref_count_load(&scope->pending_submissions) == 0);
}

iree_status_t iree_task_scope_wait_idle(iree_task_scope_t* scope,
                                        iree_time_t deadline_ns) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    // Polling for idle.
    if (iree_task_scope_is_idle(scope)) {
      status = iree_ok_status();
    } else {
      status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
  } else {
    // Wait for the scope to enter the idle state.
    if (!iree_notification_await(&scope->idle_notification,
                                 (iree_condition_fn_t)iree_task_scope_is_idle,
                                 scope, iree_make_deadline(deadline_ns))) {
      status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
