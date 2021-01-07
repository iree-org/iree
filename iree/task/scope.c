// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/task/scope.h"

void iree_task_scope_initialize(iree_string_view_t name,
                                iree_task_scope_t* out_scope) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_scope, 0, sizeof(*out_scope));

  iree_host_size_t name_length =
      iree_min(name.size, IREE_ARRAYSIZE(out_scope->name) - 1);
  memcpy(out_scope->name, name.data, name_length);
  out_scope->name[name.size] = 0;

  // TODO(benvanik): pick trace colors based on name hash.
  IREE_TRACE(out_scope->task_trace_color = 0xFFFF0000u);

  iree_notification_initialize(&out_scope->idle_notification);

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_scope_deinitialize(iree_task_scope_t* scope) {
  IREE_TRACE_ZONE_BEGIN(z0);

  assert(iree_task_scope_is_idle(scope) &&
         "pending submissions must be aborted prior to deinitializing their "
         "scope");

  // Makes it easier to see if we were incorrectly using the name even after the
  // scope is deinitialized. Since scopes may be stack allocated we don't want
  // to have anyone trying to access them (like tracy).
  memset(scope->name, 0xCD, sizeof(scope->name));

  // In most cases the status will have been consumed by the scope owner.
  iree_status_t status = (iree_status_t)iree_atomic_exchange_ptr(
      &scope->permanent_status, (uintptr_t)NULL, iree_memory_order_acquire);
  IREE_IGNORE_ERROR(status);

  iree_notification_deinitialize(&scope->idle_notification);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_task_scope_try_set_status(iree_task_scope_t* scope,
                                           iree_status_t new_status) {
  if (IREE_UNLIKELY(iree_status_is_ok(new_status))) return;

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "failed: ");
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(new_status)));

  iree_status_t old_status = iree_ok_status();
  if (!iree_atomic_compare_exchange_strong_ptr(
          &scope->permanent_status, (uintptr_t*)&old_status,
          (uintptr_t)new_status, iree_memory_order_seq_cst,
          iree_memory_order_seq_cst)) {
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(new_status);
  }

  // TODO(#4026): poke to wake idle waiters.

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_scope_abort(iree_task_scope_t* scope) {
  iree_status_t status =
      iree_make_status(IREE_STATUS_ABORTED, "entire scope aborted by user");
  iree_task_scope_try_set_status(scope, status);
}

void iree_task_scope_fail(iree_task_scope_t* scope, iree_task_t* task,
                          iree_status_t status) {
  // TODO(benvanik): logging/tracing based on task.
  iree_task_scope_try_set_status(scope, status);
}

iree_status_t iree_task_scope_consume_status(iree_task_scope_t* scope) {
  iree_status_t old_status = iree_ok_status();
  iree_status_t new_status = iree_ok_status();
  while (!iree_atomic_compare_exchange_strong_ptr(
      &scope->permanent_status, (uintptr_t*)&old_status, (uintptr_t)new_status,
      iree_memory_order_seq_cst, iree_memory_order_seq_cst)) {
    // Previous status was not OK; we have it now though and can try again.
    new_status = iree_status_from_code(iree_status_code(new_status));
  }
  return old_status;
}

iree_task_dispatch_statistics_t iree_task_scope_consume_statistics(
    iree_task_scope_t* scope) {
  iree_task_dispatch_statistics_t result = scope->dispatch_statistics;
  memset(&scope->dispatch_statistics, 0, sizeof(scope->dispatch_statistics));
  return result;
}

bool iree_task_scope_is_idle(iree_task_scope_t* scope) {
  return iree_atomic_load_int32(&scope->pending_submissions,
                                iree_memory_order_relaxed) == 0;
}

iree_status_t iree_task_scope_wait_idle(iree_task_scope_t* scope,
                                        iree_time_t deadline_ns) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait for the scope to enter the idle state.
  // NOTE: we are currently ignoring |deadline_ns|.
  iree_notification_await(&scope->idle_notification,
                          (iree_condition_fn_t)iree_task_scope_is_idle, scope);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
