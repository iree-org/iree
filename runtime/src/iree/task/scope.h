// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_SCOPE_H_
#define IREE_TASK_SCOPE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/synchronization.h"
#include "iree/task/task.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum iree_task_scope_flag_bits_e {
  IREE_TASK_SCOPE_FLAG_NONE = 0u,
  // Calls iree_status_abort on the first failure within the scope.
  // Hosting applications should properly handle the errors by retrieving the
  // failure status from the appropriate query or wait primitive.
  IREE_TASK_SCOPE_FLAG_ABORT_ON_FAILURE = 1u << 0,
};
typedef uint32_t iree_task_scope_flags_t;

// iree_task_scope_t is an atomic reference-counting helper posting a
// notification when the reference count is decremended to 0.
//
// It is used as a loose way of grouping tasks within the task system.
// Each scope represents a unique collection of tasks that have some related
// properties - most often their producer - that need to carry along some
// tracking information to act on all related tasks at once. They do not
// indicate any particular ordering of tasks or how the tasks are to be treated
// by executors.
//
// Scopes can be used to signal, propagate, and retrieve failure statuses. As
// the executor processes tasks in an unordered fashion this is the only way to
// perform cross-task operations such as "abort all of the tasks from this
// producer" or "wait until all tasks from this producer finish." In addition
// there are statistics that can be aggregated across all tasks attributed to
// the scope that allows for an efficient roll-up of activity over specific
// durations.
//
// Task producers can decide whether to create new scopes for each batch of
// tasks they submit or reuse scopes for the lifetime of their subprocess. Scope
// overhead is low and the only advantage of reusing them is that lifetime can
// become easier to manage by tying them 1:1 with producers.
//
// Thread-safe; once created scopes are modified exclusively via atomic
// operations.
typedef struct iree_task_scope_t {
  // Name used for logging and tracing.
  char name[16];

  // Flags controlling optional scope behavior.
  iree_task_scope_flags_t flags;

  // Base color used for tasks in this scope.
  // The color will be modulated based on task type.
  IREE_TRACE(uint32_t task_trace_color;)

  // A permanent status code set when a task within the scope fails. All pending
  // tasks will be aborted, though any in-flight tasks may continue executing
  // to completion.
  iree_atomic_intptr_t permanent_status;

  // Dispatch statistics aggregated from all dispatches in this scope. Updated
  // relatively infrequently and must not be used for task control as values
  // are undefined in the case of failure and may tear.
  iree_task_dispatch_statistics_t dispatch_statistics;

  // A count of pending submissions within this scope. 0 indicates idle.
  // Each submission has a fence that references this value and decrements it
  // as it is reached indicating that all memory used by all tasks within that
  // submission is available for reuse.
  iree_atomic_ref_count_t pending_submissions;

  // A notification signaled when the scope transitions to having no pending
  // tasks or completes all pending tasks after a failure.
  iree_notification_t idle_notification;
  iree_atomic_int32_t pending_idle_notification_posts;
} iree_task_scope_t;

// Initializes a caller-allocated scope.
// Callers must ensure the scope remains live for as long as there are any
// tasks that may reference it.
void iree_task_scope_initialize(iree_string_view_t name,
                                iree_task_scope_flags_t flags,
                                iree_task_scope_t* out_scope);

// Deinitializes an task scope.
// No tasks may be pending and the scope must be idle.
void iree_task_scope_deinitialize(iree_task_scope_t* scope);

// Returns the name of the scope. Informational only and may be the empty
// string.
iree_string_view_t iree_task_scope_name(iree_task_scope_t* scope);

// Returns and resets the statistics for the scope.
// Statistics may experience tearing (non-atomic update across fields) if this
// is performed while tasks are in-flight.
iree_task_dispatch_statistics_t iree_task_scope_consume_statistics(
    iree_task_scope_t* scope);

// Returns true if the scope has failed.
// iree_task_scope_consume_status can be used once to get the full status
// describing the failure and subsequent calls will return the status code.
bool iree_task_scope_has_failed(iree_task_scope_t* scope);

// Returns the permanent scope failure status to the caller (transfering
// ownership). The scope will remain in a failed state with the status code.
iree_status_t iree_task_scope_consume_status(iree_task_scope_t* scope);

// Marks the scope as having been aborted by the user with IREE_STATUS_ABORTED.
// All pending tasks will be dropped though in-flight tasks may complete
// execution. Callers must use iree_task_scope_wait_idle to ensure the scope
// state synchronizes prior to deinitializing. If the scope has already been
// aborted or failed with a permanent error then the operation is ignored and
// the previous error status is preserved.
void iree_task_scope_abort(iree_task_scope_t* scope);

// Marks the scope as having encountered an error while processing a task.
// The scope will be moved into a permanent failure state and all pending tasks
// will be aborted. In-flight tasks may continue executing prior to
// iree_task_scope_wait_idle returning true. If the scope has already been
// marked as failing then the status is ignored.
void iree_task_scope_fail(iree_task_scope_t* scope, iree_status_t status);

// Notifies the scope that a new execution task assigned to the scope has begun.
// The scope is considered active until it is notified execution has completed
// with iree_task_scope_end.
//
// Memory ordering: this does a iree_atomic_ref_count_inc.
// That typically means only a 'relaxed' order operation -- no ordering.
void iree_task_scope_begin(iree_task_scope_t* scope);

// Notifies the scope that a previously begun execution task has completed.
//
// Memory ordering: this does a iree_atomic_ref_count_dec.
// That means a guarantee that all writes made before calling
// iree_task_scope_end on one thread are visible to any other thread that has
// observed the reference count falling to zero (e.g. iree_task_scope_is_idle
// returned true).
void iree_task_scope_end(iree_task_scope_t* scope);

// Returns true if the scope has no pending or in-flight tasks.
//
// Memory ordering: this does iree_atomic_ref_count_load.
// That means a guarantee that subsequent memory read accesses can't be
// reordered before a call to iree_task_scope_is_idle, and that when
// iree_task_scope_is_idle returns true as a result of another thread
// having just called iree_task_scope_end, all memory writes made on that thread
// before calling iree_task_scope_end are visible on this thread after
// iree_task_scope_is_idle has returned.
bool iree_task_scope_is_idle(iree_task_scope_t* scope);

// Waits for the scope to become idle indicating that all pending and in-flight
// tasks have completed. If the scope is aborted or marked for permanent failure
// then the wait will only return after it is guaranteed no more tasks will ever
// be issued by the task system.
//
// Memory ordering: see iree_task_scope_is_idle.
iree_status_t iree_task_scope_wait_idle(iree_task_scope_t* scope,
                                        iree_time_t deadline_ns);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_SCOPE_H_
