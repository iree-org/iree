// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/process.h"

#include <string.h>

#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// Process lifecycle
//===----------------------------------------------------------------------===//

void iree_task_process_initialize(iree_task_process_drain_fn_t drain_fn,
                                  int32_t suspend_count, int32_t wake_budget,
                                  iree_task_process_t* out_process) {
  IREE_ASSERT(drain_fn, "drain function must not be NULL");
  IREE_ASSERT(wake_budget >= 1, "wake budget must be at least 1");

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)suspend_count);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)wake_budget);

  // Zero non-atomic fields via memset so we don't risk forgetting one when new
  // fields are added. The subsequent assignments overwrite the fields that need
  // non-zero values.
  //
  // Atomic fields are initialized explicitly via atomic stores below rather
  // than relying on the memset: writing _Atomic fields through memset is
  // undefined behavior in C11, and while practically harmless when no other
  // thread can see the struct, there's no reason to tempt the optimizer.
  memset(out_process, 0, sizeof(*out_process));

  // Non-atomic fields (non-zero values only — rest are zeroed by memset).
  out_process->drain = drain_fn;

  // Atomic fields — all initialized explicitly regardless of value.
  iree_atomic_store(&out_process->suspend_count, suspend_count,
                    iree_memory_order_relaxed);
  iree_atomic_store(&out_process->state,
                    suspend_count > 0 ? IREE_TASK_PROCESS_STATE_SUSPENDED
                                      : IREE_TASK_PROCESS_STATE_RUNNABLE,
                    iree_memory_order_relaxed);
  iree_atomic_store(&out_process->error_status, 0, iree_memory_order_relaxed);
  iree_atomic_store(&out_process->wake_budget, wake_budget,
                    iree_memory_order_relaxed);
  iree_atomic_store(&out_process->placement_epoch, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&out_process->schedule_state, 0, iree_memory_order_relaxed);
  iree_atomic_store(&out_process->needs_drain, 0, iree_memory_order_relaxed);
  iree_atomic_store(&out_process->retain_drain, 0, iree_memory_order_relaxed);
  iree_atomic_store(&out_process->retention_epoch, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&out_process->warm_retainers, 0, iree_memory_order_relaxed);
  iree_atomic_store(&out_process->retention_sleepers, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&out_process->retention_spin_deadline_ns, 0,
                    iree_memory_order_relaxed);

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_process_set_flags(iree_task_process_t* process,
                                 iree_task_process_flags_t flags) {
  process->flags = flags;
}

bool iree_task_process_wake(iree_task_process_t* process) {
  int32_t previous = iree_atomic_fetch_sub(&process->suspend_count, 1,
                                           iree_memory_order_acq_rel);
  IREE_ASSERT(previous > 0,
              "wake called on process with suspend_count already at 0");
  if (previous != 1) return false;  // Still suspended.

  // We drove suspend_count to zero. Transition SUSPENDED -> RUNNABLE.
  // If the process was already cancelled or completed (e.g., by a concurrent
  // cancel call), the CAS fails and we return false — the process should not
  // be activated.
  int32_t expected = IREE_TASK_PROCESS_STATE_SUSPENDED;
  bool activated = iree_atomic_compare_exchange_strong(
      &process->state, &expected, (int32_t)IREE_TASK_PROCESS_STATE_RUNNABLE,
      iree_memory_order_acq_rel, iree_memory_order_acquire);

  IREE_TRACE({
    if (activated) {
      IREE_TRACE_ZONE_BEGIN(z0);
      IREE_TRACE_ZONE_END(z0);
    }
  });

  return activated;
}

// Resolves a terminated process: takes ownership of the error status, calls
// the completion callback, and resolves dependents. The process must already be
// in a terminal state.
//
// The completion callback is allowed to free the process, so all fields needed
// after the callback (dependents) are snapshotted before calling it.
static void iree_task_process_resolve(
    iree_task_process_t* process, iree_task_process_t** out_activated_head,
    iree_task_process_t** out_activated_tail) {
  *out_activated_head = NULL;
  *out_activated_tail = NULL;

  // Snapshot dependent state before the completion callback, which may free
  // the process.
  uint16_t dependent_count = process->dependent_count;
  iree_task_process_t** dependents = process->dependents;

  // Load (not exchange) the accumulated error status. The field retains its
  // non-zero value after this load, which is intentional: for wake_budget > 1
  // processes, a late report_error from a concurrent worker could CAS-succeed
  // if we cleared error_status to zero here, stranding a status that nobody
  // would ever consume. Leaving the old value in place means the CAS in
  // report_error sees non-zero and frees the late error.
  //
  // After this load, the status pointer in error_status may become dangling
  // (the completion callback frees the storage). This is safe because nothing
  // dereferences error_status — has_error is a != 0 check, and report_error
  // uses CAS(expected=0) which only compares the raw intptr_t value.
  iree_status_t status = (iree_status_t)iree_atomic_load(
      &process->error_status, iree_memory_order_acquire);

  // Call the completion callback, transferring status ownership.
  // After this call, |process| may be freed and must not be dereferenced.
  if (process->completion_fn) {
    process->completion_fn(process, status);
  } else {
    // No completion callback — we must consume the status ourselves. The
    // load (not exchange) above intentionally leaves the non-zero value in
    // error_status: report_error uses CAS(expected=0), so the stale non-zero
    // value causes late reports to fail the CAS and free their error rather
    // than stranding a status that nobody would ever consume.
    iree_status_free(status);
  }

  // Resolve dependents: wake each one, building a singly-linked list of
  // newly activated processes for the caller to push to run lists.
  for (uint16_t i = 0; i < dependent_count; ++i) {
    iree_task_process_t* dependent = dependents[i];
    if (iree_task_process_wake(dependent)) {
      iree_task_process_slist_set_next(dependent, NULL);
      if (*out_activated_tail) {
        iree_task_process_slist_set_next(*out_activated_tail, dependent);
      } else {
        *out_activated_head = dependent;
      }
      *out_activated_tail = dependent;
    }
  }
}

void iree_task_process_complete(iree_task_process_t* process,
                                iree_task_process_t** out_activated_head,
                                iree_task_process_t** out_activated_tail) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Transition to COMPLETED if not already terminal. report_error or cancel
  // may have already set a terminal state — preserve CANCELLED if so.
  if (!iree_task_process_is_terminal(process)) {
    iree_atomic_store(&process->state,
                      (int32_t)IREE_TASK_PROCESS_STATE_COMPLETED,
                      iree_memory_order_release);
  }

  // Peek the error status for trace annotation (before resolve exchanges it).
  IREE_TRACE({
    iree_status_t peek = (iree_status_t)iree_atomic_load(
        &process->error_status, iree_memory_order_relaxed);
    if (!iree_status_is_ok(peek)) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "error: ");
      IREE_TRACE_ZONE_APPEND_TEXT(
          z0, iree_status_code_string(iree_status_code(peek)));
    }
  });

  // Snapshot dependent count for trace before resolve (which may free process).
  IREE_TRACE(uint16_t dependent_count = process->dependent_count);

  iree_task_process_resolve(process, out_activated_head, out_activated_tail);

  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)dependent_count);
  IREE_TRACE_ZONE_END(z0);
}

void iree_task_process_report_error(iree_task_process_t* process,
                                    iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status));

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "failed: ");
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(status)));

  // Store as the first error. Only the first error wins.
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &process->error_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    // Another error was already recorded. Free ours.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "(freed, prior error exists)");
    iree_status_free(status);
  }

  // Transition RUNNABLE -> COMPLETED so future drain() calls bail via
  // is_terminal(). Only RUNNABLE is attempted — report_error must only be
  // called on a process that is being drained. To fail a SUSPENDED process,
  // use iree_task_process_cancel instead.
  int32_t runnable = IREE_TASK_PROCESS_STATE_RUNNABLE;
  iree_atomic_compare_exchange_strong(
      &process->state, &runnable, (int32_t)IREE_TASK_PROCESS_STATE_COMPLETED,
      iree_memory_order_release, iree_memory_order_relaxed);

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_process_cancel(iree_task_process_t* process,
                              iree_task_process_t** out_activated_head,
                              iree_task_process_t** out_activated_tail) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_activated_head = NULL;
  *out_activated_tail = NULL;

  // Record the cancellation error (if no error already stored).
  iree_status_t status = iree_status_from_code(IREE_STATUS_CANCELLED);
  intptr_t expected_status = 0;
  if (!iree_atomic_compare_exchange_strong(
          &process->error_status, &expected_status, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    // An error was already recorded. The cancellation piggybacks on that —
    // the process will complete with the original error.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "prior error exists");
    iree_status_free(status);
  }

  // Transition to CANCELLED from any non-terminal state.
  int32_t expected = IREE_TASK_PROCESS_STATE_SUSPENDED;
  if (iree_atomic_compare_exchange_strong(
          &process->state, &expected,
          (int32_t)IREE_TASK_PROCESS_STATE_CANCELLED, iree_memory_order_acq_rel,
          iree_memory_order_relaxed)) {
    // Was SUSPENDED: no worker will ever drain this process, so resolve it
    // inline (completion callback + dependent activation).
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "from SUSPENDED");
    iree_task_process_resolve(process, out_activated_head, out_activated_tail);
    IREE_TRACE_ZONE_END(z0);
    return;
  }
  expected = IREE_TASK_PROCESS_STATE_RUNNABLE;
  if (iree_atomic_compare_exchange_strong(
          &process->state, &expected,
          (int32_t)IREE_TASK_PROCESS_STATE_CANCELLED, iree_memory_order_acq_rel,
          iree_memory_order_relaxed)) {
    // Was RUNNABLE: workers will see the terminal state on their next drain()
    // call and complete the process through the normal path.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "from RUNNABLE");
    IREE_TRACE_ZONE_END(z0);
    return;
  }
  // Already terminal.
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "already terminal");
  IREE_TRACE_ZONE_END(z0);
}
