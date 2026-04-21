// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Process: the universal work unit for the cooperative task scheduler.
//
// A process is any unit of work that can be drained incrementally by one or
// more workers. Workers call drain() on processes, do bounded work, and
// return — freeing them to handle other work (I/O completions, higher-priority
// processes, newly-arrived submissions) before coming back.
//
// Processes are activated when their suspend_count reaches zero. External
// events (semaphore signals, upstream process completion, user submission)
// decrement the suspend_count. The thread that drives it to zero pushes the
// process onto a run list and wakes workers.
//
// See runtime/src/iree/task/README.md for the full design.

#ifndef IREE_TASK_PROCESS_H_
#define IREE_TASK_PROCESS_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/cpu.h"
#include "iree/base/threading/futex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_task_process_t
//===----------------------------------------------------------------------===//

typedef struct iree_task_process_t iree_task_process_t;

// Result of a single drain() call on a process.
typedef struct iree_task_process_drain_result_t {
  // True when the process has finished all work or encountered a fatal error.
  // Once true, all subsequent drain() calls also return completed=true.
  bool completed;
  // True if this drain() call performed useful work. False means no work was
  // available (e.g., between region transitions, waiting for completer).
  // The executor uses this to deprioritize processes that repeatedly return
  // did_work=false (sleeping processes).
  bool did_work;

  // True if this drain() call did not perform useful work but the process
  // should remain on an active worker and be retried soon. Cooperative
  // processes use this to express active handoff policy. Unlike
  // did_work, this does not publish a shared cross-drainer progress signal.
  bool keep_active;

  // True if this drain() call did not perform useful work and the worker
  // should wait near the compute slot without remaining an active drainer.
  // Only wake_budget > 1 compute-slot processes can use this. While warm, the
  // worker is counted in the process warm-retainer count and does not call
  // drain() again until the process retention epoch changes.
  bool keep_warm;

  // True if keep_active must be visible to peer drainers. Use this when a
  // cooperative process may make mixed keep/release decisions across workers;
  // the last drainer consumes the shared publication before releasing the
  // process. This has no effect unless keep_active is also true.
  bool publish_keep_active;

  // Retention epoch observed by the drain function when setting keep_warm.
  // The warm worker waits until iree_task_process_retention_epoch(process)
  // differs from this value, then leaves warm state and rejoins scheduling.
  // Drain functions that set keep_warm must fill this from a process-owned or
  // process-associated epoch before returning no-work.
  int32_t keep_warm_epoch;
} iree_task_process_drain_result_t;

// Called by workers to do bounded work on a process. Multiple workers may call
// concurrently. Each call should do a bounded amount of work and return
// quickly — the scheduler relies on drain() returning to check for higher-
// priority work between calls.
//
// |worker_index| identifies the calling worker. It is stable for the worker
// lifetime and may include an executor-specific base offset. Drain functions
// that need dense per-executor storage should map this value into their own
// worker capacity; the executor does not manage per-process worker state.
typedef iree_status_t (*iree_task_process_drain_fn_t)(
    iree_task_process_t* process, uint32_t worker_index,
    iree_task_process_drain_result_t* out_result);

// Called exactly once when a process completes (after the last drain() returns
// completed=true). Handles cleanup, semaphore signaling, arena deallocation,
// etc. The callback runs on the worker that observed completion — it should
// be fast with no blocking or heavy allocation.
//
// For wake_budget > 1 processes, this fires eagerly as soon as the first worker
// observes completion. Other workers may still be inside drain(), so the
// callback must NOT free resources accessed during drain (processor context,
// worker state, etc.). Use release_fn for deferred resource cleanup.
//
// |status| is iree_ok_status() on success or the first error from drain().
// The callback takes ownership of the status.
typedef void (*iree_task_process_completion_fn_t)(iree_task_process_t* process,
                                                  iree_status_t status);

// Called when it is safe to free resources accessed by drain(). For
// wake_budget > 1 processes with cooperative multi-worker draining, this may
// fire after completion_fn when the last active drainer exits. For
// wake_budget == 1 processes with a single exclusive drainer, this fires
// immediately after completion_fn.
//
// Typical use: freeing the processor context, issue context wrapper, and
// other allocations that workers touch during drain().
//
// If NULL, no deferred release is needed (the completion callback handles
// everything, or the process's resources are externally managed).
typedef void (*iree_task_process_release_fn_t)(iree_task_process_t* process);

// Process state. Tracked transitions are:
//   SUSPENDED → RUNNABLE  (suspend_count driven to 0 via wake)
//   SUSPENDED → CANCELLED (cancel while still waiting for dependencies)
//   RUNNABLE  → COMPLETED (complete called after drain returns completed=true)
//   RUNNABLE  → CANCELLED (cancel while on run list or being drained)
// DRAINING is an implicit state: a RUNNABLE process that a worker currently
// holds on its stack. No enum value is needed because multiple workers may
// drain concurrently — the process stays RUNNABLE throughout.
typedef enum iree_task_process_state_e {
  // Waiting for suspend_count to reach zero before activation.
  IREE_TASK_PROCESS_STATE_SUSPENDED = 0,
  // On a run list, eligible for drain() by workers.
  IREE_TASK_PROCESS_STATE_RUNNABLE = 1,
  // Finished all work or encountered an error.
  IREE_TASK_PROCESS_STATE_COMPLETED = 2,
  // Cancelled by the user or due to scope failure.
  IREE_TASK_PROCESS_STATE_CANCELLED = 3,
} iree_task_process_state_t;

// Executor-managed scheduling state for run-list placement. The process type
// does not touch these — only the executor's schedule/drain logic does.
typedef enum iree_task_process_schedule_state_e {
  // Not on any run list. External events must push to activate.
  IREE_TASK_PROCESS_SCHEDULE_IDLE = 0,
  // On the immediate list, waiting for a worker to pop and drain.
  IREE_TASK_PROCESS_SCHEDULE_QUEUED = 1,
  // A worker has popped this process and is actively draining it.
  IREE_TASK_PROCESS_SCHEDULE_DRAINING = 2,
} iree_task_process_schedule_state_t;

// A process in the cooperative task scheduler. All fields after initialization
// are either immutable or accessed via atomics — no external locking required.
//
// Cache line layout:
//   Line 0: immutable after init (drain fn, completion fn, user_data).
//   Line 1: activation/completion (suspend_count, state, error_status).
//           Written by signaling threads (semaphore callbacks, completing
//           workers, cancellation). Read by workers at scan time.
//   Line 2: scheduling and warm retention. Written by drain functions and read
//           by the executor's wake/release scheduler.
//   Line 3: slist intrusion. The slist_next field is used by the immediate
//           list and dependent activation chains.
struct iree_task_process_t {
  //--- Cache line 0: immutable after initialization -------------------------

  // Called by workers to do bounded work. Multiple workers may call
  // concurrently. Must not be NULL.
  iree_task_process_drain_fn_t drain;

  // Called exactly once when the process completes (success or error).
  // Handles semaphore signaling, arena cleanup, etc. May be NULL if no
  // completion work is needed.
  iree_task_process_completion_fn_t completion_fn;

  // Called when it is safe to free drain-accessed resources. For
  // wake_budget > 1 processes, this fires when the last active drainer exits
  // after completion_fn. For wake_budget == 1 processes, this fires
  // immediately after completion_fn. May be NULL if no deferred release is
  // needed.
  iree_task_process_release_fn_t release_fn;

  // Opaque user data for the drain, completion, and release functions.
  // Typically points to the process-specific context (block processor
  // context, queue state, etc.). The process does not own this pointer.
  void* user_data;

  // Processes to activate when this process completes. Each dependent's
  // suspend_count is decremented; those that reach zero are activated.
  // The array is owned by the caller (typically arena-allocated alongside
  // the process). NULL if no dependents.
  iree_task_process_t** dependents;
  uint16_t dependent_count;

  uint8_t reserved0[22];

  //--- Cache line 1: activation and completion state ------------------------

  iree_alignas(iree_hardware_destructive_interference_size)

      // Number of outstanding dependencies. Process is runnable when this
      // reaches zero. Decremented by signaling threads (semaphore callbacks,
      // completing upstream processes, user activation). The thread that
      // drives this to zero is responsible for pushing the process to a run
      // list and waking workers.
      //
      // Initialized to the number of dependencies at creation time. A value
      // of 0 means the process is immediately runnable.
      iree_atomic_int32_t suspend_count;

  // Current process state. Transitions are monotonic (see state enum comment).
  iree_atomic_int32_t state;

  // First error encountered during drain(). Set via CAS; only the first error
  // wins. Stored as intptr_t because iree_status_t is pointer-tagged.
  iree_atomic_intptr_t error_status;

  //--- Cache line 2: scheduling state --------------------------------------

  iree_alignas(iree_hardware_destructive_interference_size)

      // Wake demand hint. A value of 1 uses the sequential immediate-list
      // path. Values greater than 1 use compute slots and seed the executor
      // wake tree with this many desired workers. This is not an admission
      // limit: already-active workers may still enter drain().
      //
      // Drain functions may update the hint at region transitions when the
      // amount of useful parallel work changes.
      iree_atomic_int32_t wake_budget;

  // Executor-managed scheduling state (iree_task_process_schedule_state_t).
  // Tracks whether this process is idle, queued on a run list, or being
  // drained by a worker. Used to prevent double-pushes to the immediate
  // list and to coordinate the sleeping/re-wake protocol.
  iree_atomic_int32_t schedule_state;

  // Set by external events to signal that new work is available for this
  // process. The draining worker checks this before transitioning to idle
  // to close the race between "drain returned no work" and "new work arrived
  // while we were draining." See the worker drain loop in worker.c.
  iree_atomic_int32_t needs_drain;

  // Set by drainers that need a keep_active decision to be visible to peer
  // drainers. The last compute-slot drainer consumes this before deciding to
  // release a non-terminal process.
  iree_atomic_int32_t retain_drain;

  // Monotonic process-local epoch used to release warm retainers. Scheduling
  // advances the epoch whenever external work is published; cooperative drain
  // functions may also advance it when process-local state changes in a way
  // that should make warm workers rejoin active draining.
  iree_atomic_int32_t retention_epoch;

  // Number of workers retained near this process without active drainer
  // claims. Slot release is not allowed while this is non-zero.
  iree_atomic_int32_t warm_retainers;

  // Number of warm retainers sleeping on retention_epoch.
  iree_atomic_int32_t retention_sleepers;

  // Shared warm-retainer spin deadline in host nanoseconds.
  iree_atomic_int64_t retention_spin_deadline_ns;

  //--- Cache line 3: list intrusion ----------------------------------------

  iree_alignas(iree_hardware_destructive_interference_size)

      // Intrusive slist pointer for the immediate list. Must be at a stable
      // offset for IREE_TYPED_ATOMIC_SLIST_WRAPPER.
      iree_atomic_slist_intrusive_ptr_t slist_next;
};

static_assert(offsetof(iree_task_process_t, suspend_count) %
                      iree_hardware_destructive_interference_size ==
                  0,
              "process activation state must start on its own cache line");
static_assert(offsetof(iree_task_process_t, wake_budget) %
                      iree_hardware_destructive_interference_size ==
                  0,
              "process scheduling state must start on its own cache line");
static_assert(offsetof(iree_task_process_t, retention_epoch) -
                      offsetof(iree_task_process_t, wake_budget) <
                  iree_hardware_destructive_interference_size,
              "warm-retention epoch must stay on the scheduling cache line");
static_assert(offsetof(iree_task_process_t, warm_retainers) -
                      offsetof(iree_task_process_t, wake_budget) <
                  iree_hardware_destructive_interference_size,
              "warm-retainer count must stay on the scheduling cache line");
static_assert(offsetof(iree_task_process_t, retention_sleepers) -
                      offsetof(iree_task_process_t, wake_budget) <
                  iree_hardware_destructive_interference_size,
              "retention sleeper count must stay on the scheduling cache line");
static_assert(offsetof(iree_task_process_t, retention_spin_deadline_ns) +
                      sizeof(iree_atomic_int64_t) -
                      offsetof(iree_task_process_t, wake_budget) <=
                  iree_hardware_destructive_interference_size,
              "process scheduling state must fit on one cache line");
static_assert(offsetof(iree_task_process_t, slist_next) %
                      iree_hardware_destructive_interference_size ==
                  0,
              "process list state must start on its own cache line");

// Typed slist wrapper for the immediate list.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_task_process, iree_task_process_t,
                                offsetof(iree_task_process_t, slist_next));

//===----------------------------------------------------------------------===//
// Process lifecycle
//===----------------------------------------------------------------------===//

// Initializes a process with the given drain function and configuration.
// The process starts in SUSPENDED state with the given suspend_count.
// If suspend_count is 0, the process is immediately runnable (but not yet
// on any run list — the caller must activate it).
//
// All pointer fields (completion_fn, user_data, dependents) may be set
// after initialization but before the process is activated.
void iree_task_process_initialize(iree_task_process_drain_fn_t drain_fn,
                                  int32_t suspend_count, int32_t wake_budget,
                                  iree_task_process_t* out_process);

// Decrements the process's suspend count by one. If the count reaches zero,
// returns true — the caller is responsible for pushing the process to the
// appropriate run list and waking workers. If the count is still positive,
// returns false.
//
// Thread-safe: may be called from any thread (worker, proactor, user thread).
// Uses acq_rel ordering: the release ensures all writes by the signaling
// thread are visible to the draining worker; the acquire ensures the
// activating thread sees all prior signalers' writes.
//
// If the process is cancelled, returns false (the process should not be
// activated). The cancellation path handles cleanup.
bool iree_task_process_wake(iree_task_process_t* process);

// Marks the process as completed and resolves dependents. Called by the worker
// that observed drain() returning completed=true. Must be called exactly once.
//
// Actions:
// - Sets state to COMPLETED.
// - Calls completion_fn (if set) with the process's error status.
// - For each dependent: decrements suspend_count. If any dependent reaches
//   zero, it is added to |out_activated| (caller must push to run list).
//
// |out_activated_head| and |out_activated_tail| receive a singly-linked list
// (via slist_next) of dependents that became runnable. The caller pushes
// these to the appropriate run lists.
void iree_task_process_complete(iree_task_process_t* process,
                                iree_task_process_t** out_activated_head,
                                iree_task_process_t** out_activated_tail);

// Records the first error encountered during drain(). Thread-safe via CAS.
// If another thread already recorded an error, this error is freed. If the
// process is RUNNABLE, transitions to
// COMPLETED so that future drain() calls bail immediately via is_terminal().
// Must only be called on a RUNNABLE process (i.e., during drain). To fail
// a SUSPENDED process, use iree_task_process_cancel.
void iree_task_process_report_error(iree_task_process_t* process,
                                    iree_status_t status);

// Returns true if the process has encountered an error. Cheap check for
// drain() functions to bail early.
static inline bool iree_task_process_has_error(
    const iree_task_process_t* process) {
  return iree_atomic_load(&process->error_status, iree_memory_order_relaxed) !=
         0;
}

// Cancels the process. Records a cancellation error and transitions to
// CANCELLED from any non-terminal state. If already completed or cancelled,
// this is a no-op.
//
// If the process was SUSPENDED, no worker will ever drain it, so cancel
// resolves it inline: calls completion_fn and resolves dependents. Newly
// activated dependents are returned in
// |out_activated_head|/|out_activated_tail|.
//
// If the process was RUNNABLE, workers will see the terminal state on their
// next drain() call and complete it through the normal path. The out lists
// will be empty.
//
// After cancellation of a RUNNABLE process:
// - drain() sees is_terminal()/has_error() and returns completed=true.
// - wake() returns false (the process should not be re-activated).
// - complete() resolves dependents normally (they see the error).
void iree_task_process_cancel(iree_task_process_t* process,
                              iree_task_process_t** out_activated_head,
                              iree_task_process_t** out_activated_tail);

// Returns the current state of the process. The state may change immediately
// after this call (another thread may complete or cancel the process).
static inline iree_task_process_state_t iree_task_process_state(
    const iree_task_process_t* process) {
  return (iree_task_process_state_t)iree_atomic_load(&process->state,
                                                     iree_memory_order_acquire);
}

// Returns true if the process is in a terminal state (completed or cancelled).
static inline bool iree_task_process_is_terminal(
    const iree_task_process_t* process) {
  iree_task_process_state_t state = iree_task_process_state(process);
  return state == IREE_TASK_PROCESS_STATE_COMPLETED ||
         state == IREE_TASK_PROCESS_STATE_CANCELLED;
}

// Returns the current wake demand hint. May change at any time when drain()
// reaches a region transition.
static inline int32_t iree_task_process_wake_budget(
    const iree_task_process_t* process) {
  return iree_atomic_load(&process->wake_budget, iree_memory_order_relaxed);
}

// Returns the process-local warm-retention epoch.
static inline int32_t iree_task_process_retention_epoch(
    const iree_task_process_t* process) {
  return iree_atomic_load(&process->retention_epoch, iree_memory_order_acquire);
}

// Returns the number of workers currently retained warm near the process.
static inline int32_t iree_task_process_warm_retainer_count(
    const iree_task_process_t* process) {
  return iree_atomic_load(&process->warm_retainers, iree_memory_order_acquire);
}

// Advances the process-local warm-retention epoch and returns the new value.
static inline int32_t iree_task_process_advance_retention_epoch(
    iree_task_process_t* process) {
  const int32_t new_epoch = iree_atomic_fetch_add(&process->retention_epoch, 1,
                                                  iree_memory_order_acq_rel) +
                            1;
#if defined(IREE_RUNTIME_USE_FUTEX)
  if (iree_atomic_load(&process->retention_sleepers,
                       iree_memory_order_acquire) > 0) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z_wake,
                                "iree_task_process_wake_retention_sleepers");
    iree_futex_wake((void*)&process->retention_epoch, IREE_ALL_WAITERS);
    IREE_TRACE_ZONE_END(z_wake);
  }
#endif  // IREE_RUNTIME_USE_FUTEX
  return new_epoch;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_PROCESS_H_
