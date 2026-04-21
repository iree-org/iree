// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/worker.h"

#include <stdbool.h>
#include <string.h>

#include "iree/base/internal/fpu_state.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/processor.h"
#include "iree/task/executor_impl.h"
#include "iree/task/process.h"
#include "iree/task/tuning.h"

#define IREE_TASK_WORKER_MIN_STACK_SIZE (32 * 1024)

static int iree_task_worker_main(iree_task_worker_t* worker);

static int iree_task_worker_thread_entry(void* entry_arg) {
  return iree_task_worker_main((iree_task_worker_t*)entry_arg);
}

iree_status_t iree_task_worker_initialize(
    iree_task_executor_t* executor, iree_host_size_t worker_index,
    const iree_task_topology_group_t* topology_group,
    iree_host_size_t stack_size, iree_byte_span_t local_memory,
    iree_prng_splitmix64_state_t* seed_prng, iree_task_worker_t* out_worker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  out_worker->executor = executor;
  out_worker->worker_index = executor->worker_base_index + worker_index;
  out_worker->worker_bit = iree_task_affinity_bit_for_worker(worker_index);
  out_worker->ideal_thread_affinity = topology_group->ideal_thread_affinity;
  out_worker->constructive_sharing_mask =
      topology_group->constructive_sharing_mask;
  out_worker->local_memory = local_memory;
  out_worker->processor_id = 0;
  out_worker->processor_tag = 0;

  iree_notification_initialize(&out_worker->wake_notification);
  iree_notification_initialize(&out_worker->state_notification);

  iree_task_worker_state_t initial_state = IREE_TASK_WORKER_STATE_RUNNING;
  iree_atomic_store(&out_worker->state, initial_state,
                    iree_memory_order_release);

  iree_thread_create_params_t thread_params;
  memset(&thread_params, 0, sizeof(thread_params));
  thread_params.name = iree_make_cstring_view(topology_group->name);
  thread_params.create_suspended = false;
  thread_params.priority_class = IREE_THREAD_PRIORITY_CLASS_NORMAL;
  thread_params.initial_affinity = out_worker->ideal_thread_affinity;
  thread_params.stack_size =
      iree_max(IREE_TASK_WORKER_MIN_STACK_SIZE, stack_size);

  // NOTE: if the thread creation fails we'll bail here and let the caller
  // cleanup by calling deinitialize. The guard in deinitialize checks
  // worker->executor (set above) to distinguish initialized workers from
  // never-initialized ones in the same allocation.
  iree_status_t status = iree_thread_create(
      iree_task_worker_thread_entry, out_worker, thread_params,
      executor->allocator, &out_worker->thread);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_task_worker_request_exit(iree_task_worker_t* worker) {
  if (!worker->thread) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // If the thread is already in the exiting/zombie state we don't need to do
  // anything.
  iree_task_worker_state_t prev_state =
      (iree_task_worker_state_t)iree_atomic_exchange(
          &worker->state, IREE_TASK_WORKER_STATE_EXITING,
          iree_memory_order_acq_rel);
  switch (prev_state) {
    case IREE_TASK_WORKER_STATE_ZOMBIE:
      // Worker already exited; reset state to ZOMBIE.
      iree_atomic_store(&worker->state, IREE_TASK_WORKER_STATE_ZOMBIE,
                        iree_memory_order_release);
      break;
    default:
      // Worker now set to EXITING and should exit soon.
      break;
  }

  // Kick the worker in case it is waiting for work.
  iree_notification_post(&worker->wake_notification, 1);

  IREE_TRACE_ZONE_END(z0);
}

// Returns true if the worker is in the zombie state (exited and awaiting
// teardown).
static bool iree_task_worker_is_zombie(iree_task_worker_t* worker) {
  return iree_atomic_load(&worker->state, iree_memory_order_acquire) ==
         IREE_TASK_WORKER_STATE_ZOMBIE;
}

static bool iree_task_worker_is_zombie_thunk(void* arg) {
  return iree_task_worker_is_zombie((iree_task_worker_t*)arg);
}

void iree_task_worker_await_exit(iree_task_worker_t* worker) {
  if (!worker->thread) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_task_worker_request_exit(worker);
  iree_notification_await(&worker->state_notification,
                          iree_task_worker_is_zombie_thunk, worker,
                          iree_infinite_timeout());

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_worker_deinitialize(iree_task_worker_t* worker) {
  // Skip workers that were never initialized. Their memory is zero-filled
  // from iree_allocator_malloc but notifications were never initialized —
  // calling iree_notification_deinitialize on them would operate on a
  // never-initialized pthread_mutex_t (undefined behavior on non-glibc).
  // worker->executor is set at the start of iree_task_worker_initialize,
  // before notification init, so NULL means initialize was never called.
  if (!worker->executor) return;

  IREE_TRACE_ZONE_BEGIN(z0);

  // Must have called request_exit/await_exit, OR thread creation failed during
  // initialization (thread is NULL but notifications are initialized).
  IREE_ASSERT_TRUE(!worker->thread || iree_task_worker_is_zombie(worker));

  iree_thread_release(worker->thread);
  worker->thread = NULL;

  iree_notification_deinitialize(&worker->wake_notification);
  iree_notification_deinitialize(&worker->state_notification);

  IREE_TRACE_ZONE_END(z0);
}

// Marks the worker as "active" (scheduling work or executing it).
// The idle mask is accessed with 'relaxed' order because it's just a hint.
static void iree_task_worker_mark_active(iree_task_worker_t* worker) {
  iree_atomic_task_affinity_set_clear(&worker->executor->worker_idle_mask,
                                      worker->worker_bit,
                                      iree_memory_order_relaxed);
  int32_t old_idle_count = iree_atomic_fetch_sub(
      &worker->executor->worker_idle_count, 1, iree_memory_order_relaxed);
  (void)old_idle_count;
  IREE_TRACE_PLOT_VALUE_F32(worker->executor->trace_name,
                            100.0f - 100.0f * (float)(old_idle_count - 1) /
                                         (float)worker->executor->worker_count);
}

// Marks the worker as "idle" (sleeping/spinning waiting to wake).
// The idle mask is accessed with 'relaxed' order because it's just a hint.
static void iree_task_worker_mark_idle(iree_task_worker_t* worker) {
  iree_atomic_task_affinity_set_set(&worker->executor->worker_idle_mask,
                                    worker->worker_bit,
                                    iree_memory_order_relaxed);
  int32_t old_idle_count = iree_atomic_fetch_add(
      &worker->executor->worker_idle_count, 1, iree_memory_order_relaxed);
  (void)old_idle_count;
  IREE_TRACE_PLOT_VALUE_F32(worker->executor->trace_name,
                            100.0f - 100.0f * (float)(old_idle_count + 1) /
                                         (float)worker->executor->worker_count);
}

// Pops one process from the executor's immediate list and drains it.
// Drains in a loop until the process completes or returns did_work=false
// (sleeping), then transitions the process to IDLE with a final needs_drain
// race-check to ensure no work signaled between the last drain and the
// DRAINING->IDLE transition is lost.
//
// Returns true if a process was found (whether completed or put to sleep).
// Returns false if the immediate list was empty (no processes available).
static bool iree_task_worker_drain_process(iree_task_worker_t* worker) {
  iree_task_executor_t* executor = worker->executor;
  iree_task_process_t* process =
      iree_task_process_slist_pop(&executor->immediate_list);
  if (!process) return false;

  IREE_TRACE_ZONE_BEGIN(z0);

  // Transition QUEUED → DRAINING. We own this process exclusively now.
  iree_atomic_store(&process->schedule_state,
                    (int32_t)IREE_TASK_PROCESS_SCHEDULE_DRAINING,
                    iree_memory_order_release);

  while (true) {
    // Drain bounded work from the process.
    iree_task_process_drain_result_t result;
    memset(&result, 0, sizeof(result));
    iree_status_t status =
        process->drain(process, (uint32_t)worker->worker_index, &result);
    if (!iree_status_is_ok(status)) {
      iree_task_process_report_error(process, status);
    }

    // Completed or terminal (cancelled/errored): resolve and schedule
    // activated dependents.
    if (result.completed || iree_task_process_is_terminal(process)) {
      iree_atomic_store(&process->schedule_state,
                        (int32_t)IREE_TASK_PROCESS_SCHEDULE_IDLE,
                        iree_memory_order_release);

      // Snapshot release_fn before complete — completion_fn may free the
      // process if release_fn is NULL (single-phase lifecycle).
      iree_task_process_release_fn_t release_fn = process->release_fn;

      iree_task_process_t* activated_head = NULL;
      iree_task_process_t* activated_tail = NULL;
      iree_task_process_complete(process, &activated_head, &activated_tail);

      // For wake_budget == 1 processes there is exactly one drainer (us), so
      // release is safe immediately after completion.
      if (release_fn) {
        release_fn(process);
      }

      // Schedule each activated dependent.
      iree_task_process_t* activated = activated_head;
      while (activated) {
        iree_task_process_t* next = iree_task_process_slist_get_next(activated);
        iree_task_executor_schedule_process(executor, activated);
        activated = next;
      }
      IREE_TRACE_ZONE_END(z0);
      return true;
    }

    // Process has more work — loop immediately.
    if (result.did_work) continue;

    // No work available (sleeping). Check if an external event signaled
    // between our last drain call and now.
    if (iree_atomic_exchange(&process->needs_drain, 0,
                             iree_memory_order_acq_rel)) {
      continue;  // New work signaled — drain again.
    }

    // Transition DRAINING → IDLE.
    //
    // Matches schedule_process's seq-cst store/CAS pair in the Dekker-style
    // sleep/wake handoff: the IDLE store and the final needs_drain load both
    // participate in the same seq-cst order as the scheduler's operations.
    // Without that, the worker can miss needs_drain=1 while the scheduler's
    // CAS still sees DRAINING and skips enqueueing.
    iree_atomic_store(&process->schedule_state,
                      (int32_t)IREE_TASK_PROCESS_SCHEDULE_IDLE,
                      iree_memory_order_seq_cst);

    // Final race check: an external event may have set needs_drain after our
    // exchange (saw 0) but before we stored IDLE. That event's
    // schedule_process saw DRAINING and returned without pushing, trusting us
    // to re-check. If needs_drain is set, reclaim the process.
    if (iree_atomic_load(&process->needs_drain, iree_memory_order_seq_cst)) {
      int32_t expected = IREE_TASK_PROCESS_SCHEDULE_IDLE;
      if (iree_atomic_compare_exchange_strong(
              &process->schedule_state, &expected,
              (int32_t)IREE_TASK_PROCESS_SCHEDULE_DRAINING,
              iree_memory_order_acq_rel, iree_memory_order_acquire)) {
        continue;  // Reclaimed — drain again.
      }
      // CAS failed: another thread already transitioned IDLE→QUEUED and
      // pushed to the list. The process will be picked up by a worker.
    }

    // Process is truly sleeping. We're done with it.
    IREE_TRACE_ZONE_END(z0);
    return true;
  }
}

// Eagerly completes a compute process: calls the completion callback
// (semaphore signaling, dependent activation) and schedules any activated
// dependents. Called by the first worker to claim completion via CAS on
// completion_claimed.
//
// Does NOT transition schedule_state to IDLE — that happens in
// release_compute_process after the slot is fully cleaned up. Setting IDLE
// here would allow schedule_process to reschedule the process into a new
// slot before the release callback fires, causing overlapping releases.
//
// Does NOT free drain-accessed resources — that is handled by
// iree_task_worker_release_compute_process when the last drainer exits.
static void iree_task_worker_eager_complete_compute_process(
    iree_task_worker_t* worker, iree_task_process_t* process) {
  iree_task_executor_t* executor = worker->executor;

  iree_task_process_t* activated_head = NULL;
  iree_task_process_t* activated_tail = NULL;
  iree_task_process_complete(process, &activated_head, &activated_tail);

  iree_task_process_t* activated = activated_head;
  while (activated) {
    iree_task_process_t* next = iree_task_process_slist_get_next(activated);
    iree_task_executor_schedule_process(executor, activated);
    activated = next;
  }
}

// Releases a compute slot's current process ownership. Clears the slot, resets
// the slot generation for reuse, optionally reclaims a non-terminal process if
// new work arrived during the sleep transition, and frees drain-accessed
// resources only for terminal processes. Called by the worker that
// successfully CAS'd active_drainers from gen|0 to gen|SENTINEL.
//
// |tagged_sentinel| is the full 64-bit value (gen|SENTINEL) currently in
// active_drainers. The generation bits are preserved and incremented when
// the slot is reset, preventing ABA races on subsequent slot lifetimes.
//
// Ordering: clear process pointer, reset active_drainers, promote from
// overflow, re-wake, then either put the process to sleep or run its terminal
// release callback. Each phase has ordering constraints documented inline.
static void iree_task_worker_release_compute_process(
    iree_task_worker_t* worker, iree_task_compute_slot_t* slot,
    iree_task_process_t* process, bool process_is_terminal,
    int64_t tagged_sentinel) {
  iree_task_executor_t* executor = worker->executor;
  process_is_terminal =
      process_is_terminal || iree_task_process_is_terminal(process);

  // Clear the process pointer. After this store (release), no new worker will
  // pass the quick check for this slot — except via stale relaxed reads (the
  // quick check uses relaxed ordering).
  iree_atomic_store(&slot->process, 0, iree_memory_order_release);

  // Reset completion_claimed and atomically CAS active_drainers from
  // gen|SENTINEL to (gen+1)|0.
  //
  // Workers that passed the quick check with a stale non-zero read before
  // the process pointer was cleared may still be in the bail path: they did
  // fetch_add(1) on active_drainers (seeing SENTINEL + N in the count bits),
  // found (int32_t)prev < 0, and are about to do fetch_sub(1) to undo.
  //
  // The CAS only succeeds when active_drainers is exactly our sentinel
  // value (no in-flight bailers have perturbed the count bits). The
  // generation increment on reset means any stale CAS from a worker that
  // observed a prior generation's count=0 will fail — the generation bits
  // won't match.
  //
  // completion_claimed is reset before the CAS so that a new process placed
  // during the release window does not inherit a stale claimed flag.
  iree_atomic_store(&slot->completion_claimed, 0, iree_memory_order_relaxed);
  int64_t next_generation =
      (tagged_sentinel & ~(int64_t)UINT32_MAX) + IREE_TASK_SLOT_GEN_INCREMENT;
  int64_t expected_sentinel = tagged_sentinel;
  while (!iree_atomic_compare_exchange_weak(
      &slot->active_drainers, &expected_sentinel, next_generation,
      iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_processor_yield();
    expected_sentinel = tagged_sentinel;
  }

  // Promote from the compute overflow list into this slot. If wake_budget > 1
  // processes were scheduled while all slots were occupied, they are waiting
  // in compute_overflow. Now that this slot is clean (process=0,
  // active_drainers=(gen+1)|0), try to promote one. The CAS handles the race
  // with a concurrent schedule_process that may have already filled the slot
  // during the release window.
  iree_task_process_t* overflow_process =
      iree_task_process_slist_pop(&executor->compute_overflow);
  if (overflow_process) {
    intptr_t expected = 0;
    if (iree_atomic_compare_exchange_strong(
            &slot->process, &expected, (intptr_t)overflow_process,
            iree_memory_order_release, iree_memory_order_relaxed)) {
      // Placed successfully. The re-wake below will handle waking workers.
    } else {
      // Slot was filled by a concurrent schedule_process. Try to place the
      // overflow process in any other empty slot; if none available, push it
      // back to the overflow list for the next release to pick up.
      if (!iree_task_executor_try_place_in_compute_slot(executor,
                                                        overflow_process)) {
        iree_task_process_slist_push(&executor->compute_overflow,
                                     overflow_process);
      }
    }
  }

  // Re-wake if a process was placed during the release window. Between the
  // process pointer clear and the active_drainers reset, a concurrent
  // schedule_process can place a new process in this slot, or the overflow
  // promotion above may have placed one. Workers woken by the original
  // schedule_process found active_drainers sentinel (count < 0), bailed,
  // and went back to sleep. Now that the slot is clean and drainable, we
  // must re-wake workers for whatever process is in the slot.
  intptr_t new_process =
      iree_atomic_load(&slot->process, iree_memory_order_acquire);
  if (new_process) {
    iree_task_process_t* p = (iree_task_process_t*)new_process;
    int32_t budget = iree_task_process_wake_budget(p);
    iree_task_executor_wake_workers(executor, budget);
  }

  // Transition to IDLE only if the process is not terminal (may be
  // rescheduled for more work). Terminal processes (completed/cancelled)
  // must NOT transition to IDLE — that would allow schedule_process to
  // CAS(IDLE→DRAINING) and re-place the process, causing double
  // completion and double release.
  if (!process_is_terminal) {
    // Matches schedule_process's seq-cst store/CAS pair in the same
    // Dekker-style handoff used by the wake_budget == 1 path.
    iree_atomic_store(&process->schedule_state,
                      (int32_t)IREE_TASK_PROCESS_SCHEDULE_IDLE,
                      iree_memory_order_seq_cst);

    // Final race check. If new work arrived after we cleared the slot but
    // before the IDLE store became visible, reclaim the process by CAS-ing it
    // back to DRAINING and placing it in a compute slot again.
    if (iree_atomic_load(&process->needs_drain, iree_memory_order_seq_cst)) {
      int32_t expected = IREE_TASK_PROCESS_SCHEDULE_IDLE;
      if (iree_atomic_compare_exchange_strong(
              &process->schedule_state, &expected,
              (int32_t)IREE_TASK_PROCESS_SCHEDULE_DRAINING,
              iree_memory_order_acq_rel, iree_memory_order_acquire)) {
        if (!iree_task_executor_try_place_in_compute_slot(executor, process)) {
          iree_task_process_slist_push(&executor->compute_overflow, process);
        }
        iree_task_executor_wake_workers(executor,
                                        iree_task_process_wake_budget(process));
      }
    }
  }

  // Free drain-accessed resources. The slot is fully clean and may
  // be reused by a new process placed by a concurrent schedule_process call.
  // release_fn operates on the old process via its pointer argument, which
  // is independent of whatever new process may now occupy the slot.
  //
  // Non-terminal processes are merely going idle here and must keep their
  // process-owned state live for the next activation.
  if (process_is_terminal) {
    iree_task_process_release_fn_t release_fn = process->release_fn;
    if (release_fn) {
      release_fn(process);
    }
  }
}

// Scans executor compute slots for wake_budget > 1 processes and drains bounded
// work from one of them. Workers scan round-robin starting from
// compute_slot_scan_start to distribute load evenly across slots.
//
// Two-phase active-drainer protocol:
//   Before accessing a slot's process, the worker increments active_drainers.
//   This prevents the release callback (which frees the processor context)
//   from firing while any worker is still inside drain(). The completion
//   callback (semaphore signaling, dependent activation) fires eagerly —
//   only the resource release is deferred until the last drainer exits.
//
// Returns true if any useful work was performed. The caller should loop back
// to check the immediate list before scanning again, ensuring responsive
// handling of wake_budget == 1 processes between compute work.
static bool iree_task_worker_drain_compute_slots(iree_task_worker_t* worker) {
  iree_task_executor_t* executor = worker->executor;
  bool did_work = false;

  for (iree_host_size_t i = 0; i < IREE_TASK_EXECUTOR_MAX_COMPUTE_SLOTS; ++i) {
    iree_host_size_t slot_index = (worker->compute_slot_scan_start + i) %
                                  IREE_TASK_EXECUTOR_MAX_COMPUTE_SLOTS;
    iree_task_compute_slot_t* slot = &executor->compute_slots[slot_index];

    // Quick check: skip empty slots without touching active_drainers.
    // Relaxed is sufficient — this is just a hint to avoid the expensive
    // fetch_add on the common case of empty slots.
    intptr_t quick =
        iree_atomic_load(&slot->process, iree_memory_order_relaxed);
    if (!quick) continue;

    // Register as an active drainer BEFORE accessing the process. This
    // prevents the release callback from firing while we're in drain().
    //
    // The fetch_add(1) operates on the full 64-bit tagged counter but only
    // affects the low 32-bit count (count is always small, never overflows
    // into the generation bits). If the count portion is negative (sentinel
    // bit set), the slot is being released — bail immediately.
    int64_t prev_drainers = iree_atomic_fetch_add(&slot->active_drainers, 1,
                                                  iree_memory_order_acq_rel);
    if (IREE_UNLIKELY((int32_t)prev_drainers < 0)) {
      iree_atomic_fetch_sub(&slot->active_drainers, 1,
                            iree_memory_order_release);
      continue;
    }

    // Re-verify with acquire ordering. Between our relaxed load and
    // our drainer registration, the slot may have been cleared by the
    // last drainer of a prior completion. If so, unregister and skip.
    iree_task_process_t* process = (iree_task_process_t*)iree_atomic_load(
        &slot->process, iree_memory_order_acquire);
    if (!process) {
      iree_atomic_fetch_sub(&slot->active_drainers, 1,
                            iree_memory_order_release);
      continue;
    }

    // From here, we are a registered drainer. The process pointer is valid
    // and will remain valid until we decrement active_drainers — the
    // release path waits for active_drainers to reach zero.
    bool is_terminal = iree_task_process_is_terminal(process);
    bool drained_process_work = false;

    if (!is_terminal) {
      // Drain bounded work from this process.
      iree_task_process_drain_result_t result;
      memset(&result, 0, sizeof(result));
      iree_status_t status =
          process->drain(process, (uint32_t)worker->worker_index, &result);
      if (!iree_status_is_ok(status)) {
        iree_task_process_report_error(process, status);
      }

      is_terminal = result.completed || iree_task_process_is_terminal(process);
      drained_process_work = result.did_work;
      if (drained_process_work) {
        // Sticky self-rerun signal for cross-drainer coordination. Any
        // drainer that observes forward progress publishes needs_drain=1 so
        // whichever peer ends up being the last drainer knows another scan
        // pass is warranted. Without this publication, a co-drainer who
        // raced our useful work can see did_work=false on its own drain,
        // reach the last-drainer branch, and release the slot even though
        // there may still be process-local work visible only to us. Terminal
        // processes do not need re-drain — they are about to be released.
        if (!is_terminal) {
          iree_atomic_store(&process->needs_drain, 1,
                            iree_memory_order_release);
        }
        did_work = true;
      }
    }

    // If we observed completion, try to claim the eager completion callback.
    // Exactly one worker wins this CAS and runs signaling/dependent
    // activation immediately — zero delay on downstream work.
    if (is_terminal) {
      int32_t expected_claim = 0;
      if (iree_atomic_compare_exchange_strong(
              &slot->completion_claimed, &expected_claim, 1,
              iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
        iree_task_worker_eager_complete_compute_process(worker, process);
      }
    }

    // Unregister as active drainer. The fetch_sub(1) returns the previous
    // 64-bit tagged value; extract the count from the low 32 bits to check
    // if we were the last drainer.
    //
    // If we're the last drainer out and either the process is terminal or no
    // one requested another drain pass, try to claim slot release by CAS-ing
    // active_drainers from gen|0 to gen|SENTINEL. The generation bits (from our
    // fetch_sub result) must match — if another worker completed an entire
    // release cycle between our fetch_sub and our CAS, the generation will have
    // incremented and our CAS fails harmlessly. This eliminates the ABA race
    // that existed with the 32-bit counter.
    int64_t old_drainers = iree_atomic_fetch_sub(&slot->active_drainers, 1,
                                                 iree_memory_order_acq_rel);
    int32_t remaining = (int32_t)old_drainers - 1;
    if (remaining == 0) {
      // Only the last drainer is allowed to clear needs_drain. A non-last
      // drainer that returned did_work=false may have observed a stale empty
      // process-local state while another drainer or a schedule_process call
      // was concurrently publishing more work; clearing needs_drain in that
      // non-last path could strand the work against the true last drainer's
      // release decision.
      //
      // The last drainer combines two signals into its sleep decision:
      //   1. drained_process_work — our own local did-useful-work result,
      //      which no global flag carries across drainers but is the most
      //      precise signal we have about "did this worker just see work".
      //   2. global needs_drain — the shared publication from peer drainers
      //      (via the sticky store above) and from external
      //      schedule_process callers. We consume this with an exchange so
      //      future activations start from a clean slate.
      //
      // A did_work=true last drainer skips the exchange and leaves
      // needs_drain set for the next pass to consume; worst case this is
      // one extra no-work drain before the process actually releases. A
      // did_work=false last drainer must consume the global flag to avoid
      // missing a cross-drainer wake signal.
      bool needs_drain = drained_process_work;
      if (!is_terminal) {
        if (!needs_drain) {
          needs_drain = iree_atomic_exchange(&process->needs_drain, 0,
                                             iree_memory_order_acq_rel) != 0;
        }
        if (needs_drain) {
          did_work = true;
        }
      }

      if (is_terminal || !needs_drain) {
        // Release the slot: CAS active_drainers from gen|0 to gen|SENTINEL.
        // Generation bits must match our fetch_sub snapshot — if another
        // worker completed an entire release cycle between our fetch_sub
        // and this CAS, the generation will have advanced and our CAS
        // fails harmlessly (the other worker already released this slot).
        // The generation-tagged 64-bit counter eliminates the ABA that a
        // plain 32-bit counter would hit here.
        int64_t generation = old_drainers & ~(int64_t)UINT32_MAX;
        int64_t expected_empty = generation;
        int64_t sentinel = generation | IREE_TASK_SLOT_SENTINEL;
        if (iree_atomic_compare_exchange_strong(
                &slot->active_drainers, &expected_empty, sentinel,
                iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
          iree_task_worker_release_compute_process(worker, slot, process,
                                                   is_terminal, sentinel);
        }
        // CAS failed: a new worker incremented active_drainers between our
        // decrement and this CAS, or another worker already released this
        // slot (generation advanced). Either way, release is handled.
      }
    }

    if (did_work) break;  // Return to main loop to interleave with immediate.
  }

  // Advance scan start for round-robin fairness.
  worker->compute_slot_scan_start = (worker->compute_slot_scan_start + 1) %
                                    IREE_TASK_EXECUTOR_MAX_COMPUTE_SLOTS;

  return did_work;
}

// Claims a share of the executor's desired_wake counter and wakes that many
// additional idle workers, propagating the wake tree. Called early in each
// pump iteration (after mark_active, before drain) so that the tree expands
// with minimal latency — each worker relays before doing any useful work.
//
// The CAS loop avoids over-claiming: if another worker claims between our
// load and CAS, we retry with the updated value. If desired_wake reaches
// zero, we return immediately (no workers to wake).
//
// If we claim N but can only find M < N idle workers, the excess is dropped
// (not returned to desired_wake). This is correct because the missing workers
// are already active — if they were idle, they'd appear in idle_mask. Returning
// the excess would create a livelock: active workers would claim, fail to find
// idle targets, return, and repeat forever, preventing desired_wake from
// reaching zero.
static void iree_task_worker_relay_wake(iree_task_worker_t* worker) {
  iree_task_executor_t* executor = worker->executor;

  // Claim up to IREE_TASK_WAKE_FANOUT from desired_wake via CAS.
  int32_t claimed = 0;
  int32_t desired =
      iree_atomic_load(&executor->desired_wake, iree_memory_order_acquire);
  while (desired > 0) {
    int32_t claim = desired;
    if (claim > IREE_TASK_WAKE_FANOUT) claim = IREE_TASK_WAKE_FANOUT;
    if (iree_atomic_compare_exchange_weak(
            &executor->desired_wake, &desired, desired - claim,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      claimed = claim;
      break;
    }
    // CAS failed: desired was updated by another thread. Loop retries
    // with the new value loaded into desired by CAS.
  }
  if (claimed <= 0) return;

  // Wake |claimed| idle workers. Skip ourselves (already active).
  iree_task_affinity_set_t idle_mask = iree_atomic_task_affinity_set_load(
      &executor->worker_idle_mask, iree_memory_order_relaxed);
  iree_task_affinity_set_clear(&idle_mask, worker->worker_bit);

  while (claimed > 0) {
    int target = iree_task_affinity_set_find_first(idle_mask);
    if (target < 0 || target >= (int)executor->worker_count) break;
    iree_notification_post(&executor->workers[target].wake_notification, 1);
    iree_task_affinity_set_clear_index(&idle_mask, (iree_host_size_t)target);
    --claimed;
  }
  // Any remaining |claimed| is dropped — the corresponding workers are
  // already active and will drain without needing a wake.
}

// Updates the cached processor ID field in the worker.
static void iree_task_worker_update_processor_id(iree_task_worker_t* worker) {
  iree_cpu_requery_processor_id(&worker->processor_tag, &worker->processor_id);
}

// Alternates between draining processes and waiting for more work to arrive.
// Only returns when the worker has been asked by the executor to exit.
static void iree_task_worker_pump_until_exit(iree_task_worker_t* worker) {
  // Initial processor ID assignment. We normally refresh this upon waking from
  // a wait but it's possible that there's already work pending and we want to
  // be able to process it with the proper processor ID immediately.
  iree_task_worker_update_processor_id(worker);

  // Track whether this worker is currently marked active in the executor's
  // worker_idle_mask. Workers only touch the shared mask on actual state
  // transitions (idle→active when waking, active→idle when sleeping). While
  // spinning with work, the mask is untouched — avoiding 2N atomic RMWs per
  // pump cycle on the shared cache line when N workers are all active.
  bool is_active = false;

  while (true) {
    // In order to not miss any work that is enqueued after we've already
    // checked a particular source we use an interruptible wait token that
    // will prevent the wait from happening if anyone touches the data
    // structures we use.
    iree_wait_token_t wait_token =
        iree_notification_prepare_wait(&worker->wake_notification);

    // Mark active on the first iteration or after waking from sleep.
    // While spinning with work, we stay active and skip the mask update.
    if (!is_active) {
      iree_task_worker_mark_active(worker);
      is_active = true;
    }

    // Propagate the wake tree: claim a share of desired_wake and wake
    // additional idle workers before we start draining. This runs every
    // iteration (not just on wake) since new work may arrive while draining.
    iree_task_worker_relay_wake(worker);

    // Check state to see if we've been asked to exit.
    if (iree_atomic_load(&worker->state, iree_memory_order_acquire) ==
        IREE_TASK_WORKER_STATE_EXITING) {
      iree_notification_cancel_wait(&worker->wake_notification);
      break;
    }

    // Scan compute slots first (wake_budget > 1 cooperative drain). If compute
    // work was found, skip the immediate list — workers doing tile execution
    // shouldn't contend on the immediate list's mutex. Only check the
    // immediate list when no compute work is available, so exactly one idle
    // worker picks up the wake_budget == 1 control process.
    bool did_work = false;
    if (iree_task_worker_drain_compute_slots(worker)) {
      did_work = true;
    } else {
      while (iree_task_worker_drain_process(worker)) {
        did_work = true;
      }
    }

    if (did_work) {
      // Had work to do; loop around to check for more before sleeping.
      // Stay active — schedule_process doesn't need to wake us since we'll
      // pick up new work on our next drain pass.
      iree_notification_cancel_wait(&worker->wake_notification);
    } else {
      // No work found. Mark idle so schedule_process can target us for
      // waking, then sleep. If no idle workers exist, schedule_process
      // falls through to post to any live worker's notification, which
      // we'll see when we eventually call commit_wait.
      iree_task_worker_mark_idle(worker);
      is_active = false;

      IREE_TRACE_ZONE_BEGIN_NAMED(z_wait,
                                  "iree_task_worker_main_pump_wake_wait");
      iree_notification_commit_wait(
          &worker->wake_notification, wait_token,
          /*spin_ns=*/worker->executor->worker_spin_ns,
          /*deadline_ns=*/IREE_TIME_INFINITE_FUTURE);
      IREE_TRACE_ZONE_END(z_wait);

      // Woke from a wait - query the processor ID in case we migrated during
      // the sleep.
      iree_task_worker_update_processor_id(worker);
    }
  }
}

// Thread entry point for each worker.
static int iree_task_worker_main(iree_task_worker_t* worker) {
  IREE_TRACE_ZONE_BEGIN(thread_zone);

  // We cannot rely on the global process settings for FPU state.
  // Be explicit here on what we need.
  iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);

  // Reset affinity (as it can change over time).
  // TODO(benvanik): call this after waking in case CPU hotplugging happens.
  iree_thread_request_affinity(worker->thread, worker->ideal_thread_affinity);

  // Enter the running state immediately. Note that we could have been requested
  // to exit while suspended/still starting up, so check that here before we
  // mess with any data structures.
  const bool should_run =
      iree_atomic_exchange(&worker->state, IREE_TASK_WORKER_STATE_RUNNING,
                           iree_memory_order_acq_rel) !=
      IREE_TASK_WORKER_STATE_EXITING;
  if (IREE_LIKELY(should_run)) {
    // << work happens here >>
    iree_task_worker_pump_until_exit(worker);
  }

  // Indicate idle immediately before exit.
  iree_task_worker_mark_idle(worker);

  IREE_TRACE_ZONE_END(thread_zone);
  iree_atomic_store(&worker->state, IREE_TASK_WORKER_STATE_ZOMBIE,
                    iree_memory_order_release);
  iree_notification_post(&worker->state_notification, IREE_ALL_WAITERS);
  return 0;
}
