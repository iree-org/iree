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
  out_worker->worker_bit = iree_task_affinity_for_worker(worker_index);
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
  iree_task_affinity_set_t old_idle_mask =
      iree_atomic_task_affinity_set_fetch_and(
          &worker->executor->worker_idle_mask, ~worker->worker_bit,
          iree_memory_order_relaxed);
  (void)old_idle_mask;
  IREE_TRACE_PLOT_VALUE_F32(
      worker->executor->trace_name,
      old_idle_mask
          ? (100.0f -
             100.0f * (iree_task_affinity_set_count_ones(old_idle_mask) - 1) /
                 (float)worker->executor->worker_count)
          : 100.0f);
}

// Marks the worker as "idle" (sleeping/spinning waiting to wake).
// The idle mask is accessed with 'relaxed' order because it's just a hint.
static void iree_task_worker_mark_idle(iree_task_worker_t* worker) {
  iree_task_affinity_set_t old_idle_mask =
      iree_atomic_task_affinity_set_fetch_or(
          &worker->executor->worker_idle_mask, worker->worker_bit,
          iree_memory_order_relaxed);
  (void)old_idle_mask;
  IREE_TRACE_PLOT_VALUE_F32(
      worker->executor->trace_name,
      100.0f - 100.0f * (iree_task_affinity_set_count_ones(old_idle_mask) + 1) /
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

      // For budget-1 processes there is exactly one drainer (us), so
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
    // seq_cst is required here because this is one half of a Dekker-style
    // protocol with schedule_process:
    //   Worker:    store(schedule_state=IDLE)  then load(needs_drain)
    //   Scheduler: store(needs_drain=1)        then CAS(schedule_state)
    // Both threads store one variable and load the other. Release/acquire
    // on different variables does not prevent StoreLoad reordering — on ARM
    // the load below could execute before this store is globally visible,
    // causing the worker to miss a needs_drain=1 signal and strand the
    // process. seq_cst provides the required StoreLoad barrier.
    iree_atomic_store(&process->schedule_state,
                      (int32_t)IREE_TASK_PROCESS_SCHEDULE_IDLE,
                      iree_memory_order_seq_cst);

    // Final race check: an external event may have set needs_drain after our
    // exchange (saw 0) but before we stored IDLE. That event's
    // schedule_process saw DRAINING and returned without pushing, trusting us
    // to re-check. If needs_drain is set, reclaim the process.
    if (iree_atomic_load(&process->needs_drain, iree_memory_order_acquire)) {
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

// Eagerly completes a compute process: transitions schedule_state to IDLE,
// calls the completion callback (semaphore signaling, dependent activation),
// and schedules any activated dependents. Called by the first worker to
// claim completion via CAS on completion_claimed.
//
// Does NOT free drain-accessed resources — that is handled by
// iree_task_worker_release_compute_process when the last drainer exits.
static void iree_task_worker_eager_complete_compute_process(
    iree_task_worker_t* worker, iree_task_process_t* process) {
  iree_task_executor_t* executor = worker->executor;

  iree_atomic_store(&process->schedule_state,
                    (int32_t)IREE_TASK_PROCESS_SCHEDULE_IDLE,
                    iree_memory_order_release);

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

// Releases a compute process: clears the slot, resets the slot for reuse,
// and frees drain-accessed resources. Called by the worker that successfully
// CAS'd active_drainers from gen|0 to gen|SENTINEL.
//
// |tagged_sentinel| is the full 64-bit value (gen|SENTINEL) currently in
// active_drainers. The generation bits are preserved and incremented when
// the slot is reset, preventing ABA races on subsequent slot lifetimes.
//
// Ordering:
//   1. Clear process pointer — prevents new workers from entering via the
//      quick check (relaxed load of process). After this, no new worker will
//      START a new entry sequence for this slot. However, the quick check
//      uses relaxed ordering, so workers on other cores may still see a
//      stale non-zero pointer and enter the bail path.
//   2. Reset completion_claimed and atomically CAS active_drainers from
//      gen|SENTINEL to (gen+1)|0. The CAS loop waits for any in-flight
//      bailers (who entered via the stale quick check) to complete their
//      fetch_sub before resetting. The generation increment ensures that
//      any stale CAS from a prior generation will fail.
//   3. Call release_fn — frees processor context and other drain-accessed
//      resources. This runs AFTER the slot is fully clean so that the
//      release callback can safely signal "process memory is reusable."
static void iree_task_worker_release_compute_process(
    iree_task_worker_t* worker, iree_task_compute_slot_t* slot,
    iree_task_process_t* process, int64_t tagged_sentinel) {
  iree_task_executor_t* executor = worker->executor;

  // Step 1: Clear the process pointer. After this store (release), no new
  // worker will pass the quick check for this slot — except via stale
  // relaxed reads (the quick check uses relaxed ordering).
  iree_atomic_store(&slot->process, 0, iree_memory_order_release);

  // Step 2: Reset completion_claimed and atomically CAS active_drainers
  // from gen|SENTINEL to (gen+1)|0.
  //
  // Workers that passed the quick check with a stale non-zero read before
  // step 1 may still be in the bail path: they did fetch_add(1) on
  // active_drainers (seeing SENTINEL + N in the count bits), found
  // (int32_t)prev < 0, and are about to do fetch_sub(1) to undo.
  //
  // The CAS only succeeds when active_drainers is exactly our sentinel
  // value (no in-flight bailers have perturbed the count bits). The
  // generation increment on reset means any stale CAS from a worker that
  // observed a prior generation's count=0 will fail — the generation bits
  // won't match.
  //
  // completion_claimed is reset before the CAS so that a new process
  // placed during the release window (between step 1 and the CAS) does
  // not inherit a stale claimed flag.
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

  // Step 3: Re-wake if a new process was placed during the release window.
  //
  // Between step 1 (process pointer cleared) and the CAS above, a
  // concurrent schedule_process can place a new process in this slot —
  // place_in_compute_slot only checks that process==0. Workers woken by
  // that schedule_process find active_drainers sentinel (count < 0), bail,
  // and go back to sleep. After the CAS resets active_drainers to the next
  // generation with count=0, the slot is drainable but no workers know
  // about it. Detect this and re-wake.
  intptr_t new_process =
      iree_atomic_load(&slot->process, iree_memory_order_acquire);
  if (new_process) {
    iree_task_process_t* p = (iree_task_process_t*)new_process;
    int32_t budget = iree_task_process_worker_budget(p);
    iree_task_executor_wake_workers(executor, budget);
  }

  // Step 4: Free drain-accessed resources. The slot is fully clean and may
  // be reused by a new process placed by a concurrent schedule_process call.
  // release_fn operates on the old process via its pointer argument, which
  // is independent of whatever new process may now occupy the slot.
  iree_task_process_release_fn_t release_fn = process->release_fn;
  if (release_fn) {
    release_fn(process);
  }
}

// Scans executor compute slots for budget>1 processes and drains bounded work
// from one of them. Workers scan round-robin starting from
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
// handling of budget-1 processes between compute work.
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
      if (result.did_work) {
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
    // If we're the last drainer out and the process is terminal, try to
    // claim the release right by CAS-ing active_drainers from gen|0 to
    // gen|SENTINEL. The generation bits (from our fetch_sub result) must
    // match — if another worker completed an entire release cycle between
    // our fetch_sub and our CAS, the generation will have incremented and
    // our CAS fails harmlessly. This eliminates the ABA race that existed
    // with the 32-bit counter.
    int64_t old_drainers = iree_atomic_fetch_sub(&slot->active_drainers, 1,
                                                 iree_memory_order_acq_rel);
    int32_t remaining = (int32_t)old_drainers - 1;
    if (remaining == 0 && is_terminal) {
      // Construct the expected value: same generation, count=0.
      int64_t generation = old_drainers & ~(int64_t)UINT32_MAX;
      int64_t expected_empty = generation;  // gen | count=0
      int64_t sentinel = generation | IREE_TASK_SLOT_SENTINEL;
      if (iree_atomic_compare_exchange_strong(
              &slot->active_drainers, &expected_empty, sentinel,
              iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
        iree_task_worker_release_compute_process(worker, slot, process,
                                                 sentinel);
      }
      // CAS failed: either a new worker incremented active_drainers between
      // our decrement and this CAS, or the generation changed (another worker
      // already released this slot). Either way, release is handled.
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
  idle_mask &= ~worker->worker_bit;

  while (claimed > 0 && idle_mask) {
    iree_host_size_t worker_index =
        iree_task_affinity_set_count_trailing_zeros(idle_mask);
    if (worker_index >= executor->worker_count) break;
    iree_notification_post(&executor->workers[worker_index].wake_notification,
                           1);
    idle_mask &= ~iree_task_affinity_for_worker(worker_index);
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

    // Drain all available immediate processes (budget-1), then scan compute
    // slots (budget>1) for one round of bounded work.
    bool did_work = false;
    while (iree_task_worker_drain_process(worker)) {
      did_work = true;
    }
    if (iree_task_worker_drain_compute_slots(worker)) {
      did_work = true;
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
