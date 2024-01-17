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
#include "iree/task/executor_impl.h"
#include "iree/task/post_batch.h"
#include "iree/task/submission.h"
#include "iree/task/task_impl.h"
#include "iree/task/tuning.h"

#define IREE_TASK_WORKER_MIN_STACK_SIZE (32 * 1024)

static int iree_task_worker_main(iree_task_worker_t* worker);

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
  out_worker->max_theft_attempts =
      executor->worker_count / IREE_TASK_EXECUTOR_MAX_THEFT_ATTEMPTS_DIVISOR;
  iree_prng_minilcg128_initialize(iree_prng_splitmix64_next(seed_prng),
                                  &out_worker->theft_prng);
  out_worker->local_memory = local_memory;
  out_worker->processor_id = 0;
  out_worker->processor_tag = 0;

  iree_notification_initialize(&out_worker->wake_notification);
  iree_notification_initialize(&out_worker->state_notification);
  iree_atomic_task_slist_initialize(&out_worker->mailbox_slist);
  iree_task_queue_initialize(&out_worker->local_task_queue);

  iree_task_worker_state_t initial_state = IREE_TASK_WORKER_STATE_RUNNING;
  iree_atomic_store_int32(&out_worker->state, initial_state,
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
  // cleanup by calling deinitialize (which is safe because we zero init
  // everything).
  iree_status_t status = iree_thread_create(
      (iree_thread_entry_t)iree_task_worker_main, out_worker, thread_params,
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
      (iree_task_worker_state_t)iree_atomic_exchange_int32(
          &worker->state, IREE_TASK_WORKER_STATE_EXITING,
          iree_memory_order_acq_rel);
  switch (prev_state) {
    case IREE_TASK_WORKER_STATE_ZOMBIE:
      // Worker already exited; reset state to ZOMBIE.
      iree_atomic_store_int32(&worker->state, IREE_TASK_WORKER_STATE_ZOMBIE,
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
  return iree_atomic_load_int32(&worker->state, iree_memory_order_acquire) ==
         IREE_TASK_WORKER_STATE_ZOMBIE;
}

void iree_task_worker_await_exit(iree_task_worker_t* worker) {
  if (!worker->thread) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_task_worker_request_exit(worker);
  iree_notification_await(&worker->state_notification,
                          (iree_condition_fn_t)iree_task_worker_is_zombie,
                          worker, iree_infinite_timeout());

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_worker_deinitialize(iree_task_worker_t* worker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Must have called request_exit/await_exit.
  IREE_ASSERT_TRUE(iree_task_worker_is_zombie(worker));

  iree_thread_release(worker->thread);
  worker->thread = NULL;

  // Release unfinished tasks by flushing the mailbox (which if we're here can't
  // get anything more posted to it) and then discarding everything we still
  // have a reference to.
  iree_atomic_task_slist_discard(&worker->mailbox_slist);
  iree_task_list_discard(&worker->local_task_queue.list);

  iree_notification_deinitialize(&worker->wake_notification);
  iree_notification_deinitialize(&worker->state_notification);
  iree_atomic_task_slist_deinitialize(&worker->mailbox_slist);
  iree_task_queue_deinitialize(&worker->local_task_queue);

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_worker_post_tasks(iree_task_worker_t* worker,
                                 iree_task_list_t* list) {
  // Move the list into the mailbox. Note that the mailbox is LIFO and this list
  // is concatenated with its current order preserved (which should be LIFO).
  iree_atomic_task_slist_concat(&worker->mailbox_slist, list->head, list->tail);
  memset(list, 0, sizeof(*list));
}

iree_task_t* iree_task_worker_try_steal_task(iree_task_worker_t* worker,
                                             iree_task_queue_t* target_queue,
                                             iree_host_size_t max_tasks) {
  // Try to grab tasks from the worker; if more than one task is stolen then the
  // first will be returned and the remaining will be added to the target queue.
  iree_task_t* task = iree_task_queue_try_steal(&worker->local_task_queue,
                                                target_queue, max_tasks);
  if (task) return task;

  // If we still didn't steal any tasks then let's try the slist instead.
  task = iree_atomic_task_slist_pop(&worker->mailbox_slist);
  if (task) return task;

  return NULL;
}

// Executes a task on a worker.
// Only task types that are scheduled to workers are handled; all others must be
// handled by the coordinator during scheduling.
static void iree_task_worker_execute(
    iree_task_worker_t* worker, iree_task_t* task,
    iree_task_submission_t* pending_submission) {
  // Execute the task and resolve the task and gather any tasks that are now
  // ready for submission to the executor. They'll be scheduled the next time
  // the coordinator runs.
  //
  // TODO(benvanik): think a bit more about this timing; this ensures we have
  // BFS behavior at the cost of the additional merge overhead - it's probably
  // worth it?
  // TODO(benvanik): handle partial tasks and re-queuing.
  switch (task->type) {
    case IREE_TASK_TYPE_CALL: {
      iree_task_call_execute((iree_task_call_t*)task, pending_submission);
      break;
    }
    case IREE_TASK_TYPE_DISPATCH_SHARD: {
      iree_task_dispatch_shard_execute(
          (iree_task_dispatch_shard_t*)task, worker->processor_id,
          worker->worker_index, worker->local_memory, pending_submission);
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("incorrect task type for worker execution");
      break;
  }

  // NOTE: task is invalidated above and must not be used!
  task = NULL;
}

// Pumps the worker thread once, processing a single task.
// Returns true if pumping should continue as there are more tasks remaining or
// false if the caller should wait for more tasks to be posted.
static bool iree_task_worker_pump_once(
    iree_task_worker_t* worker, iree_task_submission_t* pending_submission) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check the local work queue for any work we know we should start
  // processing immediately. Other workers may try to steal some of this work
  // if we take too long.
  iree_task_t* task = iree_task_queue_pop_front(&worker->local_task_queue);

  // Check the mailbox to see if we have incoming work that has been posted.
  // We try to greedily move it to our local work list so that we can work
  // with the full thread-local pending task list.
  if (!task) {
    // NOTE: there's a potential for theft pessimization if the queue runs too
    // low and there's nothing there when a thief goes to grab some tasks. A
    // standout there would indicate that we weren't scheduling very well in the
    // first place (large uneven workloads for various workers, bad distribution
    // in the face of heterogenous multi-core architectures where some workers
    // complete tasks faster than others, etc).
    task = iree_task_queue_flush_from_lifo_slist(&worker->local_task_queue,
                                                 &worker->mailbox_slist);
  }

#if IREE_TASK_EXECUTOR_MAX_THEFT_ATTEMPTS_DIVISOR > 0
  // If we ran out of work assigned to this specific worker try to steal some
  // from other workers that we hopefully share some of the cache hierarchy
  // with. Their tasks will be moved from their local queue into ours and the
  // the first task in the queue is popped off and returned.
  if (!task) {
    task = iree_task_executor_try_steal_task(
        worker->executor, worker->constructive_sharing_mask,
        worker->max_theft_attempts, &worker->theft_prng,
        &worker->local_task_queue);
  }
#endif  // IREE_TASK_EXECUTOR_MAX_THEFT_ATTEMPTS_DIVISOR > 0

  // No tasks to run; let the caller know we want to wait for more.
  if (!task) {
    IREE_TRACE_ZONE_END(z0);
    return false;
  }

  // Execute the task (may call out to arbitrary user code and may submit more
  // tasks for execution).
  iree_task_worker_execute(worker, task, pending_submission);

  IREE_TRACE_ZONE_END(z0);
  return true;  // try again
}

// Updates the cached processor ID field in the worker.
static void iree_task_worker_update_processor_id(iree_task_worker_t* worker) {
  iree_cpu_requery_processor_id(&worker->processor_tag, &worker->processor_id);
}

// Alternates between pumping ready tasks in the worker queue and waiting
// for more tasks to arrive. Only returns when the worker has been asked by
// the executor to exit.
static void iree_task_worker_pump_until_exit(iree_task_worker_t* worker) {
  // Initial processor ID assignment. We normally refresh this upon waking from
  // a wait but it's possible that there's already work pending and we want to
  // be able to process it with the proper processor ID immediately.
  iree_task_worker_update_processor_id(worker);

  // Pump the thread loop to process more tasks.
  while (true) {
    // If we fail to find any work to do we'll wait at the end of this loop.
    // In order not to not miss any work that is enqueued after we've already
    // checked a particular source we use an interruptable wait token that
    // will prevent the wait from happening if anyone touches the data
    // structures we use.
    iree_wait_token_t wait_token =
        iree_notification_prepare_wait(&worker->wake_notification);

    // The masks are accessed with 'relaxed' order because they are just hints.
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

    // Check state to see if we've been asked to exit.
    if (iree_atomic_load_int32(&worker->state, iree_memory_order_acquire) ==
        IREE_TASK_WORKER_STATE_EXITING) {
      // Thread exit requested - cancel pumping.
      iree_notification_cancel_wait(&worker->wake_notification);
      // TODO(benvanik): complete tasks before exiting?
      break;
    }

    // TODO(benvanik): we could try to update the processor ID here before we
    // begin a new batch of work - assuming it's not too expensive.

    iree_task_submission_t pending_submission;
    iree_task_submission_initialize(&pending_submission);

    while (iree_task_worker_pump_once(worker, &pending_submission)) {
      // All work done ^, which will return false when the worker should wait.
    }

    bool schedule_dirty = false;
    if (!iree_task_submission_is_empty(&pending_submission)) {
      iree_task_executor_merge_submission(worker->executor,
                                          &pending_submission);
      schedule_dirty = true;
    }

    // We've finished all the work we have scheduled so set our idle flag.
    // This ensures that if any other thread comes in and wants to give us
    // work we will properly coordinate/wake below.
    old_idle_mask = iree_atomic_task_affinity_set_fetch_or(
        &worker->executor->worker_idle_mask, worker->worker_bit,
        iree_memory_order_relaxed);
    (void)old_idle_mask;
    IREE_TRACE_PLOT_VALUE_F32(
        worker->executor->trace_name,
        100.0f - 100.0f *
                     (iree_task_affinity_set_count_ones(old_idle_mask) + 1) /
                     (float)worker->executor->worker_count);

    // When we encounter a complete lack of work we can self-nominate to check
    // the global work queue and distribute work to other threads. Only one
    // coordinator can be running at a time so we also ensure that if another
    // is doing its work we gracefully wait for it. It's fine to block in here
    // as the next thing we'd have done is go idle anyway.

    // First self-nominate; this *may* do something or just be ignored (if
    // another worker is already coordinating).
    iree_task_executor_coordinate(worker->executor, worker);

    // If nothing has been enqueued since we started this loop (so even
    // coordination didn't find anything) we go idle. Otherwise we fall
    // through and try the loop again.
    if (schedule_dirty ||
        !iree_task_queue_is_empty(&worker->local_task_queue)) {
      // Have more work to do; loop around to try another pump.
      iree_notification_cancel_wait(&worker->wake_notification);
    } else {
      // Spin/wait in the kernel. We don't care if the condition fails as we're
      // just using it as a pulse.
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

    // Wait completed.
    // Jump back up and try pumping any tasks that arrived.
    continue;
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
      iree_atomic_exchange_int32(&worker->state, IREE_TASK_WORKER_STATE_RUNNING,
                                 iree_memory_order_acq_rel) !=
      IREE_TASK_WORKER_STATE_EXITING;
  if (IREE_LIKELY(should_run)) {
    // << work happens here >>
    iree_task_worker_pump_until_exit(worker);
  }

  IREE_TRACE_ZONE_END(thread_zone);
  iree_atomic_store_int32(&worker->state, IREE_TASK_WORKER_STATE_ZOMBIE,
                          iree_memory_order_release);
  iree_notification_post(&worker->state_notification, IREE_ALL_WAITERS);
  return 0;
}
