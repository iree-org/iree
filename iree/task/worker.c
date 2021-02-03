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

#include "iree/task/worker.h"

#include "iree/base/internal/debugging.h"
#include "iree/base/internal/math.h"
#include "iree/task/executor_impl.h"
#include "iree/task/task_impl.h"

static int iree_task_worker_main(iree_task_worker_t* worker);

iree_status_t iree_task_worker_initialize(
    iree_task_executor_t* executor, iree_host_size_t worker_index,
    const iree_task_topology_group_t* topology_group,
    iree_prng_splitmix64_state_t* seed_prng, iree_task_worker_t* out_worker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  out_worker->executor = executor;
  out_worker->worker_bit = iree_task_affinity_for_worker(worker_index);
  out_worker->ideal_thread_affinity = topology_group->ideal_thread_affinity;
  out_worker->constructive_sharing_mask =
      topology_group->constructive_sharing_mask;
  out_worker->max_theft_attempts =
      executor->worker_count / IREE_TASK_EXECUTOR_MAX_THEFT_ATTEMPTS_DIVISOR;
  iree_prng_minilcg128_initialize(iree_prng_splitmix64_next(seed_prng),
                                  &out_worker->theft_prng);

  iree_task_worker_state_t initial_state = IREE_TASK_WORKER_STATE_RUNNING;
  if (executor->scheduling_mode &
      IREE_TASK_SCHEDULING_MODE_DEFER_WORKER_STARTUP) {
    // User is favoring startup latency vs. initial scheduling latency. Our
    // thread will be created suspended and not first scheduled until work
    // arrives for it, (almost) ensuring no context switches and 10x+ lower
    // blocking startup time.
    initial_state = IREE_TASK_WORKER_STATE_SUSPENDED;
  }
  iree_atomic_store_int32(&out_worker->state, initial_state,
                          iree_memory_order_seq_cst);

  iree_notification_initialize(&out_worker->wake_notification);
  iree_notification_initialize(&out_worker->state_notification);
  iree_atomic_task_slist_initialize(&out_worker->mailbox_slist);
  iree_task_queue_initialize(&out_worker->local_task_queue);

  iree_thread_create_params_t thread_params;
  memset(&thread_params, 0, sizeof(thread_params));
  thread_params.name = iree_make_cstring_view(topology_group->name);
  thread_params.create_suspended =
      initial_state == IREE_TASK_WORKER_STATE_SUSPENDED;
  thread_params.priority_class = IREE_THREAD_PRIORITY_CLASS_NORMAL;
  thread_params.initial_affinity = out_worker->ideal_thread_affinity;

  // NOTE: if the thread creation fails we'll bail here and let the caller
  // cleanup by calling deinitialize (which is safe because we zero init
  // everything).
  iree_status_t status = iree_thread_create(
      (iree_thread_entry_t)iree_task_worker_main, out_worker, thread_params,
      executor->allocator, &out_worker->thread);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Returns true if the worker is in the zombie state (exited and awaiting
// teardown).
static bool iree_task_worker_is_zombie(iree_task_worker_t* worker) {
  return iree_atomic_load_int32(&worker->state, iree_memory_order_seq_cst) ==
         IREE_TASK_WORKER_STATE_ZOMBIE;
}

void iree_task_worker_deinitialize(iree_task_worker_t* worker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait for the thread to enter the zombie state indicating it has exited our
  // main function - it may still be live in the OS, but it'll not be touching
  // any of our data structures again so it's fine to blast away.
  if (worker->thread) {
    iree_notification_await(&worker->state_notification,
                            (iree_condition_fn_t)iree_task_worker_is_zombie,
                            worker);
  }
  iree_thread_release(worker->thread);

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
    case IREE_TASK_WORKER_STATE_SUSPENDED:
      // Worker was suspended; resume it so that it can exit itself.
      iree_thread_resume(worker->thread);
      break;
    case IREE_TASK_WORKER_STATE_ZOMBIE:
      // Worker already exited; reset state to ZOMBIE.
      iree_atomic_store_int32(&worker->state, IREE_TASK_WORKER_STATE_ZOMBIE,
                              iree_memory_order_seq_cst);
      break;
    default:
      // Worker now set to EXITING and should exit soon.
      break;
  }

  // Kick the worker in case it is waiting for work.
  iree_notification_post(&worker->wake_notification, 1);

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
  iree_task_t* task = iree_task_queue_try_steal(
      &worker->local_task_queue, target_queue,
      /*max_tasks=*/IREE_TASK_EXECUTOR_MAX_THEFT_TASK_COUNT);
  if (task) return task;

  // If we still didn't steal any tasks then let's try the slist instead.
  task = iree_atomic_task_slist_pop(&worker->mailbox_slist);
  if (task) return task;

  return NULL;
}

// Executes a task on a worker.
// Only task types that are scheduled to workers are handled; all others must be
// handled by the coordinator during scheduling.
static iree_status_t iree_task_worker_execute(
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
      IREE_RETURN_IF_ERROR(
          iree_task_call_execute((iree_task_call_t*)task, pending_submission));
      break;
    }
    case IREE_TASK_TYPE_DISPATCH_SLICE: {
      IREE_RETURN_IF_ERROR(iree_task_dispatch_slice_execute(
          (iree_task_dispatch_slice_t*)task, pending_submission));
      break;
    }
    case IREE_TASK_TYPE_DISPATCH_SHARD: {
      IREE_RETURN_IF_ERROR(iree_task_dispatch_shard_execute(
          (iree_task_dispatch_shard_t*)task, pending_submission));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "incorrect task type for worker execution");
  }

  // NOTE: task is invalidated here!
  task = NULL;

  return iree_ok_status();
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

  // No tasks to run; let the caller know we want to wait for more.
  if (!task) {
    IREE_TRACE_ZONE_END(z0);
    return false;
  }

  // Execute the task (may call out to arbitrary user code and may submit more
  // tasks for execution).
  iree_status_t status =
      iree_task_worker_execute(worker, task, pending_submission);

  // TODO(#4026): propagate failure to task scope.
  // We currently drop the error on the floor here; that's because the error
  // should have already been propagated to the scope and everyone should be
  // checking that before running things anyway.
  //
  // Since we can host work from multiple scopes and want to ensure an error
  // in one doesn't bring down the whole system we pretend we executed
  // something here by falling through.
  IREE_ASSERT_TRUE(iree_status_is_ok(status));
  iree_status_ignore(status);

  IREE_TRACE_ZONE_END(z0);
  return true;  // try again
}

// Alternates between pumping ready tasks in the worker queue and waiting
// for more tasks to arrive. Only returns when the worker has been asked by
// the executor to exit.
static void iree_task_worker_pump_until_exit(iree_task_worker_t* worker) {
  // Pump the thread loop to process more tasks.
  while (true) {
    // If we fail to find any work to do we'll wait at the end of this loop.
    // In order not to not miss any work that is enqueued after we've already
    // checked a particular source we use an interruptable wait token that
    // will prevent the wait from happening if anyone touches the data
    // structures we use.
    iree_wait_token_t wait_token =
        iree_notification_prepare_wait(&worker->wake_notification);
    iree_atomic_task_affinity_set_fetch_and(&worker->executor->worker_idle_mask,
                                            ~worker->worker_bit,
                                            iree_memory_order_seq_cst);

    // Check state to see if we've been asked to exit.
    if (iree_atomic_load_int32(&worker->state, iree_memory_order_seq_cst) ==
        IREE_TASK_WORKER_STATE_EXITING) {
      // Thread exit requested - cancel pumping.
      iree_notification_cancel_wait(&worker->wake_notification);
      // TODO(benvanik): complete tasks before exiting?
      break;
    }

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
    iree_atomic_task_affinity_set_fetch_or(&worker->executor->worker_idle_mask,
                                           worker->worker_bit,
                                           iree_memory_order_seq_cst);

    // When we encounter a complete lack of work we can self-nominate to check
    // the global work queue and distribute work to other threads. Only one
    // coordinator can be running at a time so we also ensure that if another
    // is doing its work we gracefully wait for it. It's fine to block in here
    // as the next thing we'd have done is go idle anyway.

    // First self-nominate; this *may* do something or just be ignored (if
    // another worker is already coordinating).
    iree_task_executor_coordinate(worker->executor, worker,
                                  /*speculative=*/true);

    // If nothing has been enqueued since we started this loop (so even
    // coordination didn't find anything) we go idle. Otherwise we fall
    // through and try the loop again.
    if (schedule_dirty ||
        !iree_task_queue_is_empty(&worker->local_task_queue)) {
      // Have more work to do; loop around to try another pump.
      iree_notification_cancel_wait(&worker->wake_notification);
    } else {
      IREE_TRACE_ZONE_BEGIN_NAMED(z_wait,
                                  "iree_task_worker_main_pump_wake_wait");
      iree_notification_commit_wait(&worker->wake_notification, wait_token);
      IREE_TRACE_ZONE_END(z_wait);
    }

    // Wait completed.
    // Jump back up and try pumping any tasks that arrived.
    continue;
  }
}

// Thread entry point for each worker.
static int iree_task_worker_main(iree_task_worker_t* worker) {
  IREE_TRACE_ZONE_BEGIN(thread_zone);

  // Reset affinity (as it can change over time).
  // TODO(benvanik): call this after waking in case CPU hotplugging happens.
  iree_thread_request_affinity(worker->thread, worker->ideal_thread_affinity);

  // Enter the running state immediately. Note that we could have been requested
  // to exit while suspended/still starting up, so check that here before we
  // mess with any data structures.
  bool should_run =
      iree_atomic_exchange_int32(&worker->state, IREE_TASK_WORKER_STATE_RUNNING,
                                 iree_memory_order_seq_cst) !=
      IREE_TASK_WORKER_STATE_EXITING;
  if (IREE_LIKELY(should_run)) {
    // << work happens here >>
    iree_task_worker_pump_until_exit(worker);
  }

  IREE_TRACE_ZONE_END(thread_zone);
  iree_atomic_store_int32(&worker->state, IREE_TASK_WORKER_STATE_ZOMBIE,
                          iree_memory_order_seq_cst);
  iree_notification_post(&worker->state_notification, IREE_ALL_WAITERS);
  return 0;
}
