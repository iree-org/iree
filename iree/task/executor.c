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

#include "iree/task/executor.h"

#include "iree/base/internal/debugging.h"
#include "iree/base/internal/math.h"
#include "iree/task/executor_impl.h"
#include "iree/task/task_impl.h"

static void iree_task_executor_destroy(iree_task_executor_t* executor);

iree_status_t iree_task_executor_create(
    iree_task_scheduling_mode_t scheduling_mode,
    const iree_task_topology_t* topology, iree_allocator_t allocator,
    iree_task_executor_t** out_executor) {
  iree_host_size_t worker_count = iree_task_topology_group_count(topology);
  if (worker_count > IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "requested %zu workers but a maximum of %d is allowed", worker_count,
        IREE_TASK_EXECUTOR_MAX_WORKER_COUNT);
  }

  // TODO(benvanik): support a threadless mode where we have one dummy worker
  // that just holds the lists but is pumped from donate_caller.
  if (worker_count == 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "threadless donate-only executor mode not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_executor);
  *out_executor = NULL;

  iree_host_size_t executor_size =
      sizeof(iree_task_executor_t) + worker_count * sizeof(iree_task_worker_t);

  iree_task_executor_t* executor = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, executor_size, (void**)&executor));
  memset(executor, 0, executor_size);
  iree_atomic_ref_count_init(&executor->ref_count);
  executor->allocator = allocator;
  executor->scheduling_mode = scheduling_mode;
  iree_atomic_task_slist_initialize(&executor->incoming_ready_slist);
  iree_atomic_task_slist_initialize(&executor->incoming_waiting_slist);
  iree_slim_mutex_initialize(&executor->coordinator_mutex);
  iree_slim_mutex_initialize(&executor->wait_mutex);

  // Simple PRNG used to generate seeds for the per-worker PRNGs used to
  // distribute work. This isn't strong (and doesn't need to be); it's just
  // enough to ensure each worker gets a sufficiently random seed for itself to
  // then generate entropy with. As a hack we use out_executor's address, as
  // that should live on the caller stack and with ASLR that's likely pretty
  // random itself. I'm sure somewhere a mathemetician just cringed :)
  iree_prng_splitmix64_state_t seed_prng;
  iree_prng_splitmix64_initialize(/*seed=*/(uint64_t)(out_executor),
                                  &seed_prng);
  iree_prng_minilcg128_initialize(iree_prng_splitmix64_next(&seed_prng),
                                  &executor->donation_theft_prng);

  iree_status_t status = iree_ok_status();

  // Wait set used to batch syscalls for polling/waiting on wait handles.
  // This is currently limited to a relatively small max to make bad behavior
  // clearer with nice RESOURCE_EXHAUSTED errors.
  if (iree_status_is_ok(status)) {
    status = iree_wait_set_allocate(IREE_TASK_EXECUTOR_MAX_OUTSTANDING_WAITS,
                                    allocator, &executor->wait_set);
  }

  // Pool used for all dispatch->slice fanout tasks. These only live within the
  // executor and since we know the precise lifetime of them we can keep them
  // entirely within the system here.
  if (iree_status_is_ok(status)) {
    status = iree_task_pool_initialize(allocator, sizeof(iree_task_fence_t), 8,
                                       &executor->fence_task_pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_task_pool_initialize(
        allocator,
        iree_max(sizeof(iree_task_dispatch_shard_t),
                 sizeof(iree_task_dispatch_slice_t)),
        worker_count *
            iree_max(IREE_TASK_EXECUTOR_INITIAL_SHARD_RESERVATION_PER_WORKER,
                     IREE_TASK_EXECUTOR_INITIAL_SLICE_RESERVATION_PER_WORKER),
        &executor->dispatch_task_pool);
  }

  // Bring up the workers; the threads will be created here but be suspended
  // (if the platform supports it) awaiting the first tasks getting scheduled.
  if (iree_status_is_ok(status)) {
    executor->worker_count = worker_count;
    executor->workers = (iree_task_worker_t*)(executor + 1);
    iree_task_affinity_set_t worker_idle_mask = 0;
    iree_task_affinity_set_t worker_live_mask = 0;
    iree_task_affinity_set_t worker_suspend_mask = 0;
    for (iree_host_size_t i = 0; i < worker_count; ++i) {
      iree_task_affinity_set_t worker_bit = iree_task_affinity_for_worker(i);
      worker_idle_mask |= worker_bit;
      worker_live_mask |= worker_bit;
      if (executor->scheduling_mode &
          IREE_TASK_SCHEDULING_MODE_DEFER_WORKER_STARTUP) {
        worker_suspend_mask |= worker_bit;
      }

      iree_task_worker_t* worker = &executor->workers[i];
      status = iree_task_worker_initialize(
          executor, i, iree_task_topology_get_group(topology, i), &seed_prng,
          worker);
      if (!iree_status_is_ok(status)) break;
    }
    iree_atomic_task_affinity_set_store(&executor->worker_live_mask,
                                        worker_live_mask,
                                        iree_memory_order_relaxed);
    iree_atomic_task_affinity_set_store(&executor->worker_suspend_mask,
                                        worker_suspend_mask,
                                        iree_memory_order_relaxed);
    iree_atomic_task_affinity_set_store(&executor->worker_idle_mask,
                                        worker_idle_mask,
                                        iree_memory_order_relaxed);
  }

  if (!iree_status_is_ok(status)) {
    // NOTE: destroy will ensure that any workers we have initialized are
    // properly cleaned up.
    iree_task_executor_destroy(executor);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_executor = executor;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_task_executor_destroy(iree_task_executor_t* executor) {
  if (!executor) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // First ask all workers to exit. We do this prior to waiting on them to exit
  // so that we parallelize the shutdown logic (which may flush pending tasks).
  for (iree_host_size_t i = 0; i < executor->worker_count; ++i) {
    iree_task_worker_t* worker = &executor->workers[i];
    iree_task_worker_request_exit(worker);
  }

  // Now that all workers should be in the process of exiting we can join with
  // them. Some may take longer than others to exit but that's fine as we can't
  // return from here until they do anyway.
  for (iree_host_size_t i = 0; i < executor->worker_count; ++i) {
    iree_task_worker_t* worker = &executor->workers[i];
    iree_task_worker_deinitialize(worker);
  }

  iree_wait_set_free(executor->wait_set);
  iree_slim_mutex_deinitialize(&executor->wait_mutex);
  iree_slim_mutex_deinitialize(&executor->coordinator_mutex);
  iree_atomic_task_slist_deinitialize(&executor->incoming_ready_slist);
  iree_atomic_task_slist_deinitialize(&executor->incoming_waiting_slist);
  iree_task_pool_deinitialize(&executor->fence_task_pool);
  iree_task_pool_deinitialize(&executor->dispatch_task_pool);
  iree_allocator_free(executor->allocator, executor);

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_executor_retain(iree_task_executor_t* executor) {
  if (executor) {
    iree_atomic_ref_count_inc(&executor->ref_count);
  }
}

void iree_task_executor_release(iree_task_executor_t* executor) {
  if (executor && iree_atomic_ref_count_dec(&executor->ref_count) == 1) {
    iree_task_executor_destroy(executor);
  }
}

iree_status_t iree_task_executor_acquire_fence(iree_task_executor_t* executor,
                                               iree_task_scope_t* scope,
                                               iree_task_fence_t** out_fence) {
  *out_fence = NULL;
  iree_task_fence_t* fence = NULL;
  IREE_RETURN_IF_ERROR(iree_task_pool_acquire(&executor->fence_task_pool,
                                              (iree_task_t**)&fence));
  iree_task_fence_initialize(scope, fence);
  fence->header.pool = &executor->fence_task_pool;
  *out_fence = fence;
  return iree_ok_status();
}

// Schedules a generic task to a worker matching its affinity.
// The task will be posted to the worker mailbox and available for the worker to
// begin processing as soon as the |post_batch| is submitted.
//
// Only called during coordination and expects the coordinator lock to be held.
static void iree_task_executor_relay_to_worker(
    iree_task_executor_t* executor, iree_task_post_batch_t* post_batch,
    iree_task_t* task) {
  iree_host_size_t worker_index =
      iree_task_post_batch_select_worker(post_batch, task->affinity_set);
  iree_task_post_batch_enqueue(post_batch, worker_index, task);
}

// Schedules all ready tasks in the |pending_submission| list.
// Task may enqueue zero or more new tasks (or newly-ready/waiting tasks) to
// |pending_submission| or queue work for posting to workers via the
// |post_batch|.
//
// NOTE: the pending submission list we walk here is in FIFO order and the
// post batch we are building is in LIFO; this means that as we pop off the
// least recently added tasks from the submission (nice in-order traversal) we
// are pushing them as what will become the least recent tasks in the batch.
//
// Only called during coordination and expects the coordinator lock to be held.
void iree_task_executor_schedule_ready_tasks(
    iree_task_executor_t* executor, iree_task_submission_t* pending_submission,
    iree_task_post_batch_t* post_batch) {
  if (iree_task_list_is_empty(&pending_submission->ready_list)) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_task_t* task = NULL;
  while ((task = iree_task_list_pop_front(&pending_submission->ready_list))) {
    switch (task->type) {
      case IREE_TASK_TYPE_NOP:
        // Doesn't do anything; just retire and continue on to any dependents.
        iree_task_nop_retire((iree_task_nop_t*)task, pending_submission);
        break;
      case IREE_TASK_TYPE_CALL:
      case IREE_TASK_TYPE_DISPATCH_SLICE: {
        // Generic routing to workers for tasks that should always run there.
        iree_task_executor_relay_to_worker(executor, post_batch, task);
        break;
      }
      case IREE_TASK_TYPE_BARRIER: {
        // Retire the barrier to (possibly) ready up all dependent tasks.
        // This acts as a fan-out in cases where the dependent task count >1.
        iree_task_barrier_retire((iree_task_barrier_t*)task,
                                 pending_submission);
        break;
      }
      case IREE_TASK_TYPE_FENCE: {
        // Scope fence hit; notifies the scope so that anyone waiting on the
        // fence can be notified without us having to do so explicitly.
        iree_task_fence_retire((iree_task_fence_t*)task, pending_submission);
        break;
      }
      case IREE_TASK_TYPE_WAIT: {
        // Waits may need to be moved into the wait list (not completed) or
        // retired (after the wait condition is met).
        if (task->flags & IREE_TASK_FLAG_WAIT_COMPLETED) {
          iree_task_wait_retire((iree_task_wait_t*)task, pending_submission);
        } else {
          iree_task_submission_enqueue(pending_submission, task);
        }
        break;
      }
      case IREE_TASK_TYPE_DISPATCH: {
        // Dispatches may need to be issued (fanning out the tiles to workers)
        // or retired (after all tiles have completed).
        if (task->flags & IREE_TASK_FLAG_DISPATCH_RETIRE) {
          iree_task_dispatch_retire((iree_task_dispatch_t*)task,
                                    pending_submission);
        } else {
          if (task->flags & IREE_TASK_FLAG_DISPATCH_SLICED) {
            iree_task_dispatch_issue_sliced((iree_task_dispatch_t*)task,
                                            &executor->dispatch_task_pool,
                                            pending_submission, post_batch);
          } else {
            iree_task_dispatch_issue_sharded((iree_task_dispatch_t*)task,
                                             &executor->dispatch_task_pool,
                                             pending_submission, post_batch);
          }
        }
        break;
      }
    }
  }
  IREE_TRACE_ZONE_END(z0);
}

void iree_task_executor_merge_submission(iree_task_executor_t* executor,
                                         iree_task_submission_t* submission) {
  // Concatenate all of the incoming tasks into the submission list.
  // Note that the submission stores tasks in LIFO order such that when they are
  // put into the LIFO atomic slist they match the order across all concats
  // (earlier concats are later in the LIFO list).
  iree_atomic_task_slist_concat(&executor->incoming_ready_slist,
                                submission->ready_list.head,
                                submission->ready_list.tail);
  iree_atomic_task_slist_concat(&executor->incoming_waiting_slist,
                                submission->waiting_list.head,
                                submission->waiting_list.tail);

  // NOTE: after concatenating the intrusive next_task pointers may immediately
  // be modified by other threads. We can no longer assume anything about the
  // submission lists and can only discard them.
  iree_task_submission_reset(submission);
}

void iree_task_executor_submit(iree_task_executor_t* executor,
                               iree_task_submission_t* submission) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Concatenate the submitted tasks onto our primary LIFO incoming lists.
  iree_task_executor_merge_submission(executor, submission);

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_executor_flush(iree_task_executor_t* executor) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Mostly a no-op today as we aren't deferring submission with the scheduling
  // mode. Instead, we'll just run the coordinator inline to ensure all tasks
  // are pushed to workers.
  iree_task_executor_coordinate(executor, /*current_worker=*/NULL,
                                /*wait_on_idle=*/false);

  IREE_TRACE_ZONE_END(z0);
}

// Merges incoming likely-unresolved wait tasks into the primary executor lists.
// The handle of each task will be inserted into the wait_set (where it may be
// a duplicate).
//
// Only called during coordination and expects the coordinator lock to be held.
static void iree_task_executor_merge_wait_list(
    iree_task_executor_t* executor, iree_task_list_t* incoming_waiting_list) {
  if (iree_task_list_is_empty(incoming_waiting_list)) return;

  iree_slim_mutex_lock(&executor->wait_mutex);

  // Walk the list of incoming wait tasks and add them to our wait_set.
  iree_task_wait_t* wait_task =
      (iree_task_wait_t*)iree_task_list_front(incoming_waiting_list);
  do {
    iree_status_t status =
        iree_wait_set_insert(executor->wait_set, wait_task->wait_handle);
    // TODO(#4026): propagate failure to the task scope.
    IREE_ASSERT_TRUE(iree_status_is_ok(status));
    iree_status_ignore(status);
    wait_task = (iree_task_wait_t*)wait_task->header.next_task;
  } while (wait_task);

  iree_slim_mutex_unlock(&executor->wait_mutex);

  // Add (in undefined order) to the primary wait list used for tracking the
  // root wait tasks until they are ready.
  iree_task_list_append(&executor->waiting_list, incoming_waiting_list);
}

// Finds the waiting task corresponding to |wake_handle| and retires it.
// Any dependent tasks will be enqueued in the |pending_submission| for issuing.
// If multiple tasks were waiting on the same wait handle all will be readied.
//
// Only called during coordination and expects the coordinator lock to be held.
// The wait lock must be held as the wait_set is modified.
static void iree_task_executor_wake_waiting_task(
    iree_task_executor_t* executor, iree_wait_handle_t wake_handle,
    iree_task_submission_t* pending_submission) {
  // Walk through the waiting_list and find all waits with this handle.
  // Some may not have resolved yet and need to remain in the list.
  iree_task_t* prev_task = NULL;
  iree_task_t* task = iree_task_list_front(&executor->waiting_list);
  while (task != NULL) {
    iree_task_t* next_task = task->next_task;
    iree_task_wait_t* wait_task = (iree_task_wait_t*)task;
    if (wake_handle.type == wait_task->wait_handle.type &&
        memcmp(&wake_handle.value, &wait_task->wait_handle.value,
               sizeof(wake_handle.value)) == 0) {
      // Found one of possibly many. If its condition is met then remove from
      // the wait set and ready up.
      if (iree_task_wait_check_condition(wait_task)) {
        iree_wait_set_erase(executor->wait_set, wake_handle);
        iree_task_list_erase(&executor->waiting_list, prev_task, task);
        iree_task_submission_enqueue(pending_submission, task);
        task = prev_task;
      }
    }
    prev_task = task;
    task = next_task;
  }
}

// Polls all waiting tasks to see if they have completed and adds any newly
// ready dependencies to |pending_submission|.
//
// Only called during coordination and expects the coordinator lock to be held.
static void iree_task_executor_poll_waiting_tasks(
    iree_task_executor_t* executor,
    iree_task_submission_t* pending_submission) {
  if (iree_task_list_is_empty(&executor->waiting_list)) return;

  // Hold the wait lock for the duration we use the wait_set.
  if (!iree_slim_mutex_try_lock(&executor->wait_mutex)) {
    return;
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Poll all root waiting tasks (infinite-past duration) to see if any have
  // completed. If one or more have resolved then wake_handle will contain an
  // unspecified wake handle.
  int woken_tasks = 0;
  do {
    iree_wait_handle_t wake_handle;
    iree_status_t status = iree_wait_any(executor->wait_set,
                                         IREE_TIME_INFINITE_PAST, &wake_handle);
    if (iree_status_is_ok(status)) {
      // One or more waiters is ready. We don't support multi-wake right now so
      // we'll just take the one we got back and try again.
      iree_task_executor_wake_waiting_task(executor, wake_handle,
                                           pending_submission);
      ++woken_tasks;
      continue;
    } else if (iree_status_is_deadline_exceeded(status)) {
      // Indicates nothing was woken. Gracefully bail for now.
      break;
    } else {
      // (Spurious?) error during poll.
      // TODO(#4026): propagate failure to all scopes involved.
      // It may be ok to ignore when polling as the eventual wait will handle
      // the full propagation. For now we assert so its easy to see if we have
      // tried to perform a bad iree_wait_any.
      IREE_ASSERT_TRUE(iree_status_is_ok(status));
      iree_status_ignore(status);
      break;
    }
  } while (!iree_task_list_is_empty(&executor->waiting_list));

  iree_slim_mutex_unlock(&executor->wait_mutex);

  IREE_TRACE_ZONE_APPEND_VALUE(z0, woken_tasks);
  IREE_TRACE_ZONE_END(z0);
}

// Waits for one or more waiting tasks to be ready to execute.
// If a wait task retires any newly-ready tasks will be added to
// |pending_submission|.
//
// Only called during coordination and expects the coordinator lock to be held.
static void iree_task_executor_wait_any_task(
    iree_task_executor_t* executor, iree_task_worker_t* current_worker,
    iree_task_submission_t* pending_submission) {
  if (iree_task_list_is_empty(&executor->waiting_list)) return;

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_unlock(&executor->coordinator_mutex);

  // We can't hold the coordinator lock during the wait but also need to ensure
  // no other coordination messes with the wait set. We have a dedicated wait
  // mutex and guard wait-set accesses (polling/waiting/etc) with that. Polls
  // may try-lock and bail if the lock is held indicating that someone else has
  // a non-polling wait active.

  // TODO(benvanik): ensure coordinator wake semantics are modeled:
  // - donator:
  //   attempt 0:
  //     try steal
  //     if fail to steal: coordinate
  //   attempt 1:
  //     try steal
  //     if fail to steal: await any-posted notification?
  // - worker:
  //   attempt 0:
  //     try steal
  //     if fail to steal: coordinate

  iree_slim_mutex_lock(&executor->wait_mutex);

  iree_time_t deadline_ns = IREE_TIME_INFINITE_FUTURE;
  iree_wait_handle_t wake_handle;
  iree_status_t status =
      iree_wait_any(executor->wait_set, deadline_ns, &wake_handle);

  iree_slim_mutex_unlock(&executor->wait_mutex);

  // TODO(#4026): propagate failure to all scopes involved.
  IREE_ASSERT_TRUE(iree_status_is_ok(status));
  iree_status_ignore(status);

  iree_slim_mutex_lock(&executor->coordinator_mutex);

  int woken_tasks = 0;
  if (iree_status_is_ok(status)) {
    // One or more waiters is ready. We don't support multi-wake right now so
    // we'll just take the one we got back and try again.
    iree_task_executor_wake_waiting_task(executor, wake_handle,
                                         pending_submission);
    ++woken_tasks;
  } else if (iree_status_is_deadline_exceeded(status)) {
    // Indicates nothing was woken. Gracefully bail and return to the
    // coordinator to see if we should wait again.
  } else {
    // (Spurious?) error during wait.
    // TODO(#4026): propagate failure to all scopes involved.
    // Failures during waits are serious: ignoring them could lead to live-lock
    // as tasks further in the pipeline expect them to have completed or - even
    // worse - user code/other processes/drivers/etc may expect them to
    // complete.
    IREE_ASSERT_TRUE(iree_status_is_ok(status));
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_APPEND_VALUE(z0, woken_tasks);
  IREE_TRACE_ZONE_END(z0);
}

// Dispatches tasks in the global submission queue to workers.
// This is called by users upon submission of new tasks or by workers when they
// run out of tasks to process. |wait_on_idle| indicates whether the
// coordination request is done as a fallback in the event of there possibly
// being new work available.
//
// If a coordination run ends up with no ready tasks and one or more waiting
// tasks then the coordinator will wait for one of the tasks to become ready.
// This only happens in the |wait_on_idle| case (so it's always a worker) as in
// those cases the next step for the worker would have been to wait anyway. In
// the non-speculative case the coordinator polls the wait handles to see if
// they have resolved instead, possibly readying more tasks immediately.
void iree_task_executor_coordinate(iree_task_executor_t* executor,
                                   iree_task_worker_t* current_worker,
                                   bool wait_on_idle) {
  iree_slim_mutex_lock(&executor->coordinator_mutex);
  IREE_TRACE_ZONE_BEGIN(z0);

  // We may be adding tasks/waiting/etc on each pass through coordination - to
  // ensure we completely drain the incoming queues and satisfied waits we loop
  // until there's nothing left to coordinate.
  bool schedule_dirty = true;
  do {
    // Check for incoming submissions and move their posted tasks into our
    // local lists. Any of the tasks here are ready to execute immediately and
    // ones we should be able to distribute to workers without delay. The
    // waiting tasks are to the best of the caller's knowledge not ready yet.
    //
    // Note that we only do this once per coordination; that's so we don't
    // starve if submissions come in faster than we can schedule them.
    // Coordination will run again when workers become idle and will pick up
    // any changes then.
    //
    // As we schedule tasks we may spawn new ones (like a dispatch -> many
    // dispatch slices) and we keep track of those here. By doing a pass through
    // all ready tasks and only then merging in the new submission we get
    // breadth-first traversal of task graphs even if they originate from
    // various places and have no relation - hopefully leading to better average
    // latency.
    iree_task_submission_t pending_submission;
    iree_task_submission_initialize_from_lifo_slist(
        &executor->incoming_ready_slist, &pending_submission);
    iree_task_list_append_from_fifo_slist(&pending_submission.waiting_list,
                                          &executor->incoming_waiting_slist);

    // Scratch coordinator submission batch used during scheduling to batch up
    // all tasks that will be posted to each worker. We could stash this on the
    // executor but given that which thread is playing the role of the
    // coordinator is random it's better to ensure that these bytes never incur
    // a cache miss by making them live here in the stack of the chosen thread.
    iree_task_post_batch_t* post_batch =
        iree_alloca(sizeof(iree_task_post_batch_t) +
                    executor->worker_count * sizeof(iree_task_list_t));
    iree_task_post_batch_initialize(executor, current_worker, post_batch);

    // Poll the waiting tasks to see if any have resolved. This dramatically
    // cuts latency in cases where the wait handle completes prior to us
    // entering the real wait. When we have semaphores sequencing back-to-back
    // work this ensures that we pack in future dispatch work earlier vs.
    // waiting for a full thread hop.
    //
    // If any waits have resolved then they'll be moved to the ready list here
    // and then get processed FIFO with the tasks that were ready in the
    // request.
    iree_task_executor_poll_waiting_tasks(executor, &pending_submission);

    // Schedule all ready tasks in this batch. Some may complete inline (such
    // as ready barriers with all their dependencies resolved) while others may
    // be scheduled on workers via the post batch.
    iree_task_executor_schedule_ready_tasks(executor, &pending_submission,
                                            post_batch);

    // Merge any newly waiting tasks into the global wait list.
    iree_task_executor_merge_wait_list(executor,
                                       &pending_submission.waiting_list);

    // Post all new work to workers; they may wake and begin executing
    // immediately. Returns whether this worker has new tasks for it to work on.
    bool did_post = iree_task_post_batch_submit(post_batch);
    if (!did_post && wait_on_idle) {
      // No work was found; wait on one or more of our wait handles.
      // This will block the calling thread but that's fine as they were going
      // to wait anyway and were just speculatively seeing if there was work
      // first by requesting coordination. If work completes here we'll catch it
      // on the poll next loop around.
      iree_task_executor_wait_any_task(executor, current_worker,
                                       &pending_submission);
    }

    // Merge any new work into the submission list for future coordinators to
    // deal with - we don't want the possibility of starvation by looping on
    // this.
    if (!iree_task_submission_is_empty(&pending_submission)) {
      iree_task_executor_merge_submission(executor, &pending_submission);
      schedule_dirty = true;
    } else {
      schedule_dirty = false;
    }
  } while (schedule_dirty);

  iree_slim_mutex_unlock(&executor->coordinator_mutex);
  IREE_TRACE_ZONE_END(z0);
}

static iree_task_t* iree_task_executor_try_steal_task_from_affinity_set(
    iree_task_executor_t* executor, iree_task_affinity_set_t victim_mask,
    uint32_t max_theft_attempts, int rotation_offset,
    iree_task_queue_t* local_task_queue) {
  if (!victim_mask) return NULL;
  max_theft_attempts = iree_min(max_theft_attempts,
                                iree_task_affinity_set_count_ones(victim_mask));
  victim_mask = iree_task_affinity_set_rotr(victim_mask, rotation_offset);

  int worker_index = rotation_offset;
  iree_task_affinity_set_t mask =
      iree_task_affinity_set_rotr(victim_mask, worker_index);
  for (uint32_t i = 0; i < max_theft_attempts; ++i) {
    // Find the last set bit and skip to it. This avoids the need for doing
    // a full O(n) scan and instead gets us at O(popcnt) * O(ctz).
    //
    // Example: sharing mask = 0b01010101
    //          mask_rotation = 3 (randomly selected)
    //          mask = 0b01010101 rotr 3 = 0b10101010
    //          for (i = 0; i < 4; ++i)
    //            offset = ctz(0b10101010) = 1
    //            mask_rotation += 1 = 4
    //            mask >>= 1 = 0b01010101
    //            victim_index = 4 % 64 = 4
    int offset = iree_task_affinity_set_count_trailing_zeros(mask);
    int victim_index = (worker_index + offset) % executor->worker_count;
    worker_index += offset + 1;
    mask = iree_shr(mask, offset + 1);
    iree_task_worker_t* victim_worker = &executor->workers[victim_index];

    // Policy: steal a chunk of tasks at the tail of the victim queue.
    // This will steal multiple tasks from the victim up to the specified max
    // and move the them into our local task queue. Not all tasks will be stolen
    // and the assumption is that over a large-enough random distribution of
    // thievery taking ~half of the tasks each time (across all queues) will
    // lead to a relatively even distribution.
    iree_task_t* task = iree_task_worker_try_steal_task(
        victim_worker, local_task_queue,
        /*max_tasks=*/IREE_TASK_EXECUTOR_MAX_THEFT_TASK_COUNT);
    if (task) return task;
  }

  // No tasks found in victim_mask.
  return NULL;
}

// Tries to steal an entire task from a sibling worker (based on topology).
// Returns a task that is available (has not yet begun processing at all).
// May steal multiple tasks and add them to the |local_task_queue|.
//
// We do a scan through ideal victims indicated by the
// |constructive_sharing_mask|; these are the workers most likely to have some
// cache benefits to taking their work as they share some level of the cache
// hierarchy and should be better to steal from than any random worker.
//
// To prevent biasing any particular victim we use a fast prng function to
// select where in the set of potential victims defined by the topology
// group we steal. We (probably) don't need anything super complex here so
// instead of bouncing around at random we just select the starting point in
// our search and then go in-order.
iree_task_t* iree_task_executor_try_steal_task(
    iree_task_executor_t* executor,
    iree_task_affinity_set_t constructive_sharing_mask,
    uint32_t max_theft_attempts, iree_prng_minilcg128_state_t* theft_prng,
    iree_task_queue_t* local_task_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Limit the workers we will steal from to the ones that are currently live
  // and not idle.
  iree_task_affinity_set_t victim_mask =
      iree_atomic_task_affinity_set_load(&executor->worker_live_mask,
                                         iree_memory_order_relaxed) &
      ~iree_atomic_task_affinity_set_load(&executor->worker_idle_mask,
                                          iree_memory_order_relaxed);

  // TODO(benvanik): it may be possible to rework this such that we better
  // use the prng; for example, instead of all this rotating stuff we could just
  // generate an 8-bit number (or even split it into two 4-bit numbers) per
  // theft attempt. The current rotation strategy is biased toward the same try
  // ordering vs. what we may really want with an unbiased random selection.
  int rotation_offset = iree_prng_minilcg128_next_uint8(theft_prng) &
                        (8 * sizeof(iree_task_affinity_set_t) - 1);

  // Try first with the workers we may have some caches shared with. This
  // helps to prevent cache invalidations/availability updates as it's likely
  // that we won't need to go back to main memory (or higher cache tiers) in the
  // event that the thief and victim are running close to each other in time.
  iree_task_t* task = iree_task_executor_try_steal_task_from_affinity_set(
      executor, victim_mask & constructive_sharing_mask, max_theft_attempts,
      rotation_offset, local_task_queue);
  if (task) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "local");
  } else {
    task = iree_task_executor_try_steal_task_from_affinity_set(
        executor, victim_mask & ~constructive_sharing_mask, max_theft_attempts,
        rotation_offset, local_task_queue);
    if (task) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "non-local");
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return task;
}

iree_status_t iree_task_executor_donate_caller(iree_task_executor_t* executor,
                                               iree_wait_handle_t* wait_handle,
                                               iree_time_t deadline_ns) {
  // Not implemented; just wait. Unclear we want this yet and it's complex.
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_wait_one(wait_handle, deadline_ns);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
