// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/executor.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/internal/debugging.h"
#include "iree/base/internal/math.h"
#include "iree/task/affinity_set.h"
#include "iree/task/executor_impl.h"
#include "iree/task/list.h"
#include "iree/task/pool.h"
#include "iree/task/post_batch.h"
#include "iree/task/queue.h"
#include "iree/task/task_impl.h"
#include "iree/task/tuning.h"
#include "iree/task/worker.h"

static void iree_task_executor_destroy(iree_task_executor_t* executor);

void iree_task_executor_options_initialize(
    iree_task_executor_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
}

// Returns the size of the worker local memory required by |group| in bytes.
// We don't want destructive sharing between workers so ensure we are aligned to
// at least the destructive interference size, even if a bit larger than what
// the user asked for or the device supports.
static iree_host_size_t iree_task_topology_group_local_memory_size(
    iree_task_executor_options_t options,
    const iree_task_topology_group_t* group) {
  iree_host_size_t worker_local_memory_size = options.worker_local_memory_size;
  if (!worker_local_memory_size) {
    worker_local_memory_size = group->caches.l2_data;
  }
  if (!worker_local_memory_size) {
    worker_local_memory_size = group->caches.l1_data;
  }
  return iree_host_align(worker_local_memory_size,
                         iree_hardware_destructive_interference_size);
}

iree_status_t iree_task_executor_create(iree_task_executor_options_t options,
                                        const iree_task_topology_t* topology,
                                        iree_allocator_t allocator,
                                        iree_task_executor_t** out_executor) {
  iree_host_size_t worker_count = iree_task_topology_group_count(topology);
  if (worker_count > IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "requested %" PRIhsz
                            " workers but a maximum of %d is allowed",
                            worker_count, IREE_TASK_EXECUTOR_MAX_WORKER_COUNT);
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

  // The executor is followed in memory by worker[] + worker_local_memory[].
  iree_host_size_t total_worker_local_memory_size = 0;
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    total_worker_local_memory_size +=
        iree_task_topology_group_local_memory_size(
            options, iree_task_topology_get_group(topology, i));
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)total_worker_local_memory_size);

  iree_host_size_t executor_base_size =
      iree_host_align(sizeof(iree_task_executor_t),
                      iree_hardware_destructive_interference_size);
  iree_host_size_t worker_list_size =
      iree_host_align(worker_count * sizeof(iree_task_worker_t),
                      iree_hardware_destructive_interference_size);
  iree_host_size_t executor_size =
      executor_base_size + worker_list_size + total_worker_local_memory_size;

  iree_task_executor_t* executor = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, executor_size, (void**)&executor));
  memset(executor, 0, executor_size);
  iree_atomic_ref_count_init(&executor->ref_count);
  executor->allocator = allocator;
  executor->scheduling_mode = options.scheduling_mode;
  executor->worker_spin_ns = options.worker_spin_ns;
  iree_atomic_task_slist_initialize(&executor->incoming_ready_slist);
  iree_slim_mutex_initialize(&executor->coordinator_mutex);

  IREE_TRACE({
    static iree_atomic_int32_t executor_id = IREE_ATOMIC_VAR_INIT(0);
    char trace_name[32];
    int trace_name_length =
        snprintf(trace_name, sizeof(trace_name), "iree-executor-%d",
                 iree_atomic_fetch_add_int32(&executor_id, 1,
                                             iree_memory_order_seq_cst));
    IREE_LEAK_CHECK_DISABLE_PUSH();
    executor->trace_name = malloc(trace_name_length + 1);
    memcpy((void*)executor->trace_name, trace_name, trace_name_length + 1);
    IREE_LEAK_CHECK_DISABLE_POP();
    IREE_TRACE_SET_PLOT_TYPE(executor->trace_name,
                             IREE_TRACING_PLOT_TYPE_PERCENTAGE, /*step=*/true,
                             /*fill=*/true, /*color=*/0xFF1F883Du);
    IREE_TRACE_PLOT_VALUE_F32(executor->trace_name, 0.0f);
  });

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

  // Pool used for system events; exposed to users of the task system to ensure
  // we minimize the number of live events and reduce overheads in
  // high-frequency transient parking operations.
  if (iree_status_is_ok(status)) {
    status = iree_event_pool_allocate(IREE_TASK_EXECUTOR_EVENT_POOL_CAPACITY,
                                      allocator, &executor->event_pool);
  }

  // Pool used for all fanout tasks. These only live within the executor and
  // since we know the precise lifetime of them we can keep them entirely within
  // the system here.
  if (iree_status_is_ok(status)) {
    status = iree_task_pool_initialize(
        allocator,
        iree_max(sizeof(iree_task_fence_t), sizeof(iree_task_dispatch_shard_t)),
        worker_count * IREE_TASK_EXECUTOR_INITIAL_SHARD_RESERVATION_PER_WORKER,
        &executor->transient_task_pool);
  }

  // Wait handling polling and waiting use a dedicated thread to ensure that
  // blocking syscalls stay off the workers.
  if (iree_status_is_ok(status)) {
    // For now we allow the poller to run anywhere - we should allow callers to
    // specify it via the topology (or something).
    iree_thread_affinity_t poller_thread_affinity;
    iree_thread_affinity_set_any(&poller_thread_affinity);
    status = iree_task_poller_initialize(executor, poller_thread_affinity,
                                         &executor->poller);
  }

  // Bring up the workers; the threads will be created here but be suspended
  // (if the platform supports it) awaiting the first tasks getting scheduled.
  if (iree_status_is_ok(status)) {
    executor->worker_base_index = options.worker_base_index;
    executor->worker_count = worker_count;
    executor->workers =
        (iree_task_worker_t*)((uint8_t*)executor + executor_base_size);
    uint8_t* worker_local_memory =
        (uint8_t*)executor->workers + worker_list_size;

    iree_task_affinity_set_t worker_mask =
        iree_task_affinity_set_ones(worker_count);

    for (iree_host_size_t i = 0; i < worker_count; ++i) {
      const iree_task_topology_group_t* group =
          iree_task_topology_get_group(topology, i);
      iree_host_size_t worker_local_memory_size =
          iree_task_topology_group_local_memory_size(options, group);
      iree_task_worker_t* worker = &executor->workers[i];
      status = iree_task_worker_initialize(
          executor, i, group, options.worker_stack_size,
          iree_make_byte_span(worker_local_memory, worker_local_memory_size),
          &seed_prng, worker);
      worker_local_memory += worker_local_memory_size;
      if (!iree_status_is_ok(status)) break;
    }

    iree_atomic_task_affinity_set_store(&executor->worker_idle_mask,
                                        worker_mask, iree_memory_order_release);
    iree_atomic_task_affinity_set_store(&executor->worker_live_mask,
                                        worker_mask, iree_memory_order_release);
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

  // Also ask the poller to exit - it'll wake from any system waits it's in and
  // abort all the remaining waits.
  iree_task_poller_request_exit(&executor->poller);

  // Now that all workers and the poller should be in the process of exiting we
  // can join with them. Some may take longer than others to exit but that's
  // fine as we can't return from here until they exit anyway.
  for (iree_host_size_t i = 0; i < executor->worker_count; ++i) {
    iree_task_worker_t* worker = &executor->workers[i];
    iree_task_worker_await_exit(worker);
  }
  iree_task_poller_await_exit(&executor->poller);

  // Tear down all workers and the poller now that no more threads are live.
  // Any live threads may still be touching their own data structures or those
  // of others (for example when trying to steal work).
  for (iree_host_size_t i = 0; i < executor->worker_count; ++i) {
    iree_task_worker_t* worker = &executor->workers[i];
    iree_task_worker_deinitialize(worker);
  }
  iree_task_poller_deinitialize(&executor->poller);

  iree_event_pool_free(executor->event_pool);
  iree_slim_mutex_deinitialize(&executor->coordinator_mutex);
  iree_atomic_task_slist_deinitialize(&executor->incoming_ready_slist);
  iree_task_pool_deinitialize(&executor->transient_task_pool);
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

void iree_task_executor_trim(iree_task_executor_t* executor) {
  // TODO(benvanik): figure out a good way to do this; the pools require that
  // no tasks are in-flight to trim but our caller can't reliably make that
  // guarantee. We'd need some global executor lock that we did here and
  // on submit - or rework pools to not have this limitation.
  // iree_task_pool_trim(&executor->fence_task_pool);
  // iree_task_pool_trim(&executor->transient_task_pool);
}

iree_host_size_t iree_task_executor_worker_count(
    iree_task_executor_t* executor) {
  return executor->worker_count;
}

iree_event_pool_t* iree_task_executor_event_pool(
    iree_task_executor_t* executor) {
  return executor->event_pool;
}

iree_status_t iree_task_executor_acquire_fence(iree_task_executor_t* executor,
                                               iree_task_scope_t* scope,
                                               iree_task_fence_t** out_fence) {
  *out_fence = NULL;

  iree_task_fence_t* fence = NULL;
  IREE_RETURN_IF_ERROR(iree_task_pool_acquire(&executor->transient_task_pool,
                                              (iree_task_t**)&fence));
  iree_task_fence_initialize(scope, iree_wait_primitive_immediate(), fence);
  fence->header.pool = &executor->transient_task_pool;

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
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_task_t* task = NULL;
  while ((task = iree_task_list_pop_front(&pending_submission->ready_list))) {
    // If the scope has been marked as failing then we abort the task.
    // This needs to happen as a poll here because one or more of the tasks we
    // are joining may have failed.
    if (IREE_UNLIKELY(!task->scope ||
                      iree_task_scope_has_failed(task->scope))) {
      iree_task_list_t discard_worklist;
      iree_task_list_initialize(&discard_worklist);
      iree_task_discard(task, &discard_worklist);
      iree_task_list_discard(&discard_worklist);
      continue;
    }

    switch (task->type) {
      case IREE_TASK_TYPE_NOP:
        // Doesn't do anything; just retire and continue on to any dependents.
        iree_task_nop_retire((iree_task_nop_t*)task, pending_submission);
        break;
      case IREE_TASK_TYPE_CALL: {
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
        // We should only ever see completed waits here; ones that have yet to
        // resolve are sent to the poller.
        iree_task_wait_retire(
            (iree_task_wait_t*)task, pending_submission,
            iree_all_bits_set(task->flags, IREE_TASK_FLAG_WAIT_COMPLETED)
                ? iree_ok_status()
                : iree_make_status(IREE_STATUS_INTERNAL,
                                   "unresolved wait task ended up in the "
                                   "executor run queue"));
        break;
      }
      case IREE_TASK_TYPE_DISPATCH: {
        // Dispatches may need to be issued (fanning out the tiles to workers)
        // or retired (after all tiles have completed).
        if (task->flags & IREE_TASK_FLAG_DISPATCH_RETIRE) {
          iree_task_dispatch_retire((iree_task_dispatch_t*)task,
                                    pending_submission);
        } else {
          iree_task_dispatch_issue((iree_task_dispatch_t*)task,
                                   &executor->transient_task_pool,
                                   pending_submission, post_batch);
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

  // Enqueue waiting tasks with the poller immediately: this may issue a
  // syscall to kick the poller. If we see bad context switches here then we
  // should split this into an enqueue/flush pair.
  iree_task_poller_enqueue(&executor->poller, &submission->waiting_list);

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
  // are pushed to workers. This will not wait - but may block.
  iree_task_executor_coordinate(executor, /*current_worker=*/NULL);

  IREE_TRACE_ZONE_END(z0);
}

// Dispatches tasks in the global submission queue to workers.
// This is called by users upon submission of new tasks or by workers when they
// run out of tasks to process. If |current_worker| is provided then tasks will
// prefer to be routed back to it for immediate processing.
//
// If a coordination run ends up with no ready tasks and |current_worker| is
// provided the calling thread will enter a wait until the worker has more tasks
// posted to it.
void iree_task_executor_coordinate(iree_task_executor_t* executor,
                                   iree_task_worker_t* current_worker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // We may be adding tasks/waiting/etc on each pass through coordination - to
  // ensure we completely drain the incoming queues and satisfied waits we loop
  // until there's nothing left to coordinate.
  bool schedule_dirty = true;
  do {
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "iree_task_executor_coordinate_try");
    // TODO(#10212): remove this lock or do something more clever to avoid
    // contention when many workers try to coordinate at the same time. This can
    // create very long serialized lock chains that slow down worker wakes.
    iree_slim_mutex_lock(&executor->coordinator_mutex);

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
    // dispatch shards) and we keep track of those here. By doing a pass through
    // all ready tasks and only then merging in the new submission we get
    // breadth-first traversal of task graphs even if they originate from
    // various places and have no relation - hopefully leading to better average
    // latency.
    iree_task_submission_t pending_submission;
    iree_task_submission_initialize_from_lifo_slist(
        &executor->incoming_ready_slist, &pending_submission);
    if (iree_task_list_is_empty(&pending_submission.ready_list)) {
      iree_slim_mutex_unlock(&executor->coordinator_mutex);
      IREE_TRACE_ZONE_END(z1);
      break;
    }

    // Scratch coordinator submission batch used during scheduling to batch up
    // all tasks that will be posted to each worker. We could stash this on the
    // executor but given that which thread is playing the role of the
    // coordinator is random it's better to ensure that these bytes never incur
    // a cache miss by making them live here in the stack of the chosen thread.
    iree_task_post_batch_t* post_batch =
        iree_alloca(sizeof(iree_task_post_batch_t) +
                    executor->worker_count * sizeof(iree_task_list_t));
    iree_task_post_batch_initialize(executor, current_worker, post_batch);

    // Schedule all ready tasks in this batch. Some may complete inline (such
    // as ready barriers with all their dependencies resolved) while others may
    // be scheduled on workers via the post batch.
    iree_task_executor_schedule_ready_tasks(executor, &pending_submission,
                                            post_batch);

    // Route waiting tasks to the poller.
    iree_task_poller_enqueue(&executor->poller,
                             &pending_submission.waiting_list);

    iree_slim_mutex_unlock(&executor->coordinator_mutex);
    IREE_TRACE_ZONE_END(z1);

    // Post all new work to workers; they may wake and begin executing
    // immediately. Returns whether this worker has new tasks for it to work on.
    schedule_dirty = iree_task_post_batch_submit(post_batch);
  } while (schedule_dirty);

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
    if (iree_atomic_load_int32(&victim_worker->state,
                               iree_memory_order_acquire) !=
        IREE_TASK_WORKER_STATE_RUNNING) {
      return NULL;
    }

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

  // The masks are accessed with 'relaxed' order because they are just hints.
  iree_task_affinity_set_t worker_live_mask =
      iree_atomic_task_affinity_set_load(&executor->worker_live_mask,
                                         iree_memory_order_relaxed);
  iree_task_affinity_set_t worker_idle_mask =
      iree_atomic_task_affinity_set_load(&executor->worker_idle_mask,
                                         iree_memory_order_relaxed);
  // Limit the workers we will steal from to the ones that are currently live
  // and not idle.
  iree_task_affinity_set_t victim_mask = worker_live_mask & ~worker_idle_mask;

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
                                               iree_wait_source_t wait_source,
                                               iree_timeout_t timeout) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Perform an immediate flush/coordination (in case the caller queued).
  iree_task_executor_flush(executor);

  // Wait until completed.
  // TODO(benvanik): make this steal tasks until wait_handle resolves?
  // Somewhat dangerous as we don't know what kind of thread we are running on;
  // it may have a smaller stack than we are expecting or have some weird thread
  // local state (FPU rounding modes/etc).
  iree_status_t status = iree_wait_source_wait_one(wait_source, timeout);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
