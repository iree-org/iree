// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/poller.h"

#include "iree/base/tracing.h"
#include "iree/task/executor.h"
#include "iree/task/executor_impl.h"
#include "iree/task/submission.h"
#include "iree/task/task_impl.h"
#include "iree/task/tuning.h"

static int iree_task_poller_main(iree_task_poller_t* poller);

iree_status_t iree_task_poller_initialize(
    iree_task_executor_t* executor,
    iree_thread_affinity_t ideal_thread_affinity,
    iree_task_poller_t* out_poller) {
  IREE_TRACE_ZONE_BEGIN(z0);

  out_poller->executor = executor;
  out_poller->ideal_thread_affinity = ideal_thread_affinity;
  iree_notification_initialize(&out_poller->state_notification);
  iree_atomic_task_slist_initialize(&out_poller->mailbox_slist);
  iree_task_list_initialize(&out_poller->wait_list);

  iree_task_poller_state_t initial_state = IREE_TASK_POLLER_STATE_RUNNING;
  // TODO(benvanik): support initially suspended wait threads. This can reduce
  // startup time as we won't give the system a chance to deschedule the calling
  // thread as it performs the initial resume of the wait thread. We'll need to
  // check in enqueue to see if the wait thread needs to be resumed.
  // initial_state = IREE_TASK_POLLER_STATE_SUSPENDED;
  iree_atomic_store_int32(&out_poller->state, initial_state,
                          iree_memory_order_release);

  // Acquire an event we can use to wake the wait thread from other threads.
  iree_status_t status = iree_event_pool_acquire(
      iree_task_executor_event_pool(out_poller->executor), 1,
      &out_poller->wake_event);

  // Wait set used to batch syscalls for polling/waiting on wait handles.
  // This is currently limited to a relatively small max to make bad behavior
  // clearer with nice RESOURCE_EXHAUSTED errors. If we start to hit that limit
  // (~63+ simultaneous system waits) we'll need to shard out the wait sets -
  // possibly with multiple wait threads (one per set).
  if (iree_status_is_ok(status)) {
    status = iree_wait_set_allocate(IREE_TASK_EXECUTOR_MAX_OUTSTANDING_WAITS,
                                    executor->allocator, &out_poller->wait_set);
  }
  if (iree_status_is_ok(status)) {
    status = iree_wait_set_insert(out_poller->wait_set, out_poller->wake_event);
  }

  iree_thread_create_params_t thread_params = {0};
  thread_params.name = iree_make_cstring_view("iree-poller");
  thread_params.create_suspended = false;
  // TODO(benvanik): make high so to reduce latency? The sooner we wake the
  // sooner we get ready tasks back in the execution queue, though we don't
  // want to preempt any of the workers.
  thread_params.priority_class = IREE_THREAD_PRIORITY_CLASS_NORMAL;
  thread_params.initial_affinity = out_poller->ideal_thread_affinity;

  // NOTE: if the thread creation fails we'll bail here and let the caller
  // cleanup by calling deinitialize (which is safe because we zero init
  // everything).
  if (iree_status_is_ok(status)) {
    status = iree_thread_create((iree_thread_entry_t)iree_task_poller_main,
                                out_poller, thread_params, executor->allocator,
                                &out_poller->thread);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_task_poller_request_exit(iree_task_poller_t* poller) {
  if (!poller->thread) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // If the thread is already in the exiting/zombie state we don't need to do
  // anything.
  iree_task_poller_state_t prev_state =
      (iree_task_poller_state_t)iree_atomic_exchange_int32(
          &poller->state, IREE_TASK_POLLER_STATE_EXITING,
          iree_memory_order_acq_rel);
  switch (prev_state) {
    case IREE_TASK_POLLER_STATE_SUSPENDED:
      // Poller was suspended; resume it so that it can exit itself.
      iree_thread_resume(poller->thread);
      break;
    case IREE_TASK_POLLER_STATE_ZOMBIE:
      // Poller already exited; reset state to ZOMBIE.
      iree_atomic_store_int32(&poller->state, IREE_TASK_POLLER_STATE_ZOMBIE,
                              iree_memory_order_release);
      break;
    default:
      // Poller now set to EXITING and should exit soon.
      break;
  }

  // Kick the wait thread to exit the system wait API, if needed.
  // It'll check the state and abort ASAP.
  iree_event_set(&poller->wake_event);

  IREE_TRACE_ZONE_END(z0);
}

// Returns true if the wait thread is in the zombie state (exited and awaiting
// teardown).
static bool iree_task_poller_is_zombie(iree_task_poller_t* poller) {
  return iree_atomic_load_int32(&poller->state, iree_memory_order_acquire) ==
         IREE_TASK_POLLER_STATE_ZOMBIE;
}

void iree_task_poller_await_exit(iree_task_poller_t* poller) {
  if (!poller->thread) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_task_poller_request_exit(poller);
  iree_notification_await(&poller->state_notification,
                          (iree_condition_fn_t)iree_task_poller_is_zombie,
                          poller, iree_infinite_timeout());

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_poller_deinitialize(iree_task_poller_t* poller) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Must have called request_exit/await_exit.
  IREE_ASSERT_TRUE(iree_task_poller_is_zombie(poller));

  iree_thread_release(poller->thread);
  poller->thread = NULL;

  iree_wait_set_free(poller->wait_set);
  if (!iree_wait_handle_is_immediate(poller->wake_event)) {
    iree_event_pool_release(iree_task_executor_event_pool(poller->executor), 1,
                            &poller->wake_event);
  }

  iree_task_list_discard(&poller->wait_list);
  iree_atomic_task_slist_discard(&poller->mailbox_slist);
  iree_atomic_task_slist_deinitialize(&poller->mailbox_slist);
  iree_notification_deinitialize(&poller->state_notification);

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_poller_enqueue(iree_task_poller_t* poller,
                              iree_task_list_t* wait_tasks) {
  if (iree_task_list_is_empty(wait_tasks)) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Move the list into the mailbox. Note that the mailbox is LIFO and this list
  // is concatenated with its current order preserved (which should be LIFO),
  // though we don't really care about order here.
  iree_atomic_task_slist_concat(&poller->mailbox_slist, wait_tasks->head,
                                wait_tasks->tail);
  memset(wait_tasks, 0, sizeof(*wait_tasks));

  // Kick the wait thread to exit the system wait API, if needed.
  // It'll merge the new wait tasks and reset the event.
  iree_event_set(&poller->wake_event);

  IREE_TRACE_ZONE_END(z0);
}

// Acquires a wait handle for |task| and inserts it into |wait_set|.
static iree_status_t iree_task_poller_insert_wait_handle(
    iree_wait_set_t* wait_set, iree_task_wait_t* task) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  iree_wait_handle_t wait_handle = iree_wait_handle_immediate();
  iree_wait_handle_t* wait_handle_ptr =
      iree_wait_handle_from_source(&task->wait_source);
  if (wait_handle_ptr) {
    // Already a wait handle - can directly insert it.
    wait_handle = *wait_handle_ptr;
  } else {
    iree_wait_primitive_t wait_primitive = iree_wait_primitive_immediate();
    status =
        iree_wait_source_export(task->wait_source, IREE_WAIT_PRIMITIVE_TYPE_ANY,
                                iree_immediate_timeout(), &wait_primitive);
    if (iree_status_is_ok(status)) {
      // Swap the wait handle with the exported handle so we can wake it later.
      // It'd be ideal if we retained the wait handle separate so that we could
      // still do fast queries for local wait sources.
      iree_wait_handle_wrap_primitive(wait_primitive.type, wait_primitive.value,
                                      &wait_handle);
      status = iree_wait_source_import(wait_primitive, &task->wait_source);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_wait_set_insert(wait_set, wait_handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

enum iree_task_poller_prepare_result_bits_e {
  IREE_TASK_POLLER_PREPARE_OK = 0,
  IREE_TASK_POLLER_PREPARE_RETIRED = 1u << 0,
  IREE_TASK_POLLER_PREPARE_CANCELLED = 1u << 1,
};
typedef uint32_t iree_task_poller_prepare_result_t;

// Prepares a wait |task| for waiting.
// The task will be checked for completion or failure such as deadline exceeded
// and removed from the wait list if resolved. If unresolved the wait will be
// prepared for the system wait by ensuring a wait handle is available.
//
// When the task is retiring because it has been completed (or cancelled) the
// |out_retire_status| status will be set to the value callers must pass to
// iree_task_wait_retire.
static iree_task_poller_prepare_result_t iree_task_poller_prepare_task(
    iree_task_poller_t* poller, iree_task_wait_t* task,
    iree_task_submission_t* pending_submission, iree_time_t now_ns,
    iree_time_t* earliest_deadline_ns, iree_status_t* out_retire_status) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Status of the preparation - failures propagate to the task scope.
  iree_status_t status = iree_ok_status();
  // Wait status:
  //   OK: wait resolved successfully
  //   DEFERRED: wait unresolved
  //   DEADLINE_EXCEEDED: deadline was hit before the wait resolved
  //   CANCELLED: wait was cancelled via the cancellation flag
  iree_status_code_t wait_status_code = IREE_STATUS_DEFERRED;
  if (iree_all_bits_set(task->header.flags, IREE_TASK_FLAG_WAIT_COMPLETED)) {
    // Wait was marked as resolved and we just pass that through here.
    // This allows us to bypass more expensive queries when doing a post-wake
    // scan of tasks.
    wait_status_code = IREE_STATUS_OK;
  } else if (task->cancellation_flag != NULL &&
             iree_atomic_load_int32(task->cancellation_flag,
                                    iree_memory_order_acquire) != 0) {
    // Task was cancelled by the user (or a wait-any). These retire without
    // failure and it's up to the user to handle what happens to them.
    wait_status_code = IREE_STATUS_CANCELLED;
  } else if (iree_wait_source_is_immediate(task->wait_source)) {
    // Task has been neutered and is treated as an immediately resolved wait.
    wait_status_code = IREE_STATUS_OK;
  } else if (iree_wait_source_is_delay(task->wait_source)) {
    // Task is a delay until some future time; factor that in to our earliest
    // deadline so that we'll wait in the system until that time. If we wake
    // earlier because another wait resolved it's still possible for the delay
    // to have been reached before we get back to this check.
    iree_time_t delay_deadline_ns = (iree_time_t)task->wait_source.data;
    if (delay_deadline_ns <= now_ns + IREE_TASK_EXECUTOR_DELAY_SLOP_NS) {
      // Wait deadline reached.
      wait_status_code = IREE_STATUS_OK;
    } else {
      // Still waiting.
      *earliest_deadline_ns =
          iree_min(*earliest_deadline_ns, delay_deadline_ns);
      wait_status_code = IREE_STATUS_DEFERRED;
    }
  } else {
    // An actual wait. Ensure that the deadline has not been exceeded yet.
    // If it hasn't yet been hit we'll propagate the deadline to the system wait
    // API - then on the next pump we'll hit this case and retire the task.
    IREE_TRACE_ZONE_APPEND_VALUE(z0, task->deadline_ns);
    IREE_TRACE_ZONE_APPEND_VALUE(z0, now_ns);
    if (task->deadline_ns <= now_ns) {
      wait_status_code = IREE_STATUS_DEADLINE_EXCEEDED;
    } else {
      // Query the status of the wait source to see if it has already been
      // resolved. Under load we can get lucky and end up with resolved waits
      // before ever needing to export them for a full system wait. This query
      // can also avoid making a syscall to check the state of the source such
      // as when the source is a process-local type.
      wait_status_code = IREE_STATUS_OK;
      status = iree_wait_source_query(task->wait_source, &wait_status_code);

      // TODO(benvanik): avoid this query for wait handles: we don't want to
      // make one syscall per handle and could rely on the completed bit being
      // set to retire these.
    }

    // If the wait has not been resolved then we need to ensure there's an
    // exported wait handle in the wait set. We only do this on the first time
    // we prepare the task.
    if (wait_status_code == IREE_STATUS_DEFERRED) {
      if (!iree_all_bits_set(task->header.flags,
                             IREE_TASK_FLAG_WAIT_EXPORTED)) {
        task->header.flags |= IREE_TASK_FLAG_WAIT_EXPORTED;
        status = iree_task_poller_insert_wait_handle(poller->wait_set, task);
      }
      *earliest_deadline_ns =
          iree_min(*earliest_deadline_ns, task->deadline_ns);
    }
  }

  if (iree_status_is_ok(status) && wait_status_code == IREE_STATUS_DEFERRED) {
    // Wait is prepared for use and can be waited on.
    IREE_TRACE_ZONE_END(z0);
    return IREE_TASK_POLLER_PREPARE_OK;
  }

  // If the task was able to be retired (deadline elapsed, completed, etc)
  // then we need to unregister it from the poller and send it back to the
  // workers for completion.
  iree_task_poller_prepare_result_t result = IREE_TASK_POLLER_PREPARE_RETIRED;

  // If this was part of a wait-any operation then set the cancellation flag
  // such that other waits are cancelled.
  if (iree_any_bit_set(task->header.flags, IREE_TASK_FLAG_WAIT_ANY)) {
    if (iree_atomic_fetch_add_int32(task->cancellation_flag, 1,
                                    iree_memory_order_release) == 0) {
      // Ensure we scan again to clean up any potentially cancelled tasks.
      // If this was task 4 in a wait-any list then tasks 0-3 need to be
      // retired.
      result |= IREE_TASK_POLLER_PREPARE_CANCELLED;
    }
  }

  // Remove the system wait handle from the wait set, if assigned.
  if (iree_all_bits_set(task->header.flags, IREE_TASK_FLAG_WAIT_EXPORTED)) {
    iree_wait_handle_t* wait_handle =
        iree_wait_handle_from_source(&task->wait_source);
    if (wait_handle) {
      iree_wait_set_erase(poller->wait_set, *wait_handle);
    }
    task->header.flags &= ~IREE_TASK_FLAG_WAIT_EXPORTED;
  }

  // Retire the task and enqueue any available completion task.
  // Note that we pass in the status of the wait query above: that propagates
  // any query failure into the task/task scope.
  if (iree_status_is_ok(status) && wait_status_code != IREE_STATUS_OK) {
    // Cancellation is ok - we just ignore those.
    if (wait_status_code != IREE_STATUS_CANCELLED) {
      status = iree_status_from_code(wait_status_code);
    }
  }

  // The caller must make the iree_task_wait_retire call with this status.
  // If we were to do that here we'd be freeing the task that may still exist
  // in the caller's working set.
  *out_retire_status = status;

  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Scans all wait tasks in |poller| to see if they have resolved.
// Resolved/failed waits are enqueued on |pending_submission|.
// If there are any unresolved delay tasks the earliest deadline will be stored
// in |out_earliest_deadline_ns| and otherwise it'll be set to
// IREE_TIME_INFINITE_FUTURE.
static void iree_task_poller_prepare_wait(
    iree_task_poller_t* poller, iree_task_submission_t* pending_submission,
    iree_time_t* out_earliest_deadline_ns) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_earliest_deadline_ns = IREE_TIME_INFINITE_FUTURE;

  // TODO(benvanik): only query if there are pending delays; this is (likely) a
  // syscall that we only need to perform if we're going to delay.
  iree_time_t now_ns = iree_time_now();

  // Perform the scan over the task list; we may need to retry the scan if we
  // encounter a situation that would invalidate other waits - such as
  // cancellation or scope errors.
  bool retry_scan = false;
  do {
    retry_scan = false;

    // Note that we walk the singly-linked list inline and need to keep track of
    // the previous task in case we need to unlink one.
    iree_task_t* prev_task = NULL;
    iree_task_t* task = iree_task_list_front(&poller->wait_list);
    while (task != NULL) {
      iree_task_t* next_task = task->next_task;

      iree_status_t retire_status = iree_ok_status();
      iree_task_poller_prepare_result_t result = iree_task_poller_prepare_task(
          poller, (iree_task_wait_t*)task, pending_submission, now_ns,
          out_earliest_deadline_ns, &retire_status);
      if (iree_all_bits_set(result, IREE_TASK_POLLER_PREPARE_CANCELLED)) {
        // A task was cancelled; we'll need to retry the scan to clean up any
        // waits we may have already checked.
        retry_scan = true;
      }

      if (iree_all_bits_set(result, IREE_TASK_POLLER_PREPARE_RETIRED)) {
        // Erase the retired task from the wait list.
        iree_task_list_erase(&poller->wait_list, prev_task, task);
        iree_task_wait_retire((iree_task_wait_t*)task, pending_submission,
                              retire_status);
        task = NULL;  // task memory is now invalid
      } else {
        prev_task = task;
      }
      task = next_task;
    }
  } while (retry_scan);

  IREE_TRACE_ZONE_END(z0);
}

// Finds tasks in |poller| using the given wait handle and marks them as
// completed.
static void iree_task_poller_wake_task(iree_task_poller_t* poller,
                                       iree_wait_handle_t wake_handle) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): scan the list. We need a way to map wake_handle back to
  // the zero or more tasks that match it but don't currently store the
  // handle. Ideally we'd have the wait set tell us precisely which things
  // woke - possibly by having a bitmap of original insertions that match the
  // handle - but for now we just eat the extra query syscall.
  int woken_tasks = 0;

  (void)woken_tasks;
  IREE_TRACE_ZONE_APPEND_VALUE(z0, woken_tasks);
  IREE_TRACE_ZONE_END(z0);
}

// Commits a system wait on the current wait set in |poller|.
// The wait will time out after |deadline_ns| is reached and return even if no
// wait handles were resolved.
static void iree_task_poller_commit_wait(iree_task_poller_t* poller,
                                         iree_time_t deadline_ns) {
  if (iree_atomic_load_int32(&poller->state, iree_memory_order_acquire) ==
      IREE_TASK_POLLER_STATE_EXITING) {
    // Thread exit requested - don't block shutdown.
    return;
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Enter the system wait API.
  iree_wait_handle_t wake_handle = iree_wait_handle_immediate();
  iree_status_t status =
      iree_wait_any(poller->wait_set, deadline_ns, &wake_handle);
  if (iree_status_is_ok(status)) {
    // One or more waiters is ready. We don't support multi-wake right now so
    // we'll just take the one we got back and try again.
    //
    // To avoid extra syscalls we scan the list and mark whatever tasks were
    // using the handle the wait set reported waking as completed. On the next
    // scan they'll be retired immediately. Ideally we'd have the wait set be
    // able to tell us this precise list.
    if (iree_wait_handle_is_immediate(wake_handle)) {
      // No-op wait - ignore.
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "nop");
    } else if (wake_handle.type == poller->wake_event.type &&
               memcmp(&wake_handle.value, &poller->wake_event.value,
                      sizeof(wake_handle.value)) == 0) {
      // Woken on the wake_event used to exit the system wait early.
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "wake_event");
    } else {
      // Route to zero or more tasks using this handle.
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "task(s)");
      iree_task_poller_wake_task(poller, wake_handle);
    }
  } else if (iree_status_is_deadline_exceeded(status)) {
    // Indicates nothing was woken within the deadline. We gracefully bail here
    // and let the scan check for per-task deadline exceeded events or delay
    // completion.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "deadline exceeded");
  } else {
    // (Spurious?) error during wait.
    // TODO(#4026): propagate failure to all scopes involved.
    // Failures during waits are serious: ignoring them could lead to live-lock
    // as tasks further in the pipeline expect them to have completed or - even
    // worse - user code/other processes/drivers/etc may expect them to
    // complete.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "failure");
    IREE_ASSERT_TRUE(iree_status_is_ok(status));
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
}

// Pumps the |poller| until it is requested to exit.
static void iree_task_poller_pump_until_exit(iree_task_poller_t* poller) {
  while (true) {
    // Check state to see if we've been asked to exit.
    if (iree_atomic_load_int32(&poller->state, iree_memory_order_acquire) ==
        IREE_TASK_POLLER_STATE_EXITING) {
      // Thread exit requested - cancel pumping.
      break;
    }

    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_task_poller_pump");

    // Reset the wake event and merge any incoming tasks to the wait list.
    // To avoid races we reset and then merge: this allows another thread
    // coming in and enqueuing tasks to set the event and ensure that we'll
    // get the tasks as we'll fall through on the wait below and loop again.
    iree_event_reset(&poller->wake_event);
    iree_task_list_append_from_fifo_slist(&poller->wait_list,
                                          &poller->mailbox_slist);

    // Scan all wait tasks to see if any have resolved and if so we'll enqueue
    // their retirement on the executor and drop them from the list.
    iree_task_submission_t pending_submission;
    iree_task_submission_initialize(&pending_submission);
    iree_time_t earliest_deadline_ns = IREE_TIME_INFINITE_FUTURE;
    iree_task_poller_prepare_wait(poller, &pending_submission,
                                  &earliest_deadline_ns);
    if (!iree_task_submission_is_empty(&pending_submission)) {
      iree_task_executor_submit(poller->executor, &pending_submission);
      iree_task_executor_flush(poller->executor);
    }

    // Enter the system multi-wait API.
    // We unconditionally do this: if we have nothing to wait on we'll still
    // wait on the wake_event for new waits to be enqueued - or the first delay
    // to be reached.
    iree_task_poller_commit_wait(poller, earliest_deadline_ns);

    IREE_TRACE_ZONE_END(z0);
  }
}

// Thread entry point for the poller wait thread.
static int iree_task_poller_main(iree_task_poller_t* poller) {
  IREE_TRACE_ZONE_BEGIN(thread_zone);

  // Reset affinity (as it can change over time).
  // TODO(benvanik): call this after waking in case CPU hotplugging happens.
  iree_thread_request_affinity(poller->thread, poller->ideal_thread_affinity);

  // Enter the running state immediately. Note that we could have been requested
  // to exit while suspended/still starting up, so check that here before we
  // mess with any data structures.
  const bool should_run =
      iree_atomic_exchange_int32(&poller->state, IREE_TASK_POLLER_STATE_RUNNING,
                                 iree_memory_order_acq_rel) !=
      IREE_TASK_POLLER_STATE_EXITING;
  if (IREE_LIKELY(should_run)) {
    // << work happens here >>
    iree_task_poller_pump_until_exit(poller);
  }

  IREE_TRACE_ZONE_END(thread_zone);
  iree_atomic_store_int32(&poller->state, IREE_TASK_POLLER_STATE_ZOMBIE,
                          iree_memory_order_release);
  iree_notification_post(&poller->state_notification, IREE_ALL_WAITERS);
  return 0;
}
