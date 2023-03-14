// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_POLLER_H_
#define IREE_TASK_POLLER_H_

#include <stdbool.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/task/affinity_set.h"
#include "iree/task/list.h"
#include "iree/task/task.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_task_executor_t iree_task_executor_t;

// Indicates the current state of a poller or, in the case of EXITING, the state
// the poller should transition to.
//
// Transition graph:
//   SUSPENDED -> RUNNING -> EXITING -> ZOMBIE
//
// NOTE: state values are ordered such that </> comparisons can be used; ensure
// that for example all states after resuming are > SUSPENDED and all states
// before exiting are < EXITING.
typedef enum iree_task_poller_state_e {
  // Wait thread has been created in a suspended state and must be resumed to
  // wake for the first time.
  IREE_TASK_POLLER_STATE_SUSPENDED = 0,
  // Wait thread is running and servicing wait tasks.
  IREE_TASK_POLLER_STATE_RUNNING = 1,
  // Wait thread should exit (or is exiting) and will soon enter the zombie
  // state.
  IREE_TASK_POLLER_STATE_EXITING = 2,
  // Wait thread has exited and entered a 🧟 state (waiting for join).
  // The thread handle is still valid and must be destroyed.
  IREE_TASK_POLLER_STATE_ZOMBIE = 3,
} iree_task_poller_state_t;

// Wait task poller with a dedicated thread for performing syscalls.
// This keeps potentially-blocking syscalls off the worker threads and ensures
// the lowest possible latency for wakes as the poller will always be kept in
// the system wait queue.
//
// During coordination wait tasks are registered with the poller for handling.
// The wait thread will wake, merge the newly-registered tasks into its lists,
// and then enter the system multi-wait API to wait for either one or more waits
// to resolve or the timeout to be hit (representing sleeps). Resolved waits
// will cause the wait task to be resubmitted to the executor with a flag
// indicating that they have completed waiting and can be retired. This ensures
// that all task-related work (completion callbacks, etc) executes on the worker
// threads and the poller can immediately return to the system for more waiting.
typedef struct {
  // Parent executor used to access the global work queue and submit wakes.
  iree_task_executor_t* executor;

  // Current state of the poller (iree_task_poller_state_t).
  iree_atomic_int32_t state;
  // Notification signaled when the wait thread changes state.
  iree_notification_t state_notification;

  // Ideal affinity for the wait thread. This can be used to keep the wait
  // thread from contending with the processing threads. To allow the wait
  // thread to run anywhere use iree_thread_affinity_set_any.
  iree_thread_affinity_t ideal_thread_affinity;

  // Thread handle of the wait thread. If the thread has exited the handle will
  // remain valid so that the poller can query its state.
  iree_thread_t* thread;

  // Event used to force the wait thread to wake.
  // This allows the wait thread to remain in a syscall but still be woken when
  // new wait tasks arrive and need to be managed by the wait thread.
  // Set from threads submitting tasks to the poller and reset after the wait
  // thread has woken and processed them. All system waits have this event
  // in the wait set.
  iree_event_t wake_event;

  // A LIFO mailbox used by coordinators to post wait tasks to the poller.
  // This allows for submissions to add tasks without needing to synchronize
  // with the wait thread; tasks are pushed to the mailbox and then merged with
  // the full wait set by the wait thread the next time it wakes.
  iree_atomic_task_slist_t mailbox_slist;

  // A list of wait tasks with external handles that need to be waited on.
  // Managed by the wait thread and must not be accessed from any other thread.
  // This is the full set of waits actively being managed by the poller.
  iree_task_list_t wait_list;

  // Wait set containing wait handles from wait_list.
  // Managed by the wait thread and must not be accessed from any other thread.
  // This may only contain a subset of the wait_list in cases where some of
  // the wait tasks do not have full system handles.
  iree_wait_set_t* wait_set;
} iree_task_poller_t;

// Initializes |out_poller| with a new poller.
// |executor| will be used to submit woken tasks for processing.
iree_status_t iree_task_poller_initialize(
    iree_task_executor_t* executor,
    iree_thread_affinity_t ideal_thread_affinity,
    iree_task_poller_t* out_poller);

// Requests that the poller wait thread begin exiting (if it hasn't already).
// If the wait thread is in a syscall it will be woken as soon as possible.
//
// May be called from any thread. Any active waits will be aborted as possible.
void iree_task_poller_request_exit(iree_task_poller_t* poller);

// Blocks the caller until |poller| has exited.
//
// May be called from any thread.
void iree_task_poller_await_exit(iree_task_poller_t* poller);

// Deinitializes |poller| after the thread has exited.
// The poller must be in the IREE_TASK_POLLER_STATE_ZOMBIE state.
//
// Expected shutdown sequence:
//  - request_exit
//  - await_exit
//  - deinitialize
void iree_task_poller_deinitialize(iree_task_poller_t* poller);

// Enqueues |wait_tasks| on the poller and kicks the wait thread.
// The task pointers will be retained by the poller and must remain valid.
//
// May be called from any thread. Waits may begin and complete prior to the
// function returning.
void iree_task_poller_enqueue(iree_task_poller_t* poller,
                              iree_task_list_t* wait_tasks);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_POLLER_H_
