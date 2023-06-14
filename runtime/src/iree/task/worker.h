// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_WORKER_H_
#define IREE_TASK_WORKER_H_

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/prng.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/task/affinity_set.h"
#include "iree/task/executor.h"
#include "iree/task/list.h"
#include "iree/task/queue.h"
#include "iree/task/task.h"
#include "iree/task/topology.h"
#include "iree/task/tuning.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Indicates the current state of a worker or, in the case of EXITING, the state
// the worker should transition to.
//
// Transition graph:
//   SUSPENDED -> RUNNING (IDLE<->PROCESSING) -> EXITING -> ZOMBIE
//
// NOTE: state values are ordered such that </> comparisons can be used; ensure
// that for example all states after resuming are > SUSPENDED and all states
// before exiting are < EXITING.
typedef enum iree_task_worker_state_e {
  // Worker is idle or actively processing tasks (either its own or others).
  IREE_TASK_WORKER_STATE_RUNNING = 0,
  // Worker should exit (or is exiting) and will soon enter the zombie state.
  // Coordinators can request workers to exit by setting their state to this and
  // then waking.
  IREE_TASK_WORKER_STATE_EXITING = 1,
  // Worker has exited and entered a 🧟 state (waiting for join).
  // The thread handle is still valid and must be destroyed.
  IREE_TASK_WORKER_STATE_ZOMBIE = 2,
} iree_task_worker_state_t;

// A worker within the executor pool.
//
// NOTE: fields in here are touched from multiple threads with lock-free
// techniques. The alignment of the entire iree_task_worker_t as well as the
// alignment and padding between particular fields is carefully (though perhaps
// not yet correctly) selected; see the 'LAYOUT' comments below.
typedef struct iree_task_worker_t {
  // A LIFO mailbox used by coordinators to post tasks to this worker.
  // As workers self-nominate to be coordinators and fan out dispatch shards
  // they can directly emplace those shards into the workers that should execute
  // them based on the work distribution policy. When workers go to look for
  // more work after their local queue empties they will flush this list and
  // move all of the tasks into their local queue and restart processing.
  // LAYOUT: must be 64b away from local_task_queue.
  iree_atomic_task_slist_t mailbox_slist;

  // Current state of the worker (iree_task_worker_state_t).
  // LAYOUT: frequent access; next to wake_notification as they are always
  //         accessed together.
  iree_atomic_int32_t state;

  // Notification signaled when the worker should wake (if it is idle).
  // LAYOUT: next to state for similar access patterns; when posting other
  //         threads will touch mailbox_slist and then send a wake
  //         notification.
  iree_notification_t wake_notification;

  // Notification signaled when the worker changes any state.
  iree_notification_t state_notification;

  // Parent executor that can be used to access the global work queue or task
  // pool. Executors always outlive the workers they own.
  iree_task_executor_t* executor;

  // Globally unique worker index (worker_base_index + local worker_index).
  iree_host_size_t worker_index;

  // Bit the worker represents in the various worker bitsets.
  // Local to the executor owning the worker.
  iree_task_affinity_set_t worker_bit;

  // Ideal thread affinity for the worker thread.
  iree_thread_affinity_t ideal_thread_affinity;

  // A bitmask of other group indices that share some level of the cache
  // hierarchy. Workers of this group are more likely to constructively share
  // some cache levels higher up with these other groups. For example, if the
  // workers in a group all share an L2 cache then the groups indicated here may
  // all share the same L3 cache.
  iree_task_affinity_set_t constructive_sharing_mask;

  // Maximum number of attempts to make when trying to steal tasks from other
  // workers. This could be 64 (try stealing from all workers) or just a handful
  // (try stealing from these 3 other cores that share your L3 cache).
  uint32_t max_theft_attempts;

  // Rotation counter for work stealing (ensures we don't favor one victim).
  // Only ever touched by the worker thread as it steals work.
  iree_prng_minilcg128_state_t theft_prng;

  // Thread handle of the worker. If the thread has exited the handle will
  // remain valid so that the executor can query its state.
  iree_thread_t* thread;

  // Guess at the current processor ID.
  // This is updated infrequently as it can be semi-expensive to determine
  // (on some platforms at least 1 syscall involved). We always update it upon
  // waking as idle waits are the most likely place the worker will be migrated
  // across processors.
  iree_cpu_processor_id_t processor_id;
  // An opaque tag used to reduce the cost of processor ID queries.
  iree_cpu_processor_tag_t processor_tag;

  // Destructive interference padding between the mailbox and local task queue
  // to ensure that the worker - who is pounding on local_task_queue - doesn't
  // contend with submissions or coordinators dropping new tasks in the mailbox.
  //
  // Today we don't need this, however on 32-bit systems or if we adjust the
  // size of iree_task_affinity_t/iree_task_affinity_set_t/etc we may need to
  // add it back.
  //
  // NOTE: due to the layout requirements of this structure (to avoid cache
  // interference) this is the only place padding should be added.
  // uint8_t _padding[8];

  // Pointer to local memory available for use exclusively by the worker.
  // The base address should be aligned to avoid false sharing with other
  // workers.
  iree_byte_span_t local_memory;

  // Worker-local FIFO queue containing the tasks that will be processed by the
  // worker. This queue supports work-stealing by other workers if they run out
  // of work of their own.
  // LAYOUT: must be 64b away from mailbox_slist.
  iree_task_queue_t local_task_queue;
} iree_task_worker_t;
static_assert(offsetof(iree_task_worker_t, mailbox_slist) +
                      sizeof(iree_atomic_task_slist_t) <
                  iree_hardware_constructive_interference_size,
              "mailbox_slist must be in the first cache line");
static_assert(offsetof(iree_task_worker_t, local_task_queue) >=
                  iree_hardware_constructive_interference_size,
              "local_task_queue must be separated from mailbox_slist by "
              "at least a cache line");

// Initializes a worker by creating its thread and configuring it for receiving
// tasks. Where supported the worker will be created in a suspended state so
// that we aren't creating a thundering herd on startup:
// https://en.wikipedia.org/wiki/Thundering_herd_problem
iree_status_t iree_task_worker_initialize(
    iree_task_executor_t* executor, iree_host_size_t worker_index,
    const iree_task_topology_group_t* topology_group,
    iree_host_size_t stack_size, iree_byte_span_t local_memory,
    iree_prng_splitmix64_state_t* seed_prng, iree_task_worker_t* out_worker);

// Requests that the worker begin exiting (if it hasn't already).
// If the worker is actively processing tasks it will wait until it has
// completed all it can and is about to go idle prior to exiting.
//
// May be called from any thread (including the worker thread).
void iree_task_worker_request_exit(iree_task_worker_t* worker);

// Blocks the caller until |worker| has exited.
//
// May be called from any thread.
void iree_task_worker_await_exit(iree_task_worker_t* worker);

// Deinitializes a worker that has successfully exited.
// The worker must be in the IREE_TASK_WORKER_STATE_ZOMBIE state.
//
// Expected shutdown sequence:
//  - request_exit on all workers
//  - await_exit on all workers
//  - deinitialize all workers
void iree_task_worker_deinitialize(iree_task_worker_t* worker);

// Posts a FIFO list of tasks to the worker mailbox. The target worker takes
// ownership of the tasks and will be woken if it is currently idle.
//
// May be called from any thread (including the worker thread).
void iree_task_worker_post_tasks(iree_task_worker_t* worker,
                                 iree_task_list_t* list);

// Tries to steal up to |max_tasks| from the back of the queue.
// Returns NULL if no tasks are available and otherwise up to |max_tasks| tasks
// that were at the tail of the worker FIFO will be moved to the |target_queue|
// and the first of the stolen tasks is returned. While tasks from the FIFO
// are preferred this may also steal tasks from the mailbox.
iree_task_t* iree_task_worker_try_steal_task(iree_task_worker_t* worker,
                                             iree_task_queue_t* target_queue,
                                             iree_host_size_t max_tasks);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_WORKER_H_
