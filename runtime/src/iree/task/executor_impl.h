// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_EXECUTOR_IMPL_H_
#define IREE_TASK_EXECUTOR_IMPL_H_

#include "iree/base/internal/math.h"
#include "iree/base/internal/prng.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/tracing.h"
#include "iree/task/affinity_set.h"
#include "iree/task/executor.h"
#include "iree/task/list.h"
#include "iree/task/poller.h"
#include "iree/task/pool.h"
#include "iree/task/post_batch.h"
#include "iree/task/queue.h"
#include "iree/task/tuning.h"
#include "iree/task/worker.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_task_executor_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  // Leaked dynamically allocated name used for tracing calls.
  // This pointer - once allocated - will be valid for the lifetime of the
  // process and can be used for IREE_TRACE plotting/allocation calls.
  IREE_TRACE(const char* trace_name;)

  // Defines how work is selected across queues.
  // TODO(benvanik): make mutable; currently always the same reserved value.
  iree_task_scheduling_mode_t scheduling_mode;

  // Time each worker should spin before parking itself to wait for more work.
  // IREE_DURATION_ZERO is used to disable spinning.
  iree_duration_t worker_spin_ns;

  // State used by the work-stealing operations performed by donated threads.
  // This is **NOT SYNCHRONIZED** and relies on the fact that we actually don't
  // much care about the precise selection of workers enough to mind any tears
  // we get in the PRNG state that lives inside. Cache write-back order and
  // incidental cache line availability/visibility update frequency is like an
  // extra layer of PRNG anyway ;)
  iree_prng_minilcg128_state_t donation_theft_prng;

  // Pools of transient dispatch tasks shared across all workers.
  // Depending on configuration the task pool may allocate after creation using
  // the allocator provided upon executor creation.
  //
  // Sized to be able to fit at least:
  //   iree_task_fence_t
  //   iree_task_dispatch_shard_t
  // Increasing the size larger than these will waste memory.
  iree_task_pool_t transient_task_pool;

  // A list of incoming tasks that are ready to execute immediately.
  // The list is LIFO and we require that task lists are reversed by the
  // submitter so we can use iree_atomic_slist_concat to quickly prepend the
  // LIFO list to the atomic slist. By doing this we can construct the task
  // lists in LIFO order prior to submission, concat with a pointer swap into
  // this list, flush from the list in LIFO order during coordination, and do a
  // single LIFO->FIFO conversion while distributing work. What could have been
  // half a dozen task list pointer walks and inverted sequential memory access
  // becomes one.
  //
  // Example:
  //   existing tasks: C B A
  //        new tasks: 1 2 3
  //    updated tasks: 3 2 1 C B A
  iree_atomic_task_slist_t incoming_ready_slist;

  // iree_event_t pool used to acquire system wait handles.
  // Many subsystems interacting with the executor will need events to park
  // their work in the wait set and sharing the pool across all of them ensures
  // we limit the number we have outstanding and avoid syscalls to allocate
  // them.
  iree_event_pool_t* event_pool;

  // Guards coordination logic; only one thread at a time may be acting as the
  // coordinator.
  iree_slim_mutex_t coordinator_mutex;

  // Wait task polling and wait thread manager.
  // This handles all system waits so that we can keep the syscalls off the
  // worker threads and lower wake latencies (the wait thread can enqueue
  // completed waits immediately after they resolve instead of waiting for
  // existing computation on the workers to finish).
  iree_task_poller_t poller;

  // A bitset indicating which workers are likely to be live and usable; all
  // attempts to push work onto a particular worker should check first with this
  // mask. This may change over time either automatically or by user request
  // ("don't use these cores for awhile I'm going to be using them" etc).
  //
  // This mask is just a hint, accessed with memory_order_relaxed. Readers must
  // be OK with getting slightly out-of-date information. The only way to get
  // an authoritative answer to the question "is this worker live" is to
  // atomically query worker->state. This mask is for usage patterns where one
  // needs a cheap (single relaxed atomic op) approximation of all N workers'
  // live state without having to perform N expensive atomic ops.
  iree_atomic_task_affinity_set_t worker_live_mask;

  // A bitset indicating which workers are currently idle. Used to bias incoming
  // tasks to workers that aren't doing much else. This is a balance of latency
  // to wake the idle workers vs. latency to wait for existing work to complete
  // on already woken workers.
  //
  // This mask is just a hint, accessed with memory_order_relaxed. See the
  // comment on worker_live_mask.
  iree_atomic_task_affinity_set_t worker_idle_mask;

  // Base value added to each executor-local worker index.
  // This allows workers to uniquely identify themselves in multi-executor
  // configurations.
  iree_host_size_t worker_base_index;

  // Specifies how many workers threads there are.
  // For now this number is fixed per executor however if we wanted to enable
  // live join/leave behavior we could change this to a registration mechanism.
  iree_host_size_t worker_count;
  iree_task_worker_t* workers;  // [worker_count]
};

// Merges a submission into the primary FIFO queues.
// Coordinators will fetch items from here as workers demand them but otherwise
// not be notified of the changes (waiting until coordination runs again).
//
// May be called from any thread.
void iree_task_executor_merge_submission(iree_task_executor_t* executor,
                                         iree_task_submission_t* submission);

// Schedules all ready tasks in the |pending_submission| list.
// Only called during coordination and expects the coordinator lock to be held.
void iree_task_executor_schedule_ready_tasks(
    iree_task_executor_t* executor, iree_task_submission_t* pending_submission,
    iree_task_post_batch_t* post_batch);

// Dispatches tasks in the global submission queue to workers.
// |current_worker| will be NULL if called from a non-worker thread and
// otherwise be the current worker; used to avoid round-tripping through the
// whole system to post to oneself.
//
// If the |current_worker| has no more work remaining then the calling thread
// may wait on any pending wait tasks until one resolves or more work is
// scheduled for the worker. If no worker is provided the call will return
// without waiting.
void iree_task_executor_coordinate(iree_task_executor_t* executor,
                                   iree_task_worker_t* current_worker);

// Tries to steal an entire task from a sibling worker (based on topology).
// Returns a task that is available (has not yet begun processing at all).
// May steal multiple tasks and add them to the |local_task_queue|.
iree_task_t* iree_task_executor_try_steal_task(
    iree_task_executor_t* executor,
    iree_task_affinity_set_t constructive_sharing_mask,
    uint32_t max_theft_attempts, iree_prng_minilcg128_state_t* theft_prng,
    iree_task_queue_t* local_task_queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_EXECUTOR_IMPL_H_
