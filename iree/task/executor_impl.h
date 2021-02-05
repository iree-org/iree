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

#ifndef IREE_TASK_EXECUTOR_IMPL_H_
#define IREE_TASK_EXECUTOR_IMPL_H_

#include "iree/base/internal/math.h"
#include "iree/base/internal/prng.h"
#include "iree/base/synchronization.h"
#include "iree/base/tracing.h"
#include "iree/task/affinity_set.h"
#include "iree/task/executor.h"
#include "iree/task/list.h"
#include "iree/task/pool.h"
#include "iree/task/post_batch.h"
#include "iree/task/queue.h"
#include "iree/task/tuning.h"
#include "iree/task/worker.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_task_executor_s {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  // Defines how work is selected across queues.
  // TODO(benvanik): make mutable; currently always the same reserved value.
  iree_task_scheduling_mode_t scheduling_mode;

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
  iree_task_pool_t fence_task_pool;
  iree_task_pool_t dispatch_task_pool;

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
  // A list of incoming wait tasks that need to be waited on. Order doesn't
  // really matter here as all tasks will be waited on simultaneously.
  iree_atomic_task_slist_t incoming_waiting_slist;

  // Guards coordination logic; only one thread at a time may be acting as the
  // coordinator.
  iree_slim_mutex_t coordinator_mutex;
  // A list of wait tasks with external handles that need to be waited on.
  // Coordinators can choose to poll/wait on these.
  iree_task_list_t waiting_list;
  // Guards manipulation and use of the wait_set.
  // coordinator_mutex may be held when taking this lock.
  iree_slim_mutex_t wait_mutex;
  // Wait set containing all the tasks in waiting_list. Coordinator manages
  // keeping the waiting_list and wait_set in sync.
  iree_wait_set_t* wait_set;

  // A bitset indicating which workers are live and usable; all attempts to
  // push work onto a particular worker should check first with this mask. This
  // may change over time either automatically or by user request ("don't use
  // these cores for awhile I'm going to be using them" etc).
  iree_atomic_task_affinity_set_t worker_live_mask;

  // A bitset indicating which workers may be suspended and need to be resumed
  // via iree_thread_resume prior to them being able to execute work.
  iree_atomic_task_affinity_set_t worker_suspend_mask;

  // A bitset indicating which workers are currently idle. Used to bias incoming
  // tasks to workers that aren't doing much else. This is a balance of latency
  // to wake the idle workers vs. latency to wait for existing work to complete
  // on already woken workers.
  iree_atomic_task_affinity_set_t worker_idle_mask;

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
// If the |current_worker| has no more work remaining and |wait_on_idle| is set
// then the calling thread may wait on any pending wait tasks until one resolves
// or more work is scheduled for the worker.
void iree_task_executor_coordinate(iree_task_executor_t* executor,
                                   iree_task_worker_t* current_worker,
                                   bool wait_on_idle);

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
