// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_EXECUTOR_IMPL_H_
#define IREE_TASK_EXECUTOR_IMPL_H_

#include "iree/base/internal/math.h"
#include "iree/base/internal/prng.h"
#include "iree/base/threading/mutex.h"
#include "iree/task/affinity_set.h"
#include "iree/task/executor.h"
#include "iree/task/list.h"
#include "iree/task/pool.h"
#include "iree/task/post_batch.h"
#include "iree/task/process.h"
#include "iree/task/queue.h"
#include "iree/task/tuning.h"
#include "iree/task/worker.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A single compute slot in the executor's slot array. Holds a process
// pointer, a count of workers currently engaged with this slot, and a flag
// ensuring the completion callback fires exactly once.
//
// The active_drainers counter prevents the release callback (which frees
// the processor context) from firing while other workers are still inside
// drain(). The completion_claimed flag ensures exactly one worker runs
// the eager completion callback (semaphore signaling, dependent activation).
typedef struct iree_task_compute_slot_t {
  // Process pointer: 0 (empty) or a valid process. Set via CAS on activate,
  // cleared by the last active drainer on release.
  iree_atomic_intptr_t process;
  // Number of workers currently inside the drain protocol for this slot
  // (between their fetch_add and fetch_sub). Incremented before accessing
  // the process, decremented after drain returns. The worker whose
  // decrement reaches zero handles release (freeing resources, clearing
  // the slot).
  iree_atomic_int32_t active_drainers;
  // Set to 1 by the first worker to claim completion (via CAS). Ensures
  // the eager completion callback (signaling, dependent activation) runs
  // exactly once. Reset to 0 when the slot is cleared on release.
  iree_atomic_int32_t completion_claimed;
} iree_task_compute_slot_t;

struct iree_task_executor_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  // NUMA node this executor's workers are pinned to, or
  // IREE_TASK_TOPOLOGY_NODE_ID_ANY if unspecified.
  iree_task_topology_node_id_t node_id;

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

  // Guards coordination logic; only one thread at a time may be acting as the
  // coordinator.
  iree_slim_mutex_t coordinator_mutex;

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

  //===--------------------------------------------------------------------===//
  // Process scheduling
  //===--------------------------------------------------------------------===//

  // Lock-free MPSC list for budget-1 processes (queue management, host
  // callbacks, retire/signal). Workers try_pop from this before scanning
  // for tasks. A failed pop on an empty list costs one atomic load (~1ns).
  //
  // Multiple producers (any thread that calls schedule_process) push
  // concurrently. Each worker pops at most one process per drain cycle
  // and drains it to completion or sleep before popping the next.
  iree_task_process_slist_t immediate_list;

  // Fixed-size array of compute process slots for budget>1 processes.
  // Each slot holds a process pointer, an active drainer count, and a
  // completion-claimed flag. Workers scan these round-robin after draining
  // the immediate list, calling drain() on each occupied slot to
  // cooperatively execute bounded work.
  //
  // Two-phase lifecycle:
  //   Activate:  CAS(process: 0 → ptr) by schedule_process.
  //   Drain:     Workers increment active_drainers before entering drain(),
  //              decrement after. Multiple workers drain concurrently.
  //   Complete:  First worker to CAS completion_claimed runs completion_fn
  //              eagerly (signals semaphores, activates dependents).
  //   Release:   Last worker to decrement active_drainers to zero calls
  //              release_fn (frees processor context) and clears the slot.
  //
  // This separation ensures the completion callback fires immediately
  // when work finishes (zero latency on downstream signaling) while the
  // processor context stays alive until every worker has exited drain().
  //
  // 16 slots supports up to 16 concurrent budget>1 processes. In practice
  // this is far more than needed — the local_task driver typically has 1-3
  // active command buffer processes at a time.
  iree_task_compute_slot_t compute_slots[IREE_TASK_EXECUTOR_MAX_COMPUTE_SLOTS];
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
