// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_EXECUTOR_H_
#define IREE_TASK_EXECUTOR_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/event_pool.h"
#include "iree/task/scope.h"
#include "iree/task/submission.h"
#include "iree/task/task.h"
#include "iree/task/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//==============================================================================
// IREE Task Executor
//==============================================================================
//
// Roughly models wavefront-style GPU dispatch. Users submit task DAGs with
// fine-grained dependency information for the executor to schedule across a set
// of workers. As tasks become ready to execute they are placed into per-worker
// FIFOs and workers run through them in a breadth-first fashion executing and
// resolving tasks and building up new waves of ready tasks. Workers will always
// make forward progress and only when they run out of work will they attempt to
// self-nominate to play the role of coordinator and schedule any newly-
// submitted or readied tasks. Only once all tasks have been retired and
// waits on external resources remain does the task system suspend itself until
// more tasks are submitted or an external wait resolves.
//
// Our goal is to do the minimal amount of work to get the maximum amount of
// concurrency the user requests or allows (by way of their dependencies).
// Whether on a single core where you want to timeshare with an application or
// across hundreds the same architecture holds. Where there is inefficiency it's
// almost always surmountable with properly constructed tasks: choose the right
// granularity for dispatches, choose the right fan-out for tiles within those
// dispatches, choose the right places to insert barriers to force fan-in to
// reduce memory utilization or right places to batch barriers to allow less
// synchronization with the work queue, etc. All of those choices are ones this
// system is designed to handle dynamically via the task graphs provided that
// are themselves (in the IREE world) mapped 1:1 with the GPU-esque grid
// dispatch and command buffer model. It's a super-power if a human is authoring
// all that information but what makes it particularly powerful here is that we
// are authoring that in the compiler based on a tremendous amount of
// higher-level information we can derive from the whole program. Every bit of
// dynamism here is matched with the ability to tighten down the screws and gain
// back anything lost by way of compiler improvements while also being able to
// generalize out to far more complex systems (higher parallelism, higher and
// more efficient concurrency, etc).
//
// The design of this system allows for a spectrum of dynamic behavior based on
// desired usage scenarios:
// - variable number of persistent workers based on compute/memory topology
// - per-task scope and per-task worker affinity to control for:
//   - power islands on multi-core systems with fine-grained power management
//   - heterogenous microarchitectures in big.LITTLE/etc compute complexes
//   - task isolation between multiple active requests or users
//   - latency prioritization by partitioning workloads by priority
// - scheduling overhead tradeoffs by varying:
//   - coordination/flush frequency to reduce cross-thread communication
//   - by statically inserting dispatch shards to avoid dynamic fan-out
//   - thread donation to avoid likely context switches upon submit+wait
//   - multi-wait across all users by sharing a wait set
//   - per-worker work-stealing specification of victim workers in the topology
//   - limited work-stealing to prevent chained stealing/cascading theft
//
// Required reading:
//  https://www.usenix.org/conference/osdi20/presentation/ma
//    (closest equivalent to this scheduling model)
//  https://www.cister-labs.pt/summer2017/w3/Parallelism%20-%20Dag%20Model.pdf
//    (good overall, our worker local lists/mailboxes are work-stealing queues)
//  http://people.csail.mit.edu/shanir/publications/Flat%20Combining%20SPAA%2010.pdf
//    (what we model with the coordinator)
//  http://mcg.cs.tau.ac.il/papers/opodis2010-quasi.pdf
//    (we exploit relaxed consistency for all our cross-thread queuing, see ^)
//  https://moodycamel.com/blog/2014/a-fast-general-purpose-lock-free-queue-for-c++.htm
//    (moodycamel is the state of the art on scaling queues; read it all)
//  https://blog.molecular-matters.com/2015/08/24/job-system-2-0-lock-free-work-stealing-part-1-basics/
//  https://blog.molecular-matters.com/2015/09/08/job-system-2-0-lock-free-work-stealing-part-2-a-specialized-allocator/
//  https://blog.molecular-matters.com/2015/09/25/job-system-2-0-lock-free-work-stealing-part-3-going-lock-free/
//  https://blog.molecular-matters.com/2015/11/09/job-system-2-0-lock-free-work-stealing-part-4-parallel_for/
//  https://blog.molecular-matters.com/2016/04/04/job-system-2-0-lock-free-work-stealing-part-5-dependencies/
//    (fantastic 5 part blog series; very similar to this)
//  http://www.cs.cmu.edu/afs/cs.cmu.edu/Web/People/blelloch/papers/jacm99.pdf
//    (provably optimal dynamic nested parallelism in 1999; basically: GPUs)
//  http://www.cs.cmu.edu/~blelloch/papers/locality2000.pdf
//    (followup to jacm99; using locality now to guide work stealing)
//  https://www.cs.cmu.edu/afs/cs.cmu.edu/Web/People/blelloch/papers/CGK07.pdf
//    (worker affinity and task locality for constructive cache sharing)
//
//==============================================================================
// Life of an iree_task_t / high level algorithm
//==============================================================================
//
// 1. Users allocate (from iree_task_pool_t, slice from arenas, etc) and
//    construct a DAG of iree_task_ts.
//
//   a. Task dependency information is setup via completion_tasks for simple
//      dependencies, implicit fan-out/fan-in (dispatches), or explicit fan-in
//      (barriers).
//
//   b. Tasks are pushed into iree_task_submission_t (LIFO, thread-local list).
//      If the task has no initial unmet initial dependencies it is placed into
//      the ready_list. If it is initially waiting on an external resource such
//      as iree_wait_handle_t then it is placed into the waiting_list.
//
// 2. iree_task_executor_submit (LIFO, atomic slist)
//    Submissions have their task thread-local lists concatenated into a LIFO
//    incoming_ready_slist or the wait poller shared by the executor.
//
// 3. iree_task_executor_flush (or a worker puts on its coordinator hat 🎩)
//
//   a. Tasks are flushed from the incoming_ready_slist into a coordinator-local
//      FIFO task queue. This centralizes enqueuing from all threads into a
//      single ordered list.
//
//   b. iree_task_executor_schedule_ready_tasks: walks the FIFO task queue and
//      builds a iree_task_post_batch_t containing the per-worker tasks
//      in LIFO order.
//
//   c. iree_task_post_batch_submit: per-worker tasks are pushed to their
//      respective iree_task_worker_t mailbox_slist and the workers with new
//      tasks are notified to wake up (if not already awake).
//
// 4. iree_task_worker_main_pump_once (LIFO mailbox -> FIFO thread-local list)
//    When either woken or after completing all available thread-local work
//    each worker will check its mailbox_slist to see if any tasks have been
//    posted.
//
//    a. Tasks are flushed from the LIFO mailbox into the local_task_queue FIFO
//       for the particular worker.
//
//    b. If the mailbox is empty the worker *may* attempt to steal work from
//       another nearby worker in the topology.
//
//    c. Any tasks in the local_task_queue are executed until empty.
//       Tasks are retired and dependent tasks (via completion_task or barriers)
//       are made ready and placed in the executor incoming_ready_slist as with
//       iree_task_executor_submit.
//
//    d. If no more thread-local work is available and the mailbox_slist is
//       empty the worker will self-nominate for coordination and attempt to don
//       the coordinator hat with iree_task_executor_coordinate. If new work
//       becomes available after coordination step 5 repeats.
//
//    e. If another worker (or iree_task_executor_flush) is already wearing the
//       coordinator hat then the worker will go to sleep.
//
//==============================================================================
// Scaling Down
//==============================================================================
//
// IREE is built at all levels - and both in the compiler and runtime - to scale
// to different needs. Everything that IREE imposes on the runtime performance
// and binary size is a spectrum of choices made that allows a user to only pay
// for what they use.
//
// If a deployment scenario does not need complex multithreading and
// out-of-order execution then this task system can be used in single-threaded
// mode to at least allow for offloading from the main application thread. In
// even more constrained scenarios (or embeddings within other systems that have
// thread pools of their own) it can be used in zero-threaded mode with only
// donated threads from the user performing work when the user wants it to
// happen within its control. It still gives the benefits of wave-style
// scheduling, multi-waiting, locality-aware work distribution, etc as well as
// giving us a single target interface from the compiler to communicate
// fine-grained dependency information to the runtime.
//
// If the cost of a few KB of data structures and some cheap uncontended atomic
// linked list concatenations is still scary (it shouldn't be for 95% of uses)
// then it's also possible to have a HAL driver that doesn't use this task
// system at all and instead just executes the command buffers directly just
// like our Vulkan/Metal/etc GPU backends do. Even though I don't recommend that
// (one wouldn't be saving as much as they think and be losing a lot instead)
// the layering holds and it can be useful if there's an existing external
// sophisticated task execution system (ala taskflow) that is already in present
// in an application.
//
// One assertion of IREE is that for models that take more than milliseconds to
// execute then asynchronous scheduling is almost always worth it even on
// systems with single cores. The ability to cooperatively schedule model
// execution allows applications significant control over their total program
// scheduling behavior; just as on a Commodore 64 you'd have to interrupt work
// on vsync to begin scanning out pixels to the screen and then resume afterward
// it's rare to see any system even scaling down to double-digit MHz
// microcontrollers that doesn't benefit from the ability to cleanly suspend and
// resume execution.
//
// But even if *all* of that is too much, the compile-time representations in
// the HAL IR are designed to be lowered away: execution modeling does not need
// to bottom out on a hal.command_buffer.dispatch that maps 1:1 with the runtime
// iree_hal_command_buffer_dispatch call: dispatch can be lowered into LLVM
// IR calls and finally into native code to do precisely what you want. The HAL
// at runtime is a useful abstraction to allow for switching your target
// execution system (statically or dynamically across deployments) and to share
// the same execution system across multiple models that may be executing
// simultaneously but it is _not_ a requirement that the IREE HAL runtime
// implementation is used. It's called multi-level IR for a reason and the HAL
// IR is just one level that may have many more below it.
//
// So yeah: don't worry. It's almost certain that the thing making or breaking
// the performance of models over 1ms of execution time is not the HAL, and that
// in models at or above that scale the benefits we get from being able to
// holistically schedule the work far outstrip any specialization that can be
// done by hand. That's to say: only worry about this if your model is literally
// 4 floats coming from an IMU and a few hundred scalar instructions to predict
// whether the user is walking, and that shouldn't be using the runtime HAL at
// all and really likely doesn't benefit from using IREE at any scale - just go
// straight to LLVM IR from the source.
//
//==============================================================================
// Scaling Up
//==============================================================================
//
// The task system has an implicit limit of 64 workers. This intentional
// limitation simplifies several parts of the code while also preventing misuse:
// it rarely (if ever) makes sense to have more than 64 compute-dominated
// threads working on a single problem. Achieving high performance in such
// situations requires extremely careful control over the OS scheduler, memory
// bandwidth consumption, and synchronization. It's always possible to make the
// problem more compute-bound or very carefully try to fit in specific cache
// sizes to avoid more constrained bandwidth paths but it's a non-portable
// whack-a-mole style solution that is in conflict with a lot of what IREE seeks
// to do with respect to low-latency and multi-tenant workloads.
//
// If more than 64 unique L1/L2 caches (or realistically more than probably ~32)
// are available *and* all of them are attached to the same memory controllers
// (no NUMA involved) then the solution is straightfoward: use multiple IREE
// task executors. Either within a process or in separate processes the
// granularity is coarse enough to not be a burden and changes the problem from
// needing 100% perfect work scaling of a single task to needing a naive
// distributed workload solution at the algorithm level.
//
// Many useful effects also fall out of solving the work distribution problem.
// Even for single-tenant workloads being able to split work between two
// executors allows for natural mappings on NUMA systems or completely
// independent machines. When supporting multi-tenant workloads (even if the
// same program is acting as multiple-tenants in a minibatched-style algorithm)
// the improvements of isolation both in memory access patterns and in variance
// from potentially bad system behavior dramatically improve: there aren't many
// opportunities for contention in this system but one can guarantee zero
// contention by simply not sharing the resources!

// A bitfield specifying the scheduling mode used for configuring how (or if)
// work is balanced across queues.
enum iree_task_scheduling_mode_bits_t {
  // TODO(benvanik): batch, round-robin, FCFS, SJF, etc.
  // We can also allow for custom scheduling, though I'm skeptical of the value
  // of that. We should look into what GPUs do in hardware for balancing things
  // (if anything this sophisticated at all). The potential benefit here is that
  // we can optimize for offline workloads by allowing each queue to be drained
  // until blocking - hopefully optimizing cache coherency and reducing the
  // total memory high-water mark - or optimize for latency across all queues by
  // taking tasks from all queues equally. There are other more interesting
  // scheduling strategies such as preferring the widest tasks available from
  // any queue such that we are keeping as many workers active as possible to
  // reach peak utilization or artificially limiting which tasks we allow
  // through to keep certain CPU cores asleep unless absolutely required.
  IREE_TASK_SCHEDULING_MODE_RESERVED = 0u,
};
typedef uint32_t iree_task_scheduling_mode_t;

// Options controlling task executor behavior.
typedef struct iree_task_executor_options_t {
  // Specifies the schedule mode used for worker and workload balancing.
  iree_task_scheduling_mode_t scheduling_mode;

  // Base value added to each executor-local worker index.
  // This allows workers to uniquely identify themselves in multi-executor
  // configurations.
  iree_host_size_t worker_base_index;

  // TODO(benvanik): add a scope_spin_ns to control wait-idle and other
  // scope-related waits coming from outside of the task system.

  // Maximum duration in nanoseconds each worker should spin waiting for
  // additional work. In almost all cases this should be IREE_DURATION_ZERO as
  // spinning is often extremely harmful to system health. Only set to non-zero
  // values when latency is the #1 priority (over thermals, system-wide
  // scheduling, and the environment).
  iree_duration_t worker_spin_ns;

  // Minimum size in bytes of each worker thread stack.
  // The underlying platform may allocate more stack space but _should_
  // guarantee that the available stack space is near this amount. Note that the
  // task system will take some stack space and not all bytes should be assumed
  // usable. Note that as much as possible users should not rely on the stack
  // for storage over ~16-32KB and instead use local workgroup memory.
  iree_host_size_t worker_stack_size;

  // Defines the bytes to be allocated and reserved by each worker to use for
  // local memory operations. Will be rounded up to the next power of two.
  // Dispatches performed will be able to request up to this amount of memory
  // for their invocations and no more. May be 0 if no worker local memory is
  // required.
  // By default the CPU L2 cache size is used if such queries are supported.
  iree_host_size_t worker_local_memory_size;
} iree_task_executor_options_t;

// Initializes |out_options| to default values.
void iree_task_executor_options_initialize(
    iree_task_executor_options_t* out_options);

// Base task system executor interface.
typedef struct iree_task_executor_t iree_task_executor_t;

// Creates a task executor using the specified topology.
// |options| must be initialized with iree_task_executor_options_initialize by
// callers and then overridden as required.
// |topology| is only used during creation and need not live beyond this call.
// |out_executor| must be released by the caller.
iree_status_t iree_task_executor_create(iree_task_executor_options_t options,
                                        const iree_task_topology_t* topology,
                                        iree_allocator_t allocator,
                                        iree_task_executor_t** out_executor);

// Retains the given |executor| for the caller.
void iree_task_executor_retain(iree_task_executor_t* executor);

// Releases the given |executor| from the caller.
void iree_task_executor_release(iree_task_executor_t* executor);

// Trims pools and caches used by the executor and its workers.
void iree_task_executor_trim(iree_task_executor_t* executor);

// Returns the number of live workers usable by the executor.
// The actual number used for any particular operation is dynamic.
iree_host_size_t iree_task_executor_worker_count(
    iree_task_executor_t* executor);

// Returns an iree_event_t pool managed by the executor.
// Users of the task system should acquire their transient events from this.
// Long-lived events should be allocated on their own in order to avoid
// expending the pool and harming high-frequency event acquisition.
iree_event_pool_t* iree_task_executor_event_pool(
    iree_task_executor_t* executor);

// Acquires a fence for the given |scope| from the executor fence pool.
iree_status_t iree_task_executor_acquire_fence(iree_task_executor_t* executor,
                                               iree_task_scope_t* scope,
                                               iree_task_fence_t** out_fence);

// TODO(benvanik): scheduling mode mutation, compute quota control, etc.

// Submits a batch of tasks for execution.
// The submission represents a DAG of tasks all reachable from the initial
// submission lists.
//
// Ownership of the tasks remains with the caller for the lifetime of the
// submission unless tasks have a custom pool specified that they can be
// returned to.
//
// Safe to call from any thread. Wait-free but may block for a small duration
// during initial scheduling of the submitted tasks.
//
// NOTE: it's possible for all work in the submission to complete prior to this
// function returning.
void iree_task_executor_submit(iree_task_executor_t* executor,
                               iree_task_submission_t* submission);

// Flushes any pending task batches for execution.
//
// Safe to call from any thread. Wait-free but may block for a small duration
// during initial scheduling of the submitted tasks.
//
// NOTE: due to races it's possible for new work to arrive from other threads
// after the flush has occurred but prior to this call returning.
void iree_task_executor_flush(iree_task_executor_t* executor);

// Donates the calling thread to the executor until either |wait_source|
// resolves or |timeout| is exceeded. Flushes any pending task batches prior
// to doing any work or waiting.
//
// If there are no tasks available then the calling thread will block as if
// iree_wait_source_wait_one had been used on |wait_source|. If tasks are ready
// then the caller will not block prior to starting to perform work on behalf of
// the executor.
//
// Donation is intended as an optimization to elide context switches when the
// caller would have waited anyway; now instead of performing a kernel wait and
// most certainly incurring a context switch the caller immediately begins
// taking work from the queue - likely even prior to any of the executor workers
// waking (assuming they were idle).
//
// Note that donation may not always be strictly a win: the caller may have an
// arbitrary thread affinity that may cause oversubscription of resources within
// the topology. This can cause additional contention for compute resources and
// increase kernel scheduling overhead as threads are swapped or migrated.
// Measure, measure, measure! If there is any IO that can be performed during
// the time that a caller would otherwise donate themselves to the executor that
// should always be preferred as should smaller computation (again to not
// oversubscribe resources). Treat donation as a hail mary to prevent a kernel
// wait and not something that will magically make things execute faster.
// Especially in large applications it's almost certainly better to do something
// useful with the calling thread (even if that's go to sleep).
//
// Safe to call from any thread (though bad to reentrantly call from workers).
iree_status_t iree_task_executor_donate_caller(iree_task_executor_t* executor,
                                               iree_wait_source_t wait_source,
                                               iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_EXECUTOR_H_
