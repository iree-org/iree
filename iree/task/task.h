// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_TASK_H_
#define IREE_TASK_TASK_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/task/affinity_set.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_task_list_t iree_task_list_t;
typedef struct iree_task_pool_t iree_task_pool_t;
typedef struct iree_task_scope_t iree_task_scope_t;
typedef struct iree_task_submission_t iree_task_submission_t;

//==============================================================================
// Task header for internal tracking
//==============================================================================

// Specifies the type of a task and how executors handle it.
enum iree_task_type_bits_t {
  // Task is a no-op (performs no work) and exists for flexibility.
  IREE_TASK_TYPE_NOP = 0u,

  // Task will synchronously call a function before continuing.
  IREE_TASK_TYPE_CALL = 1u,

  // Task exists only as a barrier to join/fork tasks and has no executable
  // payload.
  IREE_TASK_TYPE_BARRIER = 2u,

  // Task is a fence indicating that a certain point in the task graph has been
  // reached. All tasks prior to this fence (by way of happens-before
  // dependencies) are guaranteed to have retired.
  IREE_TASK_TYPE_FENCE = 3u,

  // Task is a wait on an external wait handle (fd, HANDLE, etc).
  // Executors will wait on the handle until it is signaled and meets the
  // specified condition prior to readying the dependent tasks.
  IREE_TASK_TYPE_WAIT = 4u,

  // Task is a 3D grid dispatch of zero or more tiles.
  // Dispatches are issued when ready by either being split into one shard per
  // worker that should process the dispatch.
  //
  // If IREE_TASK_FLAG_DISPATCH_INDIRECT is set then the dispatch reads the
  // workgroup count from a buffer immediately prior to fan-out instead of using
  // the values embedded in the task structure.
  //
  // After a dispatch has been issued the IREE_TASK_FLAG_DISPATCH_RETIRE flag is
  // set to indicate that when the dispatch becomes ready again it will be after
  // all shards have completed.
  IREE_TASK_TYPE_DISPATCH = 5u,

  // Task is one of potentially many shards processing a larger dispatch grid.
  // Each shard may have a preference as to which parts of grid it will focus
  // on but is able to otherwise steal any available region directly from the
  // shared dispatch coordination state. Shards retire once there are no more
  // tiles remaining in the dispatch grid.
  IREE_TASK_TYPE_DISPATCH_SHARD = 6u,
};
typedef uint8_t iree_task_type_t;

enum iree_task_flag_bits_t {
  IREE_TASK_FLAG_NONE = 0u,

  // The wait handle the task is specified to wait on has resolved and the task
  // can now be considered complete.
  IREE_TASK_FLAG_WAIT_COMPLETED = 1u << 0,

  // The workgroup count for the dispatch is provided by way of a pointer to a
  // list of 3 uint32_t values that will be sampled immediately prior to
  // issuing of the dispatch. The contents of the pointer can be safely modified
  // up until the last dependency has completed and the dispatch is about to be
  // issued.
  IREE_TASK_FLAG_DISPATCH_INDIRECT = 1u << 1,

  // The dispatch has been issued and the task is waiting for one or more
  // shards to complete. After they complete the dispatch will be readied and
  // can be retired.
  //
  // Though added by the executor after issuing a dispatch users can also set
  // this to indicate that all dispatch shards for a particular dispatch have
  // been statically scheduled. Executors will then skip issuing the dispatch
  // and instead wait until all shards complete, enabling IREE_TASK_TYPE_BARRIER
  // behavior but without an additional task as dispatches are still required
  // to store information for shards.
  IREE_TASK_FLAG_DISPATCH_RETIRE = 1u << 2,

  // An error occurred at or before the task and it has been aborted.
  // Aborted tasks may continue to execute if they're already in-flight but must
  // not begin execution after the flag has been set.
  //
  // The actual error that occurred is routed to the parent task scope as it
  // happens and may be available for querying before all tasks have been
  // cleaned up.
  IREE_TASK_FLAG_ABORTED = 1u << 3,
};
typedef uint16_t iree_task_flags_t;

typedef struct iree_task_t iree_task_t;

// A function called to cleanup tasks.
// Each task has its associated cleanup function called exactly once.
// The provided |status_code| indicates the execution status of the task prior
// to cleanup and will usually be IREE_STATUS_OK indicating the task was
// successfully issued or IREE_STATUS_ABORTED if the task was discard prior to
// issuing.
typedef void(IREE_API_PTR* iree_task_cleanup_fn_t)(
    iree_task_t* task, iree_status_code_t status_code);

// A task within the task system that runs on an executor.
// Tasks have an iree_task_type_t that defines which parameters are valid and
// how the executor is to treat the task. Dependency edges can be defined that
// determine the execution order of tasks within the executors.
struct iree_alignas(iree_max_align_t) iree_task_t {
  // Intrusive pointer used to store tasks within iree_task_list_t and
  // iree_atomic_task_list_t singly-linked lists. This must come first in the
  // structure so that it is at the appropriate alignment.
  iree_task_t* next_task;

  // The scope this task is attributed to. Errors with the task will be
  // propagated to the scope and errors in the scope will cause pending tasks to
  // be skipped.
  iree_task_scope_t* scope;

  // Optional function to call to cleanup the task on completion.
  // Will be called after the task has retired or if the task fails to issue
  // (dependency failed, etc).
  iree_task_cleanup_fn_t cleanup_fn;

  // Optional task that will be notified when the task completes.
  // The task will have its pending_dependency_count decremented and will be
  // readied for execution when the count reaches 0.
  iree_task_t* completion_task;

  // Specifies which workers will be used to execute this task.
  // Forked tasks will inherit their parent task affinity (possibly with some
  // task-dependent rules) to partition workloads across workers with knowledge
  // of the specific work being performed. For example, some dispatches can be
  // limited to run on certain microarchitectures that workers have affinity
  // with at the OS scheduler level (such as little.BIG topologies).
  iree_task_affinity_set_t affinity_set;

  // Total number of dependent tasks still outstanding. Decremented each time
  // a dependent task completes. The task is considered ready to execute when
  // this value reaches 0.
  iree_atomic_int32_t pending_dependency_count;

  // Optional pool the task should be returned to after it has resolved. If the
  // task was allocated as part of a larger data structure (embedded within
  // an arena for example) then this can be NULL to prevent the task system
  // from interfering.
  iree_task_pool_t* pool;

  // Specifies the type of the task and how the executor handles it.
  iree_task_type_t type;

  // Task-specific flag bits.
  iree_task_flags_t flags;
};
static_assert(offsetof(iree_task_t, next_task) == 0,
              "next_task intrusive pointer must be at offset 0");
static_assert(sizeof(iree_task_t) <= 64,
              "the task header greatly influences pool sizes due to alignment "
              "requirements and should be kept tiny");

// Initializes a task header with the given type.
// Must be called on all tasks to ensure proper dependency tracking and list
// state prior to enqueuing. Only the task header structure is initialized and
// any additional data as part of the wrapping task type must be initialized by
// the caller.
void iree_task_initialize(iree_task_type_t type, iree_task_scope_t* scope,
                          iree_task_t* out_task);

// Sets the optional function called when the task completes (whether successful
// or not). The cleanup function will receive a status indicating whether the
// cleanup is from expected execution as the task retires (IREE_STATUS_OK)
// or because it was aborted (IREE_STATUS_ABORTED).
void iree_task_set_cleanup_fn(iree_task_t* task,
                              iree_task_cleanup_fn_t cleanup_fn);

// Sets up a dependency edge from |task| to |completion_task| such that when
// |task| completes |completion_task| will be notified and have its
// pending_dependency_count decremented.
void iree_task_set_completion_task(iree_task_t* task,
                                   iree_task_t* completion_task);

// Returns true if the |task| is ready to execute immediately.
// Though this is safe to call from any thread the test may have false-negatives
// (ready tasks are not returned as ready) due to cross-thread synchronization
// latency. Note that tasks may yield themselves during execution and switch
// from ready to waiting (such as when an indirect dispatch needs to wait for
// all tiles to complete).
bool iree_task_is_ready(iree_task_t* task);

// Discards the task and any dependent tasks.
// Any dependent tasks that need to be discarded will be added to
// |discard_worklist| for the caller to continue discarding.
void iree_task_discard(iree_task_t* task, iree_task_list_t* discard_worklist);

//==============================================================================
// IREE_TASK_TYPE_NOP
//==============================================================================

// Task is a no-op (performs no work) and exists for flexibility.
// NOP tasks can be used to link together task lists from multiple threads
// where it may otherwise not be ideal to have heavy-weight concurrency
// structures. NOP tasks can also be useful for neutering another task type
// after it has already been recorded into a list such as when cancellations
// occur.
typedef iree_alignas(iree_max_align_t) struct {
  // Task header: implementation detail, do not use.
  iree_task_t header;
} iree_task_nop_t;

void iree_task_nop_initialize(iree_task_scope_t* scope,
                              iree_task_nop_t* out_task);

//==============================================================================
// IREE_TASK_TYPE_CALL
//==============================================================================

typedef iree_status_t(IREE_API_PTR* iree_task_call_closure_fn_t)(
    void* user_context, iree_task_t* task,
    iree_task_submission_t* pending_submission);

// A function closure representing the function to call and its arguments.
typedef struct iree_task_call_closure_t {
  // Function called per tile invocation.
  iree_task_call_closure_fn_t fn;

  // Opaque pointer to a user-provided data structure.
  // No lifetime management is performed by the task system and it is required
  // that users ensure that the memory referenced is live until after the task
  // has completed.
  void* user_context;

  // TODO(benvanik): cleanup function? right now assume arg is never freed.
} iree_task_call_closure_t;

// Binds a function pointer and the arguments it should be called with.
// If the arguments represent pointers they must remain live until the task
// has completed execution.
static inline iree_task_call_closure_t iree_task_make_call_closure(
    iree_task_call_closure_fn_t fn, void* user_context) {
  iree_task_call_closure_t closure = {fn, user_context};
  return closure;
}

// A task that will synchronously call a function from the executor and wait
// for it to complete before continuing.
//
// Memory referenced by closure arguments must be kept valid until the function
// executes (in general with the same lifetime as the task itself).
typedef iree_alignas(iree_max_align_t) struct {
  // Task header: implementation detail, do not use.
  iree_task_t header;

  // Function closure to call when the task is executed.
  iree_task_call_closure_t closure;

  // Resulting status from the call available once all nested tasks have
  // completed (or would have completed). It's possible for a call to nest
  // additional work under it and then return a failure; to ensure we don't
  // discard the root call while the nested tasks are still executing we set the
  // status here and wait for the nested tasks to complete. We'll try not to
  // issue work that was enqueued while the call was executing but it's possible
  // for work to come from other angles and we need to err on the side of
  // safety.
  iree_atomic_intptr_t status;
} iree_task_call_t;

void iree_task_call_initialize(iree_task_scope_t* scope,
                               iree_task_call_closure_t closure,
                               iree_task_call_t* out_task);

//==============================================================================
// IREE_TASK_TYPE_BARRIER
//==============================================================================

// A join point for fork/join-style scheduling.
// References a set of dependent tasks that will be notified and possibly
// readied when the barrier is reached.
//
// This allows for modeling one-to-many and many-to-many relationships. The base
// task dependency system only models one-to-one and should be used if possible
// to avoid the additional overhead of a barrier task both in memory and task
// indirection/queuing.
//
// Example:
//  * [A] -> Barrier -> [C, D]
//  - A executes
//  - Barrier is processed after A completes
//  - C and D execute concurrently (in any order)
//
//  * [A, B] -> Barrier -> [C, D]
//  - A and B execute concurrently (in any order)
//  - Barrier is processed after both A and B complete
//  - C and D execute concurrently
//
//  * [A] -> Barrier -> [B]
//  - Don't do this and use the base task dependency instead; it'll work, but
//    it's much better to avoid the additional barrier indirection when
//    possible.
typedef iree_alignas(iree_max_align_t) struct {
  // Task header: implementation detail, do not use.
  iree_task_t header;

  // Number of valid tasks in the dependent_tasks list.
  iree_host_size_t dependent_task_count;
  // [0-dependent_task_count] tasks that will be notified when the barrier is
  // reached. Each task will have its pending_dependency_count decremented and
  // when the count reaches 0 be added to the ready list.
  iree_task_t* const* dependent_tasks;
} iree_task_barrier_t;

void iree_task_barrier_initialize(iree_task_scope_t* scope,
                                  iree_host_size_t dependent_task_count,
                                  iree_task_t* const* dependent_tasks,
                                  iree_task_barrier_t* out_task);

void iree_task_barrier_initialize_empty(iree_task_scope_t* scope,
                                        iree_task_barrier_t* out_task);

void iree_task_barrier_set_dependent_tasks(
    iree_task_barrier_t* task, iree_host_size_t dependent_task_count,
    iree_task_t* const* dependent_tasks);

//==============================================================================
// IREE_TASK_TYPE_FENCE
//==============================================================================

// A fence indicating that a certain point in the task graph has been reached.
// All tasks prior to this fence (by way of happens-before dependencies) are
// guaranteed to have retired.
//
// When all of the dependencies of a fence have retired the fence will notify
// the parent scope of the task by decrementing the pending_submissions count
// and publishing an idle_notification if it was the last in-flight submission.
typedef iree_alignas(iree_max_align_t) struct {
  // Task header: implementation detail, do not use.
  iree_task_t header;

  // TODO(benvanik): user-defined fence data for semaphore signaling. Optional
  // wait_handle to signal?
} iree_task_fence_t;

void iree_task_fence_initialize(iree_task_scope_t* scope,
                                iree_task_fence_t* out_task);

//==============================================================================
// IREE_TASK_TYPE_WAIT
//==============================================================================

typedef struct iree_task_wait_t {
  // Task header: implementation detail, do not use.
  iree_task_t header;

  // The external wait handle that the task is waiting on.
  // TODO(benvanik): null handle for sleep.
  // TODO(benvanik): use a wait source with cached wait handle.
  // TODO(benvanik): multiple wait handles (ptr owned by outer wrapper task).
  iree_wait_handle_t wait_handle;

  // TODO(benvanik): deadline_ns.
  // TODO(benvanik): condition (possibly a closure to evaluate) ala condvar.
  // TODO(benvanik): whether a sleep.
} iree_task_wait_t;

void iree_task_wait_initialize(iree_task_scope_t* scope,
                               iree_wait_handle_t wait_handle,
                               iree_task_wait_t* out_task);

//==============================================================================
// IREE_TASK_TYPE_DISPATCH_* structures
//==============================================================================

// Statistics tracked across an entire dispatch operation.
// Each tile contributes to these statistics as they execute to provide an
// aggregate set of statistics that can be reported to tracing/user queries.
//
// We want to keep this structure relatively compact as it does add overhead.
// If statistics are used purely for interactive tracing then they can be
// piped directly to the tracing tool using IREE_TRACE_* macros. If the
// statistics are programmatically queried for benchmarks or reporting then
// they belong here where we can efficiently move them around.
//
// If we find ourselves with a lot of hardware-specific counters (vs more
// generic ones like 'l2 cache misses' or 'ipc') then we can sprinkle in some
// #ifdefs.
typedef struct iree_task_dispatch_statistics_t {
  // TODO(benvanik): statistics counters.
  // NOTE: each of these increases the command buffer storage requirements; we
  // should always guard these with IREE_STATISTICS_ENABLE.
  iree_atomic_int32_t reserved;
} iree_task_dispatch_statistics_t;

// Merges statistics from |source| to |target| atomically per-field.
// As each field is updated independently and in a relaxed memory order it's
// possible for statistics consumers to see a tear.
void iree_task_dispatch_statistics_merge(
    const iree_task_dispatch_statistics_t* source,
    iree_task_dispatch_statistics_t* target);

typedef struct iree_task_tile_storage_t {
  // TODO(benvanik): coroutine storage.
  // Ideally we'll be able to have a fixed coroutine storage size per dispatch
  // (via @llvm.coro.size) such that we can preallocate all of the storage for
  // a dispatch in one shot. If we need to do dynamic allocation we will need a
  // ringbuffer or other kind of pool to allocate from on-demand.
  uint32_t reserved;
} iree_task_tile_storage_t;

// Per-tile context provided to each dispatch function invocation in the grid.
// This information is unique to the tile being dispatched and may contain
// specific state about the calling thread/fiber/etc.
//
// If tile execution is suspended by hitting a coroutine suspend point then the
// coroutine state will be stored within the tile context until the tile is
// resumed.
typedef iree_alignas(iree_max_align_t) struct {
  // Workgroup ID for the current invocation.
  uint32_t workgroup_xyz[3];
  // Workgroup size for each invocation.
  uint32_t workgroup_size[3];
  // Total workgroup count for the task. Can be used in conjunction with the
  // per-invocation workgroup_xyz and workgroup_size to compute offsets/indices.
  uint32_t workgroup_count[3];
  // TODO(benvanik): workgroup index to amortize calculating linear offsets.
  // (like gl_GlobalInvocationID)

  // Tile-local memory that is pinned to each worker ensuring no cache
  // thrashing. Aligned to at least the natural pointer size of the machine.
  // Contents are (today) undefined upon entry.
  iree_byte_span_t local_memory;

  // Shared statistics counters for the dispatch shard.
  iree_task_dispatch_statistics_t* statistics;

  // TODO(benvanik): cpuid uarch.
  // TODO(benvanik): per-tile coroutine storage.
} iree_task_tile_context_t;

typedef struct iree_task_dispatch_t iree_task_dispatch_t;

//==============================================================================
// Dispatch function closures
//==============================================================================

typedef iree_status_t(IREE_API_PTR* iree_task_dispatch_closure_fn_t)(
    void* user_context, const iree_task_tile_context_t* tile_context,
    iree_task_submission_t* pending_submission);

// A function closure representing the function to call and its arguments.
typedef struct iree_task_dispatch_closure_t {
  // Function called per tile invocation.
  iree_task_dispatch_closure_fn_t fn;

  // User-defined argument passed to task functions during invocation.
  // Opaque pointer-sized values that could point to user data structures or
  // contain embedded values. No lifetime management is performed by the task
  // system and it is required that users ensure that the memory referenced is
  // live until after the task has completed.
  void* user_context;
} iree_task_dispatch_closure_t;

// Binds a function pointer and the arguments it should be called with.
// If the arguments represent pointers they must remain live until the task
// has completed execution.
static inline iree_task_dispatch_closure_t iree_task_make_dispatch_closure(
    iree_task_dispatch_closure_fn_t fn, void* user_context) {
  iree_task_dispatch_closure_t closure = {fn, user_context};
  return closure;
}

//==============================================================================
// IREE_TASK_TYPE_DISPATCH
//==============================================================================

// An execution request across a tiled grid.
// Dispatches are fork points where zero or more dispatch shard tasks are
// spawned and processed prior to joining again on the dispatch completion task.
//
// The total workgroup count defines the [x,y,z] extents of the dispatch grid.
// The count may either be embedded directly into the dispatch or provided as a
// pointer to the workgroup_count[3] that will be read immediately prior to
// forking. If any dimension of the workgroup count is zero then the dispatch is
// skipped and the completion task will be readied immediately.
//
// Example:
//   dispatch([5, 1, 1])
//     forked into shards based on affinity/scheduling parameters:
//     -> dispatch_shard for core 0, processes [0-1, 1, 1]
//     -> dispatch_shard for core 1, processes [2-3, 1, 1]
//     -> dispatch_shard for core 2, processes [4-5, 1, 1]
//   completion_task run after all shards complete
typedef iree_alignas(iree_max_align_t) struct iree_task_dispatch_t {
  // Task header: implementation detail, do not use.
  iree_task_t header;

  // Function closure to call per tile.
  iree_task_dispatch_closure_t closure;

  // Workgroup size for each invocation. Passed on to tiles without
  // modification and not used for scheduling.
  uint32_t workgroup_size[3];

  // 3D workgroup count used to tile the dispatch.
  // [1,1,1] specifies single invocation of the function. A value of 0 in
  // any dimension will skip execution of the function.
  union {
    // Embedded immutable 3D workgroup count value.
    uint32_t value[3];
    // Pointer to the uint32_t[3] containing the 3D workgroup count.
    // Sampled immediately prior to execution.
    const uint32_t* ptr;
  } workgroup_count;

  // Optional transient shared memory size in bytes to allocate and pass into
  // the iree_task_tile_context_t::local_memory of each invocation of the
  // dispatch closure.
  uint32_t local_memory_size;

  // Resulting status from the dispatch available once all workgroups have
  // completed (or would have completed). If multiple shards processing the
  // workgroups hit an error the first will be taken and the result ignored. A
  // dispatch with a non-ok status will mark the parent task scope as failing
  // when it retires.
  iree_atomic_intptr_t status;

  // Statistics storage used for aggregating counters across all shards.
  iree_task_dispatch_statistics_t statistics;

  // The total number of tiles in the dispatch bounding tile_index.
  uint32_t tile_count;

  // Maximum number of tiles to fetch per tile reservation from the grid.
  // Bounded by IREE_TASK_DISPATCH_MAX_TILES_PER_SHARD_RESERVATION and a
  // reasonable number chosen based on the tile and shard counts.
  uint32_t tiles_per_reservation;

  // The tail tile index; the next reservation will start from here.
  // This is used by shards to slice off the work to perform in their inner
  // loop. Ideally we'd have no destructive interference with other shared data
  // in this structure but the shared parts (status/statistics) are updated once
  // per shard instead of once per slice and are less of a concern.
  iree_atomic_int32_t tile_index;

  // Incrementing process-lifetime dispatch identifier.
  IREE_TRACE(int64_t dispatch_id;)
} iree_task_dispatch_t;

void iree_task_dispatch_initialize(iree_task_scope_t* scope,
                                   iree_task_dispatch_closure_t closure,
                                   const uint32_t workgroup_size[3],
                                   const uint32_t workgroup_count[3],
                                   iree_task_dispatch_t* out_task);

void iree_task_dispatch_initialize_indirect(
    iree_task_scope_t* scope, iree_task_dispatch_closure_t closure,
    const uint32_t workgroup_size[3], const uint32_t* workgroup_count_ptr,
    iree_task_dispatch_t* out_task);

//==============================================================================
// IREE_TASK_TYPE_DISPATCH_SHARD
//==============================================================================

typedef iree_alignas(iree_max_align_t) struct {
  // Task header: implementation detail, do not use.
  iree_task_t header;

  // NOTE: the parent dispatch task this shard is applied to is in the
  // header.completion_task field.
} iree_task_dispatch_shard_t;

void iree_task_dispatch_shard_initialize(iree_task_dispatch_t* dispatch_task,
                                         iree_task_dispatch_shard_t* out_task);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_TASK_H_
