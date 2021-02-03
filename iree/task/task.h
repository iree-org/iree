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

#ifndef IREE_TASK_TASK_H_
#define IREE_TASK_TASK_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/synchronization.h"
#include "iree/task/affinity_set.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_task_list_s iree_task_list_t;
typedef struct iree_task_pool_s iree_task_pool_t;
typedef struct iree_task_scope_s iree_task_scope_t;
typedef struct iree_task_submission_s iree_task_submission_t;

//==============================================================================
// Task header for internal tracking
//==============================================================================

// Specifies the type of a task and how executors handle it.
enum iree_task_type_e {
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
  // Dispatches are issued when ready by either being split into zero or more
  // slices with one or more tiles each based on the workgroup count or one
  // shard per worker that should process the dispatch.
  //
  // If IREE_TASK_FLAG_DISPATCH_INDIRECT is set then the dispatch reads the
  // workgroup count from a buffer immediately prior to fan-out instead of using
  // the values embedded in the task structure.
  //
  // After a dispatch has been issued the IREE_TASK_FLAG_DISPATCH_RETIRE flag is
  // set to indicate that when the dispatch becomes ready again it will be after
  // all slices have completed.
  IREE_TASK_TYPE_DISPATCH = 5u,

  // Task is a slice of a larger contiguous dispatch tile range. The full
  // dispatch will be sliced into zero or more slices and each slice will be
  // posted to a particular worker for executiion. If work progresses unevenly
  // then entire slices will be stolen across workers to balance out the timing.
  // Slices retire once they have completed the tiles assigned to them.
  IREE_TASK_TYPE_DISPATCH_SLICE = 6u,

  // Task is one of potentially many shards processing a larger dispatch grid.
  // Each shard may have a preference as to which parts of grid it will focus
  // on but is able to otherwise steal any available region directly from the
  // shared dispatch coordination state. Shards retire once there are no more
  // tiles remaining in the dispatch grid.
  IREE_TASK_TYPE_DISPATCH_SHARD = 7u,
};
typedef uint8_t iree_task_type_t;

enum iree_task_flags_e {
  // The wait handle the task is specified to wait on has resolved and the task
  // can now be considered complete.
  IREE_TASK_FLAG_WAIT_COMPLETED = 1u << 0,

  // The workgroup count for the dispatch is provided by way of a pointer to a
  // list of 3 uint32_t values that will be sampled immediately prior to
  // issuing of the dispatch. The contents of the pointer can be safely modified
  // up until the last dependency has completed and the dispatch is about to be
  // issued.
  IREE_TASK_FLAG_DISPATCH_INDIRECT = 1u << 1,

  // The dispatch should be sliced across workers via the low-contention
  // IREE_TASK_TYPE_DISPATCH_SLICE mechanism. This moves the dispatch overhead
  // to the time when the grid is sliced for a savings during when the grid is
  // executed.
  IREE_TASK_FLAG_DISPATCH_SLICED = 1u << 2,

  // The dispatch has been issued and the task is waiting for one or more
  // slices to complete. After they complete the dispatch will be readied and
  // can be retired.
  //
  // Though added by the executor after issuing a dispatch users can also set
  // this to indicate that all dispatch slices for a particular dispatch have
  // been statically scheduled. Executors will then skip issuing the dispatch
  // and instead wait until all slices complete, enabling IREE_TASK_TYPE_BARRIER
  // behavior but without an additional task as dispatches are still required
  // to store information for slices.
  IREE_TASK_FLAG_DISPATCH_RETIRE = 1u << 3,
};
typedef uint16_t iree_task_flags_t;

typedef struct iree_task_s iree_task_t;

// A function called to cleanup tasks.
// The provided |status| is unowned and must be cloned if used beyond the scope
// of the cleanup function (such as when stored for later usage).
typedef void(IREE_API_PTR* iree_task_cleanup_fn_t)(iree_task_t* task,
                                                   iree_status_t status);

// A task within the task system that runs on an executor.
// Tasks have an iree_task_type_t that defines which parameters are valid and
// how the executor is to treat the task. Dependency edges can be defined that
// determine the execution order of tasks within the executors.
struct iree_alignas(iree_max_align_t) iree_task_s {
  // Instrusive pointer used to store tasks within iree_task_list_t and
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

// Initializes a task header with the given type.
// Must be called on all tasks to ensure proper dependency tracking and list
// state prior to enqueuing. Only the task header structure is initialized and
// any additional data as part of the wrapping task type must be initialized by
// the caller.
void iree_task_initialize(iree_task_type_t type, iree_task_scope_t* scope,
                          iree_task_t* out_task);

// Sets the optional function called when the task completes (whether successful
// or not).
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
    uintptr_t user_context, iree_task_t* task,
    iree_task_submission_t* pending_submission);

// A function closure representing the function to call and its arguments.
typedef struct {
  // Function called per tile invocation.
  iree_task_call_closure_fn_t fn;

  // User-defined argument passed to task functions during invocation.
  // Opaque pointer-sized values that could point to user data structures or
  // contain embedded values. No lifetime management is performed by the task
  // system and it is required that users ensure that the memory referenced is
  // live until after the task has completed.
  uintptr_t user_context;

  // TODO(benvanik): cleanup function? right now assume arg is never freed.
} iree_task_call_closure_t;

// Binds a function pointer and the arguments it should be called with.
// If the arguments represent pointers they must remain live until the task
// has completed execution.
static inline iree_task_call_closure_t iree_task_make_call_closure(
    iree_task_call_closure_fn_t fn, uintptr_t user_context) {
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

typedef struct {
  // Task header: implementation detail, do not use.
  iree_task_t header;

  // The external wait handle that the task is waiting on.
  // TODO(benvanik): multiple wait handles.
  iree_wait_handle_t wait_handle;

  // TODO(benvanik): deadline_ns.
  // TODO(benvanik): condition (possibly a closure to evaluate) ala condvar.
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
typedef struct {
  // TODO(benvanik): statistics counters.
  iree_atomic_int32_t reserved;
} iree_task_dispatch_statistics_t;

// Merges statistics from |source| to |target| atomically per-field.
// As each field is updated independently and in a relaxed memory order it's
// possible for statistics consumers to see a tear.
void iree_task_dispatch_statistics_merge(
    const iree_task_dispatch_statistics_t* source,
    iree_task_dispatch_statistics_t* target);

typedef struct {
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

  // Incoherent memory shared across all invocations of the task.
  // Aligned to at least the natural pointer size of the machine. Functions must
  // use atomic operations to ensure proper memory ordering.
  iree_byte_span_t shared_memory;

  // Shared statistics counters for the dispatch slice.
  iree_task_dispatch_statistics_t* statistics;

  // TODO(benvanik): cpuid uarch.
  // TODO(benvanik): per-tile coroutine storage.
} iree_task_tile_context_t;

typedef struct iree_task_dispatch_s iree_task_dispatch_t;

// Shared state for all shards processing a dispatch.
typedef iree_alignas(iree_max_align_t) struct {
  // Direct reference to the parent dispatch that all shards are processing.
  iree_task_dispatch_t* dispatch_task;

  // The tail tile index; the next reservation will start from here.
  iree_atomic_int32_t tile_index;

  // The total number of tiles in the dispatch bounding tile_index.
  uint32_t tile_count;

  // Maximum number of tiles to fetch per tile reservation from the grid.
  // Bounded by IREE_TASK_DISPATCH_MAX_TILES_PER_SHARD_RESERVATION and a
  // reasonable number chosen based on the tile and shard counts.
  uint32_t tiles_per_reservation;

  // Total workgroup count for the task. Can be used in conjunction with the
  // per-invocation workgroup_xyz and workgroup_size to compute offsets/indices.
  uint32_t workgroup_count[3];

  // Incoherent memory shared across all invocations of the task.
  // Aligned to at least the natural pointer size of the machine. Functions must
  // use atomic operations to ensure proper memory ordering.
  iree_byte_span_t shared_memory;
} iree_task_dispatch_shard_state_t;

//==============================================================================
// Dispatch function closures
//==============================================================================

typedef iree_status_t(IREE_API_PTR* iree_task_dispatch_closure_fn_t)(
    uintptr_t user_context, const iree_task_tile_context_t* tile_context,
    iree_task_submission_t* pending_submission);

// A function closure representing the function to call and its arguments.
typedef struct {
  // Function called per tile invocation.
  iree_task_dispatch_closure_fn_t fn;

  // User-defined argument passed to task functions during invocation.
  // Opaque pointer-sized values that could point to user data structures or
  // contain embedded values. No lifetime management is performed by the task
  // system and it is required that users ensure that the memory referenced is
  // live until after the task has completed.
  uintptr_t user_context;
} iree_task_dispatch_closure_t;

// Binds a function pointer and the arguments it should be called with.
// If the arguments represent pointers they must remain live until the task
// has completed execution.
static inline iree_task_dispatch_closure_t iree_task_make_dispatch_closure(
    iree_task_dispatch_closure_fn_t fn, uintptr_t user_context) {
  iree_task_dispatch_closure_t closure = {fn, user_context};
  return closure;
}

//==============================================================================
// IREE_TASK_TYPE_DISPATCH
//==============================================================================

// An execution request across a tiled grid.
// Dispatches are fork points where zero or more dispatch slice tasks are
// spawned and processed prior to joining again on the dispatch completion task.
//
// The total workgroup count indicates the [x,y,z] extents of the dispatch grid.
// The count may either be embedded directly into the dispatch or provided as a
// pointer to the workgroup_count[3] that will be read immediately prior to
// forking. If any dimension of the workgroup count is zero then the dispatch is
// skipped and the completion task will be readied immediately.
//
// Example:
//   dispatch([5, 1, 1])
//     forked into slices based on affinity/scheduling parameters:
//     -> dispatch_slice([0-1, 1, 1])
//     -> dispatch_slice([2-3, 1, 1])
//     -> dispatch_slice([4-5, 1, 1])
//   completion_task run after all slices complete
typedef iree_alignas(iree_max_align_t) struct iree_task_dispatch_s {
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

  // Optional transient shared memory size to allocate and pass into the
  // iree_task_context_t::shared_memory of each invocation of the task
  // closure.
  iree_host_size_t shared_memory_size;

  // Statistics storage used for aggregating counters across all slices.
  iree_task_dispatch_statistics_t statistics;

  // Shared state across all slices/shards/etc.
  // Stored once per dispatch and then referenced by all subtasks.
  union {
    iree_task_dispatch_shard_state_t shard_state;
  } shared;
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
// IREE_TASK_TYPE_DISPATCH_SLICE
//==============================================================================

// TODO(benvanik): per-region dependencies (allow slices to execute directly
// across dispatches).

// A slice of tiles within a larger dispatch grid.
// These tasks are designed to be synthesized by the task system when processing
// a dispatch task. Based on the workgroup count, affinity settings, and
// available executor threads zero or more slices are enqueued, executed, and
// retired as part of the complete dispatch task. The dispatch is only
// considered completed and subsquent tasks readied once all slices are
// complete.
//
// Slices aggregate statistics from all tiles within them and then upon
// completion merge those into the shared dispatch statistics. As slices may
// suspend and resume several times the dispatch-level statistics should only be
// read once all slices have completed fully.
//
// In general slices represent a contiguous range of tiles along the most
// rapidly changing dimension (x, then y, then z). This ensures that we at least
// give the opportunity for cache locality to the tiles as they are processed.
// If work stealing is enabled then slices may shed their trailing tiles to
// other threads that have completed all of their work (at a cost of power vs.
// potential latency savings).
typedef iree_alignas(iree_max_align_t) struct {
  // Task header: implementation detail, do not use.
  iree_task_t header;

  // NOTE: the following fields are mostly replicated from iree_task_dispatch_t.
  // This removes the need for touching the dispatch struct when beginning a
  // tile which would likely be a cache miss as we fan out to other cores.

  // Function closure to call per tile (same as the closure in the dispatch).
  iree_task_dispatch_closure_t closure;

  // Base workgroup ID for the slice range.
  uint32_t workgroup_base[3];
  // Total count of tiles within the slice range.
  uint32_t workgroup_range[3];

  // Workgroup size for each invocation.
  uint32_t workgroup_size[3];
  // Total workgroup count for the task. Can be used in conjunction with the
  // per-invocation workgroup_xyz and workgroup_size to compute offsets/indices.
  uint32_t workgroup_count[3];

  // Incoherent memory shared across all invocations of the task.
  // Aligned to at least the natural pointer size of the machine. Functions must
  // use atomic operations to ensure proper memory ordering.
  iree_byte_span_t shared_memory;

  // Shared statistics counters for the entire dispatch. References the storage
  // held in the parent iree_task_dispatch_t.
  iree_task_dispatch_statistics_t* dispatch_statistics;
  // Statistics just for this single slice. The statistics will be added to the
  // dispatch_statistics after the slice completes to prevent excessive
  // contention on the shared dispatch statistics across multiple threads.
  iree_task_dispatch_statistics_t slice_statistics;

  // Per-tile initialized coroutine storage for all tiles in the range
  // initialized as each tile begins execution.
  // TODO(benvanik): coroutine storage as iree_task_tile_storage_t.
} iree_task_dispatch_slice_t;

// TODO(benvanik): document initialize() for slice pre-planning/embeddeding.
// This would be useful to reduce latency when the number of slices is small
// (~<5) as the dispatch wouldn't need to be issued. This can also be used to
// implement per-region dependencies as direct slice->slice deps vs. fork/join
// dispatch->dispatch deps. Show how IREE_TASK_FLAG_DISPATCH_RETIRE is set.
void iree_task_dispatch_slice_initialize(iree_task_dispatch_t* dispatch_task,
                                         const uint32_t workgroup_base[3],
                                         const uint32_t workgroup_range[3],
                                         const uint32_t workgroup_count[3],
                                         iree_task_dispatch_slice_t* out_task);

//==============================================================================
// IREE_TASK_TYPE_DISPATCH_SHARD
//==============================================================================

typedef iree_alignas(iree_max_align_t) struct {
  // Task header: implementation detail, do not use.
  iree_task_t header;

  // Active dispatch progress shared across all shards.
  // Each shard will be read/modify/writing this and there's likely to be
  // contention.
  iree_task_dispatch_shard_state_t* shared_state;
} iree_task_dispatch_shard_t;

void iree_task_dispatch_shard_initialize(
    iree_task_dispatch_t* dispatch_task,
    iree_task_dispatch_shard_state_t* shared_state,
    iree_task_dispatch_shard_t* out_task);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_TASK_H_
