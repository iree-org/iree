// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Task queue: two persistent processes — a budget-1 control process and a
// budget-N compute process — that cooperatively execute queue operations.
//
// Submissions flow through semaphore waits into the ready list. The budget-1
// control process pops operations and handles them by type: barriers, host
// calls, and allocations are handled inline; command buffers are filled into
// recording items and pushed to the compute process's pending list.
//
// The budget-N compute process drains recording items cooperatively across
// all workers via the block processor. It occupies a single compute slot
// for the queue's lifetime. Per-recording two-phase completion ensures
// semaphores are signaled eagerly while resources stay alive until all
// workers have exited drain.
//
// Operations are arena-allocated at submit time and freed by the completion
// callback. No per-submission task allocations at issue time.

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_TASK_QUEUE_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_TASK_QUEUE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/block_processor.h"
#include "iree/task/executor.h"
#include "iree/task/process.h"
#include "iree/task/scope.h"
#include "iree/task/tuning.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A single command buffer submitted to a device queue.
// All of the wait semaphores must reach or exceed the given payload values
// prior to the batch beginning execution. Only after all commands have
// completed will the signal semaphores be updated to the provided payload
// values.
typedef struct iree_hal_task_submission_batch_t {
  // Semaphores to wait on prior to executing any command buffer.
  iree_hal_semaphore_list_t wait_semaphores;

  // Command buffer to execute and optional binding table.
  iree_hal_command_buffer_t* command_buffer;
  iree_hal_buffer_binding_table_t binding_table;

  // Semaphores to signal once all command buffers have completed execution.
  iree_hal_semaphore_list_t signal_semaphores;
} iree_hal_task_submission_batch_t;

typedef struct iree_async_file_t iree_async_file_t;
typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;
typedef struct iree_hal_resource_set_t iree_hal_resource_set_t;
typedef struct iree_hal_task_queue_t iree_hal_task_queue_t;

//===----------------------------------------------------------------------===//
// Queue operation (ready list entry)
//===----------------------------------------------------------------------===//

// Type of operation queued for execution.
typedef enum iree_hal_task_queue_op_type_e {
  IREE_HAL_TASK_QUEUE_OP_COMMANDS,
  IREE_HAL_TASK_QUEUE_OP_BARRIER,
  IREE_HAL_TASK_QUEUE_OP_HOST_CALL,
  IREE_HAL_TASK_QUEUE_OP_ALLOCA,
  IREE_HAL_TASK_QUEUE_OP_DEALLOCA,
  IREE_HAL_TASK_QUEUE_OP_READ,
  IREE_HAL_TASK_QUEUE_OP_WRITE,
  IREE_HAL_TASK_QUEUE_OP_FILL,
  IREE_HAL_TASK_QUEUE_OP_COPY,
  IREE_HAL_TASK_QUEUE_OP_UPDATE,
  IREE_HAL_TASK_QUEUE_OP_DISPATCH,
} iree_hal_task_queue_op_type_t;

typedef struct iree_hal_task_queue_op_t iree_hal_task_queue_op_t;

#if !defined(NDEBUG)
// Debug-only metadata attached to one queue operation.
typedef struct iree_hal_task_queue_op_debug_state_t {
  // Monotonic ordinal assigned at allocation time. Used only for queue dumps
  // when diagnosing lost-submit/lost-completion bugs.
  uint64_t ordinal;
} iree_hal_task_queue_op_debug_state_t;
#endif  // !defined(NDEBUG)

// A single queued operation. Arena-allocated; the arena is freed when the
// operation completes (barriers/host calls in drain, command buffers in
// the CB process completion callback).
//
// The slist_next field must be at a stable offset for the MPSC ready list.
struct iree_hal_task_queue_op_t {
  // Intrusive slist node for the ready list.
  iree_atomic_slist_intrusive_ptr_t slist_next;

  // Operation type.
  iree_hal_task_queue_op_type_t type;

  // Arena owning this operation and all transient allocations (semaphore
  // list storage, wait entries, binding table copies). The arena is
  // deinitialized when the operation completes.
  iree_arena_allocator_t arena;

  // Scope for begin/end lifecycle tracking. The matching scope_begin is
  // called at submit time; scope_end is called when the operation completes.
  iree_task_scope_t* scope;

  // Semaphores to signal on completion.
  iree_hal_semaphore_list_t signal_semaphores;

  // Resources retained until all have retired (command buffers, binding
  // table buffers, etc.). Allocated from the small block pool.
  iree_hal_resource_set_t* resource_set;

  // Frontier tracking context. On completion, the operation atomically
  // increments *epoch_counter to get a fresh epoch and advances the tracker.
  // NULL frontier_tracker disables frontier advancement.
  iree_async_frontier_tracker_t* frontier_tracker;
  iree_async_axis_t axis;
  iree_atomic_int64_t* epoch_counter;

  // Outstanding semaphore wait count. Atomically decremented by semaphore
  // timepoint callbacks. When this reaches zero, the operation is pushed
  // to the ready list. Initialized to the number of unsatisfied waits.
  iree_atomic_int32_t wait_count;

  // Back-pointer to the owning queue. Used by semaphore wait callbacks to
  // push to the ready list and schedule the queue process when all waits
  // are satisfied. The queue outlives all operations (deinitialize waits
  // for scope idle).
  iree_hal_task_queue_t* queue;

#if !defined(NDEBUG)
  // Debug-only metadata omitted from release builds.
  iree_hal_task_queue_op_debug_state_t debug;
#endif  // !defined(NDEBUG)

  // First error encountered from a failed semaphore wait. Set via CAS —
  // only the first error wins. Checked on the last wait_count decrement
  // to decide between pushing to the ready list or failing the operation.
  iree_atomic_intptr_t error_status;

  // Type-specific data.
  union {
    struct {
      iree_hal_command_buffer_t* command_buffer;
      iree_hal_buffer_binding_table_t binding_table;
      // SCOPED buffer mappings for binding table resolution. Arena-allocated,
      // indexed 1:1 with binding_table entries. Unmapped in op_destroy before
      // the resource_set releases buffers. NULL when binding_table is empty.
      iree_hal_buffer_mapping_t* binding_mappings;
    } commands;
    struct {
      iree_hal_device_t* device;
      iree_hal_queue_affinity_t queue_affinity;
      iree_hal_host_call_t call;
      uint64_t args[4];
      iree_hal_host_call_flags_t flags;
    } host_call;
    struct {
      iree_hal_buffer_t* transient_buffer;
    } alloca;
    struct {
      iree_hal_buffer_t* transient_buffer;
    } dealloca;
    struct {
      iree_hal_file_t* hal_file;
      iree_async_file_t* async_file;  // NULL = sync fallback via hal_file.
      uint64_t file_offset;
      iree_hal_buffer_t* buffer;
      iree_device_size_t buffer_offset;
      iree_device_size_t length;
    } read;
    struct {
      iree_hal_file_t* hal_file;
      iree_async_file_t* async_file;  // NULL = sync fallback via hal_file.
      uint64_t file_offset;
      iree_hal_buffer_t* buffer;
      iree_device_size_t buffer_offset;
      iree_device_size_t length;
    } write;
    struct {
      iree_hal_buffer_t* target_buffer;
      iree_device_size_t target_offset;
      iree_device_size_t length;
      uint8_t pattern[4];
      uint8_t pattern_length;
    } fill;
    struct {
      iree_hal_buffer_t* source_buffer;
      iree_device_size_t source_offset;
      iree_hal_buffer_t* target_buffer;
      iree_device_size_t target_offset;
      iree_device_size_t length;
    } copy;
    struct {
      iree_hal_buffer_t* target_buffer;
      iree_device_size_t target_offset;
      iree_device_size_t length;
      // Source data arena-allocated (pointer into operation arena).
      const void* source_data;
    } update;
    struct {
      iree_hal_executable_t* executable;
      iree_hal_executable_export_ordinal_t export_ordinal;
      iree_hal_dispatch_config_t config;
      // Constants arena-allocated (pointer into operation arena).
      const uint32_t* constants;
      uint16_t constant_count;
      // Bindings arena-allocated (pointer into operation arena).
      const iree_hal_buffer_ref_t* bindings;
      iree_host_size_t binding_count;
      iree_hal_dispatch_flags_t flags;
    } dispatch;
  };
};

// Typed MPSC slist for the ready list.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_hal_task_queue_op,
                                iree_hal_task_queue_op_t,
                                offsetof(iree_hal_task_queue_op_t, slist_next));

//===----------------------------------------------------------------------===//
// Compute recording items (pool-managed)
//===----------------------------------------------------------------------===//

// Initial pool size for compute recording items. The pool grows dynamically
// when all items are in-flight. 16 covers typical pipeline depths (10+
// concurrent recordings between submission and final release).
#define IREE_HAL_TASK_QUEUE_COMPUTE_INITIAL_POOL_SIZE 16

typedef struct iree_hal_task_queue_compute_item_t
    iree_hal_task_queue_compute_item_t;

// A single recording being drained by the compute process. Arena-allocated
// with a trailing worker_states[] FAM. Items cycle through:
//   free_pool → (budget-1 fills) → pending → (compute installs) → current
//   → (completion + release) → free_pool
//
// Each item holds the block processor context and per-worker drain state for
// one command buffer recording. The low-bit drainer count in item->drainers is
// per-recording (separate from the compute slot's per-process active_drainers):
// the compute process occupies one slot persistently, but recordings flow
// through it sequentially.
//
// Two-phase lifecycle per recording:
//   Eager close:       First worker to observe completed=true sets CLOSED and
//                      installs the next pending recording (or NULL) in
//                      compute_current so unrelated work can start
//                      immediately.
//   Final release:     Last worker to leave after CLOSED is set claims
//                      RELEASE_CLAIMED, consumes the processor result exactly
//                      once, completes/fails the operation, tears down the
//                      processor context and any item-owned recording blocks,
//                      bumps the generation, and returns the item to the free
//                      pool.
struct iree_hal_task_queue_compute_item_t {
  // Intrusive slist node for the pending list and free pool.
  iree_atomic_slist_intrusive_ptr_t slist_next;

  // Linked list of all allocated items (for shutdown cleanup enumeration).
  // Written once at allocation, read only during shutdown.
  iree_hal_task_queue_compute_item_t* next_allocated;

  // Combined generation + drainer count + release/closed flags in one 64-bit
  // atomic. This replaces three separate fields (generation, active_drainers,
  // release_pending) to eliminate the TOCTOU race between checking the
  // drainer count and setting the completion flag.
  //
  //   bits 63-32: generation (monotonic, ABA prevention for recycled items)
  //   bit 31:     CLOSED flag (set when recording completes)
  //   bit 30:     RELEASE_CLAIMED flag (one worker owns final cleanup)
  //   bits 29-0:  active drainer count
  //
  // Protocol (mirrors the compute slot protocol in worker.c):
  //   Entry:  fetch_add(1). If (int32_t)prev < 0 → CLOSED, bail.
  //           Snapshot the item's generation from item->drainers before
  //           registering, then verify fetch_add observed that same
  //           generation. Re-check queue->compute_current_revision and
  //           queue->compute_current after registering to reject recycled
  //           same-address items.
  //   Close:  fetch_or(CLOSED_BIT). First to set publishes the next item.
  //   Exit:   fetch_sub(1). If prev == (gen | CLOSED | 1), try to CAS
  //           gen|CLOSED → gen|CLOSED|RELEASE_CLAIMED. The winner performs
  //           final cleanup and operation completion/failure.
  //   Reset:  CAS(gen|CLOSED|RELEASE_CLAIMED → next_gen|0) during final
  //           cleanup, waiting for late CLOSED bailers to drain first.
  iree_atomic_int64_t drainers;
#define IREE_HAL_TASK_QUEUE_ITEM_CLOSED_BIT ((int64_t)(uint32_t)INT32_MIN)
#define IREE_HAL_TASK_QUEUE_ITEM_RELEASE_CLAIMED_BIT \
  ((int64_t)(uint32_t)(1u << 30))
#define IREE_HAL_TASK_QUEUE_ITEM_GEN_INCREMENT ((int64_t)1 << 32)

  // Block processor execution context. Separately allocated with cache-line
  // alignment by context_allocate; freed in the final release path. NULL for
  // no-block recordings, which complete immediately through the null-safe
  // processor drain path.
  iree_hal_cmd_block_processor_context_t* processor_context;

  // Number of workers participating in draining this recording.
  uint32_t worker_count;

  // Back-pointer to the queue operation that submitted this recording.
  // Owned by the item from fill time until the final release worker consumes
  // the processor result and completes/fails the operation. Keeping the
  // operation alive until the last drainer exits preserves its resource_set
  // and scope without a separate ownership handoff.
  iree_hal_task_queue_op_t* operation;

  // Allocator used to free the processor context. Snapshotted at fill time so
  // the final release path doesn't chase through queue pointers that
  // may be destroyed during shutdown.
  iree_allocator_t host_allocator;

  // For queue-built recordings (not from a command buffer): the recording
  // whose blocks must be released in the final release path. For command
  // buffer recordings, first_block is NULL and the command buffer's recording
  // remains live through item->operation->resource_set until final release.
  iree_hal_cmd_block_recording_t recording;

  // Per-worker state for block processor drain calls. Each worker maintains
  // a block_sequence counter to detect block transitions. Zero-initialized
  // when the item is filled. Trailing FAM sized to worker_count at allocation.
  iree_hal_cmd_block_processor_worker_state_t worker_states[];
};

// Typed MPSC slist for the compute pending and free pool lists.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_hal_task_queue_compute_item,
                                iree_hal_task_queue_compute_item_t,
                                offsetof(iree_hal_task_queue_compute_item_t,
                                         slist_next));

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_t
//===----------------------------------------------------------------------===//

#if !defined(NDEBUG)
// Debug-only queue operation counters. These are best-effort monotonic
// snapshots for queue dumps and are not part of queue correctness.
typedef struct iree_hal_task_queue_debug_state_t {
  iree_atomic_int64_t next_operation_ordinal;
  iree_atomic_int64_t submitted_operation_count;
  iree_atomic_int64_t completed_operation_count;
  iree_atomic_int64_t failed_operation_count;
  iree_atomic_int64_t destroyed_ok_operation_count;
  iree_atomic_int64_t destroyed_failed_operation_count;
  iree_atomic_int64_t last_submitted_operation_ordinal;
  iree_atomic_int64_t last_completed_operation_ordinal;
  iree_atomic_int64_t last_failed_operation_ordinal;
  iree_atomic_int64_t last_destroyed_operation_ordinal;
} iree_hal_task_queue_debug_state_t;
#endif  // !defined(NDEBUG)

struct iree_hal_task_queue_t {
  // Affinity mask this queue processes.
  iree_hal_queue_affinity_t affinity;

  // Shared executor that the queue submits processes to.
  iree_task_executor_t* executor;

  // Proactor for async I/O operations on this queue. Borrowed from the
  // device's proactor pool — selected at device creation time based on the
  // executor's NUMA node for NUMA-correct I/O. Valid as long as the device
  // (which retains the proactor pool) is alive.
  iree_async_proactor_t* proactor;

  // Shared frontier tracker for cross-device causal ordering. When non-NULL,
  // the queue can perform domination checks on submission to skip proactor-
  // driven semaphore waits when all predecessors are already enqueued.
  // Borrowed from the device — valid as long as the device is alive.
  iree_async_frontier_tracker_t* frontier_tracker;

  // This queue's axis identity in the frontier system.
  // Derived from the device's base_axis + queue_index at initialization.
  iree_async_axis_t axis;

  // Monotonic epoch counter for frontier advancement. Incremented at
  // COMPLETION time (not submit time) because local_task is out-of-order:
  // operations complete on arbitrary threads in arbitrary order. Completion-
  // time assignment guarantees monotonic epoch progression without head-of-
  // line blocking. The epoch means "this many completions have occurred on
  // this queue."
  iree_atomic_int64_t epoch;

  // Length (in bytes) at which queue-level fill/copy operations route through
  // the block processor framework instead of executing as a direct memcpy in
  // the queue drain thread. See iree_hal_task_device_params_t for details.
  iree_device_size_t inline_transfer_threshold;

  // Shared block pool for allocating submission transients.
  iree_arena_block_pool_t* small_block_pool;
  // Shared block pool for large allocations (command buffers/etc).
  iree_arena_block_pool_t* large_block_pool;

  // Device allocator used for transient allocations/tracking.
  iree_hal_allocator_t* device_allocator;

  // Scope used for all operations in the queue.
  // This allows for easy waits on all outstanding queue operations as well as
  // differentiation of operations within the executor.
  iree_task_scope_t scope;

#if !defined(NDEBUG)
  // Debug-only operation counters omitted from release builds.
  iree_hal_task_queue_debug_state_t debug;
#endif  // !defined(NDEBUG)

  // MPSC ready list of operations with all semaphore waits satisfied.
  // Populated by semaphore timepoint callbacks (slow path) or directly
  // by the submitting thread (fast path when all waits are satisfied).
  iree_hal_task_queue_op_slist_t ready_list;

  // Set during deinitialize to signal the queue process to complete.
  // The queue process checks this at the start of each drain call and
  // returns completed=true when set, triggering normal process completion
  // and scope_end via the release callback.
  iree_atomic_int32_t shutting_down;

  // The queue's persistent control process. Budget-1: drains operations from
  // the ready list sequentially. When the ready list is empty, the process
  // returns did_work=false and the executor's sleeping protocol parks it.
  // New submissions wake it via schedule_process. Completes when
  // shutting_down is set (during deinitialize).
  //
  // For COMMANDS operations, this process fills a recording item and pushes
  // it to compute_pending (delegating actual execution to the compute
  // process below). For all other operation types, this process handles
  // them inline.
  //
  // The process participates in the queue's scope: scope_begin at
  // initialization, scope_end in the release callback. This ensures
  // scope_wait_idle blocks until the process has fully completed and its
  // worker has exited the drain stack.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_task_process_t process;

  // Persistent budget-N compute process for executing command buffer
  // recordings. Placed in a compute slot on first recording; stays there
  // for the queue's lifetime. Workers cooperatively drain recordings from
  // this process via the block processor.
  //
  // The budget-1 control process (above) is the control plane: it pops
  // operations from the ready list, fills recording items, and pushes them
  // to compute_pending. This compute process is the data plane: its drain
  // function loads the current recording item and delegates to
  // processor_drain for cooperative tile execution.
  //
  // schedule_process behavior for a persistent process: the first call
  // CAS(IDLE->DRAINING) and places it in a compute slot. Subsequent calls
  // fail the CAS (already DRAINING) but still wake workers. The process
  // stays in its slot until shutdown.
  //
  // Participates in the queue's scope: scope_begin at initialization,
  // scope_end in the release callback after the last drainer exits.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_task_process_t compute_process;

  // MPSC list of filled recording items ready to drain. The budget-1
  // process pushes items after filling; the compute drain function's
  // completer pops the next item when the current recording finishes.
  iree_hal_task_queue_compute_item_slist_t compute_pending;

  // Free pool of recording items. All items start here at initialization.
  // Budget-1 process pops to acquire; final release pushes to return.
  iree_hal_task_queue_compute_item_slist_t compute_free_pool;

  // Current recording being drained by workers. Pointer to the active item,
  // or NULL when no recording is active.
  //
  // compute_current_revision is a seqlock-style publication sequence for this
  // pointer. Writers transition even->odd before changing compute_current and
  // odd->next-even afterwards. Workers snapshot the revision, load this
  // pointer, register on item->drainers, and re-check both revision and
  // pointer before entering the block processor.
  //
  // The publication revision matters because items are recycled at stable
  // addresses. Pointer identity alone does not distinguish "same item, next
  // lifecycle," and a stale worker must not attach itself to a recycled item
  // whose payload has already been reset.
  iree_atomic_intptr_t compute_current;
  iree_atomic_int64_t compute_current_revision;

  // Arena for recording item allocation. Items are bump-allocated with
  // cache-line alignment from blocks acquired from the large block pool.
  // Items are never individually freed — they cycle through the free slist.
  // The arena is deinitialized at queue shutdown, returning all blocks.
  iree_arena_allocator_t compute_item_arena;

  // Head of the all-items linked list for shutdown cleanup enumeration.
  iree_hal_task_queue_compute_item_t* compute_item_head;

  // Number of workers for this queue (cached from executor at init time).
  // Used to size the worker_states FAM in newly allocated items.
  uint32_t compute_worker_count;
};

iree_status_t iree_hal_task_queue_initialize(
    iree_string_view_t identifier, iree_hal_queue_affinity_t affinity,
    iree_task_scope_flags_t scope_flags, iree_task_executor_t* executor,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis,
    iree_device_size_t inline_transfer_threshold,
    iree_arena_block_pool_t* small_block_pool,
    iree_arena_block_pool_t* large_block_pool,
    iree_hal_allocator_t* device_allocator, iree_hal_task_queue_t* out_queue);

void iree_hal_task_queue_deinitialize(iree_hal_task_queue_t* queue);

void iree_hal_task_queue_trim(iree_hal_task_queue_t* queue);

iree_status_t iree_hal_task_queue_submit_barrier(
    iree_hal_task_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_commands(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_task_submission_batch_t* batches);

iree_status_t iree_hal_task_queue_submit_host_call(
    iree_hal_task_queue_t* queue, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags);

iree_status_t iree_hal_task_queue_submit_alloca(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* transient_buffer,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_dealloca(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* transient_buffer,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_read(
    iree_hal_task_queue_t* queue, iree_hal_file_t* source_file,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_write(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_file_t* target_file,
    uint64_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_fill(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_copy(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_update(
    iree_hal_task_queue_t* queue, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_dispatch(
    iree_hal_task_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_t* bindings, iree_host_size_t binding_count,
    iree_hal_dispatch_flags_t flags, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_TASK_QUEUE_H_
