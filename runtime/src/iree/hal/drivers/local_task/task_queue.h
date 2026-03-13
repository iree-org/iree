// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Task queue: a persistent budget-1 process that drains an MPSC ready list
// of queue operations (command buffer execution, barriers, host calls).
//
// Submissions flow through semaphore waits into the ready list. The queue
// process pops operations and handles them: command buffers are issued as
// separate compute processes via block_command_buffer_issue; barriers and
// host calls are handled inline by the queue process itself.
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
#include "iree/task/executor.h"
#include "iree/task/process.h"
#include "iree/task/scope.h"

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
} iree_hal_task_queue_op_type_t;

// Forward declaration for the typed slist.
typedef struct iree_hal_task_queue_op_t iree_hal_task_queue_op_t;

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

  // First error encountered from a failed semaphore wait. Set via CAS —
  // only the first error wins. Checked on the last wait_count decrement
  // to decide between pushing to the ready list or failing the operation.
  iree_atomic_intptr_t error_status;

  // Type-specific data.
  union {
    struct {
      iree_hal_command_buffer_t* command_buffer;
      iree_hal_buffer_binding_table_t binding_table;
    } commands;
    struct {
      iree_hal_device_t* device;
      iree_hal_queue_affinity_t queue_affinity;
      iree_hal_host_call_t call;
      uint64_t args[4];
      iree_hal_host_call_flags_t flags;
    } host_call;
    struct {
      iree_hal_allocator_t* device_allocator;
      iree_hal_buffer_params_t params;
      iree_device_size_t allocation_size;
      iree_hal_buffer_t* transient_buffer;
    } alloca;
    struct {
      iree_hal_buffer_t* transient_buffer;
    } dealloca;
  };
};

// Typed MPSC slist for the ready list.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_hal_task_queue_op,
                                iree_hal_task_queue_op_t,
                                offsetof(iree_hal_task_queue_op_t, slist_next));

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_t
//===----------------------------------------------------------------------===//

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

  // MPSC ready list of operations with all semaphore waits satisfied.
  // Populated by semaphore timepoint callbacks (slow path) or directly
  // by the submitting thread (fast path when all waits are satisfied).
  iree_hal_task_queue_op_slist_t ready_list;

  // Set during deinitialize to signal the queue process to complete.
  // The queue process checks this at the start of each drain call and
  // returns completed=true when set, triggering normal process completion
  // and scope_end via the completion callback.
  iree_atomic_int32_t shutting_down;

  // The queue's persistent process. Budget-1: drains operations from the
  // ready list sequentially. When the ready list is empty, the process
  // returns did_work=false and the executor's sleeping protocol parks it.
  // New submissions wake it via schedule_process. Completes when
  // shutting_down is set (during deinitialize).
  //
  // The process participates in the queue's scope: scope_begin at
  // initialization, scope_end in the completion callback. This ensures
  // scope_wait_idle blocks until the process has fully completed and no
  // worker is touching queue/device resources.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_task_process_t process;
};

void iree_hal_task_queue_initialize(
    iree_string_view_t identifier, iree_hal_queue_affinity_t affinity,
    iree_task_scope_flags_t scope_flags, iree_task_executor_t* executor,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis,
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
    iree_hal_task_queue_t* queue, iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_buffer_t* transient_buffer,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_dealloca(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* transient_buffer,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_TASK_QUEUE_H_
