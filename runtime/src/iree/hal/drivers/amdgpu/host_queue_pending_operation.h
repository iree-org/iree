// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PENDING_OPERATION_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PENDING_OPERATION_H_

#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/amdgpu/host_queue_pending.h"
#include "iree/hal/utils/resource_set.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_alloca_memory_wait_t
    iree_hal_amdgpu_alloca_memory_wait_t;
typedef struct iree_hal_amdgpu_wait_entry_t iree_hal_amdgpu_wait_entry_t;

// Operation types corresponding to virtual queue vtable entries. Each type has
// a per-operation capture struct in the pending_op_t union.
typedef enum iree_hal_amdgpu_pending_op_type_e {
  IREE_HAL_AMDGPU_PENDING_OP_FILL,
  IREE_HAL_AMDGPU_PENDING_OP_COPY,
  IREE_HAL_AMDGPU_PENDING_OP_UPDATE,
  IREE_HAL_AMDGPU_PENDING_OP_DISPATCH,
  IREE_HAL_AMDGPU_PENDING_OP_EXECUTE,
  IREE_HAL_AMDGPU_PENDING_OP_ALLOCA,
  IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA,
  IREE_HAL_AMDGPU_PENDING_OP_HOST_CALL,
  IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION,
} iree_hal_amdgpu_pending_op_type_t;

// Completion ownership for a deferred operation.
typedef enum iree_hal_amdgpu_pending_op_lifecycle_e {
  // Waiting callbacks may still resolve the op.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING = 0,
  // Queue shutdown claimed cancellation ownership.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING = 1,
  // The last wait callback claimed completion ownership.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING = 2,
  // The issuing thread is registering a cold alloca memory-readiness wait.
  // Cancellation only claims PENDING ops; the arming thread publishes PENDING
  // after registration or observes a synchronous callback as COMPLETING.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT = 3,
} iree_hal_amdgpu_pending_op_lifecycle_t;

// A deferred queue operation waiting for its waits to become satisfiable.
// Arena-allocated from the queue's block pool. All variable-size captured data
// lives in the arena alongside this struct.
struct iree_hal_amdgpu_pending_op_t {
  // Arena backing this operation and all captured data.
  iree_arena_allocator_t arena;

  // Owning queue used to emit work when all waits are satisfied.
  iree_hal_amdgpu_host_queue_t* queue;

  // Next operation in the queue's pending list.
  iree_hal_amdgpu_pending_op_t* next;

  // Back-pointer to the previous link field for O(1) unlink.
  iree_hal_amdgpu_pending_op_t** prev_next;

  // Completion-thread retry queued when submission capacity is unavailable.
  iree_hal_amdgpu_host_queue_post_drain_action_t capacity_retry;

  // Number of outstanding wait timepoints.
  iree_atomic_int32_t wait_count;

  // Completion/cancellation owner.
  iree_atomic_int32_t lifecycle_state;

  // First error from a failed wait. CAS from 0; the winner owns the status.
  iree_atomic_intptr_t error_status;

  // Wakes cancellation when a detached wait callback finishes touching the op.
  iree_notification_t callback_notification;

  // Arena-owned clone of the wait semaphore list.
  iree_hal_semaphore_list_t wait_semaphore_list;

  // Arena-owned clone of signal payload values; semaphores alias the first
  // entries of retained_resources.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Wait entries registered with the wait semaphores.
  iree_hal_amdgpu_wait_entry_t* wait_entries;

  // Flat array of all retained HAL resources.
  iree_hal_resource_t** retained_resources;

  // Number of entries currently owned in |retained_resources|.
  uint16_t retained_resource_count;

  // Operation payload selector.
  iree_hal_amdgpu_pending_op_type_t type;

  union {
    // Captured queue_fill payload.
    struct {
      // Target buffer retained until the deferred fill operation issues.
      iree_hal_buffer_t* target_buffer;
      // Target byte offset captured from queue_fill.
      iree_device_size_t target_offset;
      // Number of bytes filled by this queue operation.
      iree_device_size_t length;
      // Fill pattern bits captured in the low bytes.
      uint64_t pattern_bits;
      // Fill pattern length in bytes.
      iree_host_size_t pattern_length;
      // HAL fill flags captured from queue_fill.
      iree_hal_fill_flags_t flags;
    } fill;

    // Captured queue_copy/read/write payload.
    struct {
      // Source buffer retained until the deferred copy operation issues.
      iree_hal_buffer_t* source_buffer;
      // Source byte offset captured from queue_copy/read/write.
      iree_device_size_t source_offset;
      // Target buffer retained until the deferred copy operation issues.
      iree_hal_buffer_t* target_buffer;
      // Target byte offset captured from queue_copy/read/write.
      iree_device_size_t target_offset;
      // Number of bytes copied by this queue operation.
      iree_device_size_t length;
      // HAL copy flags captured from queue_copy/read/write.
      iree_hal_copy_flags_t flags;
      // Queue profiling event type used when the copy submission issues.
      iree_hal_profile_queue_event_type_t profile_event_type;
    } copy;

    // Captured queue_update payload.
    struct {
      // Source data is copied into the arena.
      const void* source_data;
      // Target buffer retained until the deferred update operation issues.
      iree_hal_buffer_t* target_buffer;
      // Target byte offset captured from queue_update.
      iree_device_size_t target_offset;
      // Number of bytes copied from |source_data|.
      iree_device_size_t length;
      // HAL update flags captured from queue_update.
      iree_hal_update_flags_t flags;
    } update;

    // Captured queue_dispatch payload.
    struct {
      // Executable retained until the deferred dispatch operation issues.
      iree_hal_executable_t* executable;
      // Export ordinal captured from queue_dispatch.
      iree_hal_executable_export_ordinal_t export_ordinal;
      // Dispatch workgroup configuration captured from queue_dispatch.
      iree_hal_dispatch_config_t config;
      // Arena-owned copy of dispatch constants.
      iree_const_byte_span_t constants;
      // Arena-owned copy of dispatch buffer references.
      iree_hal_buffer_ref_list_t bindings;
      // HAL dispatch flags captured from queue_dispatch.
      iree_hal_dispatch_flags_t flags;
    } dispatch;

    // Captured queue_execute payload.
    struct {
      // Command buffer retained until the deferred execute operation issues.
      iree_hal_command_buffer_t* command_buffer;
      // Arena-owned copy of the binding table prefix used by command_buffer.
      iree_hal_buffer_binding_table_t binding_table;
      // Binding resources captured until the deferred execute operation issues.
      iree_hal_resource_set_t* binding_resource_set;
      // HAL execute flags captured from queue_execute.
      iree_hal_execute_flags_t flags;
    } execute;

    // Captured queue_alloca payload.
    struct {
      // Borrowed pool resolved during queue_alloca capture.
      iree_hal_pool_t* pool;
      // Buffer parameters captured from queue_alloca.
      iree_hal_buffer_params_t params;
      // Requested allocation size in bytes.
      iree_device_size_t allocation_size;
      // HAL allocation flags captured from queue_alloca.
      iree_hal_alloca_flags_t flags;
      // Pool reservation flags used when probing the selected pool.
      iree_hal_pool_reserve_flags_t reserve_flags;
      // Transient buffer returned to the caller and committed on success.
      iree_hal_buffer_t* buffer;
      // Cold memory-readiness sidecar allocated only after user waits resolve.
      iree_hal_amdgpu_alloca_memory_wait_t* memory_wait;
    } alloca_op;

    // Captured queue_dealloca payload.
    struct {
      // Transient buffer retained until the deferred dealloca operation issues.
      iree_hal_buffer_t* buffer;
    } dealloca;

    // Captured queue_host_call payload.
    struct {
      // Host callback and user data captured from queue_host_call.
      iree_hal_host_call_t call;
      // Host call arguments copied by value.
      uint64_t args[4];
      // HAL host-call flags captured from queue_host_call.
      iree_hal_host_call_flags_t flags;
    } host_call;

    // Captured driver host-action payload.
    struct {
      // Driver-owned completion-thread action ordered by queue semaphores.
      iree_hal_amdgpu_reclaim_action_t action;
    } host_action;
  };
};

// Result of trying to issue an operation payload under the queue submission
// lock.
typedef struct iree_hal_amdgpu_pending_op_payload_issue_t {
  // Whether queue admission found enough capacity for this payload.
  bool ready;

  // Pending alloca operation that owns a prepared cold memory-readiness wait.
  iree_hal_amdgpu_pending_op_t* memory_wait_op;
} iree_hal_amdgpu_pending_op_payload_issue_t;

// Allocates and links a pending operation. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_pending_op_allocate(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_amdgpu_pending_op_type_t type, uint16_t max_resource_count,
    iree_hal_amdgpu_pending_op_t** out_op);

// Retains |resource| in |op|'s preallocated retained resource table.
void iree_hal_amdgpu_pending_op_retain(iree_hal_amdgpu_pending_op_t* op,
                                       iree_hal_resource_t* resource);

// Releases all resources retained by |op|.
void iree_hal_amdgpu_pending_op_release_retained(
    iree_hal_amdgpu_pending_op_t* op);

// Destroys a capture-time failed operation. Caller must hold
// queue->locks.submission_mutex.
void iree_hal_amdgpu_pending_op_destroy_under_lock(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status);

// Issues the operation-family payload. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_pending_op_issue_payload(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_pending_op_payload_issue_t* issue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_PENDING_OPERATION_H_
