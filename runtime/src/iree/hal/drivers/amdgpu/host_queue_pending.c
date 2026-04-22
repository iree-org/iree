// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_pending.h"

#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/amdgpu/host_queue_blit.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue_dispatch.h"
#include "iree/hal/drivers/amdgpu/host_queue_host_call.h"
#include "iree/hal/drivers/amdgpu/host_queue_memory.h"
#include "iree/hal/drivers/amdgpu/host_queue_waits.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// Pending operations (deferred submission)
//===----------------------------------------------------------------------===//
//
// LOCKING PROTOCOL
//
// submission_mutex protects all submission-path state: AQL ring reservation,
// kernarg allocation, packet emission, commit_signals, frontier mutation,
// notification ring push, and the pending list (link/unlink).
//
// The completion thread (drain, error check) does NOT acquire
// submission_mutex. It reads the notification ring (SPSC consumer) and the
// atomic error_status.
//
// Deferred operations use a two-phase protocol:
//
//   Phase 1 (under submission_mutex): resolve_waits, allocate pending_op,
//     capture operation parameters, link to pending list.
//
//   Phase 2 (WITHOUT submission_mutex): register timepoints via enqueue_waits.
//     Timepoint callbacks may fire synchronously during acquire_timepoint
//     (when the semaphore value is already reached or the semaphore is already
//     failed). The last callback to fire calls pending_op_issue or
//     pending_op_fail, both of which acquire submission_mutex internally.
//     This is safe because Phase 1 released the mutex before Phase 2 began.
//
// pending_op_issue: acquires submission_mutex to emit AQL packets, transfer
//   retained resources to the reclaim ring, commit signals, and unlink.
//
// pending_op_fail: acquires submission_mutex to unlink. Semaphore failure
//   and resource release happen outside the lock.
//
// pending_op_destroy_under_lock: for capture-time failures (arena allocation
//   errors after pending_op_allocate). Caller already holds submission_mutex.
//   Does NOT re-acquire; unlinks and cleans up directly.

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

// Per-wait timepoint entry, arena-allocated one per unsatisfied wait. The
// timepoint callback decrements the operation's atomic wait counter; the last
// callback to fire issues or fails the operation.
typedef struct iree_hal_amdgpu_wait_entry_t {
  // Async semaphore timepoint registration owned by this wait entry.
  iree_async_semaphore_timepoint_t timepoint;
  // Pending operation whose wait_count is decremented by this callback.
  iree_hal_amdgpu_pending_op_t* operation;
  // Set to 1 after the callback's final access to this entry/op completes.
  // Queue shutdown spins on this for callbacks that were already detached from
  // the semaphore before cancel_timepoint() ran.
  iree_atomic_int32_t callback_complete;
} iree_hal_amdgpu_wait_entry_t;

typedef enum iree_hal_amdgpu_alloca_memory_wait_kind_e {
  // No active memory wait or held reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE = 0,
  // Waiting for a copied pool death frontier while holding a reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER = 1,
  // Waiting for a pool release notification before retrying reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION = 2,
} iree_hal_amdgpu_alloca_memory_wait_kind_t;

// Cold-path alloca memory-readiness wait. Allocated inside a pending op's arena
// only after user semaphore waits have resolved and the pool cannot produce
// immediately-usable bytes.
typedef struct iree_hal_amdgpu_alloca_memory_wait_t {
  // Active wait source.
  iree_hal_amdgpu_alloca_memory_wait_kind_t kind;

  // Set to 1 after the callback's final access to this wait/op completes.
  iree_atomic_int32_t callback_complete;

  // Held-reservation wait state blocked on a pool death frontier.
  struct {
    // Queue-owned reservation held while waiting for its death frontier.
    iree_hal_pool_reservation_t reservation;

    // Arena-owned copy of the pool-owned death frontier.
    iree_async_frontier_t* wait_frontier;

    // Tracker waiter storage for |wait_frontier|.
    iree_async_frontier_waiter_t waiter;
  } frontier;

  // Pool notification retry state for reservation attempts.
  struct {
    // Borrowed notification returned by the pool.
    iree_async_notification_t* notification;

    // Notification epoch observed before the reservation retry.
    uint32_t wait_token;

    // Whether the pre-submit observation scope is still held. Once submit
    // returns, the submitted wait operation owns its own observation scope and
    // this bridge scope is released.
    bool pre_submit_observation_held;

    // Wait operations rotated so a callback can arm a retry before returning.
    iree_async_notification_wait_operation_t wait_ops[2];

    // Index of the active wait operation in |wait_ops|.
    uint8_t wait_slot;
  } pool_notification;
} iree_hal_amdgpu_alloca_memory_wait_t;

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

static void iree_hal_amdgpu_pending_op_issue(iree_hal_amdgpu_pending_op_t* op);
static void iree_hal_amdgpu_pending_op_capacity_post_drain(void* user_data);
static void iree_hal_amdgpu_pending_op_fail(iree_hal_amdgpu_pending_op_t* op,
                                            iree_status_t status);
// Links a pending op into the queue's pending list. Caller must hold
// submission_mutex.
static void iree_hal_amdgpu_pending_op_link(iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  op->next = queue->pending_head;
  op->prev_next = &queue->pending_head;
  if (queue->pending_head) {
    queue->pending_head->prev_next = &op->next;
  }
  queue->pending_head = op;
}

// Unlinks a pending op from the queue's pending list. Caller must hold
// submission_mutex.
static void iree_hal_amdgpu_pending_op_unlink(
    iree_hal_amdgpu_pending_op_t* op) {
  *op->prev_next = op->next;
  if (op->next) {
    op->next->prev_next = op->prev_next;
  }
  op->next = NULL;
  op->prev_next = NULL;
}

// Retains a resource and appends it to the pending op's retained_resources
// array. The caller must have allocated sufficient capacity in the array
// via the max_resource_count parameter to pending_op_allocate.
static void iree_hal_amdgpu_pending_op_retain(iree_hal_amdgpu_pending_op_t* op,
                                              iree_hal_resource_t* resource) {
  if (IREE_LIKELY(resource)) {
    iree_hal_resource_retain(resource);
    op->retained_resources[op->retained_resource_count++] = resource;
  }
}

// Releases all retained HAL resources in the flat array. Used on failure,
// cancellation, and success paths where the submit helper retained the
// resources it needs instead of consuming this pending op's refs.
static void iree_hal_amdgpu_pending_op_release_retained(
    iree_hal_amdgpu_pending_op_t* op) {
  for (uint16_t i = 0; i < op->retained_resource_count; ++i) {
    iree_hal_resource_release(op->retained_resources[i]);
  }
  op->retained_resource_count = 0;
}

static void iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type == IREE_HAL_AMDGPU_PENDING_OP_EXECUTE) {
    iree_hal_resource_set_free(op->execute.binding_resource_set);
    op->execute.binding_resource_set = NULL;
  }
}

static void iree_hal_amdgpu_pending_op_fail_host_action(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION ||
      !op->host_action.action.fn) {
    return;
  }
  op->host_action.action.fn(/*entry=*/NULL, op->host_action.action.user_data,
                            status);
  op->host_action.action.fn = NULL;
  op->host_action.action.user_data = NULL;
}

// Releases any queue-owned alloca memory-readiness reservation. This runs only
// on failure/cancellation paths or after ownership has not transferred into the
// transient buffer.
static void iree_hal_amdgpu_pending_op_release_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_ALLOCA) return;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) return;

  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_pool_release_reservation(op->alloca_op.pool,
                                        &wait->frontier.reservation,
                                        wait->frontier.wait_frontier);
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      break;
  }
}

// Clears the queued marker for a deferred dealloca that never published a
// completion epoch. Successful deallocas transfer ownership to the reclaim ring
// and must not call this.
static void iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA) return;
  iree_hal_amdgpu_transient_buffer_abort_dealloca(op->dealloca.buffer);
}

static bool iree_hal_amdgpu_alloca_memory_wait_callback_is_complete(
    void* user_data) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait =
      (iree_hal_amdgpu_alloca_memory_wait_t*)user_data;
  return iree_atomic_load(&wait->callback_complete,
                          iree_memory_order_acquire) != 0;
}

static void iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_release);
  iree_notification_post(&op->callback_notification, IREE_ALL_WAITERS);
}

// Publishes a prepared memory-readiness wait as ARMING. The release store on
// lifecycle_state makes the initialized sidecar fields visible to the callback
// or cancellation path that observes the state transition.
static void iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
    iree_hal_amdgpu_pending_op_t* op,
    iree_hal_amdgpu_alloca_memory_wait_t* wait,
    iree_hal_amdgpu_alloca_memory_wait_kind_t kind) {
  wait->kind = kind;
  iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_relaxed);
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT,
                    iree_memory_order_release);
}

static iree_status_t iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op,
    iree_hal_amdgpu_alloca_memory_wait_t** out_wait) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) {
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&op->arena, sizeof(*wait), (void**)&wait));
    memset(wait, 0, sizeof(*wait));
    iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_relaxed);
    op->alloca_op.memory_wait = wait;
    IREE_TRACE_ZONE_END(z0);
  }
  *out_wait = wait;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pending_op_prepare_alloca_frontier_wait(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_alloca_reservation_t* alloca_reservation) {
  const iree_async_frontier_t* wait_frontier =
      alloca_reservation->acquire_info.wait_frontier;
  if (IREE_UNLIKELY(!wait_frontier)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue_alloca waitable pool reservation did not provide a frontier");
  }

  iree_host_size_t wait_frontier_size = 0;
  IREE_RETURN_IF_ERROR(iree_async_frontier_size(wait_frontier->entry_count,
                                                &wait_frontier_size));
  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  iree_async_frontier_t* wait_frontier_copy = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, wait_frontier->entry_count);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&op->arena, wait_frontier_size,
                              (void**)&wait_frontier_copy));

  memcpy(wait_frontier_copy, wait_frontier, wait_frontier_size);
  wait->frontier.reservation = alloca_reservation->reservation;
  wait->frontier.wait_frontier = wait_frontier_copy;
  iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
      op, wait, IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pending_op_prepare_alloca_pool_notification_wait(
    iree_hal_amdgpu_pending_op_t* op, iree_async_notification_t* notification,
    uint32_t wait_token) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  wait->pool_notification.notification = notification;
  wait->pool_notification.wait_token = wait_token;
  wait->pool_notification.pre_submit_observation_held = true;
  wait->pool_notification.wait_slot =
      (uint8_t)((wait->pool_notification.wait_slot + 1u) & 1u);
  iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
      op, wait, IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION);
  return iree_ok_status();
}

static void iree_hal_amdgpu_alloca_pool_notification_end_observe(
    iree_hal_amdgpu_alloca_memory_wait_t* wait) {
  if (wait->pool_notification.pre_submit_observation_held) {
    wait->pool_notification.pre_submit_observation_held = false;
    iree_async_notification_end_observe(wait->pool_notification.notification);
  }
}

// Cancels any active alloca memory-readiness wait before destroying the op.
static void iree_hal_amdgpu_pending_op_cancel_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_ALLOCA) return;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) return;

  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER: {
      const bool cancelled = iree_async_frontier_tracker_cancel_wait(
          op->queue->frontier_tracker, &wait->frontier.waiter);
      if (!cancelled) {
        iree_notification_await(
            &op->callback_notification,
            iree_hal_amdgpu_alloca_memory_wait_callback_is_complete, wait,
            iree_infinite_timeout());
      }
      break;
    }
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION: {
      iree_hal_amdgpu_alloca_pool_notification_end_observe(wait);
      // Shutdown is allowed to prod the pool notification: it is a broad wake,
      // but prevents teardown from depending on a future dealloca. The callback
      // observes the CANCELLING lifecycle state and only publishes completion.
      iree_async_notification_signal(wait->pool_notification.notification,
                                     INT32_MAX);
      iree_notification_await(
          &op->callback_notification,
          iree_hal_amdgpu_alloca_memory_wait_callback_is_complete, wait,
          iree_infinite_timeout());
      break;
    }
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      break;
  }
}

static bool iree_hal_amdgpu_wait_entry_callback_is_complete(void* user_data) {
  iree_hal_amdgpu_wait_entry_t* entry =
      (iree_hal_amdgpu_wait_entry_t*)user_data;
  return iree_atomic_load(&entry->callback_complete,
                          iree_memory_order_acquire) != 0;
}

static void iree_hal_amdgpu_wait_entry_publish_callback_complete(
    iree_hal_amdgpu_wait_entry_t* entry) {
  iree_atomic_store(&entry->callback_complete, 1, iree_memory_order_release);
  iree_notification_post(&entry->operation->callback_notification,
                         IREE_ALL_WAITERS);
}

static bool iree_hal_amdgpu_pending_op_wait_callbacks_are_complete(
    void* user_data) {
  iree_hal_amdgpu_pending_op_t* op = (iree_hal_amdgpu_pending_op_t*)user_data;
  for (iree_host_size_t i = 0; i < op->wait_semaphore_list.count; ++i) {
    iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
    if (!iree_hal_amdgpu_wait_entry_callback_is_complete(entry)) {
      return false;
    }
  }
  return true;
}

// Records the first asynchronous wait failure. Takes ownership of |status|,
// storing it for the completion owner or dropping it if another failure won.
static void iree_hal_amdgpu_pending_op_record_error_status(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &op->error_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
}

static bool iree_hal_amdgpu_pending_op_mark_waits_resolved(
    iree_hal_amdgpu_pending_op_t* op) {
  int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
  return iree_atomic_compare_exchange_strong(
      &op->lifecycle_state, &expected_state,
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
}

static void iree_hal_amdgpu_pending_op_complete_resolved_waits(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_notification_await(
      &op->callback_notification,
      iree_hal_amdgpu_pending_op_wait_callbacks_are_complete, op,
      iree_infinite_timeout());
  iree_status_t error = (iree_status_t)iree_atomic_exchange(
      &op->error_status, 0, iree_memory_order_acquire);
  if (!iree_status_is_ok(error)) {
    iree_hal_amdgpu_pending_op_fail(op, error);
  } else {
    iree_hal_amdgpu_pending_op_issue(op);
  }
}

// Destroys a pending operation that failed during capture (arena allocation
// error after pending_op_allocate but before enqueue_waits). Caller MUST hold
// submission_mutex; the op is linked to the pending list by allocate and
// needs the mutex for unlinking.
//
// Unlike pending_op_fail (which acquires submission_mutex internally), this
// function assumes the caller already holds it. This is necessary because the
// capture phase runs under the mutex (Phase 1 of the two-phase protocol).
static void iree_hal_amdgpu_pending_op_destroy_under_lock(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  iree_hal_amdgpu_pending_op_fail_host_action(op, status);
  // Fail signal semaphores so downstream waiters get the error.
  iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
  iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
  // Release any queue-owned memory reservation before releasing op resources.
  iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
  iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(op);
  // Release all retained resources (signal semaphores + op resources).
  iree_hal_amdgpu_pending_op_release_retained(op);
  // Release wait semaphores (separately retained by the clone).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  // Unlink from the pending list (caller holds submission_mutex).
  iree_hal_amdgpu_pending_op_unlink(op);
  // Tear down callback wake state before returning arena blocks to the pool.
  iree_notification_deinitialize(&op->callback_notification);
  // Return arena blocks to the pool.
  iree_arena_deinitialize(&op->arena);
}

// Timepoint callback fired when a wait semaphore reaches its target value or
// fails. The last resolved wait claims completion, waits for all callbacks to
// finish touching arena-owned entries, and then issues or fails the operation.
static void iree_hal_amdgpu_wait_entry_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amdgpu_wait_entry_t* entry =
      (iree_hal_amdgpu_wait_entry_t*)user_data;
  iree_hal_amdgpu_pending_op_t* op = entry->operation;

  iree_hal_amdgpu_pending_op_record_error_status(op, status);

  int32_t previous_count =
      iree_atomic_fetch_sub(&op->wait_count, 1, iree_memory_order_acq_rel);
  bool owns_completion = false;
  if (previous_count == 1) {
    owns_completion = iree_hal_amdgpu_pending_op_mark_waits_resolved(op);
  }

  iree_hal_amdgpu_wait_entry_publish_callback_complete(entry);
  if (owns_completion) {
    iree_hal_amdgpu_pending_op_complete_resolved_waits(op);
  }
}

// Registers timepoints for all waits in the operation's wait semaphore list.
// Sets wait_count and registers one timepoint per wait; callbacks may fire
// synchronously during registration. When all waits are satisfied (the last
// callback fires), the operation is issued or failed.
//
// The wait_semaphore_list on the op (cloned into the arena by allocate)
// retains all semaphores for the lifetime of the op. Wait entries do not
// independently retain semaphores.
static iree_status_t iree_hal_amdgpu_pending_op_enqueue_waits(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_semaphore_list_t wait_semaphores = op->wait_semaphore_list;
  if (wait_semaphores.count == 0) {
    iree_hal_amdgpu_pending_op_issue(op);
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, wait_semaphores.count);

  iree_host_size_t wait_entry_bytes = 0;
  iree_status_t status =
      IREE_STRUCT_LAYOUT(0, &wait_entry_bytes,
                         IREE_STRUCT_FIELD(wait_semaphores.count,
                                           iree_hal_amdgpu_wait_entry_t, NULL));
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  status = iree_arena_allocate(&op->arena, wait_entry_bytes,
                               (void**)&op->wait_entries);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  memset(op->wait_entries, 0, wait_entry_bytes);
  // Unregistered entries never receive callbacks, so they start complete.
  // Active registrations flip their entry incomplete until the callback exits.
  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    iree_atomic_store(&op->wait_entries[i].callback_complete, 1,
                      iree_memory_order_relaxed);
  }

  // Set wait_count before registering any timepoints. A timepoint callback
  // may fire synchronously during acquire_timepoint.
  iree_atomic_store(&op->wait_count, (int32_t)wait_semaphores.count,
                    iree_memory_order_release);

  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
    entry->operation = op;
    iree_atomic_store(&entry->callback_complete, 0, iree_memory_order_relaxed);
    entry->timepoint.callback = iree_hal_amdgpu_wait_entry_resolved;
    entry->timepoint.user_data = entry;
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)wait_semaphores.semaphores[i],
        wait_semaphores.payload_values[i], &entry->timepoint);

    if (!iree_status_is_ok(status)) {
      // Registration failed at index i. Timepoints 0..i-1 are already
      // registered and their callbacks will fire asynchronously; we cannot
      // destroy the op here. Record the error and subtract the unregistered
      // count so the existing callbacks drain and destroy the op.
      iree_hal_amdgpu_pending_op_record_error_status(op, status);
      int32_t unregistered = (int32_t)(wait_semaphores.count - i);
      iree_atomic_store(&entry->callback_complete, 1,
                        iree_memory_order_release);
      int32_t previous_count = iree_atomic_fetch_sub(
          &op->wait_count, unregistered, iree_memory_order_acq_rel);
      if (previous_count == unregistered) {
        if (iree_hal_amdgpu_pending_op_mark_waits_resolved(op)) {
          iree_hal_amdgpu_pending_op_complete_resolved_waits(op);
        }
      }
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdgpu_alloca_memory_wait_resolved(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (iree_status_is_ok(status) &&
      wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION) {
    wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
  }

  int32_t expected_state =
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT;
  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_pending_op_record_error_status(op, status);
    iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
    return;
  }

  expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
    if (!iree_status_is_ok(status)) {
      iree_hal_amdgpu_pending_op_fail(op, status);
    } else {
      iree_hal_amdgpu_pending_op_issue(op);
    }
    return;
  }

  iree_hal_amdgpu_pending_op_record_error_status(op, status);
  iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
}

static void iree_hal_amdgpu_alloca_frontier_wait_resolved(
    void* user_data, iree_status_t status) {
  iree_hal_amdgpu_alloca_memory_wait_resolved(
      (iree_hal_amdgpu_pending_op_t*)user_data, status);
}

static void iree_hal_amdgpu_alloca_pool_notification_wait_resolved(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_amdgpu_alloca_memory_wait_resolved(
      (iree_hal_amdgpu_pending_op_t*)user_data, status);
}

static void iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  int32_t expected_state =
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT;
  if (iree_status_is_ok(status)) {
    if (iree_atomic_compare_exchange_strong(
            &op->lifecycle_state, &expected_state,
            IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      return;
    }
    if (expected_state == IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING) {
      iree_status_t error = (iree_status_t)iree_atomic_exchange(
          &op->error_status, 0, iree_memory_order_acquire);
      if (!iree_status_is_ok(error)) {
        iree_hal_amdgpu_pending_op_fail(op, error);
      } else {
        iree_hal_amdgpu_pending_op_issue(op);
      }
    }
    return;
  }

  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    return;
  }
  iree_hal_amdgpu_pending_op_record_error_status(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_frontier_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_atomic_store(&wait->callback_complete, 0, iree_memory_order_relaxed);
  iree_status_t status = iree_async_frontier_tracker_wait(
      queue->frontier_tracker, wait->frontier.wait_frontier,
      iree_hal_amdgpu_alloca_frontier_wait_resolved, op,
      &wait->frontier.waiter);
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_pool_notification_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_async_notification_wait_operation_t* wait_op =
      &wait->pool_notification.wait_ops[wait->pool_notification.wait_slot];
  iree_async_operation_zero(&wait_op->base, sizeof(*wait_op));
  iree_async_operation_initialize(
      &wait_op->base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_amdgpu_alloca_pool_notification_wait_resolved, op);
  wait_op->notification = wait->pool_notification.notification;
  wait_op->wait_flags = IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
  wait_op->wait_token = wait->pool_notification.wait_token;

  iree_atomic_store(&wait->callback_complete, 0, iree_memory_order_relaxed);
  iree_status_t status =
      iree_async_proactor_submit_one(op->queue->proactor, &wait_op->base);
  iree_hal_amdgpu_alloca_pool_notification_end_observe(wait);
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(op, status);
}

void iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_amdgpu_pending_op_enqueue_alloca_frontier_wait(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      iree_hal_amdgpu_pending_op_enqueue_alloca_pool_notification_wait(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      iree_hal_amdgpu_pending_op_fail(
          op, iree_make_status(IREE_STATUS_INTERNAL,
                               "pending alloca has no memory wait to enqueue"));
      break;
  }
}

static void iree_hal_amdgpu_pending_op_enqueue_capacity_retry(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
                    iree_memory_order_release);
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      op->queue, &op->capacity_retry,
      iree_hal_amdgpu_pending_op_capacity_post_drain, op);
}

static void iree_hal_amdgpu_pending_op_capacity_post_drain(void* user_data) {
  iree_hal_amdgpu_pending_op_issue((iree_hal_amdgpu_pending_op_t*)user_data);
}

iree_status_t iree_hal_amdgpu_pending_op_start(iree_hal_amdgpu_pending_op_t* op,
                                               bool wait_for_capacity) {
  if (wait_for_capacity) {
    iree_hal_amdgpu_pending_op_enqueue_capacity_retry(op);
    return iree_ok_status();
  }
  return iree_hal_amdgpu_pending_op_enqueue_waits(op);
}

static iree_status_t iree_hal_amdgpu_host_queue_clone_error_status(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_status_t error = (iree_status_t)iree_atomic_load(
      &queue->error_status, iree_memory_order_acquire);
  return iree_status_is_ok(error) ? iree_ok_status() : iree_status_clone(error);
}

// Allocates and initializes a pending operation from a fresh arena.
// Clones the wait semaphore list. Allocates the retained_resources array
// with |max_resource_count| capacity and populates the first entries with
// signal semaphores (retained). The signal_semaphore_list.semaphores pointer
// aliases into retained_resources so that commit_signals and
// semaphore_list_fail can use it directly.
//
// The caller must push operation-specific resources into retained_resources
// (via the returned op) before calling enqueue_waits.
//
// On failure, the arena is cleaned up and *out_op is set to NULL.
static iree_status_t iree_hal_amdgpu_pending_op_allocate(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_amdgpu_pending_op_type_t type, uint16_t max_resource_count,
    iree_hal_amdgpu_pending_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, type);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, max_resource_count);

  iree_arena_allocator_t arena;
  iree_arena_initialize(queue->block_pool, &arena);

  iree_hal_amdgpu_pending_op_t* op = NULL;
  iree_status_t status = iree_arena_allocate(&arena, sizeof(*op), (void**)&op);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  memset(op, 0, sizeof(*op));
  memcpy(&op->arena, &arena, sizeof(arena));
  op->queue = queue;
  op->type = type;
  iree_atomic_store(&op->wait_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING,
                    iree_memory_order_relaxed);
  iree_atomic_store(&op->error_status, 0, iree_memory_order_relaxed);
  iree_notification_initialize(&op->callback_notification);

  iree_allocator_t arena_allocator = iree_arena_allocator(&op->arena);

  // Clone the wait semaphore list (retains each wait semaphore).
  status = iree_hal_semaphore_list_clone(wait_semaphore_list, arena_allocator,
                                         &op->wait_semaphore_list);

  // Allocate the retained resources array and the signal payload values.
  if (iree_status_is_ok(status) && max_resource_count > 0) {
    iree_host_size_t retained_resource_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &retained_resource_size,
        IREE_STRUCT_FIELD(max_resource_count, iree_hal_resource_t*, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&op->arena, retained_resource_size,
                                   (void**)&op->retained_resources);
    }
  }
  uint64_t* signal_payload_values = NULL;
  if (iree_status_is_ok(status) && signal_semaphore_list->count > 0) {
    iree_host_size_t signal_payload_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &signal_payload_size,
        IREE_STRUCT_FIELD(signal_semaphore_list->count, uint64_t, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&op->arena, signal_payload_size,
                                   (void**)&signal_payload_values);
    }
  }

  if (iree_status_is_ok(status)) {
    // Signal semaphores occupy the first entries of retained_resources.
    // The signal_semaphore_list.semaphores pointer aliases this region.
    for (iree_host_size_t i = 0; i < signal_semaphore_list->count; ++i) {
      op->retained_resources[i] =
          (iree_hal_resource_t*)signal_semaphore_list->semaphores[i];
      iree_hal_resource_retain(op->retained_resources[i]);
      signal_payload_values[i] = signal_semaphore_list->payload_values[i];
    }
    op->retained_resource_count = (uint16_t)signal_semaphore_list->count;
    op->signal_semaphore_list.count = signal_semaphore_list->count;
    op->signal_semaphore_list.semaphores =
        (iree_hal_semaphore_t**)op->retained_resources;
    op->signal_semaphore_list.payload_values = signal_payload_values;

    iree_hal_amdgpu_pending_op_link(op);
    *out_op = op;
  } else {
    iree_hal_semaphore_list_release(op->wait_semaphore_list);
    iree_notification_deinitialize(&op->callback_notification);
    iree_arena_deinitialize(&op->arena);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Issues a deferred operation after all waits are satisfied. All waits are
// tier 0 (timeline_value >= waited_value); the GPU work producing those
// values has completed. No barriers are needed.
//
// Called from the last wait_entry callback (any thread). Acquires
// submission_mutex to emit AQL packets and commit signals.
static void iree_hal_amdgpu_pending_op_issue(iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_slim_mutex_lock(&queue->submission_mutex);

  iree_status_t status = iree_ok_status();
  bool ready = true;
  if (queue->is_shutting_down) {
    status = iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  } else {
    status = iree_hal_amdgpu_host_queue_clone_error_status(queue);
  }
  if (iree_status_is_ok(status)) {
    // All waits are tier 0; emit operation packets with no dependency
    // barriers.
    iree_hal_amdgpu_wait_resolution_t resolution;
    resolution.barrier_count = 0;
    resolution.needs_deferral = false;
    memset(resolution.reserved, 0, sizeof(resolution.reserved));
    resolution.wait_count = op->wait_semaphore_list.count > UINT32_MAX
                                ? UINT32_MAX
                                : (uint32_t)op->wait_semaphore_list.count;
    resolution.profile_event_flags =
        IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
    resolution.inline_acquire_scope = op->wait_semaphore_list.count > 0
                                          ? IREE_HSA_FENCE_SCOPE_SYSTEM
                                          : IREE_HSA_FENCE_SCOPE_NONE;
    resolution.barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
    switch (op->type) {
      case IREE_HAL_AMDGPU_PENDING_OP_FILL:
        status = iree_hal_amdgpu_host_queue_submit_fill(
            queue, &resolution, op->signal_semaphore_list,
            op->fill.target_buffer, op->fill.target_offset, op->fill.length,
            op->fill.pattern_bits, op->fill.pattern_length, op->fill.flags,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &ready);
        if (iree_status_is_ok(status) && ready) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_COPY:
        status = iree_hal_amdgpu_host_queue_submit_copy(
            queue, &resolution, op->signal_semaphore_list,
            op->copy.source_buffer, op->copy.source_offset,
            op->copy.target_buffer, op->copy.target_offset, op->copy.length,
            op->copy.flags, op->copy.profile_event_type,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &ready);
        if (iree_status_is_ok(status) && ready) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_UPDATE:
        status = iree_hal_amdgpu_host_queue_submit_update(
            queue, &resolution, op->signal_semaphore_list,
            op->update.source_data, /*source_offset=*/0,
            op->update.target_buffer, op->update.target_offset,
            op->update.length, op->update.flags,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &ready);
        if (iree_status_is_ok(status) && ready) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_DISPATCH:
        status = iree_hal_amdgpu_host_queue_submit_dispatch(
            queue, &resolution, op->signal_semaphore_list,
            op->dispatch.executable, op->dispatch.export_ordinal,
            op->dispatch.config, op->dispatch.constants, op->dispatch.bindings,
            op->dispatch.flags, IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE,
            &ready);
        if (iree_status_is_ok(status) && ready) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_EXECUTE:
        if (op->execute.command_buffer) {
          status = iree_hal_amdgpu_host_queue_submit_command_buffer(
              queue, &resolution, op->signal_semaphore_list,
              op->execute.command_buffer, op->execute.binding_table,
              op->execute.flags, &op->execute.binding_resource_set, &ready);
          if (iree_status_is_ok(status) && ready) {
            iree_hal_amdgpu_pending_op_release_retained(op);
          }
        } else {
          status = iree_hal_amdgpu_host_queue_try_submit_barrier(
              queue, &resolution, op->signal_semaphore_list,
              (iree_hal_amdgpu_reclaim_action_t){0},
              /*operation_resources=*/NULL,
              /*operation_resource_count=*/0,
              /*profile_event_info=*/NULL,
              iree_hal_amdgpu_host_queue_post_commit_callback_null(),
              /*resource_set=*/NULL,
              IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &ready,
              /*out_submission_id=*/NULL);
          if (iree_status_is_ok(status) && ready) {
            op->retained_resource_count = 0;
          }
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_ALLOCA: {
        iree_hal_amdgpu_pending_op_t* memory_wait_op = NULL;
        status = iree_hal_amdgpu_host_queue_submit_alloca(
            queue, &resolution, op->signal_semaphore_list, op->alloca_op.pool,
            op->alloca_op.params, op->alloca_op.allocation_size,
            op->alloca_op.flags, op->alloca_op.reserve_flags,
            op->alloca_op.buffer,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, op,
            &memory_wait_op, &ready);
        if (iree_status_is_ok(status) && memory_wait_op) {
          iree_slim_mutex_unlock(&queue->submission_mutex);
          iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(memory_wait_op);
          return;
        }
        if (iree_status_is_ok(status) && ready) {
          op->retained_resource_count = 0;
        }
        break;
      }
      case IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA:
        status = iree_hal_amdgpu_host_queue_submit_dealloca(
            queue, &resolution, op->signal_semaphore_list, op->dealloca.buffer,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &ready);
        if (iree_status_is_ok(status) && ready) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_HOST_CALL:
        status = iree_hal_amdgpu_host_queue_submit_host_call(
            queue, &resolution, op->signal_semaphore_list, op->host_call.call,
            op->host_call.args, op->host_call.flags, &ready);
        if (iree_status_is_ok(status) && ready) {
          iree_hal_amdgpu_pending_op_release_retained(op);
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION:
        status = iree_hal_amdgpu_host_queue_try_submit_barrier(
            queue, &resolution, iree_hal_semaphore_list_empty(),
            op->host_action.action, op->retained_resources,
            op->retained_resource_count,
            /*profile_event_info=*/NULL,
            iree_hal_amdgpu_host_queue_post_commit_callback_null(),
            /*resource_set=*/NULL,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &ready,
            /*out_submission_id=*/NULL);
        if (iree_status_is_ok(status) && ready) {
          op->retained_resource_count = 0;
        }
        break;
      default:
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "unrecognized pending op type %d", op->type);
        break;
    }
  }

  if (iree_status_is_ok(status) && !ready) {
    iree_hal_amdgpu_pending_op_enqueue_capacity_retry(op);
    iree_slim_mutex_unlock(&queue->submission_mutex);
    return;
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_fail_host_action(op, status);
    iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
    iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
    iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
    iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(op);
    iree_hal_amdgpu_pending_op_release_retained(op);
  }

  // Clean up the pending op. Wait semaphore list is released (the clone holds
  // separate retains). Remaining retained_resources entries are either
  // transferred to reclaim or were released by the success path above.
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_notification_deinitialize(&op->callback_notification);
  iree_arena_deinitialize(&op->arena);

  iree_slim_mutex_unlock(&queue->submission_mutex);
}

// Fails a deferred operation. Propagates the error to all signal semaphores
// so downstream waiters receive the failure instead of hanging. Takes
// ownership of |status|.
static void iree_hal_amdgpu_pending_op_fail(iree_hal_amdgpu_pending_op_t* op,
                                            iree_status_t status) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_hal_amdgpu_pending_op_fail_host_action(op, status);
  // Fail signal semaphores (records error, does not release our retains).
  iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
  iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
  // Release any queue-owned memory reservation before releasing op resources.
  iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
  iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(op);
  // Release all retained resources (signal semaphores + op resources).
  iree_hal_amdgpu_pending_op_release_retained(op);
  // Release wait semaphores (separately retained by the clone).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_slim_mutex_unlock(&queue->submission_mutex);
  iree_notification_deinitialize(&op->callback_notification);
  iree_arena_deinitialize(&op->arena);
}

// Cancels all pending operations on a queue with the given failure details.
// Creates a status only for operations that do not already carry a wait error.
// Called during deinitialize or on unrecoverable GPU fault.
// Caller must ensure no concurrent submissions (shutdown path).
void iree_hal_amdgpu_host_queue_cancel_pending(
    iree_hal_amdgpu_host_queue_t* queue, iree_status_code_t status_code,
    const char* status_message) {
  iree_slim_mutex_lock(&queue->submission_mutex);
  queue->is_shutting_down = true;
  iree_slim_mutex_unlock(&queue->submission_mutex);

  for (;;) {
    iree_hal_amdgpu_pending_op_t* op = NULL;
    iree_slim_mutex_lock(&queue->submission_mutex);
    for (iree_hal_amdgpu_pending_op_t* candidate = queue->pending_head;
         candidate != NULL; candidate = candidate->next) {
      int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
      if (iree_atomic_compare_exchange_strong(
              &candidate->lifecycle_state, &expected_state,
              IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING,
              iree_memory_order_acq_rel, iree_memory_order_acquire)) {
        iree_hal_amdgpu_pending_op_unlink(candidate);
        op = candidate;
        break;
      }
    }
    bool has_pending_ops = queue->pending_head != NULL;
    iree_slim_mutex_unlock(&queue->submission_mutex);

    if (op == NULL) {
      if (!has_pending_ops) break;
      iree_thread_yield();
      continue;
    }

    for (iree_host_size_t i = 0; i < op->wait_semaphore_list.count; ++i) {
      iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
      if (iree_hal_amdgpu_wait_entry_callback_is_complete(entry)) continue;
      if (iree_async_semaphore_cancel_timepoint(entry->timepoint.semaphore,
                                                &entry->timepoint)) {
        continue;
      }
      iree_notification_await(&op->callback_notification,
                              iree_hal_amdgpu_wait_entry_callback_is_complete,
                              entry, iree_infinite_timeout());
    }
    iree_hal_amdgpu_pending_op_cancel_alloca_memory_wait(op);

    iree_status_t op_status = (iree_status_t)iree_atomic_exchange(
        &op->error_status, 0, iree_memory_order_acquire);
    if (iree_status_is_ok(op_status)) {
      op_status = iree_make_status(status_code, "%s", status_message);
    }
    iree_hal_semaphore_list_fail(op->signal_semaphore_list, op_status);
    iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
    iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
    iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(op);
    iree_hal_amdgpu_pending_op_release_retained(op);
    iree_hal_semaphore_list_release(op->wait_semaphore_list);
    iree_notification_deinitialize(&op->callback_notification);
    iree_arena_deinitialize(&op->arena);
  }
}

//===----------------------------------------------------------------------===//
// Pending operation capture helpers
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_host_queue_defer_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_ALLOCA, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)buffer);
  op->alloca_op.pool = pool;
  op->alloca_op.params = params;
  op->alloca_op.allocation_size = allocation_size;
  op->alloca_op.flags = flags;
  op->alloca_op.reserve_flags = reserve_flags;
  op->alloca_op.buffer = buffer;
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_host_queue_submit_alloca_held_frontier_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_alloca_memory_wait_t* memory_wait, bool* out_ready) {
  iree_hal_amdgpu_alloca_reservation_t alloca_reservation = {
      .readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY,
      .acquire_result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT,
      .reservation = memory_wait->frontier.reservation,
      .acquire_info =
          {
              .wait_frontier = memory_wait->frontier.wait_frontier,
          },
      .wait_resolution = *resolution,
  };
  memory_wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
  return iree_hal_amdgpu_host_queue_submit_alloca_reservation(
      queue, &alloca_reservation, signal_semaphore_list, allocation_pool,
      params, buffer, submission_flags, out_ready);
}

static iree_status_t iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  *out_memory_wait_op = NULL;
  if (pending_op) {
    *out_memory_wait_op = pending_op;
    return iree_ok_status();
  }

  iree_hal_semaphore_list_t empty_wait_list = iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_defer_alloca(
      queue, &empty_wait_list, &signal_semaphore_list, allocation_pool, params,
      allocation_size, flags, reserve_flags, buffer, out_memory_wait_op));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    const iree_hal_amdgpu_alloca_reservation_t* alloca_reservation,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  iree_hal_amdgpu_pending_op_t* memory_wait_op = pending_op;
  iree_status_t status = iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
      queue, signal_semaphore_list, allocation_pool, params, allocation_size,
      flags, reserve_flags, buffer, pending_op, &memory_wait_op);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pending_op_prepare_alloca_frontier_wait(
        memory_wait_op, alloca_reservation);
  }
  if (iree_status_is_ok(status)) {
    *out_memory_wait_op = memory_wait_op;
  } else {
    iree_hal_pool_release_reservation(
        allocation_pool, &alloca_reservation->reservation,
        alloca_reservation->acquire_info.wait_frontier);
    if (!pending_op && memory_wait_op) {
      iree_hal_amdgpu_pending_op_destroy_under_lock(memory_wait_op,
                                                    iree_status_clone(status));
    }
  }

  return status;
}

static iree_status_t
iree_hal_amdgpu_host_queue_defer_alloca_pool_notification_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op, bool* out_ready) {
  iree_async_notification_t* notification =
      iree_hal_pool_notification(allocation_pool);
  if (IREE_UNLIKELY(!notification)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue_alloca exhausted pool did not provide a notification");
  }

  const uint32_t wait_token =
      iree_async_notification_begin_observe(notification);
  iree_hal_amdgpu_alloca_reservation_t alloca_reservation;
  iree_status_t status = iree_hal_amdgpu_host_queue_acquire_alloca_reservation(
      queue, resolution, allocation_pool, params, allocation_size, flags,
      reserve_flags, buffer, &alloca_reservation);

  bool observation_transferred = false;
  if (iree_status_is_ok(status)) {
    switch (alloca_reservation.readiness) {
      case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY:
        status = iree_hal_amdgpu_host_queue_submit_alloca_reservation(
            queue, &alloca_reservation, signal_semaphore_list, allocation_pool,
            params, buffer, submission_flags, out_ready);
        break;
      case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT:
        status = iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
            queue, signal_semaphore_list, allocation_pool, params,
            allocation_size, flags, reserve_flags, buffer, &alloca_reservation,
            pending_op, out_memory_wait_op);
        break;
      case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION:
        break;
      default:
        status =
            iree_make_status(IREE_STATUS_INTERNAL,
                             "unrecognized alloca reservation readiness %u",
                             alloca_reservation.readiness);
        break;
    }
  }

  iree_hal_amdgpu_pending_op_t* memory_wait_op = pending_op;
  if (iree_status_is_ok(status) &&
      alloca_reservation.readiness ==
          IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION) {
    status = iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
        queue, signal_semaphore_list, allocation_pool, params, allocation_size,
        flags, reserve_flags, buffer, pending_op, &memory_wait_op);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_pending_op_prepare_alloca_pool_notification_wait(
          memory_wait_op, notification, wait_token);
      observation_transferred = iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      *out_memory_wait_op = memory_wait_op;
    } else if (!pending_op && memory_wait_op) {
      iree_hal_amdgpu_pending_op_destroy_under_lock(memory_wait_op,
                                                    iree_status_clone(status));
    }
  }
  if (!observation_transferred) {
    iree_async_notification_end_observe(notification);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_submit_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op, bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  *out_memory_wait_op = NULL;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  iree_hal_amdgpu_alloca_memory_wait_t* memory_wait =
      pending_op ? pending_op->alloca_op.memory_wait : NULL;
  if (memory_wait &&
      memory_wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER) {
    return iree_hal_amdgpu_host_queue_submit_alloca_held_frontier_wait(
        queue, resolution, signal_semaphore_list, allocation_pool, params,
        buffer, submission_flags, memory_wait, out_ready);
  }

  iree_hal_amdgpu_alloca_reservation_t alloca_reservation;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_acquire_alloca_reservation(
      queue, resolution, allocation_pool, params, allocation_size, flags,
      reserve_flags, buffer, &alloca_reservation));
  switch (alloca_reservation.readiness) {
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY:
      return iree_hal_amdgpu_host_queue_submit_alloca_reservation(
          queue, &alloca_reservation, signal_semaphore_list, allocation_pool,
          params, buffer, submission_flags, out_ready);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT:
      return iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
          queue, signal_semaphore_list, allocation_pool, params,
          allocation_size, flags, reserve_flags, buffer, &alloca_reservation,
          pending_op, out_memory_wait_op);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION:
      return iree_hal_amdgpu_host_queue_defer_alloca_pool_notification_wait(
          queue, resolution, signal_semaphore_list, allocation_pool, params,
          allocation_size, flags, reserve_flags, buffer, submission_flags,
          pending_op, out_memory_wait_op, out_ready);
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized alloca reservation readiness %u",
                              alloca_reservation.readiness);
  }
}

iree_status_t iree_hal_amdgpu_host_queue_defer_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)buffer);
  op->dealloca.buffer = buffer;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_defer_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_FILL, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);
  op->fill.target_buffer = target_buffer;
  op->fill.target_offset = target_offset;
  op->fill.length = length;
  op->fill.pattern_bits = pattern_bits;
  op->fill.pattern_length = pattern_length;
  op->fill.flags = flags;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_defer_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_profile_queue_event_type_t profile_event_type,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/2, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_COPY, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)source_buffer);
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);
  op->copy.source_buffer = source_buffer;
  op->copy.source_offset = source_offset;
  op->copy.target_buffer = target_buffer;
  op->copy.target_offset = target_offset;
  op->copy.length = length;
  op->copy.flags = flags;
  op->copy.profile_event_type = profile_event_type;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_defer_update(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  const uint8_t* source_bytes = NULL;
  iree_host_size_t source_length = 0;
  uint8_t* target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_update_copy(
      target_buffer, target_offset, source_buffer, source_offset, length, flags,
      &source_bytes, &source_length, &target_device_ptr));
  (void)target_device_ptr;

  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_UPDATE, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);

  // Copy the source host data into the arena. The caller's buffer may be
  // freed after this call returns.
  void* source_copy = NULL;
  iree_status_t status =
      iree_arena_allocate(&op->arena, source_length, &source_copy);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op, status);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "arena allocation failed during defer_update");
  }
  memcpy(source_copy, source_bytes, source_length);
  op->update.source_data = source_copy;
  op->update.target_buffer = target_buffer;
  op->update.target_offset = target_offset;
  op->update.length = length;
  op->update.flags = flags;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_defer_execute(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags, iree_hal_amdgpu_pending_op_t** out_op) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_execute_flags(flags));
  if (IREE_UNLIKELY(!command_buffer && binding_table.count != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "barrier-only queue_execute must not provide a binding table "
        "(count=%" PRIhsz ")",
        binding_table.count);
  }
  const iree_host_size_t binding_count =
      command_buffer ? command_buffer->binding_count : 0;
  if (command_buffer && command_buffer->binding_count == 0) {
    binding_table = iree_hal_buffer_binding_table_empty();
  }

  iree_hal_resource_set_t* binding_resource_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_create_binding_table_resource_set(
          queue, command_buffer, binding_table, flags, &binding_resource_set));

  const iree_host_size_t operation_resource_count = command_buffer ? 1 : 0;
  uint16_t max_resources = 0;
  iree_hal_amdgpu_pending_op_t* op = NULL;
  iree_status_t status = iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count, operation_resource_count, &max_resources);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pending_op_allocate(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_AMDGPU_PENDING_OP_EXECUTE, max_resources, &op);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)command_buffer);
    op->execute.command_buffer = command_buffer;
    op->execute.binding_resource_set = binding_resource_set;
    binding_resource_set = NULL;
    op->execute.flags = flags;
  }

  if (iree_status_is_ok(status) && binding_count > 0) {
    iree_hal_buffer_binding_t* bindings_copy = NULL;
    iree_host_size_t binding_table_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &binding_table_size,
        IREE_STRUCT_FIELD(binding_count, iree_hal_buffer_binding_t, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&op->arena, binding_table_size,
                                   (void**)&bindings_copy);
    }
    if (iree_status_is_ok(status)) {
      memcpy(bindings_copy, binding_table.bindings, binding_table_size);
      op->execute.binding_table.count = binding_count;
      op->execute.binding_table.bindings = bindings_copy;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_op = op;
  } else {
    iree_hal_resource_set_free(binding_resource_set);
    if (op) {
      iree_hal_amdgpu_pending_op_destroy_under_lock(op,
                                                    iree_status_clone(status));
    }
  }
  return status;
}

// Captures a dispatch operation into a pending op. Copies constants and
// bindings into the arena.
// Caller must hold submission_mutex. Caller must call enqueue_waits after
// releasing submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_defer_dispatch(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  iree_host_size_t operation_resource_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_dispatch(
      queue, executable, export_ordinal, config, constants, bindings, flags,
      &operation_resource_count));
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count, operation_resource_count, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_DISPATCH, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)executable);
  op->dispatch.executable = executable;
  op->dispatch.export_ordinal = export_ordinal;
  op->dispatch.config = config;
  op->dispatch.flags = flags;

  // Copy constants into the arena.
  iree_status_t status = iree_ok_status();
  if (constants.data_length > 0) {
    void* constants_copy = NULL;
    status =
        iree_arena_allocate(&op->arena, constants.data_length, &constants_copy);
    if (iree_status_is_ok(status)) {
      memcpy(constants_copy, constants.data, constants.data_length);
      op->dispatch.constants.data = (const uint8_t*)constants_copy;
      op->dispatch.constants.data_length = constants.data_length;
    }
  }

  // Copy bindings array and retain all bound buffers.
  if (iree_status_is_ok(status) && bindings.count > 0 &&
      !iree_any_bit_set(flags,
                        IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    iree_hal_buffer_ref_t* bindings_copy = NULL;
    iree_host_size_t binding_ref_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &binding_ref_size,
        IREE_STRUCT_FIELD(bindings.count, iree_hal_buffer_ref_t, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&op->arena, binding_ref_size,
                                   (void**)&bindings_copy);
    }
    if (iree_status_is_ok(status)) {
      memcpy(bindings_copy, bindings.values, binding_ref_size);
      for (iree_host_size_t i = 0; i < bindings.count; ++i) {
        iree_hal_amdgpu_pending_op_retain(
            op, (iree_hal_resource_t*)bindings_copy[i].buffer);
      }
      op->dispatch.bindings.count = bindings.count;
      op->dispatch.bindings.values = bindings_copy;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_op = op;
  } else {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op,
                                                  iree_status_clone(status));
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_defer_host_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      /*signal_semaphore_count=*/0,
      /*operation_resource_count=*/operation_resource_count, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  const iree_hal_semaphore_list_t empty_signal_list =
      iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, &empty_signal_list,
      IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION, max_resources, &op));
  for (iree_host_size_t i = 0; i < operation_resource_count; ++i) {
    iree_hal_amdgpu_pending_op_retain(op, operation_resources[i]);
  }
  op->host_action.action = action;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_defer_host_call(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags, iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/0, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_HOST_CALL, max_resources, &op));
  op->host_call.call = call;
  memcpy(op->host_call.args, args, sizeof(op->host_call.args));
  op->host_call.flags = flags;
  *out_op = op;
  return iree_ok_status();
}
